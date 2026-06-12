import os
import gc
import random
import logging
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer

from unsloth import FastVisionModel

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =========================================================
# ENV / REPRO
# =========================================================
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# CONFIG
# =========================================================
BASE_MODEL        = "unsloth/Qwen3-VL-8B-Instruct"
MAX_SEQ_LEN       = 2048
INFER_BSZ         = 8
LOCAL_FILES_ONLY  = False

IMG_DIR = Path("./SVs_merged")

instruction = (
    "You are given photos of a house captured from different angles.\n"
    "Your task is to evaluate ONE specific aspect of damage based on what you see.\n"
)

FOCUS = {
    "roof":   "Focus ONLY on the ROOF surface (shingles/holes/missing sections/collapse). Ignore walls/windows/ground.\n",
    "facade": "Focus ONLY on the FACADE (walls/porch/columns). Ignore roof and ground.\n",
    "open":   "Focus ONLY on WINDOWS/DOORS. Ignore roof and most wall texture.\n",
}

# =========================================================
# DIGIT TOKEN IDS  (resolved lazily from tokenizer only —
#                   no model weights are loaded at import)
# =========================================================
DIGIT_STRS = ["1", "2"]
DIGIT_IDS_T: torch.Tensor | None = None   # populated by _ensure_digit_ids()


def _ensure_digit_ids(local_files_only: bool = LOCAL_FILES_ONLY) -> None:
    """Resolve DIGIT_IDS_T once, using the tokenizer only (no model load)."""
    global DIGIT_IDS_T
    if DIGIT_IDS_T is not None:
        return

    log.info("Loading tokenizer to resolve digit token IDs …")
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=local_files_only,
    )

    ids = []
    for s in DIGIT_STRS:
        encoded = tok.encode(s, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"'{s}' is not a single token in this tokenizer. Got ids={encoded}"
            )
        ids.append(encoded[0])

    DIGIT_IDS_T = torch.tensor(ids, dtype=torch.long)
    log.info("Digit token IDs resolved: %s → %s", DIGIT_STRS, ids)


# =========================================================
# CLEANUP
# =========================================================
def hard_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# =========================================================
# IMAGE / PROMPT HELPERS
# =========================================================
EXPECTED_SIZE = (300, 1240)   # (width, height) — update if your dataset changes

def split_merged_3(im: Image.Image, source: str = "") -> list[Image.Image]:
    """
    Split a vertically stacked 3-view image into three equal sub-images.
    If the height is not exactly divisible by 3, at most 2 bottom pixels
    are trimmed so all three crops are the same height.
    Logs a warning when the image dimensions differ from EXPECTED_SIZE.
    """
    im = im.convert("RGB")
    w, h = im.size

    if (w, h) != EXPECTED_SIZE:
        tag = f" ({source})" if source else ""
        log.warning(
            "Unexpected image size %dx%d%s — expected %dx%d.",
            w, h, tag, *EXPECTED_SIZE,
        )

    # If taller than expected, discard everything below row EXPECTED_SIZE[1]
    expected_h = EXPECTED_SIZE[1]
    if h > expected_h:
        im = im.crop((0, 0, w, expected_h))
        h  = expected_h

    h3     = h // 3
    h_trim = h3 * 3          # largest exact multiple of 3 ≤ h (drops at most 2px)
    return [
        im.crop((0, 0,      w, h3)),
        im.crop((0, h3,     w, 2 * h3)),
        im.crop((0, 2 * h3, w, h_trim)),
    ]


def make_prompt(task_key: str, question: str) -> str:
    return (
        f"{instruction}\n"
        f"Task:\n{question}\n\n"
        "You will see 3 views of the SAME house.\n"
        f"{FOCUS[task_key]}"
        "STRICT OUTPUT:\n"
        "- Return EXACTLY 1 digit.\n"
        "- Must be one of: 1 2\n"
        "- Do NOT output anything else (no spaces, no punctuation, no newlines).\n"
    )


# =========================================================
# MESSAGE BUILDER  (per-row errors are collected, not fatal)
# =========================================================
def build_messages_for_row(row, task_key: str, question: str) -> list | None:
    """
    Returns a messages list for the row, or None if the image is missing /
    cannot be opened (the caller decides how to handle None entries).
    """
    img_path = IMG_DIR / row["svi_merged"]
    if not img_path.exists():
        log.warning("Missing image, row will be skipped: %s", img_path)
        return None
    try:
        with Image.open(img_path) as im:
            views = split_merged_3(im, source=str(img_path))
    except Exception as exc:
        log.warning("Failed to open image %s (%s), row will be skipped.", img_path, exc)
        return None

    prompt       = make_prompt(task_key, question)
    user_content = [{"type": "text", "text": prompt}]
    for v in views:
        user_content.append({"type": "image", "image": v})
    return [{"role": "user", "content": user_content}]


def build_messages_list(
    df: pd.DataFrame,
    task_key: str,
    question: str,
) -> tuple[list[list], list[int]]:
    """
    Returns:
        messages  – list of valid message dicts (bad rows excluded)
        valid_idx – original DataFrame indices that correspond to each entry
    """
    messages, valid_idx = [], []
    for idx, row in df.iterrows():
        msgs = build_messages_for_row(row, task_key, question)
        if msgs is not None:
            messages.append(msgs)
            valid_idx.append(idx)

    n_skipped = len(df) - len(valid_idx)
    if n_skipped:
        log.warning("%d row(s) skipped due to missing/unreadable images.", n_skipped)

    return messages, valid_idx


# =========================================================
# MODEL LOADER
# =========================================================
def load_stage_model(model_dir: str, local_files_only: bool = LOCAL_FILES_ONLY):
    hard_cleanup()
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        device_map="auto",
        local_files_only=local_files_only,
    )
    FastVisionModel.for_inference(model)

    # FIX: enforce left-padding — batched inference reads logits at position T-1,
    # which is only the final real token when the tokenizer left-pads.
    assert getattr(processor.tokenizer, "padding_side", "left") == "left", (
        f"Tokenizer in '{model_dir}' uses right-padding — "
        "batched inference will silently read wrong logit positions."
    )

    return model, processor


# =========================================================
# CORE INFERENCE  (OOM-safe with automatic batch-size halving)
# =========================================================
def predict_classes(
    model,
    processor,
    messages_list: list[list],
    batch_size: int = INFER_BSZ,
) -> list[int]:
    """
    Returns one integer prediction per entry in messages_list.

    Padding notes
    -------------
    Qwen / most decoder-only chat models use LEFT-padding when batching,
    so real tokens occupy the *last* `seq_len` positions in the padded
    sequence.  We locate the final real-token position as:

        last_real_pos = total_len - 1

    because with left-padding the last column is always a real token.
    The attention-mask sum is used only to detect fully-padded edge cases.
    """
    assert DIGIT_IDS_T is not None, "Call _ensure_digit_ids() before inference."

    model_dtype  = next(model.parameters()).dtype
    input_device = next(model.parameters()).device
    preds: list[int] = []

    start = 0
    while start < len(messages_list):
        batch_msgs = messages_list[start : start + batch_size]
        try:
            model_inputs = processor.apply_chat_template(
                batch_msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            # FIX: build a new dict instead of mutating while iterating.
            model_inputs = {
                k: (
                    (v.to(model_dtype) if torch.is_floating_point(v) else v).to(input_device)
                    if torch.is_tensor(v) else v
                )
                for k, v in model_inputs.items()
            }

            # FIX: inference_mode as the outer context (stronger than no_grad);
            # autocast nested inside for fp16/bf16 CUDA models.
            with torch.inference_mode():
                if torch.cuda.is_available() and model_dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast(device_type="cuda", dtype=model_dtype):
                        outputs = model(**model_inputs)
                else:
                    outputs = model(**model_inputs)

            logits       = outputs.logits                        # (B, T, V)
            digit_ids_dv = DIGIT_IDS_T.to(logits.device)

            B, T, _ = logits.shape
            attn_mask = model_inputs.get("attention_mask")       # (B, T)

            for b in range(B):
                if attn_mask is not None:
                    # With left-padding, real tokens are at the end of the row.
                    # The last real position is always T-1 (the pad tokens, if
                    # any, are prepended). We verify via the mask.
                    real_len = int(attn_mask[b].sum().item())
                    if real_len == 0:
                        log.warning("Batch item %d has zero real tokens; defaulting to 1.", b)
                        preds.append(1)
                        continue
                    last_real_pos = T - 1          # left-padded → last col is real
                else:
                    last_real_pos = T - 1

                digit_logits = logits[b, last_real_pos, digit_ids_dv]   # (2,)
                pred_idx     = int(torch.argmax(digit_logits).item())
                preds.append(pred_idx + 1)         # 0-indexed → class 1 or 2

            start += batch_size                    # advance only on success

        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise RuntimeError(
                    "OOM even at batch_size=1; reduce MAX_SEQ_LEN or image resolution."
                )
            new_bs = max(1, batch_size // 2)
            log.warning(
                "CUDA OOM at batch_size=%d — retrying with batch_size=%d.", batch_size, new_bs
            )
            hard_cleanup()
            batch_size = new_bs
            # Do NOT advance `start`; retry the same slice with smaller batch.

    return preds


# =========================================================
# TASK CONFIG
# =========================================================
TASK_INFERENCE_CONFIGS: dict[str, dict] = {
    "roof": {
        "s1_model_dir": "models_2stage_qwen/roof_s1",
        "s2_model_dir": "models_2stage_qwen/roof_s2",
        "s1_question":  "ROOF (Stage 1): 1=good_condition, 2=any_damage (small or severe)",
        "s2_question":  "ROOF (Stage 2, damaged only): 1=small_damage, 2=severe_damage",
        "s1_col":       "s1_roof",
        "output_col":   "pred_roof",
    },
    "facade": {
        "s1_model_dir": "models_2stage_qwen/facade_s1",
        "s2_model_dir": "models_2stage_qwen/facade_s2",
        "s1_question":  "FACADE (Stage 1): 1=good_condition, 2=any_damage (small or severe)",
        "s2_question":  "FACADE (Stage 2, damaged only): 1=small_damage, 2=severe_damage",
        "s1_col":       "s1_facade",
        "output_col":   "pred_facade",
    },
    "open": {
        "s1_model_dir": "models_2stage_qwen/open_s1",
        "s2_model_dir": "models_2stage_qwen/open_s2",
        "s1_question":  "OPENINGS (Stage 1): 1=good_condition, 2=any_damage (uncovered/major)",
        "s2_question":  "OPENINGS (Stage 2, damaged only): 1=small_damage, 2=severe_damage",
        "s1_col":       "s1_open",
        "output_col":   "pred_open",
    },
}

# =========================================================
# STEP 1 — Stage 1 inference → intermediate CSV
# =========================================================
def run_stage1(
    df: pd.DataFrame,
    stage1_csv: str = None,
    task_keys: tuple[str, ...] = ("roof", "facade", "open"),
    batch_size: int = INFER_BSZ,
    export: bool = True,
) -> pd.DataFrame:
    """
    Runs Stage 1 for every task; writes one `s1_<task>` column per task.
    Saves an intermediate CSV so you can inspect results before Stage 2.

    Args:
        df          : Input DataFrame (must contain 'svi_merged' column).
        stage1_csv  : Path for the intermediate output file.
        task_keys   : Which damage aspects to evaluate.
        batch_size  : Initial inference batch size (halved automatically on OOM).
        export      : Write CSV when True (set False when caller handles saving).

    Returns:
        df with `s1_<task>` columns added in-place.
    """
    _ensure_digit_ids()
    df = df.copy().reset_index(drop=True)

    for task in task_keys:
        hard_cleanup()
        cfg = TASK_INFERENCE_CONFIGS[task]
        log.info("=" * 60)
        log.info("Stage 1 — Task: %s", task.upper())
        log.info("Loading model: %s", cfg["s1_model_dir"])

        m1, p1 = load_stage_model(cfg["s1_model_dir"])
        messages, valid_idx = build_messages_list(df, task, cfg["s1_question"])

        raw_preds = predict_classes(m1, p1, messages, batch_size=batch_size)
        m1 = p1 = None
        hard_cleanup()

        # Write predictions back at their original row indices; skipped rows → NaN
        s1_col = cfg["s1_col"]
        df[s1_col] = np.nan
        for idx, pred in zip(valid_idx, raw_preds):
            df.at[idx, s1_col] = pred

        n_forward = int((df[s1_col] == 2).sum())
        log.info(
            "Distribution: %s  →  %d rows flagged for Stage 2",
            df[s1_col].value_counts().sort_index().to_dict(),
            n_forward,
        )

    if export:
        df.to_csv(stage1_csv, index=False)
        log.info("Stage 1 complete. Intermediate file saved to '%s'", stage1_csv)
    return df


# =========================================================
# STEP 2 — Stage 2 inference driven by Stage 1 predictions
# =========================================================
def run_stage2(
    df: pd.DataFrame,
    output_csv: str = None,
    task_keys: tuple[str, ...] = ("roof", "facade", "open"),
    batch_size: int = INFER_BSZ,
    export: bool = True,
) -> pd.DataFrame:
    """
    For each task, only rows where `s1_<task>` == 2 are passed through
    Stage 2. Final 3-class label written to `pred_<task>`:
        1 = no damage   (Stage 1 predicted 1)
        2 = minor       (Stage 1 predicted 2, Stage 2 predicted 1)
        3 = severe      (Stage 1 predicted 2, Stage 2 predicted 2)

    Args:
        df          : DataFrame with `s1_<task>` columns from run_stage1().
                      Must have been reset_index(drop=True) beforehand —
                      run_stage1() guarantees this for its return value.
        output_csv  : Path for the final output file.
        task_keys   : Must match what was used in run_stage1().
        batch_size  : Initial inference batch size (halved automatically on OOM).
        export      : Write CSV when True (set False when caller handles saving).

    Returns:
        df with `pred_<task>` columns added in-place.
    """
    _ensure_digit_ids()
    df = df.copy().reset_index(drop=True)

    for task in task_keys:
        hard_cleanup()
        cfg    = TASK_INFERENCE_CONFIGS[task]
        s1_col = cfg["s1_col"]

        if s1_col not in df.columns:
            raise ValueError(
                f"Column '{s1_col}' not found. Run run_stage1() first."
            )

        s1_series  = df[s1_col]
        output_col = cfg["output_col"]

        # Pre-fill final predictions: Stage-1 class 1 → final class 1 (no damage)
        final_preds = pd.Series(index=df.index, dtype="Int64")
        final_preds[s1_series == 1] = 1

        stage2_mask = s1_series == 2
        n_forward   = int(stage2_mask.sum())
        n_skip      = int(s1_series.isna().sum())

        log.info("=" * 60)
        log.info("Stage 2 — Task: %s", task.upper())
        log.info(
            "%d/%d rows forwarded from Stage 1 (%d skipped due to missing images)",
            n_forward, len(df), n_skip,
        )

        if n_forward > 0:
            log.info("Loading model: %s", cfg["s2_model_dir"])
            m2, p2 = load_stage_model(cfg["s2_model_dir"])

            # Note: stage2_df inherits the reset integer index from df, so
            # valid_idx values returned by build_messages_list are valid
            # positional labels into df.index and final_preds.index.
            stage2_df           = df[stage2_mask]
            messages, valid_idx = build_messages_list(stage2_df, task, cfg["s2_question"])

            s2_raw = predict_classes(m2, p2, messages, batch_size=batch_size)
            m2 = p2 = None
            hard_cleanup()

            # Map Stage-2 output → final 3-class label
            for orig_idx, s2_pred in zip(valid_idx, s2_raw):
                final_preds[orig_idx] = 2 if s2_pred == 1 else 3

        df[output_col] = final_preds
        log.info(
            "Final distribution: %s",
            df[output_col].value_counts(dropna=False).sort_index().to_dict(),
        )

    if export:
        df.to_csv(output_csv, index=False)
        log.info("Stage 2 complete. Final predictions saved to '%s'", output_csv)
    return df


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage inference of house damage.")
    parser.add_argument("-i", "--input_csv", type=str, required=True,
                        help="CSV file with a 'svi_merged' column of image filenames.")
    parser.add_argument("-r", "--run_name", type=str, required=True,
                        help="Tag used to name the output files (e.g. 'exp01').")
    parser.add_argument("-t", "--tasks", nargs="+",
                        default=["roof", "facade", "open"],
                        choices=["roof", "facade", "open"],
                        help="Which damage aspects to evaluate (default: all three).")
    parser.add_argument("-b", "--batch_size", type=int, default=INFER_BSZ,
                        help="Initial inference batch size (auto-halved on OOM).")
    parser.add_argument("-s", "--split_size", type=int, default=None,
                        help="Split the input DataFrame into N parts to reduce peak memory usage.")
    args = parser.parse_args()

    task_keys  = tuple(args.tasks)
    input_csv  = Path(args.input_csv)
    stage1_csv = f"stage1_predictions_{args.run_name}.csv"
    output_csv = f"inference_output_{args.run_name}.csv"

    df = pd.read_csv(input_csv).reset_index(drop=True)
    log.info("Loaded %d rows from '%s'", len(df), input_csv)

    if args.split_size is not None:
        stage1_dfs = []
        output_dfs = []
        chunk_size = max(1, len(df) // args.split_size)
        chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
        for sub_df in chunks:
            df_1 = run_stage1(
                df=sub_df,
                task_keys=task_keys,
                batch_size=args.batch_size,
                export=False,
            )
            stage1_dfs.append(df_1)
            df_2 = run_stage2(
                df=df_1,
                task_keys=task_keys,
                batch_size=args.batch_size,
                export=False,
            )
            output_dfs.append(df_2)

        pd.concat(stage1_dfs, ignore_index=True).to_csv(stage1_csv, index=False)
        pd.concat(output_dfs, ignore_index=True).to_csv(output_csv, index=False)
        log.info("Split-mode complete. Results saved to '%s' and '%s'.", stage1_csv, output_csv)

    else:
        df = run_stage1(
            df=df,
            stage1_csv=stage1_csv,
            task_keys=task_keys,
            batch_size=args.batch_size,
        )
        run_stage2(
            df=df,
            output_csv=output_csv,
            task_keys=task_keys,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
