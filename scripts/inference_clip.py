"""
Two-stage cascade inference with fine-tuned CLIP.

Mirrors inference_qwen.py in structure and output format:
  s1_{task}   — Stage 1 binary:   0 = good, 1 = damaged
  pred_{task} — Final 3-class:    0 = no damage, 1 = minor, 2 = severe

Usage:
  python inference_clip.py -i input.csv -r exp01
  python inference_clip.py -i input.csv -r exp01 --tasks roof facade
  python inference_clip.py -i input.csv -r exp01 --model ViT-B/32 --batch-size 32
  python inference_clip.py -i input.csv -r exp01 --model-dir /path/to/models --img-dir /path/to/images
"""

import gc
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse

# ---------------------------------------------------------------------------
# Import model class from the training module.
# Clip_single.py has harmless module-level side-effects (seed, mkdir) only.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Clip_single import MultiViewCLIPClassifier

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MEAN  = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD   = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_MODEL      = "ViT-B/16"
DEFAULT_MODE       = "triad"
DEFAULT_INPUT_MODE = "vision"
DEFAULT_BSZ        = 16
DEFAULT_MODEL_DIR  = Path("clip_two_stage_models")
DEFAULT_IMG_DIR    = Path(r"D:\blight_fintune\SVs_merged")

# Expected raw image size — matches inference_qwen.py
EXPECTED_SIZE = (300, 1240)   # (width, height)

TASK_CONFIGS = {
    "roof":     {"s1_col": "s1_roof",     "pred_col": "pred_roof"},
    "facade":   {"s1_col": "s1_facade",   "pred_col": "pred_facade"},
    "openings": {"s1_col": "s1_openings", "pred_col": "pred_openings"},
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def split_merged_3(im: Image.Image, source: str = "") -> list[Image.Image]:
    """Split a vertically-stacked 3-view image into three equal sub-images."""
    im = im.convert("RGB")
    w, h = im.size
    if (w, h) != EXPECTED_SIZE:
        log.warning("Unexpected size %dx%d%s — expected %dx%d.",
                    w, h, f" ({source})" if source else "", *EXPECTED_SIZE)
    expected_h = EXPECTED_SIZE[1]
    if h > expected_h:
        im = im.crop((0, 0, w, expected_h))
        h  = expected_h
    h3     = h // 3
    h_trim = h3 * 3
    return [
        im.crop((0, 0,      w, h3)),
        im.crop((0, h3,     w, 2 * h3)),
        im.crop((0, 2 * h3, w, h_trim)),
    ]


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset  (inference-only: no label column required)
# ---------------------------------------------------------------------------
class InferenceDataset(Dataset):
    """Loads merged 3-view images and returns preprocessed view tensors."""

    def __init__(self, filenames: list[str], img_dir: Path, image_size: int):
        self.filenames = filenames
        self.img_dir   = Path(img_dir)
        self.transform = make_transform(image_size)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname    = self.filenames[idx]
        img_path = self.img_dir / fname
        with Image.open(img_path) as im:
            views = split_merged_3(im, source=fname)
        x = torch.stack([self.transform(v) for v in views], dim=0)  # (3, C, H, W)
        return x, fname


def collate_inference(batch):
    xs, names = zip(*batch)
    return torch.stack(xs, dim=0), list(names)


def build_inference_loader(
    filenames: list[str], img_dir: Path,
    batch_size: int, image_size: int, num_workers: int,
) -> DataLoader:
    ds = InferenceDataset(filenames, img_dir, image_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_inference,
        persistent_workers=False,
    )


# ---------------------------------------------------------------------------
# File validation (done upfront — avoids __getitem__ errors mid-batch)
# ---------------------------------------------------------------------------
def validate_files(
    filenames: list[str], img_dir: Path,
) -> tuple[list[str], list[int]]:
    """Return (valid_filenames, valid_df_indices), logging any missing files."""
    valid_fnames: list[str] = []
    valid_idxs:  list[int]  = []
    for i, fname in enumerate(filenames):
        if (img_dir / fname).exists():
            valid_fnames.append(fname)
            valid_idxs.append(i)
        else:
            log.warning("Missing image (skipped): %s", img_dir / fname)
    n_skip = len(filenames) - len(valid_fnames)
    if n_skip:
        log.warning("%d image(s) skipped due to missing files.", n_skip)
    return valid_fnames, valid_idxs


# ---------------------------------------------------------------------------
# Model loading  (standalone — all hyperparams read from checkpoint)
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path):
    """Load a MultiViewCLIPClassifier from a .pt checkpoint file."""
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MultiViewCLIPClassifier(
        num_classes     = int(ckpt["num_classes"]),
        clip_model_name = ckpt["clip_model_name"],
        input_mode      = ckpt.get("input_mode",     "vision"),
        view_agg        = ckpt.get("view_agg",        "concat"),
        dropout         = float(ckpt.get("dropout",   0.25)),
        head_hidden_dim = int(ckpt.get("head_hidden_dim", 512)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    log.info("  Loaded: %s  (classes=%d, clip=%s)",
             path, ckpt["num_classes"], ckpt["clip_model_name"])
    return model, ckpt


def resolve_checkpoint(
    model_dir: Path, task: str, stage: str,
    mode: str, model_name: str, input_mode: str,
) -> Path:
    safe = model_name.replace("/", "_").replace("@", "_")
    p = model_dir / f"{task}_{stage}_{mode}_{safe}_{input_mode}" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {p}\n"
            f"  → Train with: python benchmark_clip.py --models {model_name} --tasks {task}"
        )
    return p


# ---------------------------------------------------------------------------
# Core inference  (OOM-safe: halves batch_size on CUDA OOM)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_stage(
    model,
    filenames: list[str],
    img_dir: Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> dict[str, int]:
    """
    Returns {filename: predicted_class_index} for every filename.
    Automatically halves batch_size on CUDA OOM.
    """
    model.eval()
    results: dict[str, int] = {}
    use_amp = torch.cuda.is_available()

    start = 0
    while start < len(filenames):
        batch_fnames = filenames[start : start + batch_size]
        try:
            loader = build_inference_loader(batch_fnames, img_dir, batch_size, image_size, num_workers)
            for x, names in loader:
                x = x.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x, None)   # vision-only; text=None
                preds = logits.argmax(dim=1).cpu().tolist()
                for name, pred in zip(names, preds):
                    results[name] = int(pred)
            start += batch_size

        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise RuntimeError("CUDA OOM even at batch_size=1.")
            new_bs = max(1, batch_size // 2)
            log.warning("CUDA OOM at batch_size=%d — retrying with batch_size=%d.",
                        batch_size, new_bs)
            hard_cleanup()
            batch_size = new_bs
            # do NOT advance start — retry the same slice

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hard_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage 1 — binary good vs. damaged
# ---------------------------------------------------------------------------
def run_stage1(
    df: pd.DataFrame,
    model_dir: Path,
    img_dir: Path,
    model_name: str,
    mode: str,
    input_mode: str,
    tasks: list[str],
    batch_size: int,
    image_size: int,
    num_workers: int,
    stage1_csv: Optional[str] = None,
    export: bool = True,
) -> pd.DataFrame:
    """
    Runs Stage 1 for every task and writes one s1_{task} column per task.
    Labels: 0 = good condition, 1 = any damage.
    """
    df = df.copy().reset_index(drop=True)
    all_filenames = df["svi_merged"].tolist()

    for task in tasks:
        hard_cleanup()
        log.info("=" * 60)
        log.info("Stage 1 — Task: %s", task.upper())

        ckpt_path = resolve_checkpoint(model_dir, task, "s1", mode, model_name, input_mode)
        model, _  = load_checkpoint(ckpt_path)

        valid_fnames, valid_idxs = validate_files(all_filenames, img_dir)
        preds = predict_stage(model, valid_fnames, img_dir, batch_size, image_size, num_workers)
        del model
        hard_cleanup()

        s1_col = TASK_CONFIGS[task]["s1_col"]
        df[s1_col] = pd.NA
        for fname, idx in zip(valid_fnames, valid_idxs):
            df.at[idx, s1_col] = preds.get(fname)

        n_damaged = int((df[s1_col] == 1).sum())
        n_skipped = int(df[s1_col].isna().sum())
        log.info("Distribution: %s  →  %d forwarded to Stage 2  (%d skipped)",
                 df[s1_col].value_counts(dropna=False).sort_index().to_dict(),
                 n_damaged, n_skipped)

    if export and stage1_csv:
        df.to_csv(stage1_csv, index=False)
        log.info("Stage 1 complete. Saved to '%s'", stage1_csv)

    return df


# ---------------------------------------------------------------------------
# Stage 2 — severity for damaged buildings
# ---------------------------------------------------------------------------
def run_stage2(
    df: pd.DataFrame,
    model_dir: Path,
    img_dir: Path,
    model_name: str,
    mode: str,
    input_mode: str,
    tasks: list[str],
    batch_size: int,
    image_size: int,
    num_workers: int,
    output_csv: Optional[str] = None,
    export: bool = True,
) -> pd.DataFrame:
    """
    For each task, only rows where s1_{task} == 1 are passed through Stage 2.

    Final 3-class label in pred_{task}:
      0 = no damage   (Stage 1 predicted 0)
      1 = minor       (Stage 1 predicted 1, Stage 2 predicted 0)
      2 = severe      (Stage 1 predicted 1, Stage 2 predicted 1)
    """
    df = df.copy().reset_index(drop=True)

    for task in tasks:
        hard_cleanup()
        cfg     = TASK_CONFIGS[task]
        s1_col  = cfg["s1_col"]
        out_col = cfg["pred_col"]

        if s1_col not in df.columns:
            raise ValueError(f"Column '{s1_col}' missing — run run_stage1() first.")

        stage2_mask = df[s1_col] == 1
        n_forward   = int(stage2_mask.sum())
        n_skip      = int(df[s1_col].isna().sum())

        log.info("=" * 60)
        log.info("Stage 2 — Task: %s  |  %d/%d rows forwarded  (%d skipped)",
                 task.upper(), n_forward, len(df), n_skip)

        # Pre-fill: stage1=0 → final label 0 (no damage)
        final = pd.array([pd.NA] * len(df), dtype="Int64")
        final[df[s1_col] == 0] = 0

        if n_forward > 0:
            ckpt_path = resolve_checkpoint(model_dir, task, "s2", mode, model_name, input_mode)
            model, _  = load_checkpoint(ckpt_path)

            s2_filenames = df.loc[stage2_mask, "svi_merged"].tolist()
            s2_df_idxs   = list(df.index[stage2_mask])

            valid_fnames, valid_local_idxs = validate_files(s2_filenames, img_dir)
            preds = predict_stage(model, valid_fnames, img_dir, batch_size, image_size, num_workers)
            del model
            hard_cleanup()

            # Map valid Stage 2 predictions → final label
            valid_df_idxs = [s2_df_idxs[i] for i in valid_local_idxs]
            for fname, df_idx in zip(valid_fnames, valid_df_idxs):
                raw = preds.get(fname)
                if raw is not None:
                    # Stage 2: 0 → minor (1),  1 → severe (2)
                    final[df_idx] = raw + 1

        df[out_col] = final
        log.info("Final distribution: %s",
                 df[out_col].value_counts(dropna=False).sort_index().to_dict())

    if export and output_csv:
        df.to_csv(output_csv, index=False)
        log.info("Stage 2 complete. Saved to '%s'", output_csv)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage cascade inference with fine-tuned CLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-csv",  required=True,
                        help="CSV with a 'svi_merged' column of image filenames.")
    parser.add_argument("-r", "--run-name",   required=True,
                        help="Tag used to name output files (e.g. 'exp01').")
    parser.add_argument("-t", "--tasks",      nargs="+",
                        default=["roof", "facade", "openings"],
                        choices=["roof", "facade", "openings"],
                        help="Damage aspects to evaluate.")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BSZ,
                        help="Inference batch size (auto-halved on CUDA OOM).")
    parser.add_argument("--model",            default=DEFAULT_MODEL,
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP model variant.")
    parser.add_argument("--input-mode",       default=DEFAULT_INPUT_MODE,
                        choices=["vision", "text", "both"],
                        help="Input modality used during training.")
    parser.add_argument("--model-dir",        default=str(DEFAULT_MODEL_DIR),
                        help="Root directory for trained checkpoints.")
    parser.add_argument("--img-dir",          default=str(DEFAULT_IMG_DIR),
                        help="Directory containing merged street-view images.")
    parser.add_argument("--num-workers",      type=int, default=2,
                        help="DataLoader worker processes.")
    args = parser.parse_args()

    model_dir  = Path(args.model_dir)
    img_dir    = Path(args.img_dir)
    stage1_csv = f"stage1_predictions_{args.run_name}.csv"
    output_csv = f"inference_output_{args.run_name}.csv"
    image_size = 336 if "336" in args.model else 224

    log.info("=" * 60)
    log.info("CLIP Inference — %s  |  mode=%s  |  device=%s", args.model, DEFAULT_MODE, DEVICE)
    log.info("img_dir:   %s", img_dir)
    log.info("model_dir: %s", model_dir)
    log.info("tasks:     %s", args.tasks)
    log.info("=" * 60)

    df = pd.read_csv(args.input_csv).reset_index(drop=True)
    log.info("Loaded %d rows from '%s'", len(df), args.input_csv)

    if "svi_merged" not in df.columns:
        raise ValueError("Input CSV must contain a 'svi_merged' column.")

    shared = dict(
        model_dir  = model_dir,
        img_dir    = img_dir,
        model_name = args.model,
        mode       = DEFAULT_MODE,
        input_mode = args.input_mode,
        tasks      = args.tasks,
        batch_size = args.batch_size,
        image_size = image_size,
        num_workers= args.num_workers,
    )

    df = run_stage1(df, stage1_csv=stage1_csv, export=True, **shared)
    run_stage2(df, output_csv=output_csv, export=True, **shared)


if __name__ == "__main__":
    main()
