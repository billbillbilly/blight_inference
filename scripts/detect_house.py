import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# -----------------------
# 0) Device + precision
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

use_amp = (device.type == "cuda")  # mixed precision on GPU
amp_dtype = torch.float16          # fp16; you can switch to bfloat16 if your GPU supports it

# -----------------------
# 1) Load model/processor
# -----------------------
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    cache_dir="models",
)

model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-mapillary-vistas-semantic",
    cache_dir="models",
)
model.to(device)
model.eval()

# Optional: allow cuDNN autotuning for fixed image sizes (may help)
torch.backends.cudnn.benchmark = True


def detect_building(
    image_path,
    model=model,
    processor=processor,
    building_id=17,
    patch_fraction=(0.60, 0.30),
    threshold=0.20,
    check=False,
):
    # Always open as RGB (sometimes PNG has alpha)
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        # processor returns CPU tensors by default
        inputs = processor(images=image, return_tensors="pt")

    # Move inputs to GPU
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    # Forward pass on GPU
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    # Post-process (this returns a CPU tensor map in most HF versions)
    # target_sizes expects (height, width)
    target_sizes = [(image.size[1], image.size[0])]
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=target_sizes
    )[0]

    # Ensure tensor on CPU for slicing (it likely already is)
    if isinstance(predicted_semantic_map, torch.Tensor) and predicted_semantic_map.is_cuda:
        predicted_semantic_map = predicted_semantic_map.cpu()

    H, W = predicted_semantic_map.shape
    cx, cy = W // 2, H // 2

    wx = max(1, int(patch_fraction[0] * W))
    wy = max(1, int(patch_fraction[1] * H))

    x0, x1 = max(0, cx - wx // 2), min(W, cx + wx // 2)
    y0, y1 = max(0, cy - wy // 2), min(H, cy + wy // 2)

    patch = predicted_semantic_map[y0:y1, x0:x1]
    # patch is int labels; compute ratio on CPU
    ratio = (patch == building_id).float().mean().item()

    is_building = ratio >= threshold

    if check:
        print(f"building ratio in center patch: {ratio:.3f}")
        print("building around center?", is_building)

    return is_building, ratio


# -----------------------
# 2) Run batch
# -----------------------
dic = {
    "file": [],
    "mapillary_id": [],
    "parcel_id": [],
    "is_building": [],
    "ratio": [],
}

files = sorted(os.listdir("svi_computed"))
files = [os.path.join("svi_computed", f) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for file in tqdm(files, total=len(files)):
    filename = os.path.basename(file)
    parcel_id = filename.split("_")[0]
    mapillary_id = filename.split("_")[1].split(".png")[0]

    is_building, ratio = detect_building(file)

    dic["file"].append(file)
    dic["mapillary_id"].append(mapillary_id)
    dic["parcel_id"].append(parcel_id)
    dic["is_building"].append(is_building)
    dic["ratio"].append(ratio)

df = pd.DataFrame(dic)
df.to_csv("building_detection_results.csv", index=False)
df_stats = df.groupby("parcel_id", as_index=False).agg(detected_building=("is_building", "sum"))
df_stats[df_stats["detected_building"] == 1].to_csv("one_building_detected.csv", index=False)
df_stats[df_stats["detected_building"] == 2].to_csv("two_building_detected.csv", index=False)
df_stats[df_stats["detected_building"] == 3].to_csv("three_building_detected.csv", index=False)

# df_stats = df.groupby("parcel_id", as_index=False).agg(detected_building=("is_building", "sum"))
# os.makedirs("detection", exist_ok=True)
# df_stats[df_stats["detected_building"] >= 1].to_csv("detection/1more_buildings_detected.csv", index=False)
# df_stats[df_stats["detected_building"] >= 2].to_csv("detection/2more_buildings_detected.csv", index=False)
# df_stats[df_stats["detected_building"] >= 3].to_csv("detection/3more_buildings_detected.csv", index=False)
