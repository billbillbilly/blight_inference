# Blight Inference

Detect residential blight/damage conditions in Detroit from street view imagery (SVI) using fine-tuned Qwen vision-language models.

## Pipeline

1. Download Mapillary street view images for residential parcels (`scripts/svi.py`)
2. Detect houses in the images with Mask2Former (`scripts/detect_house.py`)
3. Merge perspectives and prepare inference inputs (`scripts/prepare_data.py`)
4. Run damage-condition inference with fine-tuned Qwen models (`scripts/inference_qwen.py`)

## Requirements

- OS: 64-bit Linux (Ubuntu 20.04+) or Windows 10/11
- Python: 3.11–3.13 (3.12 recommended)
- CUDA Toolkit: 11.8 or 12.1+ recommended (CUDA 12.8+ required for NVIDIA Blackwell GPUs)
- PyTorch: 2.0+ built with CUDA support matching your drivers
- Core dependencies: `triton`, `xformers`, `bitsandbytes`, `unsloth`

## Installation

### Option 1: conda environment file (recommended)

```sh
conda env create -f environment.yml
conda activate blight_inference
```

> Note: `environment.yml` installs PyTorch from the CUDA 12.1 wheel index. If your system needs a different CUDA build (e.g. `cu128`, `cu130`), edit the `--extra-index-url` line accordingly.

### Option 2: manual setup

```sh
conda create --name blight_inference python=3.12 -y
conda activate blight_inference

# PyTorch with CUDA (pick the index matching your CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
pip install unsloth transformers bitsandbytes xformers
pip install geopandas urban-worm pandas numpy pillow tqdm huggingface_hub
```

### Verify GPU setup

Check the NVIDIA driver is working (reinstall drivers if this fails):

```sh
nvidia-smi
```

Test that PyTorch sees the GPU:

```python
import torch
print(torch.cuda.is_available())  # should print True
```

## Checkpoints

Fine-tuned Qwen checkpoints are hosted on Hugging Face:

| HF repo | URL |
|---|---|
| xiaohaoy/qwen_facade_s1 | https://huggingface.co/xiaohaoy/qwen_facade_s1 |
| xiaohaoy/qwen_facade_s2 | https://huggingface.co/xiaohaoy/qwen_facade_s2 |
| xiaohaoy/qwen_open_s1 | https://huggingface.co/xiaohaoy/qwen_open_s1 |
| xiaohaoy/qwen_open_s2 | https://huggingface.co/xiaohaoy/qwen_open_s2 |
| xiaohaoy/qwen_roof_s1 | https://huggingface.co/xiaohaoy/qwen_roof_s1 |
| xiaohaoy/qwen_roof_s2 | https://huggingface.co/xiaohaoy/qwen_roof_s2 |

### Download checkpoints

Download all checkpoints with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

repos = [
    "xiaohaoy/qwen_facade_s1",
    "xiaohaoy/qwen_facade_s2",
    "xiaohaoy/qwen_open_s1",
    "xiaohaoy/qwen_open_s2",
    "xiaohaoy/qwen_roof_s1",
    "xiaohaoy/qwen_roof_s2",
]

for repo_id in repos:
    local_dir = f"models/{repo_id.split('/')[-1]}"
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Downloaded {repo_id} -> {local_dir}")
```

Or with the Hugging Face CLI:

```sh
pip install -U "huggingface_hub[cli]"

for repo in qwen_facade_s1 qwen_facade_s2 qwen_open_s1 qwen_open_s2 qwen_roof_s1 qwen_roof_s2; do
    hf download "xiaohaoy/${repo}" --local-dir "models/${repo}"
done
```

## Usage

To detect residential damage conditions in Detroit, run the scripts in sequence:

```sh
bash 1_get_mapillary_svi.sh   # download Mapillary SVI for residential parcels
bash 2_detect_house.sh        # detect houses with Mask2Former
bash 3_prepare_data.sh        # merge perspectives, prepare inference inputs
bash 4_inference.sh           # run Qwen damage-condition inference
```

Notes:

- `1_get_mapillary_svi.sh` requires a Mapillary API key (passed to `scripts/svi.py` via `--key`) and input data in `data/` (`buildings.geojson`, `zoning.geojson`).
- Outputs from each stage feed the next; run them in order.

## Acknowledgement

The project was supported by the City of Detroit. We acknowledge the blight survey data provided by the Detroit Land Bank Authority (DLBA).
