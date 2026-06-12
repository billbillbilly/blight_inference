## Requirement
- Operating System: 64-bit Linux (Ubuntu 20.04+) or Windows 10/11.
- Python: Version 3.11 up to (but not including) 3.14. Python 3.13 is fully supported.
- CUDA Toolkit: CUDA 11.8 or 12.1+ is recommended. If you are running newer NVIDIA Blackwell series GPUs, CUDA 12.8+ is required.
- PyTorch: Version 2.0 or newer (must be explicitly compiled with CUDA support matching your system drivers).
- Core Dependencies: The environment must support triton, xformers, and bitsandbytes

```sh
conda create --name unsloth_env python==3.12 -y
conda activate unsloth_env
```

After typing nvidia-smi in Powershell, you should see something like below. If you don't have nvidia-smi or the below fails to pop up, you need to reinstall NVIDIA drivers.
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```
Test if PyTorch is installed properly with CUDA enabled:
```python
import torch
print(torch.cuda.is_available())
```
Install Unsloth
```sh
pip install unsloth
```

## Checkpoints
|HF repo|URLs|
|---|---|
|xiaohaoy/qwen_facade_s1|https://huggingface.co/xiaohaoy/qwen_facade_s1|
|xiaohaoy/qwen_facade_s2|https://huggingface.co/xiaohaoy/qwen_facade_s2|
|xiaohaoy/qwen_open_s1|https://huggingface.co/xiaohaoy/qwen_open_s1|
|xiaohaoy/qwen_open_s2|https://huggingface.co/xiaohaoy/qwen_open_s2|
|xiaohaoy/qwen_roof_s1|https://huggingface.co/xiaohaoy/qwen_roof_s1|
|xiaohaoy/qwen_roof_s2|https://huggingface.co/xiaohaoy/qwen_roof_s2|

## Usage
To detect the residential damage condition in Detroit, run the following scripts in the sequence:
- 1_get_mapillary_svi.sh
- 2_detect_house.sh
- 3_prepare_data.sh
- 4_inference.sh

## Acknowledgement
The project was supported by the city of Detroit. We acknowledge the blight survey data provided by Detroit Land Bank Authority (DLBA)
