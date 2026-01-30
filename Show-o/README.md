# Show-o Training Guide

> **⚠️ Important Warning**: Currently, this training code **only supports single-GPU training**. Our training implementation calls the forward pass twice in one backward pass (once for image2text and once for reconstruction), which may cause issues with multi-GPU distributed training. Please use a single GPU for training.

> Note: This repository is heavily based on [showlab/Show-o](https://github.com/showlab/Show-o). This repository provides training code and evaluation scripts for the RecA-enhanced version. For the original implementation and additional resources, please refer to the original repository.

This guide provides comprehensive instructions for training Show-o models with Reconstruction Alignment (RecA).

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Preparation](#model-preparation)
- [Training Configuration](#training-configuration)
- [Training](#training)
- [Evaluation](#evaluation)

## Installation

### Environment Setup

```bash
conda create -n showo python=3.10
conda activate showo
pip install -r requirements.txt
```

## Data Preparation

### LLaVA tuning data

Show-o training needs image-text pairs in the LLaVA format. Organize your data as follows:

```
data/
├── LLaVA-Instruct-150K/
│   └── llava_v1_5_mix665k.json
│   └── tuning_data/
│       ├── coco/
│       ├── gqa/
│       ├── ocr_vqa/
│       ├── textvqa/
│       └── vg/
```

If you don't want to use this format, please modify the data loading logic in lines 153-154 of `llava/llava_data_vq_unified.py`.

```python
data_file_path = "data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"
self.image_root = "data/LLaVA-Instruct-150K/tuning_data"
```

The JSON file should contain LLaVA format data with image paths and conversations. You can download the LLaVA dataset from [here](https://huggingface.co/datasets/sanaka87/LLaVA-Instruct-150K).

### Reconstruction data

Our script will download the required reconstruction dataset `brivangl/midjourney-v6-llava` automatically.

## Training Configuration

Show-o provides two architectures:

### CLIP variant

This is the primary approach used in the paper, which uses CLIP visual encoder for reconstruction.

**Configuration files:**

- `configs/showo_reca_clip.yaml`. Resolution of output images is 512×512.

- `configs/showo_reca_256_clip.yaml`. Resolution of output images is 256×256.

### VQGAN variant (Appendix)

This approach uses the VQGAN tokenizer for reconstruction. We implement two strategies:

- Configuration: `configs/showo_reca.yaml`. Input images are resized to 256×256 for reconstruction

- Configuration: `configs/showo_reca_blur8.yaml`. Input images are downsampled by 8× then upsampled back (creates blur effect).

![alt text](assets/VQGAN.png)


## Training

Configure distributed training in `accelerate_configs/0.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 1
gpu_ids: "0"
```

### CLIP Variant (Recommended)

This is the main approach from the paper.

```bash
export PYTHONPATH=.
accelerate launch \
    --config_file accelerate_configs/0.yaml \
    --main_process_port=8881 \
    training/train_recon_w_clip_vit.py \
    config=configs/showo_reca_clip.yaml
```


### VQGAN Variant

```bash
export PYTHONPATH=.
accelerate launch \
    --config_file accelerate_configs/0.yaml \
    --main_process_port=8881 \
    training/train_recon.py \
    config=configs/showo_reca.yaml
```

## Evaluation

### Image Reconstruction Demo

Show-o provides reconstruction demo scripts for both CLIP and VQGAN variants. These scripts allow you to visualize the reconstruction quality of your trained models.

#### CLIP Variant Reconstruction

Use this script to perform image reconstruction with the CLIP-based model:

```bash
export PYTHONPATH=.
python inference_recon_clip.py \
    config=configs/showo_demo_w_clip_vit_512x512.yaml \
    model.showo.pretrained_model_path=/path/to/checkpoint-xxxx/unwrapped_model \
    batch_size=1 \
    guidance_scale=5 \
    generation_timesteps=30
```

**Key parameters:**
- `config`: Configuration file for CLIP variant (512×512 or other resolutions)
- `model.showo.pretrained_model_path`: Path to your trained checkpoint
- `guidance_scale`: Classifier-free guidance scale (default: 5)
- `generation_timesteps`: Number of denoising steps (default: 30)

The script will automatically process all images in the `recon_validation/` directory and save reconstructed images with `_recon.jpg` suffix.

#### VQGAN Variant Reconstruction

Use this script to perform image reconstruction with the VQGAN-based model:

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python inference_recon.py \
    config=configs/showo_demo_512x512.yaml \
    model.showo.pretrained_model_path=/path/to/checkpoint-xxxx/unwrapped_model \
    batch_size=1 \
    guidance_scale=5 \
    generation_timesteps=20
```

**Key parameters:**
- `config`: Configuration file for VQGAN variant
- `model.showo.pretrained_model_path`: Path to your trained checkpoint
- `guidance_scale`: Classifier-free guidance scale (default: 5)
- `generation_timesteps`: Number of denoising steps (default: 20)

**Note**: The VQGAN variant script includes options for different downsampling strategies:
- Standard resize: Direct resizing to lower resolution
- Blur strategy: 8× downsample followed by upsample (see `showo_reca_blur8.yaml`)

### Unified Inference Script

Show-o provides a unified inference script that supports multiple evaluation frameworks. Our script will automatically calculate the number of samples per prompt (12 for GenEval, 4 for DPG, 1 for WISE). 

```bash
export PYTHONPATH=.
python scripts/inference_benchmark.py \
    /path/to/your/checkpoint-xxxx/unwrapped_model \
    --config configs/showo_demo_512x512.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --framework geneval # (or dpg or wise) 
    # (optional) --num_samples 12 

```

You can evaluate multiple checkpoints in sequence. For example:

```bash
python scripts/inference_benchmark.py \
    /path/to/your/checkpoint-1000/unwrapped_model \
    /path/to/your/checkpoint-2000/unwrapped_model \
    /path/to/your/checkpoint-3000/unwrapped_model \
    --framework geneval \
    --config configs/showo_demo_512x512.yaml
    ...
```

