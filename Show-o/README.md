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

### Checkpoints

Checkpoints are saved in:

```
checkpoint-xxxx/
├── unwrapped_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── ...
```

### Visualization

Use the scripts below to perform image reconstruction with the Show-o model:

#### CLIP Variant Reconstruction

```bash
export PYTHONPATH=.
python scripts/inference_recon_clip.py \
    config=configs/showo_demo_w_clip_vit_512x512.yaml \
    model.showo.pretrained_model_path=sanaka87/Show-o-512x512-RecA \
    input_image=inpainting_validation/bedroom.jpg \
    output_image=results/bedroom_recon.jpg \
    generation_timesteps=8 \
    guidance_scale=1.75
```

### Benchmark Evaluation

We provides a unified inference script that supports multiple evaluation frameworks. Our script will automatically calculate the number of samples per prompt (12 for GenEval, 4 for DPG, 1 for WISE).

#### GenEval

```bash
export PYTHONPATH=.
python scripts/inference_benchmark.py \
    /path/to/your/checkpoint-xxxx/unwrapped_model \
    --config configs/showo_demo_512x512.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --framework geneval
```

#### DPGBench

```bash
export PYTHONPATH=.
python scripts/inference_benchmark.py \
    /path/to/your/checkpoint-xxxx/unwrapped_model \
    --config configs/showo_demo_512x512.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --framework dpg
```

#### WISE

```bash
export PYTHONPATH=.
python scripts/inference_benchmark.py \
    /path/to/your/checkpoint-xxxx/unwrapped_model \
    --config configs/showo_demo_512x512.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --framework wise
```

**Example with multiple checkpoints:**

You can evaluate multiple checkpoints in sequence:

```bash
python scripts/inference_benchmark.py \
    /path/to/your/checkpoint-1000/unwrapped_model \
    /path/to/your/checkpoint-2000/unwrapped_model \
    /path/to/your/checkpoint-3000/unwrapped_model \
    --framework geneval \
    --config configs/showo_demo_512x512.yaml \
    --gpus 0,1,2,3,4,5,6,7
```

