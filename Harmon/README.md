# Harmon Training Guide

> Note: This repository is heavily based on [wusize/Harmon](https://github.com/wusize/Harmon). This repository provides training code and evaluation scripts. For inference scripts and additional resources, please refer to the original repository.

This guide provides comprehensive instructions for training Harmon models.

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
conda create -n harmon python=3.10
conda activate harmon
pip install -r requirements.txt
```

## Data Preparation

### Dataset Structure

Harmon supports multiple training modes. Organize your data according to your training objective:

#### For Image-to-Text Training (LLaVA Format)

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

The JSON file should contain LLaVA format data with image paths and conversations. You can download the LLaVA dataset from [here](https://huggingface.co/datasets/sanaka87/LLaVA-Instruct-150K).


#### For Image Reconstruction Training (WebDataset Format)

```
data/
├── webdataset/
│   ├── data_000000.tar
│   ├── data_000001.tar
│   └── ...
```

Each tar file should contain image-caption pairs in the WebDataset format.

### Dataset Configuration Examples

#### LLaVA Understanding Dataset

Configuration file: `configs/datasets/qwen2_5_1_5b/image2text.py`

```python
data_root = 'data/LLaVA-Instruct-150K/' # You can modify this path
data_path = 'data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = 'data/LLaVA-Instruct-150K/tuning_data'
max_length = int(2048 - (336 / 14) ** 2)
```

#### Reconstruction Dataset (O3/WebDataset)

Configuration file: `configs/datasets/qwen2_5_1_5b/reconstruction_o3.py`

You can download the blip3o-60k dataset from [here](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k).

#### Reconstruction Dataset (Midjourney)

Configuration file: `configs/datasets/qwen2_5_1_5b/reconstruction_mid.py`

MidjourneyV6 dataset will be downloaded automatically when you run the training script.

#### Multi-Task Training Configuration

Use these combined configs for multi-task training:

- **Midjourney + Image2Text**: `configs/datasets/qwen2_5_1_5b/image2text_recon_mid.py`

- **Blip3o-60k + Image2Text**: `configs/datasets/qwen2_5_1_5b/image2text_recon_mid.py`

Example multi-task setup:

```python
# From image2text_recon_mid.py
group_keys = ['recon', 'image2text', 'text2image']
repeat = [2, 2, 0]  # Training ratios: 2x recon, 2x understanding, 0x generation
batch_size = 48     # for 1.5B model
# batch_size = 96   # for 0.5B model
num_workers = 32    # for multi-task training
```

## Model Preparation

### Download Base Model

Download the pre-trained Harmon model:

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download via Hugging Face CLI (recommended)
pip install -U "huggingface_hub[cli]"
huggingface-cli download wusize/harmon --local-dir checkpoints --repo-type model
```

Your checkpoint structure should look like:

```
checkpoints/
├── kl16.ckpt
├── harmon_0.5b.pth
├── harmon_1.5b.pth
└── ...
```

## Training Configuration

### Basic Configuration

Choose a configuration from `configs/examples/` or create your own. The key parameters to configure:

```python
# Model configuration
model.update(
    type=HarmonDev,
    pretrained_pth='checkpoints/harmon_1.5b.pth',  # Path to base model
    freeze_llm=False,  # Whether to freeze the language model
)

# Training parameters
max_iters = 50000  # Total training iterations
lr = 1e-5  # Learning rate
weight_decay = 0.02
warmup_ratio = 0.01
save_steps = 5000  # Save checkpoint every N steps
```

### Hardware Configuration

Configure GPU settings in `train.sh`:
```bash
NNODES=1  # Number of nodes
GPUS_PER_NODE=1  # GPUs per node
export CUDA_VISIBLE_DEVICES=0  # Specify GPU IDs
```

## Training

Configure the training environment and start the training process:

```bash
bash train.sh <config_name>
```

**Example:**

```bash
# There is a config file named realign.py in configs/examples/
bash train.sh reca
```

You can modify `train.sh` to set the number of GPUs and other parameters as needed.

```bash
MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
GPUS_PER_NODE=1                  # Number of GPUs per node
NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0.   # Specify GPU IDs
export PYTHONPATH=.
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

CONFIG_NAME=${1:-"exp0"}
CONFIG_FILE="configs/examples/${CONFIG_NAME}.py"

export LAUNCHER="torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "

export CMD="scripts/train.py \
$CONFIG_FILE \
--launcher pytorch \
--deepspeed deepspeed_zero2"

echo $LAUNCHER
echo $CMD

bash -c "$LAUNCHER $CMD"

sleep 60s
```

## Evaluation

### Checkpoints

Checkpoints are saved in:

```
work_dirs/your_experiment/
├── iter_5000.pth
├── iter_10000.pth
└── ...
```

### Evaluation

Use the parallel evaluation script to efficiently evaluate your trained models on multiple benchmarks:

#### GenEval

```bash
# Basic GenEval evaluation using multiple GPUs
python scripts/parallel_geneval.py \
    --gpus 0,1 \
    --checkpoint work_dirs/exp18/iter_5000.pth \
    --batch_size 4 \
    --outdir geneval_exp18_5000 \
    --mode geneval
```

**Example with custom config (for 0.5B model):**
```bash
python scripts/parallel_geneval.py \
    --gpus 0,1 \
    --checkpoint work_dirs/exp16/iter_3000.pth \
    --batch_size 4 \
    --outdir geneval_exp16_3000 \
    --mode geneval \
    --config configs/models/qwen2_5_0_5b_kl16_mar_b.py
```

#### DPGBench

```bash
# DPGBench evaluation
python scripts/parallel_geneval.py \
    --gpus 0,1 \
    --checkpoint work_dirs/exp18/iter_5000.pth \
    --batch_size 4 \
    --outdir dpg_exp18_5000 \
    --mode dpgbench
```
