# OpenUni Training Guide

> Note: This repository is heavily based on [wusize/OpenUni](https://github.com/wusize/OpenUni). For inference scripts and additional resources, please refer to the original repository.

This guide provides comprehensive instructions for training OpenUni models with Reconstruction Alignment (RecA).

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
conda create -n openuni python=3.10
conda activate openuni
pip install -r requirements.txt
```

Required packages include:
- mmengine
- xtuner
- transformers
- torch
- flash_attn

## Model Preparation

### Download Base Model

Download the pre-trained OpenUni model:

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download via Hugging Face CLI (recommended)
pip install -U "huggingface_hub[cli]"
huggingface-cli download wusize/openuni --local-dir checkpoints --repo-type model
```

Your checkpoint structure should look like:

```
checkpoints/
├── openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth
├── openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth
├── openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth
├── openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth
├── openuni_l_internvl3_2b_sana_1_6b_1024_hf_blip3o60k.pth
└── openuni_l_internvl3_2b_sana_1_6b_1024_hf_text2image23m.pth
```

### Configuration Files

We provide three configuration Files:

- **OpenUni_1_0.6b** : `configs/finetune/reca_1_0.6.py`
- **OpenUni_2_1.6b** : `configs/finetune/reca_2_1.6.py`
- **OpenUni_2_1.6b_blip3o** : `configs/finetune/reca_2_1.6_3o.py`

Key configuration parameters:

```python
# Model configuration
model.update(
    pretrained_pth='path/to/pretrained/checkpoint.pth',  # Path to base model
)

# Training parameters
max_iters = 10000  # Total training iterations
lr = 1e-5  # Learning rate
warmup_ratio = 0.01
save_steps = 5000  # Save checkpoint every N steps
```

## Training

### Pretraining

```bash
export PYTHONPATH=.
GPUS_PER_NODE=1 NNODES=1 bash scripts/train_ddp.sh \
     configs/finetune/reca_1_0.6.py \
     --deepspeed deepspeed_zero2
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

### Inference

#### Reconstruction Visualization

```bash
export PYTHONPATH=.
python scripts/image_edit.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \
    --checkpoint /path/to/your/ckpt \
    --input_image /path/to/your/input \
    --prompt "Describe the image in details" \
    --output /path/to/your/output \
    --height 512 --width 512 \
    --seed 42
```

#### GenEval

```bash
export PYTHONPATH=.
accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \
  --checkpoint work_dirs/your/ckpt/ \
  --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth \
  --batch_size 2 \
  --output /path/to/your/output \
  --height 512 --width 512 \
  --seed 42 
```

#### DPGBench

```bash
accelerate launch scripts/evaluation/dpg_bench.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \
  --checkpoint work_dirs/your/ckpt/ \
  --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth \
  --batch_size 2 \
  --output /path/to/your/output \
  --height 512 --width 512 \
  --seed 42 
```

#### WISE Bench

```bash
accelerate launch scripts/evaluation/dpg_bench.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py \
  --checkpoint work_dirs/your/ckpt/ \
  --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth \
  --batch_size 2 \
  --output /path/to/your/output \
  --height 512 --width 512 \
  --seed 42 \
  --data ../Benchmark/wise/data/cultural_common_sense.json

# Do not forget
# ../Benchmark/wise/data/natural_science.json 
# ../Benchmark/wise/data/spatio-temporal_reasoning.json
```
