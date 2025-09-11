# BAGEL Installation Guide

> **Note**: This repository is heavily based on [ByteDance-Seed/BAGEL](https://github.com/bytedance-seed/BAGEL). This repository provides training code and evaluation scripts. For inference scripts and additional resources, please refer to the original repository.

This guide provides comprehensive instructions for training ReAlign on BAGEL.

## Table of Contents

- [Installation](#installation)
- [Model and Data Preparation](#model-and-data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
  - [GenEval](#geneval)
  - [DPGBench](#dpgbench)
  - [WISE](#wise)
  - [ImgEdit](#imgedit)

## Installation

### Environment Setup

Create and activate a new conda environment:

```bash
conda create -n bagel python=3.10
conda activate bagel
pip install -r requirements.txt
```

### 🔥 Start Inference!

You can use the `inference.ipynb` and try out!

## Model and Data Preparation

### Download Pre-trained Model

Download the BAGEL-7B-MoT checkpoint from Hugging Face:

```bash
mkdir ckpt
cd ckpt
git clone https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
cd ..
```

> **Note**: All scripts in this repository expect the model to be located at `./ckpt/BAGEL-7B-MoT`. If you change this path, you'll need to update the scripts accordingly.

### Download Training Data

#### Text-to-Image-2M Dataset

Download the training data using wget:

```bash
mkdir -p data/train
cd data/train
wget https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_1024_10K/data_000000.tar
cd ../..
```

> **Note**: Each tar file contains 10K high-quality text-to-image pairs at 1024x1024 resolution. You can replace `data_000000.tar` with other tar files from the same dataset directory if you need more training data.

## Training

### Dataset Configuration

Before training, configure your dataset paths in `data/configs/dataset_info.py`:

```python
{
    'reconstruction': {
        'webdataset': {
            'data_dir': './data/train/data_000000.tar',  # Path containing all tar files
            'num_files': 1,  # Number of data units to be sharded across all ranks and workers
            'cache_dir': '.cache',  # Cache directory for extracted images
        },
    },
}
```

### Start Training

Configure the training environment and start the training process:

```bash
export master_addr=localhost
export master_port=12345
export output_path='./'
export ckpt_path='./checkpoints'
export PYTHONPATH=.

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --model_path 'ckpt/BAGEL-7B-MoT' \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --freeze_vae True \
  --freeze_vit True \
  --freeze_und True \
  --finetune_from_ema True \
  --resume_from 'ckpt/BAGEL-7B-MoT' \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --log_every 1 \
  --wandb_runid 1 \
  --use_flex \
  --lr 0.00004 \
```

## Evaluation

The evaluation script is located at `scripts/eval/run_geneval.sh`. You can modify the script to set the number of GPUs, model path and the output directory, generated images' resolution, and number of images per prompt.

```bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7  # Set the GPUs to use
GPUS=6  # Number of GPUs to use
export PYTHONPATH=.
export output_path='./'
export model_path='ckpt/BAGEL-7B-MoT'  # Model path

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp.py \
    --output_dir $output_path/bagel_ori_1024 \
    --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \
    --batch_size 1 \
    --num_images 12 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \
    --use-ema # Use EMA weights. If you want to use non-EMA weights, remove this line.
```

**Note:** This will generate 12 1024x1024 images per prompt for evaluation. Similarly, you can run the evaluation scripts for DPGBench, WISE, ImgEdit and GEdit datasets by modifying the respective scripts in the `scripts/eval` directory. More detailed Evaluation instructions can be found in the original [BAGEL repository](https://github.com/ByteDance-Seed/BAGEL).
