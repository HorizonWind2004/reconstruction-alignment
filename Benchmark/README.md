# Benchmark Evaluation Guide

> **Note**: This repository is based on the following open-source benchmarks:
> - GenEval: https://github.com/djghosh13/geneval
> - DPGBench: https://github.com/TencentQQGYLab/ELLA
> - WISE: https://github.com/PKU-YuanGroup/WISE
> - ImgEdit: https://github.com/PKU-YuanGroup/ImgEdit
> - GEdit: https://github.com/stepfun-ai/Step1X-Edit/tree/main

This guide provides comprehensive instructions for evaluating models on multiple benchmarks.

## Table of Contents

- [GenEval](#geneval)
- [DPGBench](#dpgbench)
- [WISE](#wise)
- [ImgEdit and GEdit](#imgedit-gedit)

## GenEval

### Environment Setup

Create a separate environment for GenEval (reference: [GenEval Issues #12](https://github.com/djghosh13/geneval/issues/12)):

```bash
conda create -n geneval python=3.10
conda activate geneval
cd geneval

# Download detection models
bash ./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"

# Install required packages
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch==2.26.1
pip install clip-benchmark
pip install -U openmim
pip install einops
pip install lightning
pip install "diffusers[torch]" transformers
pip install tomli platformdirs
pip install --upgrade setuptools 
mim install mmengine mmcv-full==1.7.2

# Install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 2.x
pip install -v -e .
cd ..

# Download evaluation models
mkdir model
bash ./evaluation/download_models.sh ./model
```

### Evaluation

```bash
cd geneval
bash calculate.sh -g <gpu_id> <image_folder>
```

**Example:**
```bash
# Use GPU 0 to evaluate images in ./geneval_output/
bash calculate.sh -g 0 ./geneval_output/
```

## DPGBench

### Environment Setup

```bash
conda create -n dpg python==3.10
conda activate dpg

# Install FairSeq
git clone https://github.com/facebookresearch/fairseq
pip install pip==24.0 
pip install omegaconf==2.0.6
pip install ./fairseq  

# Install additional requirements
pip install -r requirements-for-dpg_bench.txt
pip install addict datasets==3.6.0 simplejson sortedcontainers
```

### Evaluation

Use the following command to run the evaluation:

```bash
bash dpgbench/dist_eval.sh <image_folder> <resolution>
```

**Example:**
```bash
# Evaluate images in ./dpg_output/ at 1024x1024 resolution
bash dpgbench/dist_eval.sh ./dpg_output/ 1024
```

You can modify the script to set the number of GPUs and other parameters in the script:

```
IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
GPU_IDS=${GPU_IDS:-"0,1,2"} # here
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
PROCESSES=$NUM_GPUS
PORT=${PORT:-29504}

echo "Use GPU: $GPU_IDS ( $NUM_GPUS GPUs )"
echo "Start $PROCESSES processes"

accelerate launch --num_machines 1 --num_processes $PROCESSES --mixed_precision "fp16" --main_process_port $PORT \
  ./dpg_bench/compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug
  #  --multi_gpu
```

## WISE

### Environment Setup

No additional environment setup is required. Just make sure you have installed the `openai` package:

```bash
pip install openai
```

And set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your_api_key_here'
```

### Evaluation

Use the provided script to run WISE evaluation:

```bash
bash wise/get_wise_score.sh <INPUT_DIR> [OUTPUT_DIR]
```

**Example:**
```bash
# Evaluate images in ./wise_output/ and save results to the same directory
bash wise/get_wise_score.sh ./wise_output/

# Evaluate images in ./wise_output/ and save results to ./wise_results/
bash wise/get_wise_score.sh ./wise_output/ ./wise_results/
```

## ImgEdit, GEdit

See `../BAGEL/README.md` for more details.
