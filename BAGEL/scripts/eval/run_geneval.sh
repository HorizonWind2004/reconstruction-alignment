# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8
export output_path='/path/to/your/geneval/results'
export model_path='/path/to/your/model'

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12349 \
    ./eval/gen/gen_images_mp.py \
    --output_dir $output_path \
    --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \
    --batch_size 1 \
    --num_images 12 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \

