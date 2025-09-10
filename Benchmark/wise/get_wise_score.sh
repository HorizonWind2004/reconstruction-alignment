#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# Check if INPUT_DIR is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <INPUT_DIR> [OUTPUT_DIR]"
    echo "  INPUT_DIR: Directory containing the generated images"
    echo "  OUTPUT_DIR: Directory to save the evaluation results (optional, defaults to INPUT_DIR)"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=${2:-$INPUT_DIR}

# Check if INPUT_DIR exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY=your_api_key"
    exit 1
fi

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Using OpenAI API key: ${OPENAI_API_KEY:0:8}..."

# calculate score
python3 wise/gpt_eval_mp.py \
        --json_path wise/data/cultural_common_sense.json \
        --image_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR

python3 wise/gpt_eval_mp.py \
        --json_path wise/data/spatio-temporal_reasoning.json \
        --image_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR

python3 wise/gpt_eval_mp.py \
        --json_path wise/data/natural_science.json \
        --image_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR

python3 wise/cal_score.py \
        --output_dir $OUTPUT_DIR