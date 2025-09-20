#!/bin/bash
INPUT_CHECKPOINT_PATH="/workspace/reconstruction-alignment/BAGEL/checkpoints/0000250"

# 1. Construct the output path for the converted model (append _hf to the original path)
#    Example: /workspace/reconstruction-alignment/BAGEL/results/hf_weights/checkpoint_reg_2e5_0.1_hf
OUTPUT_HF_PATH="/workspace/reconstruction-alignment/BAGEL/results/hf_weights/reca_0000250"
TEMPLATE_MODEL="/workspace/SRUM/BAGEL-7B-MoT"
# Print the command that will be executed, for easy debugging
echo "############################################################"
echo "### Processing: ${INPUT_CHECKPOINT_PATH}"
echo "### Output to:  ${OUTPUT_HF_PATH}"
echo "############################################################"

# 2. Execute the Python conversion script
python scripts/trans2hf.py \
  --training_checkpoint_path "${INPUT_CHECKPOINT_PATH}" \
  --template_model_path "${TEMPLATE_MODEL}" \
  --output_path "${OUTPUT_HF_PATH}"

echo "Checkpoint for weight has been processed."