#!/bin/bash

GPU_ID=0

while getopts ":g:" opt; do
  case ${opt} in
    g )
      GPU_ID=$OPTARG
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

if [ $# -eq 0 ]; then
    MODEL_PATHS=("../Show-o/out_showo_256_bs100_4000")
else
    MODEL_PATHS=("$@")
fi

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATH%/}

    echo "Processing model: $MODEL_PATH"
    MODEL_NAME=$(basename "$MODEL_PATH")
    CLEAN_MODEL_NAME="${MODEL_NAME#out_}"

    echo "Output file: results_${CLEAN_MODEL_NAME}.jsonl"
    echo "Using GPU: $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluation/evaluate_images.py "$MODEL_PATH" --outfile "results_${CLEAN_MODEL_NAME}.jsonl"

    echo "Finished evaluating model $MODEL_PATH"
    echo "------------------------"
done
