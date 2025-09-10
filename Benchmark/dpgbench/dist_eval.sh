IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}  
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