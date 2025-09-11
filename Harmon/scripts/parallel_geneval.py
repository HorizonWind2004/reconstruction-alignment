#!/usr/bin/env python
# coding=utf-8

'''
python scripts/parallel_geneval.py --gpus 3,4,5 --checkpoint work_dirs/exp18/iter_5000.pth --batch_size 4 --outdir dpg_exp18_5000 --mode dpgbench
python scripts/parallel_geneval.py --gpus 8,9 --checkpoint work_dirs/exp16/iter_3000.pth --batch_size 4 --outdir dpg_exp16_3000 --mode dpgbench --config /home/jixie/Harmon/configs/models/qwen2_5_0_5b_kl16_mar_b.py
'''

import os
import sys
import argparse
import multiprocessing as mp
import subprocess
import time
from itertools import cycle

def run_geneval(gpu_id, args, start_idx, end_idx):
    cmd = [
        "CUDA_VISIBLE_DEVICES=" + str(gpu_id),
        "python", "scripts/geneval.py",
        f"--config={args.config}",
        f"--batch_size={args.batch_size}",
        f"--guidance_scale={args.guidance_scale}",
        f"--generation_timesteps={args.generation_timesteps}",
        f"--temperature={args.temperature}",
        f"--cfg_schedule={args.cfg_schedule}",
        f"--cfg_prompt='{args.cfg_prompt}'",
        f"--validation_prompts_file={args.validation_prompts_file}",
        f"--seed={args.seed}",
        f"--image_size={args.image_size}",
        f"--l={start_idx}",
        f"--r={end_idx}",
        f"--exp={args.exp}",
        f"--step={args.step}",
    ]
    
    # 添加可选参数
    if args.outdir is not None:
        cmd.append(f"--outdir={args.outdir}")
    if args.checkpoint is not None:
        cmd.append(f"--checkpoint={args.checkpoint}")
    if args.use_template:
        cmd.append("--use_template")
    if args.remove_prefix:
        cmd.append("--remove_prefix")

    cmd_str = " ".join(cmd)
    print(f"GPU {gpu_id} 运行命令: {cmd_str}")
    
    process = subprocess.Popen(
        cmd_str, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    prefix = f"[GPU {gpu_id}] "
    for line in process.stdout:
        print(prefix + line.rstrip(), flush=True)
    
    return_code = process.wait()
    if return_code != 0:
        print(f"{prefix} exit: {return_code}", flush=True)
    else:
        print(f"{prefix} done.", flush=True)

def run_dpgbench(gpu_id, args, start_idx, end_idx):
    cmd = [
        "CUDA_VISIBLE_DEVICES=" + str(gpu_id),
        "python", "scripts/dpgbench.py",
        f"--config={args.config}",
        f"--batch_size=4",
        f"--guidance_scale={args.guidance_scale}",
        f"--generation_timesteps={args.generation_timesteps}",
        f"--temperature={args.temperature}",
        f"--cfg_schedule={args.cfg_schedule}",
        f"--cfg_prompt='{args.cfg_prompt}'",
        f"--seed={args.seed}",
        f"--image_size={args.image_size}",
        f"--l={start_idx}",
        f"--r={end_idx}",
        f"--prompts_file={args.prompts_file}",
    ]
    
    if args.outdir is not None:
        cmd.append(f"--outdir={args.outdir}")
    if args.checkpoint is not None:
        cmd.append(f"--checkpoint={args.checkpoint}")
    if args.use_template:
        cmd.append("--use_template")
    
    cmd_str = " ".join(cmd)
    print(f"GPU {gpu_id} running command: {cmd_str}")
    process = subprocess.Popen(
        cmd_str, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    prefix = f"[GPU {gpu_id}] "
    for line in process.stdout:
        print(prefix + line.rstrip(), flush=True)
    
    return_code = process.wait()
    if return_code != 0:
        print(f"{prefix} exit: {return_code}", flush=True)
    else:
        print(f"{prefix} done.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path.', default='configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument("--checkpoint", "--ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--guidance_scale", "--cfg", type=float, default=3.0)
    parser.add_argument("--generation_timesteps", "--num_iter", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--cfg_prompt', type=str, default='Generate an image.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--validation_prompts_file', type=str, default='../Benchmark/geneval/prompts/evaluation_metadata.jsonl')
    parser.add_argument('--prompts_file', type=str, default='../Benchmark/dpgbench/prompts.json')
    parser.add_argument('--use_template', action='store_true')
    parser.add_argument('--exp', type=str, default='exp4')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--total_prompts', type=int, default=None)
    parser.add_argument('--mode', type=str, default='geneval', choices=['geneval', 'dpgbench'])
    parser.add_argument('--remove_prefix', action='store_true')
    
    args = parser.parse_args()
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    
    print(f"Use {num_gpus} GPUs: {gpu_ids}")
    
    if args.outdir is None:
        if args.mode == 'geneval':
            args.outdir = args.exp + f"_{args.step}"
            if args.long:
                args.outdir += "_long"
        else:
            args.outdir = f"dpg_harmon_results"
            if args.long:
                args.outdir += "_long"
    
    if args.checkpoint is None and args.mode == 'geneval':
        args.checkpoint = f"work_dirs/{args.exp}/{args.exp}_{args.step}"
        if not os.path.exists(args.checkpoint):
            args.checkpoint = f"work_dirs/{args.exp}/iter_{args.step}.pth"

    os.makedirs(args.outdir, exist_ok=True)
    
    import json
    try:
        if args.mode == 'geneval':
            with open(args.validation_prompts_file) as fp:
                prompts = [json.loads(line) for line in fp]
            total_prompts = args.total_prompts if args.total_prompts is not None else len(prompts)
            total_prompts = min(total_prompts, len(prompts))
        else:
            with open(args.prompts_file) as fp:
                prompts = json.load(fp)
            total_prompts = args.total_prompts if args.total_prompts is not None else len(prompts)
            total_prompts = min(total_prompts, len(prompts))
            print(f"Load {total_prompts} DPGBench prompts")
    except Exception as e:
        print(f"Load prompts file error: {e}")
        prompts = {"default.txt": "a dog on the left and a cat on the right."}
        total_prompts = 1
    
    prompts_per_gpu = (total_prompts + num_gpus - 1) // num_gpus
    ranges = []
    
    for i in range(num_gpus):
        start_idx = i * prompts_per_gpu
        end_idx = min((i + 1) * prompts_per_gpu, total_prompts)
        if start_idx < end_idx:
            ranges.append((start_idx, end_idx))

    print(f"Totally {total_prompts} prompts will be divided into {len(ranges)} tasks")

    processes = []
    for (start_idx, end_idx), gpu_id in zip(ranges, gpu_ids):
        print(f"GPU {gpu_id} processing range: {start_idx} - {end_idx}")
        target_func = run_geneval if args.mode == 'geneval' else run_dpgbench
        
        p = mp.Process(
            target=target_func,
            args=(gpu_id, args, start_idx, end_idx)
        )
        processes.append(p)
        p.start()
        time.sleep(1)
    
    for p in processes:
        p.join()
    
    print("All processes done!")
