# coding=utf-8
# Copyright 2024 Harmon Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import numpy as np
import torch
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from einops import rearrange
from tqdm import tqdm, trange
import json

from xtuner.model.utils import guess_load_checkpoint

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path.', default='configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument("--checkpoint", type=str, default='checkpoints/harmon_1.5b.pth')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", type=str, default="t2i")
    parser.add_argument("--guidance_scale", "--cfg", type=float, default=3.0)
    parser.add_argument("--generation_timesteps", "--num_iter", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--cfg_prompt', type=str, default='Generate an image.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--outdir', type=str, default='dpg_harmon_ori')
    parser.add_argument('--prompts_file', type=str, default='../dpg_bench/prompts.json')
    parser.add_argument('--l', type=int, default=0, help='Start index for processing')
    parser.add_argument('--r', type=int, default=None, help='End index for processing')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    config = Config.fromfile(args.config)
    
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    
    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        if os.path.isdir(args.checkpoint):
            checkpoint = guess_load_checkpoint(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False)
    info = model.load_state_dict(checkpoint, strict=False)
    
    try:
        with open(args.prompts_file, 'r') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        dataset = {"default.txt": "a dog on the left and a cat on the right."}

    l = args.l
    r = args.r if args.r is not None else len(dataset)
    
    for key, prompt in tqdm(list(dataset.items()), desc="Processing prompts"):
        index = list(dataset.keys()).index(key)
        if index < l or index >= r:
            continue
            
        os.makedirs(args.outdir, exist_ok=True)

        print(f"Prompt ({index+1}/{len(dataset)}, key={key}): '{prompt}'")

            
        full_prompt = f"Generate an image: {prompt}"
        class_info = model.prepare_text_conditions(full_prompt, args.cfg_prompt)
        
        input_ids = class_info['input_ids']
        attention_mask = class_info['attention_mask']

        batch_size = args.batch_size
        
        if args.guidance_scale != 1.0:
            input_ids = torch.cat([
                input_ids[0:1].expand(batch_size, -1),
                input_ids[1:2].expand(batch_size, -1),
            ])
            attention_mask = torch.cat([
                attention_mask[0:1].expand(batch_size, -1),
                attention_mask[1:2].expand(batch_size, -1),
            ])
        else:
            input_ids = input_ids[0:1].expand(batch_size, -1)
            attention_mask = attention_mask[0:1].expand(batch_size, -1)
        
        img_h = img_w = args.image_size // 16
        
        with torch.no_grad():
            samples = model.sample(input_ids=input_ids, 
                                  attention_mask=attention_mask,
                                  num_iter=args.generation_timesteps, 
                                  cfg=args.guidance_scale, 
                                  cfg_schedule=args.cfg_schedule,
                                  temperature=args.temperature, 
                                  progress=True, 
                                  image_shape=(img_h, img_w))
        
        for idx, sample in enumerate(samples):
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
            sample = sample.transpose(1, 2, 0)
            out_path = os.path.join(args.outdir, f"{key.split('.')[-2]}_{idx}.jpg")
            Image.fromarray(sample).save(out_path)
    
    print("Done!")
