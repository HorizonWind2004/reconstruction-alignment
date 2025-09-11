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
import argparse
from xtuner.model.utils import guess_load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path.', default='configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument('--output', help='output file path.', default='qwen2_5_1_5b_mar_h.pth')

    args = parser.parse_args()
    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        if os.path.isdir(args.checkpoint):
            checkpoint = guess_load_checkpoint(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False)
    
    torch.save(checkpoint, args.output)