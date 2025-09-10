# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
import random
from safetensors.torch import load_file

import torch
import torch.distributed as dist
import numpy as np
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from tqdm import tqdm
from modeling.bagel.qwen2_navit import NaiveCache


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None, use_template=False):
    if use_template:
        prompt = f"Describe the image: {prompt}"
    else:
        prompt = f"Generate an image: {prompt}"
        
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(resolution, resolution)] * num_images, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope, 
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution//8, resolution//8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPG benchmark images using Bagel model.")
    parser.add_argument("--outdir", type=str, default="dpg_bagel_ori", help="Directory to save the generated images.")
    parser.add_argument("--prompts_file", type=str, default="./eval/gen/dpg_bench/prompts.json", help="JSON file containing prompts for DPG benchmark.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--cfg_scale", "--guidance_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution.")
    parser.add_argument("--num_timesteps", "--generation_timesteps", type=int, default=50, help="Number of diffusion steps.")
    parser.add_argument("--model-path", type=str, default="/scratch/2025_05/jixie/BAGEL-7B-MoT", help="Path to model weights.")
    parser.add_argument("--max_latent_size", type=int, default=64, help="Maximum latent size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--l", type=int, default=0, help="Start index for processing.")
    parser.add_argument("--r", type=int, default=None, help="End index for processing.")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate per prompt.")
    parser.add_argument("--use_template", action="store_true", help="Use prompt template.")
    parser.add_argument("--cfg_prompt", type=str, default="Generate an image.", help="Prompt used for classifier-free guidance.")
    parser.add_argument('--origin-model-path', type=str, default='ckpt/BAGEL-7B-MoT/')
    parser.add_argument('--use-ema', action='store_true', help="Use EMA weights for the model.")

    args = parser.parse_args()
    
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images will be saved in {output_dir}")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.origin_model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.origin_model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.origin_model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.origin_model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    if args.use_ema:
        model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    else:
        model_state_dict_path = os.path.join(args.model_path, "model.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    model = model.to(dtype=torch.bfloat16)
    if rank == 0:
        print(msg)
    del model_state_dict

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 3.0
    num_timesteps = args.num_timesteps
    cfg_renorm_min = 0.0

    try:
        with open(args.prompts_file, 'r') as f:
            dataset = json.load(f)
            if rank == 0:
                print(f"Loaded {len(dataset)} prompts from {args.prompts_file}")
    except Exception as e:
        if rank == 0:
            print(f"Error loading prompts file: {e}")
        dataset = {"default.txt": "a dog on the left and a cat on the right."}
    
    dataset_items = list(dataset.items())
    total_metadatas = len(dataset_items)
    
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    
    processed_items = dataset_items[start:end]
    for key, prompt in tqdm(processed_items, desc=f"GPU {rank} processing"):
        set_seed(args.seed)
        
        print(f"GPU {rank} processing prompt: '{prompt}'")
        
        image_list = []
        # skip
        flag = True
        for idx in range(args.num_images):
            if not os.path.exists(os.path.join(output_dir, f"{key.split('.')[-2]}_{idx}.jpg")):
                flag = False
                break
        if flag:
            print(f"GPU {rank} skipping generation for prompt: {prompt}")
            continue
        for i in range(args.num_images // args.batch_size):
            batch_images = generate_image(
                prompt=prompt,
                cfg_scale=cfg_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                num_images=min(args.num_images, args.batch_size),
                resolution=args.resolution,
                device=device,
                use_template=args.use_template,
            )
            image_list.extend(batch_images)
        
        for idx, sample in enumerate(image_list):
            out_filename = f"{key.split('.')[-2]}_{idx}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            sample.save(out_path)
            print(f"GPU {rank} saved image to: {out_path}")

    print(f"GPU {rank} has completed all tasks")
    # dist.barrier()
