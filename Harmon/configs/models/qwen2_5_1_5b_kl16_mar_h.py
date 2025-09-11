# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers import AutoModelForCausalLM
from src.models.mar.mar import mar_huge
from src.models.mar.vae import AutoencoderKL
from src.models.harmon import Harmon
from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoTokenizer

llm_name_or_path = 'Qwen/Qwen2.5-1.5B-Instruct'
prompt_template = dict(
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'])

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=Harmon,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    vae=dict(type=AutoencoderKL,
             embed_dim=16,
             ch_mult=(1, 1, 2, 2, 4),
             ckpt_path='checkpoints/kl16.ckpt'),
    vae_scale=0.2325,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ),
    mar=dict(type=mar_huge,
             img_size=256,
             vae_stride=16,
             patch_size=1,
             vae_embed_dim=16,
             mask_ratio_min=0.7,
             label_drop_prob=0.1,
             class_num=1000,
             attn_dropout=0.1,
             proj_dropout=0.1,
             buffer_size=64,
             diffloss_d=12,
             diffloss_w=1536,
             num_sampling_steps="100",
             diffusion_batch_mul=4,
             grad_checkpointing=True,
             ),
)
