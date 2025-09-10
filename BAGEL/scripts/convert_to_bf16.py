#!/usr/bin/env python3
# Convert safetensors model to bf16 format
import os
import torch
import argparse
from safetensors.torch import load_file, save_file

def convert_to_bf16(input_file, output_file=None):
    """
    将 safetensors 文件转换为 bfloat16 格式
    
    参数:
        input_file (str): 输入的 safetensors 文件路径
        output_file (str): 输出的 safetensors 文件路径，如果为 None 则自动生成
    """
    print(f"加载 safetensors 文件: {input_file}")
    state_dict = load_file(input_file)
    
    if output_file is None:
        # 自动生成输出文件名
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_bf16{ext}"
    
    print(f"转换为 bf16 格式...")
    # 转换为 bf16
    converted_dict = {}
    for key, tensor in state_dict.items():
        # 仅转换浮点类型的张量
        if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            converted_dict[key] = tensor.to(torch.bfloat16)
        else:
            converted_dict[key] = tensor
    
    print(f"保存 bf16 格式到: {output_file}")
    save_file(converted_dict, output_file)
    print(f"转换完成！")
    
    # 显示内存占用对比
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    converted_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"原始文件大小: {original_size:.2f} MB")
    print(f"转换后文件大小: {converted_size:.2f} MB")
    print(f"大小变化: {(converted_size - original_size) / original_size * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="将 safetensors 模型转换为 bfloat16 格式")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入的 safetensors 文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出的 safetensors 文件路径")
    
    args = parser.parse_args()
    convert_to_bf16(args.input, args.output)

if __name__ == "__main__":
    main()
