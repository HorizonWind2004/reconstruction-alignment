import torch
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from einops import rearrange
import numpy as np
import random
from xtuner.model.utils import guess_load_checkpoint
import os
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Reconstruction Demo Script")
    parser.add_argument('--config', help='config file path.', default='configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument("--checkpoint", type=str, default='/home/jixie/Harmon/checkpoints/harmon_1.5b.pth')
    parser.add_argument('--input_image', help='path to input image for reconstruction', required=True)
    parser.add_argument('--output', type=str, default='recon_output.jpg', help='output filename')
    parser.add_argument('--save_original', action='store_true', help='save original image alongside reconstruction')
    parser.add_argument("--prompt", type=str, default='Describe this image in details.', help='reconstruction prompt')
    parser.add_argument("--temperature", type=float, default=1.0, help='sampling temperature')
    parser.add_argument('--num_iter', type=int, default=64, help='number of iterations for reconstruction')
    parser.add_argument('--image_size', type=int, default=512, help='output image size')
    parser.add_argument('--cfg', type=float, default=3.0, help='guidance scale for reconstruction')
    parser.add_argument('--grid_size', type=int, default=2, help='grid size for multiple reconstructions (2 means 2x2=4 images)')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {args.seed}")

    print(f"Loading model from config {args.config}", flush=True)
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)

    if os.path.isdir(args.checkpoint):
        checkpoint = guess_load_checkpoint(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint)
    
    info = model.load_state_dict(checkpoint, strict=False)
    print(f"Checkpoint loaded: {args.checkpoint}", flush=True)

    print(f"Loading input image from {args.input_image}", flush=True)
    try:
        input_pil = Image.open(args.input_image).convert('RGB')
        input_pil = input_pil.resize(size=(args.image_size, args.image_size))
        input_tensor = torch.from_numpy(np.array(input_pil)).to(dtype=model.dtype, device='cuda')
        input_tensor = rearrange(input_tensor, 'h w c -> c h w')[None]
        input_tensor = 2 * (input_tensor / 255) - 1
        
        if args.save_original:
            original_path = f"original_{os.path.basename(args.output)}"
            input_pil.save(original_path)
            print(f"Original image saved to {original_path}")
        
        print(f"Input image processed, shape: {input_tensor.shape}", flush=True)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)

    print(f"Starting image reconstruction with prompt: '{args.prompt}'", flush=True)
    print(f"Using {args.num_iter} iterations at temperature {args.temperature}", flush=True)
    print(f"Generating {args.grid_size}x{args.grid_size}={args.grid_size**2} images", flush=True)
    
    bsz = args.grid_size ** 2
    input_tensor = input_tensor.expand(bsz, -1, -1, -1)
    
    with torch.no_grad():
        recon_images = model.sample_recon(
            image=input_tensor,
            prompt=args.prompt,
            temperature=args.temperature,
            num_iter=args.num_iter,
            progress=True,
            cfg=args.cfg,
        )
    
    recon_images = torch.clamp(127.5 * recon_images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    recon_images = rearrange(recon_images, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    
    Image.fromarray(recon_images).save(args.output)
    print(f"Reconstruction grid saved to {args.output}", flush=True)
    
    print("Done!")
