from torch.utils.data import Dataset
from PIL import Image
import os
import io
import json
import random
import torch
import numpy as np
from einops import rearrange
from xtuner.registry import BUILDER
from src.datasets.utils import crop2square
from glob import glob


class Text2ImageDataset(Dataset):
    def __init__(self,
                 data_path,
                 local_folder,
                 image_size,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=True,
                 cap_source='caption',
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.unconditional = unconditional
        self.local_folder = local_folder
        self.cap_source = cap_source

        self.image_size = image_size

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image

    def _load_data(self, data_path):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        image = Image.open(os.path.join(self.local_folder, image_file))
        assert image.width > 8 and image.height > 8, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image

    def _process_text(self, text):
        if random.uniform(0, 1) < self.unconditional:
            prompt = "Generate an image."
        else:
            prompt = f"Generate an image: {text.strip()}"
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')[0]

        return dict(input_ids=input_ids[:self.max_length])

    def _process_image(self, image):
        data = dict()

        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))

        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)

        return data

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')

            caption = data_sample[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class LargeText2ImageDataset(Text2ImageDataset):
    # self.data_list only contains paths of images and captions

    def __init__(self, cap_folder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_folder = self.local_folder if cap_folder is None else cap_folder

    def _load_data(self, data_path):      # image path and annotation path are saved in a json file
        if data_path.endswith(".json"):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            self.data_list = []
            json_files = glob(f'{data_path}/*.json')
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    self.data_list += json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            with open(f"{self.cap_folder}/{data_sample['annotation']}", 'r') as f:
                caption = json.load(f)[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{data_sample}: {e}", flush=True)
            return self._retry()


class BlipO3Dataset(Text2ImageDataset):
    def __init__(self, 
                 data_path="/scratch/2025_06/jixie/BLIP3o-60k/*.tar",
                 cache_dir='/scratch/2025_06/jixie/',
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        super().__init__(data_path=data_path, *args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            data_files = glob(data_path) 
            self.dataset = load_dataset("webdataset", data_files=data_files, cache_dir=self.cache_dir, split="train", num_proc=64)

            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx,
                })
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['jpg']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")
            
            caption = sample['txt']
            
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()
        
        
class MidJourneyDataset(Text2ImageDataset):
    def __init__(self, 
                 data_path="brivangl/midjourney-v6-llava",
                 cache_dir=None,
                 use_llava=False,
                 *args, **kwargs):
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.use_llava = use_llava
        super().__init__(data_path=data_path, *args, **kwargs)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            self.dataset = load_dataset(data_path, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx, 
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")
            
            if self.use_llava:
                caption = sample['llava']
            else:
                caption = sample['prompt']
            
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()

class ReconstructionDataset(Text2ImageDataset):
    def __init__(self,
                 data_path,
                 image_size,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 cap_source='caption',
                 max_samples=None,
                 use_downscale=False,
                 cache_dir='/scratch/2025_06/jixie/journeydb'):
        
        self.data_path = data_path
        self.unconditional = unconditional
        self.local_folder = None
        self.cap_source = cap_source
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self.use_downscale = use_downscale
        self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._load_data(data_path)
        from src.datasets.text2image.consts import get_recon_prompt_list
        self.recon_prompts = get_recon_prompt_list()
        
        print(f"Loaded ReconstructionDataset with {len(self.data_list)} samples, {len(self.recon_prompts)} prompts, cache_dir: {self.cache_dir}", flush=True)

    def _extract_tar_if_needed(self, tar_path):
        import tarfile
        import hashlib
        tar_hash = hashlib.md5(tar_path.encode()).hexdigest()
        extract_dir = os.path.join(self.cache_dir, tar_hash)
        
        lock_file = os.path.join(extract_dir, '.extraction_complete')
        
        if os.path.exists(lock_file):
            print(f"Using cached extraction for {tar_path} in {extract_dir}", flush=True)
            return extract_dir
        
        print(f"Extracting {tar_path} to {extract_dir}...", flush=True)
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
            
            with open(lock_file, 'w') as f:
                f.write(f"Extracted from {tar_path} at {os.path.getmtime(tar_path)}")
                
            print(f"Extraction complete: {tar_path} -> {extract_dir}", flush=True)
            return extract_dir
        except Exception as e:
            print(f"Error extracting {tar_path}: {e}", flush=True)
            raise
    
    def _load_data(self, data_path):
        import tarfile
        import glob
        
        self.tar_files = glob.glob(os.path.expanduser(data_path.replace('{', '[').replace('}', ']')))
        self.data_list = []
        self.image_cache_paths = {}
        
        for tar_idx, tar_path in enumerate(self.tar_files):
            try:
                extract_dir = self._extract_tar_if_needed(tar_path)
                
                with tarfile.open(tar_path, 'r') as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            file_name = member.name
                            cache_path = os.path.join(extract_dir, file_name)
                            self.data_list.append({'image': file_name, 'tar_idx': tar_idx})
                            self.image_cache_paths[file_name] = cache_path
                        
                        if self.max_samples and len(self.data_list) >= self.max_samples:
                            break
                    
                    if self.max_samples and len(self.data_list) >= self.max_samples:
                        break
                    
            except Exception as e:
                print(f"Error loading tar file {tar_path}: {e}", flush=True)
                
        print(f"Loaded {len(self.data_list)} images from {len(self.tar_files)} tar files: {self.tar_files}", flush=True)

        if len(self.data_list) == 0:
            raise RuntimeError(f"No valid images found in tar archives: {data_path}")

    def _read_image(self, image_file):
        if image_file not in self.image_cache_paths:
            raise ValueError(f"Image file {image_file} not found in cache")
            
        cache_path = self.image_cache_paths[image_file]
        
        try:
            image = Image.open(cache_path)
            assert image.width > 8 and image.height > 8, f"Image too small: {image.size}"
            assert image.width / image.height > 0.1, f"Image aspect ratio too extreme: {image.size}"
            assert image.width / image.height < 10, f"Image aspect ratio too extreme: {image.size}"
            
            return image
        except Exception as e:
            raise RuntimeError(f"Error reading image from cache path {cache_path}: {e}")

    def _process_text(self, text):
        prompt = random.choice(self.recon_prompts)
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            final_prompt = f"\n{prompt}"
            
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        # print(f"Prompt: {final_prompt}", flush=True)
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        return dict(input_ids=input_ids[:self.max_length])

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            
            if self.use_downscale:
                image = image.resize(size=(self.image_size // 2, self.image_size // 2))
                image = image.resize(size=(self.image_size, self.image_size))
            
            data = self._process_image(image)
            data.update(self._process_text(""))
            data.update(type='recon')
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()

class MidjourneyReconstructionDataset(Text2ImageDataset):
    def __init__(self, 
                 image_size,
                 data_path="brivangl/midjourney-v6-llava",
                 cache_dir=None,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=False,
                 cap_source='caption',
                 max_samples=None,
                 use_downscale=False,
                 *args, **kwargs):
        self.data_path = data_path
        self.unconditional = unconditional
        self.local_folder = None
        self.cap_source = cap_source
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.max_samples = max_samples
        self.use_downscale = use_downscale
        self.cache_dir = cache_dir
        from src.datasets.text2image.consts import get_recon_prompt_list
        self.recon_prompts = get_recon_prompt_list()
        self._load_data(data_path)

    def _load_data(self, data_path):
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {data_path} with cache_dir {self.cache_dir}")
            self.dataset = load_dataset(data_path, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(self.dataset)} samples from {data_path}")
            
            self.data_list = []
            for idx in range(len(self.dataset)):
                self.data_list.append({
                    'idx': idx,
                })
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.data_list = []

        print(f"Load {len(self.data_list)} data samples from {data_path} for reconstruction", flush=True)
    
    def _process_text(self, text):
        prompt = random.choice(self.recon_prompts)
        
        if random.uniform(0, 1) < self.unconditional:
            final_prompt = "Generate an image."
        else:
            final_prompt = f"\n{prompt}"
            
        final_prompt = self.prompt_template['INSTRUCTION'].format(input=final_prompt)
        input_ids = self.tokenizer.encode(final_prompt, add_special_tokens=True, return_tensors='pt')[0]
        
        input_ids = torch.cat([
            input_ids[:3],
            torch.tensor([-200], dtype=torch.long),
            input_ids[3:],
        ], dim=0)
        
        return dict(input_ids=input_ids[:self.max_length])

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            original_idx = data_sample['idx']
            
            sample = self.dataset[original_idx]
            
            image_data = sample['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif hasattr(image_data, 'convert'):
                image = image_data.convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except:
                    raise TypeError(f"Unknown type: {type(image_data)}")

            data = self._process_image(image)
            data.update(self._process_text(""))
            data.update(type='recon')
            
            return data

        except Exception as e:
            print(f"Error when processing index {idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return self._retry()
        
    def __len__(self):

        return len(self.data_list) if self.data_list else 0