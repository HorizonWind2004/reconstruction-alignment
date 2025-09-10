# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
from glob import glob
import tarfile
import tempfile
import pyarrow.parquet as pq
import random
from PIL import Image
import hashlib
from pathlib import Path
import traceback
import numpy as np  # Add import for numpy

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 20_000_000


class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        Text-to-Image dataset loader with distributed training support.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        """Get the list of parquet file paths"""
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        """
        Iterator method for data loading.
        Returns: A training sample for each iteration
        """
        # Get data paths and ID for current worker
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # Resume from checkpoint if data status is available
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            # Otherwise start from the beginning
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, rg#{row_group_start_id}, row#{row_start_id}"
        )

        # Infinite loop to enable dataset reuse
        while True:
            # Get data paths starting from current parquet file index
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                # Initialize Arrow filesystem and open parquet file
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    # Get row group IDs
                    row_group_ids = list(range(fr.num_row_groups))
                    # Start from specified row group ID
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        # Read specified row group and convert to pandas DataFrame
                        df = fr.read_row_group(row_group_id).to_pandas()
                        # Start from specified row index
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0  # Initialize token count
                            try:
                                # Read image byte data and convert to RGB PIL image
                                image_byte = row['image']
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            # Apply image transformation
                            image_tensor = self.transform(image)
                            # Calculate image token count based on dimensions and stride
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride ** 2

                            try:
                                # Read and parse caption data for the image
                                caption_dict = row['captions']
                                caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                print(f'Error: {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue

                            # Encode all captions as token sequences
                            caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                            if len(caps_token) == 0:
                                # Use space character if no captions available
                                print(f'no caption in rg#{row_group_id}, {parquet_file_path}')
                                caption_token = self.tokenizer.encode(' ')
                            else:
                                # Randomly select one caption as training label
                                caption_token = random.choice(caps_token)

                            # Initialize sequence plan and text ID list
                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            # Update token count by adding text token count
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            
                            # Add text processing plan (caption part)
                            sequence_plan.append({
                                'type': 'text',
                                'enable_cfg': 1,
                                'loss': 0,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })
                        
                            # Add image processing plan (VAE image part)
                            sequence_plan.append({
                                'type': 'vae_image',
                                'enable_cfg': 0,
                                'loss': 1,
                                'special_token_loss': 0,
                                'special_token_label': None,
                            })

                            # Build sample dictionary with all training info
                            sample = dict(
                                image_tensor_list=[image_tensor],
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                            )
                            yield sample  # Return a sample

                        # Reset row start ID after processing a row group
                        row_start_id = 0
                    # Reset row group start ID after processing a parquet file
                    row_group_start_id = 0
            # Reset parquet file start ID after processing all parquet files (restart)
            parquet_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class ReconstructionDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
        cache_dir=None, from_huggingface=False, path_huggingface=None
    ):
        """
        Reconstruction Dataset processor for loading webdataset format data from tar packages.
        Supports caching mechanism to improve data loading efficiency.
        Also supports loading data directly from Hugging Face datasets.
        
        Args:
            from_huggingface: Whether to load data from Hugging Face instead of tar files
            path_huggingface: Path to the Hugging Face dataset, e.g., "brivangl/midjourney-v6-llava"
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.from_huggingface = from_huggingface
        self.path_huggingface = path_huggingface
        
        from data.consts import get_recon_prompt_list
        import random
        self.prompt_templates = get_recon_prompt_list()
        # Initialize cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "bagel_reconstruction_cache")
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Using cache directory: {self.cache_dir}")
        
        # Store cache path mappings for extracted tar files
        self.tar_cache_dirs = {}
        # Store filename to cache path mapping
        self.file_cache_paths = {}
        
        # Get data paths
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        if self.data_paths:  # Only set epoch when data paths are not empty
            self.set_epoch()
        else:
            print(f"Warning: No data files found for {dataset_name}. Check your data_dir_list: {data_dir_list}")
        
    def get_data_paths(self, data_dir_list, num_used_data):
        """Get tar file paths and their internal file lists"""
        # If loading from Hugging Face, use that method instead
        if self.from_huggingface and self.path_huggingface:
            return self._load_huggingface_data(num_used_data)
            
        all_items = []
        # Iterate through each data directory
        print(f"Loading data from directories: {data_dir_list}")
        
        # Expand path wildcards
        if data_dir_list and len(data_dir_list) > 0:
            data_dir_list = glob(os.path.expanduser(data_dir_list[0].replace('{', '[').replace('}', ']')))
        else:
            print("Warning: Empty data_dir_list provided")
            return all_items
            
        print(f"Found tar files: {data_dir_list}")

        for tar_path in data_dir_list:
            try:
                # Extract tar file to cache directory
                extract_dir = self._extract_tar_if_needed(tar_path)
                
                # Get all extracted image files
                image_files = []
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, extract_dir)
                        
                        # Only save image files to cache mapping
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.file_cache_paths[rel_path] = file_path
                            image_files.append(rel_path)
                
                # Each image file as a separate data item
                valid_groups = []
                for img_path in image_files:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    valid_groups.append((tar_path, base_name, [img_path]))
                
                all_items.extend(valid_groups)
                print(f"Found {len(valid_groups)} valid images in {tar_path}")
                print(f"Found {len(valid_groups)} valid image-json pairs in {tar_path}")
            except Exception as e:
                print(f"Error processing tar file {tar_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Limit data amount if needed
        print(f'Total items found: {len(all_items)}, num_used_data: {num_used_data}')
        if num_used_data and num_used_data[0] > 0 and num_used_data[0] < len(all_items):
            all_items = all_items[:num_used_data[0]]
            print(f"Limited to {len(all_items)} items")
            
        return all_items
        
    def _load_huggingface_data(self, num_used_data):
        """Load data from Hugging Face dataset"""
        all_items = []
        try:
            from datasets import load_dataset
            print(f"Loading dataset from {self.path_huggingface} with cache_dir {self.cache_dir}")
            dataset = load_dataset(self.path_huggingface, cache_dir=self.cache_dir)['train']
            print(f"Loaded {len(dataset)} samples from {self.path_huggingface}")
            
            # Create virtual items
            for idx in range(len(dataset)):
                all_items.append(('huggingface', f'sample_{idx}', [f'sample_{idx}']))
                
            print(f"Added {len(all_items)} samples from Hugging Face dataset {self.path_huggingface}")
            
            # Save the dataset for later use in __iter__
            self.hf_dataset = dataset
            
        except Exception as e:
            print(f"Error loading Hugging Face dataset: {e}")
            traceback.print_exc()
        
        # Limit data amount if needed
        if num_used_data and num_used_data[0] > 0 and num_used_data[0] < len(all_items):
            all_items = all_items[:num_used_data[0]]
            print(f"Limited to {len(all_items)} items")
            
        return all_items
                
    def _get_cached_image_path(self, tar_path, file_name):
        """Generate cached image path"""
        # Create a unique cache filename using tar path and internal filename
        tar_hash = hashlib.md5(tar_path.encode()).hexdigest()[:8]
        file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
        cache_name = f"{tar_hash}_{file_hash}_{os.path.basename(file_name)}"
        return os.path.join(self.cache_dir, cache_name)
    
    def _extract_and_cache(self, tar_path, file_name):
        """Get file path from already extracted cache directory"""
        # If file is already in cache mapping, return directly
        if file_name in self.file_cache_paths:
            return self.file_cache_paths[file_name]
        
        # Ensure tar is extracted
        extract_dir = self._extract_tar_if_needed(tar_path)
        
        # Build file path in cache
        cache_path = os.path.join(extract_dir, file_name)
        
        # If file exists, return path
        if os.path.exists(cache_path):
            self.file_cache_paths[file_name] = cache_path
            return cache_path
            
        print(f"Error: File {file_name} not found in extracted tar {tar_path}")
        return None
    
    def _extract_tar_if_needed(self, tar_path):
        """Extract tar file to cache directory if not already extracted"""
        import tarfile
        import hashlib
        
        # If this tar file is already extracted, return cache directory path
        if tar_path in self.tar_cache_dirs:
            return self.tar_cache_dirs[tar_path]
        
        # Use hash of file path as subdirectory name to avoid conflicts
        tar_hash = hashlib.md5(tar_path.encode()).hexdigest()
        extract_dir = os.path.join(self.cache_dir, tar_hash)
        
        # Create lock file path
        lock_file = os.path.join(extract_dir, '.extraction_complete')
        
        # Check if extraction is already complete
        if os.path.exists(lock_file):
            print(f"Using cached extraction for {tar_path} in {extract_dir}", flush=True)
            self.tar_cache_dirs[tar_path] = extract_dir
            return extract_dir
        
        # Need to extract
        print(f"Extracting {tar_path} to {extract_dir}...", flush=True)
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
            
            # Create lock file to mark extraction complete
            with open(lock_file, 'w') as f:
                f.write(f"Extracted from {tar_path} at {os.path.getmtime(tar_path)}")
                
            print(f"Extraction complete: {tar_path} -> {extract_dir}", flush=True)
            self.tar_cache_dirs[tar_path] = extract_dir
            return extract_dir
        except Exception as e:
            print(f"Error extracting {tar_path}: {e}", flush=True)
            if os.path.exists(extract_dir):
                # Delete incomplete extraction directory
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise
    
    def __iter__(self):
        """
        Iterator method for loading data from tar files or Hugging Face datasets
        Returns: A training sample for each iteration
        """
        # Get data paths and ID for current worker
        data_items_per_worker, worker_id = self.get_data_paths_per_worker()
        
        # Resume from checkpoint if data status is available
        if self.data_status is not None:
            item_start_id = self.data_status[worker_id] + 1
        else:
            item_start_id = 0
            
        transform_stride = self.transform.stride  # Image transform stride
        
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at item#{item_start_id}"
        )
        
        # Infinite loop to enable dataset reuse
        while True:
            # Get data items starting from current index
            data_items_ = data_items_per_worker[item_start_id:]
            
            for item_idx, (tar_path, base_name, files) in enumerate(data_items_, start=item_start_id):
                try:
                    # If using Hugging Face dataset
                    if self.from_huggingface and tar_path == 'huggingface':
                        # Get the index from base_name (sample_X)
                        idx = int(base_name.split('_')[1]) 
                        sample = self.hf_dataset[idx]
                        
                        # Process the image from Hugging Face dataset
                        image_data = sample['image']
                        if isinstance(image_data, dict) and 'bytes' in image_data:
                            image = pil_img2rgb(Image.open(io.BytesIO(image_data['bytes'])))
                        elif hasattr(image_data, 'convert'):
                            image = pil_img2rgb(image_data.convert('RGB'))
                        elif isinstance(image_data, bytes):
                            image = pil_img2rgb(Image.open(io.BytesIO(image_data)))
                        else:
                            # Try to convert using numpy array
                            try:
                                image = pil_img2rgb(Image.fromarray(image_data).convert('RGB'))
                            except:
                                print(f"Unable to process image type: {type(image_data)}")
                                continue
                    else:
                        # Regular tar file processing
                        # Find image file
                        image_file = next((f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))), None)
                        
                        if not image_file:
                            continue
                        
                        # Extract and cache image file
                        image_cache_path = self._extract_and_cache(tar_path, image_file)
                        if not image_cache_path:
                            continue
                        
                        # Load image
                        try:
                            image = pil_img2rgb(Image.open(image_cache_path))
                        except Exception as e:
                            print(f"Error loading image {image_cache_path}: {e}")
                            continue
                    
                    # Directly randomly select a prompt from the template, not relying on JSON metadata
                    prompt = random.choice(self.prompt_templates)
                    
                    # Apply image transformation
                    image_tensor = self.transform(image)
                    vit_image_tensor = self.vit_transform(image)
                    
                    num_tokens = 0
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                    
                    height_vit, width_vit = vit_image_tensor.shape[1:]
                    num_tokens += width_vit * height_vit // self.vit_transform.stride ** 2
                    
                    # Encode prompt text as token sequence
                    caption_token = self.tokenizer.encode(prompt)
                    
                    # Initialize sequence plan and text ID list
                    sequence_plan, text_ids_list = [], []
                    text_ids = caption_token
                    
                    # Update token count by adding text token count
                    num_tokens += len(caption_token)
                    text_ids_list.append(text_ids)

                    sequence_plan.append({
                        'type': 'vit_image',
                        'enable_cfg': 0,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Add text processing plan (prompt part)
                    sequence_plan.append({
                        'type': 'text',
                        'enable_cfg': 1,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Add image processing plan (reconstruction target)
                    sequence_plan.append({
                        'type': 'vae_image',
                        'enable_cfg': 0,
                        'loss': 1,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                    
                    # Build sample dictionary
                    sample = dict(
                        image_tensor_list=[vit_image_tensor, image_tensor],
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": item_idx,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )
                    
                    yield sample  # Return a sample
                
                except Exception as e:
                    print(f"Error processing item {base_name} from {tar_path}: {e}")
                    traceback.print_exc()
                    continue
            
            # Reset start ID after processing all data (restart)
            item_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
            
    def cleanup_cache(self):
        """Clean up temporary files in cache directory"""
        if os.path.exists(self.cache_dir):
            print(f"Cleaning up cache directory: {self.cache_dir}")
            for cached_file in os.listdir(self.cache_dir):
                try:
                    os.remove(os.path.join(self.cache_dir, cached_file))
                except Exception as e:
                    print(f"Error removing cached file {cached_file}: {e}")
