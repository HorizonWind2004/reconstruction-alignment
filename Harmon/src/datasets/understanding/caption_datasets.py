from torch.utils.data import Dataset
from PIL import Image
import os
import io
import json
import random
import torch
import numpy as np
from einops import rearrange
try:
    from aoss_client.client import Client
except:
    try:
        from petrel_client.client import Client
    except:
        Client = None
from glob import glob
from xtuner.registry import BUILDER
from xtuner.dataset.utils import expand2square
from src.datasets.utils import crop2square, encode_fn
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.datasets.understanding.caption_prompts import dense_prompts, short_prompts


class CaptionDataset(Dataset):
    def __init__(self,
                 data_path,
                 local_folder,
                 image_size,
                 ceph_folder=None,
                 ceph_config=None,
                 tokenizer=None,
                 template_map_fn=None,
                 max_length=2048,
                 min_image_size=80,
                 image_length=256,
                 pad_image=True,
                 brief=False,
                 cap_folder=None,
                 cap_source='caption',
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.local_folder = local_folder
        self.cap_folder = local_folder if cap_folder is None else cap_folder
        self.cap_source = cap_source

        self.image_size = image_size

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = template_map_fn['template']
        self.template_map_fn = BUILDER.build(template_map_fn)
        self.max_length = max_length
        self.image_length = image_length
        self.pad_image = pad_image
        self.min_image_size = min_image_size

        self.FILE_CLIENT = None
        self.ceph_folder = ceph_folder
        self.ceph_config = ceph_config
        self.use_ceph = ((Client is not None) and (ceph_folder is not None)
                         and (ceph_config is not None) and os.path.exists(ceph_config))

        self.brief = brief
        self.caption_prompts = short_prompts if self.brief else dense_prompts

    def _load_data(self, data_path: str):      # image path and annotation path are saved in a json file
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            json_files = glob(f"{data_path}/*.json")
            data_list = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data_list += json.load(f)

            self.data_list = data_list

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_ceph(self, ceph_path):
        if self.FILE_CLIENT is None:
            self.FILE_CLIENT = Client(self.ceph_config)
        data_bytes = self.FILE_CLIENT.get(ceph_path)

        return io.BytesIO(data_bytes)

    def _read_image(self, image_file):
        if self.use_ceph:
            image = Image.open(
                self._read_ceph(
                    os.path.join(self.ceph_folder, image_file)
                )
            )
        else:
            image = Image.open(
                os.path.join(self.local_folder, image_file)
            )
        assert image.width > self.min_image_size and image.height > self.min_image_size, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image.convert('RGB')

    def _read_json(self, annotation_file):
        if self.use_ceph:
            annotation = json.load(
                self._read_ceph(
                    os.path.join(self.ceph_folder, annotation_file)
                )
            )
        else:
            with open(os.path.join(self.local_folder, annotation_file), 'r') as f:
                annotation = json.load(f)

        return annotation

    def _process_image(self, image):
        data = dict()
        if self.pad_image:
            image = expand2square(image, (127, 127, 127))
        else:
            image = crop2square(image)

        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)
        return data

    def _process_text(self, text):
        assert DEFAULT_IMAGE_TOKEN not in text, text
        data_dict = dict(conversation=[{'input': f"{DEFAULT_IMAGE_TOKEN}\n{random.choice(self.caption_prompts)}",
                                        'output': text.strip()}])
        data_dict.update(self.template_map_fn(data_dict))
        data_dict.update(encode_fn(data_dict, self.tokenizer, self.max_length,
                                   self.image_length, True, True))

        assert (torch.tensor(data_dict['input_ids']).long() == IMAGE_TOKEN_INDEX).sum() == self.image_length, \
            "Error in image format"

        data_dict['type'] = 'image2text'
        return data_dict

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            data = self._process_image(image)
            del image
            with open(f"{self.cap_folder}/{data_sample['annotation']}", 'r') as f:
                caption = json.load(f)[self.cap_source]
            data.update(self._process_text(caption))

            data.update(image_dir=self.local_folder, image_file=data_sample['image'])

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{data_sample['image']}: {e}", flush=True)
            return self._retry()
