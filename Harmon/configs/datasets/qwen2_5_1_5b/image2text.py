from src.datasets.understanding.caption_datasets import CaptionDataset
# from src.datasets.understanding.llava_datasets import LLaVADataset
from mmengine.config import read_base
from src.datasets.collate_functions import collate_func_und, CollateConcat
# from mmengine.dataset import DefaultSampler
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler
# from xtuner.dataset import LLaVADataset
from src.datasets.understanding.llava_datasets import LLaVADataset

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.utils import expand2square

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index, image_length


data_root = 'data/LLaVA-Instruct-150K/'
data_path = 'data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = 'data/LLaVA-Instruct-150K/tuning_data'
max_length = int(2048 - (336 / 14) ** 2)

dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
)


group_keys = ['image2text',]
repeat = [1]
batch_size = 32
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=FixedBatchMultiSourceSampler,
                 repeat=repeat,
                 batch_size=batch_size,    # fixed batch size for all sources
                 shuffle=True),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=collate_func_und,
                                      pad_index=pad_index),
                                 # dict(type=collate_func_gen,
                                 #      pad_index=pad_index),
                                 ],
                    keys=group_keys
                    )
)
