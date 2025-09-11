from src.datasets.text2image.text2image import MidjourneyReconstructionDataset
from mmengine.config import read_base
from src.datasets.collate_functions import collate_func_gen, CollateConcat
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index


dataset = dict(
    type=MidjourneyReconstructionDataset,
    image_size=image_size,
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    crop_image=False,
    max_length=128,
    unconditional=0.1,
)

group_keys = ['recon']
repeat = [1]
batch_size = 16

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(
        type=FixedBatchMultiSourceSampler,
        repeat=repeat,
        batch_size=batch_size,
        shuffle=True
    ),
    collate_fn=dict(
        type=CollateConcat,
        collate_fns=[
            dict(
                type=collate_func_gen,
                pad_index=pad_index
            ),
        ],
        keys=group_keys
    )
)
