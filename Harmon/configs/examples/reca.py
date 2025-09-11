from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from torch.optim import AdamW
from mmengine.config import read_base
from src.models.harmon_dev import HarmonDev

with read_base():
    from ..models.qwen2_5_1_5b_kl16_mar_h import model
    from ..datasets.qwen2_5_1_5b.image2text_recon_mid import train_dataloader, repeat


#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
model.update(
    type=HarmonDev,
    pretrained_pth='checkpoints/harmon_1.5b.pth',
    freeze_llm=False,)

# Scheduler & Optimizer
accumulative_counts = sum(repeat)
dataloader_num_workers = 4
max_iters = 50000
optim_type = AdamW
lr = 1e-5
betas = (0.9, 0.95)
weight_decay = 0.02
max_norm = 1.0  # grad clip
warmup_ratio = 0.01

# Save
save_steps = 1000
save_total_limit = -1  # Maximum checkpoints to keep (-1 means unlimited)


#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataloader = train_dataloader

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    constructor='MAROptimWrapperConstructor',
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=warmup_ratio * max_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=False,
        begin=warmup_ratio * max_iters,
        end=max_iters)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_iters=max_iters)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed but disable `deterministic` to avoid CuBLAS issues
randomness = dict(seed=42, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
