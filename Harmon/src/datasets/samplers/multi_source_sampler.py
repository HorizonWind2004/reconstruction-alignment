# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Iterator, List, Optional, Sized, Union
import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler


class FixedBatchMultiSourceSampler(Sampler):
    r"""Multi-Source Infinite Sampler.

    According to the sampling ratio, sample data from different
    datasets to form batches.

    Args:
        repeat (tuple): repeat factor
        dataset (Sized): The dataset.
        batch_size (int): Size of mini-batch.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """

    def __init__(self,
                 repeat,
                 dataset: Sized,
                 batch_size: int,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:

        assert hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        assert len(repeat) == len(dataset.cumulative_sizes), \
            'The length of repeat must be equal to ' \
            f'the number of datasets, but got repeat={repeat}'

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.repeat = repeat
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        self.batch_size = batch_size

        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.source2inds = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size: int) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size: int) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.world_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """Not supported in `epoch-based runner."""
        pass

    def __iter__(self) -> Iterator[int]:
        while True:
            for source, repeat in enumerate(self.repeat):
                for _ in range(repeat):
                    batch_buffer_per_source = []
                    while len(batch_buffer_per_source) < self.batch_size:
                        idx = next(self.source2inds[source])
                        idx += self.cumulative_sizes[source]
                        batch_buffer_per_source.append(idx)

                    yield from batch_buffer_per_source
