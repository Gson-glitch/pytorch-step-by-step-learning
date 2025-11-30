import torch
from torch.utils.data import random_split, WeightedRandomSampler
from torch import Tensor, LongTensor
from typing import List, Sequence


def index_splitter(n: int, splits: Sequence[float], seed: int = 13) -> list[Tensor]:
    """Split indices 0..n-1 into parts based on given proportions."""
    idx: Tensor = torch.arange(n)
    splits_tensor: Tensor = torch.as_tensor(splits)
    total: Tensor = splits_tensor.sum().float()
    
    if not total.isclose(torch.ones(1)[0]):
        splits_tensor = splits_tensor / total
    
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)


def index_splitter_v2(n: int, train_frac: float = 0.8, seed: int = 13) -> tuple[LongTensor, LongTensor]:
    """Return shuffled train and validation indices."""
    torch.manual_seed(seed)
    idx = torch.randperm(n)
    train_size = int(train_frac * n)
    return idx[:train_size], idx[train_size:]

def make_balanced_sampler(y: Tensor) -> WeightedRandomSampler:
    """Return a sampler that balances classes in y."""
    classes, counts = y.unique(return_counts=True)
    weights: Tensor = 1.0 / counts.float()
    sample_weights: Tensor = weights[y.squeeze().long()]
    
    generator = torch.Generator()
    sampler: WeightedRandomSampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True
    )
    return sampler
