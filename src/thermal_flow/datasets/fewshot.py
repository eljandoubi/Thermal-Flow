"""Few-shot subsampling protocol: restrict thermal supervision to N training pairs
while validation/test splits stay untouched.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset


def few_shot_subset(dataset: Dataset, num_samples: int | None, seed: int = 0) -> Dataset:
    """Return a deterministic subset of `num_samples` pairs from `dataset`.

    `num_samples=None` (or a value >= len(dataset)) returns `dataset` unchanged,
    representing the "Full" setting in the few-shot sweep (10/50/100/500/1000/Full).
    """
    total = len(dataset)
    if num_samples is None or num_samples >= total:
        return dataset
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:num_samples].tolist()
    return Subset(dataset, indices)
