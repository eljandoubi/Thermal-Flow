import pytest
from torch.utils.data import Dataset

from thermal_flow.datasets.fewshot import few_shot_subset


class _RangeDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


def test_few_shot_subset_returns_requested_count() -> None:
    dataset = _RangeDataset(100)

    subset = few_shot_subset(dataset, num_samples=10, seed=0)

    assert len(subset) == 10


def test_few_shot_subset_is_deterministic_given_seed() -> None:
    dataset = _RangeDataset(100)

    subset_a = few_shot_subset(dataset, num_samples=10, seed=42)
    subset_b = few_shot_subset(dataset, num_samples=10, seed=42)

    assert list(subset_a) == list(subset_b)


def test_few_shot_subset_none_returns_full_dataset() -> None:
    dataset = _RangeDataset(20)
    assert few_shot_subset(dataset, num_samples=None) is dataset


def test_few_shot_subset_clamped_to_full_dataset_when_too_large() -> None:
    dataset = _RangeDataset(5)
    assert few_shot_subset(dataset, num_samples=1000) is dataset


def test_few_shot_subset_rejects_negative() -> None:
    dataset = _RangeDataset(5)
    with pytest.raises(ValueError):
        few_shot_subset(dataset, num_samples=-1)
