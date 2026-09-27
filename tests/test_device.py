import pytest
import torch

from thermal_flow.device import get_device


def test_auto_selects_available_device() -> None:
    device = get_device()
    assert device.type in {"cuda", "mps", "cpu"}


def test_explicit_cpu_always_available() -> None:
    assert get_device("cpu").type == "cpu"


def test_unknown_device_type_raises() -> None:
    with pytest.raises(ValueError):
        get_device("tpu")


def test_explicit_cuda_matches_availability() -> None:
    if torch.cuda.is_available():
        assert get_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            get_device("cuda")


def test_explicit_mps_matches_availability() -> None:
    if torch.backends.mps.is_available():
        assert get_device("mps").type == "mps"
    else:
        with pytest.raises(RuntimeError):
            get_device("mps")
