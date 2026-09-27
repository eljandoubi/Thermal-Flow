# Thermal-Flow

Few-shot RGB-to-Thermal domain adaptation using a lightweight conditional Flow Matching model,
built incrementally as a set of small, independently testable milestones.

## Overview

Thermal cameras help autonomous-driving perception in conditions where RGB struggles (night,
low light, haze), but thermal datasets are scarce and expensive to collect. This project learns
to translate RGB images into synthetic thermal images using only a small number of paired
RGB-thermal examples, then evaluates the result both visually (MAE/RMSE/PSNR/SSIM) and
functionally (does it help a downstream object detector?).

**Research question:** Can a lightweight conditional Flow Matching model learn a useful
RGB-to-thermal mapping when only a small amount of paired thermal data is available?

The full proposal (motivation, architecture, training pipeline, ablations, evaluation plan) lives
in `tmp/` locally and is intentionally not tracked in git (see `.gitignore`).

## Hardware / device notes

This model is developed on macOS (CPU/MPS) but is meant to **train on an NVIDIA GPU on a separate
machine**. No special configuration is needed to get there:

- `uv.lock` is resolved for all platforms ("universal resolution"), so it already pins
  CUDA-capable Linux wheels for `torch` (with the matching `nvidia-*` runtime packages) alongside
  the macOS build. Running `uv sync` on the GPU machine installs the CUDA build automatically.
- All code selects its compute device through [`thermal_flow.device.get_device`](src/thermal_flow/device.py),
  which prefers CUDA, then Apple MPS, then CPU — the same code runs unmodified on both machines.
- If the GPU machine's driver doesn't support the CUDA version bundled by default, pin a specific
  `pytorch-cuXXX` index per [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/);
  not required out of the box.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) (manages the Python version and virtual environment for
you — no manual `venv`/`pip` steps).

```bash
# install dependencies (creates .venv, installs the pinned Python 3.11)
uv sync

# run the test suite
uv run pytest

# lint / format check
uv run ruff check .
uv run ruff format --check .
```

## Project status

Built step by step; every milestone below shipped with its own unit tests and its own commit.

- [x] **M0 — Project scaffolding**: uv-managed package (`thermal_flow`), pytest + ruff, device
      selection utility, README.
- [x] **M1 — Flow Matching core**: time embedding, lightweight conditional U-Net, probability path
      / velocity target, Flow Matching loss.
- [x] **M2 — ODE sampler / inference**: Euler/RK4 solvers integrating RGB → thermal, end-to-end
      sampling.
- [x] **M3 — Parameter-efficient adapter**: frozen backbone + trainable bottleneck adapter for
      few-shot fine-tuning.
- [x] **M4 — Additional losses**: SSIM + edge-preservation losses, combined weighted objective.
- [x] **M5 — Dataset pipeline**: FLIR ADAS / LLVIP loaders, preprocessing, fixed val/test splits,
      few-shot subsampling protocol.
- [x] **M6 — Baselines**: CNN regression, U-Net, Pix2Pix (generator + discriminator).
- [x] **M7 — Training pipeline**: full-supervision / few-shot / adapter-finetuning training loops,
      YAML configs, checkpointing.
- [x] **M8 — Evaluation metrics**: MAE, RMSE, PSNR, SSIM, efficiency (params/size/time/FLOPs).
- [x] **M9 — Downstream detection evaluation**: mAP@50, mAP@50:95, precision/recall on
      real/synthetic/RGB thermal inputs.
- [x] **M10 — Experiments, ablations & visualization**: few-shot sweep/ablation orchestration,
      efficiency benchmarking, qualitative comparisons, plots.

What's intentionally **not** included: the actual FLIR ADAS/LLVIP image files (must be obtained
from their providers and converted into the expected layout, see `Datasets` below), a trained
pretrained object detector (`evaluate_detector` is detector-agnostic — plug in any torchvision-style
detector), and a CLI/argparse wrapper (the training/experiment entry points are plain importable
functions, exercised directly from Python or a notebook).

## Project structure

```
src/thermal_flow/
├── device.py             # compute device selection: cuda > mps > cpu (M0)
├── models/                # timestep embedding, conditional U-Net, flow matching, adapter (M1, M3)
├── inference/             # Euler/RK4 ODE solvers, end-to-end sampling (M2)
├── losses/                # flow matching, SSIM, edge losses, combined objective (M1, M4)
├── datasets/              # FLIR/LLVIP loaders, preprocessing, few-shot subsampling (M5)
├── baselines/             # CNN, U-Net, Pix2Pix baselines (M6)
├── training/               # config schema, checkpointing, train/fewshot/adapter loops (M7)
├── evaluation/             # image metrics, efficiency, downstream detection (M8, M9)
├── experiments/            # fewshot sweep, ablation runner, efficiency benchmarking (M10)
└── visualization/          # metric plots, qualitative comparison grids (M10)
configs/                   # example YAML training configs (flir/llvip/fewshot)
tests/                     # unit tests (flat, one file per module, synthetic data/fixtures only)
```

## Datasets

- **FLIR ADAS** (primary) — paired RGB/thermal automotive imagery with object annotations.
- **LLVIP** (secondary) — aligned visible/infrared pedestrian imagery, used to validate that the
  approach generalizes beyond one dataset.

Both require obtaining the data from their respective providers directly; dataset loaders (M5)
will document the expected on-disk layout and are unit-tested against small synthetic fixtures so
the test suite never needs the real datasets.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
