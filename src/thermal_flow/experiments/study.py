"""Few-shot study: Flow Matching against three baselines, trained on the same pairs
with the same number of optimizer steps and scored on the same test images.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from thermal_flow.baselines.cnn import CNNRegressionBaseline
from thermal_flow.baselines.pix2pix import PatchGANDiscriminator, Pix2PixGenerator
from thermal_flow.baselines.unet import UNetBaseline
from thermal_flow.datasets.common import PairedRGBThermalDataset
from thermal_flow.datasets.fewshot import few_shot_subset
from thermal_flow.device import get_device
from thermal_flow.evaluation.image_metrics import all_metrics
from thermal_flow.experiments.benchmark import benchmark_models
from thermal_flow.inference.sample import sample
from thermal_flow.losses.combined import CombinedFlowMatchingLoss
from thermal_flow.models.flow_matching import ConditionalFlowMatching
from thermal_flow.models.unet import ConditionalUNet
from thermal_flow.visualization.plots import plot_metric_vs_x
from thermal_flow.visualization.qualitative import save_comparison_grid

METHODS = ("cnn", "unet", "pix2pix", "flow_matching")
ODE_STEPS = 8
Predictor = Callable[[torch.Tensor], torch.Tensor]


class CachedPairs(Dataset):
    """Resized pairs held in memory as uint8, so a run never decodes the large
    source images twice. With a limit, the pairs are evenly spaced over the
    dataset, because neighbouring files are consecutive video frames.
    """

    def __init__(self, dataset: Dataset, limit: int | None = None) -> None:
        total = len(dataset)
        if limit is None or limit >= total:
            indices = list(range(total))
        else:
            indices = torch.linspace(0, total - 1, limit).round().long().tolist()
        items = [dataset[i] for i in indices]
        self.rgb = torch.stack([(x["rgb"] * 255).round().byte() for x in items])
        self.thermal = torch.stack([(x["thermal"] * 255).round().byte() for x in items])

    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "rgb": self.rgb[index].float() / 255,
            "thermal": self.thermal[index].float() / 255,
        }


def endless_batches(dataset: Dataset, batch_size: int, seed: int) -> Iterator[dict]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) > batch_size,
        generator=generator,
    )
    while True:
        yield from loader


def train_regressor(model: nn.Module, stream, steps: int, lr: float, device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        batch = next(stream)
        rgb, thermal = batch["rgb"].to(device), batch["thermal"].to(device)
        loss = F.l1_loss(model(rgb), thermal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_pix2pix(
    generator: nn.Module,
    discriminator: nn.Module,
    stream,
    steps: int,
    lr: float,
    device,
    l1_weight: float = 100.0,
) -> None:
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    generator.train()
    discriminator.train()
    for _ in range(steps):
        batch = next(stream)
        rgb, thermal = batch["rgb"].to(device), batch["thermal"].to(device)
        fake = generator(rgb)

        real_score = discriminator(rgb, thermal)
        fake_score = discriminator(rgb, fake.detach())
        d_loss = 0.5 * (
            F.mse_loss(real_score, torch.ones_like(real_score))
            + F.mse_loss(fake_score, torch.zeros_like(fake_score))
        )
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        fake_score = discriminator(rgb, fake)
        g_loss = F.mse_loss(fake_score, torch.ones_like(fake_score)) + l1_weight * F.l1_loss(
            fake, thermal
        )
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


def train_flow_matching(
    model: ConditionalFlowMatching, stream, steps: int, lr: float, device
) -> None:
    loss_fn = CombinedFlowMatchingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        batch = next(stream)
        rgb, thermal = batch["rgb"].to(device), batch["thermal"].to(device)
        loss = loss_fn(model, rgb, thermal)["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def fit_method(
    name: str,
    data: Dataset,
    steps: int,
    batch_size: int,
    lr: float,
    base_channels: int,
    device,
    seed: int,
) -> tuple[nn.Module, Predictor]:
    """Train one method and return its model with a function mapping RGB to thermal."""
    torch.manual_seed(seed)
    stream = endless_batches(data, batch_size, seed)
    if name == "cnn":
        model = CNNRegressionBaseline().to(device)
        train_regressor(model, stream, steps, lr, device)
        return model.eval(), model
    if name == "unet":
        model = UNetBaseline(base_channels=base_channels).to(device)
        train_regressor(model, stream, steps, lr, device)
        return model.eval(), model
    if name == "pix2pix":
        generator = Pix2PixGenerator(base_channels=base_channels).to(device)
        discriminator = PatchGANDiscriminator(base_channels=base_channels).to(device)
        train_pix2pix(generator, discriminator, stream, steps, lr, device)
        return generator.eval(), generator
    if name == "flow_matching":
        model = ConditionalFlowMatching(ConditionalUNet(base_channels=base_channels)).to(device)
        train_flow_matching(model, stream, steps, lr, device)
        return model, lambda rgb: sample(model, rgb, num_steps=ODE_STEPS)
    raise ValueError(f"unknown method {name!r}, expected one of {METHODS}")


@torch.no_grad()
def evaluate(predict: Predictor, test_data: Dataset, batch_size: int, device) -> dict[str, float]:
    """MAE, RMSE, PSNR and SSIM of each test batch, averaged over the batches."""
    totals: dict[str, float] = {}
    batches = 0
    for batch in DataLoader(test_data, batch_size=batch_size):
        rgb, thermal = batch["rgb"].to(device), batch["thermal"].to(device)
        for key, value in all_metrics(predict(rgb).clamp(0, 1), thermal).items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / batches for key, value in totals.items()}


def efficiency_table(base_channels: int, image_size: int, device) -> dict[str, dict[str, float]]:
    models: dict[str, nn.Module] = {
        "cnn": CNNRegressionBaseline(),
        "unet": UNetBaseline(base_channels=base_channels),
        "pix2pix": Pix2PixGenerator(base_channels=base_channels),
        "flow_matching": ConditionalUNet(base_channels=base_channels),
    }
    for model in models.values():
        model.to(device).eval()

    def inputs(name: str, _model: nn.Module) -> tuple[torch.Tensor, ...]:
        rgb = torch.rand(1, 3, image_size, image_size, device=device)
        if name == "flow_matching":
            return rgb, torch.full((1,), 0.5, device=device), rgb
        return (rgb,)

    with torch.no_grad():
        table = benchmark_models(models, inputs, num_iterations=10, warmup_iterations=3)
    table["flow_matching"]["network_evaluations"] = float(ODE_STEPS)
    return table


def run_study(
    root: str,
    sizes: tuple[int | None, ...],
    methods: tuple[str, ...],
    steps: int,
    batch_size: int,
    lr: float,
    image_size: int,
    base_channels: int,
    test_images: int,
    seeds: tuple[int, ...],
    output_dir: str,
    device_name: str | None = None,
    max_train: int | None = None,
) -> list[dict]:
    device = get_device(device_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_full = CachedPairs(
        PairedRGBThermalDataset(root, "train", image_size=image_size), max_train
    )
    test_data = CachedPairs(
        PairedRGBThermalDataset(root, "test", image_size=image_size), test_images
    )

    rows: list[dict] = []
    grid_inputs = next(iter(DataLoader(test_data, batch_size=4)))
    grids: dict[tuple[str, int | None], torch.Tensor] = {}
    for size in sizes:
        for method in methods:
            for seed in seeds:
                data = few_shot_subset(train_full, size, seed=seed)
                start = time.time()
                model, predict = fit_method(
                    method, data, steps, batch_size, lr, base_channels, device, seed
                )
                seconds = time.time() - start
                metrics = evaluate(predict, test_data, batch_size, device)
                rows.append(
                    {
                        "method": method,
                        "num_samples": size,
                        "train_pairs": len(data),
                        "seed": seed,
                        "steps": steps,
                        "train_seconds": round(seconds, 1),
                        **metrics,
                    }
                )
                print(rows[-1], flush=True)
                if seed == seeds[0]:
                    with torch.no_grad():
                        grids[(method, size)] = predict(grid_inputs["rgb"].to(device)).cpu()
                del model, predict

    efficiency = efficiency_table(base_channels, image_size, device)
    write_outputs(rows, efficiency, grids, grid_inputs, out, sizes)
    return rows


def write_outputs(rows, efficiency, grids, grid_inputs, out: Path, sizes) -> None:
    (out / "study.json").write_text(json.dumps(rows, indent=2))
    (out / "efficiency.json").write_text(json.dumps(efficiency, indent=2))

    methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = ["| method | pairs | MAE | RMSE | PSNR (dB) | SSIM |", "|---|---|---|---|---|---|"]
    for size in sizes:
        for method in methods:
            subset = [r for r in rows if r["method"] == method and r["num_samples"] == size]
            if not subset:
                continue
            mean = {
                k: sum(r[k] for r in subset) / len(subset) for k in ("mae", "rmse", "psnr", "ssim")
            }
            label = "all" if size is None else str(size)
            lines.append(
                f"| {method} | {label} | {mean['mae']:.4f} | {mean['rmse']:.4f} "
                f"| {mean['psnr']:.2f} | {mean['ssim']:.3f} |"
            )
    (out / "study.md").write_text("\n".join(lines) + "\n")

    for method in methods:
        series = [
            {
                "num_samples": size,
                "psnr": sum(
                    r["psnr"] for r in rows if r["method"] == method and r["num_samples"] == size
                )
                / max(
                    1, sum(1 for r in rows if r["method"] == method and r["num_samples"] == size)
                ),
            }
            for size in sizes
        ]
        plot_metric_vs_x(series, "num_samples", "psnr", out / f"psnr_{method}.png", title=method)

    for size in sizes:
        for method in methods:
            if (method, size) in grids:
                save_comparison_grid(
                    grid_inputs["rgb"],
                    grid_inputs["thermal"],
                    grids[(method, size)],
                    out / f"grid_{method}_{'all' if size is None else size}.png",
                )


def parse_sizes(values: list[str]) -> tuple[int | None, ...]:
    return tuple(None if v in ("full", "all") else int(v) for v in values)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Few-shot RGB to thermal study")
    parser.add_argument(
        "--root", required=True, help="Folder with train/ and test/ in the paired layout"
    )
    parser.add_argument("--sizes", nargs="+", default=["10", "100", "1000", "full"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--steps", type=int, default=2000, help="Optimizer steps per run")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--test-images", type=int, default=500)
    parser.add_argument(
        "--max-train", type=int, default=None, help="Keep N evenly spaced training pairs"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output", default="results/study")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    run_study(
        args.root,
        parse_sizes(args.sizes),
        tuple(args.methods),
        args.steps,
        args.batch_size,
        args.lr,
        args.image_size,
        args.base_channels,
        args.test_images,
        tuple(args.seeds),
        args.output,
        args.device,
        args.max_train,
    )


if __name__ == "__main__":
    main()
