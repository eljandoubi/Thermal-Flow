"""Line/bar plots for experiment result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display server required
import matplotlib.pyplot as plt


def plot_metric_vs_x(
    results: list[dict],
    x_key: str,
    y_key: str,
    output_path: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
) -> Path:
    """Save a line plot of `y_key` vs `x_key` from a list of result dicts (e.g. the
    output of `experiments.fewshot.run_fewshot_sweep`).

    A `None` `x_key` value (the "Full" few-shot setting) is plotted after all
    numeric values and labeled "Full" on the x-axis.
    """
    numeric_entries = [(r[x_key], r[y_key]) for r in results if r[x_key] is not None]
    none_values = [r[y_key] for r in results if r[x_key] is None]

    xs = [x for x, _ in numeric_entries]
    ys = [y for _, y in numeric_entries]
    tick_positions = list(xs)
    tick_labels = [str(x) for x in xs]

    if none_values:
        full_x = max(xs) * 1.2 if xs else 1.0
        xs = xs + [full_x] * len(none_values)
        ys = ys + none_values
        tick_positions.append(full_x)
        tick_labels.append("Full")

    order = sorted(range(len(xs)), key=lambda i: xs[i])

    fig, ax = plt.subplots()
    ax.plot([xs[i] for i in order], [ys[i] for i in order], marker="o")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(x_label or x_key)
    ax.set_ylabel(y_label or y_key)
    if title:
        ax.set_title(title)

    return _save_and_close(fig, output_path)


def plot_ablation_bar(
    results: dict[str, dict[str, float]],
    metric_key: str,
    output_path: str | Path,
    y_label: str | None = None,
    title: str | None = None,
) -> Path:
    """Bar chart of `metric_key` across named ablation configs (the dict returned
    by `experiments.ablation.run_ablation`).
    """
    names = list(results)
    values = [results[name][metric_key] for name in names]

    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_ylabel(y_label or metric_key)
    if title:
        ax.set_title(title)
    fig.autofmt_xdate(rotation=30)

    return _save_and_close(fig, output_path)


def _save_and_close(fig: matplotlib.figure.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
