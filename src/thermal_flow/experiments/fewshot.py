"""Few-shot sweep orchestration: 10 -> 50 -> 100 -> 500 -> 1000 -> Full thermal pairs."""

from __future__ import annotations

from collections.abc import Callable

# None represents the "Full" (unrestricted) setting.
FEWSHOT_SWEEP: tuple[int | None, ...] = (10, 50, 100, 500, 1000, None)


def run_fewshot_sweep(
    evaluate_fn: Callable[[int | None], dict[str, float]],
    num_samples_sweep: tuple[int | None, ...] = FEWSHOT_SWEEP,
) -> list[dict]:
    """Call `evaluate_fn(num_samples)` for each setting in the sweep and collect results.

    `evaluate_fn` is expected to train (or load a checkpoint for) a model restricted
    to `num_samples` thermal pairs and return its evaluation metrics; `num_samples=None`
    represents the "Full" setting. Returns one result dict per setting, each augmented
    with its `num_samples` value (suitable input for `visualization.plots.plot_metric_vs_x`).
    """
    return [{"num_samples": n, **evaluate_fn(n)} for n in num_samples_sweep]
