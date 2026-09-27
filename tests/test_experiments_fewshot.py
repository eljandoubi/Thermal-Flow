from thermal_flow.experiments.fewshot import FEWSHOT_SWEEP, run_fewshot_sweep


def test_run_fewshot_sweep_calls_once_per_setting_in_order() -> None:
    calls = []

    def evaluate_fn(num_samples: int | None) -> dict[str, float]:
        calls.append(num_samples)
        return {"mae": 1.0 / (num_samples or 10_000)}

    results = run_fewshot_sweep(evaluate_fn)

    assert calls == list(FEWSHOT_SWEEP)
    assert [r["num_samples"] for r in results] == list(FEWSHOT_SWEEP)
    assert all("mae" in r for r in results)


def test_run_fewshot_sweep_supports_custom_sweep() -> None:
    results = run_fewshot_sweep(lambda n: {"score": float(n or -1)}, num_samples_sweep=(1, 2, None))

    assert [r["num_samples"] for r in results] == [1, 2, None]
    assert results[-1]["score"] == -1.0
