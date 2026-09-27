from thermal_flow.experiments.ablation import run_ablation


def test_run_ablation_maps_each_named_config() -> None:
    configs = {"FM": 1, "FM+SSIM": 2, "FM+SSIM+Edge": 3}

    results = run_ablation(configs, evaluate_fn=lambda x: {"value": x * 10})

    assert set(results) == set(configs)
    assert results["FM+SSIM"]["value"] == 20


def test_run_ablation_calls_evaluate_fn_with_the_config_value() -> None:
    seen = []

    run_ablation({"a": "config-a"}, evaluate_fn=lambda cfg: seen.append(cfg) or {})

    assert seen == ["config-a"]
