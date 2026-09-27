from pathlib import Path

from thermal_flow.visualization.plots import plot_ablation_bar, plot_metric_vs_x


def test_plot_metric_vs_x_creates_file_with_full_setting(tmp_path: Path) -> None:
    results = [
        {"num_samples": 10, "mae": 0.5},
        {"num_samples": 100, "mae": 0.3},
        {"num_samples": None, "mae": 0.1},
    ]
    output_path = tmp_path / "fewshot.png"

    returned_path = plot_metric_vs_x(
        results, x_key="num_samples", y_key="mae", output_path=output_path
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_metric_vs_x_without_full_setting(tmp_path: Path) -> None:
    results = [{"num_samples": 10, "mae": 0.5}, {"num_samples": 100, "mae": 0.3}]
    output_path = tmp_path / "fewshot_no_full.png"

    plot_metric_vs_x(results, x_key="num_samples", y_key="mae", output_path=output_path)

    assert output_path.exists()


def test_plot_ablation_bar_creates_file(tmp_path: Path) -> None:
    results = {"FM": {"mae": 0.4}, "FM+SSIM": {"mae": 0.3}}
    output_path = tmp_path / "ablation.png"

    plot_ablation_bar(results, metric_key="mae", output_path=output_path)

    assert output_path.exists()
