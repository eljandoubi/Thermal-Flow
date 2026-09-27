import torch

from thermal_flow.evaluation.detection import (
    evaluate_detector,
    mean_average_precision,
    precision_recall_at_iou,
)


def _perfect_detection() -> tuple[list[dict], list[dict]]:
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]
    return preds, targets


def test_mean_average_precision_perfect_match() -> None:
    preds, targets = _perfect_detection()

    result = mean_average_precision(preds, targets)

    assert result["map_50"] == 1.0
    assert result["map_50_95"] == 1.0


def test_mean_average_precision_no_predictions() -> None:
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]
    preds = [
        {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    ]

    result = mean_average_precision(preds, targets)

    assert result["map_50"] == 0.0


def test_precision_recall_perfect_match() -> None:
    preds, targets = _perfect_detection()

    result = precision_recall_at_iou(preds, targets)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_precision_recall_false_positive_and_false_negative() -> None:
    preds = [
        {
            "boxes": torch.tensor([[100.0, 100.0, 110.0, 110.0]]),  # far from any GT box
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]

    result = precision_recall_at_iou(preds, targets)

    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_precision_recall_ignores_low_confidence_predictions() -> None:
    preds, targets = _perfect_detection()
    preds[0]["scores"] = torch.tensor([0.1])  # below default score_threshold=0.5

    result = precision_recall_at_iou(preds, targets, score_threshold=0.5)

    assert result["true_positives"] == 0
    assert result["false_negatives"] == 1


def test_precision_recall_handles_no_ground_truth() -> None:
    preds = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]

    result = precision_recall_at_iou(preds, targets)

    assert result["false_positives"] == 1
    assert result["true_positives"] == 0


def test_evaluate_detector_wraps_detector_and_targets() -> None:
    def dummy_detector(images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
            for _ in images
        ]

    images = [torch.rand(3, 32, 32)]
    targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]

    metrics = evaluate_detector(dummy_detector, images, targets)

    assert metrics["map_50"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
