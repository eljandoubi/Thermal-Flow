"""Downstream object-detection evaluation (project proposal section 16).

Compares how well a detector performs on real thermal, synthetic (generated)
thermal, and RGB inputs, using COCO-style mAP (via torchmetrics) plus simple
fixed-threshold precision/recall.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

Detections = list[dict[str, torch.Tensor]]
Detector = Callable[[list[torch.Tensor]], Detections]


def mean_average_precision(preds: Detections, targets: Detections) -> dict[str, float]:
    """COCO-style mAP@50, mAP@50:95, and mean average recall (@100 detections)."""
    metric = MeanAveragePrecision(iou_type="bbox", backend="faster_coco_eval")
    metric.update(preds, targets)
    result = metric.compute()
    return {
        "map_50": result["map_50"].item(),
        "map_50_95": result["map"].item(),
        "mar_100": result["mar_100"].item(),
    }


def precision_recall_at_iou(
    preds: Detections,
    targets: Detections,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict[str, float]:
    """Single-threshold precision/recall via greedy, confidence-ordered IoU matching.

    Predictions below `score_threshold` are discarded; a prediction is a true
    positive if its IoU with the best still-unmatched, same-label ground-truth
    box is >= `iou_threshold`.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, target in zip(preds, targets, strict=True):
        keep = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][keep]
        pred_labels = pred["labels"][keep]
        pred_scores = pred["scores"][keep]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        order = torch.argsort(pred_scores, descending=True)
        for idx in order:
            label = pred_labels[idx]
            same_label = gt_labels == label
            if gt_boxes.shape[0] == 0 or not bool(same_label.any()):
                false_positives += 1
                continue

            ious = box_iou(pred_boxes[idx : idx + 1], gt_boxes).squeeze(0)
            ious = ious.masked_fill(~same_label | matched, 0.0)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou >= iou_threshold:
                true_positives += 1
                matched[best_idx] = True
            else:
                false_positives += 1

        false_negatives += int((~matched).sum().item())

    denom_p = true_positives + false_positives
    denom_r = true_positives + false_negatives
    return {
        "precision": true_positives / denom_p if denom_p > 0 else 0.0,
        "recall": true_positives / denom_r if denom_r > 0 else 0.0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


@torch.no_grad()
def evaluate_detector(
    detector: Detector,
    images: list[torch.Tensor],
    targets: Detections,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict[str, float]:
    """Run `detector` over `images` and score it against `targets`.

    `detector` follows the torchvision detection API: given a list of (C, H, W)
    image tensors, it returns a list of dicts with "boxes"/"scores"/"labels".
    Call this once per input type (real thermal / synthetic thermal / RGB) with
    the same `detector` and `targets` to reproduce the section 16 comparison.
    """
    preds = detector(images)
    metrics = mean_average_precision(preds, targets)
    metrics.update(precision_recall_at_iou(preds, targets, iou_threshold, score_threshold))
    return metrics
