"""LLVIP paired visible/infrared dataset.

The raw LLVIP download is not organized as a normalized `rgb/`/`thermal/` pair of
folders out of the box. Convert or symlink the extracted visible and infrared
images into `root/<split>/{rgb,thermal}/<name>.jpg` (matching filenames across the
two folders) before using this loader; see `PairedRGBThermalDataset`.
"""

from __future__ import annotations

from thermal_flow.datasets.common import PairedRGBThermalDataset


class LLVIPDataset(PairedRGBThermalDataset):
    """LLVIP: aligned visible/infrared pedestrian imagery (both loaded as 3-channel)."""
