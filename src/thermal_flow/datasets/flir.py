"""FLIR ADAS paired RGB/thermal dataset.

The raw FLIR ADAS download is not organized as a normalized `rgb/`/`thermal/` pair
of folders out of the box. Convert or symlink the extracted RGB and thermal images
into `root/<split>/{rgb,thermal}/<name>.jpg` (matching filenames across the two
folders) before using this loader; see `PairedRGBThermalDataset`.
"""

from __future__ import annotations

from thermal_flow.datasets.common import PairedRGBThermalDataset


class FLIRDataset(PairedRGBThermalDataset):
    """FLIR ADAS: RGB + thermal automotive imagery (both loaded as 3-channel)."""
