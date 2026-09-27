"""Convert raw FLIR ADAS / LLVIP downloads into the normalized on-disk layout
`PairedRGBThermalDataset` expects: `root/<split>/{rgb,thermal}/<name>.ext`.

Neither dataset can be fetched with a plain guessed URL: FLIR ADAS is only
available after filling out a registration form on the vendor's site, and
LLVIP is distributed as a single archive shared from the dataset authors' own
Google Drive / Baidu Netdisk (see
https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md). This
module automates what can actually be automated: downloading the LLVIP
archive from that verified, publicly linked Google Drive file, and
reorganizing either dataset's extracted contents once it is on disk.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# From the dataset authors' own download page (linked above): a single file,
# not a folder, so it can be fetched directly with gdown without a browser.
LLVIP_GOOGLE_DRIVE_FILE_ID = "1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"


@dataclass
class SplitReport:
    """Outcome of converting one split: how many pairs matched, and what didn't."""

    split: str
    matched: int
    unmatched_rgb: list[str] = field(default_factory=list)
    unmatched_thermal: list[str] = field(default_factory=list)


def _list_images(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.iterdir()) if p.suffix.lower() in _IMAGE_EXTENSIONS}


def _pair_by_stem(
    rgb_dir: Path, thermal_dir: Path
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    rgb_images = _list_images(rgb_dir)
    thermal_images = _list_images(thermal_dir)
    shared = sorted(set(rgb_images) & set(thermal_images))
    pairs = [(rgb_images[name], thermal_images[name]) for name in shared]
    # report actual filenames (with extension), not the bare stems used for matching
    unmatched_rgb = sorted(rgb_images[name].name for name in set(rgb_images) - set(thermal_images))
    unmatched_thermal = sorted(
        thermal_images[name].name for name in set(thermal_images) - set(rgb_images)
    )
    return pairs, unmatched_rgb, unmatched_thermal


def _find_split_dir(root: Path, modality: str, split: str) -> Path | None:
    """Find a directory named `split` whose parent is named `modality`, under `root`."""
    for path in sorted(root.rglob(split)):
        if path.is_dir() and path.parent.name == modality:
            return path
    return None


def _place(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _convert_split(
    rgb_dir: Path, thermal_dir: Path, output_root: Path, split: str, mode: str, dry_run: bool
) -> SplitReport:
    pairs, unmatched_rgb, unmatched_thermal = _pair_by_stem(rgb_dir, thermal_dir)
    for rgb_path, thermal_path in pairs:
        _place(rgb_path, output_root / split / "rgb" / rgb_path.name, mode, dry_run)
        _place(thermal_path, output_root / split / "thermal" / thermal_path.name, mode, dry_run)
    return SplitReport(
        split=split,
        matched=len(pairs),
        unmatched_rgb=unmatched_rgb,
        unmatched_thermal=unmatched_thermal,
    )


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extract a .zip/.tar(.gz) archive into `destination`, returning `destination`."""
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(destination, filter="data")
    else:
        raise ValueError(f"Unrecognized archive format: {archive_path}")
    return destination


def _resolve_source(source: Path) -> Path:
    """Extract `source` first if it's an archive; pass directories through unchanged."""
    if source.is_dir():
        return source
    if source.is_file():
        return extract_archive(source, source.parent / f"{source.stem}_extracted")
    raise FileNotFoundError(f"{source} does not exist")


def convert_llvip(
    source: Path,
    output: Path,
    splits: tuple[str, ...] = ("train", "test"),
    mode: str = "copy",
    dry_run: bool = False,
) -> dict[str, SplitReport]:
    """Reorganize an LLVIP download (extracted, or the archive itself) into
    `output/<split>/{rgb,thermal}`.

    LLVIP ships `visible/<split>` and `infrared/<split>` folders; those are
    located anywhere under `source` (searched by directory name), so this
    works regardless of the exact top-level folder name inside the archive.
    """
    source = _resolve_source(Path(source))
    reports: dict[str, SplitReport] = {}
    for split in splits:
        visible_dir = _find_split_dir(source, "visible", split)
        infrared_dir = _find_split_dir(source, "infrared", split)
        if visible_dir is None or infrared_dir is None:
            raise FileNotFoundError(
                f"Could not find both 'visible/{split}' and 'infrared/{split}' under {source}."
            )
        reports[split] = _convert_split(visible_dir, infrared_dir, output, split, mode, dry_run)
    return reports


def convert_flir(
    source: Path,
    output: Path,
    rgb_dirname: str = "RGB",
    thermal_dirname: str = "thermal_8_bit",
    splits: tuple[str, ...] = ("train", "val"),
    mode: str = "copy",
    dry_run: bool = False,
) -> dict[str, SplitReport]:
    """Reorganize a FLIR ADAS download (extracted, or the archive itself) into
    `output/<split>/{rgb,thermal}`.

    FLIR's exact folder names have changed across releases; the defaults match
    the commonly published `<split>/RGB` + `<split>/thermal_8_bit` layout, but
    pass `rgb_dirname`/`thermal_dirname`/`splits` to match whatever you
    actually downloaded if these don't apply. Use `dry_run=True` to check the
    match counts before copying anything.
    """
    source = _resolve_source(Path(source))
    reports: dict[str, SplitReport] = {}
    for split in splits:
        rgb_dir = source / split / rgb_dirname
        thermal_dir = source / split / thermal_dirname
        if not rgb_dir.is_dir() or not thermal_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {rgb_dir} and {thermal_dir} to exist. Pass --rgb-dirname/"
                "--thermal-dirname/--splits to match your download's actual layout."
            )
        reports[split] = _convert_split(rgb_dir, thermal_dir, output, split, mode, dry_run)
    return reports


def download_llvip(output_dir: Path) -> Path:
    """Download the official LLVIP archive from Google Drive via `gdown`.

    Requires the optional `data` dependency group: `uv sync --extra data`.
    """
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download LLVIP; install it with `uv sync --extra data`."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={LLVIP_GOOGLE_DRIVE_FILE_ID}"
    downloaded = gdown.download(url, output=f"{output_dir}/", quiet=False)
    if downloaded is None:
        raise RuntimeError("gdown failed to download the LLVIP archive.")
    return Path(downloaded)


def _report_summary(reports: dict[str, SplitReport]) -> str:
    lines = []
    for split, report in reports.items():
        lines.append(
            f"{split}: {report.matched} pairs matched, "
            f"{len(report.unmatched_rgb)} rgb-only, {len(report.unmatched_thermal)} thermal-only"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermal-flow-data", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-llvip", help="Download the LLVIP archive from Google Drive"
    )
    download_parser.add_argument("--output", type=Path, default=Path("data/llvip_raw"))

    convert_llvip_parser = subparsers.add_parser(
        "convert-llvip", help="Reorganize an LLVIP download into the expected layout"
    )
    convert_llvip_parser.add_argument("--source", type=Path, required=True)
    convert_llvip_parser.add_argument("--output", type=Path, required=True)
    convert_llvip_parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    convert_llvip_parser.add_argument("--dry-run", action="store_true")

    convert_flir_parser = subparsers.add_parser(
        "convert-flir", help="Reorganize a FLIR ADAS download into the expected layout"
    )
    convert_flir_parser.add_argument("--source", type=Path, required=True)
    convert_flir_parser.add_argument("--output", type=Path, required=True)
    convert_flir_parser.add_argument("--rgb-dirname", default="RGB")
    convert_flir_parser.add_argument("--thermal-dirname", default="thermal_8_bit")
    convert_flir_parser.add_argument("--splits", nargs="+", default=["train", "val"])
    convert_flir_parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    convert_flir_parser.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "download-llvip":
            print(f"Downloaded {download_llvip(args.output)}")
        elif args.command == "convert-llvip":
            print(
                _report_summary(
                    convert_llvip(args.source, args.output, mode=args.mode, dry_run=args.dry_run)
                )
            )
        elif args.command == "convert-flir":
            print(
                _report_summary(
                    convert_flir(
                        args.source,
                        args.output,
                        rgb_dirname=args.rgb_dirname,
                        thermal_dirname=args.thermal_dirname,
                        splits=tuple(args.splits),
                        mode=args.mode,
                        dry_run=args.dry_run,
                    )
                )
            )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
