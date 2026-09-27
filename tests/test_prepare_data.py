import tarfile
import zipfile
from pathlib import Path

import pytest

import thermal_flow.datasets.prepare as prepare
from thermal_flow.datasets.prepare import (
    LLVIP_GOOGLE_DRIVE_FILE_ID,
    convert_flir,
    convert_llvip,
    download_llvip,
    extract_archive,
    main,
)


def _touch_images(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def _make_llvip_tree(root: Path) -> None:
    _touch_images(root / "visible" / "train", ["010001.jpg", "010002.jpg"])
    _touch_images(root / "infrared" / "train", ["010001.jpg", "010002.jpg"])
    _touch_images(root / "visible" / "test", ["190001.jpg"])
    _touch_images(root / "infrared" / "test", ["190001.jpg"])


def test_convert_llvip_reorganizes_into_expected_layout(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _make_llvip_tree(source)
    output = tmp_path / "prepared"

    reports = convert_llvip(source, output)

    assert reports["train"].matched == 2
    assert reports["test"].matched == 1
    assert (output / "train" / "rgb" / "010001.jpg").exists()
    assert (output / "train" / "thermal" / "010001.jpg").exists()
    assert (output / "test" / "rgb" / "190001.jpg").exists()


def test_convert_llvip_locates_nested_directories(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "LLVIP-extracted" / "LLVIP"
    _make_llvip_tree(source)
    output = tmp_path / "prepared"

    reports = convert_llvip(tmp_path / "raw", output)

    assert reports["train"].matched == 2


def test_convert_llvip_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _make_llvip_tree(source)
    output = tmp_path / "prepared"

    reports = convert_llvip(source, output, dry_run=True)

    assert reports["train"].matched == 2
    assert not output.exists()


def test_convert_llvip_reports_unmatched_files(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _touch_images(source / "visible" / "train", ["a.jpg", "b.jpg"])
    _touch_images(source / "infrared" / "train", ["a.jpg", "c.jpg"])
    _touch_images(source / "visible" / "test", [])
    _touch_images(source / "infrared" / "test", [])

    reports = convert_llvip(source, tmp_path / "prepared")

    assert reports["train"].matched == 1
    assert reports["train"].unmatched_rgb == ["b.jpg"]
    assert reports["train"].unmatched_thermal == ["c.jpg"]


def test_convert_llvip_missing_directories_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_llvip(tmp_path, tmp_path / "prepared")


def test_convert_llvip_nonexistent_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_llvip(tmp_path / "does-not-exist", tmp_path / "prepared")


def test_convert_llvip_symlink_mode_links_to_original_file(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _make_llvip_tree(source)
    output = tmp_path / "prepared"

    convert_llvip(source, output, mode="symlink")

    linked = output / "train" / "rgb" / "010001.jpg"
    assert linked.is_symlink()
    assert linked.resolve() == (source / "visible" / "train" / "010001.jpg").resolve()


def test_convert_llvip_accepts_zip_archive_directly(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_llvip_tree(raw)
    archive = tmp_path / "LLVIP.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in raw.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(tmp_path))

    reports = convert_llvip(archive, tmp_path / "prepared")

    assert reports["train"].matched == 2


def test_convert_flir_reorganizes_with_default_dirnames(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _touch_images(source / "train" / "RGB", ["FLIR_00001.jpg", "FLIR_00002.jpg"])
    _touch_images(source / "train" / "thermal_8_bit", ["FLIR_00001.jpeg", "FLIR_00002.jpeg"])
    _touch_images(source / "val" / "RGB", ["FLIR_09001.jpg"])
    _touch_images(source / "val" / "thermal_8_bit", ["FLIR_09001.jpeg"])
    output = tmp_path / "prepared"

    reports = convert_flir(source, output)

    assert reports["train"].matched == 2
    assert reports["val"].matched == 1
    assert (output / "train" / "rgb" / "FLIR_00001.jpg").exists()
    assert (output / "train" / "thermal" / "FLIR_00001.jpeg").exists()


def test_convert_flir_supports_custom_dirnames_and_splits(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _touch_images(source / "images_train" / "rgb8", ["a.jpg"])
    _touch_images(source / "images_train" / "ir8", ["a.jpg"])

    reports = convert_flir(
        source,
        tmp_path / "prepared",
        rgb_dirname="rgb8",
        thermal_dirname="ir8",
        splits=("images_train",),
    )

    assert reports["images_train"].matched == 1


def test_convert_flir_missing_directories_raises_with_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rgb-dirname"):
        convert_flir(tmp_path, tmp_path / "prepared")


def test_extract_archive_handles_zip(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/hello.txt", "hi")

    destination = extract_archive(archive, tmp_path / "extracted")

    assert (destination / "nested" / "hello.txt").read_text() == "hi"


def test_extract_archive_handles_tar(tmp_path: Path) -> None:
    payload = tmp_path / "hello.txt"
    payload.write_text("hi")
    archive = tmp_path / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname="nested/hello.txt")

    destination = extract_archive(archive, tmp_path / "extracted")

    assert (destination / "nested" / "hello.txt").read_text() == "hi"


def test_extract_archive_rejects_unknown_format(tmp_path: Path) -> None:
    not_an_archive = tmp_path / "data.bin"
    not_an_archive.write_bytes(b"not an archive")

    with pytest.raises(ValueError, match="Unrecognized archive"):
        extract_archive(not_an_archive, tmp_path / "extracted")


def test_llvip_file_id_matches_the_official_download_page() -> None:
    # https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md
    assert LLVIP_GOOGLE_DRIVE_FILE_ID == "1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"


def test_download_llvip_uses_the_verified_file_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdown = pytest.importorskip("gdown")
    captured = {}

    def fake_download(url: str, output: str, quiet: bool) -> str:
        captured["url"] = url
        captured["output"] = output
        return f"{output}LLVIP.zip"

    monkeypatch.setattr(gdown, "download", fake_download)

    result = download_llvip(tmp_path / "raw")

    assert LLVIP_GOOGLE_DRIVE_FILE_ID in captured["url"]
    assert result == tmp_path / "raw" / "LLVIP.zip"


def test_download_llvip_without_gdown_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "gdown":
            raise ImportError("simulated: gdown not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="uv sync --extra data"):
        download_llvip(Path("unused"))


def test_download_llvip_raises_when_gdown_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gdown = pytest.importorskip("gdown")
    monkeypatch.setattr(gdown, "download", lambda url, output, quiet: None)

    with pytest.raises(RuntimeError, match="gdown failed"):
        download_llvip(tmp_path / "raw")


def test_cli_convert_llvip_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "raw"
    _make_llvip_tree(source)
    output = tmp_path / "prepared"

    exit_code = main(["convert-llvip", "--source", str(source), "--output", str(output)])

    assert exit_code == 0
    assert (output / "train" / "rgb" / "010001.jpg").exists()
    assert "matched" in capsys.readouterr().out


def test_cli_reports_errors_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["convert-llvip", "--source", str(tmp_path), "--output", str(tmp_path / "out")]
    )

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_convert_flir_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "raw"
    _touch_images(source / "train" / "RGB", ["FLIR_00001.jpg"])
    _touch_images(source / "train" / "thermal_8_bit", ["FLIR_00001.jpeg"])
    output = tmp_path / "prepared"

    exit_code = main(
        [
            "convert-flir",
            "--source",
            str(source),
            "--output",
            str(output),
            "--splits",
            "train",
        ]
    )

    assert exit_code == 0
    assert (output / "train" / "rgb" / "FLIR_00001.jpg").exists()
    assert "matched" in capsys.readouterr().out


def test_cli_download_llvip_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(prepare, "download_llvip", lambda output: output / "LLVIP.zip")

    exit_code = main(["download-llvip", "--output", str(tmp_path / "raw")])

    assert exit_code == 0
    assert "Downloaded" in capsys.readouterr().out
