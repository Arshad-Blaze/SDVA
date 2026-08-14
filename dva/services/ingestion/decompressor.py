"""Decompressor (Boundary 2): verified raw -> decompressed local temp.

Gzip and zip inputs are expanded byte-by-byte (streaming), so even a very
large compressed file is decompressed incrementally instead of being held
in memory. Decompression never performs business validation; the orchestrator
owns the resulting temp file's lifecycle (deleted only after COMPLETE).
"""

from __future__ import annotations

import gzip
import zipfile
from dataclasses import dataclass
from pathlib import Path

_COMPRESSED_SUFFIXES = (".gz", ".gzip", ".zip")


class DecompressionError(Exception):
    """Raised when a compressed file cannot be expanded safely."""


@dataclass(frozen=True, slots=True)
class DecompressedInput:
    """A verified raw file plus its expanded local temporary copy."""
    raw_path: Path
    decompressed_path: Path


def is_compressed(path: str | Path) -> bool:
    """True when the file looks like a gzip or zip archive (by suffix)."""
    return Path(path).name.lower().endswith(_COMPRESSED_SUFFIXES)


def decompressed_name(raw_path: str | Path) -> str:
    """Unique temp name so the output can never collide with a raw file."""
    return f"{Path(raw_path).name}.decompressed"


def prepare_input(
    raw_path: str | Path, dest_dir: str | Path
) -> DecompressedInput | None:
    """Expand ``raw_path`` into ``dest_dir`` when it is compressed.

    Returns ``None`` for already-readable files (the caller parses the raw
    path directly). Transient-only: never deletes the source archive.
    """
    raw = Path(raw_path)
    if not is_compressed(raw):
        return None
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / decompressed_name(raw)
    decompress_to(raw, target)
    return DecompressedInput(raw_path=raw, decompressed_path=target)


def decompress_to(raw_path: str | Path, target: str | Path) -> Path:
    """Stream-expand gzip/zip into ``target`` (overwrites any previous file)."""
    raw = Path(raw_path)
    target = Path(target)
    try:
        if raw.name.lower().endswith(".zip"):
            _expand_zip(raw, target)
        else:
            _expand_gzip(raw, target)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DecompressionError(
            f"cannot decompress {raw.name}: {exc}"
        ) from exc
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise DecompressionError(f"decompressed {raw.name} is empty")
    return target


def _expand_gzip(raw: Path, target: Path) -> None:
    """Incremental gzip expansion (1 MiB chunks)."""
    with gzip.open(raw, "rb") as src, open(target, "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)


def _expand_zip(raw: Path, target: Path) -> None:
    """Expand the first regular member of a zip archive (streamed).

    Multi-member archives are deliberately ambiguous for a flat file
    pipeline; expanding a single representative member keeps the output a
    single file the Detector can handle.
    """
    with zipfile.ZipFile(raw) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if not members:
            raise DecompressionError(f"zip archive {raw.name} contains no files")
        with archive.open(members[0]) as src, open(target, "wb") as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)