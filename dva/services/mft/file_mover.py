"""File Mover (Boundary 1).

Owns MFT discovery, metadata, download and completeness verification.
It never parses retailer layouts and never performs business validation.

Download contract (03, section 2; 04, Boundary 1):

    1. Download into ``<raw_dir>/<name>.part``.
    2. Verify completeness (size, then optional checksum).
    3. Atomically rename to ``<raw_dir>/<name>`` (the "ready" file).
    4. Only the ready file may ever be consumed; ``*.part`` never is.

Raw files are transient. Deletion is the orchestrator's job and only
happens after successful Parquet verification (04, Boundary 8); the
mover exposes ``cleanup_raw_file`` for that purpose.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from models.file_models import SourceFile
from services.mft.transport import Transport
from utils.checksums import checksum_file, checksums_match


class DownloadError(Exception):
    """Base class for download/verification failures."""


class DownloadIncomplete(DownloadError):
    """Downloaded size does not match the expected MFT size."""


class ChecksumMismatch(DownloadError):
    """Downloaded content does not match the MFT checksum."""


@dataclass(slots=True)
class DownloadResult:
    """Outcome of one file download attempt."""

    file_id: str
    local_path: Path | None
    success: bool
    message: str = ""
    already_present: bool = False


class FileMover:
    """Transfers files from MFT into the local raw staging area."""

    def __init__(self, transport: Transport, raw_dir: str | Path) -> None:
        self._transport = transport
        self._raw_dir = Path(raw_dir)

    # ------------------------------------------------------------------
    # Path helpers: the two staging forms of a source file.
    # ------------------------------------------------------------------
    def _part_path(self, name: str) -> Path:
        """Destination while a download is in progress."""
        return self._raw_dir / f"{name}.part"

    def _ready_path(self, name: str) -> Path:
        """Destination after successful verification (the parseable file)."""
        return self._raw_dir / name

    # ------------------------------------------------------------------
    # Discovery & metadata (Boundary 1).
    # ------------------------------------------------------------------
    def discover_files(
        self, remote_path: str, pattern: str | None = None
    ) -> list[SourceFile]:
        """List non-directory files under ``remote_path`` as SourceFiles.

        Args:
            remote_path: directory on the MFT share to scan.
            pattern: optional glob (e.g. ``*.txt``) to filter by name.
        """
        discovered: list[SourceFile] = []
        for info in self._transport.list_files(remote_path):
            if info.is_dir:
                continue  # files only; directories are never ingested
            if pattern is not None and not fnmatch.fnmatch(info.name, pattern):
                continue
            discovered.append(
                SourceFile(
                    name=info.name,
                    remote_path=f"{remote_path.rstrip('/')}/{info.name}",
                    expected_size=info.size,
                )
            )
        return discovered

    def get_metadata(self, source_file: SourceFile) -> SourceFile:
        """Refresh the expected size of one file from the MFT."""
        size = self._transport.get_size(source_file.remote_path)
        source_file.expected_size = size
        return source_file

    # ------------------------------------------------------------------
    # Download + verification (Boundary 1).
    # ------------------------------------------------------------------
    def download_file(self, source_file: SourceFile) -> DownloadResult:
        """Download one file and leave a verified ready file in raw/.

        On restart-based recovery a verified ready file may already exist
        (raw is retained on parse failure); it is not re-downloaded.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        ready = self._ready_path(source_file.name)
        if ready.exists() and self._size_matches(ready, source_file.expected_size):
            return DownloadResult(
                file_id=source_file.file_id,
                local_path=ready,
                success=True,
                message="already present",
                already_present=True,
            )

        part = self._part_path(source_file.name)
        try:
            self._transport.get(source_file.remote_path, str(part))
        except Exception as exc:  # transport-level failure (network, auth...)
            self._remove_if_exists(part)
            return DownloadResult(
                file_id=source_file.file_id,
                local_path=None,
                success=False,
                message=f"download failed: {exc}",
            )

        try:
            self._verify_part(part, source_file)
        except DownloadError as exc:
            self._remove_if_exists(part)
            return DownloadResult(
                file_id=source_file.file_id,
                local_path=None,
                success=False,
                message=str(exc),
            )

        os.replace(part, ready)  # atomic within the same filesystem
        return DownloadResult(
            file_id=source_file.file_id,
            local_path=ready,
            success=True,
            message="downloaded",
        )

    def verify_download(
        self, part_path: str | Path, source_file: SourceFile
    ) -> None:
        """Raise DownloadError unless the ``.part`` file is complete.

        Exposed for explicit verification before any rename; the normal
        path calls this internally from ``download_file``.
        """
        self._verify_part(Path(part_path), source_file)

    def _verify_part(self, part: Path, source_file: SourceFile) -> None:
        """Size check first (cheap), then optional checksum check."""
        if source_file.expected_size is not None:
            actual = part.stat().st_size
            if actual != source_file.expected_size:
                raise DownloadIncomplete(
                    f"size mismatch for {source_file.name}: expected "
                    f"{source_file.expected_size}, got {actual}."
                )

        if source_file.checksum:
            actual_checksum = checksum_file(part, algorithm="md5")
            if not checksums_match(actual_checksum, source_file.checksum):
                raise ChecksumMismatch(
                    f"checksum mismatch for {source_file.name}."
                )

    def _size_matches(self, path: Path, expected: int | None) -> bool:
        """True when ``path`` is a non-empty file matching the expected size."""
        if expected is None:
            return False  # unknown size: treat as stale, re-download
        try:
            return path.is_file() and path.stat().st_size == expected
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Cleanup (Boundary 8). Orchestrator calls this only after Parquet
    # write + dataset verification succeed.
    # ------------------------------------------------------------------
    def cleanup_raw_file(self, source_file: SourceFile) -> bool:
        """Delete the raw staging file (ready or leftover ``.part``).

        Returns True when the file was removed (or never existed).
        """
        targets = [
            self._ready_path(source_file.name),
            self._part_path(source_file.name),
        ]
        removed = False
        for target in targets:
            self._remove_if_exists(target)
            if not target.exists():
                removed = True
        return removed

    @staticmethod
    def _remove_if_exists(path: Path) -> None:
        """Best-effort delete; never raises."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
