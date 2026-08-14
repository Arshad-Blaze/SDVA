"""Registry persistence and crash recovery for the pipeline (Phase 9).

The registry (``<workspace>/registry.json``) is the crash-recovery
contract: every status transition rewrites it atomically, and a restarted
pipeline rehydrates from it. ``recover_inflight`` pulls transient working
states back to stable checkpoints; ``retry_failed`` re-queues every failed
file; ``cleanup_stale_parts`` sweeps orphaned ``.part`` files left by an
interrupted run.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from models.detection_models import ApprovedConfig
from models.file_models import FileStatus, SourceFile


class RegistryStore:
    """Atomic read/write of the pipeline registry on disk."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = threading.RLock()

    def path(self) -> Path:
        return self._root / "registry.json"

    # ------------------------------------------------------------------
    def persist(
        self,
        files: dict[str, SourceFile],
        approved: dict[str, ApprovedConfig],
    ) -> None:
        """Atomically write the registry.

        The whole write+rename runs under a lock so concurrent transitions
        (download and parser threads) can never clobber each other's
        ``.part`` file mid-rename.
        """
        with self._lock:
            payload = {
                "version": 1,
                "files": {
                    file_id: source.to_json_dict()
                    for file_id, source in files.items()
                },
                "approved": {
                    file_id: config.to_json_dict()
                    for file_id, config in approved.items()
                },
            }
            part = self.path().with_suffix(".json.part")
            part.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(part, self.path())

    # ------------------------------------------------------------------
    def load(self) -> tuple[dict[str, SourceFile], dict[str, ApprovedConfig]]:
        """Rehydrate files + approved configs from disk.

        A corrupt registry is not fatal: the pipeline starts clean and
        re-discovers. Individual malformed entries are skipped.
        """
        if not self.path().exists():
            return {}, {}

        try:
            payload = json.loads(self.path().read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"[orchestrator] ignoring unreadable registry: {exc}")
            return {}, {}

        files: dict[str, SourceFile] = {}
        for file_id, data in payload.get("files", {}).items():
            try:
                files[file_id] = SourceFile.from_json_dict(data)
            except (ValueError, KeyError, TypeError):
                continue  # one malformed entry does not sink the registry

        approved: dict[str, ApprovedConfig] = {}
        for file_id, data in payload.get("approved", {}).items():
            try:
                approved[file_id] = ApprovedConfig.from_json_dict(data)
            except (ValueError, KeyError, TypeError):
                continue
        return files, approved


def recover_inflight(files: dict[str, SourceFile]) -> int:
    """Re-queue transient states at safe checkpoints after a crash.

    A process dying mid-stage leaves a file in a working state with no
    worker to finish it. Every such state is pulled back to the last
    stable checkpoint so the next ``process()`` call re-runs it:

      - DOWNLOADING -> DOWNLOAD_QUEUED  (mover skips an existing raw)
      - DETECTING, PARSING, PARQUET_WRITING, DATASET_VERIFYING
                      -> DOWNLOAD_VERIFIED (re-detect / re-parse)
      - RAW_CLEANUP stays put: ``process()`` finishes the deletion.

    Returns the number of files that needed recovery.
    """
    requeue: dict[FileStatus, FileStatus] = {
        FileStatus.DISCOVERED: FileStatus.DOWNLOAD_QUEUED,
        FileStatus.DOWNLOADING: FileStatus.DOWNLOAD_QUEUED,
        FileStatus.DETECTING: FileStatus.DOWNLOAD_VERIFIED,
        FileStatus.PARSING: FileStatus.DOWNLOAD_VERIFIED,
        FileStatus.PARQUET_WRITING: FileStatus.DOWNLOAD_VERIFIED,
        FileStatus.DATASET_VERIFYING: FileStatus.DOWNLOAD_VERIFIED,
    }
    recovered = 0
    for source in files.values():
        target = requeue.get(source.status)
        if target is None:
            continue
        # Recovery is a special case and bypasses the transition table
        # (those edges are not legal moves, by design).
        source.status = target
        source.error = None
        recovered += 1
    return recovered


def retry_failed(files: dict[str, SourceFile]) -> int:
    """Re-queue every failed file at its earliest retryable stage."""
    failed = [source for source in files.values() if source.is_failure()]
    for source in failed:
        source.mark_ready_to_retry()
    return len(failed)


def cleanup_stale_parts(root: str | Path) -> int:
    """Sweep abandoned ``.part`` files left by a crash.

    Returns how many files were removed. ``.part`` is the in-progress
    suffix for both raw downloads (``raw/*.part``) and Parquet writes
    (``datasets/*/*.part``); neither is ever consumed directly.
    """
    root = Path(root)
    removed = 0
    for directory in (root / "raw", root / "datasets"):
        if not directory.exists():
            continue
        for part in directory.glob("*.part"):
            try:
                part.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
        for dataset_dir in directory.iterdir():
            if not dataset_dir.is_dir():
                continue
            for part in dataset_dir.glob("*.part"):
                try:
                    part.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    continue
    return removed
