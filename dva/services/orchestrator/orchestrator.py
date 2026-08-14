"""Pipeline orchestrator (Boundary 2/6): wires the stages together.

Topology: 2 download workers and 1 parser worker (requirement doc 02).
Files progress per file, not per batch: as soon as a download finishes it
is fed to the parser queue instead of waiting for the whole batch.

Per-file flow handled here:

    discover -> DOWNLOAD_QUEUED -> [download pool] DOWNLOAD_VERIFIED
      -> DETECTING -> (needs review? AWAITING_APPROVAL) | (PARSING
      -> PARQUET_WRITING -> DATASET_VERIFYING -> COMPLETE -> RAW_CLEANUP
      -> RAW_DELETED)

Every transition goes through ``SourceFile.record_status`` so illegal
moves are impossible (models/file_models.py). Approval decisions are
applied later by the operator (approve/reprocess), which the UI exposes.

Persistence (Phase 9): the whole registry (files + approved configs) is
written to ``<workspace>/registry.json`` on every transition, so a crash
mid-flight can be recovered on restart. ``recover_inflight()`` re-queues
transient states at a safe checkpoint; ``retry_failed()`` re-queues every
failed file; ``cleanup_stale_parts()`` sweeps abandoned ``.part`` files.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from models.dataset_models import (
    DatasetMetadata,
    dataset_directory,
)
from models.detection_models import (
    ApprovedConfig,
    DetectionResult,
    FileFormat,
    StructureType,
)
from models.file_models import FileStatus, SourceFile
from services.ingestion.detector import Detector
from services.mft.file_mover import FileMover
from services.mft.transport import Transport
from services.orchestrator.registry import (
    RegistryStore,
    cleanup_stale_parts,
    recover_inflight,
    retry_failed,
)
from services.parser.parser import ParseReport, parse_file
from services.writer.parquet_writer import verify_parquet, write_parquet


@dataclass(frozen=True, slots=True)
class Workspace:
    """Directory layout where the pipeline stages its work."""
    root: Path
    raw_dir: Path
    datasets_dir: Path

    @classmethod
    def at(cls, root: str | Path) -> "Workspace":
        root = Path(root)
        return cls(root=root, raw_dir=root / "raw", datasets_dir=root / "datasets")

    def ensure(self) -> "Workspace":
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Static configuration for one pipeline run."""
    source_path: str                       # remote directory to scan
    pattern: str | None = None             # glob filter
    transport_factory: Callable[[], Transport] | None = None
    fixed_width_layout: str | Path | None = None
    download_workers: int = 2
    parser_workers: int = 1


@dataclass(slots=True)
class RunSummary:
    """Current work done by the last process() call."""
    discovered: int = 0
    downloaded: int = 0
    parsed: int = 0
    pending_approval: int = 0
    failed: int = 0


class Pipeline:
    """Coordinates mover -> detector -> parser -> writer for all files.

    All per-file state lives on the SourceFile objects in ``files`` keyed
    by file_id; dataset metadata persists to each dataset directory. The
    registry itself is kept in memory (persistence is a Phase-9 concern).
    """

    def __init__(
        self,
        workspace: Workspace,
        settings: PipelineSettings,
        *,
        rescue_complete_datasets: bool = True,
    ) -> None:
        self.workspace = workspace.ensure()
        self.settings = settings
        if settings.transport_factory is None:
            raise ValueError("PipelineSettings requires a transport_factory.")

        self.files: dict[str, SourceFile] = {}
        self.approved: dict[str, ApprovedConfig] = {}
        self.datasets: dict[str, DatasetMetadata] = {}
        self._detector = Detector(settings.fixed_width_layout)
        self._lock = threading.RLock()
        self._registry = RegistryStore(self.workspace.root)

        self._recover_registry()
        self.recovered_on_start = self.recover_inflight()
        if rescue_complete_datasets:
            self._recover_datasets()

    # ==================================================================
    # Registry persistence (Phase 9): every transition writes the whole
    # registry to registry.json so a restart can resume where we left off.
    # ==================================================================
    def _persist_registry(self) -> None:
        self._registry.persist(self.files, self.approved)

    def _recover_registry(self) -> None:
        self.files, self.approved = self._registry.load()

    def recover_inflight(self) -> int:
        """Re-queue transient states at safe checkpoints after a crash."""
        with self._lock:
            recovered = recover_inflight(self.files)
        if recovered:
            self._persist_registry()
        return recovered

    def retry_failed(self) -> int:
        """Re-queue every failed file at its earliest retryable stage."""
        with self._lock:
            requeued = retry_failed(self.files)
        if requeued:
            self._persist_registry()
        return requeued

    def retry_file(self, file_id: str) -> SourceFile:
        """Re-queue a single failed file and persist the registry."""
        source = self.get_file(file_id)
        if not source.is_failure():
            raise ValueError(
                f"Only failed files can be retried, got "
                f"{source.status.value} for {source.name}."
            )
        with self._lock:
            source.mark_ready_to_retry()
        self._persist_registry()
        return source

    def cleanup_stale_parts(self) -> int:
        """Sweep abandoned ``.part`` files left by a crash."""
        return cleanup_stale_parts(self.workspace.root)

    # ==================================================================
    # Lifecycle control (thread-safe registry).
    # ==================================================================
    @staticmethod
    def _new_mover(settings: PipelineSettings, workspace: Workspace) -> FileMover:
        return FileMover(settings.transport_factory(), workspace.raw_dir)

    def discover(self) -> list[SourceFile]:
        """Scan the MFT and queue every new file for download.

        Files already known in this session (by MFT name) are skipped so
        a finished file is never re-queued; a raw file still present is
        recovered as "already present" by the mover instead.
        """
        mover = self._new_mover(self.settings, self.workspace)
        entries = mover.discover_files(
            self.settings.source_path, self.settings.pattern
        )
        with self._lock:
            known = {file.name for file in self.files.values()}
            fresh = [file for file in entries if file.name not in known]
        for source in fresh:
            self._register(source)
        return fresh

    def _register(self, source: SourceFile) -> None:
        with self._lock:
            source.record_status(FileStatus.DOWNLOAD_QUEUED)
            self.files[source.file_id] = source
        self._persist_registry()

    def get_file(self, file_id: str) -> SourceFile:
        with self._lock:
            return self.files[file_id]

    def save_metadata(self, dataset: DatasetMetadata) -> None:
        """Persist a dataset's metadata.json alongside its parquet files."""
        dataset_dir = dataset_directory(self.workspace.root, dataset.dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "metadata.json").write_text(
            json.dumps(dataset.to_json_dict(), indent=2), encoding="utf-8"
        )

    def _recover_datasets(self) -> None:
        """Re-import persisted dataset metadata (restart recovery)."""
        for metadata_path in self.workspace.datasets_dir.glob("*/metadata.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                meta = DatasetMetadata.from_json_dict(payload)
                self.datasets[meta.dataset_id] = meta
            except (ValueError, OSError, KeyError):
                continue  # malformed metadata is not fatal to startup

    # ==================================================================
    # The two-stage worker pool.
    # ==================================================================
    def process(self) -> RunSummary:
        """Advance every eligible file: downloads, then parses, then finishes
        recovered raw cleanups (Phase 9)."""
        summary = RunSummary()
        summary.discovered = self.discover_count()

        with self._lock:
            to_download = [
                file for file in self.files.values()
                if file.status is FileStatus.DOWNLOAD_QUEUED
            ]
        self._download_pool(to_download, summary)

        with self._lock:
            to_parse = [
                file for file in self.files.values()
                if file.status is FileStatus.DOWNLOAD_VERIFIED
            ]
        self._parse_pool(to_parse, summary)

        with self._lock:
            to_cleanup = [
                file for file in self.files.values()
                if file.status is FileStatus.RAW_CLEANUP
            ]
        self._cleanup_pool(to_cleanup)

        with self._lock:
            summary.pending_approval = sum(
                file.status is FileStatus.AWAITING_APPROVAL
                for file in self.files.values()
            )
            summary.failed = sum(file.is_failure() for file in self.files.values())
        return summary

    def _cleanup_pool(self, files: list[SourceFile]) -> None:
        """Finish raw deletion for files recovered mid-cleanup (single thread)."""
        for source in files:
            self._cleanup_raw(source)

    def _download_pool(self, files: list[SourceFile], summary: RunSummary) -> None:
        if not files:
            return
        with ThreadPoolExecutor(max_workers=self.settings.download_workers) as pool:
            futures = [pool.submit(self._download_one, file, summary) for file in files]
            for future in futures:
                future.result()

    def _parse_pool(self, files: list[SourceFile], summary: RunSummary) -> None:
        if not files:
            return
        with ThreadPoolExecutor(max_workers=self.settings.parser_workers) as pool:
            futures = [pool.submit(self._parse_one, file, summary) for file in files]
            for future in futures:
                future.result()

    # ==================================================================
    # Stage implementations.
    # ==================================================================
    def _download_one(self, source: SourceFile, summary: RunSummary) -> None:
        self._transition(source, FileStatus.DOWNLOADING)
        mover = self._new_mover(self.settings, self.workspace)
        result = mover.download_file(source)
        if not result.success:
            self._transition(source, FileStatus.DOWNLOAD_FAILED, error=result.message)
            return
        source.local_path = result.local_path
        self._transition(source, FileStatus.DOWNLOAD_VERIFIED)
        summary.downloaded += 1

    def _parse_one(self, source: SourceFile, summary: RunSummary) -> None:
        self._transition(source, FileStatus.DETECTING)
        try:
            result = self._detector.detect(source.local_path)
        except Exception as exc:  # unreadable file -> permanent detection failure
            self._transition(source, FileStatus.DETECTION_FAILED, error=str(exc))
            return

        config = self._auto_config(source, result)
        if config is None:
            self._transition(source, FileStatus.AWAITING_APPROVAL)
            return

        self.approved[source.file_id] = config
        self._transition(source, FileStatus.PARSING)
        self._parse_and_write(source, config)
        summary.parsed += 1

    def _parse_and_write(
        self, source: SourceFile, config: ApprovedConfig
    ) -> None:
        try:
            frame, report = parse_file(source.local_path, config)
        except Exception as exc:
            self._transition(source, FileStatus.PARSE_FAILED, error=str(exc))
            return

        self._transition(source, FileStatus.PARQUET_WRITING)
        dataset = self._new_dataset(source, config)
        try:
            parquet_path = self._dataset_parquet(dataset, config)
            write_result = write_parquet(
                frame,
                parquet_path,
                source_path=str(source.local_path),
                schema_overrides=config.schema_overrides,
            )
        except Exception as exc:
            self._transition(source, FileStatus.WRITE_FAILED, error=str(exc))
            return

        self._transition(source, FileStatus.DATASET_VERIFYING)
        try:
            verify = verify_parquet(
                parquet_path,
                expected_rows=frame.height,
                expected_columns=list(frame.columns),
            )
            if not (verify["rows_matched"] and verify["columns_matched"]):
                raise RuntimeError("written parquet does not match the parsed frame")
        except Exception as exc:
            self._transition(source, FileStatus.VERIFICATION_FAILED, error=str(exc))
            return

        dataset.row_count = verify["rows"]
        dataset.schema = {
            name: str(dtype) for name, dtype in frame.schema.items()
        }
        dataset.extra["parse_report"] = report.to_json_dict()
        dataset.mark_verifying()
        dataset.mark_complete()
        self.datasets[dataset.dataset_id] = dataset
        self.save_metadata(dataset)

        self._transition(source, FileStatus.COMPLETE)
        self._cleanup_raw(source)

    def _cleanup_raw(self, source: SourceFile) -> None:
        """Delete the raw file after verification; recovered files are already
        in RAW_CLEANUP, so only transition when coming from a completed write."""
        if source.status is not FileStatus.RAW_CLEANUP:
            self._transition(source, FileStatus.RAW_CLEANUP)
        mover = self._new_mover(self.settings, self.workspace)
        if mover.cleanup_raw_file(source):
            self._transition(source, FileStatus.RAW_DELETED)
        else:
            self._transition(
                source, FileStatus.CLEANUP_FAILED, error="raw file not removable"
            )

    # ==================================================================
    # Detection -> approval / configuration.
    # ==================================================================
    @staticmethod
    def _auto_config(
        source: SourceFile, result: DetectionResult
    ) -> ApprovedConfig | None:
        """Build a parse config when detection is confident+runnable.

        None means the file must be reviewed (low confidence, unknown
        header, missing layout, or a deferred structure type).
        """
        if (
            not result.is_parseable()
            or result.needs_approval()
            or result.confidence <= 0.0
        ):
            return None

        if result.format is FileFormat.DELIMITED:
            if result.header_present is None:
                return None  # header is unknown -> ask the operator
            if not result.delimiter or not result.layout:
                return None
            return ApprovedConfig(
                source_format=FileFormat.DELIMITED,
                delimiter=result.delimiter,
                encoding=result.encoding,
                header_present=result.header_present,
                columns=list(result.layout),
                skip_rows=result.preamble_lines,
            )

        if result.format is FileFormat.FIXED_WIDTH:
            if result.layout is None or result.structure_type is not StructureType.FLAT:
                return None
            return ApprovedConfig(
                source_format=FileFormat.FIXED_WIDTH,
                encoding=result.encoding,
                header_present=False,
                columns=result.layout.column_names(),
                fixed_width_layout=result.layout,
            )

        return None  # UNSUPPORTED and friends

    def _new_dataset(self, source: SourceFile, config: ApprovedConfig) -> DatasetMetadata:
        dataset = DatasetMetadata(
            source_file_id=source.file_id,
            source_file_name=source.name,
        )
        dataset.mark_writing()
        self.datasets[dataset.dataset_id] = dataset
        return dataset

    def _dataset_parquet(
        self, dataset: DatasetMetadata, config: ApprovedConfig
    ) -> Path:
        dataset_dir = dataset_directory(self.workspace.root, dataset.dataset_id)
        path = dataset_dir / f"{dataset.source_file_name}.parquet"
        dataset.parquet_files = [path.name]
        return path

    # ==================================================================
    # Approval published by the operator.
    # ==================================================================
    def approve(self, file_id: str, config: ApprovedConfig) -> None:
        """Accept an operator-supplied config and resume parsing."""
        config.validate()
        source = self.get_file(file_id)
        if source.status is not FileStatus.AWAITING_APPROVAL:
            raise ValueError(
                f"Cannot approve {source.name}: status is "
                f"{source.status.value}, not AWAITING_APPROVAL."
            )
        self.approved[file_id] = config
        self._transition(source, FileStatus.PARSING)
        self._parse_to_completion(source)

    def reprocess(self, file_id: str) -> None:
        """Discard any config and re-detect the file."""
        source = self.get_file(file_id)
        if source.status is not FileStatus.AWAITING_APPROVAL:
            raise ValueError(
                f"Cannot reprocess {source.name}: status is "
                f"{source.status.value}, not AWAITING_APPROVAL."
            )
        self.approved.pop(file_id, None)
        self._parse_one(source, RunSummary())

    def _parse_to_completion(self, source: SourceFile) -> None:
        """Run the parse/write/verify/cleanup chain synchronously."""
        config = self.approved.get(source.file_id)
        if config is None:
            self._transition(source, FileStatus.PARSE_FAILED, error="no approved config")
            return
        self._parse_and_write(source, config)

    # ==================================================================
    # Small helpers. -----------------------------------------------------
    # ==================================================================
    def _transition(
        self, source: SourceFile, status: FileStatus, error: str | None = None
    ) -> None:
        with self._lock:
            source.record_status(status, error=error)
        # The registry is the cheap, crash-recovery contract for the UI.
        self._persist_registry()

    def discover_count(self) -> int:
        with self._lock:
            return sum(
                file.status is FileStatus.DOWNLOAD_QUEUED
                for file in self.files.values()
            )

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for file in self.files.values():
                counts[file.status.value] = counts.get(file.status.value, 0) + 1
            return counts