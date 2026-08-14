"""Per-file pipeline stages (Boundary 2/6) for the orchestrator.

Kept in a separate module so no source file exceeds the 500-line project
limit. These methods are mixed into ``Pipeline``; they rely on the state
``Pipeline`` owns (``settings``, ``workspace``, ``approved``, ``datasets``,
``_decompressed``, ``_detector``) and the helpers it provides
(``_transition``, ``_new_mover``, ``_auto_config``, ``_new_dataset``,
``_dataset_parquet``, ``save_metadata``).

File flow driven here:

    DOWNLOAD_VERIFIED -> DETECTING -> (AWAITING_APPROVAL | PARSING
      -> PARQUET_WRITING -> DATASET_VERIFYING -> COMPLETE -> RAW_CLEANUP
      -> RAW_DELETED)

Compressed raws are expanded first (Boundary 2); the temporary copy is
tracked so cleanup removes it on completion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from models.detection_models import ApprovedConfig
from models.file_models import FileStatus, SourceFile
from services.ingestion.decompressor import prepare_input
from services.parser.parser import parse_file
from services.writer.parquet_writer import verify_parquet, write_parquet


@dataclass(slots=True)
class RunSummary:
    """Current work done by the last process() call."""
    discovered: int = 0
    downloaded: int = 0
    parsed: int = 0
    pending_approval: int = 0
    failed: int = 0


class PipelineStages:
    """Stage implementations mixed into :class:`Pipeline`."""

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
            parse_path = self._prepare_input(source)
        except Exception as exc:  # corrupt/unsupported archive -> decompression failure
            self._transition(source, FileStatus.DECOMPRESSION_FAILED, error=str(exc))
            return
        try:
            result = self._detector.detect(parse_path)
        except Exception as exc:  # unreadable file -> permanent detection failure
            self._transition(source, FileStatus.DETECTION_FAILED, error=str(exc))
            return

        config = self._auto_config(source, result)
        if config is None:
            self._transition(source, FileStatus.AWAITING_APPROVAL)
            return

        self.approved[source.file_id] = config
        self._transition(source, FileStatus.PARSING)
        self._parse_and_write(source, config, parse_path)
        summary.parsed += 1

    def _prepare_input(self, source: SourceFile) -> Path:
        """Expand compressed raws (Boundary 2); track the temp for cleanup."""
        decompressed = prepare_input(source.local_path, self.workspace.raw_dir)
        if decompressed is None:
            return source.local_path
        self._decompressed[source.file_id] = decompressed.decompressed_path
        return decompressed.decompressed_path

    def _parse_and_write(
        self, source: SourceFile, config: ApprovedConfig, parse_path: Path | None = None
    ) -> None:
        path = parse_path or source.local_path
        try:
            frame, report = parse_file(path, config)
        except Exception as exc:
            self._transition(source, FileStatus.PARSE_FAILED, error=str(exc))
            return

        self._transition(source, FileStatus.PARQUET_WRITING)
        dataset = self._new_dataset(source, config)
        try:
            parquet_path = self._dataset_parquet(dataset, config)
            write_parquet(
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
        in RAW_CLEANUP, so only transition when coming from a completed write.
        Any decompressed temp for this file is removed first."""
        if source.status is not FileStatus.RAW_CLEANUP:
            self._transition(source, FileStatus.RAW_CLEANUP)
        temp = self._decompressed.pop(source.file_id, None)
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                self._transition(
                    source, FileStatus.CLEANUP_FAILED,
                    error="decompressed temp not removable",
                )
                return
        mover = self._new_mover(self.settings, self.workspace)
        if mover.cleanup_raw_file(source):
            self._transition(source, FileStatus.RAW_DELETED)
        else:
            self._transition(
                source, FileStatus.CLEANUP_FAILED, error="raw file not removable"
            )