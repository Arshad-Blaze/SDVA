"""Per-file lifecycle states and the source-file model.

Canonical lifecycle (DVA requirement package, docx section 5):

    DISCOVERED
      -> DOWNLOAD_QUEUED -> DOWNLOADING -> DOWNLOAD_VERIFIED
      -> DETECTING -> AWAITING_APPROVAL -> PARSING -> PARQUET_WRITING
      -> DATASET_VERIFYING -> COMPLETE -> RAW_CLEANUP -> RAW_DELETED

Failure states terminate the lifecycle and never progress further:

    DOWNLOAD_FAILED, DETECTION_FAILED, PARSE_FAILED, WRITE_FAILED,
    VERIFICATION_FAILED, CLEANUP_FAILED

Recovery (Phase 1, restart-based) allows a *failed* file to be retried by
re-queuing it; a file marked RAW_DELETED is permanently finished.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class FileStatus(str, Enum):
    """All reachable states of a single source file's lifecycle."""

    # ---- happy path -------------------------------------------------
    DISCOVERED = "DISCOVERED"
    DOWNLOAD_QUEUED = "DOWNLOAD_QUEUED"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOAD_VERIFIED = "DOWNLOAD_VERIFIED"
    DETECTING = "DETECTING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PARSING = "PARSING"
    PARQUET_WRITING = "PARQUET_WRITING"
    DATASET_VERIFYING = "DATASET_VERIFYING"
    COMPLETE = "COMPLETE"
    RAW_CLEANUP = "RAW_CLEANUP"
    RAW_DELETED = "RAW_DELETED"

    # ---- failure ----------------------------------------------------
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    DETECTION_FAILED = "DETECTION_FAILED"
    PARSE_FAILED = "PARSE_FAILED"
    WRITE_FAILED = "WRITE_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    CLEANUP_FAILED = "CLEANUP_FAILED"

    # ----------------------------------------------------------------
    # Failure states are always terminal from the pipeline's point of
    # view; explicit retry re-queues the file afterwards.
    FAILURE_STATES: frozenset["FileStatus"] = frozenset(
        {
            DOWNLOAD_FAILED,
            DETECTION_FAILED,
            PARSE_FAILED,
            WRITE_FAILED,
            VERIFICATION_FAILED,
            CLEANUP_FAILED,
        }
    )

    # States that permanently close the lifecycle.
    TERMINAL_STATES: frozenset["FileStatus"] = FAILURE_STATES | {RAW_DELETED}

    # ------------------------------------------------------------------
    def is_failure(self) -> bool:
        """True when the file ended in a failure state."""
        return self in FileStatus.FAILURE_STATES

    def is_terminal(self) -> bool:
        """True when no further transition is possible."""
        return self in FileStatus.TERMINAL_STATES


# ---------------------------------------------------------------------
# Legal transitions. A transition not listed here raises an error, which
# keeps the orchestrator honest about the lifecycle ordering.
# ---------------------------------------------------------------------
_TRANSITIONS: dict[FileStatus, frozenset[FileStatus]] = {
    FileStatus.DISCOVERED: frozenset({FileStatus.DOWNLOAD_QUEUED}),
    FileStatus.DOWNLOAD_QUEUED: frozenset({FileStatus.DOWNLOADING}),
    FileStatus.DOWNLOADING: frozenset(
        {FileStatus.DOWNLOAD_VERIFIED, FileStatus.DOWNLOAD_FAILED}
    ),
    FileStatus.DOWNLOAD_VERIFIED: frozenset({FileStatus.DETECTING}),
    FileStatus.DETECTING: frozenset(
        {
            FileStatus.AWAITING_APPROVAL,
            FileStatus.PARSING,
            FileStatus.DETECTION_FAILED,
        }
    ),
    # Approval can be accepted, modified, or reprocessed (re-detect).
    FileStatus.AWAITING_APPROVAL: frozenset(
        {FileStatus.PARSING, FileStatus.DETECTING, FileStatus.DETECTION_FAILED}
    ),
    FileStatus.PARSING: frozenset(
        {FileStatus.PARQUET_WRITING, FileStatus.PARSE_FAILED}
    ),
    FileStatus.PARQUET_WRITING: frozenset(
        {FileStatus.DATASET_VERIFYING, FileStatus.WRITE_FAILED}
    ),
    FileStatus.DATASET_VERIFYING: frozenset(
        {FileStatus.COMPLETE, FileStatus.VERIFICATION_FAILED}
    ),
    FileStatus.COMPLETE: frozenset({FileStatus.RAW_CLEANUP}),
    FileStatus.RAW_CLEANUP: frozenset(
        {FileStatus.RAW_DELETED, FileStatus.CLEANUP_FAILED}
    ),
    # Failure states only move forward via explicit retry.
    FileStatus.DOWNLOAD_FAILED: frozenset({FileStatus.DOWNLOAD_QUEUED}),
    FileStatus.PARSE_FAILED: frozenset({FileStatus.PARSING}),
    FileStatus.DETECTION_FAILED: frozenset({FileStatus.DETECTING}),
    FileStatus.WRITE_FAILED: frozenset({FileStatus.PARQUET_WRITING}),
    FileStatus.VERIFICATION_FAILED: frozenset({FileStatus.DATASET_VERIFYING}),
    FileStatus.CLEANUP_FAILED: frozenset({FileStatus.RAW_CLEANUP}),
}

# Terminal states have no outgoing edges.
for _state in FileStatus.TERMINAL_STATES:
    _TRANSITIONS.setdefault(_state, frozenset())


@dataclass(slots=True)
class SourceFile:
    """Immutable metadata for one MFT source file plus its mutable lifecycle."""

    name: str
    remote_path: str
    expected_size: int | None = None
    checksum: str | None = None
    status: FileStatus = FileStatus.DISCOVERED
    file_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    local_path: Path | None = None
    dataset_dir: Path | None = None
    error: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ------------------------------------------------------------------
    def record_status(
        self, new_status: FileStatus, error: str | None = None
    ) -> None:
        """Transition this file to ``new_status``, enforcing legal moves.

        Raises:
            ValueError: when the transition is not allowed or the file is
                already irreversibly finished (RAW_DELETED).
        """
        if self.status is FileStatus.RAW_DELETED:
            raise ValueError(
                f"Cannot transition {self.name}: already terminal "
                f"({self.status.value})."
            )
        if new_status not in _TRANSITIONS[self.status]:
            raise ValueError(
                f"Invalid transition {self.status.value} -> "
                f"{new_status.value} for file {self.name}."
            )

        self.status = new_status
        self.error = error
        self.updated_at = datetime.now(timezone.utc)

    def mark_ready_to_retry(self) -> None:
        """Re-queue a failed file so the orchestrator can process it again."""
        if not self.status.is_failure():
            raise ValueError(
                f"Only failed files can be retried, got {self.status.value}."
            )

        # Re-queue the file at the earliest sensible pipeline step.
        retry_from = {
            FileStatus.DOWNLOAD_FAILED: FileStatus.DOWNLOAD_QUEUED,
            FileStatus.DETECTION_FAILED: FileStatus.DETECTING,
            FileStatus.PARSE_FAILED: FileStatus.PARSING,
            FileStatus.WRITE_FAILED: FileStatus.PARQUET_WRITING,
            FileStatus.VERIFICATION_FAILED: FileStatus.DATASET_VERIFYING,
            FileStatus.CLEANUP_FAILED: FileStatus.RAW_CLEANUP,
        }
        self.record_status(retry_from[self.status])
        self.error = None

    def is_failure(self) -> bool:
        """True when this file ended in a failure state."""
        return self.status.is_failure()

    def is_terminal(self) -> bool:
        """True when no further lifecycle transition is possible."""
        return self.status.is_terminal()

    def is_complete(self) -> bool:
        """True once the dataset is published (raw may still be present)."""
        return self.status is FileStatus.COMPLETE

    def is_done(self) -> bool:
        """True once the whole lifecycle, cleanup included, is finished."""
        return self.status is FileStatus.RAW_DELETED

    # ------------------------------------------------------------------
    def to_json_dict(self) -> dict:
        """Plain-dict form for registry persistence."""
        return {
            "name": self.name,
            "remote_path": self.remote_path,
            "expected_size": self.expected_size,
            "checksum": self.checksum,
            "status": self.status.value,
            "file_id": self.file_id,
            "local_path": str(self.local_path) if self.local_path else None,
            "dataset_dir": str(self.dataset_dir) if self.dataset_dir else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "SourceFile":
        """Rehydrate a SourceFile from ``to_json_dict`` output."""
        return cls(
            name=payload["name"],
            remote_path=payload["remote_path"],
            expected_size=payload.get("expected_size"),
            checksum=payload.get("checksum"),
            status=FileStatus(payload["status"]),
            file_id=payload["file_id"],
            local_path=(
                Path(payload["local_path"]) if payload.get("local_path") else None
            ),
            dataset_dir=(
                Path(payload["dataset_dir"]) if payload.get("dataset_dir") else None
            ),
            error=payload.get("error"),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
        )
