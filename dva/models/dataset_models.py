"""Dataset lifecycle and metadata models.

A *dataset* is the structured working artifact produced from one source
file: a directory of Parquet part-files plus ``metadata.json`` and an
atomic completion marker (03, section 8; 04, Boundary 6/7).

Only datasets in COMPLETE status may be consumed by the Validator.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class DatasetStatus(str, Enum):
    """Lifecycle of a Parquet dataset directory."""
    CREATED = "created"
    WRITING = "writing"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


# ---------------------------------------------------------------------
# Legal transitions for dataset status.
# ---------------------------------------------------------------------
_DATASET_TRANSITIONS: dict[DatasetStatus, frozenset[DatasetStatus]] = {
    DatasetStatus.CREATED: frozenset({DatasetStatus.WRITING}),
    DatasetStatus.WRITING: frozenset(
        {DatasetStatus.VERIFYING, DatasetStatus.FAILED}
    ),
    DatasetStatus.VERIFYING: frozenset(
        {DatasetStatus.COMPLETE, DatasetStatus.FAILED}
    ),
    DatasetStatus.COMPLETE: frozenset(),       # published, immutable
    DatasetStatus.FAILED: frozenset(),         # terminal, must be rebuilt
}


@dataclass(slots=True)
class DatasetMetadata:
    """Persisted metadata describing one Parquet dataset.

    Written by the Parquet Writer as ``metadata.json`` alongside the
    part-files (Boundary 6 responsibilities: schema, row groups, file
    rollover, metadata, completion state).
    """

    dataset_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source_file_id: str = ""
    source_file_name: str = ""
    status: DatasetStatus = DatasetStatus.CREATED
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None
    parquet_files: list[str] = field(default_factory=list)
    row_count: int | None = None
    # Column name -> dtype string recorded at the end of writing.
    schema: dict[str, str] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def record_status(self, new_status: DatasetStatus) -> None:
        """Transition the dataset to ``new_status``, enforcing legal moves."""
        if new_status not in _DATASET_TRANSITIONS[self.status]:
            raise ValueError(
                f"Invalid dataset transition {self.status.value} -> "
                f"{new_status.value}."
            )
        self.status = new_status
        if new_status is DatasetStatus.COMPLETE:
            self.completed_at = datetime.now(timezone.utc)

    def mark_writing(self) -> None:
        self.record_status(DatasetStatus.WRITING)

    def mark_verifying(self) -> None:
        self.record_status(DatasetStatus.VERIFYING)

    def mark_complete(self) -> None:
        self.record_status(DatasetStatus.COMPLETE)

    def mark_failed(self) -> None:
        self.record_status(DatasetStatus.FAILED)

    def is_complete(self) -> bool:
        """Only COMPLETE datasets are valid Validator inputs."""
        return self.status is DatasetStatus.COMPLETE

    # ------------------------------------------------------------------
    def to_json_dict(self) -> dict:
        """Serialisable form used by the Parquet Writer for metadata.json."""
        return {
            "dataset_id": self.dataset_id,
            "source_file_id": self.source_file_id,
            "source_file_name": self.source_file_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "parquet_files": self.parquet_files,
            "row_count": self.row_count,
            "schema": self.schema,
            "extra": self.extra,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "DatasetMetadata":
        """Rehydrate metadata from a metadata.json payload."""
        meta = cls(
            dataset_id=payload["dataset_id"],
            source_file_id=payload.get("source_file_id", ""),
            source_file_name=payload.get("source_file_name", ""),
            status=DatasetStatus(payload["status"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            completed_at=(
                datetime.fromisoformat(payload["completed_at"])
                if payload.get("completed_at")
                else None
            ),
            parquet_files=payload.get("parquet_files", []),
            row_count=payload.get("row_count"),
            schema=payload.get("schema", {}),
            extra=payload.get("extra", {}),
        )
        return meta


def dataset_directory(root: Path, dataset_id: str) -> Path:
    """Conventional location of a dataset directory under a workspace root."""
    return root / "datasets" / dataset_id
