"""Unit tests for the dataset lifecycle and metadata models."""

import pytest

from models.dataset_models import (
    DatasetMetadata,
    DatasetStatus,
    dataset_directory,
)
from pathlib import Path


def make_metadata() -> DatasetMetadata:
    return DatasetMetadata(source_file_id="abc", source_file_name="sales.txt")


def test_default_status_is_created():
    meta = make_metadata()
    assert meta.status is DatasetStatus.CREATED
    assert not meta.is_complete()


def test_complete_lifecycle():
    meta = make_metadata()
    meta.mark_writing()
    meta.mark_verifying()
    meta.mark_complete()
    assert meta.is_complete()
    assert meta.completed_at is not None


def test_invalid_transition_raises():
    meta = make_metadata()
    with pytest.raises(ValueError, match="Invalid dataset transition"):
        meta.mark_complete()  # CREATED -> COMPLETE is not allowed


def test_failure_from_writing():
    meta = make_metadata()
    meta.mark_writing()
    meta.mark_failed()
    assert meta.status is DatasetStatus.FAILED
    assert not meta.is_complete()


def test_json_round_trip():
    meta = make_metadata()
    meta.mark_writing()
    meta.mark_verifying()
    meta.mark_complete()
    meta.parquet_files = ["part-000.parquet"]
    meta.row_count = 42
    meta.schema = {"store": "Utf8"}

    payload = meta.to_json_dict()
    restored = DatasetMetadata.from_json_dict(payload)

    assert restored.dataset_id == meta.dataset_id
    assert restored.status is DatasetStatus.COMPLETE
    assert restored.row_count == 42
    assert restored.parquet_files == ["part-000.parquet"]
    assert restored.schema == {"store": "Utf8"}
    assert restored.completed_at is not None


def test_dataset_directory_convention():
    root = Path("/tmp/workspace")
    assert dataset_directory(root, "d1") == Path("/tmp/workspace/datasets/d1")
