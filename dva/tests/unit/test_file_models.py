"""Unit tests for the source-file lifecycle model."""

import pytest

from models.file_models import FileStatus, SourceFile

HAPPY_PATH = [
    FileStatus.DISCOVERED,
    FileStatus.DOWNLOAD_QUEUED,
    FileStatus.DOWNLOADING,
    FileStatus.DOWNLOAD_VERIFIED,
    FileStatus.DETECTING,
    FileStatus.PARSING,  # approval skipped (high-confidence detection)
    FileStatus.PARQUET_WRITING,
    FileStatus.DATASET_VERIFYING,
    FileStatus.COMPLETE,
    FileStatus.RAW_CLEANUP,
    FileStatus.RAW_DELETED,
]


def make_file() -> SourceFile:
    return SourceFile(
        name="sales.txt",
        remote_path="/data/inbound/sales.txt",
        expected_size=1024,
    )


def walk_to(file: SourceFile, target: FileStatus) -> None:
    """Advance a file along the happy path until it reaches ``target``."""
    for status in HAPPY_PATH:
        if file.status is target:
            return
        if status is file.status:
            continue  # already at this step, do not self-transition
        file.record_status(status)
    raise AssertionError(f"target {target} is not on the happy path")


# A failure state is reached by walking to its happy-path predecessor and
# then applying the failing transition.
FAILURE_PREDECESSOR = {
    FileStatus.DOWNLOAD_FAILED: FileStatus.DOWNLOADING,
    FileStatus.DECOMPRESSION_FAILED: FileStatus.PARSING,
    FileStatus.DETECTION_FAILED: FileStatus.DETECTING,
    FileStatus.PARSE_FAILED: FileStatus.PARSING,
    FileStatus.WRITE_FAILED: FileStatus.PARQUET_WRITING,
    FileStatus.VERIFICATION_FAILED: FileStatus.DATASET_VERIFYING,
    FileStatus.CLEANUP_FAILED: FileStatus.RAW_CLEANUP,
}


def walk_to_failure(file: SourceFile, failed_state: FileStatus) -> None:
    """Walk the happy path and then apply the failing transition."""
    walk_to(file, FAILURE_PREDECESSOR[failed_state])
    file.record_status(failed_state)


def test_default_status_is_discovered():
    f = make_file()
    assert f.status is FileStatus.DISCOVERED
    assert not f.is_terminal()


def test_happy_path_transitions():
    f = make_file()
    for status in HAPPY_PATH[1:]:
        f.record_status(status)
    assert f.is_done()
    assert f.is_complete() is False  # cleanup already done


def test_invalid_transition_raises():
    f = make_file()
    with pytest.raises(ValueError, match="Invalid transition"):
        # DOWNLOAD_VERIFIED directly from DISCOVERED is not allowed.
        f.record_status(FileStatus.DOWNLOAD_VERIFIED)


def test_terminal_state_cannot_transition():
    f = make_file()
    for status in HAPPY_PATH[1:]:
        f.record_status(status)
    with pytest.raises(ValueError, match="already terminal"):
        f.record_status(FileStatus.DISCOVERED)


def test_approval_branch_back_to_detecting():
    f = make_file()
    f.record_status(FileStatus.DOWNLOAD_QUEUED)
    f.record_status(FileStatus.DOWNLOADING)
    f.record_status(FileStatus.DOWNLOAD_VERIFIED)
    f.record_status(FileStatus.DETECTING)
    f.record_status(FileStatus.AWAITING_APPROVAL)
    # User chose REPROCESS -> re-detect.
    f.record_status(FileStatus.DETECTING)
    assert f.status is FileStatus.DETECTING


def test_failure_states_are_terminal_and_flagged():
    f = make_file()
    f.record_status(FileStatus.DOWNLOAD_QUEUED)
    f.record_status(FileStatus.DOWNLOADING)
    f.record_status(FileStatus.DOWNLOAD_FAILED, error="connection reset")
    assert f.status.is_failure()
    assert f.status.is_terminal()
    assert f.error == "connection reset"


@pytest.mark.parametrize(
    "failed_state, retry_state",
    [
        (FileStatus.DOWNLOAD_FAILED, FileStatus.DOWNLOAD_QUEUED),
        (FileStatus.DECOMPRESSION_FAILED, FileStatus.DOWNLOAD_VERIFIED),
        (FileStatus.DETECTION_FAILED, FileStatus.DOWNLOAD_VERIFIED),
        (FileStatus.PARSE_FAILED, FileStatus.DOWNLOAD_VERIFIED),
        (FileStatus.WRITE_FAILED, FileStatus.DOWNLOAD_VERIFIED),
        (FileStatus.VERIFICATION_FAILED, FileStatus.DOWNLOAD_VERIFIED),
        (FileStatus.CLEANUP_FAILED, FileStatus.RAW_CLEANUP),
    ],
)
def test_retry_requeues_failed_file(failed_state, retry_state):
    f = make_file()
    walk_to_failure(f, failed_state)  # file is now in the failure state
    assert f.status.is_failure()

    f.mark_ready_to_retry()
    assert f.status is retry_state
    assert f.error is None


def test_retry_rejects_non_failed_files():
    f = make_file()
    with pytest.raises(ValueError, match="Only failed files"):
        f.mark_ready_to_retry()


def test_error_recorded_on_transition():
    f = make_file()
    f.record_status(FileStatus.DOWNLOAD_QUEUED)
    f.record_status(FileStatus.DOWNLOADING)
    f.record_status(FileStatus.DOWNLOAD_VERIFIED)
    f.record_status(FileStatus.DETECTING)
    f.record_status(FileStatus.AWAITING_APPROVAL)
    f.record_status(FileStatus.PARSING)
    f.record_status(FileStatus.PARSE_FAILED, error="bad delimiter")
    assert f.status is FileStatus.PARSE_FAILED
    assert f.error == "bad delimiter"


def test_source_file_json_round_trip():
    from pathlib import Path

    f = make_file()
    f.record_status(FileStatus.DOWNLOAD_QUEUED)
    f.record_status(FileStatus.DOWNLOADING)
    f.record_status(FileStatus.DOWNLOAD_VERIFIED)
    f.local_path = Path("/raw/sales.txt")
    f.error = None

    restored = SourceFile.from_json_dict(f.to_json_dict())
    assert restored.name == "sales.txt"
    assert restored.remote_path == "/data/inbound/sales.txt"
    assert restored.expected_size == 1024
    assert restored.status is FileStatus.DOWNLOAD_VERIFIED
    assert restored.file_id == f.file_id
    assert restored.local_path == Path("/raw/sales.txt")
    assert restored.created_at == f.created_at


def test_source_file_json_round_trip_preserves_failure():
    f = make_file()
    f.record_status(FileStatus.DOWNLOAD_QUEUED)
    f.record_status(FileStatus.DOWNLOADING)
    f.record_status(FileStatus.DOWNLOAD_FAILED, error="connection reset")
    restored = SourceFile.from_json_dict(f.to_json_dict())
    assert restored.status is FileStatus.DOWNLOAD_FAILED
    assert restored.error == "connection reset"
