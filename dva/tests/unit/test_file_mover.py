"""Unit tests for the File Mover against a local/mock MFT source.

LocalTransport stands in for the real SFTP MFT (04, "Test with mock/local
source"), so no server or paramiko is required.
"""

import pytest

from models.file_models import FileStatus
from services.mft.file_mover import (
    ChecksumMismatch,
    DownloadIncomplete,
    FileMover,
)
from services.mft.transport import LocalTransport, RemoteFileInfo


@pytest.fixture()
def mover(tmp_path):
    """A FileMover whose 'MFT' is a local folder, staging into another."""
    source = tmp_path / "mft"
    raw = tmp_path / "raw"
    source.mkdir()
    raw.mkdir()

    (source / "sales.txt").write_text("STORE|UNITS|PRICE\n1|10|5.00\n", encoding="utf-8")
    (source / "other.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (source / "nested").mkdir()  # directory must be skipped

    transport = LocalTransport(source)
    return FileMover(transport=transport, raw_dir=raw), raw


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------
def test_discover_lists_only_files(mover):
    mover, _ = mover
    files = mover.discover_files("/")
    names = {f.name for f in files}
    assert names == {"sales.txt", "other.csv"}


def test_discover_filters_by_pattern(mover):
    mover, _ = mover
    files = mover.discover_files("/", pattern="*.csv")
    assert [f.name for f in files] == ["other.csv"]


def test_discover_builds_remote_paths(mover):
    mover, _ = mover
    files = mover.discover_files("/inbound")
    assert all(f.remote_path.startswith("/inbound/") for f in files)


def test_discover_empty_dir(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    mover = FileMover(LocalTransport(source), raw_dir=tmp_path / "raw")
    assert mover.discover_files("/") == []


# ---------------------------------------------------------------------
# Download + verification
# ---------------------------------------------------------------------
def test_download_leaves_ready_file_only(mover):
    mover, raw = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    result = mover.download_file(source)

    assert result.success
    assert result.local_path == raw / "sales.txt"
    assert result.local_path.exists()
    assert not (raw / "sales.txt.part").exists()  # no leftover .part


def test_download_verifies_size(mover):
    mover, raw = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    source.expected_size = 99999  # lie about the real size

    result = mover.download_file(source)

    assert not result.success
    assert "size mismatch" in result.message
    assert not (raw / "sales.txt").exists()       # never renamed to ready
    assert not (raw / "sales.txt.part").exists()  # part cleaned up


def test_download_verifies_checksum(mover):
    mover, raw = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    source.checksum = "00000000000000000000000000000000"  # wrong on purpose

    result = mover.download_file(source)

    assert not result.success
    assert "checksum mismatch" in result.message
    assert not (raw / "sales.txt").exists()


def test_download_accepts_matching_checksum(mover):
    from utils.checksums import checksum_file

    mover, raw = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    mft_file = mover._transport._root / "sales.txt"
    source.checksum = checksum_file(mft_file)  # correct on purpose

    result = mover.download_file(source)
    assert result.success
    assert result.local_path.exists()


def test_download_skips_when_ready_exists(mover):
    mover, raw = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]

    first = mover.download_file(source)
    assert first.success and not first.already_present

    second = mover.download_file(source)
    assert second.success
    assert second.already_present


def test_verify_download_raises_on_mismatch(mover):
    mover, raw = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    part = raw / "sales.txt.part"
    part.write_text("partial data", encoding="utf-8")
    source.expected_size = 12345

    with pytest.raises(DownloadIncomplete):
        mover.verify_download(part, source)


def test_get_metadata_updates_expected_size(mover):
    mover, _ = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    source.expected_size = None
    mover.get_metadata(source)
    assert source.expected_size == (mover._transport._root / "sales.txt").stat().st_size


# ---------------------------------------------------------------------
# Cleanup (Boundary 8)
# ---------------------------------------------------------------------
def test_cleanup_raw_file_removes_ready_and_part(mover):
    mover, raw = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    mover.download_file(source)
    (raw / "sales.txt.part").write_text("stale", encoding="utf-8")

    assert mover.cleanup_raw_file(source)
    assert not (raw / "sales.txt").exists()
    assert not (raw / "sales.txt.part").exists()


def test_cleanup_raw_file_is_safe_when_missing(mover):
    mover, _ = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    assert mover.cleanup_raw_file(source)  # nothing there -> True


# ---------------------------------------------------------------------
# Transport failure handling
# ---------------------------------------------------------------------
class _FailingTransport:
    """Minimal fake that simulates a network failure during download."""

    def list_files(self, remote_path):
        return [RemoteFileInfo(name="x.txt", size=3, is_dir=False)]

    def get_size(self, remote_path):
        return 3

    def get(self, remote_path, local_path):
        raise ConnectionError("boom")

    def close(self):
        pass


def test_download_surfaces_transport_failure(tmp_path):
    raw = tmp_path / "raw"
    mover = FileMover(_FailingTransport(), raw_dir=raw)
    source = mover.discover_files("/")[0]

    result = mover.download_file(source)

    assert not result.success
    assert "download failed" in result.message
    assert not (raw / "x.txt").exists()


def test_discovery_returns_models_in_discovered_state(mover):
    mover, _ = mover
    source = mover.discover_files("/", pattern="sales.txt")[0]
    assert source.status is FileStatus.DISCOVERED
    assert source.file_id
