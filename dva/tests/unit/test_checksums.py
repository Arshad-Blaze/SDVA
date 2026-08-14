"""Unit tests for checksum helpers."""

import hashlib

import pytest

from utils.checksums import SUPPORTED_ALGORITHMS, checksum_file, checksums_match


def test_md5_checksum_matches_hashlib(tmp_path):
    payload = b"STORE|UNITS|PRICE\n1|10|5.00\n"
    path = tmp_path / "data.bin"
    path.write_bytes(payload)

    expected = hashlib.md5(payload).hexdigest()
    assert checksum_file(path) == expected


def test_sha256_checksum(tmp_path):
    payload = b"hello world"
    path = tmp_path / "data.bin"
    path.write_bytes(payload)
    assert checksum_file(path, algorithm="sha256") == hashlib.sha256(payload).hexdigest()


def test_unsupported_algorithm_raises(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported checksum algorithm"):
        checksum_file(path, algorithm="crc32")


def test_checksums_match_is_case_insensitive():
    assert checksums_match("ABC123", "abc123")
    assert not checksums_match("abc", "def")


def test_supported_algorithms_are_exposed():
    assert {"md5", "sha1", "sha256"}.issubset(SUPPORTED_ALGORITHMS)
