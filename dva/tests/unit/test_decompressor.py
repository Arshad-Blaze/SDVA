"""Unit tests for the Decompressor (Boundary 2)."""

import gzip
import zipfile

import pytest

from services.ingestion.decompressor import (
    DecompressionError,
    decompressed_name,
    decompress_to,
    is_compressed,
    prepare_input,
)


def write_bytes(tmp_path, name, payload: bytes):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_is_compressed_by_suffix(tmp_path):
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "a.csv.gz").write_text("x", encoding="utf-8")
    (tmp_path / "a.zip").write_text("x", encoding="utf-8")
    assert is_compressed(tmp_path / "a.csv.gz")
    assert is_compressed(tmp_path / "a.zip")
    assert is_compressed("a.GZ")  # case-insensitive
    assert not is_compressed(tmp_path / "a.csv")
    assert not is_compressed("a.dat")


def test_decompress_gzip_stream(tmp_path):
    payload = b"STORE,UNITS,PRICE\n1001,2,10\n" * 1000
    source = write_bytes(tmp_path, "sales.csv.gz", gzip.compress(payload))
    target = tmp_path / "sales.csv.gz.decompressed"
    out = decompress_to(source, target)
    assert out.read_bytes() == payload


def test_prepare_input_returns_none_for_plain(tmp_path):
    plain = write_bytes(tmp_path, "plain.csv", b"a,b\n1,2\n")
    assert prepare_input(plain, tmp_path) is None


def test_prepare_input_expands_into_dest_dir(tmp_path):
    source = write_bytes(tmp_path, "sales.csv.gz", gzip.compress(b"a,b\n1,2\n"))
    dest = tmp_path / "out"
    result = prepare_input(source, dest)
    assert result is not None
    assert result.raw_path == source
    assert result.decompressed_path == dest / "sales.csv.gz.decompressed"
    assert result.decompressed_path.exists()
    assert result.decompressed_path.read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_decompress_zip_first_member(tmp_path):
    payload = b"store,units\n1001,5\n"
    buf = tmp_path / "z.zip"
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("data.csv", payload)
        archive.writestr("extra.txt", b"ignored")
    out = decompress_to(buf, tmp_path / "z.zip.decompressed")
    assert out.read_bytes() == payload


def test_decompressed_name_is_unique(tmp_path):
    assert decompressed_name("sales.csv.gz") == "sales.csv.gz.decompressed"
    assert decompressed_name("a.b.zip") == "a.b.zip.decompressed"


def test_corrupt_gzip_raises(tmp_path):
    source = write_bytes(tmp_path, "bad.csv.gz", b"this is not gzip data")
    with pytest.raises(DecompressionError, match="cannot decompress"):
        decompress_to(source, tmp_path / "bad.csv.gz.decompressed")


def test_corrupt_zip_raises(tmp_path):
    source = write_bytes(tmp_path, "bad.zip", b"PK\x03\x04not really a zip")
    with pytest.raises(DecompressionError, match="cannot decompress"):
        decompress_to(source, tmp_path / "bad.zip.decompressed")


def test_empty_zip_raises(tmp_path):
    source = tmp_path / "empty.zip"
    with zipfile.ZipFile(source, "w"):
        pass  # archive with no members
    with pytest.raises(DecompressionError, match="contains no files"):
        decompress_to(source, tmp_path / "empty.zip.decompressed")