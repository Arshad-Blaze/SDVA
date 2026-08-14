import polars as pl
import pytest

from services.writer.parquet_writer import (
    ParquetWriteError,
    apply_schema_overrides,
    verify_parquet,
    write_parquet,
)


def sample_frame():
    return pl.DataFrame(
        {"STORE": ["1001", "1002"], "UNITS": ["45", "60"], "PRICE": ["500", "700"]}
    )


def test_write_parquet_atomic_and_reports(tmp_path):
    target = tmp_path / "out" / "data.parquet"
    result = write_parquet(sample_frame(), target, source_path="raw/sales.csv")
    assert target.exists()
    assert not target.with_name(target.name + ".part").exists()
    assert result.rows == 2
    assert result.columns == ["STORE", "UNITS", "PRICE"]
    assert result.bytes_written > 0
    assert len(result.sha256) == 64  # sha256 hex


def test_write_overwrites_cleanly(tmp_path):
    target = tmp_path / "data.parquet"
    write_parquet(sample_frame(), target)
    result = write_parquet(sample_frame(), target)
    assert result.target_path == str(target)


def test_verify_parquet_matches(tmp_path):
    target = tmp_path / "data.parquet"
    write_parquet(sample_frame(), target)
    report = verify_parquet(
        target, expected_rows=2, expected_columns=["STORE", "UNITS", "PRICE"]
    )
    assert report["rows"] == 2
    assert report["rows_matched"] and report["columns_matched"]


def test_verify_parquet_detects_mismatch(tmp_path):
    target = tmp_path / "data.parquet"
    write_parquet(sample_frame(), target)
    report = verify_parquet(target, expected_rows=99)
    assert not report["rows_matched"]


def test_write_parquet_missing_parent_is_created(tmp_path):
    target = tmp_path / "a" / "b" / "c" / "d.parquet"
    write_parquet(sample_frame(), target)
    assert target.exists()


def test_verify_missing_file_raises(tmp_path):
    with pytest.raises(ParquetWriteError, match="missing"):
        verify_parquet(tmp_path / "nope.parquet")


def test_apply_schema_overrides_casts(tmp_path):
    frame = sample_frame()
    result = apply_schema_overrides(
        frame, {"STORE": "Int64", "UNITS": "Int64", "PRICE": "Float64"}
    )
    assert result.schema["STORE"] == pl.Int64
    assert result.schema["PRICE"] == pl.Float64


def test_apply_schema_overrides_rejects_bad_values(tmp_path):
    frame = sample_frame()
    with pytest.raises(ParquetWriteError, match="unknown columns"):
        apply_schema_overrides(frame, {"NOPE": "Int64"})
    with pytest.raises(ParquetWriteError, match="Unsupported override"):
        apply_schema_overrides(frame, {"STORE": "List"})


def test_overrides_persist_through_write(tmp_path):
    target = tmp_path / "data.parquet"
    write_parquet(
        sample_frame(),
        target,
        schema_overrides={"STORE": "Int64", "UNITS": "Int64", "PRICE": "Float64"},
    )
    schema = pl.read_parquet_schema(target)
    assert schema["STORE"] == pl.Int64
    assert schema["PRICE"] == pl.Float64


def test_round_trip_row_count_preserved(tmp_path):
    frame = pl.DataFrame({"A": range(1000), "B": ["x"] * 1000})
    target = tmp_path / "big.parquet"
    result = write_parquet(frame, target)
    report = verify_parquet(target, expected_rows=1000, expected_columns=["A", "B"])
    assert report["rows"] == result.rows == 1000
    assert report["rows_matched"]


# ---------------------------------------------------------------------
# Schema override semantics: "trust but verify" casts with nulls.
# ---------------------------------------------------------------------
def test_schema_override_uncastable_values_become_null(tmp_path):
    frame = pl.DataFrame({"price": ["12", "3.5", "not-a-number", "100"]})
    result = apply_schema_overrides(frame, {"price": "Float64"})
    values = result["price"].to_list()
    assert values[0] == 12.0
    assert values[1] == 3.5
    assert values[2] is None  # uncastable -> null, not an exception
    assert values[3] == 100.0


def test_schema_override_int_uncastable_becomes_null():
    frame = pl.DataFrame({"n": ["1", "2", "x"]})
    result = apply_schema_overrides(frame, {"n": "Int64"})
    assert result["n"].to_list()[:2] == [1, 2]
    assert result["n"].null_count() == 1


def test_schema_override_boolean_from_token_strings():
    frame = pl.DataFrame({"f": ["true", "false", "0", "1", "x", None]})
    result = apply_schema_overrides(frame, {"f": "Boolean"})
    assert result["f"].to_list() == [True, False, False, True, None, None]


def test_schema_override_unknown_column_raises(tmp_path):
    with pytest.raises(ParquetWriteError, match="unknown columns"):
        apply_schema_overrides(sample_frame(), {"NOPE": "Int64"})


def test_empty_typed_frame_round_trip(tmp_path):
    target = tmp_path / "empty.parquet"
    frame = pl.DataFrame(
        {
            "id": pl.Series("id", [], dtype=pl.Int64),
            "flag": pl.Series("flag", [], dtype=pl.Boolean),
        }
    )
    write_parquet(frame, target)
    report = verify_parquet(target, expected_rows=0, expected_columns=["id", "flag"])
    assert report["rows_matched"] and report["columns_matched"]


def test_all_null_column_round_trip(tmp_path):
    target = tmp_path / "allnull.parquet"
    frame = pl.DataFrame({"a": [None, None], "b": [1, 2]})
    write_parquet(frame, target)
    back = pl.read_parquet(target)
    assert back["a"].null_count() == 2
    assert back["b"].to_list() == [1, 2]