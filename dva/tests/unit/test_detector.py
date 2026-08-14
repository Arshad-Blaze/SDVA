"""Unit tests for the Detector against representative sample files."""

from pathlib import Path

import pytest

from models.detection_models import FileFormat, StructureType
from services.ingestion.detector import Detector, APPROVAL_THRESHOLD
from services.ingestion.fields import first_key, parse_fields
from services.ingestion.layout import (
    LayoutLoadError,
    load_layout_csv,
    validate_layout_against_sample,
)
from services.ingestion.sampler import detect_encoding, read_bounded_sample


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# Delimited files.
# ---------------------------------------------------------------------
def test_detects_pipe_delimited_with_header(tmp_path):
    path = write(
        tmp_path,
        "sales.txt",
        "STORE|UNITS|PRICE\n1001|45|500\n1002|60|700\n1003|12|180\n",
    )
    result = Detector().detect(path)

    assert result.format is FileFormat.DELIMITED
    assert result.structure_type is StructureType.FLAT
    assert result.delimiter == "|"
    assert result.header_present is True
    assert result.layout == ["STORE", "UNITS", "PRICE"]
    assert set(result.schema) == {"STORE", "UNITS", "PRICE"}
    assert result.is_parseable()
    assert result.confidence >= APPROVAL_THRESHOLD


def test_detects_comma_delimited_no_header(tmp_path):
    path = write(
        tmp_path,
        "data.csv",
        "1234,56,7890\n2345,67,8901\n3456,78,9012\n",
    )
    result = Detector().detect(path)

    assert result.format is FileFormat.DELIMITED
    assert result.delimiter == ","
    assert result.header_present is False
    assert result.layout == ["COL_1", "COL_2", "COL_3"]
    assert result.is_parseable()


def test_detects_tab_delimited(tmp_path):
    path = write(
        tmp_path,
        "data.tsv",
        "STORE\tUNITS\tPRICE\n1\t2\t3\n4\t5\t6\n",
    )
    result = Detector().detect(path)
    assert result.delimiter == "\t"
    assert result.header_present is True


# ---------------------------------------------------------------------
# Fixed-width files.
# ---------------------------------------------------------------------
@pytest.fixture()
def layout_csv(tmp_path):
    path = tmp_path / "layout.csv"
    path.write_text(
        "Field,From,Length\n"
        "STORE,1,4\n"
        "UNITS,5,4\n"
        "PRICE,9,6\n",
        encoding="utf-8",
    )
    return path


def test_detects_fixed_width_requires_layout(tmp_path):
    # Consistent line length, no delimiter.
    path = write(
        tmp_path,
        "fixed.txt",
        "10010045000500\n10020060000700\n10030012000180\n",
    )
    result = Detector().detect(path)

    assert result.format is FileFormat.FIXED_WIDTH
    assert result.structure_type is StructureType.FLAT
    assert result.delimiter is None
    assert result.layout is None  # user must supply it
    assert result.needs_approval()  # confidence penalized -> review
    assert any("layout" in warning for warning in result.warnings)


def test_fixed_width_with_layout(tmp_path, layout_csv):
    path = write(
        tmp_path,
        "fixed.txt",
        "10010045000500\n10020060000700\n10030012000180\n",
    )
    result = Detector().detect(path, layout_csv=layout_csv)

    assert result.format is FileFormat.FIXED_WIDTH
    assert result.layout is not None
    assert result.layout.column_names() == ["STORE", "UNITS", "PRICE"]
    assert set(result.schema) == {"STORE", "UNITS", "PRICE"}
    assert result.warnings == []


def test_layout_validation_warns_when_lines_too_short(tmp_path, layout_csv):
    path = write(
        tmp_path,
        "short.txt",
        "10010045\n10020060\n",  # only 8 chars, layout needs 14
    )
    result = Detector().detect(path, layout_csv=layout_csv)
    assert result.layout is not None
    assert any("width" in warning for warning in result.warnings)


def test_load_layout_csv_bad_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Field,From\nA,1\n", encoding="utf-8")
    with pytest.raises(LayoutLoadError, match="columns"):
        load_layout_csv(bad)


def test_load_layout_csv_bad_numbers(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Field,From,Length\nA,notanumber,4\n", encoding="utf-8")
    with pytest.raises(LayoutLoadError, match="Invalid"):
        load_layout_csv(bad)


# ---------------------------------------------------------------------
# Structures (Phase 1 scoping).
# ---------------------------------------------------------------------
def test_multiline_delimited_via_continuation_marker(tmp_path):
    path = write(
        tmp_path,
        "cont.txt",
        "STORE|UNITS|PRICE\n1001|45|500\\\n1002|60|700\n",
    )
    result = Detector().detect(path)
    assert result.structure_type is StructureType.MULTILINE
    assert result.is_parseable()  # delimited multiline is supported


def test_quoted_delimiter_is_flat_and_consistent(tmp_path):
    # A quoted field containing the delimiter is NOT a multiline record;
    # quote-aware counting keeps the delimiter signal consistent.
    path = write(
        tmp_path,
        "quoted.txt",
        'STORE,UNITS,PRICE\n1001,"45,000",500\n1002,60,700\n',
    )
    result = Detector().detect(path)
    assert result.structure_type is StructureType.FLAT
    assert result.format is FileFormat.DELIMITED
    assert result.delimiter == ","
    assert result.is_parseable()
    assert result.sample_metadata["delimiter_stats"]["consistency"] >= 0.9


def test_multiline_delimited_via_unclosed_quote(tmp_path):
    # A record whose quoted field continues onto the next physical line.
    path = write(
        tmp_path,
        "quoted_multiline.txt",
        '123,"multi\nline",abc\n456,def,ghi\n',
    )
    result = Detector().detect(path)
    assert result.structure_type is StructureType.MULTILINE
    assert result.is_parseable()
    assert result.confidence < APPROVAL_THRESHOLD  # multiline -> review


def test_record_typed_is_flagged_not_parsed(tmp_path):
    path = write(
        tmp_path,
        "header_detail.txt",
        "H|ORDER|TOTAL\nD|1001|45|500\nD|1002|60|700\n",
    )
    result = Detector().detect(path)
    assert result.structure_type is StructureType.RECORD_TYPED
    assert not result.is_parseable()
    assert result.confidence < APPROVAL_THRESHOLD


def test_multiline_fixed_width_is_flagged_not_parsed(tmp_path):
    path = write(
        tmp_path,
        "multi_fixed.txt",
        "AB1234\nAB1234567890\nAB9999\n",
    )
    result = Detector().detect(path)
    assert result.structure_type is StructureType.MULTILINE
    assert not result.is_parseable()


# ---------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------
def test_empty_file_is_unsupported(tmp_path):
    path = write(tmp_path, "empty.txt", "")
    result = Detector().detect(path)
    assert result.format is FileFormat.UNSUPPORTED
    assert not result.is_parseable()
    assert result.confidence == 0.0


def test_single_long_column_has_low_confidence(tmp_path):
    # No delimiter, inconsistent line lengths -> not fixed-width.
    path = write(
        tmp_path,
        "junk.txt",
        "a short line\n"
        "a considerably longer line that keeps going\n"
        "tiny\n",
    )
    result = Detector().detect(path)
    assert result.format is FileFormat.DELIMITED
    assert result.confidence < APPROVAL_THRESHOLD


def test_confidence_always_bounded(tmp_path):
    path = write(tmp_path, "x.txt", "A|B\n1|2\n")
    result = Detector().detect(path)
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------
# Regression tests from the edge-case sweep.
# ---------------------------------------------------------------------
def test_escaped_delimiter_not_seen_as_field_separator(tmp_path):
    # ``\,`` inside a field must not split a record into extra columns.
    path = write(
        tmp_path,
        "escaped_delim.csv",
        "STORE,UNITS,PRICE\n1001,45\\,000,500\n1002,60,700\n",
    )
    result = Detector().detect(path)
    assert result.format is FileFormat.DELIMITED
    assert result.structure_type is StructureType.FLAT
    assert result.delimiter == ","
    fields = parse_fields(r"1001,45\,000,500", ",")
    assert fields == ["1001", "45,000", "500"]  # escaped delimiter restored


def test_blank_lines_do_not_look_like_record_typed(tmp_path):
    path = write(
        tmp_path,
        "empty_lines_mid.csv",
        "STORE,UNITS,PRICE\n1001,4,500\n\n\n1002,6,700\n",
    )
    result = Detector().detect(path)
    assert result.structure_type is StructureType.FLAT
    assert result.delimiter == ","
    assert result.header_present is True
    assert result.is_parseable()


def test_blank_lines_before_header_keep_header_detection(tmp_path):
    path = write(
        tmp_path,
        "blank_before_header.csv",
        "\nSTORE,UNITS,PRICE\n1001,4,500\n",
    )
    result = Detector().detect(path)
    assert result.structure_type is StructureType.FLAT
    assert result.header_present is True
    assert result.layout == ["STORE", "UNITS", "PRICE"]
    assert result.is_parseable()


def test_duplicate_header_names_disambiguated(tmp_path):
    path = write(tmp_path, "dup_header.csv", "A,A,B\n1,2,3\n4,5,6\n")
    result = Detector().detect(path)
    assert result.layout == ["A", "A_2", "B"]


def test_first_key_tolerates_blank_lines():
    assert first_key("", ",") == ""
    assert first_key("   ", ",") == ""


def test_cp1252_header_and_schema_detected(tmp_path):
    # Non-UTF-8 header must be inferrable using the detected encoding.
    path = tmp_path / "accents.csv"
    path.write_bytes(b"NAME,PRICE\ncaf\xe9,5\nm\xfcnchen,7\n")
    result = Detector().detect(path)
    assert result.encoding == "cp1252"
    assert result.header_present is True
    assert result.layout == ["NAME", "PRICE"]
    assert result.is_parseable()


# ---------------------------------------------------------------------
# Leading cover paragraphs ("preamble").
# ---------------------------------------------------------------------
def test_preamble_prose_detected_and_skipped(tmp_path):
    path = write(
        tmp_path,
        "preamble.csv",
        "Monthly Sales Report\n"
        "Generated by Data Operations\n"
        "Covers all stores\n"
        "STORE,UNITS,PRICE\n"
        "1001,45,500\n"
        "1002,60,700\n",
    )
    result = Detector().detect(path)
    assert result.format is FileFormat.DELIMITED
    assert result.structure_type is StructureType.FLAT
    assert result.preamble_lines == 3
    assert result.header_present is True
    assert result.layout == ["STORE", "UNITS", "PRICE"]
    assert result.is_parseable()
    # A preamble warrants human sign-off: confidence drops below threshold.
    assert result.confidence < APPROVAL_THRESHOLD
    assert any("non-data line" in warning for warning in result.warnings)


def test_preamble_with_delimiters_in_prose(tmp_path):
    # Prose lines may themselves contain the delimiter; the data region
    # still resolves to the stable header + rows.
    path = write(
        tmp_path,
        "preamble_commas.csv",
        "Report dated August 14, 2026, covers Q3.\n"
        "Author: Data Ops, Finance team.\n"
        "STORE,UNITS,PRICE\n"
        "1001,45,500\n"
        "1002,60,700\n",
    )
    result = Detector().detect(path)
    assert result.preamble_lines == 2
    assert result.header_present is True
    assert result.layout == ["STORE", "UNITS", "PRICE"]


def test_preamble_banner_and_blank_gap(tmp_path):
    path = write(
        tmp_path,
        "banner.csv",
        "================================\n"
        "MONTHLY SALES REPORT - AUTHORIZED\n"
        "================================\n"
        "\n"
        "STORE,UNITS,PRICE\n"
        "1001,45,500\n"
        "1002,60,700\n",
    )
    result = Detector().detect(path)
    assert result.preamble_lines == 4
    assert result.header_present is True
    assert result.layout == ["STORE", "UNITS", "PRICE"]
    assert result.is_parseable()


def test_single_row_shaped_odd_line_is_not_preamble(tmp_path):
    # One leading "H|..." row over uniform D rows is a record-type
    # prefix, NOT a preamble: it must keep its RECORD_TYPED flag.
    path = write(
        tmp_path,
        "header_detail.txt",
        "H|ORDER|TOTAL\nD|1001|45|500\nD|1002|60|700\n",
    )
    result = Detector().detect(path)
    assert result.preamble_lines == 0
    assert result.structure_type is StructureType.RECORD_TYPED
    assert not result.is_parseable()


def test_single_prose_line_is_preamble(tmp_path):
    # A single non-row-shaped lead line IS preamble (it parses to one
    # field, unlike a record-type tag row).
    path = write(
        tmp_path,
        "note.csv",
        "Internal note:\nSTORE,UNITS,PRICE\n1001,45,500\n1002,60,700\n",
    )
    result = Detector().detect(path)
    assert result.preamble_lines == 1
    assert result.header_present is True
    assert result.layout == ["STORE", "UNITS", "PRICE"]


# ---------------------------------------------------------------------
# Encoding detection.
# ---------------------------------------------------------------------
def test_encoding_cp1252(tmp_path):
    path = tmp_path / "cp.txt"
    path.write_bytes("caf\xe9|prix\n1|2\n".encode("latin-1"))  # é = 0xE9
    encoding, method = detect_encoding(path)
    assert encoding == "cp1252"
    assert method == "cp1252"


def test_encoding_utf8_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfA|B\n1|2\n")
    encoding, method = detect_encoding(path)
    assert encoding == "utf-8"
    assert method == "utf-8-bom"


def test_sampling_strips_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfA|B\n1|2\n")
    sample = read_bounded_sample(path)
    assert sample.lines[0] == "A|B"
    assert sample.encoding == "utf-8"


def test_sampling_is_bounded_by_lines(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("\n".join(f"row-{i}" for i in range(100_000)), encoding="utf-8")
    sample = read_bounded_sample(path, max_lines=1000)
    assert sample.line_count == 1000


def test_validate_layout_against_sample_reports_bad_coverage(tmp_path, layout_csv):
    from models.detection_models import ColumnSpec, FixedWidthLayout

    layout = load_layout_csv(layout_csv)
    # All lines shorter than the 14-char layout width.
    warnings = validate_layout_against_sample(layout, ["10010045", "10020060"])
    assert len(warnings) == 1
    assert "width" in warnings[0]

    # Clean sample produces no warnings.
    assert validate_layout_against_sample(layout, ["10010045000500"] * 5) == []

    # A layout object is validated through ColumnSpec as well.
    assert ColumnSpec(field="X", start=0, end=4).width == 4
