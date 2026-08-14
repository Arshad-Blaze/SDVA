import pytest

from models.detection_models import (
    ApprovedConfig,
    ColumnSpec,
    FileFormat,
    FixedWidthLayout,
)
from services.parser.fixed_width_parser import parse_fixed_width
from services.parser.parser import (
    ISSUE_COLUMN_MISMATCH,
    ISSUE_QUOTED_DELIMITER,
    ISSUE_SHORT_LINE,
    ParseReport,
    clean_illegal_chars,
    parse_file,
)
from services.parser.delimited_parser import parse_delimited


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def delimited_config(*, columns=None, delimiter=",", header=True):
    return ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter=delimiter,
        header_present=header,
        columns=list(columns) if columns is not None else ["STORE", "UNITS", "PRICE"],
    )


def fixed_config(*specs, header=True):
    layout = FixedWidthLayout.from_specs(
        [ColumnSpec(**spec) for spec in specs]
    )
    return ApprovedConfig(
        source_format=FileFormat.FIXED_WIDTH,
        header_present=header,
        fixed_width_layout=layout,
        columns=[spec["field"] for spec in specs],
    )


def test_unparseable_format_raises(tmp_path):
    path = write(tmp_path, "x.txt", "a\nb\n")
    config = ApprovedConfig(source_format=FileFormat.UNSUPPORTED)
    with pytest.raises(ValueError):
        parse_file(path, config)


def test_delimited_requires_columns(tmp_path):
    path = write(tmp_path, "x.csv", "1,2\n")
    config = delimited_config(columns=[])
    with pytest.raises(ValueError, match="column names"):
        parse_delimited(path, config)


# ---------------------------------------------------------------------
# Delimited: flat.
# ---------------------------------------------------------------------
def test_flat_delimited_parses_all_records(tmp_path):
    path = write(tmp_path, "flat.csv", "STORE,UNITS,PRICE\n1001,45,500\n1002,60,700\n")
    frame, report = parse_file(path, delimited_config())
    assert frame.shape == (2, 3)
    assert list(frame.columns) == ["STORE", "UNITS", "PRICE"]
    assert report.total_records == 3  # includes the header
    assert report.parsed_records == 2
    assert report.bad_rows_count == 0


def test_delimited_without_header_uses_positional_names(tmp_path):
    path = write(tmp_path, "nohdr.csv", "1001,45,500\n1002,60,700\n")
    config = delimited_config(columns=["STORE", "UNITS", "PRICE"], header=False)
    frame, report = parse_file(path, config)
    assert frame.shape == (2, 3)
    assert report.total_records == 2
    assert report.parsed_records == 2


def test_quoted_delimiter_field_survives(tmp_path):
    path = write(tmp_path, "q.csv", 'STORE,UNITS,PRICE\n1001,"45,000",500\n1002,60,700\n')
    frame, report = parse_file(path, delimited_config())
    assert frame["UNITS"][0] == "45,000"
    assert frame["UNITS"][1] == "60"
    # A quoted delimiter is informational and does not fail the record.
    assert report.counters.get(ISSUE_QUOTED_DELIMITER) == 1


def test_aliases_and_escaped_delimiter(tmp_path):
    path = write(tmp_path, "esc.txt", 'STORE,UNITS,PRICE\n1001,45\\,000,500\n1002,60,700\n')
    frame, report = parse_file(path, delimited_config())
    assert frame["UNITS"][0] == "45,000"  # escaped delimiter restored


# ---------------------------------------------------------------------
# Delimited: multiline.
# ---------------------------------------------------------------------
def test_multiline_quoted_field_assembles_one_record(tmp_path):
    path = write(
        tmp_path,
        "multi.csv",
        'STORE,UNITS,PRICE\n1001,"45,000",500\n1003,"multi\nline value",800\n1002,60,700\n',
    )
    frame, report = parse_file(path, delimited_config())
    assert frame.shape == (3, 3)
    assert frame["UNITS"][1] == "multi\nline value"
    # Physical line 4 holds the end of the second data record.
    assert report.bad_rows_count == 0
    assert report.total_records == 4


def test_multiline_continuation_marker(tmp_path):
    # "one,two\\" + "three" merge into one logical record via the marker.
    path = write(tmp_path, "cont.csv", "A,B\none,two\\\nthree\nfive,six\n")
    frame, report = parse_file(
        path,
        delimited_config(columns=["A", "B"], delimiter=","),
    )
    assert frame.height == 2
    assert frame["B"][0] == "twothree"
    assert report.bad_rows_count == 0


# ---------------------------------------------------------------------
# Delimited: blank lines & duplicate headers (edge-case sweep fixes).
# ---------------------------------------------------------------------
def test_blank_records_are_dropped_not_parsed(tmp_path):
    path = write(
        tmp_path,
        "blanks.csv",
        "A,B\n1,2\n\n   \n3,4\n",
    )
    frame, report = parse_file(
        path, delimited_config(columns=["A", "B"], delimiter=",")
    )
    assert frame.height == 2  # blanks are padding, not rows
    assert report.total_records == 3  # header + 2 data records only


def test_blank_lines_before_header_are_skipped(tmp_path):
    path = write(tmp_path, "leadblank.csv", "\nA,B\n1,2\n3,4\n")
    frame, report = parse_file(
        path, delimited_config(columns=["A", "B"], delimiter=",")
    )
    assert frame.height == 2
    assert frame.columns == ["A", "B"]


def test_duplicate_header_columns_disambiguated(tmp_path):
    # Detection dedupes the layout; the parser must tolerate it too rather
    # than crashing on a column_data key collision.
    path = write(tmp_path, "dup.csv", "A,A,B\n1,2,3\n4,5,6\n")
    frame, report = parse_file(
        path,
        delimited_config(columns=["A", "A_2", "B"], delimiter=","),
    )
    assert frame.shape == (2, 3)
    assert list(frame.columns) == ["A", "A_2", "B"]
    assert frame["A"].to_list() == ["1", "4"]
    assert frame["A_2"].to_list() == ["2", "5"]


# ---------------------------------------------------------------------
# Delimited: skip_rows (leading cover paragraphs).
# ---------------------------------------------------------------------
def test_delimited_skip_rows_skips_preamble(tmp_path):
    path = write(
        tmp_path,
        "pre.csv",
        "Monthly Sales Report\nGenerated by Ops\n"
        "STORE,UNITS,PRICE\n1001,45,500\n1002,60,700\n",
    )
    config = delimited_config()
    config.skip_rows = 2
    frame, report = parse_file(path, config)
    assert frame.shape == (2, 3)
    assert frame["STORE"].to_list() == ["1001", "1002"]
    # Preamble never counts; header + 2 data records only.
    assert report.total_records == 3
    assert report.parsed_records == 2
    assert report.bad_rows_count == 0


def test_delimited_skip_rows_no_header(tmp_path):
    path = write(tmp_path, "pre.csv", "banner line\n1001,45,500\n1002,60,700\n")
    config = delimited_config(columns=["A", "B", "C"], header=False)
    config.skip_rows = 1
    frame, report = parse_file(path, config)
    assert frame.height == 2
    assert frame["A"].to_list() == ["1001", "1002"]


def test_fixed_width_skip_rows_skips_banner(tmp_path):
    path = write(tmp_path, "fw.txt", "BANNER\n1001  45\n1002  60\n")
    config = fixed_config(
        dict(field="STORE", start=0, end=4),
        dict(field="UNITS", start=5, end=9),
        header=False,
    )
    config.skip_rows = 1
    frame, report = parse_file(path, config)
    assert frame.height == 2
    assert frame["STORE"].to_list() == ["1001", "1002"]


# ---------------------------------------------------------------------
# Delimited: issues.
# ---------------------------------------------------------------------
def test_column_mismatch_logged_and_padded(tmp_path):
    path = write(tmp_path, "bad.csv", "STORE,UNITS,PRICE\n1,2\n3,4,5,6\n7,8,9\n")
    frame, report = parse_file(path, delimited_config())
    assert report.total_records == 4  # header + 3 data rows
    assert report.counters[ISSUE_COLUMN_MISMATCH] == 2
    # Plus a quoted-delimiter signal on the over-long row shares an issue record.
    assert [issue.line_number for issue in report.issues] == [2, 3]
    # Short row padded, long row truncated -> frame stays 3 columns wide.
    assert frame.width == 3
    assert frame["PRICE"][0] == ""


def test_odd_rows_are_preserved_and_logged(tmp_path):
    # An over-wide row is logged as a mismatch, padded/truncated to the
    # schema, and still lands in the frame (nothing silently dropped).
    path = write(tmp_path, "oops.csv", "STORE,UNITS,PRICE\n1,2,3\n!!!,?,,,\n7,8,9\n")
    frame, report = parse_file(path, delimited_config())
    assert report.counters[ISSUE_COLUMN_MISMATCH] == 1
    assert frame.height == 3
    assert frame["STORE"][1] == "!!!"


def test_report_json_round_trip(tmp_path):
    path = write(tmp_path, "rt.csv", "A,B\nx\nC,D\n")
    _, report = parse_file(path, delimited_config(columns=["A", "B"]))
    restored = ParseReport.from_json_dict(report.to_json_dict())
    assert restored.total_records == report.total_records
    assert restored.bad_rows_count == report.bad_rows_count
    assert restored.issues[0].issue == report.issues[0].issue


# ---------------------------------------------------------------------
# Fixed-width.
# ---------------------------------------------------------------------
def test_fixed_width_parses_by_layout(tmp_path):
    layout = [
        {"field": "STORE", "start": 0, "end": 4},
        {"field": "UNITS", "start": 4, "end": 9},
        {"field": "PRICE", "start": 9, "end": 14},
    ]
    path = write(tmp_path, "fixed.txt", "STOREUNITS PRIC\n1001 0045500  \n1002 0060700  \n")
    frame, report = parse_file(path, fixed_config(*layout))
    assert frame.shape == (2, 3)
    assert frame["STORE"][0] == "1001"
    assert report.total_records == 3
    assert report.bad_rows_count == 0


def test_fixed_width_short_line_logged(tmp_path):
    layout = [
        {"field": "A", "start": 0, "end": 4},
        {"field": "B", "start": 4, "end": 8},
    ]
    # Second line is 5 chars, shorter than the 8-char layout.
    path = write(tmp_path, "short.txt", "1001AAAA\n100\n9999BBBB\n")
    frame, report = parse_file(path, fixed_config(*layout, header=False))
    assert report.counters[ISSUE_SHORT_LINE] == 1
    assert frame.shape == (3, 2)
    assert frame["B"][1] == ""


def test_fixed_width_requires_layout(tmp_path):
    path = write(tmp_path, "x.txt", "1\n")
    config = ApprovedConfig(source_format=FileFormat.FIXED_WIDTH)
    with pytest.raises(ValueError, match="layout"):
        parse_fixed_width(path, config)


def test_clean_illegal_chars_strips_control_characters():
    assert clean_illegal_chars("a\x00b\x1fc") == "abc"
    assert clean_illegal_chars("a\nb\tc") == "a\nb\tc"