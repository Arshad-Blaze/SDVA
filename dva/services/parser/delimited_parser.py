"""Delimited (CSV-like) parser supporting flat and multiline records.

Uses Python's ``csv`` module as a *record* reader over the decoded file
streamed line by line (never the whole file at once). Unlike the legacy
per-physical-line approach, a quoted field may span several physical
lines and the logical record is still assembled correctly.

Memory is bounded: ``_PhysicalLineReader`` feeds ``csv.reader`` one
physical line at a time and remembers only the lines of the current
record so raw text can be rebuilt for the report.

Continuation markers (a backslash at end of a physical line, reference
``delimited_file_processor_final.py``) are handled at the *record* level:
a marker-terminated record is held back and merged with the next record's
raw text before parsing, exactly like the legacy ``text.replace("\\\\n", "")``
pre-pass, but streaming.

``reader.line_num`` reports how many physical lines csv has consumed;
subtracting the newlines embedded inside quoted fields yields the
record's start line for accurate reports.

Issues recorded per record:

  - escaped_delimiter : the escape char falls just before the delimiter,
  - quoted_delimiter  : naive split count differs from the parsed count,
  - column_mismatch   : parsed field count differs from the schema,
  - parse_error       : csv parsing itself failed for the record.
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

from models.detection_models import ApprovedConfig
from services.parser.parser import (
    ISSUE_COLUMN_MISMATCH,
    ISSUE_ESCAPED_DELIMITER,
    ISSUE_PARSE_ERROR,
    ISSUE_QUOTED_DELIMITER,
    ParseIssue,
    ParseReport,
    clean_illegal_chars,
)

# Marker swapped in for an escaped delimiter while csv parses the record,
# then swapped back. Private-use char keeps it out of real data.
_ESCAPE_MARKER = "\ue000"

# Continuation marker: "\" as the last char of a physical line means the
# logical record continues on the next line.
CONTINUATION_MARKER = "\\"


class _PhysicalLineReader:
    """Streams physical lines into csv.reader and remembers each line.

    ``index`` tracks the next physical line number. Consumed lines are
    kept in ``_buffer`` until the caller purges them, so raw text of the
    current record can be rebuilt without holding the whole file.
    """

    def __init__(self, handle) -> None:
        self._handle = handle
        self.index = 0  # physical line number of the next line returned
        self._buffer: dict[int, str] = {}

    def __iter__(self):
        return self

    def __next__(self) -> str:
        raw = self._handle.readline()
        if raw == "":
            raise StopIteration
        self.index += 1
        line = raw.rstrip("\r\n")
        self._buffer[self.index] = line
        # Keep the newline terminator: csv's parser needs it to assemble
        # multiline quoted records.
        return raw

    def raw_lines(self, start: int, end: int) -> str:
        """Raw text of physical lines in the half-open range [start, end)."""
        return "\n".join(
            self._buffer[i] for i in range(start, end) if i in self._buffer
        )

    def purge_before(self, index: int) -> None:
        """Drop remembered lines strictly before ``index`` (bounded memory)."""
        for key in [key for key in self._buffer if key < index]:
            del self._buffer[key]


def _has_quoted_delimiter(
    raw_line: str, parsed_fields: list[str], delimiter: str
) -> bool:
    """True when quotes hide a delimiter, i.e. naive and parsed counts differ."""
    if '"' not in raw_line:
        return False
    return len(raw_line.split(delimiter)) != len(parsed_fields)


def _parse_raw(
    raw: str,
    delimiter: str,
    escape_pattern: str,
    expected: int,
    report: ParseReport,
    line_number: int,
) -> tuple[list[str], int, bool]:
    """CSV-parse one raw record, protecting escaped delimiters first.

    Returns ``(fields, found_count, parse_error)``. On a genuine parse
    error the whole record collapses to a single raw field so no input is
    silently dropped.
    """
    try:
        safe = raw.replace(escape_pattern, _ESCAPE_MARKER)
        fields = next(csv.reader([safe], delimiter=delimiter, skipinitialspace=True))
        fields = [field.replace(_ESCAPE_MARKER, delimiter) for field in fields]
        return fields, len(fields), False
    except (csv.Error, StopIteration):
        report.add_issue(
            ParseIssue(
                line_number=line_number,
                issue=ISSUE_PARSE_ERROR,
                expected_columns=expected,
                found_columns="N/A",
                raw_data=clean_illegal_chars(raw),
            )
        )
        return [clean_illegal_chars(raw)], 1, True


def parse_delimited(
    path: Path, config: ApprovedConfig
) -> tuple[pl.DataFrame, ParseReport]:
    """Parse a delimited file into a structured frame plus a report.

    The schema of ``config.columns`` is authoritative: every record is
    compared against it and mismatches are logged, not dropped.
    """
    if not config.columns:
        raise ValueError(
            "Delimited parsing requires column names in ApprovedConfig."
        )

    delimiter = config.delimiter
    escape_pattern = f"\\{delimiter}"
    expected = len(config.columns)
    columns = config.columns
    # Duplicate names would collide in ``column_data``; disambiguate the
    # same way the detector does (name, name_2, name_3, ...).
    seen: dict[str, int] = {}
    disambiguated: list[str] = []
    for name in columns:
        if name in seen:
            seen[name] += 1
            disambiguated.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            disambiguated.append(name)
    expected = len(disambiguated)
    columns = disambiguated

    report = ParseReport(source_path=str(path))
    # Column-wise accumulation avoids a separate list-of-lists copy.
    column_data: dict[str, list[str]] = {name: [] for name in columns}

    with path.open(encoding=config.encoding, errors="replace") as handle:
        line_reader = _PhysicalLineReader(handle)
        reader = csv.reader(line_reader, delimiter=delimiter)
        pending_fields: list[str] | None = None
        pending_start = 0
        pending_raw = ""

        for record in reader:
            embedded_newlines = sum(field.count("\n") for field in record)
            start_line = reader.line_num - embedded_newlines
            # Physical line numbers are 1-based and inclusive: a record on
            # line L is [L, L], hence the +1 on the exclusive end.
            raw = line_reader.raw_lines(start_line, reader.line_num + 1)
            line_reader.purge_before(reader.line_num + 1)

            # Blank / whitespace-only records are padding, not data: drop
            # them so they never count toward the header or the frame.
            if not raw.strip():
                continue

            # Leading preamble lines (cover paragraph, banners) are not
            # data either: skip the first ``skip_rows`` physical lines
            # entirely, before any header/data accounting.
            if start_line <= config.skip_rows:
                continue

            report.total_records += 1

            # Header (never a continuation) is skipped, not counted.
            if config.header_present and report.total_records == 1:
                continue

            # A record whose raw ends with the continuation marker waits
            # for its partner before being parsed as one logical record.
            if raw.rstrip().endswith(CONTINUATION_MARKER):
                pending_fields = ["?PLACEHOLDER?"]
                pending_start = start_line
                pending_raw = raw
                continue

            if pending_fields is not None:
                # Merge: drop the marker + line break, then concatenate.
                merged_raw = pending_raw.rstrip()[:-1] + raw
                merged_start = pending_start
                raw = merged_raw
                start_line = merged_start
                pending_fields = None

            found = len(record)
            issues: list[ParseIssue] = []

            if escape_pattern in raw:
                issues.append(
                    ParseIssue(
                        line_number=start_line,
                        issue=ISSUE_ESCAPED_DELIMITER,
                        expected_columns=expected,
                        found_columns=found,
                        raw_data=clean_illegal_chars(raw),
                    )
                )

            parsed, found, parsed_error = _parse_raw(
                raw, delimiter, escape_pattern, expected, report, start_line
            )

            if (
                not parsed_error
                and embedded_newlines == 0
                and _has_quoted_delimiter(raw, parsed, delimiter)
            ):
                issues.append(
                    ParseIssue(
                        line_number=start_line,
                        issue=ISSUE_QUOTED_DELIMITER,
                        expected_columns=expected,
                        found_columns=found,
                        raw_data=clean_illegal_chars(raw),
                    )
                )

            if found != expected and not parsed_error:
                issues.append(
                    ParseIssue(
                        line_number=start_line,
                        issue=ISSUE_COLUMN_MISMATCH,
                        expected_columns=expected,
                        found_columns=found,
                        raw_data=clean_illegal_chars(raw),
                    )
                )

            for issue in issues:
                report.add_issue(issue)

            # Pad short rows / truncate over-long rows to the schema width
            # so the frame keeps exactly ``expected`` columns.
            if len(parsed) < expected:
                parsed = parsed + [""] * (expected - len(parsed))
            else:
                parsed = parsed[:expected]
            for index, value in enumerate(parsed):
                column_data[columns[index]].append(value)
            report.parsed_records += 1

    frame = pl.DataFrame(column_data)
    return frame, report