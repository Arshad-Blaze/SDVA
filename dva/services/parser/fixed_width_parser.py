"""Fixed-width parser (flat records only, using a user layout CSV).

Every physical line is one record. Each field is sliced out of
``FixedWidthLayout``'s ordered, non-overlapping columns and stripped.
Lines shorter than ``layout.width`` are logged as ``short_line`` and the
missing columns are filled with empty strings so the frame is complete.
Multiline fixed-width and record-typed files never reach here: the
detector blocks them before parsing (is_parseable is False). Encoding is
the one the sampler detected, applied at read time.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from models.detection_models import ApprovedConfig
from services.parser.parser import (
    ISSUE_SHORT_LINE,
    ParseIssue,
    ParseReport,
    clean_illegal_chars,
)


def parse_fixed_width(
    path: Path, config: ApprovedConfig
) -> tuple[pl.DataFrame, ParseReport]:
    """Parse a fixed-width file into a structured frame plus a report."""
    if config.fixed_width_layout is None:
        raise ValueError(
            "Fixed-width parsing requires an approved layout."
        )

    layout = config.fixed_width_layout
    columns = [column.field for column in layout.columns]
    width = layout.width

    report = ParseReport(source_path=str(path))

    rows: list[dict[str, str]] = []
    with path.open(encoding=config.encoding, errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")

            # Leading preamble lines (banners, cover text) are not data
            # and never count toward the header/data accounting.
            if line_number <= config.skip_rows:
                continue

            report.total_records += 1

            if config.header_present and report.total_records == 1:
                report.parsed_records += 1
                continue

            # Lines shorter than the layout may also be blank padding at
            # the end of the file; the row is still padded to full width.
            if len(line) < width:
                report.add_issue(
                    ParseIssue(
                        line_number=line_number,
                        issue=ISSUE_SHORT_LINE,
                        expected_columns=width,
                        found_columns=len(line),
                        raw_data=clean_illegal_chars(line),
                    )
                )

            rows.append(
                {
                    column.field: line[column.start : column.end].strip()
                    for column in layout.columns
                }
            )
            report.parsed_records += 1

    frame = pl.DataFrame(rows, schema=columns)
    return frame, report