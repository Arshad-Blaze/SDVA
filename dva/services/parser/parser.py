"""Parser orchestration and reporting models (Boundary 4).

The parser turns one verified local raw file into a structured in-memory
frame plus a ``ParseReport`` describing every row that could not be read
as-is. The Parquet writer (Phase 5) persists the frame; the report feeds
the per-file verification step.

The decision made in Phase 3 carries through here:

  - delimited, flat / multiline         -> delimited_parser.py
  - fixed-width, flat                   -> fixed_width_parser.py
  - fixed-width multiline, record_typed -> detection blocks these before
    parsing ever starts (DetectionResult.is_parseable is False).

Report semantics are ported from the legacy tool
``delimited_file_processor_final.py``: per-record issues are collected
with their physical start line, expected/found column counts and the raw
text, so failures stay human-verifiable instead of being silently lost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from models.detection_models import ApprovedConfig, FileFormat

# Issue categories shared by both parsers. Kept as plain strings so the
# report survives JSON round-trips without an extra enum.
ISSUE_COLUMN_MISMATCH = "column_mismatch"
ISSUE_QUOTED_DELIMITER = "quoted_delimiter"
ISSUE_ESCAPED_DELIMITER = "escaped_delimiter"
ISSUE_PARSE_ERROR = "parse_error"
ISSUE_SHORT_LINE = "short_line"


def clean_illegal_chars(value: str) -> str:
    """Drop control characters so raw rows can be written to reports."""
    return "".join(ch for ch in value if ord(ch) >= 32 or ch in "\t\n\r")


@dataclass(frozen=True, slots=True)
class ParseIssue:
    """One problematic input record (mirrors the legacy bad-row log)."""
    line_number: int
    issue: str
    expected_columns: int | str
    found_columns: int | str
    raw_data: str


@dataclass(slots=True)
class ParseReport:
    """Summary of one parse run; feeds the per-file verification step."""
    source_path: str
    total_records: int = 0
    parsed_records: int = 0
    issues: list[ParseIssue] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    @property
    def bad_rows_count(self) -> int:
        # Only blocking issues (data lost/reshaped) count as bad rows;
        # quoted/escaped delimiters are informational signals.
        blocking = {
            ISSUE_COLUMN_MISMATCH,
            ISSUE_PARSE_ERROR,
            ISSUE_SHORT_LINE,
        }
        return sum(1 for issue in self.issues if issue.issue in blocking)

    def add_issue(self, issue: ParseIssue) -> None:
        self.issues.append(issue)
        self.counters[issue.issue] = self.counters.get(issue.issue, 0) + 1

    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict) -> "ParseReport":
        report = cls(
            source_path=data["source_path"],
            total_records=data["total_records"],
            parsed_records=data["parsed_records"],
            counters=dict(data["counters"]),
        )
        report.issues = [
            ParseIssue(**issue) for issue in data["issues"]
        ]
        return report


def parse_file(
    path: str | Path,
    config: ApprovedConfig,
) -> tuple[pl.DataFrame, ParseReport]:
    """Parse one raw file under an approved configuration.

    Returns:
        (structured frame, parse report). The caller is responsible for
        downstream verification and powering reports; parsing itself
        never raises on malformed input - it records issues instead.
    """
    config.validate()
    path = Path(path)

    if config.source_format is FileFormat.DELIMITED:
        from services.parser.delimited_parser import parse_delimited

        return parse_delimited(path, config)
    if config.source_format is FileFormat.FIXED_WIDTH:
        from services.parser.fixed_width_parser import parse_fixed_width

        return parse_fixed_width(path, config)

    raise ValueError(
        f"Unsupported format for parsing: {config.source_format}."
    )