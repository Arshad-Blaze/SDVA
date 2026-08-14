"""Format/structure/schema detector (Boundary 3).

Operates only on bounded samples (sampler.py) and produces a complete
``DetectionResult``: format, structure, encoding, delimiter, header,
layout, schema, confidence and sample metadata (04, Boundary 3).

Phase 1 structure handling (agreed with the user):
  - delimited, flat / multiline          -> parseable
  - fixed-width, flat                    -> parseable (layout CSV required)
  - fixed-width multiline, record_typed  -> flagged, NOT parsed

Confidence is a deterministic 0..1 score. Results below
``APPROVAL_THRESHOLD`` route the file to ``AWAITING_APPROVAL`` so a
human reviews the detection before parsing (02 lifecycle).
"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import polars as pl

from models.detection_models import (
    DetectionResult,
    FileFormat,
    FixedWidthLayout,
    StructureType,
)
from services.ingestion.fields import (
    first_key,
    is_multiline_delimited,
    is_multiline_fixed_width,
    is_record_typed,
    numeric_like_fraction,
    parse_fields,
    strip_quoted,
)
from services.ingestion.layout import (
    LayoutLoadError,
    load_layout_csv,
    validate_layout_against_sample,
)
from services.ingestion.sampler import read_bounded_sample

# Candidates tried in order; the one with the most consistent per-line
# count wins. Tab is tried last to avoid clashing with column padding.
DELIMITER_CANDIDATES = [",", ";", "|", "\t"]

# Fraction of sampled lines that must agree on the delimiter count.
MIN_DELIMITER_CONSISTENCY = 0.9

# Below this confidence the orchestrator sends the file for review.
APPROVAL_THRESHOLD = 0.8

# Fixed-width signal: this fraction of lines must share one length.
FIXED_WIDTH_CONSISTENCY = 0.95

# How many rows Polars reads when inferring schema/header.
_INFER_ROWS = 20_000
_INFER_SCHEMA_LENGTH = 2_000


class DetectionError(Exception):
    """Raised when detection cannot proceed at all."""


class Detector:
    """Detects format/structure/header/schema of one raw file."""

    def __init__(self, layout_csv: str | Path | None = None) -> None:
        self._default_layout_csv = Path(layout_csv) if layout_csv else None

    # ==================================================================
    # Public entry point.
    # ==================================================================
    def detect(
        self,
        path: str | Path,
        layout_csv: str | Path | None = None,
    ) -> DetectionResult:
        """Detect everything about the file at ``path``.

        Args:
            path: verified local raw file.
            layout_csv: optional layout for fixed-width files; overrides
                the layout given at construction time.
        """
        path = Path(path)
        sample = read_bounded_sample(path)
        layout_csv = Path(layout_csv) if layout_csv else self._default_layout_csv

        delimiter, delim_stats = self._detect_delimiter(sample.lines)
        preamble_lines = self._find_preamble_lines(
            sample.lines, delimiter, delim_stats
        )
        # Everything below estimates format/structure/schema from the
        # DATA region only, so a cover paragraph cannot skew the result.
        data_lines = sample.lines[preamble_lines:]
        file_format = self._decide_format(data_lines, delimiter)
        structure = self._detect_structure(data_lines, delimiter)

        # No delimiter + multiline records: this is multiline fixed-width
        # (or an unknown single-column structure) — never auto-parseable.
        if (
            file_format is FileFormat.DELIMITED
            and delimiter is None
            and structure is StructureType.MULTILINE
        ):
            file_format = FileFormat.FIXED_WIDTH

        if file_format is FileFormat.DELIMITED and delimiter is not None:
            header = self._detect_header(sample, delimiter, preamble_lines)
            layout = self._build_delimited_layout(
                sample, delimiter, header, preamble_lines
            )
            schema = self._infer_delimited_schema(
                sample, delimiter, header, preamble_lines
            )
        else:
            # No reliable delimiter: either fixed-width or a single
            # unstructured column; header is unknown in both cases.
            header = None
            layout = None
            schema = None

        if file_format is FileFormat.FIXED_WIDTH:
            layout, warnings = self._load_fixed_width_layout(
                layout_csv, sample
            )
            if layout is not None:
                schema = self._infer_fixed_width_schema(
                    sample.lines[preamble_lines:], layout
                )
        else:
            warnings = []
            if preamble_lines > 0:
                warnings.append(
                    f"skipped {preamble_lines} leading non-data line(s); "
                    f"data starts at physical line {preamble_lines + 1}"
                )

        sample_metadata = self._build_sample_metadata(
            sample, delimiter, delim_stats, structure, preamble_lines
        )
        confidence = self._score_confidence(
            file_format=file_format,
            structure=structure,
            delimiter=delimiter,
            delimiter_consistency=delim_stats.get("consistency", 0.0),
            encoding_method=sample.encoding_method,
            layout_present=layout is not None,
            header=header,
            preamble_lines=preamble_lines,
        )

        return DetectionResult(
            format=file_format,
            structure_type=structure,
            encoding=sample.encoding,
            delimiter=delimiter,
            header_present=header,
            layout=layout,
            schema=schema,
            confidence=confidence,
            preamble_lines=preamble_lines,
            sample_metadata=sample_metadata,
            warnings=warnings,
        )

    # ==================================================================
    # Format & delimiter.
    # ==================================================================
    def _detect_delimiter(self, lines: list[str]) -> tuple[str | None, dict]:
        """Return ``(delimiter or None, stats)`` from per-line counts.

        Counts are computed on lines with quoted segments stripped so a
        delimiter inside a quoted field does not skew the signal.

        Lines with ZERO of a candidate are ambiguous — they may be a
        cover paragraph before the data — so the consensus count is taken
        from the lines that actually contain it. A single stray line is
        never enough to declare a format.
        """
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            return None, {}

        best: str | None = None
        best_score = 0.0
        best_stats: dict = {}
        for delimiter in DELIMITER_CANDIDATES:
            counts = [
                strip_quoted(line).count(delimiter)
                for line in non_empty
            ]
            nonzero = [count for count in counts if count > 0]
            if len(nonzero) < 2:
                continue  # delimiter absent or only one stray occurrence
            common, occurrences = Counter(nonzero).most_common(1)[0]
            consistency = occurrences / len(nonzero)
            # Prefer higher consistency; break ties with total occurrences.
            score = (consistency, occurrences)
            if best is None or score > (best_score, best_stats.get("occurrences", 0)):
                best = delimiter
                best_score = consistency
                best_stats = {
                    "per_line_counts": counts[:20],
                    "most_common": common,
                    "occurrences": occurrences,
                    "consistency": round(consistency, 4),
                }
        return best, best_stats

    @staticmethod
    def _find_preamble_lines(
        lines: list[str], delimiter: str | None, delim_stats: dict
    ) -> int:
        """Number of leading physical lines that are not data.

        The data region begins where the consensus delimiter count
        becomes STABLE over at least two consecutive lines. A cover
        paragraph may itself contain the delimiter, so one matching line
        is not enough. A single odd leading line is treated as a
        record-type prefix (NOT a preamble) so ``record_typed`` files
        keep their flag. Blank lines before the data count as preamble.
        """
        if delimiter is None:
            return 0
        common = delim_stats.get("most_common")
        if common is None or common <= 0:
            return 0
        counts = [
            strip_quoted(line).count(delimiter) if line.strip() else -1
            for line in lines
        ]
        run = 0
        run_start = 0
        for index, count in enumerate(counts):
            if count == common:
                if run == 0:
                    run_start = index
                run += 1
                if run == 2:
                    return Detector._preamble_decision(lines, counts, delimiter, run_start)
            else:
                run = 0
        return 0  # no stable data region found -> treat as no preamble

    @staticmethod
    def _preamble_decision(
        lines: list[str], counts: list[int], delimiter: str, run_start: int
    ) -> int:
        """Decide whether the block before the stable data run is a
        preamble (skip it) or a record-type prefix (keep it, so
        ``record_typed`` detection still fires).

        - Prose/banner lines parse to a single field -> real preamble.
        - A block of >= 2 row-shaped odd lines (>= 2 fields) -> preamble.
        - A single row-shaped odd line -> a record-type prefix, NOT
          preamble, so structure detection classifies the file.
        - A block of only blank lines -> padding, treated as preamble.
        """
        source = lines[:run_start]
        row_odd = 0
        for index, line in enumerate(source):
            if counts[index] == -1:
                continue  # blank line
            if len(parse_fields(line, delimiter)) >= 2:
                row_odd += 1  # delimited, could be a record-type row
            else:
                return run_start  # prose/banner line -> real preamble
        if row_odd >= 2:
            return run_start
        if row_odd == 0:
            return run_start  # only blank padding
        return 0  # exactly one row-shaped odd line -> record-type prefix

    def _decide_format(self, lines: list[str], delimiter: str | None) -> FileFormat:
        """Delimited when a delimiter exists; else fixed-width if line
        lengths are consistent; else delimited single-column, low trust."""
        if delimiter is not None:
            return FileFormat.DELIMITED

        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            return FileFormat.UNSUPPORTED

        lengths = [len(line) for line in non_empty]
        common, occurrences = Counter(lengths).most_common(1)[0]
        if occurrences / len(lengths) >= FIXED_WIDTH_CONSISTENCY:
            return FileFormat.FIXED_WIDTH
        return FileFormat.DELIMITED  # single unstructured column

    # ==================================================================
    # Structure (flat / multiline / record-typed).
    # ==================================================================
    def _detect_structure(
        self, lines: list[str], delimiter: str | None
    ) -> StructureType:
        # Multiline first: an unterminated quote is a continuation of one
        # logical record, not interleaved header/detail rows.
        if delimiter is not None and is_multiline_delimited(lines):
            return StructureType.MULTILINE

        if is_record_typed(lines, delimiter):
            return StructureType.RECORD_TYPED

        if delimiter is None and is_multiline_fixed_width(lines):
            return StructureType.MULTILINE

        return StructureType.FLAT

    # ==================================================================
    # Header & layout.
    # ==================================================================
    @staticmethod
    def _detect_header(
        sample, delimiter: str, preamble_lines: int = 0
    ) -> bool | None:
        """Decide whether the first line of the DATA region is a header.

        Works on the decoded sampler lines so it is immune to polars
        skip/blank quirks and honours non-UTF-8 encodings (the sampler
        already decoded them). A header line is text; a data row is
        numeric, auto-generated, or a row of separators.
        """
        data_lines = [line for line in sample.lines[preamble_lines:] if line.strip()]
        if not data_lines:
            return None
        first = [field.strip() for field in parse_fields(data_lines[0], delimiter)]
        if not first or all(field == "" for field in first):
            return False  # a row of separators is no header
        if numeric_like_fraction(first) >= 0.5:
            return False  # the "header" is really a row of data
        if len(data_lines) >= 2:
            second = [field.strip() for field in parse_fields(data_lines[1], delimiter)]
            if numeric_like_fraction(second) >= 0.8:
                return True  # text headings over numeric data
        return True

    @staticmethod
    def _build_delimited_layout(
        sample, delimiter: str, header: bool | None, preamble_lines: int = 0
    ) -> list[str]:
        """Column names for a delimited file: header names or generated."""
        # The schema must come from the first line of the DATA region,
        # so a leading cover paragraph or blank lines are skipped over.
        line = sample.lines[preamble_lines] if len(sample.lines) > preamble_lines else ""
        if header:
            # Reuse the first content line as the header, disambiguating
            # duplicate names the way Polars does (name, name_2, ...).
            columns = [
                column.strip() for column in line.split(delimiter) if column.strip()
            ]
            seen: dict[str, int] = {}
            unique: list[str] = []
            for column in columns:
                if column in seen:
                    seen[column] += 1
                    unique.append(f"{column}_{seen[column]}")
                else:
                    seen[column] = 1
                    unique.append(column)
            return unique
        # No header: positional names derived from the first content line.
        count = len(line.split(delimiter)) if line else 0
        return [f"COL_{index + 1}" for index in range(count)]

    def _load_fixed_width_layout(
        self, layout_csv: Path | None, sample
    ) -> tuple[FixedWidthLayout | None, list[str]]:
        """Load the user layout, or return a warning when it is missing."""
        if layout_csv is None:
            return None, ["fixed-width file requires a layout CSV."]
        try:
            layout = load_layout_csv(layout_csv)
        except LayoutLoadError as exc:
            return None, [f"layout could not be loaded: {exc}"]
        return layout, validate_layout_against_sample(layout, sample.lines)

    # ==================================================================
    # Schema inference.
    # ==================================================================
    def _infer_delimited_schema(
        self, sample, delimiter: str, header: bool | None,
        preamble_lines: int = 0,
    ) -> dict[str, str]:
        """Infer column dtypes from the decoded DATA region only.

        The region is re-read from the already-decoded sampler lines, so
        a lead paragraph and non-UTF-8 encodings are handled correctly.
        """
        if header is None:
            return {}
        data_text = "\n".join(sample.lines[preamble_lines:])
        if not data_text.strip():
            return {}
        try:
            frame = pl.read_csv(
                io.StringIO(data_text),
                separator=delimiter,
                has_header=header,
                infer_schema_length=_INFER_SCHEMA_LENGTH,
            )
        except Exception:
            return {}
        return {name: str(dtype) for name, dtype in frame.schema.items()}

    def _infer_fixed_width_schema(
        self, lines: list[str], layout: FixedWidthLayout
    ) -> dict[str, str]:
        rows = []
        for line in lines:
            if len(line) >= layout.width:
                rows.append(
                    {
                        column.field: line[column.start : column.end].strip()
                        for column in layout.columns
                    }
                )
        if not rows:
            return {}
        frame = pl.DataFrame(rows)
        return {name: str(dtype) for name, dtype in frame.schema.items()}

    # ==================================================================
    # Metadata & confidence.
    # ==================================================================
    def _build_sample_metadata(
        self, sample, delimiter, delim_stats, structure, preamble_lines
    ) -> dict:
        non_empty = [line for line in sample.lines if line.strip()]
        lengths = [len(line) for line in non_empty]
        common, occurrences = Counter(lengths).most_common(1)[0] if lengths else (0, 0)
        return {
            "bytes_read": sample.bytes_read,
            "lines_sampled": sample.line_count,
            "encoding_method": sample.encoding_method,
            "delimiter": delimiter,
            "delimiter_stats": delim_stats,
            "preamble_lines": preamble_lines,
            "data_start_line": preamble_lines + 1,
            "length_mode": common,
            "length_consistency": (
                round(occurrences / len(lengths), 4) if lengths else 0.0
            ),
            "structure": structure.value,
        }

    def _score_confidence(
        self,
        file_format: FileFormat,
        structure: StructureType,
        delimiter: str | None,
        delimiter_consistency: float,
        encoding_method: str,
        layout_present: bool,
        header: bool | None,
        preamble_lines: int = 0,
    ) -> float:
        """Deterministic 0..1 confidence; see module docstring."""
        score = 1.0

        if file_format is FileFormat.UNSUPPORTED:
            return 0.0

        if file_format is FileFormat.DELIMITED:
            if delimiter is None:
                score -= 0.5  # single unstructured column
            else:
                score -= (1.0 - delimiter_consistency) * 0.5

        if file_format is FileFormat.FIXED_WIDTH and not layout_present:
            score -= 0.4  # needs a user-provided layout

        if structure is not StructureType.FLAT:
            score -= 0.3  # flagged structures always get human review

        if encoding_method == "latin-1-fallback":
            score -= 0.1

        if header is None:
            score -= 0.2

        # A non-trivial preamble deserves human sign-off: a wrong skip
        # count would silently drop real rows. One short preamble line is
        # predictable enough to auto-parse; two or more always review.
        if preamble_lines > 0:
            score -= 0.15 * min(preamble_lines, 2)

        return round(max(0.0, min(1.0, score)), 2)
