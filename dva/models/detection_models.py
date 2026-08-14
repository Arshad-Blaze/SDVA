"""Detection, layout, and user-approval models.

The Detector (services/ingestion/detector.py) produces a
``DetectionResult``; the user reviews it and the pipeline persists the
accepted/edited values as an ``ApprovedConfig`` (Boundary 4).

Structure handling decision for Phase 1 (confirmed with the user):
  - MULTILINE + delimited  -> fully supported (quoted newlines and
    explicit continuation markers).
  - MULTILINE + fixed-width, RECORD_TYPED -> *detected and flagged*,
    actual parsing is deferred to a later phase. Such files must never
    be silently mis-parsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FileFormat(str, Enum):
    """Top-level source format."""
    DELIMITED = "delimited"
    FIXED_WIDTH = "fixed_width"
    UNSUPPORTED = "unsupported"


class StructureType(str, Enum):
    """How logical records map to physical lines in the source file."""
    FLAT = "flat"                    # one record per physical line
    MULTILINE = "multiline"          # a record spans multiple lines
    RECORD_TYPED = "record_typed"    # interleaved header/detail records


class ApprovalDecision(str, Enum):
    """User decisions when reviewing a DetectionResult (03, section 5)."""
    ACCEPT = "accept"
    MODIFY = "modify"
    REPROCESS = "reprocess"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One fixed-width column: 0-based [start, end) character range."""
    field: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}.")
        if self.end <= self.start:
            raise ValueError(
                f"end ({self.end}) must be greater than start ({self.start})."
            )

    @property
    def width(self) -> int:
        """Column width in characters."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Plain-dict form for JSON persistence."""
        return {"field": self.field, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, payload: dict) -> "ColumnSpec":
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FixedWidthLayout:
    """Ordered, non-overlapping column layout for a fixed-width file."""
    columns: tuple[ColumnSpec, ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("A fixed-width layout needs at least one column.")

        # Columns must be sorted and must not overlap.
        previous_end = 0
        for column in self.columns:
            if column.start < previous_end:
                raise ValueError(
                    f"Overlapping columns: {column.field} starts at "
                    f"{column.start}, previous ended at {previous_end}."
                )
            previous_end = column.end

    @property
    def width(self) -> int:
        """Total line width covered by the layout."""
        return self.columns[-1].end

    def column_names(self) -> list[str]:
        """Canonical field names in layout order."""
        return [column.field for column in self.columns]

    @classmethod
    def from_specs(cls, specs: list[ColumnSpec]) -> "FixedWidthLayout":
        """Build a layout from a list of column specs (sorted first)."""
        ordered = sorted(specs, key=lambda spec: spec.start)
        return cls(tuple(ordered))

    def to_json_dict(self) -> dict:
        """Plain-dict form for JSON persistence."""
        return {"columns": [column.to_dict() for column in self.columns]}

    @classmethod
    def from_json_dict(cls, payload: dict) -> "FixedWidthLayout":
        """Rehydrate a layout from ``to_json_dict`` output."""
        return cls(
            tuple(ColumnSpec.from_dict(spec) for spec in payload["columns"])
        )


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Everything the Detector learns about a source file (Boundary 3).

    The object is immutable so it can be safely shared with the UI and
    the orchestrator without accidental mutation.
    """

    format: FileFormat
    structure_type: StructureType
    encoding: str = "cp1252"
    delimiter: str | None = None
    header_present: bool | None = None
    # Delimited: list[str] of column names. Fixed-width: FixedWidthLayout.
    layout: list[str] | FixedWidthLayout | None = None
    # Column name -> Polars/PyArrow dtype string, e.g. {"STORE": "Utf8"}.
    schema: dict[str, str] | None = None
    confidence: float = 0.0
    # Number of leading physical lines that are not part of the data
    # (cover paragraphs, banners, blank lines). Header/layout/schema are
    # inferred from the data region AFTER these lines.
    preamble_lines: int = 0
    sample_metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be within [0, 1], got {self.confidence}."
            )

    # ------------------------------------------------------------------
    def is_parseable(self) -> bool:
        """True when Phase 1 parsing can safely handle this file.

        Multiline delimited files are supported; record-typed files and
        multiline fixed-width files are flagged but not parsed yet.
        """
        if self.format is FileFormat.UNSUPPORTED:
            return False
        if self.structure_type is StructureType.RECORD_TYPED:
            return False
        if self.format is FileFormat.DELIMITED and self.delimiter is None:
            return False  # no confirmed delimiter -> needs human review
        if self.format is FileFormat.FIXED_WIDTH:
            if self.layout is None:
                return False  # layout CSV required
            if self.structure_type is StructureType.MULTILINE:
                return False
        return True

    def needs_approval(self, threshold: float = 0.8) -> bool:
        """True when the detection is uncertain enough to require review."""
        return self.confidence < threshold


@dataclass(slots=True)
class ApprovedConfig:
    """Persisted, user-approved parsing configuration (Boundary 4).

    Persisted outside Streamlit session state by the caller. Small and
    JSON-serialisable by design.
    """

    source_format: FileFormat
    delimiter: str | None = None
    encoding: str = "cp1252"
    header_present: bool = True
    columns: list[str] = field(default_factory=list)
    fixed_width_layout: FixedWidthLayout | None = None
    # Column name -> dtype override, applied during parsing (trust but
    # verify: cast with strict=False and let the writer re-check).
    schema_overrides: dict[str, str] = field(default_factory=dict)
    # Leading physical lines to skip entirely before the header/data.
    # Set by detection when a cover paragraph is found; editable on approval.
    skip_rows: int = 0

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError when the configuration is internally inconsistent."""
        if self.source_format is FileFormat.UNSUPPORTED:
            raise ValueError("Cannot approve an unsupported format.")

        if self.source_format is FileFormat.DELIMITED:
            if not self.delimiter:
                raise ValueError("Delimited files require a delimiter.")
            if len(self.delimiter) != 1:
                raise ValueError("Delimiter must be a single character.")
            if self.fixed_width_layout is not None:
                raise ValueError(
                    "A delimited file must not carry a fixed-width layout."
                )

        if self.source_format is FileFormat.FIXED_WIDTH:
            if self.fixed_width_layout is None:
                raise ValueError(
                    "Fixed-width files require an approved layout."
                )
            if self.delimiter is not None:
                raise ValueError(
                    "A fixed-width file must not carry a delimiter."
                )

        if self.skip_rows < 0:
            raise ValueError("skip_rows must be >= 0.")

    # ------------------------------------------------------------------
    def to_json_dict(self) -> dict:
        """Plain-dict form for registry persistence."""
        return {
            "source_format": self.source_format.value,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "header_present": self.header_present,
            "columns": list(self.columns),
            "fixed_width_layout": (
                self.fixed_width_layout.to_json_dict()
                if self.fixed_width_layout is not None
                else None
            ),
            "schema_overrides": dict(self.schema_overrides),
            "skip_rows": self.skip_rows,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "ApprovedConfig":
        """Rehydrate a config from ``to_json_dict`` output."""
        layout = None
        if payload.get("fixed_width_layout") is not None:
            layout = FixedWidthLayout.from_json_dict(payload["fixed_width_layout"])
        return cls(
            source_format=FileFormat(payload["source_format"]),
            delimiter=payload.get("delimiter"),
            encoding=payload.get("encoding", "cp1252"),
            header_present=payload.get("header_present", True),
            columns=payload.get("columns", []),
            fixed_width_layout=layout,
            schema_overrides=payload.get("schema_overrides", {}),
            skip_rows=payload.get("skip_rows", 0),
        )
