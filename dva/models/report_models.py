"""Item-level report configuration and result models.

The reporting service (services/reporting/report_service.py) turns two
COMPLETE Parquet datasets (BAU vs TEST) into the item-level comparison
artifacts proven in the reference engine's ``itemlevel_validation()``:
per-UPC|DESC summaries, a joined comparison, a Present-In summary and a
metrics table. Config stays dataset-agnostic and is kept separate from the
ingestion configuration (Engineering rule 13).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from models.validation_models import ColumnMapping, PriceType


@dataclass(slots=True)
class ItemValidationConfig:
    """Column mapping + money semantics for one item-level report run.

    Both sides need a UPC column, a description column, a units column and
    a price column. ``weighted_units_col`` (optional) overrides the raw
    units column when present; ``units_type`` records whether the units
    column counts quantity (qty) or weight (weight). Price handling
    follows ``PriceType`` and the ``implied_dollars_*`` flags exactly like
    the Validator.
    """

    bau: ColumnMapping = field(default_factory=ColumnMapping)
    test: ColumnMapping = field(default_factory=ColumnMapping)

    price_type_bau: PriceType = PriceType.TOTAL_PRICE
    price_type_test: PriceType = PriceType.TOTAL_PRICE
    implied_dollars_bau: bool = False
    implied_dollars_test: bool = False

    output_prefix: str = "item_validation"

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError when either side lacks the required columns."""
        required = ["upc_col", "desc_col", "units_col", "price_col"]
        problems = []
        for side in ("bau", "test"):
            mapping = getattr(self, side)
            for role in required:
                if getattr(mapping, role) is None:
                    problems.append(f"{side}.{role}")
        if problems:
            raise ValueError(
                "Item validation config incomplete: missing "
                + ", ".join(problems) + "."
            )


@dataclass(frozen=True, slots=True)
class ItemReportResult:
    """Output of one report run: small UI frames + written artifact paths.

    Only the small frames are held in memory; the full comparison stays on
    disk as Parquet (Engineering rule 3 - no large DataFrames in session
    state).
    """

    metrics: object = None  # pl.DataFrame: metric -> value
    top_5_sales: object = None
    bottom_5_sales: object = None
    top_5_units: object = None
    bottom_5_units: object = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    elapsed_seconds: float = 0.0
