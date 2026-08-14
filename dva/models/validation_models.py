"""Validation configuration and report models.

These mirror the analysis already proven in the reference engine.py /
storelistvalidation.py but are expressed against *structured Parquet
datasets* only (Boundary 9): the Validator never parses raw retailer
formats. Config is intentionally source-format agnostic and kept
separate from the ingestion configuration (Engineering rule 13).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class PriceType(str, Enum):
    """Whether the price column holds a total or a per-unit price."""
    TOTAL_PRICE = "total_price"
    UNIT_PRICE = "unit_price"


class UnitType(str, Enum):
    """What the units column actually counts: quantity or weight."""
    QTY = "qty"
    WEIGHT = "weight"


@dataclass(slots=True)
class ColumnMapping:
    """Column names (in the Parquet dataset) used for each semantic role.

    Mirrors the mapping UX from the reference tool; ``None`` means the
    role is not used. ``weighted_units_col`` overrides the raw units
    column per row (weighted value when present, else raw units) and
    applies to BOTH the store-level and item-level analyses.
    """

    store_col: str | None = None
    units_col: str | None = None
    price_col: str | None = None
    upc_col: str | None = None
    desc_col: str | None = None
    weighted_units_col: str | None = None
    units_type: UnitType = UnitType.QTY

    def required_for(self, analysis: str) -> list[str]:
        """Columns that must be set for a given analysis to run."""
        requirements: dict[str, list[str]] = {
            "store": ["store_col", "units_col", "price_col"],
            "sales": ["store_col"],
            "upc": ["upc_col", "desc_col", "units_col", "price_col"],
        }
        return [name for name in requirements.get(analysis, []) if getattr(self, name) is None]


@dataclass(slots=True)
class ValidationConfig:
    """User-selected analyses and per-side column mapping (BAU vs TEST).

    ``sales_analysis`` is the store-list / missing-stores check from the
    reference tool (storelistvalidation_main): a BAU dataset is compared
    with a TEST dataset for missing store numbers.
    """

    store_analysis: bool = False
    sales_analysis: bool = False
    upc_analysis: bool = False

    bau: ColumnMapping = field(default_factory=ColumnMapping)
    test: ColumnMapping = field(default_factory=ColumnMapping)

    price_type_bau: PriceType = PriceType.TOTAL_PRICE
    price_type_test: PriceType = PriceType.TOTAL_PRICE

    # Implied decimals: divide the raw value by 100 before aggregating.
    implied_dollars_bau: bool = False
    implied_units_bau: bool = False
    implied_dollars_test: bool = False
    implied_units_test: bool = False

    # ------------------------------------------------------------------
    def enabled_analyses(self) -> list[str]:
        """Names of the analyses the user turned on."""
        return [
            name
            for name, flag in (
                ("store", self.store_analysis),
                ("sales", self.sales_analysis),
                ("upc", self.upc_analysis),
            )
            if flag
        ]

    def missing_columns(self) -> dict[str, list[str]]:
        """Map of analysis -> missing required columns (empty when valid)."""
        missing: dict[str, list[str]] = {}
        for analysis in self.enabled_analyses():
            missing_bau = self.bau.required_for(analysis)
            missing_test = self.test.required_for(analysis)
            combined = sorted(set(missing_bau + missing_test))
            if combined:
                missing[analysis] = combined
        return missing

    def validate(self) -> None:
        """Raise ValueError when an enabled analysis lacks required columns."""
        missing = self.missing_columns()
        if missing:
            details = "; ".join(
                f"{analysis}: missing {', '.join(cols)}"
                for analysis, cols in missing.items()
            )
            raise ValueError(f"Validation config incomplete: {details}.")

    def to_json_dict(self) -> dict:
        """Serialisable form (kept small, no data frames)."""
        bau = asdict(self.bau)
        bau["units_type"] = bau["units_type"].value
        test = asdict(self.test)
        test["units_type"] = test["units_type"].value
        return {
            "store_analysis": self.store_analysis,
            "sales_analysis": self.sales_analysis,
            "upc_analysis": self.upc_analysis,
            "bau": bau,
            "test": test,
            "price_type_bau": self.price_type_bau.value,
            "price_type_test": self.price_type_test.value,
            "implied_dollars_bau": self.implied_dollars_bau,
            "implied_units_bau": self.implied_units_bau,
            "implied_dollars_test": self.implied_dollars_test,
            "implied_units_test": self.implied_units_test,
        }


@dataclass(slots=True)
class ValidationReport:
    """Result of one validation run across the enabled analyses.

    ``artifacts`` maps a short name (store / sales / upc) to a list of
    generated report file paths (Excel etc., produced by the reporting
    service). ``metrics`` holds timing counters for the UI.
    """

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    bau_dataset: str = ""
    test_dataset: str = ""
    artifacts: dict[str, list[Path]] = field(default_factory=dict)
    metrics: dict[str, str] = field(default_factory=dict)

    def add_artifact(self, analysis: str, path: Path) -> None:
        """Attach a generated report file to this validation run."""
        self.artifacts.setdefault(analysis, []).append(path)
