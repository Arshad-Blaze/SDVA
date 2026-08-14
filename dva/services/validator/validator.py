"""Validator (Boundary 9): analyses over structured Parquet datasets.

The Validator deliberately never parses retailer raw formats: its input
is a COMPLETE dataset directory (validated Parquet file per Boundary 6).
All reads go through ``pl.scan_parquet`` so memory stays flat no matter
how large a dataset is; results are collected streaming.

Three analyses mirror the reference tool (storelistvalidation.py /
engine.py) against BAU and TEST datasets:

  - store : store-list comparison. Normalised (strip, lowercase) unique
    store sets on each side, anti-joined to list missing numbers plus
    record/unique counts.
  - sales : per-store totals: record count, units and dollars (units x
    price, honouring price type and implied decimals), plus the reference
    tool's full-join comparison with difference / percentage columns and
    top-five / bottom-five store tabs.
  - upc   : per-UPC totals (units, dollars) on each side plus missing UCP
    counts and uniques.

Money semantics from ValidationConfig: ``price_type`` decides whether the
price column is already a line total or a per-unit price; ``implied_*``
divides the raw value by 100 (e.g. cents stored as integers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from models.validation_models import (
    ColumnMapping,
    PriceType,
    ValidationConfig,
)


@dataclass(frozen=True, slots=True)
class ValidationRunResult:
    """Everything produced by one validation run, UI-friendly."""
    metrics: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    frames: dict[str, pl.DataFrame] = field(default_factory=dict)


class ValidationError(Exception):
    """Raised when a dataset or mapping cannot be validated."""


# ---------------------------------------------------------------------
# Column resolution helpers.
# ---------------------------------------------------------------------
def _mapping_for(config: ValidationConfig, side: str) -> ColumnMapping:
    return config.bau if side == "bau" else config.test


def _effective_units(
    config: ValidationConfig,
    side: str,
    units_col: str | None,
    weighted_col: str | None = None,
) -> pl.Expr:
    """Per-row units, honouring weighted fallback and implied decimals.

    A weighted column, when set, overrides the raw units per row (weighted
    value when non-null, else raw units); implied units then divide the
    effective value by 100. Matches the reference item-level ``Final_Units``
    semantics.
    """
    if units_col:
        units = pl.col(units_col).cast(pl.Float64, strict=False).fill_null(0)
    else:
        units = pl.lit(1, dtype=pl.Int8)
    if weighted_col:
        weighted = pl.col(weighted_col).cast(pl.Float64, strict=False)
        units = pl.when(weighted.is_not_null()).then(weighted).otherwise(units)
    if getattr(config, f"implied_units_{side}"):
        units = units / 100
    return units


def _dollars_expr(
    config: ValidationConfig,
    side: str,
    units_col: str | None,
    price_col: str | None,
    weighted_col: str | None = None,
) -> pl.Expr:
    """Per-row dollar value, honouring price type and implied decimals.

    Mirrors the reference engine's store-level order: price is scaled by
    implied_dollars first, then multiplied by the (weighted-aware,
    implied-scaled) units when the price column holds a per-unit price.
    """
    if price_col is None:
        raise ValidationError(f"{side} price column is not mapped.")
    price = pl.col(price_col).cast(pl.Float64, strict=False).fill_null(0)
    if getattr(config, f"implied_dollars_{side}"):
        price = price / 100
    dollars = (
        price * _effective_units(config, side, units_col, weighted_col)
        if getattr(config, f"price_type_{side}") is PriceType.UNIT_PRICE
        else price
    )
    return dollars


def _scan(path: str | Path) -> pl.LazyFrame:
    """Lazy scan of one validated dataset parquet file."""
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Dataset not found: {path}")
    return pl.scan_parquet(path)


def _normalised_keys(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Unique, stripped, lowercased key column (store numbers, UPCs)."""
    return (
        lf.select(pl.col(column).drop_nulls())
        .select(pl.col(column).cast(pl.String).str.strip_chars().str.to_lowercase())
        .select(pl.col(column).alias("key"))
        .unique()
    )


# ---------------------------------------------------------------------
# Store-list analysis.
# ---------------------------------------------------------------------
def _store_analysis(
    bau_path: str | Path,
    test_path: str | Path,
    cfg: ValidationConfig,
) -> tuple[list[tuple[str, str]], dict[str, pl.DataFrame]]:
    bau_col = cfg.bau.store_col
    test_col = cfg.test.store_col
    if not bau_col or not test_col:
        raise ValidationError("store analysis needs both store columns mapped.")

    bau_lf = _scan(bau_path)
    test_lf = _scan(test_path)

    bau_unique = _normalised_keys(bau_lf, bau_col)
    test_unique = _normalised_keys(test_lf, test_col)
    missing_in_test = bau_unique.join(test_unique, on="key", how="anti")
    missing_in_bau = test_unique.join(bau_unique, on="key", how="anti")

    bau_keys = bau_unique.collect(engine="streaming")
    test_keys = test_unique.collect(engine="streaming")
    missing_test_df = missing_in_test.collect(engine="streaming")
    missing_bau_df = missing_in_bau.collect(engine="streaming")

    bau_count = bau_lf.select(pl.len()).collect().item()
    test_count = test_lf.select(pl.len()).collect().item()

    missing_in_test_list = sorted(missing_test_df["key"].to_list())
    missing_in_bau_list = sorted(missing_bau_df["key"].to_list())

    metrics = [
        ("Total number of records (BAU)", str(bau_count)),
        ("Total number of unique stores in BAU", str(bau_keys.height)),
        ("Total number of records (TEST)", str(test_count)),
        ("Total number of unique stores in TEST", str(test_keys.height)),
        ("Stores present in TEST (missing in BAU)", str(len(missing_in_bau_list))),
        ("Stores present in BAU (missing in TEST)", str(len(missing_in_test_list))),
        ("Missing store numbers in BAU", ", ".join(missing_in_bau_list)),
        ("Missing store numbers in TEST", ", ".join(missing_in_test_list)),
    ]
    frames = {
        "missing_in_bau": pl.DataFrame(
            {"store": pl.Series(missing_in_bau_list, dtype=pl.String)}
        ),
        "missing_in_test": pl.DataFrame(
            {"store": pl.Series(missing_in_test_list, dtype=pl.String)}
        ),
    }
    return metrics, frames


# ---------------------------------------------------------------------
# Per-store sales analysis.
# ---------------------------------------------------------------------
def _sales_comparison(
    bau: pl.DataFrame, test: pl.DataFrame, units_type: str
) -> pl.DataFrame:
    """Full-join per-store comparison, mirroring the reference summary.

    Columns follow the reference ``storelevelvalidation``: both sides'
    units/dollars, the differences and the percentage differences (0/0
    guarded to -100 exactly as the tool does). ``units_type`` (qty/weight)
    labels what the units column counts.
    """
    bau_side = bau.select(
        pl.col("store").cast(pl.String).alias("STORE_NUMBER"),
        pl.col("units").alias("BAU UNITS"),
        pl.col("dollars").alias("BAU TOTAL DOLLARS"),
    )
    test_side = test.select(
        pl.col("store").cast(pl.String).alias("STORE_NUMBER"),
        pl.col("units").alias("TEST UNITS_SOLD"),
        pl.col("dollars").alias("TEST TOTAL_DOLLARS"),
    )
    return (
        bau_side.join(test_side, on="STORE_NUMBER", how="full", coalesce=True)
        .with_columns([pl.lit(units_type).alias("Units Type")])
        .with_columns(
            [
                pl.col("BAU UNITS").fill_null(0),
                pl.col("BAU TOTAL DOLLARS").fill_null(0),
                pl.col("TEST UNITS_SOLD").fill_null(0).round(2),
                pl.col("TEST TOTAL_DOLLARS").fill_null(0).round(2),
            ]
        )
        .with_columns(
            [
                (pl.col("BAU UNITS") - pl.col("TEST UNITS_SOLD")).alias(
                    "UNITS DIFFERENCE"
                ),
                (pl.col("BAU TOTAL DOLLARS") - pl.col("TEST TOTAL_DOLLARS"))
                .round(2)
                .alias("DOLLAR DIFFERENCE"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("BAU UNITS") != 0)
                .then(
                    (pl.col("UNITS DIFFERENCE") / pl.col("BAU UNITS") * 100).round(2)
                )
                .otherwise(-100)
                .alias("Unit % Difference"),
                pl.when(pl.col("BAU TOTAL DOLLARS") != 0)
                .then(
                    (pl.col("DOLLAR DIFFERENCE") / pl.col("BAU TOTAL DOLLARS") * 100)
                    .round(2)
                )
                .otherwise(-100)
                .alias("Dollar % Difference"),
            ]
        )
        .sort("Dollar % Difference", descending=True)
    )


def _sales_analysis(
    bau_path: str | Path,
    test_path: str | Path,
    cfg: ValidationConfig,
) -> tuple[list[tuple[str, str]], dict[str, pl.DataFrame]]:
    bau_col = cfg.bau.store_col
    test_col = cfg.test.store_col
    if not bau_col or not test_col:
        raise ValidationError("sales analysis needs both store columns mapped.")

    def per_store(lf: pl.LazyFrame, store_col: str, side: str) -> pl.LazyFrame:
        mapping = cfg.bau if side == "bau" else cfg.test
        expr: list[pl.Expr] = [pl.len().alias("records")]
        # Totals for units only when a units/weighted column is mapped.
        if mapping.price_col:
            dollars = _dollars_expr(
                cfg, side, mapping.units_col, mapping.price_col,
                mapping.weighted_units_col,
            )
            expr.append(dollars.fill_null(0).sum().alias("dollars"))
        if mapping.units_col or mapping.weighted_units_col:
            expr.append(
                _effective_units(
                    cfg, side, mapping.units_col, mapping.weighted_units_col
                )
                .sum()
                .alias("units")
            )
            expr.append(pl.lit(mapping.units_type.value).alias("unit_type"))
        return (
            lf.group_by(pl.col(store_col).cast(pl.String).alias("store"))
            .agg(expr)
            .sort("store")
        )

    bau = per_store(_scan(bau_path), bau_col, "bau").collect(engine="streaming")
    test = per_store(_scan(test_path), test_col, "test").collect(engine="streaming")
    bau_units_type = cfg.bau.units_type.value
    test_units_type = cfg.test.units_type.value
    comparison = _sales_comparison(bau, test, bau_units_type)

    total_bau_dollars = comparison["BAU TOTAL DOLLARS"].sum()
    total_test_dollars = comparison["TEST TOTAL_DOLLARS"].sum()
    dollar_diff = total_bau_dollars - total_test_dollars
    dollar_pct = (
        round(dollar_diff / total_bau_dollars * 100, 2)
        if total_bau_dollars
        else float("nan")
    )
    total_bau_units = comparison["BAU UNITS"].sum()
    total_test_units = comparison["TEST UNITS_SOLD"].sum()
    unit_diff = total_bau_units - total_test_units
    unit_pct = (
        round(unit_diff / total_bau_units * 100, 2)
        if total_bau_units
        else float("nan")
    )
    unique_store_count = (
        comparison.select(
            pl.col("STORE_NUMBER")
            .cast(pl.String)
            .str.strip_chars()
            .alias("sn")
        )
        .filter(pl.col("sn") != "")
        .select(pl.col("sn").drop_nulls().alias("sn"))
        .unique()
        .height
    )

    def _top_bottom(sort_col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        ranked = comparison.sort(sort_col, descending=True)
        return ranked.head(5), ranked.tail(5)

    top_5_sales, bottom_5_sales = _top_bottom("Dollar % Difference")
    top_5_units, bottom_5_units = _top_bottom("Unit % Difference")

    metrics = [
        ("Unique Store Count", str(unique_store_count)),
        ("Total BAU Sales", str(total_bau_dollars)),
        ("Total Test Sales", str(total_test_dollars)),
        ("Total Dollar Difference", str(dollar_diff)),
        ("Dollar % Difference", str(dollar_pct)),
        ("Total BAU Units", str(total_bau_units)),
        ("Total Test Units", str(total_test_units)),
        ("Total Unit Difference", str(unit_diff)),
        ("Units % Difference", str(unit_pct)),
    ]
    frames = {
        "bau": bau,
        "test": test,
        "comparison": comparison,
        "top_5_sales": top_5_sales,
        "bottom_5_sales": bottom_5_sales,
        "top_5_units": top_5_units,
        "bottom_5_units": bottom_5_units,
    }
    return metrics, frames


# ---------------------------------------------------------------------
# Per-UPC analysis.
# ---------------------------------------------------------------------
def _upc_analysis(
    bau_path: str | Path,
    test_path: str | Path,
    cfg: ValidationConfig,
) -> tuple[list[tuple[str, str]], dict[str, pl.DataFrame]]:
    bau_col = cfg.bau.upc_col
    test_col = cfg.test.upc_col
    if not bau_col or not test_col:
        raise ValidationError("upc analysis needs both UPC columns mapped.")

    def per_upc(lf: pl.LazyFrame, upc_col: str) -> pl.LazyFrame:
        return (
            lf.select(pl.col(upc_col).drop_nulls())
            .select(
                pl.col(upc_col).cast(pl.String).str.strip_chars().alias("upc")
            )
            .unique()
        )

    bau_lf = per_upc(_scan(bau_path), bau_col)
    test_lf = per_upc(_scan(test_path), test_col)
    bau_upc = bau_lf.collect(engine="streaming")
    test_upc = test_lf.collect(engine="streaming")
    missing_in_test = bau_lf.join(test_lf, on="upc", how="anti").collect(engine="streaming")
    missing_in_bau = test_lf.join(bau_lf, on="upc", how="anti").collect(engine="streaming")

    metrics = [
        ("Unique UPCs in BAU", str(bau_upc.height)),
        ("Unique UPCs in TEST", str(test_upc.height)),
        ("UPCs missing in TEST", str(missing_in_test.height)),
        ("UPCs missing in BAU", str(missing_in_bau.height)),
    ]
    frames = {
        "missing_in_bau": missing_in_bau,
        "missing_in_test": missing_in_test,
    }
    return metrics, frames


# ---------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------
def validate(
    bau_path: str | Path,
    test_path: str | Path,
    config: ValidationConfig,
) -> ValidationRunResult:
    """Run every enabled analysis; returns metrics + result frames."""
    config.validate()
    result = ValidationRunResult()
    for analysis in config.enabled_analyses():
        if analysis == "store":
            metrics, frames = _store_analysis(bau_path, test_path, config)
        elif analysis == "sales":
            metrics, frames = _sales_analysis(bau_path, test_path, config)
        elif analysis == "upc":
            metrics, frames = _upc_analysis(bau_path, test_path, config)
        else:
            raise ValidationError(f"Unknown analysis: {analysis}.")
        result.metrics[analysis] = metrics
        result.frames[analysis] = frames
    return result