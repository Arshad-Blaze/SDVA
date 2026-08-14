"""Report service: item-level BAU vs TEST comparison reports.

Consumes COMPLETE Parquet datasets only (Boundary 9 output, never raw
formats) and reproduces the artifacts of the reference engine's
``itemlevel_validation()``:

  {prefix}_bau_summary.parquet   per-UPC|DESC totals for BAU
  {prefix}_test_summary.parquet  per-UPC|DESC totals for TEST
  {prefix}_comparison.parquet    joined detail with "Present In"
  {prefix}_summary.parquet       Present-In groups + Grand Total

plus in-memory metrics and top/bottom-5 frames for the UI. All reads are
lazy (``pl.scan_parquet``); report-sized aggregates are streamed to disk
via ``sink_parquet`` so memory stays flat regardless of dataset size.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from models.report_models import ItemReportResult, ItemValidationConfig
from models.validation_models import PriceType

_REPORT_COLUMNS = [
    "UPC|DESC",
    "UPC",
    "BAU Description",
    "TEST Description",
    "BAU Units",
    "BAU Dollars",
    "TEST Units",
    "TEST Dollars",
]


class ReportError(Exception):
    """Raised when a dataset or mapping cannot be reported on."""


def _scan(path: str | Path) -> pl.LazyFrame:
    """Lazy scan of one validated dataset parquet file."""
    path = Path(path)
    if not path.exists():
        raise ReportError(f"Dataset not found: {path}")
    return pl.scan_parquet(path)


def _dollars_expr(
    config: ItemValidationConfig,
    side: str,
    units_col: str,
    price_col: str,
    weighted_col: str | None = None,
) -> pl.Expr:
    """Per-row dollar value, honouring price type and implied decimals.

    With a weighted column present, the Unit-Price multiplication uses the
    effective units (weighted value when non-null, else raw units) so item
    dollars stay consistent with store-level sales dollars.
    """
    implied = config.implied_dollars_bau if side == "bau" else config.implied_dollars_test
    price = pl.col(price_col).cast(pl.Float64, strict=False)
    units: pl.Expr = (
        pl.col(units_col).cast(pl.Float64, strict=False)
        if units_col
        else pl.lit(1.0, dtype=pl.Float64)
    )
    if weighted_col:
        weighted = pl.col(weighted_col).cast(pl.Float64, strict=False)
        units = pl.when(weighted.is_not_null()).then(weighted).otherwise(units)
    dollars = (
        price * units
        if getattr(config, f"price_type_{side}") is PriceType.UNIT_PRICE
        else price
    )
    if implied:
        dollars = dollars / 100
    return dollars


def _item_summary(
    lf: pl.LazyFrame,
    mapping: object,
    config: ItemValidationConfig,
    side: str,
) -> pl.LazyFrame:
    """Per-UPC|DESC units and dollars for one side of the comparison."""
    weighted = mapping.weighted_units_col
    select_expr = [
        pl.col(mapping.upc_col).cast(pl.String).str.strip_chars().alias("UPC"),
        pl.col(mapping.desc_col)
        .fill_null("")
        .cast(pl.String)
        .str.strip_chars()
        .alias("Product_Description"),
        pl.col(mapping.units_col)
        .cast(pl.Float64, strict=False)
        .fill_null(0)
        .alias("Units"),
        _dollars_expr(config, side, mapping.units_col, mapping.price_col, mapping.weighted_units_col)
            .fill_null(0)
            .alias("Totalprice"),
    ]
    if weighted:
        select_expr.append(
            pl.col(weighted).cast(pl.Float64, strict=False).alias("Weighted_Units")
        )

    units_total: pl.Expr = pl.col("Units")
    if weighted:
        units_total = (
            pl.when(pl.col("Weighted_Units").is_not_null())
            .then(pl.col("Weighted_Units"))
            .otherwise(pl.col("Units"))
        )

    return (
        lf.select(select_expr)
        .with_columns(
            (pl.col("UPC") + pl.lit("|") + pl.col("Product_Description")).alias(
                "UPC|DESC"
            )
        )
        .group_by("UPC|DESC")
        .agg(
            [
                pl.col("UPC").first().alias("UPC"),
                pl.col("Product_Description").first().alias("Product_Description"),
                units_total.sum().alias("TOTAL_UNITS"),
                pl.col("Totalprice").sum().alias("TOTAL_DOLLARS"),
            ]
        )
        .with_columns(pl.lit(mapping.units_type.value).alias("Unit Type"))
    )


def _build_comparison(
    bau_lf: pl.LazyFrame, test_lf: pl.LazyFrame, units_type: str
) -> pl.LazyFrame:
    """Left + anti join, classify presence, add difference columns."""
    bau_compare = bau_lf.rename(
        {
            "Product_Description": "BAU Description",
            "TOTAL_UNITS": "BAU Units",
            "TOTAL_DOLLARS": "BAU Dollars",
        }
    )
    test_compare = test_lf.rename(
        {
            "Product_Description": "TEST Description",
            "TOTAL_UNITS": "TEST Units",
            "TOTAL_DOLLARS": "TEST Dollars",
        }
    )

    left = bau_compare.join(test_compare, on="UPC|DESC", how="left").with_columns(
        [pl.col("TEST Units").fill_null(0), pl.col("TEST Dollars").fill_null(0)]
    )
    right_only = test_compare.join(bau_compare, on="UPC|DESC", how="anti").with_columns(
        [
            pl.lit(None, dtype=pl.String).alias("BAU Description"),
            pl.lit(0.0).alias("BAU Units"),
            pl.lit(0.0).alias("BAU Dollars"),
        ]
    )
    left = left.select(_REPORT_COLUMNS)
    right_only = right_only.select(_REPORT_COLUMNS)

    return pl.concat([left, right_only]).with_columns(
        [
            pl.lit(units_type).alias("Unit Type"),
            pl.when(pl.col("BAU Description").is_null())
            .then(pl.lit("Present only in TEST"))
            .otherwise(
                pl.when(pl.col("TEST Description").is_null())
                .then(pl.lit("Present only in BAU"))
                .otherwise(pl.lit("Present in Both"))
            )
            .alias("Present In"),
            (
                pl.col("BAU Units").fill_null(0) - pl.col("TEST Units").fill_null(0)
            ).alias("Units Difference"),
            (
                pl.col("BAU Dollars").fill_null(0)
                - pl.col("TEST Dollars").fill_null(0)
            ).alias("Dollar Difference"),
        ]
    )


def _build_present_in_summary(comparison: pl.LazyFrame) -> pl.LazyFrame:
    """Group the comparison by presence class plus a Grand Total row."""
    summary = comparison.group_by("Present In").agg(
        [
            pl.len().alias("Count"),
            pl.col("Unit Type").first().alias("Unit Type"),
            pl.col("Units Difference").sum().alias("Units Difference"),
            pl.col("Dollar Difference").sum().alias("Dollar Difference"),
        ]
    )
    grand_total = summary.select(
        [
            pl.lit("Grand Total").alias("Present In"),
            pl.col("Count").sum().alias("Count"),
            pl.col("Unit Type").first().alias("Unit Type"),
            pl.col("Units Difference").sum().alias("Units Difference"),
            pl.col("Dollar Difference").sum().alias("Dollar Difference"),
        ]
    )
    return pl.concat([summary, grand_total])


def _build_metrics(
    comparison: pl.LazyFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Collect the comparison into small UI frames.

    The comparison is UPC|DESC aggregate rows (not raw records), so
    collecting it here is bounded and mirrors the reference tool.
    """
    comp = comparison.collect(engine="streaming")
    bau_mask = comp["BAU Description"].is_not_null()
    test_mask = comp["TEST Description"].is_not_null()

    total_bau_dollars = comp["BAU Dollars"].sum()
    total_test_dollars = comp["TEST Dollars"].sum()
    total_bau_units = comp["BAU Units"].sum()
    total_test_units = comp["TEST Units"].sum()

    metrics = pl.DataFrame(
        {
            "Metric": [
                "Unique UPC BAU",
                "Unique UPC TEST",
                "New UPC indicator",
                "Total Test Sales",
                "Total BAU Sales",
                "Dollar Difference",
                "Total Units Test",
                "Total Units BAU",
                "Units Difference",
            ],
            "Value": pl.Series(
                [
                    int(bau_mask.sum()),
                    int(test_mask.sum()),
                    int(bau_mask.sum() - test_mask.sum()),
                    round(total_test_dollars, 2),
                    round(total_bau_dollars, 2),
                    round(total_bau_dollars - total_test_dollars, 2),
                    round(total_test_units, 2),
                    round(total_bau_units, 2),
                    round(total_bau_units - total_test_units, 2),
                ],
                dtype=pl.Float64,
            ),
        }
    )

    sales_cols = [
        "UPC|DESC",
        "UPC",
        "BAU Description",
        "TEST Description",
        "BAU Dollars",
        "TEST Dollars",
        "Dollar Difference",
    ]
    units_cols = [
        "UPC|DESC",
        "UPC",
        "BAU Description",
        "TEST Description",
        "BAU Units",
        "TEST Units",
        "Units Difference",
    ]
    top_5_sales = comp.select(sales_cols).sort("Dollar Difference", descending=True).head(5)
    bottom_5_sales = comp.select(sales_cols).sort("Dollar Difference").head(5)
    top_5_units = comp.select(units_cols).sort("Units Difference", descending=True).head(5)
    bottom_5_units = comp.select(units_cols).sort("Units Difference").head(5)

    return metrics, top_5_sales, bottom_5_sales, top_5_units, bottom_5_units


def generate_reports(
    bau_path: str | Path,
    test_path: str | Path,
    config: ItemValidationConfig,
    output_dir: str | Path,
) -> ItemReportResult:
    """Write the four report artifacts and return the UI frames.

    Args:
        bau_path: parquet path of the BAU dataset.
        test_path: parquet path of the TEST dataset.
        config: column mapping + money semantics.
        output_dir: directory to write the ``.parquet`` artifacts into.
    """
    config.validate()
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output_prefix

    bau_lf = _item_summary(_scan(bau_path), config.bau, config, "bau")
    test_lf = _item_summary(_scan(test_path), config.test, config, "test")
    comparison = _build_comparison(bau_lf, test_lf, config.bau.units_type.value)
    summary = _build_present_in_summary(comparison)

    artifacts = {
        "bau_summary": output_dir / f"{prefix}_bau_summary.parquet",
        "test_summary": output_dir / f"{prefix}_test_summary.parquet",
        "comparison": output_dir / f"{prefix}_comparison.parquet",
        "summary": output_dir / f"{prefix}_summary.parquet",
    }
    bau_lf.sink_parquet(artifacts["bau_summary"])
    test_lf.sink_parquet(artifacts["test_summary"])
    comparison.sink_parquet(artifacts["comparison"])
    summary.sink_parquet(artifacts["summary"])

    metrics, top_5_sales, bottom_5_sales, top_5_units, bottom_5_units = _build_metrics(
        comparison
    )
    return ItemReportResult(
        metrics=metrics,
        top_5_sales=top_5_sales,
        bottom_5_sales=bottom_5_sales,
        top_5_units=top_5_units,
        bottom_5_units=bottom_5_units,
        artifacts=artifacts,
        elapsed_seconds=round(time.time() - start, 2),
    )
