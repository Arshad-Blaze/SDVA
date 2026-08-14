# DVA Developer Guide

End-to-end processing engine: **MFT → detect → parse → Parquet → verify →
cleanup → validate → reports**, with per-file independent progression.

- **Framework**: Streamlit (thin UI) + Polars (processing)
- **Tests**: pytest, `tests/unit/` + `tests/integration/` (204 passing)
- **Language**: Python 3.12

> See `docs/COMPLIANCE_MATRIX.md` for a boundary-by-boundary status report and
> `docs/USER_GUIDE.md` for how operators use the tool.

---

## 1. Repository map

```
dva/
├── app.py                        Streamlit entry point (5 tabs, thin orchestration)
├── models/                       Pure dataclasses, no IO — the contracts
│   ├── file_models.py            FileStatus lifecycle + transitions, SourceFile
│   ├── detection_models.py       DetectionResult, ApprovedConfig, FixedWidthLayout, ColumnSpec
│   ├── dataset_models.py         DatasetMetadata, dataset_directory()
│   ├── validation_models.py      ValidationConfig, ColumnMapping, PriceType, UnitType
│   └── report_models.py          ItemValidationConfig, ItemReportResult
├── services/
│   ├── mft/                      Boundary 1 — file mover (Local + SFTP transport)
│   │   ├── transport.py          LocalTransport / SftpTransport (download abstraction)
│   │   └── file_mover.py         discover, get_metadata, download, verify, cleanup
│   ├── ingestion/                Boundary 2+3 — decompressor + detector
│   │   ├── decompressor.py       gzip/zip stream expansion (1 MiB chunks), temp naming
│   │   ├── sampler.py            bounded read of the head of a file
│   │   ├── fields.py             delimiter/record-type helpers (shared with parser)
│   │   ├── layout.py             fixed-width layout inference / validation
│   │   └── detector.py           Detector -> DetectionResult
│   ├── parser/                   Boundary 5 — parse ApprovedConfig → batches
│   │   ├── parser.py             dispatch + ParseReport
│   │   ├── delimited_parser.py   streaming CSV-style records (handles preamble, quoting, multiline)
│   │   └── fixed_width_parser.py positional slices per approved layout
│   ├── writer/                   Boundary 6 — Parquet writer + verifier
│   │   └── parquet_writer.py     write_parquet (atomic, snappy, schema overrides), verify_parquet
│   ├── orchestrator/             Boundaries 7+8 + concurrency + recovery
│   │   ├── orchestrator.py       Pipeline, Workspace, PipelineSettings (2/1 workers)
│   │   ├── stages.py             per-file stage implementations (mixin, keeps files <500 lines)
│   │   └── registry.py           on-disk state (statuses, approved configs), recovery
│   ├── validator/                Boundary 9 — analyses over COMPLETE Parquet only
│   │   └── validator.py          store / sales / upc analyses via pl.scan_parquet
│   └── reporting/                Boundary 10 — item-level report artifacts
│       └── report_service.py     UPC|DESC summaries, comparison, Present-In summary, metrics, Excel
├── ui/report_view.py             Reports tab adapter (thin)
├── utils/checksums.py            md5 helpers (MFT verification)
└── tests/
    ├── unit/                     unit tests (one file per service/model)
    ├── integration/              end-to-end: mocked MFT → validate → report
    └── performance/              DVA_BENCH=1 benchmark harness
```

## 2. Data flow & lifecycle

### End-to-end

```
discover → download (.part) → verify size+checksum → rename (ready)
  → decompress if gz/zip (.decompressed temp, 1 MiB stream)
  → detect (bounded sample) → [AWAITING_APPROVAL if confidence low]
  → parse → write Parquet → verify dataset → mark COMPLETE → cleanup raw
  → Validator (scan_parquet only) → Reports
```

Compressed raws are expanded before detection (Boundary 2). The temp output
is named `<name>.decompressed`, tracked per file id and unlinked during
cleanup — the source archive is never deleted. A corrupt/empty archive
becomes `DECOMPRESSION_FAILED` and is retryable. `cleanup_stale_parts()`
also sweeps abandoned `*.decompressed` temp files from crashes.

### File lifecycle (`models/file_models.py`)

`FileStatus` is a strict state machine. Each transition is validated by
`SourceFile.record_status()`; an illegal transition raises. The oracle for
per-file independence is `Pipeline.process()`, which reads the ready queue and
moves each file through its own stages. Failures are terminal; `retry_file`
re-queues a file at its last safe checkpoint (`registry.py`).

### Concurrency

- `PipelineSettings.download_workers = 2`, `parser_workers = 1`
  (`concurrent.futures.ThreadPoolExecutor`). Two worker pools, no unbounded
  growth (Engineering rule 9). Keep the defaults tight until you have
  benchmark data (rule 10).

### Storage

- `raw/` is transient (`*.part` then ready-file; removed after COMPLETE).
- `datasets/<id>/` holds `part-*.parquet` + schema metadata; only
  `DatasetStatus.COMPLETE` datasets are visible to Validator/Reports.
- The registry (`<workspace>/registry.json`) persists file status and
  approved configs **outside** the Streamlit session so restarts do not
  lose work.

## 3. The contracts

### `DetectionResult` (Boundary 3)

Carries `format`, `structure_type`, `encoding`, `delimiter`,
`header_present`, `layout`, `schema`, `confidence`, `sample_metadata`,
`warnings`, and `preamble_lines`. `ApprovedConfig` (Boundary 4) is derived
from it, may be edited by the operator, and is serialized via
`to_json_dict` / `from_json_dict` for the registry.

Key behaviours baked into the Detector:

- **Delimiter detection** counts only lines *containing* the candidate, so
  prose around the data does not skew it.
- **Preamble handling** — leading banner/title/prose lines are detected as
  `preamble_lines`, carried into `ApprovedConfig.skip_rows`, and skipped by
  the parsers. Confidence drops when >1 prose line is skipped, forcing a
  human decision.
- **Record-type prefixes** (a single odd top line) are kept so
  `RECORD_TYPED` files still flag for review.

### Parsers (Boundary 5)

- Delimited: streams records (`csv` module), assembles multi-line quoted
  fields and continuation markers, drops blank records *before* counting,
  applies `skip_rows` (preamble + header), dedupes duplicate header names,
  and pads/logs column mismatches instead of crashing.
- Fixed-width: positional slices per approved `FixedWidthLayout`, `skip_rows`
  honoured before accounting, short lines logged.
- Parser output is a `ParseReport` (counters + issues) and row batches; the
  parser never reads `*.part`, never validates business rules, never deletes
  raw files.

### Parquet writer (Boundary 6)

`write_parquet` validates a schema, writes with `snappy` compression into the
dataset directory, persists `schema`/`metadata.json`, and is atomic. Schema
overrides apply strict-but-tolerating casts (un-castable values → null).
`verify_parquet` re-reads the file and compares shape/schema/dtypes.

### Validator (Boundary 9)

`validate(bau_path, test_path, config)` runs the enabled analyses over
`pl.scan_parquet` only:

- **store** — normalised (strip+lowercase) unique store sets, anti-joined for
  missing numbers.
- **sales** — per-store records/units/dollars; then a FULL-join comparison with
  `UNITS DIFFERENCE`, `DOLLAR DIFFERENCE`, `Unit/Dollar % Difference`
  (guarded to `-100` when BAU is 0), plus top/bottom 5 frames. Money
  semantics: `price_type` (total vs unit price), `implied_dollars_*`,
  `implied_units_*` (÷100), weighted units (`weighted_units_col` fallback),
  and `units_type` (`qty`/`weight`) carried in an output column.
- **upc** — unique UPC sets per side + missing counts.

Consistency rule: the *same* `_effective_units` semantic (weighted-when-else
raw, then implied scaling) is used for the per-store units and for Unit-Price
dollars, and mirrors the item-level report — so store and item dollars agree.

### Report service (Boundary 10)

`generate_reports(bau_path, test_path, config, out_dir)` streams to four
Parquet artifacts (`{prefix}_bau_summary`, `_test_summary`, `_comparison`,
`_summary`) with `sink_parquet` (flat memory), returns small UI frames
(metrics + top/bottom 5), and writes an **Excel workbook** (`{prefix}.xlsx`)
with sheets `BAU Summary`, `TEST Summary`, `Comparison`, `Summary` streamed
via `openpyxl.Workbook(write_only=True)` (bounded memory; note openpyxl is
the slowest artefact — ~each row appended, so it is opt-in in the perf
benchmark). Every artifact carries a `Unit Type` column.

## 4. Money / unit semantics in one place

| Flag | Effect | Where used |
|---|---|---|
| `price_type = UNIT_PRICE` | dollars = units × price (else dollars = price) | Validator + Reports |
| `implied_dollars_*` | divide price by 100 before totals/dollars | Validator + Reports |
| `implied_units_*` | divide units by 100 before totals/dollars | Validator (sales) |
| `weighted_units_col` | per row: weighted value when non-null else raw units | Validator (sales) + Reports |
| `units_type` (`qty`/`weight`) | informational column on every result | Validator + Reports |

## 5. Adding a feature

1. **Contract first**: extend the right model (`models/`) with defaults,
   `to_json_dict`, validation. Add unit tests in `tests/unit/test_<model>.py`.
2. **Service second**: the logic lives in `services/<area>/` and must be
   importable/run without Streamlit (rule 14). Keep files under 500 lines;
   split helpers into sibling modules if they grow.
3. **UI last**: add a thin adapter in `app.py` / `ui/` that calls the service.
4. Run the full suite and `compileall` before finishing.

## 6. Testing

```bash
cd dva
python3 -m pytest tests -q          # 204 tests (unit + integration)
python3 -m compileall -q app.py services models ui config utils
```

Every service has a dedicated test file. Failure paths are covered
(checksum mismatch, transport failure, decompression failure, verify
mismatch, retries, corrupt registry, schema-override rejects).

- `tests/unit/` — one file per service/model; `test_registry_recovery.py`
  holds the Phase-9 recovery/retry tests separately so no file exceeds 500
  lines.
- `tests/integration/test_end_to_end.py` — real Pipeline over a mock MFT
  (including a gzip-compressed TEST file), then the real Validator and
  Report service; plus a 10-file concurrency batch with an injected
  decompression failure proving per-file isolation, and the stale-part /
  decompressed-temp sweep.
- `tests/performance/bench_pipeline.py` — run only when asked:
  `DVA_BENCH=1 python3 tests/performance/bench_pipeline.py` (default 40 MB,
  `--mb` to scale, `--with-report` to also time the Excel writer). It
  measures pipeline/validate/report wall time + peak RSS
  (`resource.getrusage`) and asserts the peak stays bounded (no eager
  full-file load). pytest never collects it.

## 7. Design rules to respect

1. No pandas as the processing engine; pandas only in UI `to_pandas()` views.
2. No eager full-file loads (sample/batch/stream everywhere).
3. No large DataFrames in Streamlit session state.
4. No parsing of `*.part`; no Validator parsing of raw formats.
5. No business validation in the parser; no raw-file deletion by the parser.
6. Only COMPLETE datasets are ever consumed by Validator/Reports.
7. Bounded worker pools (2 download / 1 parser) until benchmarked.
8. Source-format config (`detection_models`) is separate from validation
   config (`validation_models`).
9. Failures are isolated per file; recover on restart via the registry.
10. Keep files under 500 lines and code commented / readable.