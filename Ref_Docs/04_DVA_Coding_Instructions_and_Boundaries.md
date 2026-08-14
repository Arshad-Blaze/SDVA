# DVA Coding Instructions & Engineering Boundaries

## Objective

Implement the architecture using Streamlit + Polars while keeping acquisition, detection, parsing, storage, validation and UI independently testable.

## Recommended project structure

```text
dva/
├── app.py
├── config/
│   ├── settings.py
│   └── schemas.py
├── models/
│   ├── file_models.py
│   ├── detection_models.py
│   ├── dataset_models.py
│   └── validation_models.py
├── services/
│   ├── mft/
│   │   └── file_mover.py
│   ├── ingestion/
│   │   ├── orchestrator.py
│   │   ├── detector.py
│   │   ├── parser.py
│   │   ├── delimited_parser.py
│   │   ├── fixed_width_parser.py
│   │   ├── decompressor.py
│   │   └── parquet_writer.py
│   ├── validation/
│   │   ├── validator.py
│   │   ├── store_validation.py
│   │   ├── sales_validation.py
│   │   └── upc_validation.py
│   └── reporting/
│       └── report_service.py
├── ui/
│   ├── connection_manager.py
│   ├── detection_view.py
│   ├── ingestion_view.py
│   ├── validation_view.py
│   └── report_view.py
├── utils/
│   ├── filesystem.py
│   ├── logging.py
│   └── checksums.py
└── tests/
    ├── unit/
    ├── integration/
    └── performance/
```

## Boundary 1 — File Mover

Implement:

```python
discover_files()
get_metadata()
download_file()
verify_download()
```

Do not put parsing logic here.

The File Mover should write to:

```text
<workspace>/raw/<file>.part
```

and only rename to:

```text
<workspace>/raw/<file>
```

after successful verification.

## Boundary 2 — Decompressor

Input:

```text
verified local raw file
```

Output:

```text
local decompressed temporary file
```

Do not combine decompression with business validation.

If the decompressed file is itself large, process it incrementally.

## Boundary 3 — Detector

Input:

```python
Path
```

Output:

```python
DetectionResult
```

DetectionResult should contain:

```python
format
structure_type
encoding
delimiter
header_present
layout
schema
confidence
sample_metadata
```

Detector should operate on bounded samples, not entire files.

## Boundary 4 — User Configuration

Streamlit displays DetectionResult and allows:

```text
Accept
Modify
Override
Reprocess
```

Persist approved configuration outside Streamlit session state.

Do not store large DataFrames in session state.

## Boundary 5 — Parser

Input:

```python
Path + ApprovedConfig
```

Output:

```text
structured batches / Polars LazyFrame where applicable
```

### Delimited

Prefer Polars native CSV scanning where the format is compatible.

```python
pl.scan_csv(...)
```

### Fixed Width

Use the user-approved layout.

Process controlled batches and emit Polars DataFrames.

Do not load the entire fixed-width file into memory.

## Boundary 6 — Parquet Writer

Input:

```text
structured Polars batches
```

Output:

```text
dataset directory
```

Responsibilities:

- schema consistency
- compression
- row groups
- file rollover
- metadata
- completion state

Do not create one tiny file per input chunk.

## Boundary 7 — Dataset Manager

Implement:

```python
create_dataset()
mark_writing()
verify_dataset()
mark_complete()
mark_failed()
get_dataset_status()
```

Only COMPLETE datasets can be consumed by Validator.

## Boundary 8 — Cleanup

Implement:

```python
cleanup_raw_file()
```

Only call it after:

```text
Parquet write successful
+
dataset verification successful
+
dataset marked COMPLETE
```

Never let Parser delete source files.

## Boundary 9 — Validator

Validator accepts only:

```python
dataset_path
validation_config
```

It should begin with:

```python
lf = pl.scan_parquet(...)
```

Do not add raw CSV/fixed-width parsing into Validator.

## Boundary 10 — Streamlit

Streamlit should orchestrate:

```text
Connect
→ Discover
→ Select
→ Detect
→ Approve
→ Ingest
→ Validate
→ Report
```

The UI should display per-file status:

```text
DOWNLOADING
READY
DETECTING
WAITING FOR APPROVAL
PARSING
WRITING PARQUET
VERIFYING
COMPLETE
FAILED
```

Do not put heavy loops directly in UI rendering code.

## Concurrency

Start with:

```python
max_download_workers = 2
max_parser_workers = 1
```

Then benchmark.

Do not assume more workers means faster processing.

Measure:

- MFT throughput
- disk read/write throughput
- CPU
- memory
- Parquet write throughput
- total batch completion time

## Testing requirements

Every phase must end with tests before the next phase begins.

### Unit tests

Test:

- file completeness checks
- decompression
- delimiter detection
- header detection
- fixed-width layout
- schema inference
- datatype conversion
- Parquet metadata
- dataset lifecycle
- cleanup rules

### Integration tests

Test:

```text
MFT/file-mover mock
→ local staging
→ detector
→ parser
→ parquet
→ validator
→ report
```

### Failure tests

Test:

- incomplete download
- checksum mismatch
- decompression failure
- invalid delimiter
- invalid fixed-width layout
- malformed record
- schema mismatch
- disk-full simulation where practical
- Parquet write failure
- validator against incomplete dataset

### Concurrency tests

Test 10-file batches where:

- small files finish before large files
- one file fails
- multiple files finish simultaneously
- downloads and parsing overlap
- completed files are cleaned independently

### Performance tests

Use representative sizes:

```text
100 MB
500 MB
1 GB
5 GB
10 GB
```

and measure:

```text
download time
parse time
Parquet write time
validation time
peak RAM
peak disk usage
CPU utilization
total batch time
```

## Engineering rules

1. Do not use pandas as the primary processing engine.
2. Do not load huge source files eagerly.
3. Do not keep large DataFrames in Streamlit session state.
4. Do not parse incomplete `.part` files.
5. Do not let Validator parse raw formats.
6. Do not let Parser perform business validation.
7. Do not let Parser delete raw files.
8. Do not expose incomplete Parquet datasets.
9. Do not create unbounded worker pools.
10. Do not optimize concurrency before measuring.
11. Prefer simple local-file processing over unnecessary network-stream complexity.
12. Keep MFT-specific code isolated from parsing code.
13. Keep source-format configuration separate from validation configuration.
14. Make every service callable independently from Streamlit.
15. Keep failures isolated to individual files wherever possible.

## Implementation sequence

### Phase 1 — Contracts and models
Build:
- File models
- DetectionResult
- ApprovedConfig
- Dataset metadata
- File lifecycle states

Test all models.

### Phase 2 — File Mover
Build:
- discovery
- metadata
- download
- completeness verification
- `.part` handling

Test with mock/local source.

### Phase 3 — Detector
Build:
- bounded sampling
- delimiter detection
- fixed-width detection
- header detection
- schema/layout generation

Test against representative files.

### Phase 4 — Parser
Build:
- delimited parser
- fixed-width parser
- datatype conversion
- bounded processing

Test independently.

### Phase 5 — Parquet Writer
Build:
- incremental writes
- schema validation
- row groups
- rollover
- metadata
- completion state

Test dataset integrity.

### Phase 6 — Orchestrator
Build:
- download pool
- ready queue
- parser pool
- per-file lifecycle
- retries
- cleanup

Test 10-file mixed-size batches.

### Phase 7 — Validator Integration
Move Validator to:

```python
pl.scan_parquet(...)
```

and verify existing reports remain correct.

### Phase 8 — Streamlit
Build:
- connection UI
- discovery
- detection/configuration
- progress
- validation
- reports

Keep UI thin.

### Phase 9 — Performance and failure hardening
Benchmark and tune concurrency, disk usage, memory and processing throughput.

## Definition of done

The implementation is complete only when:

```text
MFT
 ↓
per-file download
 ↓
download verification
 ↓
detector
 ↓
user-approved config
 ↓
parser
 ↓
Parquet
 ↓
dataset verification
 ↓
raw cleanup
 ↓
Validator
 ↓
reports
```

works end-to-end for multiple files, while a failure in one file does not require successful files to be reprocessed.
