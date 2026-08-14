# DVA End-to-End Logic & Pseudocode

## 1. Core orchestration

```python
def run_batch(selected_files, config):
    batch = create_batch(selected_files)

    download_pool = create_download_pool(config.max_download_workers)
    parser_pool = create_parser_pool(config.max_parser_workers)

    for source_file in selected_files:
        submit(download_pool, download_file, source_file, batch)

    while batch.has_pending_work():
        completed_downloads = collect_completed_downloads()

        for result in completed_downloads:
            if result.success:
                ready_queue.put(result.local_file)
            else:
                mark_failed(result.file_id, "DOWNLOAD_FAILED")

        ready_files = ready_queue.get_available()

        for local_file in ready_files:
            submit(parser_pool, process_local_file, local_file, batch)

        completed_parses = collect_completed_parses()

        for result in completed_parses:
            if result.success:
                mark_complete(result.file_id)
                cleanup_raw_if_allowed(result)
            else:
                mark_failed(result.file_id, result.error)

    finalize_batch(batch)
```

## 2. File download

```python
def download_file(source_file):
    source_meta = mft.get_metadata(source_file)

    part_path = raw_dir / f"{source_file.name}.part"
    ready_path = raw_dir / source_file.name

    download_to_file(source_file, part_path)

    if source_meta.size is not None:
        if part_path.stat().st_size != source_meta.size:
            raise DownloadIncomplete()

    if source_meta.checksum:
        verify_checksum(part_path, source_meta.checksum)

    atomic_rename(part_path, ready_path)

    return DownloadResult(
        file_id=source_file.id,
        local_path=ready_path,
        success=True,
    )
```

## 3. Decompression

```python
def prepare_input(local_path):
    if not is_compressed(local_path):
        return local_path

    decompressed_path = decompress_to_temp(local_path)

    verify_decompressed_file(decompressed_path)

    remove_compressed_source_if_policy_allows(local_path)

    return decompressed_path
```

If decompression creates a large temporary raw file, treat that file as transient and delete it only after successful Parquet verification.

## 4. Detection

```python
def detect(local_path):
    sample = read_bounded_sample(local_path)

    file_type = detect_file_type(sample)
    structure = detect_structure(sample, file_type)
    header = detect_header(sample, structure)

    if structure == "fixed_width":
        layout = get_or_request_fixed_width_layout()
    else:
        layout = infer_or_load_delimited_configuration(sample)

    schema = infer_schema(sample, layout)

    return DetectionResult(
        file_type=file_type,
        structure=structure,
        header=header,
        layout=layout,
        schema=schema,
    )
```

## 5. User approval

```python
def approve_configuration(detection_result):
    show_detection_in_streamlit(detection_result)

    decision = wait_for_user_action()

    if decision == "ACCEPT":
        return detection_result.config

    if decision == "MODIFY":
        return validate_modified_config(get_user_config())

    if decision == "REPROCESS":
        return None

    raise ApprovalRequired()
```

## 6. Delimited parsing

```python
def parse_delimited(path, config):
    lazy_frame = (
        pl.scan_csv(
            path,
            separator=config.delimiter,
            has_header=config.has_header,
            schema_overrides=config.schema_overrides,
        )
    )

    return lazy_frame
```

Use lazy execution where it provides value. Avoid collecting the entire file simply to write it.

## 7. Fixed-width parsing

```python
def parse_fixed_width(path, layout):
    # Read controlled batches/lines.
    # Extract fields using approved start/end positions.
    # Convert to Polars DataFrames in bounded batches.

    for raw_batch in read_line_batches(path):
        structured_batch = apply_fixed_width_layout(
            raw_batch,
            layout
        )

        yield structured_batch
```

## 8. Parquet writing

```python
def build_parquet(parsed_data, dataset_dir):
    writer = create_parquet_writer(dataset_dir)

    for batch in parsed_data:
        validate_batch_schema(batch)
        writer.write(batch)

    writer.close()

    write_dataset_metadata(dataset_dir)
    verify_dataset(dataset_dir)

    mark_dataset_complete(dataset_dir)
```

For a lazy delimited pipeline, choose a bounded materialization strategy compatible with the chosen Polars/PyArrow writer implementation rather than calling a huge eager `collect()`.

## 9. Dataset verification

```python
def verify_dataset(dataset_dir):
    assert parquet_files_exist(dataset_dir)
    assert schema_is_consistent(dataset_dir)
    assert metadata_exists(dataset_dir)

    lf = pl.scan_parquet(dataset_dir / "*.parquet")

    # Lightweight validation only.
    verify_required_columns(lf)
    verify_readability(lf)

    return True
```

## 10. Cleanup

```python
def cleanup_raw_if_allowed(result):
    if result.dataset_status != "COMPLETE":
        return

    safe_delete(result.raw_path)
```

The Parser must not delete files. Cleanup belongs to the orchestrator.

## 11. Validator

```python
def run_validation(dataset_dir, validation_config):
    lf = pl.scan_parquet(dataset_dir / "*.parquet")

    if validation_config.store_validation:
        store_report = validate_store_level(lf, validation_config)

    if validation_config.sales_validation:
        sales_report = validate_sales(lf, validation_config)

    if validation_config.upc_validation:
        upc_report = validate_upc(lf, validation_config)

    return generate_reports(...)
```

## 12. Batch independence

```python
for file in batch.files:
    file.status is independent

file1 COMPLETE
file2 PARSING
file3 DOWNLOADING
file4 FAILED
file5 COMPLETE
```

A failure in file4 must not force file1/file5 to restart.

## 13. Recovery rule

Phase 1 recovery is restart-based:

```text
download failure -> retry that file
parse failure    -> retain raw file and retry after config correction
write failure    -> discard incomplete dataset and retry
verification failure -> dataset is not published
```

Complex checkpoint/resume across MFT streams is explicitly not required for Phase 1.

## 14. Resource-control rule

Never create unbounded workers.

```python
MAX_DOWNLOAD_WORKERS = config.max_download_workers
MAX_PARSER_WORKERS = config.max_parser_workers
```

Benchmark before increasing concurrency.
