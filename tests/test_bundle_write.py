"""Local pyspark tests for the bundle write path.

Mirrors the Databricks environment by running the SAME Spark code locally
against a `local[4]` SparkSession. This gives sub-30-second feedback on:

  - layout correctness (multiple files per bucket, no overlapping sku ranges)
  - PyArrow row-group skipping behaving as designed
  - per-task memory bound holding with real-size blobs

Requires `openjdk@17` (Databricks Runtime 17.3 also uses Java 17). Set up:

  brew install openjdk@17
  export JAVA_HOME=/opt/homebrew/opt/openjdk@17
  export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"

Then run:

  pytest tests/test_bundle_write.py -v -s

The tests split into a fast layer (`*_layout` — 1 KB blobs, <1 s per test)
and a slow layer (`*_memory` — 4.4 MB blobs, ~20-30 s). The fast layer runs
on every code save; the slow layer runs before pushing for Databricks tests.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession


# Run BEFORE pyspark import: set JAVA_HOME so the gateway boots.
if not os.environ.get("JAVA_HOME"):
    candidate = "/opt/homebrew/opt/openjdk@17"
    if Path(candidate).is_dir():
        os.environ["JAVA_HOME"] = candidate
        os.environ["PATH"] = f"{candidate}/bin:" + os.environ.get("PATH", "")


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder
        .master("local[4]")
        .appName("test_bundle_write")
        .config("spark.driver.memory", "2g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def _make_plan_df(spark, n_skus: int, n_regions: int, blob_size_bytes: int, model_names: list[str]):
    """Build a plan_df with the same schema the strategy expects."""
    from itertools import product
    blob = b"x" * blob_size_bytes
    skus = [f"SKU-{i:06d}" for i in range(n_skus)]
    regions = [f"REGION-{j:02d}" for j in range(n_regions)]
    rows = [
        (r, s, m, blob, {"learning_rate": "0.01"}, {"rmse": 0.1})
        for r, s, m in product(regions, skus, model_names)
    ]
    schema = "region string, sku string, model_name string, model_blob_bytes binary, params map<string,string>, metrics map<string,double>"
    return spark.createDataFrame(rows, schema=schema)


def _run_bundle_write(spark, plan_df, region_name, bundle_dir, budget):
    """Replica of UCTableLoggingStrategy._write_region_subset_to_artifact_bundle."""
    from math import ceil
    from pyspark.sql.functions import col as _col, expr as _expr

    from dbrx_multimodel_registration.adapters.logging.uc_table import (
        _MODEL_BLOB_SIZE_BYTES,
        PARQUET_ROW_GROUP_SIZE_BYTES,
    )

    region_rows = plan_df.where(_col("region") == region_name).count()
    total_bytes = region_rows * _MODEL_BLOB_SIZE_BYTES
    n_partitions = max(
        budget.bucket_count,
        ceil(total_bytes / budget.target_bytes_per_task),
    )

    (
        plan_df
        .where(_col("region") == region_name)
        .select("sku", "model_name", "model_blob_bytes", "params", "metrics")
        .withColumn(
            "bucket",
            _expr(f"cast(substring(sku, 5, 10) as int) % {budget.bucket_count}"),
        )
        .repartitionByRange(n_partitions, "bucket", "sku")
        .sortWithinPartitions("bucket", "sku")
        .write
        .mode("overwrite")
        .partitionBy("bucket")
        .option("parquet.block.size", str(PARQUET_ROW_GROUP_SIZE_BYTES))
        .parquet(bundle_dir)
    )
    return n_partitions


def _bucket_dir_files(bundle_dir: str, bucket_value: int) -> list[Path]:
    bp = Path(bundle_dir) / f"bucket={bucket_value}"
    if not bp.is_dir():
        return []
    return sorted(p for p in bp.iterdir() if p.suffix == ".parquet")


def _sku_min_max(file_path: Path) -> tuple[int, int]:
    """Read parquet column statistics: min/max of the `sku` column.

    Returns (min_int, max_int) using the numeric suffix of `SKU-NNNNNN`.
    """
    pf = pq.ParquetFile(str(file_path))
    sku_col_idx = pf.schema_arrow.get_field_index("sku")
    mins, maxes = [], []
    for rg_idx in range(pf.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        col = rg.column(sku_col_idx)
        stats = col.statistics
        if stats is not None:
            mins.append(int(stats.min.split("-")[-1]))
            maxes.append(int(stats.max.split("-")[-1]))
    if not mins:
        return (0, 0)
    return (min(mins), max(maxes))


# ─── Phase 0a: fast layout tests (1 KB blobs, <1 s each) ──────────────


def test_layout_multi_file_per_bucket(spark, tmp_path):
    """At 200 SKUs with 8 buckets and 50 MB/task budget, expect ≥1 file
    per bucket with all 8 bucket=N dirs present."""
    from dbrx_multimodel_registration.domains.entities import WorkloadBudget
    budget = WorkloadBudget(target_bytes_per_task=50 * 1024 * 1024, bucket_count=8)
    plan_df = _make_plan_df(spark, n_skus=200, n_regions=1, blob_size_bytes=1024, model_names=["A", "B", "C"])
    n_parts = _run_bundle_write(spark, plan_df, "REGION-00", str(tmp_path / "bundle"), budget)
    print(f"\n[layout] n_partitions={n_parts}")
    for bucket in range(8):
        files = _bucket_dir_files(str(tmp_path / "bundle"), bucket)
        assert len(files) >= 1, f"bucket={bucket} has no files: {files}"


def test_layout_non_overlapping_sku_ranges(spark, tmp_path):
    """Within each bucket dir, multiple files MUST have non-overlapping
    sku integer ranges. This is what lets PyArrow skip files via
    parquet stats at read time."""
    from dbrx_multimodel_registration.domains.entities import WorkloadBudget
    # Force multiple files per bucket: low task-size budget + large n_skus
    budget = WorkloadBudget(target_bytes_per_task=10 * 1024, bucket_count=4)  # 10 KB/task
    plan_df = _make_plan_df(spark, n_skus=400, n_regions=1, blob_size_bytes=1024, model_names=["A", "B", "C"])
    _run_bundle_write(spark, plan_df, "REGION-00", str(tmp_path / "bundle"), budget)

    multi_file_buckets_checked = 0
    for bucket in range(4):
        files = _bucket_dir_files(str(tmp_path / "bundle"), bucket)
        if len(files) < 2:
            continue
        multi_file_buckets_checked += 1
        ranges = [_sku_min_max(f) for f in files]
        ranges.sort()
        # Adjacent ranges should not overlap: range[i].max < range[i+1].min
        for (a_min, a_max), (b_min, b_max) in zip(ranges, ranges[1:]):
            assert a_max < b_min, (
                f"bucket={bucket}: overlapping sku ranges {(a_min, a_max)} vs {(b_min, b_max)} "
                f"— row-group skipping will not work file-by-file."
            )
    print(f"\n[layout] checked {multi_file_buckets_checked} multi-file buckets, all non-overlapping")
    assert multi_file_buckets_checked >= 1, (
        "test setup failed to produce any multi-file buckets — "
        "increase n_skus or reduce target_bytes_per_task."
    )


def test_layout_pyarrow_read_finds_correct_row(spark, tmp_path):
    """End-to-end: write via Spark, read via pyarrow.dataset with the
    same (bucket, sku) filter the production PyArrowLruModel uses.
    Validates the read API doesn't break with the multi-file layout."""
    from dbrx_multimodel_registration.adapters.logging.uc_table import _sku_bucket
    from dbrx_multimodel_registration.domains.entities import WorkloadBudget

    budget = WorkloadBudget(target_bytes_per_task=50 * 1024 * 1024, bucket_count=8)
    plan_df = _make_plan_df(spark, n_skus=200, n_regions=1, blob_size_bytes=1024, model_names=["A", "B", "C"])
    _run_bundle_write(spark, plan_df, "REGION-00", str(tmp_path / "bundle"), budget)

    target_sku = "SKU-000123"
    target_bucket = _sku_bucket(target_sku, budget.bucket_count)

    dataset = ds.dataset(str(tmp_path / "bundle"), format="parquet", partitioning="hive")
    table = dataset.to_table(
        filter=(ds.field("bucket") == target_bucket) & (ds.field("sku") == target_sku),
        columns=["model_blob_bytes"],
    )
    assert table.num_rows >= 1, f"no row found for {target_sku} (bucket={target_bucket})"
    blob = table.column("model_blob_bytes")[0].as_py()
    assert blob == b"x" * 1024, "blob bytes mismatch"


# ─── Phase 0b: real-blob memory test (~20-30 s) ──────────────────────


@pytest.mark.slow
def test_memory_real_size_blobs(spark, tmp_path):
    """With real 4.4 MB blobs at modest scale, confirm the write completes
    without OOM under a 100 MB target_bytes_per_task budget."""
    from dbrx_multimodel_registration.domains.entities import WorkloadBudget

    budget = WorkloadBudget(target_bytes_per_task=100 * 1024 * 1024, bucket_count=8)
    # 50 SKUs × 3 models × 4.4 MB = 660 MB total → 7 partitions at 100 MB budget
    plan_df = _make_plan_df(spark, n_skus=50, n_regions=1, blob_size_bytes=4_400_000, model_names=["A", "B", "C"])
    n_parts = _run_bundle_write(spark, plan_df, "REGION-00", str(tmp_path / "bundle"), budget)
    print(f"\n[memory] n_partitions={n_parts} (expected ≥7 from 660MB / 100MB)")
    assert n_parts >= 7

    # Confirm read still works
    target_sku = "SKU-000042"
    from dbrx_multimodel_registration.adapters.logging.uc_table import _sku_bucket
    target_bucket = _sku_bucket(target_sku, budget.bucket_count)
    dataset = ds.dataset(str(tmp_path / "bundle"), format="parquet", partitioning="hive")
    table = dataset.to_table(
        filter=(ds.field("bucket") == target_bucket) & (ds.field("sku") == target_sku),
        columns=["model_blob_bytes"],
    )
    assert table.num_rows >= 1
