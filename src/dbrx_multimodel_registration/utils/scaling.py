"""Scale up demand + run-plan tables by duplicate-and-rekey instead of regenerating.

Why: at 20k+ SKUs the synthetic data path (SparkDataGenerator with Faker on
driver, then TrainingSimulator's mapInPandas with 4.4MB blobs) takes 15-30 min.
For benchmarking the bundle-write + serving-read path, the row VALUES don't
matter — only row count and unique SKU IDs. So we read the existing N=10k
table and duplicate it K times with shifted SKU IDs to reach N×K rows.

Bucket distribution: SKU IDs are `SKU-NNNNNN`. Bucket = `NNNNNN % BUCKET_COUNT`.
Shifting by N=10000 → new SKUs start at 10000, 20000, etc. Since 10000 is not
a multiple of 64, the shifted SKUs distribute evenly across buckets.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def expand_table_by_duplicate_and_rekey(
    spark: SparkSession,
    table_name: str,
    target_skus: int,
    base_skus: int,
) -> bool:
    """Expand `table_name` from `base_skus` distinct SKUs to `target_skus`.

    Returns True if expansion happened, False if not applicable (e.g.
    target == base, or target < base, or target is not a multiple of base).
    Callers should fall back to full regeneration when this returns False.
    """
    if target_skus == base_skus:
        return False
    if target_skus < base_skus:
        return False
    if target_skus % base_skus != 0:
        return False

    multiplier = target_skus // base_skus
    base = spark.read.table(table_name)

    # Synthesize K-1 copies with SKU integer-suffix offsets of N, 2N, 3N…
    # `cast(substring(sku,5,10) as int) + offset` then re-formatted with
    # zero-pad to 6 digits matches the canonical SKU-NNNNNN shape used
    # everywhere (training simulator, bucket calculator, GenerationSpec.of).
    expansions = [base]
    for i in range(1, multiplier):
        offset = i * base_skus
        shifted = base.withColumn(
            "sku",
            F.format_string(
                "SKU-%06d",
                F.substring("sku", 5, 10).cast("int") + F.lit(offset),
            ),
        )
        expansions.append(shifted)

    expanded: DataFrame = expansions[0]
    for e in expansions[1:]:
        expanded = expanded.unionByName(e)

    # Memory-bounded repartition. Without this, `.repartition("region")` gives
    # only 5 partitions; at 20k+ SKUs that's ~264 GB per task with 4.4 MB
    # blobs → OOM. Pick partition count so each task holds ~500 MB.
    # The blob-size constant lives in uc_table.py; import lazily to keep
    # scaling.py independent of the logging adapter.
    from math import ceil
    from dbrx_multimodel_registration.adapters.logging.uc_table import (
        _MODEL_BLOB_SIZE_BYTES,
    )
    has_blob_column = "model_blob_bytes" in expanded.columns
    if has_blob_column:
        approx_bytes = base.count() * multiplier * _MODEL_BLOB_SIZE_BYTES
        n_partitions = max(
            spark.sparkContext.defaultParallelism * 2,
            ceil(approx_bytes / (500 * 1024 * 1024)),
        )
    else:
        # Demand table — no blob column, small per-row size. Default is fine.
        n_partitions = spark.sparkContext.defaultParallelism * 2

    (
        expanded
        # repartition(N, "region") only gives 5 non-empty buckets when there
        # are 5 region values — the hash collapses to one partition per
        # distinct value. Use multi-column hash so N partitions actually
        # get N non-empty splits.
        .repartition(n_partitions, "region", "sku")
        .write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    return True


def existing_sku_count(spark: SparkSession, table_name: str) -> int:
    """Return distinct SKU count in `table_name`, or 0 if table doesn't exist."""
    if not spark.catalog.tableExists(table_name):
        return 0
    return spark.read.table(table_name).select("sku").distinct().count()
