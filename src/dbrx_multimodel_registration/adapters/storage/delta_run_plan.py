"""Persist the trainer output as a resumable, streamable run plan.

On Databricks the `table` arg is a 3-part UC name and we write a Delta table.
For local smoke tests it can be a filesystem path and we write Parquet.
Streaming uses `toLocalIterator()` so the driver never materializes the full
trained-model set in memory.

The table schema mirrors `TrainedModelRecord` field-for-field — `params` and
`metrics` are MapType columns (no JSON encoding) — derived from
`TrainedModelRecord.spark_schema()`.
"""
from __future__ import annotations

from typing import Iterator

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from dbrx_multimodel_registration.domains.entities import TrainedModelRecord


class DeltaRunPlanRepository:
    name = "delta_or_parquet"

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark

    def _is_uc_table(self, table: str) -> bool:
        return table.count(".") >= 1 and "/" not in table

    def persist(self, training_output: DataFrame, table: str) -> None:
        # overwriteSchema=true lets us replace the existing table even when
        # the dataclass-derived schema has evolved (e.g. float→double).
        # Memory-bounded repartition: the previous `.repartition("region")`
        # produced only 5 partitions; at 20k+ SKUs with 4.4 MB blobs that
        # put ~264 GB per task → Python worker OOM (Phase 3 retry 1+2).
        # Hash on (region, sku) so N partitions actually get N non-empty
        # splits — `.repartition(N, "region")` alone hashes by region and
        # collapses to one partition per distinct region value (5 partitions).
        n_partitions = self.spark.sparkContext.defaultParallelism * 2
        writer = (
            training_output
            .repartition(n_partitions, "region", "sku")
            .write
            .mode("overwrite")
            .option("overwriteSchema", "true")
        )
        if self._is_uc_table(table):
            writer.format("delta").saveAsTable(table)
        else:
            writer.format("parquet").save(table)

    def _read(self, table: str) -> DataFrame:
        if self._is_uc_table(table):
            return self.spark.read.table(table)
        return self.spark.read.parquet(table)

    def distinct_regions(self, table: str) -> list[str]:
        return [row.region for row in self._read(table).select("region").distinct().collect()]

    def stream_by_region(self, table: str, region: str) -> Iterator[TrainedModelRecord]:
        # sortWithinPartitions instead of orderBy — `persist()` already
        # `repartition("region")`s on write, so all rows for one region land
        # in one Spark partition. Local-only sort is enough to make
        # itertools.groupby see consecutive same-SKU rows, and avoids the
        # ~25-min tail caused by full-shuffle sort+toLocalIterator at scale.
        df = self._read(table).where(col("region") == region).sortWithinPartitions("sku", "model_name")
        for row in df.toLocalIterator():
            yield TrainedModelRecord(
                region=row["region"],
                sku=row["sku"],
                model_name=row["model_name"],
                # Spark binary columns come back as bytearray; we want immutable bytes.
                model_blob_bytes=bytes(row["model_blob_bytes"]),
                # MapType columns come back as Python dicts — no JSON parse.
                params=dict(row["params"]),
                metrics={k: float(v) for k, v in row["metrics"].items()},
            )
