"""Writer for the unified `model_artifacts` table (PRD item 1).

ONE bucketed UC Delta table that is BOTH the serving source of truth (carries
the model blob) AND the Genie-queryable telemetry store — replacing the old
split of `run_plan` (blobs) + a separate blob-free telemetry table. Skips a
whole copy of the data.

- `bucket` column = `cast(substring(sku,5,10) as int) % bucket_count`, matching
  `_sku_bucket` / the logging writer, so the serving fleet's per-shard bucket
  filter is consistent build-to-serve.
- Top-level `rmse`/`mape`/`r2` are pulled out of the `metrics` map for cheap
  analytical scans (same convention as the old telemetry table).
- A blob-excluding VIEW (`<table>_genie`) is created for the Genie space —
  virtual, zero copy; column pruning means NL→SQL never reads the blob.

Schema + comments derive from `ModelArtifactRow` (single source of truth).
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dbrx_multimodel_registration.domains.entities import ModelArtifactRow
from dbrx_multimodel_registration.utils.helpers import column_comments_from_dataclass


class ModelArtifactsTable:
    """Persist trained-model output as the bucketed `model_artifacts` UC table
    + a blob-excluding Genie view."""

    name = "model_artifacts"

    def __init__(self, spark: SparkSession, bucket_count: int) -> None:
        self.spark = spark
        self.bucket_count = bucket_count

    def write(self, trained: DataFrame, table: str) -> None:
        """Write `trained` (region, sku, model_name, model_blob_bytes, params,
        metrics) as the bucketed model_artifacts table at `table` (3-part UC
        name), then set comments + create the Genie view.
        """
        enriched = (
            trained
            .withColumn("bucket", F.expr(f"cast(substring(sku, 5, 10) as int) % {self.bucket_count}"))
            .withColumn("rmse", F.col("metrics")["rmse"].cast("double"))
            .withColumn("mape", F.col("metrics")["mape"].cast("double"))
            .withColumn("r2", F.col("metrics")["r2"].cast("double"))
            .select(*[f.name for f in ModelArtifactRow.spark_schema().fields])
        )
        # Partition by bucket so per-shard reads prune to whole buckets. Range
        # repartition keeps per-file sku ranges tight (footer-stat skipping).
        n_parts = max(self.bucket_count, self.spark.sparkContext.defaultParallelism * 2)
        (
            enriched
            .repartitionByRange(n_parts, "bucket", "sku")
            .sortWithinPartitions("bucket", "sku")
            .write.mode("overwrite").option("overwriteSchema", "true")
            .partitionBy("bucket")
            .format("delta").saveAsTable(table)
        )
        self._apply_comments(table)
        self._create_genie_view(table)

    def _apply_comments(self, table: str) -> None:
        self.spark.sql(
            f"COMMENT ON TABLE {table} IS "
            f"'Per-(region, sku, model) demand-forecasting model artifacts: blob + telemetry, "
            f"bucketed for serving-shard fan-out. Query the {table}_genie view for NL/SQL "
            f"(excludes the model blob).'"
        )
        for col, comment in column_comments_from_dataclass(ModelArtifactRow).items():
            safe = comment.replace("'", "''")
            self.spark.sql(f"ALTER TABLE {table} ALTER COLUMN {col} COMMENT '{safe}'")

    def _create_genie_view(self, table: str) -> None:
        """Blob-excluding view for Genie — virtual, no data copy."""
        cols = ", ".join(ModelArtifactRow.GENIE_VIEW_COLUMNS)
        self.spark.sql(
            f"CREATE OR REPLACE VIEW {table}_genie "
            f"COMMENT 'Genie-facing view of {table} — all columns except the model blob.' "
            f"AS SELECT {cols} FROM {table}"
        )
