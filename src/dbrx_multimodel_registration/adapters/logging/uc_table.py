"""UC-table-backed logging strategy: bypasses MLflow's create_run ceiling.

Story: writing 100k+ runs through MLflow's tracking API hits a workspace-side
ceiling around 8 SKU/s (see [[Iter2]] benchmarks). For demand-forecasting at
scale, the per-SKU run hierarchy isn't the load-bearing structure — what the
business actually needs is:

  1. Searchable per-SKU training telemetry (params, metrics, model_name) →
     write to a Unity Catalog Delta table once per region (Spark, GB/s)
  2. Natural-language query over that telemetry → create a Databricks Genie
     space pointed at the Delta table, with canned example questions
  3. Per-SKU model artifact retrieval at serving time → write per-region
     parquet partitioned by `sku` (PyArrow-readable from a Model Serving
     endpoint without Spark, partition-prune to one file per inference)
  4. N deployable MLflow runs per region — each a serving-endpoint candidate
     that carries tag pointers to the table, the Genie space, and the
     model bundle

Stage-5 wall-clock at 100k SKUs/region drops from ~17 hours (Iter2 collapsed_sku)
to seconds, because the only MLflow ops are 15 cheap create_run calls (5 regions
× N=3 endpoints).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from dbrx_multimodel_registration.domains.entities import (
    LoggingConfig,
    LoggingMetrics,
    RegionSpec,
    TrainedModelRecord,
    TrainedModelTelemetry,
    WorkloadBudget,
)
from dbrx_multimodel_registration.ports.storage.run_plan_repository import RunPlanRepositoryPort
from dbrx_multimodel_registration.utils.helpers import column_comments_from_dataclass

log = logging.getLogger(__name__)


_TELEMETRY_TABLE_NAME = "demand_forecasting_artifacts"
_GENIE_SPACE_NAME = "Demand Forecasting Insights"
_WAREHOUSE_NAME = "demand-forecasting-genie-warehouse"

# Phase 3 bundle partition fan-out. Default for small bundles; for larger
# bundles use `compute_bucket_count(n_skus)` to pick a value that keeps
# per-bucket file size and dir enumeration both bounded.
BUCKET_COUNT = 64

# Target SKUs per bucket. Picked from the iter sweep at 10k:
#   - 64 buckets (~156 SKUs/bucket) → cold p50 = 704ms (iter 10)
#   - 256 buckets (~39 SKUs/bucket) → cold p50 = 2559ms (worse — too many dirs)
# Cap total buckets at 512 so PyArrow load_context dir enumeration stays cheap.
_TARGET_SKUS_PER_BUCKET = 156
_BUCKET_COUNT_MAX = 512


def compute_bucket_count(n_skus: int) -> int:
    """Pick a BUCKET_COUNT given the total SKU count of the bundle.

    Goal: keep per-bucket file size manageable AND keep dir count bounded
    (PyArrow load_context scales with dir enumeration overhead).

    Examples:
      compute_bucket_count(1_000)    == 64    (default — small bundle)
      compute_bucket_count(10_000)   == 64    (~156 SKUs/bucket — validated)
      compute_bucket_count(100_000)  == 512   (capped — ~195 SKUs/bucket)
      compute_bucket_count(1_500_000) == 512  (capped — ~2930 SKUs/bucket;
                                               row groups still skippable)

    Serving code must use the SAME bucket count the writer used. Pass it as
    a constructor arg to the PyFunc model.
    """
    return min(_BUCKET_COUNT_MAX, max(BUCKET_COUNT, n_skus // _TARGET_SKUS_PER_BUCKET))


# Parquet row group size (bytes). Spark default is ~128MB. Each cold serving
# lookup reads ONE row group (PyArrow uses min/max stats to skip the rest).
# With 4.4MB model blobs, 16MB row groups = ~4 SKUs per group → cold lookup
# reads ~16MB instead of ~128MB → ~8× less IO at UC Volume FUSE bandwidth.
PARQUET_ROW_GROUP_SIZE_BYTES = 16 * 1024 * 1024

# Approximate per-row size used to size Spark partitions for the bundle
# write. The reference model pickles to ~4.4 MB; the bundle stores one row
# per (sku, model_name) with that blob inline. The number is approximate —
# it only sizes partitions, not correctness. If real-world blobs are
# materially different, override via WorkloadBudget.target_bytes_per_task.
_MODEL_BLOB_SIZE_BYTES = 4_400_000


def _sku_bucket(sku: str, bucket_count: int = BUCKET_COUNT) -> int:
    """Bucket function used both at write (in Spark via SQL `substring`+`%`)
    and at serving (in Python via this helper). The numeric suffix of the
    canonical `SKU-NNNNNN` ID is the bucket source so neither side depends
    on hash-function compatibility across languages.

    The serving caller MUST pass the same `bucket_count` the writer used
    (see `compute_bucket_count`). The default exists only for back-compat
    with code that pre-dates dynamic sizing."""
    return int(sku.split("-")[-1]) % bucket_count

_SAMPLE_QUESTIONS = [
    "How many SKUs are tracked per region?",
    "Which 10 SKUs in NORTHEAST have the highest RMSE?",
    "What is the average MAPE per region?",
    "Show me the top 10 worst-performing SKUs by R² across all regions",
    "Compare the RMSE distributions between NORTHEAST and WEST",
    "Which model_name has the lowest mean RMSE overall?",
    "List the 20 best-performing SKUs in MIDWEST by RMSE",
    "What proportion of SKUs have MAPE below 0.10?",
    "Which regions are underperforming on average accuracy?",
]

# (question, sql) pairs — Genie learns the SQL pattern for these and reuses
# the shape for similar future questions.
_EXAMPLE_QUESTION_SQLS = [
    (
        "What is the average RMSE per region?",
        "SELECT region, AVG(rmse) AS avg_rmse FROM {table} GROUP BY region ORDER BY avg_rmse",
    ),
    (
        "Top 10 SKUs by lowest RMSE in NORTHEAST",
        "SELECT sku, model_name, rmse FROM {table} WHERE region = 'NORTHEAST' ORDER BY rmse ASC LIMIT 10",
    ),
    (
        "How many SKUs have MAPE below 0.10 in each region?",
        "SELECT region, COUNT(*) AS good_skus FROM {table} WHERE mape < 0.10 GROUP BY region",
    ),
]

_GENIE_INSTRUCTIONS = [
    "When computing model quality, lower rmse / lower mape / higher r2 = better.",
    "When asked about a region, filter on the `region` column (NORTHEAST/SOUTHEAST/MIDWEST/SOUTHWEST/WEST).",
    "When asked about a model, filter on `model_name` (AutoArima/Prophet/RandomForest).",
    "Each row represents one (region, sku, model_name) trained model.",
    "Use the top-level `rmse`/`mape`/`r2` columns for quick aggregations; the `metrics` MAP has the full set if needed.",
]


def _gid() -> str:
    """32-char lowercase hex UUID, the format the Genie API expects for id fields."""
    return uuid.uuid4().hex


class UCTableLoggingStrategy:
    """`ModelLoggingStrategyPort` — Delta + Genie + N deployable runs/region.

    Args:
        n_endpoints_per_region: How many MLflow experiments+runs to create per
            region. Each run is an independent deployable serving endpoint that
            reads from the shared Delta table and the per-region parquet bundle.
        catalog / schema: Unity Catalog target for the telemetry table.
        spark: SparkSession (needed for the Delta + parquet writes). The
            strategy reaches up to Spark; pass it via constructor so we don't
            hide the dependency.
    """

    name = "uc_table"

    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        n_endpoints_per_region: int = 3,
        budget: WorkloadBudget | None = None,
    ) -> None:
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.n_endpoints_per_region = max(1, n_endpoints_per_region)
        self.budget = budget or WorkloadBudget()
        # Convenience alias — serving code initialises PyArrowLruModel with
        # `budget.bucket_count`, so the property keeps the API stable.
        self.bucket_count = self.budget.bucket_count

    @property
    def telemetry_table_fqn(self) -> str:
        return f"{self.catalog}.{self.schema}.{_TELEMETRY_TABLE_NAME}"

    def log_all(
        self,
        regions: list[RegionSpec],
        plan_repo: RunPlanRepositoryPort,
        config: LoggingConfig,
        plan_df: DataFrame | None = None,
    ) -> LoggingMetrics:
        metrics = LoggingMetrics(strategy=self.name)
        started = time.perf_counter()
        client = MlflowClient(tracking_uri=config.tracking_uri)

        if plan_df is None:
            plan_df = self.spark.read.table(config.run_plan_table)

        log.info(
            f"[{self.name}] log_all start: {len(regions)} region(s), "
            f"n_endpoints_per_region={self.n_endpoints_per_region}, table={self.telemetry_table_fqn}"
        )

        # ─── Phase 1 — UC TELEMETRY TABLE (independent of any region) ──────
        # Idempotent: existence check first, create-empty-only-if-absent. The
        # Delta schema is derived dynamically from TrainedModelTelemetry via
        # SparkSchemaMixin.spark_schema(); changing the dataclass auto-updates
        # the create path. We do NOT alter an existing table — its schema is
        # its existing schema, by design.
        t_p1 = time.perf_counter()
        self._ensure_telemetry_table()

        # Phase 1 write — ALL rows from ALL regions go into the telemetry table.
        # This is one Spark write, completely independent of any per-region
        # artifact bundle write below.
        self._write_full_telemetry_to_uc(plan_df)
        p1_elapsed = time.perf_counter() - t_p1
        log.info(f"[{self.name}] phase 1 complete: telemetry rows in {self.telemetry_table_fqn} (elapsed={p1_elapsed:.1f}s)")

        # ─── Phase 2 — GENIE SPACE (one-shot, independent of regions) ──────
        t_p2 = time.perf_counter()
        genie_space_id, genie_space_name = self._ensure_genie_space()
        p2_elapsed = time.perf_counter() - t_p2
        log.info(f"[{self.name}] phase 2 complete: genie space ready (id={genie_space_id}, elapsed={p2_elapsed:.1f}s)")
        t_p3 = time.perf_counter()

        # ─── Phase 3 — PER-REGION ARTIFACT BUNDLES + DEPLOYABLE RUNS ───────
        # Separate operation from phase 1: each region writes its subset of
        # rows to the experiment-run artifact location as parquet partitioned
        # by SKU (for fast PyArrow lookups at serving time).
        #
        # PARALLELIZED ACROSS REGIONS: each region's per-region work (Spark
        # bundle write + N endpoint MLflow runs) runs in its own driver thread.
        # Spark schedules the 5 simultaneous write jobs across the 6 workers,
        # so we get ~5× wall-clock speedup vs the previous sequential loop.
        with ThreadPoolExecutor(max_workers=len(regions), thread_name_prefix="region") as ex:
            futures = {
                ex.submit(
                    self._build_region,
                    client,
                    plan_df,
                    region,
                    genie_space_id,
                    genie_space_name,
                ): region
                for region in regions
            }
            for fut in as_completed(futures):
                region = futures[fut]
                try:
                    runs_created = fut.result()
                    metrics.total_runs += runs_created
                    metrics.total_artifacts += 1  # 1 bundle per region
                except Exception as e:  # noqa: BLE001
                    metrics.errors.append(f"{region.name}: {e}")
                    log.error(f"  [{region.name}] region build failed: {e}")

        p3_elapsed = time.perf_counter() - t_p3
        metrics.elapsed_seconds = time.perf_counter() - started
        log.info(
            f"[{self.name}] log_all complete: total_runs={metrics.total_runs}, "
            f"total_artifacts={metrics.total_artifacts}, errors={len(metrics.errors)}, "
            f"elapsed={metrics.elapsed_seconds:.1f}s "
            f"[phase1={p1_elapsed:.1f}s phase2={p2_elapsed:.1f}s phase3={p3_elapsed:.1f}s]"
        )
        return metrics

    # ─── Phase 1 — Telemetry table (UC Delta) ─────────────────────────────

    def _ensure_telemetry_table(self) -> None:
        """Create the telemetry table if (and only if) it doesn't exist.

        Strict idempotence: if the table exists, do NOTHING — we don't run a
        CREATE statement, don't add columns, don't touch comments. The
        existing schema is the schema of record. If the dataclass changes
        later, you migrate the table out-of-band; this method won't drift it.

        Schema is derived dynamically from `TrainedModelTelemetry` via the
        SparkSchemaMixin, so adding/removing a field on the dataclass
        affects only fresh creates.
        """
        if self.spark.catalog.tableExists(self.telemetry_table_fqn):
            log.info(f"[{self.name}] telemetry table exists, skipping create: {self.telemetry_table_fqn}")
            return

        log.info(f"[{self.name}] creating telemetry table from TrainedModelTelemetry dataclass…")
        schema = TrainedModelTelemetry.spark_schema()
        empty = self.spark.createDataFrame([], schema=schema)
        (
            empty
            .write
            .format("delta")
            .partitionBy("region")
            .mode("errorifexists")  # belt + suspenders: still fail if we raced with another writer
            .saveAsTable(self.telemetry_table_fqn)
        )

        # Table-level COMMENT — narrative context Genie picks up.
        self.spark.sql(
            f"COMMENT ON TABLE {self.telemetry_table_fqn} IS "
            f"'Per-(region, sku, model) demand-forecasting training telemetry. "
            f"Backs the Genie space `{_GENIE_SPACE_NAME}` for natural-language analysis. "
            f"Model binaries are NOT in this table — they live in the per-region parquet "
            f"bundle referenced by each MLflow endpoint runs `model_bundle_uri` tag, "
            f"partitioned by `sku` for fast lookup.'"
        )

        # Column-level COMMENTs derived from the `Annotated[T, "..."]` metadata
        # on TrainedModelTelemetry. Belt-and-suspenders: the StructField metadata
        # also carries the comment, but ALTER COLUMN guarantees it lands in the
        # Delta column catalog regardless of Spark version behavior.
        for col_name, comment in column_comments_from_dataclass(TrainedModelTelemetry).items():
            escaped = comment.replace("'", "''")
            self.spark.sql(
                f"ALTER TABLE {self.telemetry_table_fqn} "
                f"ALTER COLUMN {col_name} COMMENT '{escaped}'"
            )
        log.info(f"[{self.name}] created telemetry table: {self.telemetry_table_fqn}")

    def _write_full_telemetry_to_uc(self, plan_df: DataFrame) -> None:
        """Replace ALL rows in the telemetry table with this run's rows.

        Uses `TRUNCATE TABLE` + `insertInto` (not `saveAsTable(overwrite)`) so
        we wipe the data but PRESERVE the table's schema, comments, and
        TBLPROPERTIES (notably `genie_space_id` — the cross-run idempotency
        anchor for the Genie space). Without truncation, multi-run dev sessions
        accumulate rows from prior scales (1k, 10k, …) and the next bundle
        write + telemetry write slow down because the table grows unbounded.

        Independent of any per-region artifact-bundle write — this is one
        Spark job that completes before any bundle is written.
        """
        from pyspark.sql.functions import current_timestamp, expr

        # TRUNCATE preserves schema + TBLPROPERTIES; only data is wiped.
        self.spark.sql(f"TRUNCATE TABLE {self.telemetry_table_fqn}")

        telemetry_df = (
            plan_df
            .withColumn("rmse", expr("metrics['rmse']").cast("double"))
            .withColumn("mape", expr("metrics['mape']").cast("double"))
            .withColumn("r2", expr("metrics['r2']").cast("double"))
            .withColumn("logged_at", current_timestamp())
            # Select in the dataclass field order so insertInto matches the
            # existing table schema without positional surprises.
            .select(
                "region", "sku", "model_name",
                "rmse", "mape", "r2",
                "params", "metrics",
                "logged_at",
            )
        )
        # insertInto (not saveAsTable) appends into an existing table without
        # touching its definition or properties. We just truncated above, so
        # the result is a clean replace.
        telemetry_df.write.insertInto(self.telemetry_table_fqn, overwrite=False)

    # ─── Phase 3 helper — one region's full work, runs per-thread ────────

    def _build_region(
        self,
        client: MlflowClient,
        plan_df: DataFrame,
        region: RegionSpec,
        genie_space_id: str,
        genie_space_name: str,
    ) -> int:
        """Do everything for one region in parallel-safe sequence.

        Returns: number of endpoint runs created.
        """
        t_bundle = time.perf_counter()
        bundle_uri = self._write_region_subset_to_artifact_bundle(plan_df, region)
        log.info(f"  [{region.name}] artifact bundle → {bundle_uri} (elapsed={time.perf_counter()-t_bundle:.1f}s)")

        runs_created = 0
        for endpoint_idx in range(self.n_endpoints_per_region):
            experiment_id = self._ensure_endpoint_experiment(client, region, endpoint_idx)
            run = self._create_endpoint_run(
                client=client,
                experiment_id=experiment_id,
                region=region,
                endpoint_idx=endpoint_idx,
                bundle_uri=bundle_uri,
                genie_space_id=genie_space_id,
                genie_space_name=genie_space_name,
            )
            client.set_terminated(run.info.run_id, status="FINISHED")
            runs_created += 1
            log.info(
                f"  [{region.name}] endpoint {endpoint_idx} → run={run.info.run_id} "
                f"(experiment={experiment_id})"
            )
        return runs_created

    # ─── Phase 3 — Per-region artifact bundle (parquet partitioned by SKU) ─

    def _wipe_uc_volume_dir(self, uri: str) -> None:
        from databricks.sdk import WorkspaceClient
        try:
            WorkspaceClient().dbutils.fs.rm(uri, recurse=True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"  wipe skipped for {uri}: {e}")

    def _write_region_subset_to_artifact_bundle(
        self, plan_df: DataFrame, region: RegionSpec
    ) -> str:
        """Write ONE region's subset of rows to a Hive-partitioned parquet bundle.

        Separate operation from `_write_full_telemetry_to_uc`: same source
        DataFrame, different sink (per-region artifact location vs. shared
        UC Delta table) and different shape (partitioned by `sku` for
        constant-time PyArrow lookups at serving time).

        Returns the bundle URI that gets attached to each endpoint run as
        the `model_bundle_uri` tag.

        EXPLICIT WIPE before write: `mode("overwrite").partitionBy(...)`
        does NOT reliably remove stale partition directories on UC Volume.
        Wipe via native UC API: `dbutils.fs.rm(uri, recurse=True)` (single RPC).

        BUCKETED PARTITIONING: previous design `partitionBy("sku")` created
        one directory per SKU → 10k dirs × 5 regions = 50,000 small dirs at
        scale. Phase 3 wall-clock grew super-linearly (iter 2 5k=417s, iter 3
        10k=1161s = 2.8× for 2× scale) due to per-directory metadata
        coordination, not data volume.

        New design buckets SKUs into N=`BUCKET_COUNT` directories. Lookup at
        serving time stays partition-prune-fast because the bucket is a
        deterministic function of the SKU: `_sku_bucket(sku)`. PyArrow can
        still apply both the bucket partition prune AND a predicate filter
        on sku within the bucket.

        Bucket function uses the numeric suffix of `SKU-NNNNNN` IDs so
        Python (serving) and Spark (write) compute the same value without
        depending on hash-function equivalence across languages.
        """
        from pyspark.sql.functions import col as _col, substring as _substring, cast as _cast
        from pyspark.sql.functions import expr as _expr

        base = region.artifact_location or ""
        bundle_uri = f"{base.rstrip('/')}/bundle_parquet"

        t_wipe = time.perf_counter()
        self._wipe_uc_volume_dir(bundle_uri)
        log.info(f"  [{region.name}] wipe done (elapsed={time.perf_counter()-t_wipe:.1f}s)")

        # Memory-bounded partition sizing. Pick n_partitions so each Spark
        # task holds at most `budget.target_bytes_per_task` of model blobs;
        # never go below `bucket_count` so we always have ≥1 file per bucket.
        # Iter 13's OOM was the previous `repartition(BUCKET_COUNT, "bucket")`
        # forcing one task per bucket, which at 20k SKU held ~2 GB per task.
        from math import ceil
        region_rows = self._estimated_region_rows(plan_df, region)
        total_bytes = region_rows * _MODEL_BLOB_SIZE_BYTES
        n_partitions = max(
            self.budget.bucket_count,
            ceil(total_bytes / self.budget.target_bytes_per_task),
        )
        log.info(
            f"  [{region.name}] write plan: rows≈{region_rows:,}, "
            f"n_partitions={n_partitions}, bucket_count={self.budget.bucket_count}, "
            f"target_bytes_per_task={self.budget.target_bytes_per_task // (1024*1024)} MB"
        )

        t_write = time.perf_counter()
        (
            plan_df
            .where(_col("region") == region.name)
            .select("sku", "model_name", "model_blob_bytes", "params", "metrics")
            # bucket = int(sku.split("-")[-1]) % bucket_count
            # SKU shape is `SKU-NNNNNN`; we lift the integer suffix in SQL
            # so Python and Spark agree without sharing a hash impl.
            .withColumn(
                "bucket",
                _expr(f"cast(substring(sku, 5, 10) as int) % {self.bucket_count}"),
            )
            # repartitionByRange distributes by non-overlapping (bucket, sku)
            # ranges ACROSS partitions, but does NOT guarantee sort WITHIN
            # each partition. Without an explicit sortWithinPartitions, the
            # parquet writer emits row groups whose min/max sku spans the
            # whole partition range — defeating row-group skipping at read
            # time. (Verified empirically: Phase 2 retry 4 at 10k with no
            # local sort gave cold p50 = 12s.) Always pair the two.
            .repartitionByRange(n_partitions, "bucket", "sku")
            .sortWithinPartitions("bucket", "sku")
            .write
            .mode("overwrite")
            .partitionBy("bucket")
            # 16MB row groups (vs Spark default 128MB) — cold lookup reads
            # one row group, so a smaller size means less IO per lookup.
            .option("parquet.block.size", str(PARQUET_ROW_GROUP_SIZE_BYTES))
            .parquet(bundle_uri)
        )
        log.info(f"  [{region.name}] parquet write done (elapsed={time.perf_counter()-t_write:.1f}s)")
        return bundle_uri

    def _estimated_region_rows(self, plan_df: DataFrame, region: RegionSpec) -> int:
        """Estimate row count for this region's slice without materialising.

        At scale, `.count()` would scan all rows. We use the plan's total
        row count divided by region count as the estimate. Approximate is
        fine — the result only sizes Spark partitions, not correctness.
        """
        from pyspark.sql.functions import col as _col
        return plan_df.where(_col("region") == region.name).count()

    # ─── Per-endpoint MLflow experiment + run ─────────────────────────────

    def _ensure_endpoint_experiment(
        self,
        client: MlflowClient,
        region: RegionSpec,
        endpoint_idx: int,
    ) -> str:
        """Idempotent: lookup-by-name then create. One experiment per (region, endpoint)."""
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[3]
        name = str(
            project_root
            / f"{region.experiment_name}_endpoint_{endpoint_idx:02d}"
        )
        existing = client.get_experiment_by_name(name)
        if existing is not None:
            return existing.experiment_id
        return client.create_experiment(
            name=name,
            artifact_location=region.artifact_location,
        )

    def _create_endpoint_run(
        self,
        client: MlflowClient,
        experiment_id: str,
        region: RegionSpec,
        endpoint_idx: int,
        bundle_uri: str,
        genie_space_id: str,
        genie_space_name: str,
    ):
        """Create ONE deployable run. Tags are the pointer set the serving endpoint needs."""
        return client.create_run(
            experiment_id=experiment_id,
            tags={
                "mlflow.runName": f"{region.name}_endpoint_{endpoint_idx:02d}",
                "layer": "deployable_endpoint",
                "region": region.name,
                "endpoint_index": str(endpoint_idx),
                "demand_forecasting_table": self.telemetry_table_fqn,
                "model_bundle_uri": bundle_uri,
                "genie_space_id": genie_space_id,
                "genie_space_name": genie_space_name,
            },
        )

    # ─── Genie space + warehouse (idempotent) ─────────────────────────────

    def _ensure_genie_space(self) -> tuple[str, str]:
        """Idempotent: lookup-by-table-property, else create + write back.

        The SDK doesn't expose `list_spaces`, so we can't cheaply find a
        space by name in the workspace. Instead, we persist the space_id
        as a Delta TBLPROPERTY on the telemetry table on first create —
        subsequent runs read it back and reuse. The table is the natural
        anchor: the space points at the table, so the space_id lives WITH
        the table.

        Returns (space_id, space_name); ("", "") on failure (non-fatal —
        strategy continues with empty genie tags on the runs).
        """
        existing_id = self._read_genie_space_id_from_table()
        if existing_id:
            log.info(f"  reusing existing Genie space (from table property): id={existing_id}")
            return (existing_id, _GENIE_SPACE_NAME)

        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
        except Exception as e:  # noqa: BLE001
            log.warning(f"databricks-sdk WorkspaceClient init failed: {e}; genie space disabled")
            return ("", "")

        warehouse_id = self._ensure_warehouse(w)
        if not warehouse_id:
            log.warning("no warehouse available — genie space cannot be created")
            return ("", "")

        try:
            space_id, space_name = self._create_genie_space(w, warehouse_id)
        except Exception as e:  # noqa: BLE001
            log.warning(f"genie space create failed: {e}; continuing without genie tags")
            return ("", "")

        if space_id:
            self._write_genie_space_id_to_table(space_id)
        return (space_id, space_name)

    def _read_genie_space_id_from_table(self) -> str:
        """Read the `genie_space_id` TBLPROPERTY off the telemetry table.

        Empty string if the table doesn't exist or the property isn't set.
        Used for cross-run idempotency on Genie space creation.
        """
        if not self.spark.catalog.tableExists(self.telemetry_table_fqn):
            return ""
        try:
            rows = self.spark.sql(
                f"SHOW TBLPROPERTIES {self.telemetry_table_fqn}"
            ).collect()
            for row in rows:
                if row["key"] == "genie_space_id" and row["value"]:
                    return row["value"]
        except Exception as e:  # noqa: BLE001
            log.warning(f"reading genie_space_id table property failed: {e}")
        return ""

    def _write_genie_space_id_to_table(self, space_id: str) -> None:
        """Persist the genie space id back to the telemetry table as a
        TBLPROPERTY so the next run reuses it instead of creating duplicates."""
        escaped = space_id.replace("'", "''")
        try:
            self.spark.sql(
                f"ALTER TABLE {self.telemetry_table_fqn} "
                f"SET TBLPROPERTIES ('genie_space_id' = '{escaped}')"
            )
            log.info(f"  persisted genie_space_id={space_id} → {self.telemetry_table_fqn} (TBLPROPERTY)")
        except Exception as e:  # noqa: BLE001
            log.warning(f"persisting genie_space_id to table property failed (non-fatal): {e}")

    def _ensure_warehouse(self, w) -> str:
        """Find-or-create a small serverless SQL warehouse to back the Genie space."""
        try:
            for wh in w.warehouses.list():
                if wh.name == _WAREHOUSE_NAME:
                    return wh.id
        except Exception as e:  # noqa: BLE001
            log.warning(f"warehouse list failed: {e}")

        try:
            from databricks.sdk.service.sql import (
                CreateWarehouseRequestWarehouseType,
                EndpointInfoWarehouseType,
            )
            wh = w.warehouses.create_and_wait(
                name=_WAREHOUSE_NAME,
                cluster_size="2X-Small",
                min_num_clusters=1,
                max_num_clusters=1,
                auto_stop_mins=15,
                enable_serverless_compute=True,
                warehouse_type=CreateWarehouseRequestWarehouseType.PRO,
            )
            log.info(f"  created serverless warehouse: {wh.name} (id={wh.id})")
            return wh.id
        except Exception as e:  # noqa: BLE001
            log.warning(f"warehouse create failed: {e}; trying to find any existing serverless warehouse")
            try:
                for wh in w.warehouses.list():
                    if wh.enable_serverless_compute:
                        log.info(f"  falling back to existing serverless warehouse: {wh.name} (id={wh.id})")
                        return wh.id
            except Exception:  # noqa: BLE001
                pass
            return ""

    def _create_genie_space(self, w, warehouse_id: str) -> tuple[str, str]:
        """Create a Genie space via `w.genie.create_space(serialized_space=…)`.

        The Genie create API takes a JSON-stringified `serialized_space` config
        with `version: 2` and a structured spec containing data sources,
        column configs, sample questions, instructions, and example
        question/SQL pairs. This is the schema Genie uses for NL → SQL.
        """
        # Resolve the current-user workspace home for parent_path. Genie spaces
        # live as workspace objects under a user's namespace. If the SDK call
        # fails (rare — typically only when run_as identity is misconfigured),
        # raise rather than fall back to a stale placeholder.
        me = w.current_user.me()
        user_name = me.user_name
        if not user_name:
            raise RuntimeError(
                "WorkspaceClient.current_user.me() returned no user_name; "
                "Genie space creation requires a workspace user identity."
            )
        parent_path = f"/Workspace/Users/{user_name}"

        # Build the structured space config — see the API schema doc. Note that
        # many fields are lists of strings (not strings) — the API expects that.
        # All id-bearing lists must be sorted by id (Genie's validator enforces
        # this for deterministic export-proto comparison).
        sample_questions = sorted(
            [{"id": _gid(), "question": [q]} for q in _SAMPLE_QUESTIONS],
            key=lambda x: x["id"],
        )
        # text_instructions MUST be a single item; the API stacks the instruction
        # strings inside its `content` list (validator: "must contain at most one item").
        text_instructions = [
            {"id": _gid(), "content": list(_GENIE_INSTRUCTIONS)},
        ]
        example_question_sqls = sorted(
            [
                {
                    "id": _gid(),
                    "question": [q],
                    "sql": [sql.format(table=self.telemetry_table_fqn)],
                }
                for q, sql in _EXAMPLE_QUESTION_SQLS
            ],
            key=lambda x: x["id"],
        )

        space_config = {
            "version": 2,
            "config": {
                "sample_questions": sample_questions,
            },
            "data_sources": {
                "tables": [
                    {
                        "identifier": self.telemetry_table_fqn,
                        "description": [
                            "Per-(region, sku, model_name) demand-forecasting training telemetry. "
                            "Each row is one trained model with rmse/mape/r2 as headline metrics "
                            "and full params/metrics maps for detail."
                        ],
                        # column_configs MUST be sorted by column_name — Genie validates this.
                        "column_configs": sorted(
                            [
                                {"column_name": "region", "enable_entity_matching": True},
                                {"column_name": "sku", "enable_entity_matching": True},
                                {"column_name": "model_name", "enable_entity_matching": True},
                                {"column_name": "logged_at", "enable_format_assistance": True},
                            ],
                            key=lambda c: c["column_name"],
                        ),
                    }
                ],
            },
            "instructions": {
                "text_instructions": text_instructions,
                "example_question_sqls": example_question_sqls,
            },
        }
        # Databricks validates these lists as sorted — be defensive even with
        # a single table.
        space_config["data_sources"]["tables"] = sorted(
            space_config["data_sources"]["tables"],
            key=lambda t: t["identifier"],
        )
        serialized_space = json.dumps(space_config, separators=(",", ":"))
        description = (
            f"Genie space for demand-forecasting training telemetry "
            f"(table: {self.telemetry_table_fqn}). Ask in natural language."
        )

        # SDK 0.117.0 doesn't expose `w.genie.create_space`; older versions did
        # under different names. Try the high-level SDK call first; if it isn't
        # available, fall back to the raw REST endpoint which IS public.
        space_id = ""
        try:
            created = w.genie.create_space(
                warehouse_id=warehouse_id,
                title=_GENIE_SPACE_NAME,
                description=description,
                parent_path=parent_path,
                serialized_space=serialized_space,
            )
            space_id = (
                getattr(created, "space_id", None)
                or getattr(created, "id", None)
                or ""
            )
            log.info(f"  genie space created via SDK: id={space_id}")
        except AttributeError as e:
            log.info(f"  w.genie.create_space not available ({e}); falling back to raw REST")
            body = {
                "warehouse_id": warehouse_id,
                "title": _GENIE_SPACE_NAME,
                "description": description,
                "parent_path": parent_path,
                "serialized_space": serialized_space,
            }
            resp = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)
            if isinstance(resp, dict):
                space_id = resp.get("space_id") or resp.get("id") or ""
            log.info(f"  genie space created via REST: id={space_id}")

        if not space_id:
            log.warning(f"genie create returned no id; space may still exist — check the workspace")
            return ("", "")
        log.info(f"  created Genie space: {_GENIE_SPACE_NAME} (id={space_id}, parent={parent_path})")
        return (str(space_id), _GENIE_SPACE_NAME)


# Re-export so adapters/__init__.py picks it up via from-module imports
__all__ = ["UCTableLoggingStrategy"]
