"""Domain entities for the multi-model registration demo.

Each dataclass inherits `SparkSchemaMixin` so adapters can call
`Entity.spark_schema()` to get a `StructType` derived directly from the
declared fields. The mixin is lazy: subclasses with non-mappable field
types (e.g. `list[str]`) only fail when `.spark_schema()` is called, so
inheriting is essentially free.

`DemandRecord` and `TrainedModelRecord` are self-generating: instantiating
them with only the structural args (region/sku/etc.) auto-populates the
rest via faker `default_factory` calls. Adapters only need to pass the
required args.

Faker is imported LAZILY (see `_fake()`), so importing entities on the
import-light serving/routing path does not pull in the data-generation library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Annotated

from dbrx_multimodel_registration.utils.helpers import SparkSchemaMixin

# Faker is imported LAZILY (not at module top-level): entities are value objects
# shared across build-time AND import-light inference/routing code, and only the
# synthetic-data factories below actually need Faker. A top-level `from faker
# import Faker` forced every entity consumer — including the serving hot path —
# to depend on the data-generation library (it's what broke the serving-fleet
# deploy with ModuleNotFoundError: faker). `_fake()` defers the import + instance
# to first use, inside the default_factory calls. On Spark workers each process
# still gets its own Faker the first time a factory fires.
_FAKE = None


def _fake():
    global _FAKE
    if _FAKE is None:
        from faker import Faker
        _FAKE = Faker()
    return _FAKE

_CANONICAL_REGIONS = [
    "NORTHEAST", "SOUTHEAST", "MIDWEST", "SOUTHWEST", "WEST",
    "PACIFIC", "MOUNTAIN", "GREAT_LAKES", "GULF_COAST", "MID_ATLANTIC",
]


@dataclass(frozen=True)
class GenerationSpec(SparkSchemaMixin):
    """Sizing config for synthetic demand data: regions × skus × rows_per_sku."""
    regions: list[str]
    skus: list[str]
    rows_per_sku: int = 100

    @property
    def n_rows(self) -> int:
        return len(self.regions) * len(self.skus) * self.rows_per_sku

    @classmethod
    def of(cls, n_regions: int, n_skus: int, rows_per_sku: int = 100) -> "GenerationSpec":
        if n_regions <= len(_CANONICAL_REGIONS):
            regions = _CANONICAL_REGIONS[:n_regions]
        else:
            regions = [f"REGION_{i:03d}" for i in range(1, n_regions + 1)]
        # Deterministic SKU IDs (not Faker UUIDs) — guarantees each region's
        # run-plan slice has *exactly* n_skus distinct SKUs after the
        # `product(regions, skus, models)` cartesian. UUIDs were fine for
        # realism but made per-region counts vary slightly under shuffles.
        skus = [f"SKU-{i:06d}" for i in range(n_skus)]
        return cls(regions=regions, skus=skus, rows_per_sku=rows_per_sku)


@dataclass(frozen=True)
class RunPlanKey(SparkSchemaMixin):
    """Identity columns of one run-plan entry — the cross-product input."""
    region: str
    sku: str
    model_name: str


@dataclass
class DemandRecord(SparkSchemaMixin):
    """One row of synthetic demand-forecasting data.

    region/sku are required (set by the generator from the GenerationSpec).
    All other fields self-populate via faker default_factory.
    """
    region: str
    sku: str
    date: date = field(default_factory=lambda: _fake().date_between(start_date="-1y", end_date="today"))
    product_id: str = field(default_factory=lambda: f"PROD_{_fake().random_int(min=1, max=50):03d}")
    demand: int = field(default_factory=lambda: _fake().random_int(min=10, max=500))
    price: float = field(default_factory=lambda: round(_fake().pyfloat(min_value=9.99, max_value=299.99), 2))
    day_of_week: int = field(init=False)
    promotion: bool = field(default_factory=lambda: _fake().boolean(chance_of_getting_true=30))
    inventory: int = field(default_factory=lambda: _fake().random_int(min=50, max=1000))

    def __post_init__(self):
        # init=False fields must be set in __post_init__. On frozen dataclasses
        # this would need object.__setattr__; we keep this one non-frozen so
        # day_of_week can be derived from the (also-random) date.
        self.day_of_week = self.date.weekday()


def _mock_params() -> dict[str, str]:
    """10 mock MLflow params. Module-level so default_factory can call it per-instance."""
    return {
        "learning_rate":     f"{_fake().pyfloat(min_value=1e-4, max_value=1e-2):.5f}",
        "batch_size":        str(_fake().random_element(elements=(32, 64, 128, 256))),
        "epochs":            str(_fake().random_int(min=10, max=100)),
        "dropout":           f"{_fake().pyfloat(min_value=0.0, max_value=0.5):.2f}",
        "l2_regularization": f"{_fake().pyfloat(min_value=1e-5, max_value=1e-3):.6f}",
        "optimizer":         _fake().random_element(elements=("adam", "sgd", "rmsprop")),
        "model_type":        "RandomForest",
        "dataset_version":   f"v{_fake().random_int(min=1, max=10)}",
        "features_count":    str(_fake().random_int(min=4, max=20)),
        "target_metric":     _fake().random_element(elements=("demand", "revenue", "units_sold")),
    }


def _mock_metrics() -> dict[str, float]:
    """10 mock MLflow metrics. Module-level so default_factory can call it per-instance."""
    return {
        "rmse":                  round(_fake().pyfloat(min_value=5.0, max_value=25.0), 4),
        "mape":                  round(_fake().pyfloat(min_value=0.05, max_value=0.30), 4),
        "r2":                    round(_fake().pyfloat(min_value=0.40, max_value=0.95), 4),
        "mae":                   round(_fake().pyfloat(min_value=3.0, max_value=20.0), 4),
        "mse":                   round(_fake().pyfloat(min_value=25.0, max_value=625.0), 4),
        "accuracy":              round(_fake().pyfloat(min_value=0.70, max_value=0.95), 4),
        "precision":             round(_fake().pyfloat(min_value=0.70, max_value=0.95), 4),
        "recall":                round(_fake().pyfloat(min_value=0.70, max_value=0.95), 4),
        "f1":                    round(_fake().pyfloat(min_value=0.70, max_value=0.95), 4),
        "training_time_seconds": round(_fake().pyfloat(min_value=1.0, max_value=120.0), 2),
    }


@dataclass
class TrainedModelRecord(SparkSchemaMixin):
    """One trained-model entry emitted by the training simulator.

    Structural fields (region, sku, model_name, model_blob_bytes) are required.
    params and metrics auto-generate from `_mock_params()` / `_mock_metrics()`
    via faker — the simulator never touches them.
    """
    region: str
    sku: str
    model_name: str
    model_blob_bytes: bytes
    params: dict[str, str] = field(default_factory=_mock_params)
    metrics: dict[str, float] = field(default_factory=_mock_metrics)


@dataclass
class TrainedModelTelemetry(SparkSchemaMixin):
    """One row of per-(region, sku, model) training telemetry.

    Schema for the `demand_forecasting_artifacts` UC Delta table backing the
    Genie space. Same shape as TrainedModelRecord but without the blob (blobs
    live in the per-region parquet artifact bundle, queried at serving time
    by partition pruning on `sku`).

    Field order = table column order. `UCTableLoggingStrategy` derives the
    Spark schema from this dataclass via `SparkSchemaMixin.spark_schema()`,
    AND extracts the `Annotated[..., "comment"]` strings via
    `column_comments_from_dataclass()` to set Delta column comments — both
    used by Genie to write better NL→SQL.
    """
    region: Annotated[str, "Sales region the (sku, model) was trained for. One of NORTHEAST, SOUTHEAST, MIDWEST, SOUTHWEST, WEST."]
    sku: Annotated[str, "Stock-keeping unit identifier. Each SKU has one trained model per `model_name` per region."]
    model_name: Annotated[str, "Model algorithm used to train. One of AutoArima, Prophet, RandomForest."]
    rmse: Annotated[float, "Root-mean-squared error on holdout data. Lower is better. Use to compare model fit across SKUs."]
    mape: Annotated[float, "Mean absolute percentage error. Lower is better. Scale-free — comparable across SKUs of different demand magnitudes."]
    r2: Annotated[float, "Coefficient of determination (R-squared). Closer to 1.0 is better; can be negative for very bad fits."]
    params: Annotated[dict[str, str], "Full training hyperparameters as a MAP<STRING, STRING> — e.g. learning_rate, batch_size, n_estimators, dropout."]
    metrics: Annotated[dict[str, float], "Full evaluation metrics as a MAP<STRING, DOUBLE> — includes rmse/mape/r2 (also broken out as top-level columns) plus accuracy/precision/recall/f1 etc."]
    logged_at: Annotated[datetime, "When this telemetry row was written to the table."]


@dataclass
class ModelArtifactRow(SparkSchemaMixin):
    """One row of the unified `model_artifacts` table — the single source of
    truth that replaces the separate `run_plan` (blobs) and telemetry tables.

    Carries BOTH the model blob (serving source) AND the Genie-queryable
    telemetry columns, plus a `bucket` column (bucket = f(sku)) so the serving
    populate step can filter per shard by whole buckets. Genie reads a
    blob-excluding VIEW over this table (no data copy) — column pruning means
    NL→SQL over rmse/mape/etc never touches the blob.

    Field order = table column order. Annotated comment strings drive Genie's
    NL→SQL (extracted via `column_comments_from_dataclass`). `model_blob_bytes`
    is intentionally NOT in the Genie view.
    """
    region: Annotated[str, "Sales region the (sku, model) was trained for. One of NORTHEAST, SOUTHEAST, MIDWEST, SOUTHWEST, WEST."]
    sku: Annotated[str, "Stock-keeping unit identifier. Each SKU has one trained model per `model_name` per region."]
    model_name: Annotated[str, "Model algorithm used to train. One of AutoArima, Prophet, RandomForest."]
    bucket: Annotated[int, "Serving shard bucket = cast(substring(sku,5,10) as int) %% bucket_count. The unit the serving fleet groups into per-endpoint shards."]
    rmse: Annotated[float, "Root-mean-squared error on holdout data. Lower is better. Use to compare model fit across SKUs."]
    mape: Annotated[float, "Mean absolute percentage error. Lower is better. Scale-free — comparable across SKUs of different demand magnitudes."]
    r2: Annotated[float, "Coefficient of determination (R-squared). Closer to 1.0 is better; can be negative for very bad fits."]
    params: Annotated[dict[str, str], "Full training hyperparameters as a MAP<STRING, STRING>."]
    metrics: Annotated[dict[str, float], "Full evaluation metrics as a MAP<STRING, DOUBLE> — rmse/mape/r2 broken out above plus accuracy/precision/recall/f1 etc."]
    model_blob_bytes: Annotated[bytes, "Pickled trained model bytes. Serving source — EXCLUDED from the Genie view."]

    # Columns the Genie view exposes (everything except the blob).
    GENIE_VIEW_COLUMNS = ("region", "sku", "model_name", "bucket", "rmse", "mape", "r2", "params", "metrics")


@dataclass(frozen=True)
class RegionSpec(SparkSchemaMixin):
    """A region and its MLflow placement.

    `experiment_name` follows the PRD convention `Demand_Forecasting-[REGION]`.
    `artifact_location` is a UC Volume path; None lets MLflow pick a default.
    """
    name: str
    experiment_name: str
    artifact_location: str | None = None


@dataclass(frozen=True)
class LoggingConfig(SparkSchemaMixin):
    """Tunables for the logging strategies."""
    concurrency: int = 16
    async_logging: bool = True
    http_retries: int = 7
    http_timeout: int = 120
    run_plan_table: str = "main.demo.run_plan"
    artifact_volume: str | None = None
    tracking_uri: str | None = None


@dataclass(frozen=True)
class WorkloadBudget:
    """Cluster-independent Spark task sizing for the bundle write.

    Three explicit knobs control resource usage so the same code scales
    across cluster sizes without per-test tuning:

    - `target_bytes_per_task`: Spark partition count is sized so each task
      holds at most this many bytes of model blobs. Avoids Python worker OOM
      regardless of `n_skus`.
    - `max_concurrent_tasks`: hard cap on cluster-wide concurrent tasks.
      Useful when shuffle managers / GC need headroom; default leaves some
      cluster capacity unused intentionally.
    - `bucket_count`: read-side directory fan-out (independent of write-time
      partition count). Picked from cold-lookup tuning.

    The write path uses `repartitionByRange(n_partitions, "bucket", "sku")`
    where `n_partitions = max(bucket_count, total_bytes / target_bytes_per_task)`,
    producing multiple files per bucket dir with non-overlapping sku ranges.
    PyArrow row-group skipping still applies file-by-file via parquet stats.
    """
    # 2 GB per task is the empirical sweet spot on this cluster (m5d.2xlarge,
    # 32 GB workers, ~8 cores each, ~2 concurrent tasks/worker for safety).
    # - 500 MB target at 10k → 4 files/bucket → cold p50 = 3286 ms (Phase 2)
    # - 2 GB target at 10k → ~1 file/bucket → cold p50 ≈ 704 ms (matches iter 10)
    # At 20k it becomes ~2 files/bucket (slight regression OK; primary
    # constraint is avoiding the iter-11 OOM at 4 GB/task).
    target_bytes_per_task: int = 2 * 1024 * 1024 * 1024
    max_concurrent_tasks: int = 12
    bucket_count: int = 64


@dataclass(frozen=True)
class EndpointCapacityPlan:
    """How many Model Serving endpoints hold `n_models` under a per-endpoint
    disk cap, given a measured on-disk footprint per model.

    Disk-bound, not RAM-bound: the v3 serving design keeps the whole per-region
    parquet bundle on the endpoint's local disk and unpickles only the working
    set per request. So the constraint is `per_endpoint_bytes` of bundle on
    disk, and capacity = floor(cap / bytes_per_model).

    `bytes_per_model` must come from a MEASUREMENT of DISTINCT models written
    through the real (snappy parquet) bundle path — not the demo's raw 4.4 MB
    blob constant. The demo broadcasts one model and reuses its bytes, so a
    demo bundle compresses ~losslessly and would report a fraction of the true
    footprint. Feed the number from `measure_bytes_per_model(...)`.

    `pack_by_region=True` mirrors the diagram's region-sharded fleet: models
    split evenly across `n_regions`, each region packed onto its own endpoints
    so no shard straddles two endpoints (a SKU's region determines its
    endpoint). This rounds up per region and is what governs the real fleet
    size. Set False for the region-agnostic lower bound (global bin-packing).
    """
    n_models: int
    bytes_per_model: float
    per_endpoint_bytes: int
    n_regions: int = 1
    pack_by_region: bool = True

    def __post_init__(self) -> None:
        if self.bytes_per_model <= 0:
            raise ValueError("bytes_per_model must be positive")
        if self.per_endpoint_bytes <= 0:
            raise ValueError("per_endpoint_bytes must be positive")
        if self.bytes_per_model > self.per_endpoint_bytes:
            raise ValueError(
                f"one model ({self.bytes_per_model:.0f} B) exceeds the per-endpoint "
                f"cap ({self.per_endpoint_bytes} B) — no packing is possible"
            )
        if self.n_regions < 1:
            raise ValueError("n_regions must be >= 1")

    @property
    def total_bytes(self) -> float:
        return self.n_models * self.bytes_per_model

    @property
    def models_per_endpoint(self) -> int:
        """Max models that fit on one endpoint (floor — no partial model)."""
        return int(self.per_endpoint_bytes // self.bytes_per_model)

    @property
    def endpoints_needed(self) -> int:
        """Endpoints to hold all models, respecting the sharding policy."""
        from math import ceil

        cap = self.models_per_endpoint
        if not self.pack_by_region:
            return ceil(self.n_models / cap)
        # Region-sharded: each region packed independently. Distribute models
        # as evenly as possible, then size the fleet from the heaviest region.
        base, extra = divmod(self.n_models, self.n_regions)
        per_region_counts = [base + (1 if i < extra else 0) for i in range(self.n_regions)]
        return sum(ceil(c / cap) for c in per_region_counts if c > 0)

    @property
    def utilization(self) -> float:
        """Fraction of provisioned endpoint disk actually used (0..1)."""
        provisioned = self.endpoints_needed * self.per_endpoint_bytes
        return self.total_bytes / provisioned if provisioned else 0.0

    @classmethod
    def from_gb(
        cls,
        n_models: int,
        bytes_per_model: float,
        per_endpoint_gb: float,
        n_regions: int = 1,
        pack_by_region: bool = True,
    ) -> "EndpointCapacityPlan":
        """Build a plan from a per-endpoint cap expressed in GB (10^9 bytes)."""
        return cls(
            n_models=n_models,
            bytes_per_model=bytes_per_model,
            per_endpoint_bytes=int(per_endpoint_gb * 1_000_000_000),
            n_regions=n_regions,
            pack_by_region=pack_by_region,
        )


# Eng-confirmed hard ceiling for a baked serving-endpoint artifact (2026-07-28).
# A shard bundle must stay under this or the endpoint's container build fails
# ("internal error"). Governs the minimum SHARD_COUNT for a region.
MAX_SERVING_ARTIFACT_GB = 768.0


@dataclass(frozen=True)
class ShardPlacement:
    """One shard of a region's models → the endpoint that serves it.

    A shard is a group of bundle buckets (see `ShardMappingPort`). This value
    object is the unit the fleet builds (one per endpoint) and the row the
    routing index stores. `sku_min`/`sku_max` are the SKU-range the shard's
    bundle actually holds (read from parquet footer stats), so the router can
    resolve `(region, sku) → endpoint` by range without recomputing the hash.
    """
    region: str
    shard: int
    shard_count: int
    endpoint_name: str
    bundle_uri: str
    sku_min: str | None = None
    sku_max: str | None = None


@dataclass(frozen=True)
class ServingDeployment:
    """Outcome of deploying a serving fleet — the return type of the serving
    deployment-strategy port. Describes what was created so callers can route,
    inspect, or tear down without re-deriving it.
    """
    experiment: str
    shard_count: int
    placements: list[ShardPlacement]
    router_endpoint: str | None = None
    index_uri: str | None = None

    @property
    def endpoint_names(self) -> list[str]:
        return [p.endpoint_name for p in self.placements]


@dataclass
class LoggingMetrics(SparkSchemaMixin):
    """Outcome of a logging run, returned by strategies."""
    strategy: str
    total_runs: int = 0
    total_artifacts: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def throughput_runs_per_sec(self) -> float:
        return self.total_runs / self.elapsed_seconds if self.elapsed_seconds else 0.0