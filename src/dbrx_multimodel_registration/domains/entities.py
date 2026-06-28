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

Module-level `fake = Faker()` runs on import — on Spark workers, each Python
process gets its own Faker instance the first time it imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Annotated

from faker import Faker

from dbrx_multimodel_registration.utils.helpers import SparkSchemaMixin

fake = Faker()

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
    date: date = field(default_factory=lambda: fake.date_between(start_date="-1y", end_date="today"))
    product_id: str = field(default_factory=lambda: f"PROD_{fake.random_int(min=1, max=50):03d}")
    demand: int = field(default_factory=lambda: fake.random_int(min=10, max=500))
    price: float = field(default_factory=lambda: round(fake.pyfloat(min_value=9.99, max_value=299.99), 2))
    day_of_week: int = field(init=False)
    promotion: bool = field(default_factory=lambda: fake.boolean(chance_of_getting_true=30))
    inventory: int = field(default_factory=lambda: fake.random_int(min=50, max=1000))

    def __post_init__(self):
        # init=False fields must be set in __post_init__. On frozen dataclasses
        # this would need object.__setattr__; we keep this one non-frozen so
        # day_of_week can be derived from the (also-random) date.
        self.day_of_week = self.date.weekday()


def _mock_params() -> dict[str, str]:
    """10 mock MLflow params. Module-level so default_factory can call it per-instance."""
    return {
        "learning_rate":     f"{fake.pyfloat(min_value=1e-4, max_value=1e-2):.5f}",
        "batch_size":        str(fake.random_element(elements=(32, 64, 128, 256))),
        "epochs":            str(fake.random_int(min=10, max=100)),
        "dropout":           f"{fake.pyfloat(min_value=0.0, max_value=0.5):.2f}",
        "l2_regularization": f"{fake.pyfloat(min_value=1e-5, max_value=1e-3):.6f}",
        "optimizer":         fake.random_element(elements=("adam", "sgd", "rmsprop")),
        "model_type":        "RandomForest",
        "dataset_version":   f"v{fake.random_int(min=1, max=10)}",
        "features_count":    str(fake.random_int(min=4, max=20)),
        "target_metric":     fake.random_element(elements=("demand", "revenue", "units_sold")),
    }


def _mock_metrics() -> dict[str, float]:
    """10 mock MLflow metrics. Module-level so default_factory can call it per-instance."""
    return {
        "rmse":                  round(fake.pyfloat(min_value=5.0, max_value=25.0), 4),
        "mape":                  round(fake.pyfloat(min_value=0.05, max_value=0.30), 4),
        "r2":                    round(fake.pyfloat(min_value=0.40, max_value=0.95), 4),
        "mae":                   round(fake.pyfloat(min_value=3.0, max_value=20.0), 4),
        "mse":                   round(fake.pyfloat(min_value=25.0, max_value=625.0), 4),
        "accuracy":              round(fake.pyfloat(min_value=0.70, max_value=0.95), 4),
        "precision":             round(fake.pyfloat(min_value=0.70, max_value=0.95), 4),
        "recall":                round(fake.pyfloat(min_value=0.70, max_value=0.95), 4),
        "f1":                    round(fake.pyfloat(min_value=0.70, max_value=0.95), 4),
        "training_time_seconds": round(fake.pyfloat(min_value=1.0, max_value=120.0), 2),
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
