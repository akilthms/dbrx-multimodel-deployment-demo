"""Demo entrypoint: 5-stage pipeline.

  1. Generate mock demand dataset (regions × skus × rows_per_sku)
  2. Train ONE reference RandomForest on one (region, sku) slice
  3. Broadcast the reference model bytes
  4. Simulate distributed training (cross-product → TrainedModelRecord DataFrame)
  5. Multithreaded MLflow logging to one experiment per region

Run locally with the package installed editable:
    pip install -e .
    python src/main.py --strategy collapsed_sku --regions 2 --skus 100 --rows-per-sku 10

On Databricks, the DAB job runs this file via `spark_python_task.python_file`.
The `dbrx_multimodel_registration` package is installed on the cluster as a
wheel library; this script just imports from it.
"""
from __future__ import annotations

import os

# Must precede any mlflow import anywhere in the process.
os.environ.setdefault("DISABLE_MLFLOWDBFS", "true")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")

import argparse
import logging
import sys
import uuid
from dataclasses import asdict

from pyspark.sql import SparkSession

from dbrx_multimodel_registration.adapters import (
    CollapsedSkuLoggingStrategy,
    DeltaRunPlanRepository,
    NestedModelLoggingStrategy,
    ReferenceModelTrainer,
    RegionArtifactOnlyStrategy,
    SparkDataGenerator,
    TrainingSimulator,
    UCTableLoggingStrategy,
)
from dbrx_multimodel_registration.domains.entities import (
    DemandRecord,
    GenerationSpec,
    LoggingConfig,
    RegionSpec,
)
from dbrx_multimodel_registration.ports.logging import ModelLoggingStrategyPort
from dbrx_multimodel_registration.utils.helpers import configure_mlflow_http_pool

log = logging.getLogger("dbrx_multimodel_registration")

_DEFAULT_MODELS = ["AutoArima", "Prophet", "RandomForest"]
_MAX_REGIONS = 10  # bounded by _CANONICAL_REGIONS list in entities.py


def _build_strategy(args: argparse.Namespace, spark: "SparkSession") -> ModelLoggingStrategyPort:
    name = args.strategy
    if name == "uc_table":
        if not args.catalog or not args.schema:
            raise SystemExit("--catalog and --schema are required for uc_table strategy")
        return UCTableLoggingStrategy(
            spark=spark,
            catalog=args.catalog,
            schema=args.schema,
            n_endpoints_per_region=args.n_endpoints_per_region,
        )
    registry: dict[str, ModelLoggingStrategyPort] = {
        "collapsed_sku": CollapsedSkuLoggingStrategy(),
        "nested_model": NestedModelLoggingStrategy(per_model_artifact=args.per_model_artifact),
        "region_artifact_only": RegionArtifactOnlyStrategy(),
    }
    if name not in registry:
        raise SystemExit(f"unknown strategy: {name!r}. choices: {sorted(registry)} or uc_table")
    return registry[name]


def _configure_mlflow(tracking_uri: str | None) -> None:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.autolog(disable=True)
    try:
        mlflow.tracing.disable()
    except Exception:
        pass
    try:
        mlflow.disable_system_metrics_logging()
    except Exception:
        pass
    try:
        mlflow.config.enable_async_logging()
    except Exception:
        log.warning("async logging unavailable; falling back to sync")


def _build_spark(app_name: str) -> SparkSession:
    builder = SparkSession.builder.appName(app_name).config(
        "spark.sql.execution.arrow.pyspark.enabled", "true"
    )
    return builder.getOrCreate()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dbrx-multimodel-register")
    p.add_argument(
        "--strategy",
        default="collapsed_sku",
        choices=["collapsed_sku", "nested_model", "region_artifact_only", "uc_table"],
    )
    p.add_argument("--catalog", default=None,
                   help="Unity Catalog (uc_table strategy).")
    p.add_argument("--schema", default=None,
                   help="UC schema for the telemetry table (uc_table strategy).")
    p.add_argument("--n-endpoints-per-region", type=int, default=3,
                   help="uc_table strategy: number of deployable MLflow runs per region (one per serving endpoint).")
    p.add_argument("--per-model-artifact", action="store_true",
                   help="Only honored by nested_model; uploads one artifact per model run (slow).")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--regions", type=int, default=5,
                   help=f"Number of regions (1..{_MAX_REGIONS} picks from the canonical pool).")
    p.add_argument("--skus", type=int, default=1_000,
                   help="Number of SKUs in the pool (each region gets exactly this many in the run plan).")
    p.add_argument("--rows-per-sku", type=int, default=100,
                   help="Demand records per (region, sku) pair.")
    p.add_argument("--models", nargs="+", default=_DEFAULT_MODELS)
    p.add_argument("--ref-model-n-estimators", type=int, default=500,
                   help="RandomForest n_estimators for the reference model.")
    p.add_argument("--ref-model-max-depth", type=int, default=20,
                   help="RandomForest max_depth for the reference model.")
    p.add_argument("--run-plan-table", default=None,
                   help="UC table (e.g. main.demo.run_plan) or local path. Defaults to a tmp parquet path.")
    p.add_argument("--artifact-volume", default=None,
                   help="UC Volume base path for experiment artifacts (one subdir per region).")
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI. None = MLflow default (local mlruns/ when off-Databricks).")
    p.add_argument("--experiment-prefix", default="Demand_Forecasting")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv or sys.argv[1:])

    if args.regions < 1:
        raise SystemExit("--regions must be >= 1")

    configure_mlflow_http_pool(n_regions=args.regions, concurrency_per_region=args.concurrency)

    run_plan_table = args.run_plan_table or f"/tmp/dbrx_runplan_{uuid.uuid4().hex[:8]}"

    spec = GenerationSpec.of(args.regions, args.skus, args.rows_per_sku)
    log.info(
        "spec: %d regions × %d skus × %d rows/sku = %d demand rows; "
        "run plan = %d × |models|=%d entries",
        len(spec.regions), len(spec.skus), spec.rows_per_sku, spec.n_rows,
        len(spec.regions) * len(spec.skus), len(args.models),
    )
    log.info(
        "strategy=%s concurrency=%d plan=%s",
        args.strategy, args.concurrency, run_plan_table,
    )

    spark = _build_spark(f"dbrx-multimodel-register-{args.strategy}")
    _configure_mlflow(args.tracking_uri)

    # Stage 1: generate mock demand dataset
    generator = SparkDataGenerator(spark=spark, entity_type=DemandRecord)
    demand_df = generator.generate(spec)

    # Stage 2: train ONE reference model on one (region, sku) slice
    ref_region, ref_sku = spec.regions[0], spec.skus[0]
    log.info("training reference model on (region=%s, sku=%s)…", ref_region, ref_sku)
    ref_bytes = ReferenceModelTrainer(
        n_estimators=args.ref_model_n_estimators,
        max_depth=args.ref_model_max_depth,
    ).train(demand_df, ref_region, ref_sku)
    log.info("reference model: %.2f MB (n_estimators=%d, max_depth=%d)",
             len(ref_bytes) / (1024 * 1024), args.ref_model_n_estimators, args.ref_model_max_depth)

    # Persist the reference model to UC Volume — same artifact preservation
    # step the notebook does in Stage 2. The simulator still consumes bytes
    # (not the path), but writing the blob to UC gives us a stable, inspectable
    # artifact and parity with the notebook flow.
    if args.artifact_volume:
        ref_model_path = f"{args.artifact_volume.rstrip('/')}/reference_model/model.pkl"
        import errno as _errno
        try:
            os.makedirs(os.path.dirname(ref_model_path), exist_ok=True)
        except OSError as e:
            # WSFS quirk: os.makedirs walks the UC volume root and FUSE returns
            # errno 95 (EOPNOTSUPP) instead of EEXIST.
            if e.errno not in (_errno.EEXIST, _errno.EOPNOTSUPP):
                raise
        with open(ref_model_path, "wb") as f:
            f.write(ref_bytes)
        log.info("reference model saved: %s", ref_model_path)

    # Stages 3+4: simulate distributed training — broadcast the reference bytes
    # and emit model_blob per row via the pandas_udf's `model_training_fn`.
    plan_df = TrainingSimulator(spark=spark).simulate(ref_bytes, spec, args.models)

    # Persist run plan (resumable, streamable)
    plan_repo = DeltaRunPlanRepository(spark=spark)
    plan_repo.persist(plan_df, run_plan_table)
    log.info("run plan persisted: %s", run_plan_table)

    # Build region specs from the GenerationSpec. MLflow REST requires a URI
    # scheme on artifact_location; we use dbfs:/Volumes/... (resolves to the
    # FUSE mount /Volumes/... that the parquet writer reads/writes).
    regions = [
        RegionSpec(
            name=r,
            experiment_name=f"{args.experiment_prefix}-{r}",
            artifact_location=(
                f"dbfs:{args.artifact_volume.rstrip('/')}/{r}"
                if args.artifact_volume else None
            ),
        )
        for r in spec.regions
    ]

    # Stage 5: run the logging strategy
    config = LoggingConfig(
        concurrency=args.concurrency,
        run_plan_table=run_plan_table,
        artifact_volume=args.artifact_volume,
        tracking_uri=args.tracking_uri,
    )
    strategy = _build_strategy(args, spark)
    metrics = strategy.log_all(regions, plan_repo, config)

    log.info("=" * 60)
    log.info("LoggingMetrics: %s", asdict(metrics))
    log.info("throughput=%.1f runs/sec", metrics.throughput_runs_per_sec)
    if metrics.errors:
        log.error("errors=%d (first 5: %s)", len(metrics.errors), metrics.errors[:5])
    log.info("=" * 60)

    return 0 if not metrics.errors else 2


if __name__ == "__main__":
    sys.exit(main())
