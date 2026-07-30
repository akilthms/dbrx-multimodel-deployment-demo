"""dbrx-mmd CLI — serving endpoint capacity planning.

Answers: given a per-endpoint disk cap, how many Model Serving endpoints does
a model fleet of size N need? Measures the real on-disk bytes/model through the
production parquet bundle path (distinct models, snappy) rather than assuming
the raw pickle size, then does the packing math.
"""
from __future__ import annotations

import typer

from dbrx_multimodel_registration.adapters.serving.capacity import (
    measure_bundle,
    plan_endpoints,
)

app = typer.Typer(help="Serving endpoint capacity planning.")


def _fmt_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1_000_000_000:.2f} GB"


@app.command()
def plan(
    n_models: int = typer.Option(500_000, "-n", "--models", help="Total models to serve."),
    per_endpoint_gb: float = typer.Option(30.0, "--endpoint-gb", help="Disk (bundle) an endpoint can hold, in GB."),
    regions: int = typer.Option(5, "-r", "--regions", help="Regions to shard across (fleet is region-packed)."),
    global_pack: bool = typer.Option(False, "--global-pack", help="Ignore regions; bin-pack globally (lower bound)."),
    sample_models: int = typer.Option(64, "--sample", help="Distinct models to train when measuring bytes/model."),
    bytes_per_model: float = typer.Option(0.0, "--bytes-per-model", help="Skip measurement; use this on-disk bytes/model."),
) -> None:
    """Measure on-disk bytes/model, then report endpoints needed for N models."""
    measurement = None
    if bytes_per_model > 0:
        effective = bytes_per_model
        typer.echo(f"Using supplied bytes/model: {effective:,.0f} B ({_fmt_gb(effective)}/model)")
    else:
        typer.echo(f"Measuring on-disk footprint from {sample_models} distinct models (snappy parquet)…")
        measurement = measure_bundle(n_models=sample_models)
        effective = measurement.bytes_per_model
        typer.echo(
            f"  raw pickle:  {measurement.raw_bytes_per_model:,.0f} B/model\n"
            f"  on disk:     {effective:,.0f} B/model  "
            f"(snappy {measurement.compression_ratio:.2f}× vs raw)"
        )

    plan_result, _ = plan_endpoints(
        n_models=n_models,
        per_endpoint_gb=per_endpoint_gb,
        n_regions=regions,
        pack_by_region=not global_pack,
        measurement=measurement,
        sample_models=sample_models,
    )

    policy = "global bin-pack" if global_pack else f"region-sharded ({regions} regions)"
    typer.echo("")
    typer.echo(f"Fleet plan — {policy}")
    typer.echo(f"  models:               {n_models:,}")
    typer.echo(f"  total bundle size:    {_fmt_gb(plan_result.total_bytes)}")
    typer.echo(f"  per-endpoint cap:     {per_endpoint_gb:g} GB → {plan_result.models_per_endpoint:,} models/endpoint")
    typer.echo(f"  ENDPOINTS NEEDED:     {plan_result.endpoints_needed}")
    typer.echo(f"  disk utilization:     {plan_result.utilization:.0%}")


@app.command()
def stress(
    source: str = typer.Argument(..., help="Existing bundle dir, e.g. /Volumes/<cat>/<sch>/mlflow_artifacts/<REGION>/bundle_parquet."),
    model_name: str = typer.Option(..., "--model-name", help="UC model name (catalog.schema.name) to register the probe under."),
    target_gb: float = typer.Option(30.0, "--target-gb", help="On-disk artifact size to build and deploy."),
    endpoint: str = typer.Option("mmd-storage-probe", "--endpoint", help="Serving endpoint name to create/update."),
    experiment: str = typer.Option(..., "--experiment", help="MLflow experiment workspace path (required — wheel tasks have no default)."),
    workdir: str = typer.Option("/local_disk0/mmd_probe_artifact", "--workdir", help="Cluster-local scratch to assemble the artifact (NOT a FUSE Volume)."),
    workload_size: str = typer.Option("Small", "--workload-size", help="Serving workload size (Small/Medium/Large)."),
    build_only: bool = typer.Option(False, "--build-only", help="Only build + log the model; skip endpoint create (dry run)."),
) -> None:
    """Design-A storage-ceiling test: assemble a TARGET_GB artifact from an
    existing bundle, bake it into a probe model, deploy, and report acceptance.

    Auto-picks the assembly strategy: if the source is LARGER than target, it
    carves a subset of whole bucket=* partitions down to target; if smaller, it
    replicates the source up to target. Intended to run ON a cluster (via the
    `storage_probe` bundle job) so the source Volume is a local FUSE read.

    Creates real, billable resources — deploys ONE endpoint; delete it after
    reading the result.
    """
    import shutil
    from pathlib import Path

    from dbrx_multimodel_registration.adapters.serving.stress import (
        build_probe_artifact,
        deploy_probe_endpoint,
        get_endpoint_status,
        log_probe_model,
        select_subset_to_size,
    )

    shutil.rmtree(workdir, ignore_errors=True)  # clean any prior run's scratch
    src = Path(source)
    src_bytes = (
        sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        if src.is_dir()
        else src.stat().st_size
    )
    if src.is_dir() and src_bytes >= int(target_gb * 1_000_000_000):
        typer.echo(f"Carving ~{target_gb:g} GB subset from {source} ({_fmt_gb(src_bytes)} available) …")
        plan = select_subset_to_size(source, workdir, target_gb)
        typer.echo(f"  assembled {_fmt_gb(plan.source_bytes)} at {workdir}")
    else:
        typer.echo(f"Replicating {source} ({_fmt_gb(src_bytes)}) up to ~{target_gb:g} GB …")
        plan = build_probe_artifact(source, workdir, target_gb)
        typer.echo(f"  {plan.n_copies} copies → {_fmt_gb(plan.resulting_bytes)} at {workdir}")

    typer.echo(f"Logging probe model to UC: {model_name} (experiment {experiment}) …")
    model_uri = log_probe_model(workdir, model_name, experiment)
    version = model_uri.rsplit("/", 1)[-1]
    typer.echo(f"  registered {model_uri}")

    if build_only:
        typer.echo("--build-only set; skipping endpoint deploy.")
        raise typer.Exit(code=0)

    typer.echo(f"Deploying endpoint '{endpoint}' (workload={workload_size}) …")
    deploy_probe_endpoint(endpoint, model_name, version, workload_size=workload_size)
    status = get_endpoint_status(endpoint)
    typer.echo(
        f"  submitted. status: ready={status['ready']} config_update={status['config_update']}\n"
        f"  Poll readiness with: dbrx-mmd capacity endpoint-status --endpoint {endpoint}\n"
        f"  A size rejection / image-build failure surfaces as UPDATE_FAILED."
    )


@app.command("endpoint-status")
def endpoint_status(
    endpoint: str = typer.Option("mmd-storage-probe", "--endpoint", help="Serving endpoint name."),
) -> None:
    """Print current readiness/config-update state of the probe endpoint."""
    from dbrx_multimodel_registration.adapters.serving.stress import get_endpoint_status

    status = get_endpoint_status(endpoint)
    typer.echo(f"{endpoint}: ready={status['ready']} config_update={status['config_update']}")


@app.command()
def validate(
    source: str = typer.Argument(..., help="Existing bundle dir to carve the (tiny) validation artifact from."),
    model_name: str = typer.Option(..., "--model-name", help="UC model name (catalog.schema.name) to register the probe under."),
    experiment: str = typer.Option(..., "--experiment", help="MLflow experiment workspace path (required — wheel/serverless tasks have no default)."),
    target_gb: float = typer.Option(0.01, "--target-gb", help="Validation artifact size — keep TINY; packaging errors are size-independent and a big download can exceed serverless local disk."),
    workdir: str = typer.Option("/local_disk0/mmd_probe_validate", "--workdir", help="Cluster/serverless-local scratch to assemble the artifact."),
) -> None:
    """Fast packaging gate: log the probe model, then rebuild its env and
    load+predict via `mlflow.models.predict(env_manager="virtualenv")`.

    This reproduces what the serving container does — env build, dependency
    install, model load, predict — in a local subprocess in ~1 min, WITHOUT
    deploying a billable endpoint. Run it on serverless for fast iteration.
    A green result here means `capacity stress` (the real 10 GB endpoint deploy)
    won't die on packaging errors (phantom project wheel, ModuleNotFound, etc.).
    """
    import shutil

    import mlflow.models

    from dbrx_multimodel_registration.adapters.serving.stress import (
        log_probe_model,
        select_subset_to_size,
    )

    shutil.rmtree(workdir, ignore_errors=True)
    typer.echo(f"Carving ~{target_gb:g} GB validation artifact from {source} …")
    plan = select_subset_to_size(source, workdir, target_gb)
    typer.echo(f"  assembled {_fmt_gb(plan.source_bytes)} at {workdir}")

    typer.echo(f"Logging probe model to UC: {model_name} …")
    model_uri = log_probe_model(workdir, model_name, experiment)
    typer.echo(f"  registered {model_uri}")

    typer.echo("Validating via mlflow.models.predict(env_manager='virtualenv') — rebuilds the serving env …")
    mlflow.models.predict(
        model_uri=model_uri,
        input_data={"ping": 1},
        env_manager="virtualenv",
    )
    typer.echo("VALIDATION PASSED — env builds, model loads and predicts. Safe to deploy the endpoint.")


@app.command()
def measure(
    sample_models: int = typer.Option(64, "--sample", help="Distinct models to train and pack."),
    n_estimators: int = typer.Option(500, help="RandomForest n_estimators (match ReferenceModelTrainer)."),
    max_depth: int = typer.Option(20, help="RandomForest max_depth (match ReferenceModelTrainer)."),
) -> None:
    """Measure and print on-disk bytes/model only (no fleet math)."""
    typer.echo(f"Training {sample_models} distinct models (n_estimators={n_estimators}, max_depth={max_depth})…")
    m = measure_bundle(n_models=sample_models, n_estimators=n_estimators, max_depth=max_depth)
    typer.echo(
        f"  raw pickle:  {m.raw_bytes_per_model:,.0f} B/model  ({_fmt_gb(m.raw_bytes_per_model)})\n"
        f"  on disk:     {m.bytes_per_model:,.0f} B/model  ({_fmt_gb(m.bytes_per_model)})\n"
        f"  snappy ratio: {m.compression_ratio:.2f}× (raw / on-disk)"
    )
