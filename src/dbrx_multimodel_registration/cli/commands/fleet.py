"""dbrx-mmd CLI — deploy the multi-model serving fleet.

Two entry points for customers:

  dbrx-mmd fleet smoke   — deploy a TINY mock fleet end-to-end (build table →
                           populate shards → deploy endpoints + router → query
                           through the router). Zero customer data; the fastest
                           way to see the architecture work.

  dbrx-mmd fleet deploy  — deploy a real fleet from the customer's existing
                           `model_artifacts` table (region · sku · model_blob_bytes
                           · bucket + telemetry). This is the production command.

Both need Spark + Unity Catalog, so run them on a Databricks cluster/serverless
(e.g. via `databricks bundle run` or a notebook). Heavy imports are lazy so the
CLI loads fast; `_get_spark()` picks up the active session on-cluster.
"""
from __future__ import annotations

import typer

app = typer.Typer(help="Deploy the multi-model serving fleet (smoke or real data).")


def _get_spark():
    """Active Spark session (on a Databricks cluster/serverless)."""
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(
            "No active Spark session. Run this on a Databricks cluster or "
            "serverless (via `databricks bundle run` or a notebook)."
        )
    return spark


def _creds():
    """Host + token for the router to call sibling endpoints. Reads env first
    (works in jobs), falls back to the notebook context on interactive clusters."""
    import os

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if host and token:
        return host.rstrip("/"), token
    try:  # interactive notebook fallback
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
        import IPython  # noqa: F401 - only present in notebooks
        dbutils = IPython.get_ipython().user_ns["dbutils"]
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        return host, token
    except Exception:
        return host, token  # may be (None, None) — deploy_router still logs the model


@app.command()
def smoke(
    catalog: str = typer.Option(..., "--catalog", help="UC catalog you can write to."),
    schema: str = typer.Option(..., "--schema", help="UC schema you can write to."),
    experiment: str = typer.Option(..., "--experiment", help="MLflow experiment workspace path, e.g. /Users/you/mmd_fleet."),
    regions: int = typer.Option(2, "--regions", help="Mock regions."),
    shard_count: int = typer.Option(2, "--shard-count", help="Shards per region → regions×shard_count endpoints."),
    skus_per_region: int = typer.Option(40, "--skus", help="Mock SKUs per region."),
    bucket_count: int = typer.Option(8, "--buckets", help="Bucket partitioning."),
) -> None:
    """Deploy a tiny MOCK fleet end-to-end so you can see the architecture work.

    Builds a small `<catalog>.<schema>.mmd_fleet_smoke` table with trivial
    RandomForest models, then runs the full pipeline and queries through the
    router. Creates regions×shard_count small serving endpoints (billable) —
    delete them after (see printed teardown).
    """
    import pickle

    import numpy as np
    from pyspark.sql import functions as F
    from sklearn.ensemble import RandomForestRegressor

    from dbrx_multimodel_registration.adapters.logging.uc_table import _sku_bucket
    from dbrx_multimodel_registration.adapters.serving import ServingFleet
    from dbrx_multimodel_registration.adapters.storage import ModelArtifactsTable

    spark = _get_spark()
    table = f"{catalog}.{schema}.mmd_fleet_smoke"
    vol = f"/Volumes/{catalog}/{schema}/mlflow_artifacts/_fleet_smoke"

    typer.echo(f"[1/4] building mock model_artifacts table {table} …")
    m = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0)
    m.fit(np.random.default_rng(0).normal(size=(16, 4)), np.random.default_rng(0).normal(size=16))
    blob = pickle.dumps(m)
    canonical = ["NORTHEAST", "SOUTHEAST", "MIDWEST", "SOUTHWEST", "WEST"]
    rows = [
        (canonical[r], f"SKU-{i:06d}", "RandomForest", blob,
         {"n_estimators": "3"}, {"rmse": 0.1, "mape": 0.2, "r2": 0.9})
        for r in range(regions) for i in range(skus_per_region)
    ]
    trained = spark.createDataFrame(
        rows, "region string, sku string, model_name string, model_blob_bytes binary, "
              "params map<string,string>, metrics map<string,double>")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    ModelArtifactsTable(spark, bucket_count=bucket_count).write(trained, table)

    host, token = _creds()
    fleet = ServingFleet(
        regions=canonical[:regions], shard_count=shard_count, artifact_volume=vol,
        catalog=catalog, schema=schema, experiment=experiment,
        bucket_count=bucket_count, workload_size="Small",
    )
    typer.echo(f"[2/4] populate shard bundles from table …")
    placements = fleet.populate_shard_bundles_from_table(spark, table)
    typer.echo(f"[3/4] deploy {len(placements)} shard endpoints + router …")
    fleet.deploy_fleet(placements)
    index_uri = fleet.build_routing_index(placements)
    router = fleet.deploy_router(index_uri, host=host, token=token)

    typer.echo(f"[4/4] fleet submitted. router endpoint: {router}")
    typer.echo(f"  shard endpoints: {[p.endpoint_name for p in placements]}")
    typer.echo(f"  → poll readiness: dbrx-mmd capacity endpoint-status --endpoint {router}")
    typer.echo(f"  → once READY, query the router with (region, sku) records")
    typer.echo(f"  → teardown: databricks serving-endpoints delete {router}  (+ each shard endpoint)")


@app.command()
def deploy(
    table: str = typer.Option(..., "--table", help="Your model_artifacts table (catalog.schema.name) with region·sku·model_blob_bytes·bucket."),
    catalog: str = typer.Option(..., "--catalog", help="UC catalog for shard models."),
    schema: str = typer.Option(..., "--schema", help="UC schema for shard models."),
    experiment: str = typer.Option(..., "--experiment", help="MLflow experiment workspace path."),
    artifact_volume: str = typer.Option(..., "--artifact-volume", help="Volume root for shard bundles, e.g. /Volumes/<cat>/<schema>/mlflow_artifacts."),
    shard_count: int = typer.Option(..., "--shard-count", help="Shards per region → regions×shard_count endpoints. Keep each shard bundle small (<=30GB proven; larger sizes fail — see reports)."),
    bucket_count: int = typer.Option(64, "--buckets", help="Bucket count the table was written with (must match)."),
    workload_size: str = typer.Option("Large", "--workload-size", help="Serving workload size."),
    enforce_size_limit: bool = typer.Option(True, "--enforce-size-limit/--no-enforce-size-limit", help="Fail fast if any shard bundle exceeds the safe cap."),
    max_shard_gb: float = typer.Option(30.0, "--max-shard-gb", help="Per-shard cap. 30GB is the proven-safe ceiling."),
) -> None:
    """Deploy a real fleet from YOUR model_artifacts table.

    Regions are read from the table. Splits each region's buckets into
    shard_count endpoints, bakes each shard's bundle, deploys endpoints + a
    router, and writes the routing index. Run on a Databricks cluster.
    """
    from pyspark.sql import functions as F

    from dbrx_multimodel_registration.adapters.serving import ServingFleet

    spark = _get_spark()
    regions = [r.region for r in spark.read.table(table).select("region").distinct().collect()]
    if not regions:
        raise typer.Exit(code=1)
    typer.echo(f"Deploying fleet: regions={regions} × shard_count={shard_count} "
               f"= {len(regions) * shard_count} endpoints (+1 router)")

    host, token = _creds()
    fleet = ServingFleet(
        regions=regions, shard_count=shard_count, artifact_volume=artifact_volume,
        catalog=catalog, schema=schema, experiment=experiment,
        bucket_count=bucket_count, workload_size=workload_size, max_shard_gb=max_shard_gb,
    )
    typer.echo("[1/4] populate shard bundles from table (parallel) …")
    placements = fleet.populate_shard_bundles_from_table(
        spark, table, enforce_size_limit=enforce_size_limit)
    typer.echo(f"[2/4] deploy {len(placements)} shard endpoints …")
    fleet.deploy_fleet(placements)
    typer.echo("[3/4] build routing index …")
    index_uri = fleet.build_routing_index(placements)
    typer.echo("[4/4] deploy router endpoint …")
    router = fleet.deploy_router(index_uri, host=host, token=token)

    typer.echo(f"Fleet submitted. router: {router}; index: {index_uri}")
    typer.echo(f"  shard endpoints: {len(placements)} → poll each with "
               f"`dbrx-mmd capacity endpoint-status --endpoint <name>`")
    typer.echo(f"  query the router with (region, sku, <feature cols>) records")
