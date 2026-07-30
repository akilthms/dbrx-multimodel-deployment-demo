"""`ServingDeploymentStrategyPort` adapter — baked-artifact, region×shard fleet.

The production serving topology from the v3 architecture, grounded in this
project's empirical findings:

  - Model Serving endpoints have NO /Volumes FUSE at inference; artifacts are
    baked into the model at deploy time. So each shard's parquet bundle is
    baked into a per-shard model (models-from-code, see shard_model_entrypoint).
  - A baked artifact must stay under MAX_SERVING_ARTIFACT_GB (768, eng-confirmed)
    or the container build fails — this floors SHARD_COUNT per region.
  - Hierarchy: ONE experiment, region parent runs (grouping), shard child runs
    (endpoint sources). Bundles live at pre-allocated Volume paths
    <region>/shard_<N>_<s>/ so shard-count sweeps don't collide.
  - Two-tier index: this fleet builds a coarse master (region, sku-range)->
    endpoint index the router loads; each endpoint owns its fine sku->row-group
    index internally.

`deploy()` is a thin orchestrator over independently-runnable steps
(populate_shard_bundles / deploy_fleet / build_routing_index / deploy_router) so
the perf-sweep can re-shard + redeploy without re-training or re-populating
unchanged inputs. Heavy imports (Spark, MLflow, serving SDK) are lazy so the
hot path — `route()` — stays import-light.
"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from dbrx_multimodel_registration.adapters.serving.shard_mapping import ModuloShardMapping
from dbrx_multimodel_registration.domains.entities import (
    MAX_SERVING_ARTIFACT_GB,
    ServingDeployment,
    ShardPlacement,
)

if TYPE_CHECKING:
    from dbrx_multimodel_registration.ports.serving.shard_mapping import ShardMappingPort

_ROUTER_ENTRYPOINT = os.path.join(os.path.dirname(__file__), "shard_model_entrypoint.py")


def _endpoint_name(region: str, shard: int, shard_count: int) -> str:
    """Deterministic per-shard endpoint name (lowercased, hyphenated)."""
    return f"mmd-{region.lower()}-s{shard}-of{shard_count}"


def _dir_size(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _d, files in os.walk(path) for f in files
    )


class ServingFleet:
    """Baked-artifact region×shard serving fleet.

    Args:
        regions: region names to deploy.
        shard_count: shards per region (perf-sweep knob; also floored by
            MAX_SERVING_ARTIFACT_GB — see `assert_shard_sizes`).
        artifact_volume: Volume root holding <region>/bundle_parquet (source)
            and where <region>/shard_<N>_<s>/ bundles get written.
        catalog, schema: UC target for the registered per-shard models.
        experiment: single MLflow experiment for all region/shard runs.
        bucket_count: bucket partitioning the source bundles were written with.
        mapping: bucket<->shard strategy (defaults to modulo).
        workload_size: serving workload tier.
    """

    name = "baked_region_shard_fleet"

    def __init__(
        self,
        regions: list[str],
        shard_count: int,
        artifact_volume: str,
        catalog: str,
        schema: str,
        experiment: str,
        bucket_count: int,
        mapping: "ShardMappingPort | None" = None,
        workload_size: str = "Large",
        max_shard_gb: float = MAX_SERVING_ARTIFACT_GB,
    ) -> None:
        self.regions = regions
        self.shard_count = shard_count
        self.artifact_volume = artifact_volume.rstrip("/")
        self.catalog = catalog
        self.schema = schema
        self.experiment = experiment
        self.bucket_count = bucket_count
        self.mapping = mapping or ModuloShardMapping()
        self.workload_size = workload_size
        self.max_shard_gb = max_shard_gb
        self._index: list[ShardPlacement] | None = None  # loaded lazily by route()

    # ─── build-time steps (heavy; lazy imports) ──────────────────────────

    def _source_bundle(self, region: str) -> str:
        return f"{self.artifact_volume}/{region}/bundle_parquet"

    def _shard_dir(self, region: str, shard: int) -> str:
        return f"{self.artifact_volume}/{region}/shard_{self.shard_count}_{shard}"

    def populate_shard_bundles(self, enforce_size_limit: bool = False) -> list[ShardPlacement]:
        """Split each region's bucketed bundle into per-shard bundle dirs.

        Copies whole `bucket=N` dirs into `<region>/shard_<N>_<s>/` grouped by
        the mapping. Parallel-safe by construction: each shard owns a disjoint
        bucket set, so shards never write the same path. Observes per-shard size
        by default; set `enforce_size_limit` to fail-fast when a shard exceeds
        `max_shard_gb` (increase shard_count).
        """
        placements: list[ShardPlacement] = []
        limit_bytes = int(self.max_shard_gb * 1_000_000_000)
        for region in self.regions:
            src = self._source_bundle(region)
            for shard in range(self.shard_count):
                dest = self._shard_dir(region, shard)
                shutil.rmtree(dest, ignore_errors=True)
                os.makedirs(dest, exist_ok=True)
                buckets = self.mapping.buckets_for_shard(shard, self.bucket_count, self.shard_count)
                for b in buckets:
                    bsrc = os.path.join(src, f"bucket={b}")
                    if os.path.isdir(bsrc):
                        shutil.copytree(bsrc, os.path.join(dest, f"bucket={b}"))
                size = _dir_size(dest)
                if enforce_size_limit and size > limit_bytes:
                    raise ValueError(
                        f"{region} shard {shard} = {size/1e9:.1f} GB > {self.max_shard_gb} GB "
                        f"limit; increase shard_count."
                    )
                placements.append(
                    ShardPlacement(
                        region=region, shard=shard, shard_count=self.shard_count,
                        endpoint_name=_endpoint_name(region, shard, self.shard_count),
                        bundle_uri=dest,
                    )
                )
        return placements

    def populate_shard_bundles_from_table(
        self,
        spark,
        table: str,
        enforce_size_limit: bool = False,
        max_workers: int | None = None,
    ) -> list[ShardPlacement]:
        """Write per-shard parquet bundles by reading the `model_artifacts` table.

        The PRD source-of-truth path (vs the file-copy `populate_shard_bundles`):
        the bucketed `model_artifacts` Delta table is filtered per shard to its
        bucket set and written to `<region>/shard_<N>_<s>/bundle_parquet/` as a
        bucket-partitioned bundle — the exact layout the shard lookup model reads.

        Conflict-free parallel by construction: each (region, shard) writes a
        DISJOINT destination from a DISJOINT bucket filter, so shards run
        concurrently with no write contention. Parallelized across shards with a
        thread pool (Spark executes each shard's read+write as a distributed job;
        threads just overlap the driver-side orchestration).

        Requires the table to carry a `bucket` column (bucket = f(sku)) plus
        `region, sku, model_blob_bytes`. Observes per-shard size by default;
        `enforce_size_limit` fails fast past `max_shard_gb`.
        """
        from concurrent.futures import ThreadPoolExecutor
        from pyspark.sql import functions as F

        limit_bytes = int(self.max_shard_gb * 1_000_000_000)

        def _one(region: str, shard: int) -> ShardPlacement:
            buckets = self.mapping.buckets_for_shard(shard, self.bucket_count, self.shard_count)
            dest = f"{self._shard_dir(region, shard)}/bundle_parquet"
            # Overwrite is safe: this (region, shard) owns this path exclusively.
            (
                spark.read.table(table)
                .where((F.col("region") == region) & (F.col("bucket").isin(buckets)))
                .select("sku", "model_blob_bytes", "bucket")
                .repartition("bucket")
                .sortWithinPartitions("bucket", "sku")
                .write.mode("overwrite").partitionBy("bucket").parquet(dest)
            )
            # dest is a /Volumes/... path — FUSE-readable on the cluster driver.
            size = _dir_size(dest) if os.path.isdir(dest) else 0
            if enforce_size_limit and size > limit_bytes:
                raise ValueError(
                    f"{region} shard {shard} = {size/1e9:.1f} GB > {self.max_shard_gb} GB "
                    f"limit; increase shard_count."
                )
            return ShardPlacement(
                region=region, shard=shard, shard_count=self.shard_count,
                endpoint_name=_endpoint_name(region, shard, self.shard_count),
                bundle_uri=dest,
            )

        jobs = [(r, s) for r in self.regions for s in range(self.shard_count)]
        with ThreadPoolExecutor(max_workers=max_workers or len(jobs)) as ex:
            return list(ex.map(lambda rs: _one(*rs), jobs))

    def _log_shard_model(self, placement: ShardPlacement) -> str:
        """Log the per-shard lookup model (models-from-code) with its bundle
        baked in; return the model version URI."""
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, Schema
        from mlflow.tracking import MlflowClient

        from dbrx_multimodel_registration.adapters.serving.shard_model_entrypoint import ARTIFACT_KEY

        model_name = f"{self.catalog}.{self.schema}.mmd_shard_{placement.region.lower()}_{placement.shard}_of{self.shard_count}"
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(self.experiment)
        sig = ModelSignature(
            inputs=Schema([ColSpec("string", "sku"), ColSpec("double", "price"),
                           ColSpec("long", "day_of_week"), ColSpec("long", "promotion"),
                           ColSpec("double", "inventory")]),
            outputs=Schema([ColSpec("double")]),
        )
        with mlflow.start_run(run_name=f"{placement.region}-shard-{placement.shard}"):
            mlflow.pyfunc.log_model(
                name="shard_model",
                python_model=_ROUTER_ENTRYPOINT,
                artifacts={ARTIFACT_KEY: placement.bundle_uri},
                registered_model_name=model_name,
                signature=sig,
                pip_requirements=["mlflow", "pyarrow", "scikit-learn"],
            )
        versions = MlflowClient().search_model_versions(f"name='{model_name}'")
        return f"models:/{model_name}/{max(int(v.version) for v in versions)}"

    def deploy_fleet(self, placements: list[ShardPlacement]) -> None:
        """Log + deploy one endpoint per shard placement."""
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

        w = WorkspaceClient()
        for p in placements:
            uri = self._log_shard_model(p)
            name, version = uri.split("models:/")[1].rsplit("/", 1)
            entity = ServedEntityInput(
                entity_name=name, entity_version=version,
                workload_size=self.workload_size, scale_to_zero_enabled=True,
            )
            if any(e.name == p.endpoint_name for e in w.serving_endpoints.list()):
                w.serving_endpoints.update_config(name=p.endpoint_name, served_entities=[entity])
            else:
                w.serving_endpoints.create(
                    name=p.endpoint_name,
                    config=EndpointCoreConfigInput(name=p.endpoint_name, served_entities=[entity]),
                )

    def build_routing_index(self, placements: list[ShardPlacement]) -> str:
        """Scan each shard bundle's parquet footers for its SKU range, tag with
        (region, shard, endpoint), and write the master index parquet under
        `_master_index/shard_<N>/`. Returns the index URI."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for p in placements:
            lo, hi = self._bundle_sku_range(p.bundle_uri)
            rows.append({
                "region": p.region, "shard": p.shard, "shard_count": self.shard_count,
                "bucket_count": self.bucket_count,
                "endpoint_name": p.endpoint_name, "bundle_uri": p.bundle_uri,
                "sku_min": lo, "sku_max": hi,
            })
        index_dir = f"{self.artifact_volume}/_master_index/shard_{self.shard_count}"
        os.makedirs(index_dir, exist_ok=True)
        index_uri = f"{index_dir}/index.parquet"
        pq.write_table(pa.Table.from_pylist(rows), index_uri)
        return index_uri

    @staticmethod
    def _bundle_sku_range(bundle_dir: str) -> tuple[str | None, str | None]:
        """Min/max sku across a bundle's parquet footer stats (no data read)."""
        import pyarrow.parquet as pq

        lo = hi = None
        for dp, _d, files in os.walk(bundle_dir):
            for fn in files:
                if not fn.endswith(".parquet"):
                    continue
                pf = pq.ParquetFile(os.path.join(dp, fn))
                ci = pf.schema_arrow.get_field_index("sku")
                for rg in range(pf.num_row_groups):
                    st = pf.metadata.row_group(rg).column(ci).statistics
                    if st is None or st.min is None:
                        continue
                    lo = st.min if lo is None or st.min < lo else lo
                    hi = st.max if hi is None or st.max > hi else hi
        return lo, hi

    # ─── inference-time hot path (import-light) ──────────────────────────

    def load_index(self, index_uri: str | None = None) -> list[ShardPlacement]:
        """Load the master index into memory (cached). Pure pyarrow read."""
        import pyarrow.parquet as pq

        uri = index_uri or f"{self.artifact_volume}/_master_index/shard_{self.shard_count}/index.parquet"
        tbl = pq.read_table(uri).to_pylist()
        self._index = [
            ShardPlacement(
                region=r["region"], shard=r["shard"], shard_count=r["shard_count"],
                endpoint_name=r["endpoint_name"], bundle_uri=r["bundle_uri"],
                sku_min=r.get("sku_min"), sku_max=r.get("sku_max"),
            )
            for r in tbl
        ]
        return self._index

    def route(self, region: str, sku: str) -> str:
        """Resolve (region, sku) -> endpoint, mapping-agnostically.

        Compute the shard via the mapping (authoritative for ANY mapping), then
        look up the endpoint bound to (region, shard) in the loaded index. This
        is correct for modulo mapping too — where a shard holds SCATTERED buckets
        and its SKU range OVERLAPS other shards', so range-matching would route
        wrong. The index's sku_min/max are observability metadata, not the
        routing key. Falls back to the deterministic endpoint name if no index
        is loaded.

        Import-light: mapping arithmetic + an in-memory dict lookup.
        """
        shard = self.mapping.shard_for_sku(sku, self.bucket_count, self.shard_count)
        if self._index is not None:
            for p in self._index:
                if p.region == region and p.shard == shard:
                    return p.endpoint_name
        return _endpoint_name(region, shard, self.shard_count)

    # ─── teardown ────────────────────────────────────────────────────────

    def teardown(self, delete: bool = False) -> None:
        """Delete the fleet's endpoints. Per project preference the default is
        to leave endpoints in place (they scale to zero); pass delete=True to
        actually remove them."""
        if not delete:
            return
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        live = {e.name for e in w.serving_endpoints.list()}
        for region in self.regions:
            for shard in range(self.shard_count):
                nm = _endpoint_name(region, shard, self.shard_count)
                if nm in live:
                    w.serving_endpoints.delete(name=nm)

    # ─── router endpoint ─────────────────────────────────────────────────

    def deploy_router(
        self,
        index_uri: str,
        router_endpoint: str | None = None,
        host: str | None = None,
        token: str | None = None,
    ) -> str:
        """Log + deploy the single default ROUTER endpoint (models-from-code)
        with the master index baked in as an artifact. The router forwards
        (region, sku) to the owning shard endpoint. Returns the endpoint name.

        A serving container has NO default Databricks credentials, so the router
        cannot call sibling endpoints without an injected token. `host`/`token`
        are set as DATABRICKS_HOST/DATABRICKS_TOKEN env vars on the served entity
        (prefer secret refs like `{{secrets/scope/key}}` in production). The
        token identity must have CAN_QUERY on the shard endpoints.
        """
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, Schema
        from mlflow.tracking import MlflowClient
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

        from dbrx_multimodel_registration.adapters.serving.router_entrypoint import INDEX_KEY

        ep_name = router_endpoint or f"mmd-router-of{self.shard_count}"
        model_name = f"{self.catalog}.{self.schema}.mmd_router_of{self.shard_count}"
        entrypoint = os.path.join(os.path.dirname(__file__), "router_entrypoint.py")

        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(self.experiment)
        sig = ModelSignature(
            inputs=Schema([ColSpec("string", "region"), ColSpec("string", "sku"),
                           ColSpec("double", "price"), ColSpec("long", "day_of_week"),
                           ColSpec("long", "promotion"), ColSpec("double", "inventory")]),
            outputs=Schema([ColSpec("double")]),
        )
        with mlflow.start_run(run_name="router"):
            mlflow.pyfunc.log_model(
                name="router",
                python_model=entrypoint,
                artifacts={INDEX_KEY: index_uri},
                registered_model_name=model_name,
                signature=sig,
                # Router calls downstream endpoints over HTTP with requests.
                pip_requirements=["mlflow", "pyarrow", "requests"],
            )
        versions = MlflowClient().search_model_versions(f"name='{model_name}'")
        version = str(max(int(v.version) for v in versions))

        env_vars = {}
        if host:
            env_vars["DATABRICKS_HOST"] = host
        if token:
            env_vars["DATABRICKS_TOKEN"] = token

        w = WorkspaceClient()
        entity = ServedEntityInput(
            entity_name=model_name, entity_version=version,
            workload_size="Small", scale_to_zero_enabled=True,
            environment_vars=env_vars or None,
        )
        if any(e.name == ep_name for e in w.serving_endpoints.list()):
            w.serving_endpoints.update_config(name=ep_name, served_entities=[entity])
        else:
            w.serving_endpoints.create(
                name=ep_name,
                config=EndpointCoreConfigInput(name=ep_name, served_entities=[entity]),
            )
        return ep_name

    # ─── orchestrator ────────────────────────────────────────────────────

    def deploy(
        self,
        enforce_size_limit: bool = False,
        with_router: bool = True,
        host: str | None = None,
        token: str | None = None,
    ) -> ServingDeployment:
        """Full pipeline: populate bundles → deploy shard endpoints → build
        master index → (optionally) deploy the router endpoint. `host`/`token`
        are injected into the router so it can call sibling endpoints."""
        placements = self.populate_shard_bundles(enforce_size_limit=enforce_size_limit)
        self.deploy_fleet(placements)
        index_uri = self.build_routing_index(placements)
        router = self.deploy_router(index_uri, host=host, token=token) if with_router else None
        return ServingDeployment(
            experiment=self.experiment,
            shard_count=self.shard_count,
            placements=placements,
            index_uri=index_uri,
            router_endpoint=router,
        )
