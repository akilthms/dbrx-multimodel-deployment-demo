"""Default logging strategy: ONE run per SKU, ONE artifact bundle per region.

For each region, this opens a parent run, then submits one nested SKU run per
SKU through an asyncio queue + consumer-task pool. Each SKU run carries the
three model variants' params and metrics as a single batched `log_batch` call
with `model_name.*` prefixed keys. The model blobs are NOT logged per-run —
they accumulate into a per-region bundle via `ParentRunArtifactWriterPort`
(default adapter writes ONE parquet file directly into the parent run's
artifact dir).

Why asyncio + run_in_executor: MlflowClient is sync (and `enable_async_logging`
already handles log_batch latency on MLflow's side). The asyncio layer gives us
clean control flow and — via `asyncio.Queue(maxsize=…)` — natural backpressure
on the producer, which is what actually keeps driver memory bounded when there
are 100k+ SKUs per region.

This collapses API call counts by ~3x relative to a model-per-run shape and
collapses artifact uploads by ~300,000x, making the workspace ceilings
(120 RPS tracking, 35 RPS artifact) the binding constraint rather than an
explosive bottleneck.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from dbrx_multimodel_registration.adapters.storage.parquet_bundle import (
    ParquetBundleArtifactWriter,
)
from dbrx_multimodel_registration.domains.entities import (
    LoggingConfig,
    LoggingMetrics,
    RegionSpec,
    TrainedModelRecord,
)
from dbrx_multimodel_registration.ports.storage.parent_run_artifact_writer import (
    ParentRunArtifactWriterPort,
)
from dbrx_multimodel_registration.ports.storage.run_plan_repository import RunPlanRepositoryPort
from dbrx_multimodel_registration.utils.helpers import RpcTimings, run_coro_blocking

log = logging.getLogger(__name__)


class CollapsedSkuLoggingStrategy:
    name = "collapsed_sku"

    def __init__(
        self,
        artifact_writer: ParentRunArtifactWriterPort | None = None,
        n_shards: int = 1,
    ) -> None:
        # Bundle format/destination is the adapter's choice. Default = parquet,
        # direct-write into the parent run's artifact dir.
        self.artifact_writer = artifact_writer or ParquetBundleArtifactWriter()
        # n_shards > 1 splits each region's SKU runs across N sub-experiments
        # to probe whether per-experiment write serialization at the workspace
        # is the throughput bottleneck.
        self.n_shards = max(1, n_shards)

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

        log.info(f"[{self.name}] log_all start: {len(regions)} region(s) in parallel, "
              f"concurrency={config.concurrency}/region")

        async def _run_all() -> None:
            # return_exceptions=True so one region's failure can't cancel
            # the others (cancellation cascade was killing throughput).
            results = await asyncio.gather(
                *(self._log_region_async(client, region, plan_repo, config, metrics, plan_df)
                  for region in regions),
                return_exceptions=True,
            )
            for region, result in zip(regions, results):
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    metrics.errors.append(f"{region.name}: region task crashed: {result}")
                    log.error("region %s failed: %s", region.name, result)

        run_coro_blocking(_run_all())

        log.info(f"[{self.name}] log_all complete: total_runs={metrics.total_runs:,}, "
              f"total_artifacts={metrics.total_artifacts}, errors={len(metrics.errors)}")

        metrics.elapsed_seconds = time.perf_counter() - started
        return metrics

    async def _log_region_async(
        self,
        client: MlflowClient,
        region: RegionSpec,
        plan_repo: RunPlanRepositoryPort,
        config: LoggingConfig,
        metrics: LoggingMetrics,
        plan_df: DataFrame | None = None,
    ) -> None:
        # Sync MLflow RPCs offloaded to the default thread pool so the
        # asyncio event loop yields to sibling regions during setup — this
        # is what eliminates the ~18s per-region startup stagger we saw
        # before when 5 regions ran via asyncio.gather.
        loop = asyncio.get_running_loop()

        log.info(f"  [{region.name}] ensuring {self.n_shards} experiment shard(s)…")
        shard_experiment_ids = await loop.run_in_executor(
            None, _ensure_shard_experiments, client, region, self.n_shards
        )
        log.info(f"  [{region.name}] shard_experiment_ids={shard_experiment_ids} → creating region parent run")
        # Region parent run lives in shard 0; bundle artifact attaches here.
        parent = await loop.run_in_executor(
            None,
            lambda: client.create_run(
                experiment_id=shard_experiment_ids[0],
                tags={"mlflow.runName": f"region_{region.name}", "layer": "region"},
            ),
        )
        log.info(f"  [{region.name}] parent_run={parent.info.run_id} → reading run plan")

        if plan_df is not None:
            records: Iterable[TrainedModelRecord] = (
                TrainedModelRecord(
                    region=row["region"],
                    sku=row["sku"],
                    model_name=row["model_name"],
                    model_blob_bytes=bytes(row["model_blob"]),
                    params=json.loads(row["params_json"]),
                    metrics={k: float(v) for k, v in json.loads(row["metrics_json"]).items()},
                )
                for row in plan_df
                .where(col("region") == region.name)
                .sortWithinPartitions("sku", "model_name")
                .toLocalIterator()
            )
        else:
            records = plan_repo.stream_by_region(config.run_plan_table, region.name)

        sku_groups = _group_by_sku(records)

        # ─── asyncio orchestration ────────────────────────────────────────
        # Bounded queue → backpressure: the producer below blocks on put()
        # once `concurrency * 4` items are pending, so we never accumulate
        # an unbounded list of work.
        queue: asyncio.Queue = asyncio.Queue(maxsize=config.concurrency * 4)
        executor = ThreadPoolExecutor(
            max_workers=config.concurrency,
            thread_name_prefix=f"mlflow-{region.name}",
        )

        # progress tick — print every N completed SKU runs. Race-safe enough
        # for a progress indicator (duplicate prints around the threshold are OK).
        progress_step = 1_000

        sku_run_ids: list[str] = []

        async def consume() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:  # poison pill
                        return
                    sku, sku_records = item
                    try:
                        # Pick a shard experiment by deterministic hash so the same
                        # SKU always lands in the same shard (and so the distribution
                        # is even). When n_shards=1 this collapses to single experiment.
                        shard_idx = abs(hash(sku)) % self.n_shards
                        shard_exp_id = shard_experiment_ids[shard_idx]
                        run_id = await loop.run_in_executor(
                            executor,
                            _log_one_sku_run,
                            client,
                            shard_exp_id,
                            parent.info.run_id,
                            region.name,
                            sku,
                            sku_records,
                            timings,
                        )
                        sku_run_ids.append(run_id)
                        metrics.total_runs += 1
                        if metrics.total_runs % progress_step == 0:
                            log.info(f"  [{region.name}] logged {metrics.total_runs:,} SKU runs")
                    except Exception as e:  # noqa: BLE001
                        metrics.errors.append(f"{region.name}: {e}")
                finally:
                    queue.task_done()

        log.info(f"  [{region.name}] spawning {config.concurrency} async consumers")
        timings = RpcTimings(region.name)
        consumers = [asyncio.create_task(consume()) for _ in range(config.concurrency)]

        # Cascade-safe orchestration: regardless of producer outcome, ensure
        # consumers receive poison pills and drain BEFORE we shutdown the
        # executor. With return_exceptions=True on gather, one consumer's
        # uncaught failure can't cancel its siblings and trigger an early
        # executor.shutdown that breaks the survivors.
        try:
            try:
                with self.artifact_writer.open(client, parent.info.run_id, region) as bundle:
                    for sku, sku_records in sku_groups:
                        await queue.put((sku, sku_records))
                        bundle.write(sku_records)
                bundle_uri = bundle.artifact_uri
                log.info(f"  [{region.name}] producer done, draining queue…")
            finally:
                # Always feed poison pills, even if the producer raised, so
                # consumers can exit cleanly.
                for _ in consumers:
                    await queue.put(None)

            results = await asyncio.gather(*consumers, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    metrics.errors.append(f"{region.name}: consumer crashed: {r}")
            log.info(f"  [{region.name}] create+log_batch phase done ({metrics.total_runs:,} so far)")
            log.info(f"  [{region.name}] bundle persisted → {bundle_uri}")

            # Batch-terminate: fire all set_terminated calls in parallel via the
            # same executor. Previously this was 60% of per-SKU latency (450ms);
            # batching means it's still rate-limited by the workspace API but
            # the producer→consumer pipeline ran without it blocking.
            log.info(f"  [{region.name}] batch-terminating {len(sku_run_ids):,} SKU runs…")
            terminate_started = time.perf_counter()
            term_futures = [
                loop.run_in_executor(executor, client.set_terminated, run_id, "FINISHED")
                for run_id in sku_run_ids
            ]
            term_results = await asyncio.gather(*term_futures, return_exceptions=True)
            term_errors = sum(1 for r in term_results if isinstance(r, BaseException))
            log.info(
                f"  [{region.name}] batch-terminate done in "
                f"{time.perf_counter() - terminate_started:.1f}s "
                f"({len(sku_run_ids) - term_errors}/{len(sku_run_ids)} succeeded)"
            )
            if term_errors:
                metrics.errors.append(f"{region.name}: {term_errors} terminate failures")

            timings.final_summary()
        finally:
            # Safe now — gather has returned, meaning all consumers exited.
            executor.shutdown(wait=True)

        metrics.total_artifacts += 1
        await loop.run_in_executor(None, client.set_terminated, parent.info.run_id, "FINISHED")


def _ensure_experiment(client: MlflowClient, region: RegionSpec) -> str:
    # Databricks workspace requires experiment names to be absolute workspace
    # paths (e.g. "/Users/<user>/exp" or "/Workspace/.../bundle/.../exp").
    # Prefix region.experiment_name with the bundle's deploy directory so the
    # experiment lives next to the code and resolves under the user's workspace.
    project_root = Path(__file__).resolve().parents[3]
    experiment_path = str(project_root / region.experiment_name)
    existing = client.get_experiment_by_name(experiment_path)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(
        name=experiment_path,
        artifact_location=region.artifact_location,
    )


def _ensure_shard_experiments(
    client: MlflowClient, region: RegionSpec, n_shards: int
) -> list[str]:
    """Return N experiment_ids for this region — one per shard.

    n_shards=1 → singleton list, exactly today's behavior (single experiment).
    n_shards>1 → N experiments named `<base>/shard_{i:02d}`, each with its own
    artifact_location subdir. Used to probe whether per-experiment write
    serialization at the workspace caps create_run throughput.
    """
    if n_shards <= 1:
        return [_ensure_experiment(client, region)]

    project_root = Path(__file__).resolve().parents[3]
    base_path = str(project_root / region.experiment_name)
    base_artifact = region.artifact_location or ""

    ids: list[str] = []
    for i in range(n_shards):
        shard_path = f"{base_path}/shard_{i:02d}"
        existing = client.get_experiment_by_name(shard_path)
        if existing is not None:
            ids.append(existing.experiment_id)
            continue
        shard_artifact = (
            f"{base_artifact.rstrip('/')}/shard_{i:02d}" if base_artifact else None
        )
        ids.append(
            client.create_experiment(name=shard_path, artifact_location=shard_artifact)
        )
    return ids


def _group_by_sku(
    records: Iterable[TrainedModelRecord],
) -> Iterable[tuple[str, list[TrainedModelRecord]]]:
    """Stream records grouped into per-SKU lists.

    Records arrive already filtered to one region; this groups *within* that
    region's records by SKU. Caller must guarantee records arrive with
    consecutive same-SKU rows — `DeltaRunPlanRepository.stream_by_region`
    enforces this via `.sortWithinPartitions("sku", "model_name")`.
    """
    for sku, group in itertools.groupby(records, key=lambda r: r.sku):
        yield sku, list(group)


def _log_one_sku_run(
    client: MlflowClient,
    experiment_id: str,
    parent_run_id: str,
    region: str,
    sku: str,
    records: list[TrainedModelRecord],
    timings: RpcTimings,
) -> str:
    """Open one nested SKU run + batch-log params/metrics. Returns the run_id.

    Per-SKU set_terminated is deferred: the strategy collects all returned
    run_ids and terminates them in parallel at end of region. set_terminated
    was 60% of per-SKU latency in profiling (450ms vs 750ms total).
    """
    params: list[Param] = []
    mlflow_metrics: list[Metric] = []
    ts = int(time.time() * 1000)
    for rec in records:
        prefix = rec.model_name
        for k, v in rec.params.items():
            params.append(Param(f"{prefix}.{k}", str(v)))
        for k, v in rec.metrics.items():
            mlflow_metrics.append(Metric(f"{prefix}.{k}", float(v), ts, 0))

    t0 = time.perf_counter()
    sku_run = client.create_run(
        experiment_id=experiment_id,
        tags={
            "mlflow.runName": f"sku_{sku}",
            "mlflow.parentRunId": parent_run_id,
            "layer": "sku",
            "region": region,
            "sku": sku,
        },
    )
    t1 = time.perf_counter()
    client.log_batch(
        run_id=sku_run.info.run_id,
        metrics=mlflow_metrics,
        params=params,
        tags=[RunTag("model_count", str(len(records)))],
    )
    t2 = time.perf_counter()
    # set_terminated deferred to end-of-region batch (was 60% of per-SKU cost).
    timings.record(t1 - t0, t2 - t1, 0.0)
    return sku_run.info.run_id
