"""Alternate logging strategy: 3-level nesting Region → SKU → Model.

Honors the strict Approach-3 hierarchy from the PRD. Each region experiment
holds region parent runs, SKU child runs, and model grand-child runs (1 per
model variant). Used for benchmarking the "what the PRD literally asks for"
shape against the collapsed default.

Asyncio orchestration (same pattern as CollapsedSkuLoggingStrategy): a bounded
`asyncio.Queue` applies backpressure on the producer; consumer tasks call the
sync MlflowClient via `loop.run_in_executor` for natural concurrency without
losing the supported client API.

By default, model blobs still bundle into ONE per-region artifact (via
`ParentRunArtifactWriterPort`, parquet adapter). Set `per_model_artifact=True`
to additionally upload each model blob as a per-run artifact — this is the
1.5M-uploads path that hits the workspace's ~35 RPS artifact ceiling and
typically runs in ~10+ hours; it exists to demonstrate the anti-pattern.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

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


class NestedModelLoggingStrategy:
    name = "nested_model"

    def __init__(
        self,
        per_model_artifact: bool = False,
        artifact_writer: ParentRunArtifactWriterPort | None = None,
    ) -> None:
        self.per_model_artifact = per_model_artifact
        self.artifact_writer = artifact_writer or ParquetBundleArtifactWriter()

    def log_all(
        self,
        regions: list[RegionSpec],
        plan_repo: RunPlanRepositoryPort,
        config: LoggingConfig,
    ) -> LoggingMetrics:
        metrics = LoggingMetrics(strategy=self.name)
        started = time.perf_counter()
        client = MlflowClient(tracking_uri=config.tracking_uri)

        log.info(f"[{self.name}] log_all start: {len(regions)} region(s) in parallel, "
              f"concurrency={config.concurrency}/region, per_model_artifact={self.per_model_artifact}")

        async def _run_all() -> None:
            results = await asyncio.gather(
                *(self._log_region_async(client, region, plan_repo, config, metrics)
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
    ) -> None:
        # Sync MLflow RPCs offloaded so sibling regions get event-loop time
        # during setup (no more 18s per-region stagger under asyncio.gather).
        loop = asyncio.get_running_loop()

        log.info(f"  [{region.name}] ensuring experiment…")
        experiment_id = await loop.run_in_executor(None, _ensure_experiment, client, region)
        log.info(f"  [{region.name}] experiment_id={experiment_id} → creating parent run")
        parent = await loop.run_in_executor(
            None,
            lambda: client.create_run(
                experiment_id=experiment_id,
                tags={"mlflow.runName": f"region_{region.name}", "layer": "region"},
            ),
        )
        log.info(f"  [{region.name}] parent_run={parent.info.run_id} → reading run plan")

        records_iter = plan_repo.stream_by_region(config.run_plan_table, region.name)
        sku_groups = itertools.groupby(records_iter, key=lambda r: r.sku)

        queue: asyncio.Queue = asyncio.Queue(maxsize=config.concurrency * 4)
        executor = ThreadPoolExecutor(
            max_workers=config.concurrency,
            thread_name_prefix=f"mlflow-{region.name}",
        )

        # progress tick — print every N completed SKU groups (each group = sku run + model children).
        # Nested strategy creates ~4 runs per SKU (1 sku + 3 model), so 1000 SKUs ≈ 4000 runs logged.
        progress_step = 1_000

        async def consume() -> None:
            local_skus = 0
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    sku_, sku_records = item
                    try:
                        runs_created = await loop.run_in_executor(
                            executor,
                            _log_one_sku_with_model_children,
                            client,
                            experiment_id,
                            parent.info.run_id,
                            region.name,
                            sku_,
                            sku_records,
                            self.per_model_artifact,
                            timings,
                        )
                        metrics.total_runs += runs_created
                        local_skus += 1
                        if local_skus % progress_step == 0:
                            log.info(f"  [{region.name}] logged {metrics.total_runs:,} runs "
                                  f"(~{local_skus:,} skus on this consumer)")
                    except Exception as e:  # noqa: BLE001
                        metrics.errors.append(f"{region.name}: {e}")
                finally:
                    queue.task_done()

        log.info(f"  [{region.name}] spawning {config.concurrency} async consumers")
        timings = RpcTimings(region.name)
        consumers = [asyncio.create_task(consume()) for _ in range(config.concurrency)]

        # Cascade-safe orchestration: poison pills always sent (even on
        # producer error), consumer gather uses return_exceptions=True, and
        # executor.shutdown only runs after gather has confirmed all consumers
        # exited. Mirrors the fix in collapsed_sku_logger.py.
        try:
            try:
                with self.artifact_writer.open(client, parent.info.run_id, region) as bundle:
                    for sku, group in sku_groups:
                        sku_records = list(group)
                        await queue.put((sku, sku_records))
                        bundle.write(sku_records)
                bundle_uri = bundle.artifact_uri
                log.info(f"  [{region.name}] producer done, draining queue…")
            finally:
                for _ in consumers:
                    await queue.put(None)

            results = await asyncio.gather(*consumers, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    metrics.errors.append(f"{region.name}: consumer crashed: {r}")
            log.info(f"  [{region.name}] all SKU + model runs done ({metrics.total_runs:,} so far)")
            log.info(f"  [{region.name}] bundle persisted → {bundle_uri}")
            timings.final_summary()
        finally:
            executor.shutdown(wait=True)

        metrics.total_artifacts += 1
        await loop.run_in_executor(None, client.set_terminated, parent.info.run_id, "FINISHED")


def _ensure_experiment(client: MlflowClient, region: RegionSpec) -> str:
    # Databricks workspace requires experiment names to be absolute workspace paths.
    # Prefix with the bundle deploy dir (matches collapsed_sku_logger).
    project_root = Path(__file__).resolve().parents[3]
    experiment_path = str(project_root / region.experiment_name)
    existing = client.get_experiment_by_name(experiment_path)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(
        name=experiment_path,
        artifact_location=region.artifact_location,
    )


def _log_one_sku_with_model_children(
    client: MlflowClient,
    experiment_id: str,
    parent_run_id: str,
    region: str,
    sku: str,
    records: list[TrainedModelRecord],
    per_model_artifact: bool,
    timings: RpcTimings,
) -> int:
    """Open the SKU run + one grand-child run per model variant.

    Returns the total number of runs created (SKU + model children).

    `timings` records the SKU-level RPCs only (create_run + the *final*
    set_terminated). log_batch isn't called on the SKU-layer run in this
    strategy — the model grand-children carry the params/metrics — so the
    log_batch slot is recorded as 0.
    """
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
    created = 1

    ts = int(time.time() * 1000)
    for rec in records:
        model_run = client.create_run(
            experiment_id=experiment_id,
            tags={
                "mlflow.runName": f"{sku}_{rec.model_name}",
                "mlflow.parentRunId": sku_run.info.run_id,
                "layer": "model",
                "region": region,
                "sku": sku,
                "model_name": rec.model_name,
            },
        )
        client.log_batch(
            run_id=model_run.info.run_id,
            params=[Param(k, str(v)) for k, v in rec.params.items()],
            metrics=[Metric(k, float(v), ts, 0) for k, v in rec.metrics.items()],
            tags=[RunTag("layer", "model")],
        )
        if per_model_artifact:
            blob_dir = Path(tempfile.mkdtemp(prefix=f"mdl_{sku}_{rec.model_name}_"))
            blob_path = blob_dir / "model.pkl"
            blob_path.write_bytes(rec.model_blob_bytes)
            client.log_artifact(model_run.info.run_id, str(blob_path), artifact_path="model")
        client.set_terminated(model_run.info.run_id, status="FINISHED")
        created += 1

    t2 = time.perf_counter()
    client.set_terminated(sku_run.info.run_id, status="FINISHED")
    t3 = time.perf_counter()
    timings.record(t1 - t0, 0.0, t3 - t2)
    return created
