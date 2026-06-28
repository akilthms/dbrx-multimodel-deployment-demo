"""Baseline strategy: 5 runs total, 5 artifact bundles. Sub-minute.

Maps to Approach 1 from the sibling PRD. Kept for benchmarking — it's what
you'd actually do at scale if hierarchy weren't a requirement.

Bundle shape is owned by the injected `ParentRunArtifactWriterPort` adapter
(default = parquet, direct-write into the parent run's artifact dir).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from mlflow.tracking import MlflowClient

from dbrx_multimodel_registration.adapters.storage.parquet_bundle import (
    ParquetBundleArtifactWriter,
)
from dbrx_multimodel_registration.domains.entities import (
    LoggingConfig,
    LoggingMetrics,
    RegionSpec,
)
from dbrx_multimodel_registration.ports.storage.parent_run_artifact_writer import (
    ParentRunArtifactWriterPort,
)
from dbrx_multimodel_registration.ports.storage.run_plan_repository import RunPlanRepositoryPort
from dbrx_multimodel_registration.utils.helpers import run_coro_blocking

log = logging.getLogger(__name__)


class RegionArtifactOnlyStrategy:
    name = "region_artifact_only"

    def __init__(self, artifact_writer: ParentRunArtifactWriterPort | None = None) -> None:
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

        log.info(f"[{self.name}] log_all start: {len(regions)} region(s) in parallel")

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

        log.info(f"  [{region.name}] ensuring experiment + parent run")
        experiment_id = await loop.run_in_executor(None, _ensure_experiment, client, region)
        run = await loop.run_in_executor(
            None,
            lambda: client.create_run(
                experiment_id=experiment_id,
                tags={"mlflow.runName": f"region_{region.name}", "layer": "region"},
            ),
        )
        log.info(f"  [{region.name}] parent_run={run.info.run_id} → streaming run plan")

        n = 0
        progress_step = 10_000  # log every 10k streamed records
        with self.artifact_writer.open(client, run.info.run_id, region) as bundle:
            buffer: list = []
            for rec in plan_repo.stream_by_region(config.run_plan_table, region.name):
                buffer.append(rec)
                n += 1
                if n % progress_step == 0:
                    log.info(f"  [{region.name}] streamed {n:,} records into bundle")
                if len(buffer) >= 200:
                    bundle.write(buffer)
                    buffer = []
            if buffer:
                bundle.write(buffer)

        log.info(f"  [{region.name}] bundle persisted ({n:,} records) → {bundle.artifact_uri}")

        await loop.run_in_executor(
            None,
            lambda: client.log_metric(run.info.run_id, "model_count", float(n)),
        )
        metrics.total_runs += 1
        metrics.total_artifacts += 1
        await loop.run_in_executor(None, client.set_terminated, run.info.run_id, "FINISHED")


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
