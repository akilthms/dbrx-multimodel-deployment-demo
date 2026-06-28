"""Port: the swappable MLflow logging strategy.

Each adapter encodes a different (run-hierarchy, artifact-shape, concurrency)
trade-off. The PRD's "Approach 3" maps to CollapsedSkuLoggingStrategy by
default; alternates exist for benchmarking and "what not to do" comparison.
"""
from __future__ import annotations

from typing import Protocol

from dbrx_multimodel_registration.domains.entities import (
    LoggingConfig,
    LoggingMetrics,
    RegionSpec,
)
from dbrx_multimodel_registration.ports.storage.run_plan_repository import RunPlanRepositoryPort


class ModelLoggingStrategyPort(Protocol):
    name: str

    def log_all(
        self,
        regions: list[RegionSpec],
        plan_repo: RunPlanRepositoryPort,
        config: LoggingConfig,
    ) -> LoggingMetrics:
        """Drive the full logging workflow across regions and return metrics."""
        ...
