"""Port: persists the trainer output as a resumable run plan."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Protocol

from dbrx_multimodel_registration.domains.entities import TrainedModelRecord

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class RunPlanRepositoryPort(Protocol):
    def persist(self, training_output: "DataFrame", table: str) -> None:
        """Write the trainer output to the run-plan store."""
        ...

    def stream_by_region(self, table: str, region: str) -> Iterator[TrainedModelRecord]:
        """Yield records for a region without materializing them all in memory."""
        ...

    def distinct_regions(self, table: str) -> list[str]:
        ...
