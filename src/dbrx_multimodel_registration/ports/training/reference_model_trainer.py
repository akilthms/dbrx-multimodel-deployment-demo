"""Port: trains the single reference model used for broadcast reuse."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class ReferenceModelTrainerPort(Protocol):
    def train(self, demand_df: "DataFrame", region: str, sku: str) -> bytes:
        """Train one model on the (region, sku) slice. Return the pickled bytes."""
        ...
