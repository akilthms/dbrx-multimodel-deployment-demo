"""Port: simulates distributed training via broadcast + cross-product."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from dbrx_multimodel_registration.domains.entities import GenerationSpec


class TrainingSimulatorPort(Protocol):
    def simulate(
        self,
        reference_model_bytes: bytes,
        spec: "GenerationSpec",
        model_names: list[str],
    ) -> "DataFrame":
        """Emit the run plan: one row per (region, sku, model_name).

        Each row carries the model_blob bytes — this simulates real distributed
        training, where workers emit serialized model bytes per (region, sku).
        For benchmarking we reuse one broadcast reference; the serialization
        path through pandas_udf → Arrow → Delta stays realistic.
        """
        ...
