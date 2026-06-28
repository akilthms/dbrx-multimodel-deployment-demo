"""Port: generates a Spark DataFrame of rows shaped like a domain entity."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from dbrx_multimodel_registration.domains.entities import GenerationSpec


class DataGeneratorPort(Protocol):
    """Generates rows of `entity_type`, sized by a `GenerationSpec`.

    Adapters bind to one entity type via `entity_type` (e.g. `DemandRecord`).
    The returned DataFrame's schema must match the field names and types of
    `entity_type` field-for-field.
    """

    entity_type: type

    def generate(self, spec: "GenerationSpec") -> "DataFrame":
        """Return a Spark DataFrame containing `spec.n_rows` rows of `entity_type`."""
        ...
