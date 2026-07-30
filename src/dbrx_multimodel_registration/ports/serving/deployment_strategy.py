"""Port: deploy a multi-model serving topology.

The contract names the RESPONSIBILITY — stand up serving endpoint(s) for a set
of models and route requests to them — not any particular topology. The first
adapter (`ServingFleet`) bakes per-shard parquet bundles into region×shard
endpoints; a future adapter could serve differently (e.g. per-request Files-API
lookup) behind the same three methods.

`deploy` and `teardown` are heavy (Spark / MLflow / serving APIs); `route` is
the inference-time hot path and must stay import-light. Adapters keep the heavy
imports lazy so an endpoint can call `route` without the build-time deps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dbrx_multimodel_registration.domains.entities import ServingDeployment


class ServingDeploymentStrategyPort(Protocol):
    def deploy(self) -> "ServingDeployment":
        """Stand up the serving topology and return what was created.

        Implementations choose how models map to endpoints (sharding, bundling,
        routing). Returns a `ServingDeployment` describing endpoints + index so
        callers can route and tear down without re-deriving placement.
        """
        ...

    def route(self, region: str, sku: str) -> str:
        """Resolve `(region, sku)` to the endpoint name that serves it.

        The composability seam: callers route the same way regardless of
        topology (a single-endpoint adapter returns its one name). Hot path —
        must not trigger heavy imports.
        """
        ...

    def teardown(self) -> None:
        """Delete/stop the endpoint(s) this strategy created."""
        ...
