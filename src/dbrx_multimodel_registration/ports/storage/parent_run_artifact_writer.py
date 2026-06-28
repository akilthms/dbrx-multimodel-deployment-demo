"""Port: streams model object data into a parent run's artifact location.

Port answers WHERE the bundle bytes land (one MLflow parent run's artifact
storage). Adapters answer HOW the bundle is shaped on disk (parquet today;
CSV-summary / directory-of-pickle-files / etc. are future PRs).

The two-Protocol shape (`ParentRunArtifactWriterPort` + the per-region
`ParentRunArtifactBundle` returned from `.open(...)`) exists because the
strategies need to interleave artifact writes with the asyncio
producer/consumer loop — they hand the writer one SKU group at a time
rather than handing it the full record stream upfront.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ContextManager, Protocol, Sequence

from dbrx_multimodel_registration.domains.entities import RegionSpec, TrainedModelRecord

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient


class ParentRunArtifactBundle(Protocol):
    """Incremental writer for one parent-run's artifact bundle.

    Lifetime is the parent run's logging loop: opened before SKU iteration
    starts, fed records as they're produced, closed on `__exit__` of the
    enclosing `with` block (adapter flushes + finalizes there).
    """

    def write(self, records: Sequence[TrainedModelRecord]) -> None:
        """Append `records` to the bundle. Adapter buffers + flushes as it sees fit."""
        ...

    @property
    def artifact_uri(self) -> str:
        """Final URI of the bundle after close. Valid to read after `__exit__`."""
        ...


class ParentRunArtifactWriterPort(Protocol):
    """Streams model object data into a parent run's artifact location."""

    def open(
        self,
        client: "MlflowClient",
        parent_run_id: str,
        region: RegionSpec,
    ) -> ContextManager[ParentRunArtifactBundle]:
        """Return a context-managed bundle writer for this parent run."""
        ...
