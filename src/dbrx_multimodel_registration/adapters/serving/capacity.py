"""Measure the on-disk footprint of a model bundle so endpoint-count math
uses a real number, not the raw blob constant.

The open question this answers: given a Model Serving endpoint that can hold
`per_endpoint_gb` of bundle on local disk, how many of our models fit — and
therefore how many endpoints does 500k models need?

The naive answer divides by the raw pickle size (`_MODEL_BLOB_SIZE_BYTES`,
4.4 MB). That is wrong for two independent reasons, both of which this module
controls for:

  1. The serving bundle is snappy-compressed parquet with column + row-group
     overhead. Effective bytes/model ≠ raw pickle size.
  2. The demo broadcasts ONE reference model and reuses its bytes for every
     row (see `TrainingSimulator`). A bundle of identical blobs compresses
     almost to nothing — measuring it would report a fraction of the true
     footprint. Production has DISTINCT models per SKU. So we train distinct
     RandomForests on jittered data and measure THOSE.

Writes through the SAME parquet schema + codec the production bundle uses
(`_BUNDLE_SCHEMA`, snappy, `_ROW_GROUP`), so the measured bytes/model matches
what actually lands on the endpoint. Pure pyarrow + sklearn — no Spark, no
Databricks — so it runs locally in seconds (fast signal before any deploy).
"""
from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dbrx_multimodel_registration.adapters.storage.parquet_bundle import (
    _BUNDLE_SCHEMA,
    _ROW_GROUP,
)
from dbrx_multimodel_registration.domains.entities import EndpointCapacityPlan


@dataclass(frozen=True)
class BundleMeasurement:
    """Result of writing `n_models` distinct models to a real parquet bundle."""

    n_models: int
    on_disk_bytes: int
    raw_pickle_bytes: int

    @property
    def bytes_per_model(self) -> float:
        """Effective on-disk bytes per model — the number capacity math needs."""
        return self.on_disk_bytes / self.n_models

    @property
    def raw_bytes_per_model(self) -> float:
        return self.raw_pickle_bytes / self.n_models

    @property
    def compression_ratio(self) -> float:
        """raw pickle bytes / on-disk bytes. >1 means the bundle is smaller."""
        return self.raw_pickle_bytes / self.on_disk_bytes if self.on_disk_bytes else 0.0


def _train_distinct_models(n: int, n_estimators: int, max_depth: int) -> list[bytes]:
    """Train `n` DISTINCT RandomForests and return their pickled bytes.

    Each model fits on independently-jittered synthetic data so the trees
    differ — this is what makes the bundle incompressible in the same way a
    real per-SKU fleet is. Sizing tracks `ReferenceModelTrainer` (same
    n_estimators / max_depth defaults) so the measured blob matches the demo's.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    blobs: list[bytes] = []
    for i in range(n):
        # Same 4-feature shape as ReferenceModelTrainer (price, day_of_week,
        # promotion, inventory). Per-model seed → distinct fitted trees.
        x = rng.normal(size=(256, 4)) + i
        y = rng.normal(size=256) + i
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=i,
        )
        model.fit(x, y)
        blobs.append(pickle.dumps(model))
    return blobs


def measure_bundle(
    n_models: int = 64,
    n_estimators: int = 500,
    max_depth: int = 20,
    dest_dir: str | None = None,
) -> BundleMeasurement:
    """Train `n_models` distinct models, write them through the production
    parquet bundle path, and measure the resulting on-disk size.

    `n_models` is a SAMPLE — a few dozen distinct models is enough to get a
    stable bytes/model, since parquet overhead amortizes and each RF is the
    same size class. Defaults train fast (~seconds) while matching the demo's
    model dimensions.

    Pass `dest_dir` to inspect the written parquet; omit to use a tempdir.
    """
    blobs = _train_distinct_models(n_models, n_estimators, max_depth)
    raw_pickle_bytes = sum(len(b) for b in blobs)

    rows = [
        {
            "sku": f"SKU-{i:06d}",
            "model_name": "rf_reference",
            "model_blob": blob,
            "rmse": 0.1,
            "mape": 0.1,
            "r2": 0.9,
        }
        for i, blob in enumerate(blobs)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(dest_dir) if dest_dir else Path(tmp) / "models.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Same schema + codec + row-group size the production bundle writer uses,
        # so the measured footprint is what actually lands on the endpoint.
        writer = pq.ParquetWriter(str(out), _BUNDLE_SCHEMA, compression="snappy")
        for start in range(0, len(rows), _ROW_GROUP):
            chunk = rows[start : start + _ROW_GROUP]
            writer.write_table(pa.Table.from_pylist(chunk, schema=_BUNDLE_SCHEMA))
        writer.close()
        on_disk_bytes = out.stat().st_size

    return BundleMeasurement(
        n_models=n_models,
        on_disk_bytes=on_disk_bytes,
        raw_pickle_bytes=raw_pickle_bytes,
    )


def plan_endpoints(
    n_models: int,
    per_endpoint_gb: float,
    n_regions: int = 1,
    pack_by_region: bool = True,
    measurement: BundleMeasurement | None = None,
    sample_models: int = 64,
) -> tuple[EndpointCapacityPlan, BundleMeasurement]:
    """Measure bytes/model (unless supplied), then build the capacity plan.

    Returns (plan, measurement) so callers can report both the derived
    endpoint count and the measured footprint it rests on.
    """
    if measurement is None:
        measurement = measure_bundle(n_models=sample_models)
    plan = EndpointCapacityPlan.from_gb(
        n_models=n_models,
        bytes_per_model=measurement.bytes_per_model,
        per_endpoint_gb=per_endpoint_gb,
        n_regions=n_regions,
        pack_by_region=pack_by_region,
    )
    return plan, measurement
