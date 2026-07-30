"""Fast unit tests for endpoint capacity planning.

Two layers (cheapest first):
  - pure math on `EndpointCapacityPlan` — no deps, microseconds
  - a real measurement smoke test (`test_measure_*`, ~seconds) that trains a
    few distinct models and writes them through the parquet bundle path,
    guarding the property that DISTINCT models don't compress to nothing.
"""
from __future__ import annotations

import pytest

from dbrx_multimodel_registration.domains.entities import EndpointCapacityPlan


# ─── pure math ────────────────────────────────────────────────────────

def test_models_per_endpoint_floors():
    # 30 GB / 4.4 MB = 6818.18 → floor 6818
    plan = EndpointCapacityPlan.from_gb(
        n_models=500_000, bytes_per_model=4_400_000, per_endpoint_gb=30.0, n_regions=1,
        pack_by_region=False,
    )
    assert plan.models_per_endpoint == 6818


def test_endpoints_global_pack():
    # ceil(500000 / 6818) = 74
    plan = EndpointCapacityPlan.from_gb(
        n_models=500_000, bytes_per_model=4_400_000, per_endpoint_gb=30.0,
        pack_by_region=False,
    )
    assert plan.endpoints_needed == 74


def test_endpoints_region_sharded_rounds_up_per_region():
    # 500k / 5 = 100k per region; ceil(100000 / 6818) = 15 per region → 75.
    # Region-sharding costs one extra endpoint vs the global lower bound (74).
    plan = EndpointCapacityPlan.from_gb(
        n_models=500_000, bytes_per_model=4_400_000, per_endpoint_gb=30.0, n_regions=5,
        pack_by_region=True,
    )
    assert plan.endpoints_needed == 75


def test_uneven_regions_size_from_heaviest():
    # 7 models, 3 regions, 2 models/endpoint → regions get [3,2,2] →
    # ceil(3/2)+ceil(2/2)+ceil(2/2) = 2+1+1 = 4
    plan = EndpointCapacityPlan(
        n_models=7, bytes_per_model=1.0, per_endpoint_bytes=2, n_regions=3, pack_by_region=True,
    )
    assert plan.endpoints_needed == 4


def test_utilization_bounded():
    plan = EndpointCapacityPlan.from_gb(
        n_models=500_000, bytes_per_model=4_400_000, per_endpoint_gb=30.0, pack_by_region=False,
    )
    assert 0.0 < plan.utilization <= 1.0


def test_rejects_model_larger_than_endpoint():
    with pytest.raises(ValueError, match="exceeds the per-endpoint cap"):
        EndpointCapacityPlan(n_models=1, bytes_per_model=100, per_endpoint_bytes=50)


def test_rejects_nonpositive_bytes():
    with pytest.raises(ValueError):
        EndpointCapacityPlan(n_models=1, bytes_per_model=0, per_endpoint_bytes=50)


# ─── real measurement smoke test (~seconds) ─────────────────────────────

def test_measure_distinct_models_do_not_overcompress():
    """Guards the core correctness property: measuring DISTINCT models must
    not report a near-zero footprint the way identical demo blobs would.

    Uses a small/fast RF (few trees) — we assert the RELATIONSHIP (distinct
    models stay a real fraction of their raw size), not an absolute size.
    """
    from dbrx_multimodel_registration.adapters.serving.capacity import measure_bundle

    m = measure_bundle(n_models=8, n_estimators=25, max_depth=8)
    assert m.n_models == 8
    assert m.on_disk_bytes > 0
    # Distinct RF pickles compress ~2.7–2.9× under snappy (numpy tree arrays
    # have internal redundancy). IDENTICAL blobs — the demo's broadcast-reuse
    # bug for capacity purposes — compress 20×+ and keep climbing with count.
    # A threshold of 5× cleanly separates the two: if this trips, the sample
    # models have collapsed to identical and the measured footprint is a lie.
    assert m.compression_ratio < 5.0, (
        f"suspiciously high compression ({m.compression_ratio:.2f}×) — are the "
        f"models identical rather than distinct?"
    )
    assert m.bytes_per_model > 1000
