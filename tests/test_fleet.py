"""Local integration test for the serving-fleet build lane — no Databricks.

Builds tiny bucketed parquet bundles for 2 regions with real (small) RF models,
then exercises the pure/filesystem parts of ServingFleet end-to-end:
populate_shard_bundles → build_routing_index → load_index → route, plus the
per-shard ShardLookupModel loading + predicting from a shard bundle. This proves
the whole pipeline shape works before any cluster deploy.
"""
from __future__ import annotations

import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dbrx_multimodel_registration.adapters.logging.uc_table import _sku_bucket
from dbrx_multimodel_registration.adapters.serving.fleet import ServingFleet, _endpoint_name

BUCKET_COUNT = 8
SHARD_COUNT = 2
REGIONS = ["WEST", "NORTHEAST"]
SKUS_PER_REGION = 40


def _tiny_rf_bytes(seed: int) -> bytes:
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(32, 4))
    y = rng.normal(size=32)
    m = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=seed)
    m.fit(x, y)
    return pickle.dumps(m)


def _write_bundle(bundle_dir, skus):
    """Write a bucketed bundle (bucket=N/part.parquet) with real RF blobs."""
    from collections import defaultdict

    by_bucket = defaultdict(list)
    for i, sku in enumerate(skus):
        by_bucket[_sku_bucket(sku, BUCKET_COUNT)].append(
            {"sku": sku, "model_blob_bytes": _tiny_rf_bytes(i)}
        )
    for bucket, rows in by_bucket.items():
        rows.sort(key=lambda r: r["sku"])  # sorted → footer stats give tight ranges
        d = bundle_dir / f"bucket={bucket}"
        d.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=pa.schema([("sku", pa.string()), ("model_blob_bytes", pa.binary())])),
            str(d / "part.parquet"),
        )


@pytest.fixture
def fleet(tmp_path):
    vol = tmp_path / "vol"
    for region in REGIONS:
        skus = [f"SKU-{i:06d}" for i in range(SKUS_PER_REGION)]
        _write_bundle(vol / region / "bundle_parquet", skus)
    return ServingFleet(
        regions=REGIONS, shard_count=SHARD_COUNT, artifact_volume=str(vol),
        catalog="c", schema="s", experiment="/exp", bucket_count=BUCKET_COUNT,
    )


def test_populate_creates_disjoint_shard_bundles(fleet):
    placements = fleet.populate_shard_bundles()
    assert len(placements) == len(REGIONS) * SHARD_COUNT
    for p in placements:
        assert p.endpoint_name == _endpoint_name(p.region, p.shard, SHARD_COUNT)
        import os
        buckets = [d for d in os.listdir(p.bundle_uri) if d.startswith("bucket=")]
        # every bucket in this shard obeys bucket % shard_count == shard
        assert all(int(b.split("=")[1]) % SHARD_COUNT == p.shard for b in buckets)


def test_index_and_route_resolve_correct_endpoint(fleet):
    placements = fleet.populate_shard_bundles()
    index_uri = fleet.build_routing_index(placements)
    fleet.load_index(index_uri)
    # Every SKU must route to an endpoint whose shard actually holds its bucket.
    for i in range(SKUS_PER_REGION):
        sku = f"SKU-{i:06d}"
        ep = fleet.route("WEST", sku)
        expected_shard = _sku_bucket(sku, BUCKET_COUNT) % SHARD_COUNT
        assert ep == _endpoint_name("WEST", expected_shard, SHARD_COUNT)


def test_shard_model_loads_and_predicts_from_bundle(fleet):
    """The baked ShardLookupModel must load a shard bundle and predict a SKU
    that lives in it."""
    placements = fleet.populate_shard_bundles()
    from dbrx_multimodel_registration.adapters.serving.shard_model_entrypoint import (
        ARTIFACT_KEY,
        ShardLookupModel,
    )

    # pick a WEST shard-0 SKU
    target = next(
        f"SKU-{i:06d}" for i in range(SKUS_PER_REGION)
        if _sku_bucket(f"SKU-{i:06d}", BUCKET_COUNT) % SHARD_COUNT == 0
    )
    shard0 = next(p for p in placements if p.region == "WEST" and p.shard == 0)

    class _Ctx:
        artifacts = {ARTIFACT_KEY: shard0.bundle_uri}

    model = ShardLookupModel()
    model.load_context(_Ctx())
    import pandas as pd
    out = model.predict(None, pd.DataFrame([{
        "sku": target, "price": 10.99, "day_of_week": 3, "promotion": 0, "inventory": 500,
    }]))
    assert len(out) == 1 and isinstance(out[0], float)
