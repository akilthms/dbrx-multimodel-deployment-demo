"""Fast unit tests for the serving-fleet foundation: modulo shard mapping +
the ShardPlacement / ServingDeployment value objects. Pure, no Databricks."""
from __future__ import annotations

from dbrx_multimodel_registration.adapters.serving.shard_mapping import ModuloShardMapping
from dbrx_multimodel_registration.domains.entities import (
    MAX_SERVING_ARTIFACT_GB,
    ServingDeployment,
    ShardPlacement,
)


def test_buckets_partition_cleanly_across_shards():
    """Every bucket lands in exactly one shard; union == all buckets."""
    m = ModuloShardMapping()
    bucket_count, shard_count = 64, 4
    groups = [m.buckets_for_shard(s, bucket_count, shard_count) for s in range(shard_count)]
    flat = [b for g in groups for b in g]
    assert sorted(flat) == list(range(bucket_count))       # complete
    assert len(flat) == len(set(flat))                     # disjoint


def test_build_and_route_agree():
    """The shard a SKU routes to must equal the shard whose bucket-set holds
    that SKU's bucket — build-time and route-time consistency."""
    m = ModuloShardMapping()
    bucket_count, shard_count = 64, 4
    for i in (0, 1, 123, 48907, 999999):
        sku = f"SKU-{i:06d}"
        shard = m.shard_for_sku(sku, bucket_count, shard_count)
        from dbrx_multimodel_registration.adapters.logging.uc_table import _sku_bucket
        assert _sku_bucket(sku, bucket_count) in m.buckets_for_shard(shard, bucket_count, shard_count)


def test_shard_for_bucket_is_modulo():
    m = ModuloShardMapping()
    assert m.shard_for_bucket(10, 4) == 2
    assert m.shard_for_bucket(3, 4) == 3


def test_single_shard_holds_everything():
    m = ModuloShardMapping()
    assert m.buckets_for_shard(0, 64, 1) == list(range(64))
    assert all(m.shard_for_sku(f"SKU-{i:06d}", 64, 1) == 0 for i in range(50))


def test_serving_deployment_endpoint_names():
    placements = [
        ShardPlacement(region="WEST", shard=0, shard_count=2, endpoint_name="e-west-0", bundle_uri="/v/west/shard_2_0"),
        ShardPlacement(region="WEST", shard=1, shard_count=2, endpoint_name="e-west-1", bundle_uri="/v/west/shard_2_1"),
    ]
    dep = ServingDeployment(experiment="Demand_Forecasting", shard_count=2, placements=placements)
    assert dep.endpoint_names == ["e-west-0", "e-west-1"]


def test_max_serving_artifact_ceiling_recorded():
    # Eng-confirmed hard ceiling; governs minimum SHARD_COUNT.
    assert MAX_SERVING_ARTIFACT_GB == 768.0
