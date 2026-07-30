"""`ShardMappingPort` adapter — modulo mapping of buckets to shards.

Shard = the buckets where `bucket % shard_count == shard`. Self-balancing:
spreading buckets modulo keeps shard sizes even even when some buckets are
heavier, and the route-time function is one modulo over the existing
`_sku_bucket`. This is adapter #1; a contiguous-range mapping is the natural
adapter #2.

Pure arithmetic + the stdlib-only `_sku_bucket` — safe on the router hot path.
"""
from __future__ import annotations

from dbrx_multimodel_registration.adapters.logging.uc_table import _sku_bucket


class ModuloShardMapping:
    """`ShardMappingPort` — `shard = bucket % shard_count`."""

    name = "modulo"

    def shard_for_bucket(self, bucket: int, shard_count: int) -> int:
        return bucket % shard_count

    def buckets_for_shard(self, shard: int, bucket_count: int, shard_count: int) -> list[int]:
        return [b for b in range(bucket_count) if b % shard_count == shard]

    def shard_for_sku(self, sku: str, bucket_count: int, shard_count: int) -> int:
        return _sku_bucket(sku, bucket_count) % shard_count
