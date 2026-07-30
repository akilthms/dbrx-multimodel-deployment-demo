"""Port: maps buckets ↔ shards, and SKUs → shard.

A shard is a group of a region's bundle buckets. The SAME mapping must be used
at build time (to group buckets into per-shard bundles) and at route time (to
resolve which shard — hence endpoint — owns a SKU), so both live behind one
port. Adapters choose the strategy (modulo, contiguous range, …).

Kept import-light on purpose: `shard_for_sku` is the router's hot path, so
implementations must be pure arithmetic — no Spark/MLflow.
"""
from __future__ import annotations

from typing import Protocol


class ShardMappingPort(Protocol):
    def shard_for_bucket(self, bucket: int, shard_count: int) -> int:
        """Which shard a given bucket belongs to."""
        ...

    def buckets_for_shard(self, shard: int, bucket_count: int, shard_count: int) -> list[int]:
        """The buckets that make up a shard (build-time grouping)."""
        ...

    def shard_for_sku(self, sku: str, bucket_count: int, shard_count: int) -> int:
        """Which shard serves a SKU (route-time hot path)."""
        ...
