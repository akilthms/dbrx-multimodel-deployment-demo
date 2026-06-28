"""Distribute synthetic data generation across a Spark cluster.

The adapter is entity-agnostic: pass in any self-generating dataclass (one
with `region` + `sku` required args and the rest backed by `default_factory`)
and a `GenerationSpec`, and you get a Spark DataFrame of `n_rows` instances.

The schema is derived from the dataclass via `struct_type_from_dataclass` —
no hand-maintained `_SCHEMA` block here.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterator

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from dbrx_multimodel_registration.domains.entities import GenerationSpec
from dbrx_multimodel_registration.utils.helpers import struct_type_from_dataclass


_PAIR_SCHEMA = StructType(
    [
        StructField("region", StringType(), False),
        StructField("sku", StringType(), False),
    ]
)
_YIELD_CHUNK = 10_000  # cap pandas DataFrame size per yield to bound worker memory


class SparkDataGenerator:
    """`DataGeneratorPort` — distributes entity instantiation across Spark.

    Each Spark partition receives a slice of the (region, sku) cross-product
    and expands every pair into `spec.rows_per_sku` records by instantiating
    `entity_type(region=..., sku=...)`. The entity's `default_factory` calls
    populate the remaining fields on the worker side.
    """

    name = "spark"

    def __init__(self, spark: SparkSession, entity_type: type, partitions: int | None = None) -> None:
        self.spark = spark
        self.entity_type = entity_type
        self.schema = struct_type_from_dataclass(entity_type)
        # Default to 4× cluster parallelism so the mapInPandas fan-out actually
        # saturates the workers. The previous fixed 16 left ~80% of cores idle
        # at scale (cluster has ~80 cores; 16 partitions used 20%).
        self.partitions = partitions if partitions is not None else spark.sparkContext.defaultParallelism * 4

    def generate(self, spec: GenerationSpec) -> DataFrame:
        pairs = [(r, s) for r in spec.regions for s in spec.skus]
        pair_df = self.spark.createDataFrame(pairs, schema=_PAIR_SCHEMA).repartition(self.partitions)

        entity_cls = self.entity_type
        rows_per_sku = spec.rows_per_sku
        out_schema = self.schema

        def _emit(itr: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            for batch in itr:
                buffer: list[dict] = []
                for region, sku in zip(batch["region"], batch["sku"]):
                    for _ in range(rows_per_sku):
                        buffer.append(asdict(entity_cls(region=region, sku=sku)))
                        if len(buffer) >= _YIELD_CHUNK:
                            yield pd.DataFrame(buffer)
                            buffer = []
                if buffer:
                    yield pd.DataFrame(buffer)

        return pair_df.mapInPandas(_emit, schema=out_schema)
