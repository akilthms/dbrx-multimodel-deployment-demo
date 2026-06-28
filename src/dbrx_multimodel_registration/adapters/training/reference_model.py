"""Train ONE reference RandomForestRegressor on one (region, sku) slice.

Used in stage 2 of the pipeline. The resulting pickled bytes are then
broadcast in stage 4 (`TrainingSimulator`) and reused for every logged
entry — the demo benchmarks *logging* throughput, not training quality.
"""
from __future__ import annotations

import pickle

from pyspark.sql import DataFrame
from pyspark.sql.functions import col


class ReferenceModelTrainer:
    """Target output: ~5MB pickled RF. Defaults are tuned for that ballpark.

    If the resulting blob is too small for the demo's narrative, bump
    `n_estimators` or `max_depth` — both increase model size superlinearly.
    """

    name = "rf_reference"

    def __init__(self, n_estimators: int = 500, max_depth: int = 20) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train(self, demand_df: DataFrame, region: str, sku: str) -> bytes:
        from sklearn.ensemble import RandomForestRegressor

        sample = (
            demand_df.where((col("region") == region) & (col("sku") == sku))
            .toPandas()
        )
        if sample.empty:
            raise ValueError(f"no demand rows for (region={region}, sku={sku})")

        x = sample[["price", "day_of_week", "promotion", "inventory"]].astype(float)
        y = sample["demand"].astype(float)
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=0,
        )
        model.fit(x, y)
        return pickle.dumps(model)
