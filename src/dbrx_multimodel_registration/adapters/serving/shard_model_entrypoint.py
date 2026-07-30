"""Models-from-code entrypoint for a per-shard serving model.

One of these is baked (with its shard's parquet bundle as the `bundle` artifact)
into each fleet endpoint. At inference it does the per-SKU lookup INSIDE its
bundle: parquet footer-stat index → row-group read → unpickle → predict. This
is the fine tier of the two-tier index (the router owns the coarse
(region,sku)->endpoint tier).

Models-from-code (set_model at the end) so the serving container EXECUTES this
script instead of unpickling a class from the project package — avoids the
ModuleNotFound / phantom-wheel failure modes. Self-contained: stdlib + mlflow +
pyarrow + sklearn (all present on the serving image), so pip_requirements stays
minimal.
"""
import bisect
import os
import pickle

import mlflow
import pyarrow.parquet as pq
from mlflow.models import set_model

ARTIFACT_KEY = "bundle"
MODEL_BLOB_COLUMN = "model_blob_bytes"
FEATURE_COLS = ["price", "day_of_week", "promotion", "inventory"]


def _sku_to_int(sku: str) -> int:
    return int(sku.split("-")[-1])


class ShardLookupModel(mlflow.pyfunc.PythonModel):
    """Per-SKU model lookup over the shard's baked parquet bundle."""

    def load_context(self, context):
        root = context.artifacts[ARTIFACT_KEY]
        # Build the footer-stat index: (sku_min, sku_max, file, rg_idx), sorted.
        entries = []
        for bucket_dir in os.listdir(root):
            if not bucket_dir.startswith("bucket="):
                continue
            bp = os.path.join(root, bucket_dir)
            for fn in os.listdir(bp):
                if not fn.endswith(".parquet"):
                    continue
                fp = os.path.join(bp, fn)
                pf = pq.ParquetFile(fp)
                sku_col = pf.schema_arrow.get_field_index("sku")
                for rg in range(pf.num_row_groups):
                    stats = pf.metadata.row_group(rg).column(sku_col).statistics
                    if stats is None or stats.min is None:
                        continue
                    entries.append((_sku_to_int(stats.min), _sku_to_int(stats.max), fp, rg))
        entries.sort()
        self._mins = [e[0] for e in entries]
        self._entries = entries
        self._cache: dict = {}

    def _find(self, sku_int: int):
        idx = bisect.bisect_right(self._mins, sku_int) - 1
        if idx < 0:
            return None
        lo, hi, fp, rg = self._entries[idx]
        return (fp, rg) if sku_int <= hi else None

    def _model_for(self, sku: str):
        if sku in self._cache:
            return self._cache[sku]
        loc = self._find(_sku_to_int(sku))
        if loc is None:
            raise KeyError(f"no model for sku={sku}")
        fp, rg = loc
        table = pq.ParquetFile(fp).read_row_group(rg, columns=[MODEL_BLOB_COLUMN, "sku"])
        skus = table.column("sku").to_pylist()
        for i, s in enumerate(skus):
            if s == sku:
                model = pickle.loads(table.column(MODEL_BLOB_COLUMN)[i].as_py())
                self._cache[sku] = model
                return model
        raise KeyError(f"sku={sku} indexed but not found in {fp}#{rg}")

    def predict(self, context, model_input):
        results = []
        for _, row in model_input.iterrows():
            model = self._model_for(row["sku"])
            results.append(float(model.predict([[row[c] for c in FEATURE_COLS]])[0]))
        return results


set_model(ShardLookupModel())
