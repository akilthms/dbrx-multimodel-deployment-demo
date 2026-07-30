"""Models-from-code entrypoint for the fleet ROUTER endpoint.

The single default entry point for the serving fleet: takes `(region, sku)`,
resolves the owning shard endpoint from the baked master index, forwards the
scoring request to that endpoint, and relays the reply.

Routing is mapping-agnostic and matches ServingFleet.route: compute the shard
via modulo (`bucket(sku) % shard_count`), then look up the endpoint bound to
(region, shard) in the master index. The index (a small parquet, ~one row per
region×shard) is baked in as the `index` artifact — no Volume/FUSE read at
inference (endpoints can't read Volumes).

models-from-code (set_model at end). Deps: pyarrow + requests only.

AUTH: a serving container has NO default Databricks credentials — calling
sibling endpoints requires an explicit token. `deploy_router` injects
DATABRICKS_HOST + DATABRICKS_TOKEN (secret-backed) as endpoint env vars; this
model reads them and calls the shard endpoint's /invocations with requests. The
token's identity (service principal) must have CAN_QUERY on the shard endpoints.
"""
import os

import pyarrow.parquet as pq
import mlflow
from mlflow.models import set_model

INDEX_KEY = "index"


def _sku_bucket(sku: str, bucket_count: int) -> int:
    # Mirror uc_table._sku_bucket: numeric suffix of SKU-NNNNNN modulo bucket_count.
    return int(sku.split("-")[-1]) % bucket_count


class RouterModel(mlflow.pyfunc.PythonModel):
    """Routes (region, sku) → shard endpoint and relays the prediction."""

    def load_context(self, context):
        rows = pq.read_table(context.artifacts[INDEX_KEY]).to_pylist()
        # (region, shard) -> endpoint_name
        self._route = {(r["region"], int(r["shard"])): r["endpoint_name"] for r in rows}
        self._shard_count = int(rows[0]["shard_count"]) if rows else 1
        self._bucket_count = int(rows[0].get("bucket_count", 64)) if rows else 64
        # Serving containers have no default auth — read the injected creds.
        self._host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        self._token = os.environ.get("DATABRICKS_TOKEN", "")

    def _endpoint_for(self, region: str, sku: str) -> str:
        shard = _sku_bucket(sku, self._bucket_count) % self._shard_count
        ep = self._route.get((region, shard))
        if ep is None:
            raise KeyError(f"no endpoint for region={region} shard={shard}")
        return ep

    def predict(self, context, model_input):
        import requests

        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        results = []
        for _, row in model_input.iterrows():
            ep = self._endpoint_for(row["region"], row["sku"])
            rec = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in model_input.columns}
            url = f"{self._host}/serving-endpoints/{ep}/invocations"
            resp = requests.post(url, headers=headers, json={"dataframe_records": [rec]}, timeout=60)
            resp.raise_for_status()
            results.append(resp.json()["predictions"][0])
        return results


set_model(RouterModel())
