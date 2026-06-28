"""Per-SKU model lookup via Databricks Lakebase (managed Postgres OLTP).

Scaling target: million-SKU bundles where neither preload (OOMs above ~1k
SKUs/replica) nor parquet partition-prune (cold p50 ~280ms is dominated by
filesystem listing) is fast enough at p99.

Lakebase serves point lookups from the page cache → typically sub-10ms. The
serving endpoint holds ONE pooled Postgres connection, not a model dict, so
memory is constant in SKU count.

The class is a drop-in `mlflow.pyfunc.PythonModel` so it can be
`mlflow.pyfunc.save_model(...)`d and deployed to a Model Serving endpoint.

**No assets are created by this module.** It assumes the following are
provisioned externally:

  1. A Lakebase database instance in the workspace.
  2. A Postgres table mirroring the (sku, region, model_blob_bytes) shape of
     the bundle parquet — typically synced from a UC Delta table.
  3. A workspace credential (PAT, OAuth token, or service principal) with
     SELECT on that table.

When ready to provision, wire the instance name + table name into
`LakebaseLookupModel(...)` and deploy. The integration points marked
`# INTEGRATION:` below show where the SDK calls go; exact method names depend
on the databricks-sdk-python version (the Lakebase API surface is named
`workspace_client.database` as of v0.30+).
"""
from __future__ import annotations

import pickle
import time
from typing import Any

import mlflow


FEATURE_COLS = ["price", "day_of_week", "promotion", "inventory"]


class LakebaseLookupModel(mlflow.pyfunc.PythonModel):
    """PyFunc model that fetches per-SKU model bytes from a Lakebase table.

    Args:
        instance_name: Lakebase database instance name (workspace-scoped).
        database: Postgres database within the instance.
        table: Fully qualified Postgres table name holding `(sku, region,
            model_blob_bytes)` rows. Schema-qualified (e.g. `public.models`).
        region: Which region's models to look up. The synced source is the
            cross-region telemetry-with-blob table; we filter to one region
            per endpoint to keep query plans simple.
        pool_min_size: Min connections in the psycopg connection pool.
        pool_max_size: Max connections — must be ≥ serving worker count.
    """

    def __init__(
        self,
        instance_name: str,
        database: str,
        table: str,
        region: str,
        pool_min_size: int = 2,
        pool_max_size: int = 16,
    ) -> None:
        self._instance_name = instance_name
        self._database = database
        self._table = table
        self._region = region
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: Any = None

    def load_context(self, context) -> None:
        from databricks.sdk import WorkspaceClient
        from psycopg_pool import ConnectionPool

        w = WorkspaceClient()

        # INTEGRATION: Lakebase instance discovery + credential generation.
        # Exact SDK shape varies by version — confirm against the installed
        # databricks-sdk before deploying.
        instance = w.database.get_database_instance(name=self._instance_name)
        host = instance.read_write_dns

        cred = w.database.generate_database_credential(
            instance_names=[self._instance_name],
            request_id=f"lakebase-lookup-{self._region}",
        )

        conninfo = (
            f"host={host} port=5432 dbname={self._database} "
            f"user={w.current_user.me().user_name} password={cred.token} "
            f"sslmode=require"
        )
        self._pool = ConnectionPool(
            conninfo=conninfo,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            open=True,
        )

    def _fetch_blob(self, sku: str) -> bytes:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT model_blob_bytes FROM {self._table} "
                    f"WHERE sku = %s AND region = %s LIMIT 1",
                    (sku, self._region),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(f"no model for sku={sku} region={self._region}")
        return bytes(row[0])

    def predict(self, context, model_input):
        """Vectorized per-SKU predict.

        model_input: pandas DataFrame with columns:
            sku, price, day_of_week, promotion, inventory
        Returns: list[float] of predictions, one per row.
        """
        results = []
        for _, row in model_input.iterrows():
            blob = self._fetch_blob(row["sku"])
            model = pickle.loads(blob)
            features = [[row[c] for c in FEATURE_COLS]]
            results.append(float(model.predict(features)[0]))
        return results

    def _predict_with_timing(self, sku: str, features=None):
        if features is None:
            features = [[10.99, 3, 0, 500]]
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        blob = self._fetch_blob(sku)
        timings["t_lakebase_query"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        model = pickle.loads(blob)
        timings["t_unpickle"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        result = model.predict(features)
        timings["t_predict"] = time.perf_counter() - t2

        return result, timings
