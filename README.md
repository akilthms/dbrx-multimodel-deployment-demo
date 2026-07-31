# Multi-Model Deployment on Databricks

Serve **hundreds of thousands of models** behind a sharded fleet of Databricks
Model Serving endpoints. Models are grouped into per-shard bundles (one bundle
baked into each endpoint), and a lightweight **router** endpoint resolves every
`(region, sku)` request to the endpoint that owns that model.

```
model_artifacts table (bucketed, Unity Catalog)          ← source of truth (+ Genie view)
        │  populate shard bundles (parallel, per-shard bucket subsets)
        ▼
/Volumes/.../<region>/shard_<N>_<s>/  bundles             ← baked into endpoints
        │
        ▼
 N_REGIONS × SHARD_COUNT serving endpoints  +  1 router   ← router: (region, sku) → shard endpoint
```

---

## Quickstart

### Prerequisites
- Databricks CLI installed and authenticated:
  ```bash
  databricks auth login --profile <your-profile>
  ```
- A Unity Catalog **catalog + schema** you can write to, and a **Volume** for artifacts.
- Create a `.env` in the repo root (git-ignored):
  ```bash
  DATABRICKS_CONFIG_PROFILE=<your-profile>
  BUNDLE_VAR_run_as_user=<you>@<company>.com
  BUNDLE_VAR_catalog=<your_catalog>
  BUNDLE_VAR_schema=<your_schema>
  ```

### Step 1 — Deploy the bundle (builds + uploads the wheel and jobs)
```bash
databricks bundle deploy -t dev
```

### Step 2 — Smoke test (mock data — see the architecture work end-to-end)
Builds a tiny `model_artifacts` table with mock models, shards it, deploys
`regions × shard_count` endpoints **plus a router**, and prints their names.
No customer data required.
```bash
databricks bundle run fleet_smoke
```
When it finishes, poll readiness and query **through the router**:
```bash
# poll (repeat until ready=READY)
databricks serving-endpoints get mmd-router-of2

# query the router — it routes (region, sku) to the owning shard endpoint
databricks serving-endpoints query mmd-router-of2 --json '{
  "dataframe_records": [
    {"region": "NORTHEAST", "sku": "SKU-000007",
     "price": 10.99, "day_of_week": 3, "promotion": 0, "inventory": 500}
  ]
}'
# → {"predictions": [ <forecast> ]}
```
**Tear down when done** (endpoints are billable):
```bash
for ep in mmd-router-of2 mmd-northeast-s0-of2 mmd-northeast-s1-of2 \
          mmd-southeast-s0-of2 mmd-southeast-s1-of2; do
  databricks serving-endpoints delete $ep
done
```

### Step 3 — Full pipeline (your data)
Point the fleet at **your** `model_artifacts` table
(`region · sku · model_name · model_blob_bytes · bucket` + optional telemetry).
```bash
databricks bundle run fleet_deploy -- \
  --table            <catalog>.<schema>.model_artifacts \
  --catalog          <catalog> \
  --schema           <schema> \
  --experiment       /Users/<you>/mmd_fleet \
  --artifact-volume  /Volumes/<catalog>/<schema>/mlflow_artifacts \
  --shard-count      4 \
  --buckets          64        # must match how the table was bucketed
```
Regions are read from the table. The run deploys all shard endpoints + a router
and writes the routing index; query the router exactly as in Step 2.

#### Sizing `--shard-count`
Each shard's bundle is **baked into** its endpoint, and baked artifacts have a
size ceiling. **≤ 30 GB per shard is proven-safe; larger sizes have failed in
testing.** Size the shard count so each shard stays under the cap:
```
shard_count ≥ ceil(region_bytes / 30 GB)      # region_bytes ≈ models_per_region × ~4 MB
```
`--enforce-size-limit` (default on) fails fast if any shard exceeds `--max-shard-gb`
(default 30 GB) — raise `--shard-count` if it trips.

---

## Building the `model_artifacts` table
If you don't already have one, `ModelArtifactsTable` builds it from a
trained-model DataFrame (`region, sku, model_name, model_blob_bytes, params,
metrics`) — it derives the `bucket` column, top-level `rmse/mape/r2`, table
comments, and a **blob-excluding Genie view** for natural-language querying.
See `src/dbrx_multimodel_registration/adapters/storage/model_artifacts.py`.

## More
- **Detailed guide:** [`docs/serving_fleet_quickstart.md`](docs/serving_fleet_quickstart.md)
- **Architecture rationale + design notes:** [`prds/PRD_serving_fleet.md`](prds/PRD_serving_fleet.md)
- **Known limits:** Model Serving endpoints have no `/Volumes` FUSE at inference
  (artifacts are baked in at deploy); the per-endpoint baked-artifact size ceiling
  is the main scaling constraint (see the shard-sizing note above).
