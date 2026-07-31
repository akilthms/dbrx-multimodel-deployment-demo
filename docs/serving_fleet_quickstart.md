# Serving Fleet — Quickstart

Deploy many models behind a sharded fleet of Databricks Model Serving endpoints,
with a router that resolves `(region, sku)` to the endpoint owning that model.

Two commands: **`smoke`** (mock data, see the architecture work) and **`deploy`**
(your data). Both run on Databricks via the bundle.

## Prerequisites
- Databricks CLI configured (`databricks auth login --profile <profile>`).
- A UC catalog + schema you can write to, and a Volume for artifacts.
- Clone this repo; set `.env` with `DATABRICKS_CONFIG_PROFILE`, `BUNDLE_VAR_run_as_user`,
  `BUNDLE_VAR_catalog`, `BUNDLE_VAR_schema`.

## 1. Deploy the bundle (builds + uploads the wheel)
```bash
databricks bundle deploy -t dev
```

## 2. Smoke test — mock data, full architecture
Builds a tiny `model_artifacts` table, shards it, deploys `regions × shard_count`
endpoints + a router, and (once ready) you query through the router.
```bash
databricks bundle run fleet_smoke
```
Output prints the router + shard endpoint names. Poll readiness:
```bash
databricks bundle run capacity -- endpoint-status --endpoint mmd-router-of2
```
Then query the router with `(region, sku, price, day_of_week, promotion, inventory)`
records. **Tear down when done:** `databricks serving-endpoints delete <name>` for the
router and each shard endpoint.

## 3. Deploy with your data
Your source is a **`model_artifacts` table**: `region · sku · model_name ·
model_blob_bytes · bucket` (+ optional telemetry columns). `bucket` =
`cast(substring(sku,5,10) as int) % bucket_count`. (See `ModelArtifactsTable` to
build it from a trained-model DataFrame.)

```bash
databricks bundle run fleet_deploy -- \
  --table       <catalog>.<schema>.model_artifacts \
  --catalog     <catalog> \
  --schema      <schema> \
  --experiment  /Users/<you>/mmd_fleet \
  --artifact-volume /Volumes/<catalog>/<schema>/mlflow_artifacts \
  --shard-count 4 \
  --buckets     64            # must match how the table was bucketed
```

### Sizing `--shard-count` (important)
Each shard's bundle is **baked into** its endpoint. Baked artifacts have a size
ceiling — **≤ 30 GB per shard is proven-safe**; larger sizes have failed in testing
(see `reports/serving_ceiling/` / the ES ticket). So:

```
shard_count >= ceil(region_bytes / 30 GB)      # region_bytes ≈ models_in_region × ~4 MB
```

`--enforce-size-limit` (default on) fails fast if any shard exceeds `--max-shard-gb`
(default 30) — raise `--shard-count` if it trips.

## Architecture (what gets created)
```
model_artifacts table (bucketed, UC managed)         ← source of truth (+ Genie view)
        │  populate_shard_bundles_from_table (parallel, per-shard bucket subsets)
        ▼
/Volumes/.../<region>/shard_<N>_<s>/  bundles         ← ≤30GB each
        │  baked into
        ▼
region×shard_count serving endpoints  +  1 router     ← router: (region,sku) → shard endpoint
```

## Notes / limits
- **No `/Volumes` FUSE at inference** — bundles are baked into each endpoint at deploy.
- **Router auth:** the router calls sibling endpoints; it needs `DATABRICKS_HOST` +
  `DATABRICKS_TOKEN` (a service principal with `CAN_QUERY` on the shard endpoints in
  production). The jobs inject the run identity's token automatically.
- **Scale:** validated end-to-end at small scale; run a real-volume test before production.
