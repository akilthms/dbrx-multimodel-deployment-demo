# PRD — Multi-Model Serving Fleet ("04 Serve", corrected)

## Context

The customer (GCP) must serve ~500k demand-forecasting models (5 regions × ~20–100k
SKU × 3 model_names, ~4 MB/model on disk ≈ ~2 TB total) behind the fewest Databricks
Model Serving endpoints. The v3 architecture diagram's "04 Serve" lane shows a
`PyArrowLruModel` reading the parquet bundle from a UC Volume at inference — **this
session proved that does not work**: Model Serving endpoints have no `/Volumes` FUSE
mount at inference; artifacts are baked into the model image at deploy time and served
from the endpoint's local disk. Lakebase point-lookup (the diagram's "next step") is
also out — not available on GCP.

This PRD specifies the **corrected** serving architecture, grounded in empirical results
from this session, and the `ServingFleet` implementation that realizes it (already built
and validated end-to-end — see Status).

## Empirical findings that anchor the design

- **Baked artifacts work; there is a size ceiling.** 10 GB ✅, 30 GB ✅ deploy and serve.
  1 TB ❌ and 750 GB ❌ fail with "Container image creation failed: internal error / Build
  could not start" — tier-independent (Small == Large). Eng confirmed a ~768 GB limit, but
  750 GB failed below it, so **usable artifact headroom is < 768 GB**. 500 GB result: <PENDING>.
  → **Shard bundles must be sized conservatively** (≤ a re-confirmed safe bound; do NOT
  assume 768 is usable).
- **Packaging must be explicit** or the container build/load fails:
  - Log serving models via **models-from-code** (script path + `set_model()`), not a pickled
    object — a pickled class needs its defining module importable on the endpoint.
  - Pass explicit **`pip_requirements`** — MLflow otherwise bakes the project wheel as a dep
    and the serving `pip install` fails (not on PyPI).
- **Serverless dep install** = Environment `dependencies` spec (or `sys.path` to the
  bundle-synced `src/`), NOT `%pip install <wheel>`.

## Architecture (decisions from the grill)

- **Serving mechanism:** baked-artifact endpoints (design A). Each endpoint ↔ one MLflow
  child run ↔ one shard's parquet bundle (1:1:1).
- **Partitioning:** `N_REGIONS × SHARD_COUNT`. `SHARD_COUNT` is a parameter (perf-sweep
  knob) with a correctness floor: `SHARD_COUNT ≥ ceil(region_bytes / SAFE_SHARD_GB)`.
- **MLflow hierarchy:** ONE experiment `Demand_Forecasting`; region **parent runs**
  (grouping only); shard **child runs** (the endpoint sources). Sweeps add child-run sets
  tagged by `shard_count`.
- **Source of truth:** ONE bucketed UC managed Delta table `model_artifacts` (renamed from
  `run_plan`) carrying blobs + telemetry columns. Genie reads a **blob-excluding view** (no
  copy). A parallel, conflict-free populate step (one worker per disjoint bucket set) writes
  per-shard parquet bundles to pre-allocated Volume paths.
- **Artifact locations:** pinned at **region** level at run creation; shard bundles written
  by the populate step to `<region>/shard_<SHARD_COUNT>_<s>/` (shard count in the path so
  sweeps don't collide).
- **Router:** pure `route(region, sku) -> endpoint` function (+ a default single router
  endpoint wrapper). Two-tier index: build-time **master** index at Volume root
  `_master_index/shard_<N>/` (router loads it) resolving `(region, shard) -> endpoint`; each
  endpoint owns its **fine** `sku -> row-group` footer index internally. Routing is
  mapping-agnostic: compute shard via the mapping, look up the endpoint binding (NOT SKU-range
  matching — modulo shards hold scattered buckets with overlapping ranges).

## Ports & adapters (domain/port/adapter discipline)

- `ports/serving/deployment_strategy.py` → `ServingDeploymentStrategyPort` (`deploy`,
  `route`, `teardown`). Port names the responsibility; baked-fleet is adapter #1, a
  Files-API/single-endpoint adapter is the justified future #2.
- `ports/serving/shard_mapping.py` → `ShardMappingPort` (`shard_for_bucket`,
  `buckets_for_shard`, `shard_for_sku`). Adapter #1 `ModuloShardMapping`; contiguous-range
  is #2.
- `adapters/serving/fleet.py` → `ServingFleet` (one class): `populate_shard_bundles`,
  `deploy_fleet`, `build_routing_index`, `load_index`, `route`, `teardown`, and a thin
  `deploy()` orchestrator. Heavy imports (Spark/MLflow/serving SDK) lazy so `route` stays
  import-light.
- `adapters/serving/shard_model_entrypoint.py` → per-shard models-from-code lookup model
  (footer index → row-group read → unpickle → predict).
- Domain: `ShardPlacement`, `ServingDeployment`, `MAX_SERVING_ARTIFACT_GB`.

## Work items

1. **`model_artifacts` table** — rename `run_plan`; write bucketed; carry blobs + telemetry
   columns + comments; add a blob-excluding Genie view. (`adapters/logging/uc_table.py`,
   `adapters/storage/…`, `domains/entities.py`.)
2. **Parallel populate step** — read table buckets, write per-shard Volume bundles
   (`<region>/shard_<N>_<s>/`), one worker per disjoint bucket set. (Extend `ServingFleet`
   or a build-lane module.)
3. **`ServingFleet`** — DONE (validated e2e). Remaining: wire `populate` to read from the
   `model_artifacts` table rather than an existing `bundle_parquet` dir; deploy the router
   endpoint wrapper; enforce `SAFE_SHARD_GB`.
4. **Router endpoint** — models-from-code wrapper over `route` + master index.
5. **`deploy()` orchestrator + `enforce_size_limit`** with `SAFE_SHARD_GB` default.
6. **Fix `entities.py` faker import** — make `from faker import Faker` lazy so the
   import-light serving path doesn't require faker (violates CLAUDE.md hot-path rule).
7. **DABs**: serverless build job (Environment `dependencies` for the wheel);
   `max_concurrent_runs > 1` (or one run deploying shards concurrently) for parallel deploys.
8. **Notebook `demo.ipynb`** — replace the broken Volume-FUSE `PyArrowLruModel` serving
   cells with the `ServingFleet` deploy + route demo.

## Verification

- **Local (fast):** `pytest tests/test_fleet.py tests/test_shard_mapping.py` — pure build-lane
  + routing correctness, no Databricks. (Currently green.)
- **Serverless smoke:** tiny 5-region × 2-shard fleet → deploy 10 endpoints → route → query.
  DONE — `FLEET_SMOKE_OK: [-0.96…]` (notebooks/_scratch_fleet_smoke.py).
- **Scale smoke:** once a safe shard size is confirmed, deploy the full fleet at
  SAFE_SHARD_GB partitions and validate route→query across regions.

## Open items
- Confirm SAFE_SHARD_GB from the 30–750 GB band (500 GB test in flight).
- Q14 ratified: separate callable steps + thin `deploy()` orchestrator (already built this way).
