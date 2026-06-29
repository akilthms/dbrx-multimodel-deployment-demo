# Multi-Model Logging + Serving at Scale — Architecture v2

## Headline

Per-region MLflow logging + per-SKU serving lookup for **1M+ models** on Databricks. Validated at **20k SKU × 5 regions × 3 model_names = 300k models** with cold lookup latency **839 ms p50** (sub-linear scale from 1k).

## The architecture in four moves

### 1. `uc_table` strategy — N experiments × 1 run, tagged to a UC Delta table

Move every per-SKU detail OUT of the MLflow tracking API and INTO Unity Catalog. The MLflow side keeps the **bare minimum** the serving endpoint needs: a single deployable run, tagged with pointers.

```
per region:
  experiment "Demand_Forecasting-<REGION>_endpoint_00"   →  1 MLflow run
  experiment "Demand_Forecasting-<REGION>_endpoint_01"   →  1 MLflow run
  experiment "Demand_Forecasting-<REGION>_endpoint_02"   →  1 MLflow run
                                                              ↓ tags:
                                                              model_bundle_uri = /Volumes/.../<region>/bundle_parquet
                                                              demand_forecasting_table = <your_catalog>.<your_schema>.demand_forecasting_artifacts
                                                              genie_space_id = …
```

**MLflow runs at full scale**: `n_endpoints_per_region × n_regions` = **15** (at 3 × 5).
**Not** 1.5M. The tracking API ceiling stops mattering — we never touch it per SKU.

Per-SKU detail (params, metrics, model bytes) lands in two UC objects:

- **`demand_forecasting_artifacts` Delta table** — `(region, sku, model_name, rmse, mape, r2, params MAP, metrics MAP, logged_at)`. Searchable. SQL-queryable. This is the **system of record** for which SKU's model performs how.
- **Per-region `bundle_parquet/` directory in a UC Volume** — `(sku, model_name, model_blob_bytes, params, metrics)`, partitioned for serving-time point lookup (see §3).

### 2. Genie space — natural-language query over the artifact table

A Databricks Genie space is provisioned in the same UC schema, pointed at the `demand_forecasting_artifacts` table, and seeded with example questions + canned SQL. The customer's analyst opens the Genie UI, types "which 10 SKUs in NORTHEAST have the highest RMSE?", and gets the rows back without writing SQL.

Seeded patterns:

- `What is the average RMSE per region?`
- `Top 10 SKUs by lowest RMSE in NORTHEAST` 
- `How many SKUs have MAPE below 0.10 in each region?`
- Plus 9 sample questions in the side panel

Column comments are pulled from the `TrainedModelTelemetry` dataclass's `Annotated[T, "description"]` metadata — Genie reads them and learns the schema's semantics automatically.

### 3. Bundle parquet — intelligent write design

**Goal**: write per-region model bundles that scale to 100k+ SKU per region without OOM AND support sub-second cold lookup at serving time. The write design has three knobs and one structural rule.

| Knob | Value | Why |
|---|---|---|
| `bucket_count` | 64 | Read-side dir fan-out. Picked from iter sweep — 64 buckets gives the best cold-lookup latency at 10-20k SKU; more buckets hurt `load_context` (PyArrow dir enumeration cost) |
| `target_bytes_per_task` | 2 GB | Per-Spark-task memory bound. At 10k SKU = 66 partitions; at 20k = 132 partitions. Each task = 1-2 GB Python worker memory — fits in 32 GB worker without OOM |
| `parquet.block.size` | 16 MB | Row group size. Spark default is 128 MB; smaller row groups = less IO per cold lookup (PyArrow reads ONE row group per query). 16 MB ≈ 4 model blobs |

The structural rule: **write produces multiple files per bucket with non-overlapping SKU ranges**, so PyArrow row-group skipping works file-by-file.

```python
plan_df
  .withColumn("bucket", expr("cast(substring(sku,5,10) as int) % 64"))
  .repartitionByRange(n_partitions, "bucket", "sku")    # range-partition by (bucket, sku)
  .sortWithinPartitions("bucket", "sku")                # local sort → ordered row groups
  .write
    .partitionBy("bucket")                              # Hive-style bucket=N dirs
    .option("parquet.block.size", 16 * 1024 * 1024)
    .parquet(bundle_uri)
```

Critical pairing: **`repartitionByRange` alone is not enough**. It distributes by range across partitions but doesn't sort WITHIN each partition. Without the explicit `sortWithinPartitions`, row groups have arbitrary SKU order — min/max stats are useless and PyArrow scans the whole file. Empirically: skipping the sort took cold p50 from 784 ms → 12,140 ms at 10k.

### 4. Bundle parquet — intelligent read design

Two patterns ship in the package. The customer picks based on their serving endpoint cold-start vs per-request latency tradeoff.

#### Approach 1 — `PyArrowLruModel` (recommended starting point)

```python
self._dataset = ds.dataset(bundle_uri, format="parquet", partitioning="hive")
self._get_model = lru_cache(maxsize=cache_size)(self._load_model_uncached)

def _load_model_uncached(self, sku):
    bucket = _sku_bucket(sku, bucket_count)
    table = self._dataset.to_table(
        filter=(ds.field("bucket") == bucket) & (ds.field("sku") == sku),
        columns=["model_blob_bytes"],
    )
    return pickle.loads(table.column("model_blob_bytes")[0].as_py())
```

- **Cold lookup**: ~700-840 ms at 10-20k scale. PyArrow opens the bucket dir, reads each file's parquet footer, applies row-group skipping via min/max sku stats, reads the one matching row group.
- **Warm lookup**: ~30 ms (`lru_cache` hit + `model.predict()`).
- **Endpoint cold-start (`load_context`)**: ~2-6 seconds (just opens the dataset; no metadata read until first query).

#### Approach 1b — `PyArrowFooterIndexModel` (lower per-request cold)

```python
# load_context: walk all parquet files, read row-group min/max stats from footer,
# build a sorted index list[(sku_min, sku_max, file_path, rg_idx)]
# lookup: binary search the index → 1 row group read

def load_context(self, context):
    entries = []
    for bucket_dir in os.listdir(bundle_uri):
        for fname in os.listdir(bucket_dir):
            pf = pq.ParquetFile(fp)
            for rg_idx in range(pf.num_row_groups):
                stats = pf.metadata.row_group(rg_idx).column(sku_idx).statistics
                entries.append((to_int(stats.min), to_int(stats.max), fp, rg_idx))
    entries.sort()
    self._sku_mins = [e[0] for e in entries]
    self._entries = entries

def _load_model_uncached(self, sku):
    sku_int = to_int(sku)
    idx = bisect.bisect_right(self._sku_mins, sku_int) - 1
    if idx >= 0 and sku_int <= self._entries[idx][1]:
        fp, rg_idx = self._entries[idx][2], self._entries[idx][3]
        return pickle.loads(read_one_row_group(fp, rg_idx)[0])
```

- **Cold lookup**: ~186 ms at 10k (4× faster than Approach 1).
- **Endpoint cold-start (`load_context`)**: depends on file count — ~30 s at 10k scale when footer reads are sequential. Parallelize footer reads to keep startup bounded at scale.
- **Tradeoff**: pays the per-replica startup cost ONCE in exchange for per-request latency. Best for endpoints with long-running replicas serving many cold lookups.

#### Approach 2 (production scale) — Lakebase point lookup

Ships as code only (`adapters.LakebaseLookupModel`). When the customer's serving traffic exceeds what UC Volume FUSE can handle under concurrency, sync the artifact table to a Lakebase (managed Postgres) instance and serve point lookups from there. Sub-10 ms typical, SLA-backed.

## Scaling proof — the cold latency curve

Bundle write + serving benchmark, same hardware (6 × m5d.2xlarge workers + 1 × i3.4xlarge driver), same code path, increasing SKU scale per region:

| Scale (SKU/region) | Cold p50 | Cold p95 | Warm p50 | Strategy write | Wall-clock |
|---:|---:|---:|---:|---:|---:|
| 1,000 | **650 ms** | 1,244 ms | 31 ms | 76 s | 6.8 min |
| 10,000 | **784 ms** | 2,531 ms | 33 ms | 722 s | 21 min |
| 20,000 | **839 ms** | 2,335 ms | 33 ms | 3,540 s | 77 min |

**Cold p50 grew only 29 % across a 20× SKU range.** The bucketed-partition + row-group-skipping read path is what makes this possible. Total models logged at 20k: 5 regions × 20k SKU × 3 model_names = **300k models**.

Why we believe it goes further (100k+ per region):

- Per-Spark-task memory is **bounded by configuration**, not by scale. `target_bytes_per_task=2 GB` keeps Python workers safe regardless of N_SKU.
- Cold lookup cost is **bounded by per-bucket file count**, not total SKU count. 64 buckets stays cold-fast up to ~thousand-SKU-per-bucket density.
- The MLflow tracking API is **completely off the critical path** — 15 `create_run` calls total at any scale.

Strategy write wall-clock DOES grow with N_SKU (it's bound by the bundle parquet shuffle volume). Acceptable for offline batch logging; not on the serving hot path.

## What was tried and discarded

| Attempt | Result | Why discarded |
|---|---|---|
| `partitionBy("sku")` — one parquet dir per SKU | 10k SKU = 10k dirs/region. `mode("overwrite")` couldn't wipe via Hadoop FS (UC Volumes reject the API) | Doesn't scale, infrastructure-hostile |
| `shutil.rmtree` over FUSE mount for wipe | 50k POSIX unlinks per region, ~30 min wipe time | Slow; switched to `WorkspaceClient.dbutils.fs.rm` |
| `PreloadAllModel` (load every model into a serving-time dict) | OOM at 5k SKU during `pyarrow.Table.to_pylist()` (~66 GB peak Python memory) | Doesn't fit any serving replica — removed from demo |
| `repartition(BUCKET_COUNT, "bucket")` (one task per bucket) | OOM at 20k SKU — each task held 2-4 GB of model blobs | Replaced with memory-bounded `repartitionByRange` |
| `BUCKET_COUNT=256` at 10k SKU | Cold p50 = 2,559 ms (vs 704 ms with 64 buckets) | More buckets = more dir enumeration overhead, hurt cold lookup. Reverted |
| `repartitionByRange` without `sortWithinPartitions` | Cold p50 = 12,140 ms — row groups unsorted, stats useless | Added explicit sortWithinPartitions |
| `.repartition(N, "region")` with N > num_regions | Hash collapsed to 5 partitions — same OOM | Switched to `.repartition(N, "region", "sku")` multi-column hash |

## File map

| Component | Path |
|---|---|
| `uc_table` strategy + bundle write | `src/dbrx_multimodel_registration/adapters/logging/uc_table.py` |
| `LakebaseLookupModel` (production-scale read) | `src/dbrx_multimodel_registration/adapters/serving/lakebase.py` |
| `WorkloadBudget` config entity | `src/dbrx_multimodel_registration/domains/entities.py` |
| Run plan persist (memory-bounded) | `src/dbrx_multimodel_registration/adapters/storage/delta_run_plan.py` |
| Demand + run plan expand-and-rekey | `src/dbrx_multimodel_registration/utils/scaling.py` |
| `PyArrowLruModel` + `PyArrowFooterIndexModel` benchmark | `notebooks/demo.ipynb` cells 31 + 32 |
| Local pyspark tests (pre-deploy validation) | `tests/test_bundle_write.py` |

## What ships to the customer

1. The `uc_table` strategy as the recommended logging path.
2. `PyArrowLruModel` as the production serving class (drop-in PyFunc).
3. `LakebaseLookupModel` as the next-step code template when point-lookup latency at scale matters.
4. `WorkloadBudget` knobs to tune per-cluster (no hardcoded constants).
5. Scaling numbers above as the talking point: **300k models logged + sub-1-second cold serving lookup**.
