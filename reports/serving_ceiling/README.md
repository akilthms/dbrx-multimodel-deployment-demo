# Model Serving — large baked-artifact deploy failures (eng triage)

**TL;DR:** Deploying an MLflow model with a large **baked artifact**
(`log_model(artifacts=…)`) to Model Serving fails above ~30 GB, in **two distinct
failure modes** by size. Small artifacts deploy fine. Blocking a **GCP** customer serving
~500k models (~2 TB total) via sharded bundles behind multiple endpoints — the max
baked-artifact size sets the endpoint count and is undocumented.

_This file consolidates the full investigation. Raw logs/events are in this directory and
`eng_500gb/`._

---

## Environment
- **Workspace:** `e2-demo-field-eng` · id `1444828305810485` · https://e2-demo-field-eng.cloud.databricks.com
- **Cloud:** customer target is **GCP**; this reproduction is on the AWS test workspace above.
- **Model (constant across every test):** `users.akil_thomas.mmd_storage_probe` — a trivial
  hello-world `python_function` (models-from-code). **Only the baked artifact SIZE varies**;
  model code is irrelevant to the failure.

## Results — same model, varying baked-artifact size
| Artifact | Workload | Result | Failure mode |
|---|---|---|---|
| 10 GB  | Small | ✅ READY, served (~20 min) | — |
| 30 GB  | Small | ✅ READY, served (~21 min) | — |
| 100 GB | Large | ❌ FAILED (~6h) | **A — slow timeout** |
| 200 GB | Large | ❌ FAILED (`UPDATE_FAILED`) | **A — slow timeout** |
| 500 GB | Large | ❌ FAILED (~4h) | **B — instant internal error** |
| 750 GB | Large | ❌ FAILED | **B — instant internal error** |
| 1 TB   | Small & Large | ❌ FAILED | **B — instant internal error** |

**➡️ Safe baked-artifact ceiling: `30 GB ≤ max < 100 GB`** — far below the ~768 GB figure
previously cited. Practical guidance: keep per-endpoint baked artifacts **≤ 30 GB**.

## ⭐ Two distinct failure modes (key finding)
### Mode A — slow timeout (~100–200 GB)
- Container image creation **initiated repeatedly** (100 GB: 3× at 17:39, 19:44, 21:48 UTC);
  build **succeeded** through conda-env creation.
- Then `Endpoint update failed for endpoint …, config version 1` at **~6h** (100 GB: 07-30 23:39 UTC).
- No explicit error surfaced → looks like a **build-duration / deployment-window timeout**, not a size rejection.

### Mode B — instant internal error (≥500 GB)
- `Container image creation failed: internal error` →
  `Build could not start due to an internal error - please contact your Databricks representative.`
- Fails **before the build runs**; tier-independent (Small == Large).

## Deployment coordinates for the failed runs
> Model Serving exposes **no separate "deployment ID."** A failed deploy is identified by
> **endpoint (name + id) + served-entity + config version**. `entity_version` = the UC model
> version baked. Some endpoints were deleted before the "keep failures" rule, so their
> `endpoint_id` is gone (deletion destroys event history) — served-entity/model-version remain.

| Size | Mode | endpoint name | endpoint_id | served entity | cfg ver | model ver | status |
|---|---|---|---|---|---|---|---|
| 100 GB | A | `mmd-storage-probe-100gb` | `06ecf05cc72d4783b7be3131bc1d1c60` | `mmd_storage_probe-13` | 1 | 13 | **LIVE** (kept) |
| 500 GB | B | `mmd-storage-probe-500gb` | `b06a447217044633861407709436f810` | `mmd_storage_probe-12` | 1 | 12 | **LIVE** (kept) |
| 200 GB | A | `mmd-storage-probe-200gb` | (deleted) | `mmd_storage_probe-14` | 1 | 14 | deleted |
| 750 GB | B | `mmd-storage-probe-750gb` | (deleted) | `mmd_storage_probe-11` | 1 | 11 | deleted |
| 1 TB   | B | `mmd-storage-probe-1tb` / `-1tb-large` | (deleted) | `mmd_storage_probe-10` | 1 | 10 | deleted |

**Live endpoints (100 GB + 500 GB) cover both failure modes** for inspection.
Build-logs locator: `GET /api/2.0/serving-endpoints/<ep>/served-models/<served-entity>/build-logs?config_version=1`

## Questions for eng
1. **Mode A:** Is there a build/deployment **timeout** a ~100 GB artifact exceeds? What is it,
   is it tunable, and which stage times out (image build, artifact push, replica pull/start)?
2. **Mode B:** What internal error blocks the build from **starting** at ≥500 GB? Hard
   artifact-size / image-layer cap?
3. **Overall:** What is the **supported max baked-artifact size** for Model Serving?
   (30 GB works, 100 GB times out, ≥500 GB rejected — undocumented.)
4. Recommended pattern for serving many-GB of models per endpoint **other than baking**,
   given GCP rules out Lakebase and endpoints have no `/Volumes` FUSE mount at inference?

## Filing (ES ticket)
- Type: Incident · Severity: **SEV1 High** (`customfield_11500` id `10601`) ·
  External Customer Facing: **No** (`customfield_14677` id `13345`) · Affects Versions: N/A ·
  ES Component: Model Serving · Cloud: GCP · Assign: Unassigned.
- **Note:** JIRA MCP was unusable during the session (IP-ACL/timeout hangs from a bouncing
  client IP). File via `go/FEfileaticket`; attach the artifacts below.
- Slack thread (#model-serving): https://databricks.slack.com/archives/CPS3MLXUJ/p1784825239849739?thread_ts=1784552406.538799

## Artifacts in this directory
- `eng_500gb/` — 500 GB (Mode B) packet: `events.json`, `build_logs.json`, `endpoint.json`, `failure_report.json`
- `mmd-storage-probe-100gb_events.json`, `mmd-storage-probe-100gb_buildlogs.json` — 100 GB (Mode A)
- `probe_*.log` — run logs (100/200/500/750 GB)
