# 500 GB baked-artifact serving deploy — FAILED (for eng)

## Ask
Root-cause the "internal error" that prevents a large baked-artifact model from
deploying to Model Serving, and confirm the real max baked-artifact size.

## Endpoint (LEFT RUNNING for inspection — not deleted)
- Endpoint: `mmd-storage-probe-500gb`
- Workspace: https://e2-demo-field-eng.cloud.databricks.com (id 1444828305810485)
- Served model: `users.akil_thomas.mmd_storage_probe` v12, served entity `mmd_storage_probe-12`, config version 1
- Workload size: Large; artifact ≈ 500 GB baked parquet bundle

## Result
`UPDATE_FAILED` after ~2h in `Container creation pending`. Events show 4×
"Container image creation initiated" then "Container image creation failed:
internal error". Build logs (config_version=1):

```
Build could not start due to an internal error - please contact your Databricks representative.
```

Build never started → no deeper logs exposed via API. Same signature as the 750 GB
and 1 TB failures (tier-independent).

## Confirmed size grid (same trivial hello-world model; only artifact size differs)
| Artifact | Workload | Result |
|---|---|---|
| 10 GB | Small | READY, served |
| 30 GB | Small | READY, served |
| 500 GB | Large | FAILED (internal error) |
| 750 GB | Large | FAILED (internal error) |
| 1 TB | Small & Large | FAILED (internal error) |

→ Empirical safe ceiling is between **30 GB and 500 GB** (eng previously cited ~768 GB,
but 500 GB already fails — usable headroom is far below 768).

## Questions for eng
1. What is the internal error behind "Build could not start" for large baked artifacts?
2. What is the real max baked-artifact size? (30 GB works, 500 GB doesn't.)
3. Is the limit on artifact bytes, image layer size, build-host disk, or a pull timeout?

## Attached
- `events.json` — full endpoint event timeline
- `build_logs.json` — build-logs at config_version=1
- `endpoint.json` — full endpoint config/state
