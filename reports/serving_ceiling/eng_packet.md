t# Model Serving — large baked-artifact deploy failure (eng triage packet)

## Summary
Deploying an MLflow model with a very large baked artifact (`log_model(artifacts=...)`)
to Model Serving fails at container image creation. 10 GB and 30 GB succeed; ~1 TB fails
on both Small and Large workload sizes with a platform-side internal error. Need the
authoritative per-endpoint baked-artifact size ceiling and the root cause of the internal
error.

## Workspace
- URL: https://e2-demo-field-eng.cloud.databricks.com
- workspace_id: 1444828305810485
- user: akil.thomas@databricks.com (id 3061324493684629)
- profile: e2-field-demo-west

## Model under test (same model at all sizes; only artifact size differs)
- UC model: `users.akil_thomas.mmd_storage_probe`
- Trivial hello-world pyfunc (models-from-code, `pip_requirements=["mlflow"]`) — the model
  code is minimal; the artifact is a replicated parquet bundle. Same model served fine at
  10/30 GB, so packaging is NOT the issue.
- 1 TB artifact = version 10, run_id `782f72c493cf4566b51c901e8774444d`,
  source `models:/m-c929f6b149064673bb7013bb8502f6d5` (~1,012 GB, 14× replicated NORTHEAST bundle)

## Results
| Artifact size | Workload | Outcome | Time to fail |
|---|---|---|---|
| 10 GB | Small | READY, served | ~20 min |
| 30 GB | Small | READY, served | ~21 min |
| 1 TB (v10) | Small | UPDATE_FAILED | ~22 min |
| 1 TB (v10) | Large | UPDATE_FAILED | ~22 min |

## Exact error (verbatim, from build-logs at config_version=1)
```
Container image creation failed: internal error
Build could not start due to an internal error - please contact your Databricks representative.
```
- Failure is at "build could not start" — BEFORE image build executes (contrast: an earlier
  dependency bug produced full conda/pip build logs; this produces none).
- Tier-independent: Small and Large failed identically, ruling out workload disk/mem size.
- NOTE: the 1 TB endpoints (`mmd-storage-probe-1tb`, `-1tb-large`) were deleted, so their
  live event history is gone. The 100 GB / 500 GB endpoints below are LEFT RUNNING with
  captured logs for triage.

## Questions for eng
1. What is the internal error behind "Build could not start"? (control-plane build-service logs)
2. What is the hard max baked-artifact size for Model Serving? (30 GB works, 1 TB doesn't)
3. Is the limit on artifact bytes, image layer size, build-host disk, or a pull timeout?

## Bracketing tests in flight (see accompanying log files)
- 100 GB → endpoint `mmd-storage-probe-100gb` → log: `.isaac/probe_100gb.log`
- 500 GB → endpoint `mmd-storage-probe-500gb` → log: `.isaac/probe_500gb.log`
- Endpoint events/build-logs captured to: `.isaac/probe_100gb_events.json`, `.isaac/probe_500gb_events.json`