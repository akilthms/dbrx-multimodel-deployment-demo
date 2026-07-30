# ES ticket — ready to file at go/FEfileaticket
(JIRA MCP was unusable — IP-ACL/timeout hangs from a bouncing client IP. File manually.)

## Fields
- Project: ES
- Issue type: Incident
- Severity: SEV1 (High) — field customfield_11500, id 10601
- External Customer Facing?: No — field customfield_14677, id 13345
- Affects Versions: N/A
- ES Component (customfield_18150): select the Model Serving component in the portal
- Cloud: GCP (customer target; test workspace on AWS e2-demo-field-eng)
- Assign: Unassigned
- Workspace: e2-demo-field-eng, workspace_id 1444828305810743... -> 1444828305810485
- URL: https://e2-demo-field-eng.cloud.databricks.com

## Summary
[Field Eng / GCP multi-model] Model Serving deploy fails for large baked artifacts (>=500GB) with internal "Build could not start" error

## Description
Problem: Deploying an MLflow model with a large BAKED artifact (log_model(artifacts=...)) to Model Serving fails at container image creation for large artifact sizes. Small artifacts deploy fine; large ones fail with an internal error before the build starts. Blocks a Field Engineering customer (GCP) serving ~500k models (~4MB each, ~2TB total) by bundling sharded model artifacts behind multiple endpoints — the max baked-artifact size determines endpoint count and is undocumented.

Verbatim failure:
- Event: "Container image creation failed: internal error"
- Build logs (config_version=1): "Build could not start due to an internal error - please contact your Databricks representative."
- Build retried hourly 4x then failed; no deeper logs exposed via API.

Failed run (500GB) identifiers (endpoint LEFT RUNNING for inspection):
- Endpoint: mmd-storage-probe-500gb (endpoint_id b06a447217044633861407709436f810)
- Served entity: mmd_storage_probe-12, config version 1; registered model users.akil_thomas.mmd_storage_probe v12
- Workload: Large; artifact ~500GB baked parquet; flavor python_function (models-from-code, trivial hello-world — only artifact SIZE varies)
- Container creation started 2026-07-29 15:29:21 UTC; retries 16:30:22, 17:31:33, 18:32:52; failed 19:33:50 UTC; DEPLOYMENT_FAILED 19:34:27 UTC (~4h4m)
- Build-logs locator: GET /api/2.0/serving-endpoints/mmd-storage-probe-500gb/served-models/mmd_storage_probe-12/build-logs?config_version=1

Empirical size grid (same trivial model, only artifact size differs):
10GB/Small READY; 30GB/Small READY; 500GB/Large FAILED; 750GB/Large FAILED; 1TB/Small&Large FAILED. => safe ceiling 30GB <= max < 500GB (eng cited ~768GB but 500GB already fails).
(100GB / 200GB tests in flight — will tighten this bound.)

Questions:
(1) root cause of the internal "Build could not start" for large baked artifacts?
(2) real max baked-artifact size?
(3) is the limit artifact bytes, image layer size, build-host disk, or a pull timeout?

Related Slack (#model-serving): https://databricks.slack.com/archives/CPS3MLXUJ/p1784825239849739?thread_ts=1784552406.538799

Attachments to add from reports/serving_ceiling/eng_500gb/: events.json, build_logs.json, endpoint.json, failure_report.json
