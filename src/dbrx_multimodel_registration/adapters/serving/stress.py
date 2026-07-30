"""Drive the design-A storage-ceiling test: build an N-GB artifact from
existing bundle files, bake it into a trivial hello-world models-from-code
pyfunc (see storage_probe_entrypoint.py), deploy to a Model Serving endpoint,
and report whether the endpoint accepted and loaded it.

Approach (per the decision to reuse existing artifacts rather than train fresh
distinct models with Spark): replicate a real bundle file to a target on-disk
size. Replication is done at the FILE level — N copies at distinct paths are N
real objects on disk (no dedup, no cross-file compression to worry about), so
the endpoint downloads a genuine N GB. This tests STORAGE capacity, which is
blob-content-independent — exactly why representative (not distinct) blobs are
fine here.

Every Databricks/MLflow import is lazy so this module loads without a live
workspace (keeps `from ...serving import build_probe_artifact` cheap and lets
the sizing helpers be unit-tested offline).
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Artifact key the baked payload is logged under. MUST match the value in
# storage_probe_entrypoint.py (which redefines it standalone so the served
# script stays self-contained at load time).
ARTIFACT_KEY = "payload"


@dataclass(frozen=True)
class ProbeArtifactPlan:
    """How to reach `target_gb` by replicating a source of `source_bytes`."""

    source_bytes: int
    target_gb: float

    @property
    def target_bytes(self) -> int:
        return int(self.target_gb * 1_000_000_000)

    @property
    def n_copies(self) -> int:
        """Copies of the source needed to meet or exceed the target."""
        from math import ceil

        if self.source_bytes <= 0:
            raise ValueError("source_bytes must be positive")
        return max(1, ceil(self.target_bytes / self.source_bytes))

    @property
    def resulting_bytes(self) -> int:
        return self.n_copies * self.source_bytes


def _dir_size(path: str) -> int:
    total = 0
    for dp, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dp, f))
    return total


def build_probe_artifact(source_path: str, dest_dir: str, target_gb: float) -> ProbeArtifactPlan:
    """Populate `dest_dir` with copies of `source_path` until it reaches
    `target_gb` on disk. `source_path` may be a file or a directory (e.g. an
    existing `bundle_parquet` dir). Returns the plan actually executed.

    Uses hardlink-free copies (`shutil.copy2` / `copytree`) so each replica is
    an independent object with its own disk footprint — a hardlink would share
    inodes and the endpoint would download less than target_gb.
    """
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"source artifact not found: {source_path}")
    source_bytes = _dir_size(source_path) if src.is_dir() else src.stat().st_size
    plan = ProbeArtifactPlan(source_bytes=source_bytes, target_gb=target_gb)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(plan.n_copies):
        if src.is_dir():
            shutil.copytree(src, dest / f"copy_{i:04d}")
        else:
            shutil.copy2(src, dest / f"copy_{i:04d}{src.suffix}")
    return plan


def select_subset_to_size(bundle_dir: str, dest_dir: str, target_gb: float) -> ProbeArtifactPlan:
    """Assemble `dest_dir` from a SUBSET of `bundle_dir`'s bucket partitions,
    accumulating whole `bucket=*` dirs until the total meets `target_gb`.

    Use when the source bundle is already LARGER than the target (e.g. a 72 GB
    region and a 30 GB test). Copies whole bucket partitions so the subset is
    itself a valid partitioned parquet dataset the probe can open. Returns a
    plan whose `resulting_bytes` is the actual assembled size (>= target, or
    the whole bundle if it's smaller than target).

    Copies (not symlinks) so the staged artifact is self-contained for
    `log_model`, which snapshots from a local path.
    """
    src = Path(bundle_dir)
    if not src.is_dir():
        raise NotADirectoryError(f"expected a bundle directory: {bundle_dir}")
    target_bytes = int(target_gb * 1_000_000_000)
    buckets = sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("bucket="))
    if not buckets:
        raise ValueError(f"no bucket=* partitions under {bundle_dir}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    assembled = 0
    n_buckets = 0
    for b in buckets:
        shutil.copytree(b, dest / b.name)
        assembled += _dir_size(str(b))
        n_buckets += 1
        if assembled >= target_bytes:
            break
    # Report via ProbeArtifactPlan shape: treat the assembled subset as the
    # "source" of size `assembled`, one copy. n_buckets travels in the log line.
    plan = ProbeArtifactPlan(source_bytes=assembled, target_gb=assembled / 1_000_000_000)
    return plan


def log_probe_model(dest_dir: str, registered_model_name: str, experiment: str) -> str:
    """Log a trivial hello-world pyfunc with `dest_dir` baked in as its artifact.

    `experiment` is REQUIRED: a python_wheel_task has no default MLflow
    experiment (unlike a notebook), so `start_run()` would fail with
    "Could not find experiment with ID None" without an explicit set. Pass a
    workspace path like `/Users/<you>/mmd_storage_probe`; it's created if absent.

    Returns the model version URI (`models:/<name>/<version>`). Baking the
    artifact is what forces the endpoint to download it to local disk — the
    design-A path whose ceiling we're probing. The MODEL is trivial on purpose;
    only the artifact size is under test.
    """
    import os as _os

    import mlflow

    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import ColSpec, Schema

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)
    # UC requires a signature with both inputs and outputs. Trivial ping→string
    # schema matching the hello-world probe (see storage_probe_entrypoint.py).
    signature = ModelSignature(
        inputs=Schema([ColSpec("long", "ping")]),
        outputs=Schema([ColSpec("string")]),
    )
    # Models-from-code: pass the SCRIPT PATH (not an object) as python_model, so
    # the serving container EXECUTES the script to rebuild the model instead of
    # unpickling + importing its defining module. Two prior failure modes this
    # avoids: (1) MLflow inferring the project wheel as a pip dep (fails to
    # install from PyPI), and (2) cloudpickle recording a class's deep module
    # path `dbrx_multimodel_registration...` and hitting ModuleNotFoundError at
    # load because we don't install the wheel. The script is fully self-contained
    # (mlflow only) and ends with set_model().
    probe_script = _os.path.join(_os.path.dirname(__file__), "storage_probe_entrypoint.py")
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="storage_probe",
            python_model=probe_script,
            artifacts={ARTIFACT_KEY: dest_dir},
            registered_model_name=registered_model_name,
            signature=signature,
            pip_requirements=["mlflow"],
        )
    from mlflow.tracking import MlflowClient

    # UC does NOT support `order_by` on search_model_versions — fetch all and
    # pick the newest by integer version locally.
    versions = MlflowClient().search_model_versions(f"name='{registered_model_name}'")
    version = max(int(v.version) for v in versions)
    return f"models:/{registered_model_name}/{version}"


def deploy_probe_endpoint(
    endpoint_name: str,
    registered_model_name: str,
    model_version: str,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> Any:
    """Create (or update) a serving endpoint hosting the probe model.

    Returns the SDK endpoint object. Poll `get_endpoint_status` for readiness;
    a failed image build / size rejection surfaces there — that's the signal.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    w = WorkspaceClient()
    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=registered_model_name,
                entity_version=model_version,
                workload_size=workload_size,
                scale_to_zero_enabled=scale_to_zero,
            )
        ]
    )
    existing = [e for e in w.serving_endpoints.list() if e.name == endpoint_name]
    if existing:
        return w.serving_endpoints.update_config(name=endpoint_name, served_entities=config.served_entities)
    return w.serving_endpoints.create(name=endpoint_name, config=config)


def get_endpoint_status(endpoint_name: str) -> dict:
    """Return a compact status dict: state + the last config-update result.

    A size rejection or image-build failure shows up as
    config_update='UPDATE_FAILED' (or the endpoint stuck NOT_READY); read the
    workspace UI / SDK error detail for the exact cause.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    ep = w.serving_endpoints.get(name=endpoint_name)
    state = ep.state
    return {
        "name": endpoint_name,
        "ready": getattr(state, "ready", None).value if getattr(state, "ready", None) else None,
        "config_update": getattr(state, "config_update", None).value
        if getattr(state, "config_update", None)
        else None,
    }
