"""Adapter: writes the per-region model bundle as ONE parquet file landed
directly in the parent run's artifact directory (no tempdir hop).

Replaces the inline pattern that was duplicated across all three logging
strategies: open ParquetWriter in a tempdir → flush rows every N → call
`client.log_artifact(...)` to upload. By resolving the parent run's
artifact_uri up front and converting the `dbfs:/Volumes/...` MLflow URI to
its FUSE mount (`/dbfs/Volumes/...`), the ParquetWriter writes straight to
final storage — one I/O pass instead of two.

Schema + row-group constant live here (single source of truth). The
strategy files no longer carry pyarrow knowledge.
"""
from __future__ import annotations

import errno
import os
from contextlib import contextmanager
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from mlflow.tracking import MlflowClient

from dbrx_multimodel_registration.domains.entities import RegionSpec, TrainedModelRecord
from dbrx_multimodel_registration.ports.storage.parent_run_artifact_writer import (
    ParentRunArtifactBundle,
)


# Mirrors TrainedModelRecord, flattened for parquet. Model metrics that are
# common to the regression strategies (rmse/mape/r2) are pulled to top-level
# columns for cheap analytical scans; everything else stays embedded inside
# the model_blob bytes.
_BUNDLE_SCHEMA = pa.schema(
    [
        ("sku", pa.string()),
        ("model_name", pa.string()),
        ("model_blob", pa.binary()),
        ("rmse", pa.float64()),
        ("mape", pa.float64()),
        ("r2", pa.float64()),
    ]
)
# 200 rows × ~5MB blob ≈ 1GB per flush — comfortable on the i3.4xlarge driver.
_ROW_GROUP = 200

_BUNDLE_SUBPATH = "models_bundle/models.parquet"


def _to_fuse_path(artifact_uri: str) -> str:
    """Convert an MLflow artifact_uri to a directly-writable local FUSE path.

    On Databricks:
      - `dbfs:/Volumes/...`  →  `/Volumes/...`  (UC Volume FUSE mount)
      - `dbfs:/...`          →  `/dbfs/...`     (legacy DBFS FUSE mount)

    Other schemes (s3://, gs://, file://) aren't supported here on purpose —
    this adapter is direct-write or bust. If a different scheme shows up,
    fail loudly so the user picks (or writes) a different adapter.
    """
    if artifact_uri.startswith("dbfs:/Volumes/"):
        return "/" + artifact_uri[len("dbfs:/"):]
    if artifact_uri.startswith("dbfs:/"):
        return "/dbfs/" + artifact_uri[len("dbfs:/"):]
    if artifact_uri.startswith("/"):
        return artifact_uri
    raise NotImplementedError(
        f"ParquetBundleArtifactWriter only supports FUSE-writable artifact URIs "
        f"(dbfs:/Volumes/..., dbfs:/..., or absolute paths). Got: {artifact_uri!r}. "
        f"Implement a different ParentRunArtifactWriterPort adapter for this scheme."
    )


class _ParquetParentRunBundle:
    """Concrete bundle returned by ParquetBundleArtifactWriter.open()."""

    def __init__(self, fuse_path: str, artifact_uri: str) -> None:
        self._fuse_path = fuse_path
        self._artifact_uri = artifact_uri
        # WSFS quirk on UC volumes: os.makedirs walks up to the volume root and
        # FUSE returns errno 95 (EOPNOTSUPP) instead of EEXIST. Suppress both.
        try:
            os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        except OSError as e:
            if e.errno not in (errno.EEXIST, errno.EOPNOTSUPP):
                raise
        self._writer = pq.ParquetWriter(fuse_path, _BUNDLE_SCHEMA, compression="snappy")
        self._buffer: list[dict] = []

    def write(self, records: Sequence[TrainedModelRecord]) -> None:
        for r in records:
            self._buffer.append(
                {
                    "sku": r.sku,
                    "model_name": r.model_name,
                    "model_blob": r.model_blob_bytes,
                    "rmse": float(r.metrics.get("rmse", 0.0)),
                    "mape": float(r.metrics.get("mape", 0.0)),
                    "r2": float(r.metrics.get("r2", 0.0)),
                }
            )
        if len(self._buffer) >= _ROW_GROUP:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._writer.write_table(pa.Table.from_pylist(self._buffer, schema=_BUNDLE_SCHEMA))
        self._buffer.clear()

    def close(self) -> None:
        self._flush()
        self._writer.close()

    @property
    def artifact_uri(self) -> str:
        return self._artifact_uri


class ParquetBundleArtifactWriter:
    """`ParentRunArtifactWriterPort` — one parquet file per parent run, direct-write."""

    name = "parquet_bundle"

    @contextmanager
    def open(
        self,
        client: MlflowClient,
        parent_run_id: str,
        region: RegionSpec,
    ) -> Iterator[ParentRunArtifactBundle]:
        run = client.get_run(parent_run_id)
        run_artifact_uri = run.info.artifact_uri  # e.g. dbfs:/Volumes/.../<run_id>/artifacts
        fuse_path = os.path.join(_to_fuse_path(run_artifact_uri), _BUNDLE_SUBPATH)
        final_artifact_uri = run_artifact_uri.rstrip("/") + "/" + _BUNDLE_SUBPATH

        bundle = _ParquetParentRunBundle(fuse_path=fuse_path, artifact_uri=final_artifact_uri)
        try:
            yield bundle
        finally:
            bundle.close()
