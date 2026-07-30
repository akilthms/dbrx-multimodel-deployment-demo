"""Fast tests for the design-A storage stress harness.

Two layers:
  - pure sizing math on `ProbeArtifactPlan` (microseconds)
  - a real filesystem build test with a tiny target (milliseconds) — proves
    replication lands independent, non-shared copies that actually sum to the
    requested size on disk.

No Databricks/MLflow imports here — the deploy path needs a live workspace and
is exercised manually via the `dbrx-mmd capacity stress` command.
"""
from __future__ import annotations

import pytest

from dbrx_multimodel_registration.adapters.serving.stress import (
    ProbeArtifactPlan,
    build_probe_artifact,
    select_subset_to_size,
)


# ─── pure math ────────────────────────────────────────────────────────

def test_n_copies_rounds_up():
    # 500 MB source, 30 GB target → ceil(30e9 / 500e6) = 60
    plan = ProbeArtifactPlan(source_bytes=500_000_000, target_gb=30.0)
    assert plan.n_copies == 60
    assert plan.resulting_bytes == 60 * 500_000_000


def test_n_copies_min_one():
    # source already bigger than target → still make one copy
    plan = ProbeArtifactPlan(source_bytes=40_000_000_000, target_gb=30.0)
    assert plan.n_copies == 1


def test_rejects_empty_source():
    with pytest.raises(ValueError, match="source_bytes must be positive"):
        _ = ProbeArtifactPlan(source_bytes=0, target_gb=30.0).n_copies


# ─── real filesystem build (tiny target) ───────────────────────────────

def test_build_replicates_to_target_on_disk(tmp_path):
    """A 1 KB source replicated to a ~5 KB target must produce independent
    copies whose bytes sum to >= target (no hardlink sharing)."""
    src = tmp_path / "bundle.parquet"
    src.write_bytes(b"x" * 1024)  # 1 KB

    dest = tmp_path / "artifact"
    plan = build_probe_artifact(str(src), str(dest), target_gb=5 / 1_000_000)  # ~5 KB

    assert plan.n_copies == 5
    copies = list(dest.iterdir())
    assert len(copies) == 5
    # Independent objects: distinct inodes (no hardlink dedup).
    inodes = {c.stat().st_ino for c in copies}
    assert len(inodes) == 5
    total = sum(c.stat().st_size for c in copies)
    assert total >= plan.target_bytes


def test_build_replicates_directory_source(tmp_path):
    """Source can be a directory (like bundle_parquet) — each replica is a
    full copytree, so per-copy size == source dir size."""
    src = tmp_path / "bundle_parquet"
    src.mkdir()
    (src / "part-0.parquet").write_bytes(b"a" * 2048)
    (src / "part-1.parquet").write_bytes(b"b" * 2048)  # 4 KB dir total

    dest = tmp_path / "artifact"
    plan = build_probe_artifact(str(src), str(dest), target_gb=12 / 1_000_000)  # ~12 KB

    assert plan.source_bytes == 4096
    assert plan.n_copies == 3  # ceil(12000 / 4096)
    assert (dest / "copy_0000" / "part-0.parquet").exists()


# ─── subset selection (source larger than target) ──────────────────────

def test_subset_accumulates_whole_buckets_to_target(tmp_path):
    """From a bundle bigger than target, copy whole bucket=* dirs until the
    total meets target — and STOP (don't copy the whole thing)."""
    bundle = tmp_path / "bundle_parquet"
    bundle.mkdir()
    (bundle / "_SUCCESS").write_bytes(b"")
    # 5 buckets × 2 KB = 10 KB total; target ~5 KB → expect 3 buckets (6 KB).
    for i in range(5):
        b = bundle / f"bucket={i}"
        b.mkdir()
        (b / "part.parquet").write_bytes(b"z" * 2048)

    dest = tmp_path / "artifact"
    plan = select_subset_to_size(str(bundle), str(dest), target_gb=5 / 1_000_000)  # ~5 KB

    copied = sorted(p.name for p in dest.iterdir())
    assert copied == ["bucket=0", "bucket=1", "bucket=2"]  # stopped at >= target
    assert plan.source_bytes >= 5000
    assert plan.source_bytes == 3 * 2048


def test_subset_takes_whole_bundle_when_smaller_than_target(tmp_path):
    """If the bundle is smaller than target, take all of it (can't exceed)."""
    bundle = tmp_path / "bundle_parquet"
    bundle.mkdir()
    for i in range(2):
        b = bundle / f"bucket={i}"
        b.mkdir()
        (b / "part.parquet").write_bytes(b"z" * 1024)

    dest = tmp_path / "artifact"
    plan = select_subset_to_size(str(bundle), str(dest), target_gb=1.0)  # 1 GB, way bigger

    assert sorted(p.name for p in dest.iterdir()) == ["bucket=0", "bucket=1"]
    assert plan.source_bytes == 2048
