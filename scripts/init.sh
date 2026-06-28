#!/usr/bin/env bash
set -euo pipefail

# ─── MLflow logging best practices (Albertsons + asyncio refactor) ──────────
# Must be set BEFORE any process imports mlflow. Routes artifact uploads
# through the Python HTTP path with thread-safe creds, instead of mlflowdbfs
# FUSE (which 401s under concurrency).
export DISABLE_MLFLOWDBFS=true
# urllib3 connection pool — must be ≥ concurrency to avoid socket starvation
# on the driver. 32 = 2× the default 16-thread concurrency for headroom.
export MLFLOW_HTTP_POOL_CONNECTIONS="${MLFLOW_HTTP_POOL_CONNECTIONS:-32}"
export MLFLOW_HTTP_POOL_MAXSIZE="${MLFLOW_HTTP_POOL_MAXSIZE:-32}"
# Retries + exponential backoff (2/4/8/16s) absorb 429/5xx without crashing.
export MLFLOW_HTTP_REQUEST_MAX_RETRIES="${MLFLOW_HTTP_REQUEST_MAX_RETRIES:-9}"
export MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR="${MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR:-2}"
export MLFLOW_HTTP_REQUEST_TIMEOUT="${MLFLOW_HTTP_REQUEST_TIMEOUT:-120}"
# Disable MLflow's built-in trace logging (we have our own asyncio orchestration).
export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING="${MLFLOW_ENABLE_ASYNC_TRACE_LOGGING:-false}"

SKILLS_DIR=".claude/skills"

# ─── MLflow skills (from mlflow/skills via raw URL) ─────────────────────────
# Single SKILL.md (+ optional ref/example) per skill — small enough to fetch
# directly without cloning anything.
MLFLOW_RAW_URL="https://raw.githubusercontent.com/mlflow/skills/main"
MLFLOW_SKILLS=(
  agent-evaluation
  analyze-mlflow-chat-session
  analyze-mlflow-trace
  instrumenting-with-mlflow-tracing
  mlflow-onboarding
  querying-mlflow-metrics
  retrieving-mlflow-traces
  searching-mlflow-docs
)

# ─── Databricks skills (from databricks-solutions/ai-dev-kit via git) ───────
# Multi-file structure (SKILL.md + reference docs + examples/), so we
# sparse-checkout only the directories we want into a temp clone, then copy
# them in. No repo/ left in the project tree afterward.
AI_DEV_KIT_REPO="https://github.com/databricks-solutions/ai-dev-kit.git"
DATABRICKS_SKILLS=(
  databricks-python-sdk
  databricks-bundles
)

# ─── Legacy cleanup ─────────────────────────────────────────────────────────
# Older installer versions cloned the full ai-dev-kit into ./repo/. Drop it
# if it's still around so the project tree stays clean.
if [ -d ./repo/.git ] && grep -q 'databricks-solutions/ai-dev-kit' ./repo/.git/config 2>/dev/null; then
  echo "Removing legacy ./repo/ ai-dev-kit checkout..."
  rm -rf ./repo
fi

mkdir -p "$SKILLS_DIR"

# ─── Install MLflow skills ──────────────────────────────────────────────────
echo "Installing MLflow skills → $SKILLS_DIR/"
for skill in "${MLFLOW_SKILLS[@]}"; do
  dest="$SKILLS_DIR/$skill"
  mkdir -p "$dest"
  if curl -fsSL "$MLFLOW_RAW_URL/$skill/SKILL.md" -o "$dest/SKILL.md"; then
    # Optional reference files — silently skip if absent upstream.
    # Mirrors install.sh:1248-1250 in the ai-dev-kit installer.
    for ref in reference.md examples.md api.md; do
      curl -fsSL "$MLFLOW_RAW_URL/$skill/$ref" -o "$dest/$ref" 2>/dev/null || true
    done
    echo "  ✓ $skill"
  else
    rm -rf "$dest"
    echo "  ✗ $skill (not found upstream — skipped)" >&2
  fi
done

# ─── Install Databricks skills ──────────────────────────────────────────────
echo "Installing Databricks skills → $SKILLS_DIR/"
tmp_kit="$(mktemp -d -t ai-dev-kit.XXXXXX)"
trap 'rm -rf "$tmp_kit"' EXIT

git -c advice.detachedHead=false clone -q --depth 1 --filter=blob:none --no-checkout --no-tags \
  "$AI_DEV_KIT_REPO" "$tmp_kit"

sparse_paths=()
for skill in "${DATABRICKS_SKILLS[@]}"; do
  sparse_paths+=("databricks-skills/$skill")
done

(
  cd "$tmp_kit"
  git sparse-checkout init --cone
  git sparse-checkout set "${sparse_paths[@]}"
  git checkout -q
)

for skill in "${DATABRICKS_SKILLS[@]}"; do
  src="$tmp_kit/databricks-skills/$skill"
  dest="$SKILLS_DIR/$skill"
  if [ -d "$src" ]; then
    rm -rf "$dest"
    cp -r "$src" "$dest"
    echo "  ✓ $skill"
  else
    echo "  ✗ $skill (not found in ai-dev-kit) — skipped" >&2
  fi
done

# tmp_kit cleaned up by EXIT trap

# ─── Install the project package ────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
    uv pip install -e .
else
    pip install -e .
fi
