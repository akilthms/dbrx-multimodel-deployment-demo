"""Helpers shared across adapters.

`SparkSchemaMixin` lets dataclasses describe their own Spark `StructType`.
Entities inherit the mixin (see `domains/entities.py`); adapters call
`TheEntity.spark_schema()` instead of hand-declaring StructType blocks.

`run_coro_blocking` runs an asyncio coroutine to completion from sync code,
even when an event loop is already running (Databricks notebooks own a loop,
so plain `asyncio.run` raises "cannot be called from a running event loop").
"""
from __future__ import annotations

import asyncio
import logging
import os
import statistics
import threading
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from typing import Any, Awaitable, get_args, get_origin, get_type_hints

from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


_PY_TO_SPARK: dict[type, DataType] = {
    int: IntegerType(),
    # Python `float` is always 64-bit; map to DoubleType so metric values
    # (rmse/mape/r2/etc.) round-trip cleanly. FloatType would truncate to 32-bit.
    float: DoubleType(),
    bool: BooleanType(),
    str: StringType(),
    date: DateType(),
    datetime: TimestampType(),
    bytes: BinaryType(),
}


def _unwrap_annotated(py_type: object) -> tuple[object, str | None]:
    """Unwrap `typing.Annotated[T, "comment", …]` → (T, first-string-or-None).

    `typing.Annotated[T, *meta]` lets the dataclass field author attach a
    human-readable column comment alongside the type:

        region: Annotated[str, "Sales region (NORTHEAST, …)."]

    `get_type_hints(cls, include_extras=True)` returns the full Annotated
    wrapper. We then pull (a) the underlying type T for Spark-type resolution
    and (b) the first string metadata, which we treat as the column comment.
    If a field isn't annotated, returns (py_type, None) — no comment.
    """
    if hasattr(py_type, "__metadata__"):
        args = get_args(py_type)
        if args:
            underlying = args[0]
            comment = next((m for m in args[1:] if isinstance(m, str)), None)
            return underlying, comment
    return py_type, None


def _resolve_spark_type(py_type: object, field_label: str) -> DataType:
    """Map a Python type annotation to a Spark `DataType`.

    Handles primitives via `_PY_TO_SPARK` and `dict[K, V]` via `MapType`.
    Anything else (Optional/Union/list/tuple/nested dataclasses) raises
    `TypeError` with `field_label` so callers can locate the offending field.

    `typing.Annotated[T, …]` wrappers are unwrapped first — the metadata
    is the caller's responsibility (used elsewhere for column comments).
    """
    # Unwrap Annotated[T, …] first so primitive lookup sees the underlying T.
    py_type, _ = _unwrap_annotated(py_type)

    primitive = _PY_TO_SPARK.get(py_type)  # type: ignore[arg-type]
    if primitive is not None:
        return primitive

    origin = get_origin(py_type)
    if origin is dict:
        key_t, val_t = get_args(py_type)
        # MapType key/value types come from the same primitives table — recurse
        # so `dict[str, float]` lands at MapType(StringType, DoubleType).
        key_spark = _resolve_spark_type(key_t, f"{field_label}.<key>")
        val_spark = _resolve_spark_type(val_t, f"{field_label}.<value>")
        return MapType(key_spark, val_spark, valueContainsNull=False)

    raise TypeError(
        f"Cannot map field {field_label}: {py_type!r} to a Spark type. "
        f"Supported primitives: {sorted(t.__name__ for t in _PY_TO_SPARK)}; "
        f"plus dict[K, V] for MapType. Add Optional/list/nested-dataclass support "
        f"to utils.helpers when an entity needs it."
    )


def struct_type_from_dataclass(cls: type) -> StructType:
    """Derive a Spark StructType from a Python dataclass.

    Walks `dataclasses.fields(cls)` in declaration order — includes `init=False`
    fields, since they still appear in instantiated rows. Field type annotations
    are resolved via `typing.get_type_hints(cls, include_extras=True)` so string
    annotations from `from __future__ import annotations` work AND so
    `typing.Annotated[T, "comment"]` wrappers survive. The first string in the
    Annotated metadata becomes the StructField's `comment` metadata, which the
    UC table writer can propagate to Delta column comments.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    hints = get_type_hints(cls, include_extras=True)
    struct_fields: list[StructField] = []
    for f in fields(cls):
        py_type = hints.get(f.name, f.type)
        _, comment = _unwrap_annotated(py_type)
        spark_type = _resolve_spark_type(py_type, f"{cls.__name__}.{f.name}")
        metadata = {"comment": comment} if comment else {}
        struct_fields.append(
            StructField(f.name, spark_type, nullable=False, metadata=metadata)
        )
    return StructType(struct_fields)


def column_comments_from_dataclass(cls: type) -> dict[str, str]:
    """Return {field_name: comment_string} for every field whose type is
    `typing.Annotated[..., "comment"]`. Fields without an Annotated string
    metadata are omitted.

    Used by the UC table writer to issue `ALTER TABLE … ALTER COLUMN …
    COMMENT '…'` statements after table creation, since not every Spark
    write path propagates StructField metadata into the Delta column
    catalog.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = get_type_hints(cls, include_extras=True)
    out: dict[str, str] = {}
    for f in fields(cls):
        _, comment = _unwrap_annotated(hints.get(f.name, f.type))
        if comment:
            out[f.name] = comment
    return out


def run_coro_blocking(coro: Awaitable[Any]) -> Any:
    """Run an awaitable to completion from sync code, robust to a parent loop.

    Why this exists: Databricks notebooks already run on top of an asyncio loop.
    Calling `asyncio.run(coro)` from a notebook cell raises:
        RuntimeError: asyncio.run() cannot be called from a running event loop

    To work in both notebook and plain-script (main.py) contexts, we:
      - detect whether there's a running loop in this thread
      - if not: just `asyncio.run(coro)` directly (cheap, no thread)
      - if yes: hand the coroutine to a worker thread that owns its own loop,
        and block the caller on `Thread.join`. The notebook's outer loop stays
        untouched; our coroutine runs in full isolation.

    Exceptions from the coroutine are re-raised on the caller's thread, so
    `try/except` around `run_coro_blocking(...)` works the same as around
    `asyncio.run(...)`.
    """
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(coro)  # type: ignore[arg-type]

    result: list[Any] = [None]
    err: list[BaseException | None] = [None]

    def _target() -> None:
        try:
            result[0] = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as e:  # noqa: BLE001
            err[0] = e

    t = threading.Thread(target=_target, name="dbrx-multimodel-asyncio", daemon=True)
    t.start()
    t.join()
    if err[0] is not None:
        raise err[0]
    return result[0]


_rpc_log = logging.getLogger("dbrx_multimodel_registration.rpc_timings")


class RpcTimings:
    """Thread-safe accumulator of per-SKU MLflow RPC durations.

    Records (create_run, log_batch, set_terminated) seconds per logged SKU.
    Emits a running-stats log line every `sample_step` records and a final
    per-region summary on `final_summary()`.
    """

    def __init__(self, region: str, sample_step: int = 100) -> None:
        self._region = region
        self._sample_step = sample_step
        self._lock = threading.Lock()
        self._create: list[float] = []
        self._batch: list[float] = []
        self._terminate: list[float] = []
        self._next_log_at = sample_step

    def record(self, create_s: float, batch_s: float, terminate_s: float) -> None:
        with self._lock:
            self._create.append(create_s)
            self._batch.append(batch_s)
            self._terminate.append(terminate_s)
            if len(self._create) >= self._next_log_at:
                self._emit_window_log(len(self._create) - self._sample_step, len(self._create))
                self._next_log_at += self._sample_step

    def _emit_window_log(self, lo: int, hi: int) -> None:
        c = self._create[lo:hi]
        b = self._batch[lo:hi]
        t = self._terminate[lo:hi]
        _rpc_log.info(
            "[%s] rpc_timings n=%d-%d create_run p50=%.0fms p95=%.0fms | "
            "log_batch p50=%.0fms p95=%.0fms | set_terminated p50=%.0fms p95=%.0fms",
            self._region, lo + 1, hi,
            _p(c, 0.50) * 1000, _p(c, 0.95) * 1000,
            _p(b, 0.50) * 1000, _p(b, 0.95) * 1000,
            _p(t, 0.50) * 1000, _p(t, 0.95) * 1000,
        )

    def final_summary(self) -> None:
        with self._lock:
            n = len(self._create)
            if n == 0:
                return
            _rpc_log.info(
                "[%s] FINAL rpc_timings n=%d create_run mean=%.0fms p95=%.0fms | "
                "log_batch mean=%.0fms p95=%.0fms | set_terminated mean=%.0fms p95=%.0fms | "
                "total per-SKU mean=%.0fms",
                self._region, n,
                statistics.mean(self._create) * 1000, _p(self._create, 0.95) * 1000,
                statistics.mean(self._batch) * 1000, _p(self._batch, 0.95) * 1000,
                statistics.mean(self._terminate) * 1000, _p(self._terminate, 0.95) * 1000,
                (statistics.mean(self._create) + statistics.mean(self._batch)
                 + statistics.mean(self._terminate)) * 1000,
            )


def _p(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


def configure_mlflow_http_pool(n_regions: int, concurrency_per_region: int, headroom: int = 16) -> None:
    """Size urllib3's connection pool to the actual concurrent-thread count.

    Total active driver threads during stage 5 = n_regions × concurrency_per_region.
    Both POOL_CONNECTIONS (host count) and POOL_MAXSIZE (per-host cap) should
    equal that number plus headroom so threads don't queue on socket allocation.

    urllib3 reads the env var lazily (per-host LRU-cached session), so this
    must run BEFORE the first MlflowClient RPC — but it's fine to run AFTER
    `import mlflow`.
    """
    pool_size = (n_regions * concurrency_per_region) + headroom
    os.environ["MLFLOW_HTTP_POOL_CONNECTIONS"] = str(pool_size)
    os.environ["MLFLOW_HTTP_POOL_MAXSIZE"] = str(pool_size)
    logging.getLogger(__name__).info(
        "MLFLOW_HTTP_POOL_CONNECTIONS / MAXSIZE = %d (n_regions=%d × concurrency=%d + %d headroom)",
        pool_size, n_regions, concurrency_per_region, headroom,
    )


class SparkSchemaMixin:
    """Mixin: a dataclass that knows its own Spark `StructType`.

    Subclasses get `spark_schema()` for free. The schema is computed lazily on
    each call, so dataclasses with field types the helper can't map (e.g.
    `list[str]`) only fail when the method is actually called — inheriting the
    mixin is free.
    """

    @classmethod
    def spark_schema(cls) -> StructType:
        return struct_type_from_dataclass(cls)
