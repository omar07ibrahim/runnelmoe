#!/usr/bin/env python3
"""Run the reproducible M2 verified-data-plane evidence experiment.

The harness intentionally has no package dependencies. It executes only the
already-built release CLI and writes a new, append-only experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HARNESS_VERSION = "runnel.m2-evidence/2"
BINARY_RELATIVE = "target/release/runnel"
OUTPUT_BASE_RELATIVE = "benchmarks/raw"
COMMAND = (BINARY_RELATIVE, "data-plane-demo", "--json")
CONTROLLED_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
BUILD_ENVIRONMENT = {
    **CONTROLLED_ENVIRONMENT,
    "CARGO_BUILD_JOBS": "2",
    "CARGO_INCREMENTAL": "0",
    "CARGO_TERM_COLOR": "never",
}
BUILD_COMMAND = (
    "cargo",
    "build",
    "--release",
    "--locked",
    "--offline",
    "-p",
    "runnel",
)
DEFAULT_WARMUPS = 3
DEFAULT_REPETITIONS = 30
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_BOOTSTRAP_SEED = 20_260_803
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MIN_BUILD_FREE_BYTES = 2 * 1024 * 1024 * 1024
BUILD_TIMEOUT_SECONDS = 15 * 60
EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,94}[a-z0-9]$|^[a-z0-9]$")


class EvidenceError(RuntimeError):
    """A user-actionable experiment contract violation."""


@dataclass(frozen=True)
class BoundedProcessResult:
    return_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    output_exceeded: bool
    launch_error: bool


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_bounded_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    maximum_capture_bytes: int = MAX_CAPTURE_BYTES,
) -> BoundedProcessResult:
    """Run a process with concurrent, hard-bounded stdout/stderr capture."""

    if maximum_capture_bytes < 1:
        raise EvidenceError("the subprocess capture limit must be positive")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        return BoundedProcessResult(None, b"", b"", False, False, True)

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    output_exceeded = threading.Event()

    def drain(stream: Any, name: str) -> None:
        try:
            try:
                while chunk := stream.read(64 * 1024):
                    remaining = maximum_capture_bytes + 1 - len(buffers[name])
                    if remaining > 0:
                        buffers[name].extend(chunk[:remaining])
                    if len(chunk) > remaining or len(buffers[name]) > maximum_capture_bytes:
                        output_exceeded.set()
                        _kill_process_group(process)
                        break
            except OSError:
                pass
        finally:
            stream.close()

    assert process.stdout is not None
    assert process.stderr is not None
    readers = [
        threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
        return_code = process.wait()
    for reader in readers:
        reader.join(timeout=5)
    if any(reader.is_alive() for reader in readers):
        _kill_process_group(process)
        process.stdout.close()
        process.stderr.close()
        for reader in readers:
            reader.join(timeout=5)
        if any(reader.is_alive() for reader in readers):
            raise EvidenceError("subprocess output readers did not terminate")
    return BoundedProcessResult(
        return_code=return_code,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
        timed_out=timed_out,
        output_exceeded=output_exceeded.is_set(),
        launch_error=False,
    )


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp without local host information."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for content-addressing."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return f"sha256:{sha256_bytes(canonical_json(value))}"


def validate_experiment_id(value: str) -> str:
    """Validate a single portable path component used as an experiment ID."""

    if not isinstance(value, str) or not EXPERIMENT_ID.fullmatch(value):
        raise EvidenceError(
            "experiment ID must be 1-96 lowercase ASCII letters, digits, dots, "
            "underscores, or hyphens, and must start and end alphanumerically"
        )
    if value in {".", ".."} or ".." in value:
        raise EvidenceError("experiment ID may not contain a traversal component")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise EvidenceError("experiment ID may not be absolute")
    if len(PurePosixPath(value).parts) != 1 or len(PureWindowsPath(value).parts) != 1:
        raise EvidenceError("experiment ID must be one path component")
    return value


def validate_repository_relative_path(value: str) -> PurePosixPath:
    """Reject absolute, Windows, empty, and traversal-bearing paths."""

    if not isinstance(value, str) or not value:
        raise EvidenceError("repository-relative path must be non-empty")
    if "\\" in value:
        raise EvidenceError("repository-relative paths must use forward slashes")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise EvidenceError("path traversal and empty path components are forbidden")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise EvidenceError("absolute paths are forbidden")
    return posix


def resolve_repository_path(value: str) -> Path:
    """Resolve a validated path and prove that it remains under the repository."""

    relative = validate_repository_relative_path(value)
    root = ROOT.resolve(strict=True)
    candidate = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise EvidenceError("repository-relative path escapes the repository") from error
    return candidate


def ensure_new_output_destination(experiment_id: str) -> Path:
    validate_experiment_id(experiment_id)
    base = resolve_repository_path(OUTPUT_BASE_RELATIVE)
    output = (base / experiment_id).resolve(strict=False)
    root = ROOT.resolve(strict=True)
    try:
        output.relative_to(root)
    except ValueError as error:
        raise EvidenceError("experiment output escapes the repository") from error
    if output.exists() or output.is_symlink():
        raise EvidenceError(
            f"experiment output already exists: {OUTPUT_BASE_RELATIVE}/{experiment_id}"
        )
    return output


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise EvidenceError(f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def require_clean_worktree() -> str:
    """Return HEAD after proving tracked and untracked repository state is clean."""

    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise EvidenceError("git worktree must be clean before an experiment starts")
    commit = git_output("rev-parse", "--verify", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise EvidenceError("HEAD did not resolve to a full Git commit")
    return commit


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"data-plane output field {label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(str(key) for key in actual_keys - expected_keys)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise EvidenceError(f"data-plane output field {label} has " + "; ".join(details))


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvidenceError(f"data-plane output field {label} must be a nonnegative integer")
    return value


def validate_observability(demo: Mapping[str, Any]) -> None:
    """Require every volatile M2 observation without pinning its value."""

    metrics = _mapping(demo.get("metrics"), "metrics")
    _nonnegative_integer(metrics.get("wait_nanoseconds"), "metrics.wait_nanoseconds")
    _nonnegative_integer(metrics.get("io_nanoseconds"), "metrics.io_nanoseconds")
    rss = _mapping(metrics.get("observed_rss"), "metrics.observed_rss")
    _require_exact_keys(rss, ("resident_bytes", "peak_bytes"), "metrics.observed_rss")
    _nonnegative_integer(
        rss.get("resident_bytes"), "metrics.observed_rss.resident_bytes"
    )
    _nonnegative_integer(rss.get("peak_bytes"), "metrics.observed_rss.peak_bytes")


def project_correctness(demo: Mapping[str, Any]) -> dict[str, Any]:
    """Project deterministic correctness/accounting fields from one CLI result.

    Wall-clock I/O counters and observed RSS are deliberately excluded. Every
    other public M2 behavior field is required and becomes exact-match evidence.
    """

    validate_observability(demo)
    metrics = _mapping(demo.get("metrics"), "metrics")
    stable_metric_names = (
        "demand_bytes",
        "physical_read_bytes",
        "hits",
        "misses",
        "admissions",
        "evictions",
        "coalesced_demands",
        "prefetch",
        "accounted",
        "trace_events_dropped",
    )
    volatile_metric_names = ("wait_nanoseconds", "io_nanoseconds", "observed_rss")
    _require_exact_keys(
        metrics,
        stable_metric_names + volatile_metric_names,
        "metrics",
    )
    top_level_names = (
        "schema_version",
        "fixtures",
        "prompt",
        "max_new_tokens",
        "generated_ids",
        "generated_text",
        "parity",
        "generation_cache",
        "forced_eviction_generation",
        "trace",
        "cache_capacity_bytes",
    )
    _require_exact_keys(demo, top_level_names + ("metrics",), "root")
    projection = {name: _copy_json(demo[name]) for name in top_level_names}
    projection["metrics"] = {
        name: _copy_json(metrics[name]) for name in stable_metric_names
    }
    return projection


def validate_correctness_projection(projection: Mapping[str, Any]) -> None:
    """Enforce the fixed deterministic M2 parity and eviction gate."""

    def require_exact_integer(
        container: Mapping[str, Any], name: str, expected: int, label: str
    ) -> int:
        value = _nonnegative_integer(container.get(name), f"{label}.{name}")
        if value != expected:
            raise EvidenceError(f"unexpected {label} field {name}")
        return value

    def require_exact_integer_list(
        value: Any, expected: Sequence[int], label: str
    ) -> list[int]:
        if not isinstance(value, list) or len(value) != len(expected):
            raise EvidenceError(f"unexpected {label}")
        checked = []
        for index, expected_value in enumerate(expected):
            actual = _nonnegative_integer(value[index], f"{label}[{index}]")
            if actual != expected_value:
                raise EvidenceError(f"unexpected {label}")
            checked.append(actual)
        return checked

    _require_exact_keys(
        projection,
        (
            "schema_version",
            "fixtures",
            "prompt",
            "max_new_tokens",
            "generated_ids",
            "generated_text",
            "parity",
            "generation_cache",
            "forced_eviction_generation",
            "trace",
            "cache_capacity_bytes",
            "metrics",
        ),
        "correctness projection",
    )
    require_exact_integer(projection, "schema_version", 2, "root")
    if projection.get("parity") is not True:
        raise EvidenceError("data-plane numerical/token parity failed")
    if not isinstance(projection.get("prompt"), str) or projection["prompt"] != "moe":
        raise EvidenceError("data-plane demo used an unexpected prompt")
    require_exact_integer(projection, "max_new_tokens", 4, "root")
    generated_ids = projection.get("generated_ids")
    if not isinstance(generated_ids, list) or len(generated_ids) != 4:
        raise EvidenceError("data-plane generated IDs differ from the golden fixture")
    for index, expected in enumerate((15, 11, 20, 9)):
        value = _nonnegative_integer(generated_ids[index], f"generated_ids[{index}]")
        if value != expected:
            raise EvidenceError("data-plane generated IDs differ from the golden fixture")
    if (
        not isinstance(projection.get("generated_text"), str)
        or projection["generated_text"] != "njsh"
    ):
        raise EvidenceError("data-plane generated text differs from the golden fixture")

    fixtures = _mapping(projection.get("fixtures"), "fixtures")
    _require_exact_keys(fixtures, ("tiny", "multi_page"), "fixtures")
    tiny = _mapping(fixtures.get("tiny"), "fixtures.tiny")
    multi_page = _mapping(fixtures.get("multi_page"), "fixtures.multi_page")
    fixture_fields = (
        "artifact_id",
        "object_digest",
        "object_length",
        "page_table_digest",
        "page_table_length",
    )
    expected_fixtures = {
        "tiny": {
            "artifact_id": "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3",
            "object_digest": "sha256:6b2b8a1bbb2854084b1e1fe1e5787a9cfdb021b397e774fc7dbef79ac9d24bf6",
            "object_length": 7_904,
            "page_table_digest": "sha256:29383b56a150f9e5705f3666ca7707f21bc3fbd663ffd7249f4ecb938da6a62d",
            "page_table_length": 96,
        },
        "multi_page": {
            "artifact_id": "sha256:15feda585327bf8e25c124692de1b2e101315185e427b811a07b8eaca54e1630",
            "object_digest": "sha256:9bb8fcc8e9d6f6ca3512bcb6daf6f89b6f0134874c9803f8782b100b39c854ae",
            "object_length": 131_089,
            "page_table_digest": "sha256:d748ae45084baf6759c34321fb1f110151dfb33be390f4118d8a196cf0b2a4ab",
            "page_table_length": 160,
        },
    }
    for label, fixture in (("tiny", tiny), ("multi_page", multi_page)):
        _require_exact_keys(fixture, fixture_fields, f"fixtures.{label}")
        for name, expected in expected_fixtures[label].items():
            if isinstance(expected, int):
                require_exact_integer(fixture, name, expected, f"fixtures.{label}")
            elif not isinstance(fixture.get(name), str) or fixture[name] != expected:
                raise EvidenceError(f"unexpected pinned fixture field {label}.{name}")

    generation = _mapping(projection.get("generation_cache"), "generation_cache")
    _require_exact_keys(
        generation,
        ("completed_generated_tokens", "physical_read_bytes", "bytes_per_generated_token"),
        "generation_cache",
    )
    ratio = _mapping(
        generation.get("bytes_per_generated_token"),
        "generation_cache.bytes_per_generated_token",
    )
    _require_exact_keys(
        ratio,
        ("numerator_bytes", "denominator_tokens"),
        "generation_cache.bytes_per_generated_token",
    )
    expected_generation = {
        "completed_generated_tokens": 4,
        "physical_read_bytes": 7_904,
    }
    for name, expected in expected_generation.items():
        require_exact_integer(generation, name, expected, "generation_cache")
    require_exact_integer(
        ratio, "numerator_bytes", 7_904, "generation_cache.bytes_per_generated_token"
    )
    require_exact_integer(
        ratio, "denominator_tokens", 4, "generation_cache.bytes_per_generated_token"
    )

    forced = _mapping(
        projection.get("forced_eviction_generation"),
        "forced_eviction_generation",
    )
    _require_exact_keys(
        forced,
        (
            "full_generation_parity",
            "cache_capacity_bytes",
            "tensor_count",
            "tensor_page_accesses",
            "interference_page_accesses",
            "tensor_page",
            "interference_page",
            "metrics",
            "trace",
        ),
        "forced_eviction_generation",
    )
    if forced.get("full_generation_parity") is not True:
        raise EvidenceError("full-generation parity under forced eviction failed")
    expected_forced_fields = {
        "cache_capacity_bytes": 65_536,
        "tensor_count": 22,
        "tensor_page_accesses": 22,
        "interference_page_accesses": 1,
    }
    for name, expected in expected_forced_fields.items():
        require_exact_integer(forced, name, expected, "forced_eviction_generation")
    if forced["cache_capacity_bytes"] != projection.get("cache_capacity_bytes"):
        raise EvidenceError("forced and fixed-trace cache capacities differ")
    tensor_page = _mapping(
        forced.get("tensor_page"),
        "forced_eviction_generation.tensor_page",
    )
    _require_exact_keys(
        tensor_page,
        ("object_digest", "page_size", "page_index", "logical_bytes"),
        "forced_eviction_generation.tensor_page",
    )
    expected_tensor_page = {
        "object_digest": tiny["object_digest"],
        "page_size": 65_536,
        "page_index": 0,
        "logical_bytes": 7_904,
    }
    for name, expected in expected_tensor_page.items():
        if isinstance(expected, int):
            require_exact_integer(
                tensor_page,
                name,
                expected,
                "forced_eviction_generation.tensor_page",
            )
        elif tensor_page.get(name) != expected:
            raise EvidenceError(f"unexpected forced-eviction tensor-page field {name}")
    interference = _mapping(
        forced.get("interference_page"),
        "forced_eviction_generation.interference_page",
    )
    _require_exact_keys(
        interference,
        ("object_digest", "page_size", "page_index", "logical_bytes"),
        "forced_eviction_generation.interference_page",
    )
    expected_interference = {
        "object_digest": multi_page["object_digest"],
        "page_size": 65_536,
        "page_index": 0,
        "logical_bytes": 65_536,
    }
    for name, expected in expected_interference.items():
        if isinstance(expected, int):
            require_exact_integer(
                interference,
                name,
                expected,
                "forced_eviction_generation.interference_page",
            )
        elif interference.get(name) != expected:
            raise EvidenceError(f"unexpected forced-eviction interference field {name}")
    forced_metrics = _mapping(
        forced.get("metrics"),
        "forced_eviction_generation.metrics",
    )
    _require_exact_keys(
        forced_metrics,
        (
            "demand_bytes",
            "physical_read_bytes",
            "hits",
            "misses",
            "admissions",
            "evictions",
            "coalesced_demands",
            "prefetch",
            "accounted",
            "trace_events_dropped",
        ),
        "forced_eviction_generation.metrics",
    )
    expected_forced_metrics = {
        "demand_bytes": 239_424,
        "physical_read_bytes": 81_344,
        "hits": 20,
        "misses": 3,
        "admissions": 3,
        "evictions": 2,
        "coalesced_demands": 0,
        "trace_events_dropped": 0,
    }
    for name, expected in expected_forced_metrics.items():
        require_exact_integer(
            forced_metrics,
            name,
            expected,
            "forced_eviction_generation.metrics",
        )
    expected_demand_bytes = (
        forced["tensor_page_accesses"] * tensor_page["logical_bytes"]
        + forced["interference_page_accesses"] * interference["logical_bytes"]
    )
    if forced_metrics["demand_bytes"] != expected_demand_bytes:
        raise EvidenceError("forced-eviction demand-byte equation does not balance")
    if forced_metrics["misses"] != forced_metrics["admissions"]:
        raise EvidenceError("forced-eviction miss/admission counts differ")
    if forced_metrics["evictions"] != forced_metrics["admissions"] - 1:
        raise EvidenceError("forced-eviction eviction/admission counts are inconsistent")
    if (
        forced_metrics["hits"] + forced_metrics["misses"]
        != forced["tensor_page_accesses"] + forced["interference_page_accesses"]
    ):
        raise EvidenceError("forced-eviction demand outcome counts do not balance")
    forced_prefetch = _mapping(
        forced_metrics.get("prefetch"),
        "forced_eviction_generation.metrics.prefetch",
    )
    _require_exact_keys(
        forced_prefetch,
        ("bytes", "coalesced", "late", "useful", "wasted", "redundant", "dropped"),
        "forced_eviction_generation.metrics.prefetch",
    )
    for name in (
        "bytes",
        "coalesced",
        "late",
        "useful",
        "wasted",
        "redundant",
        "dropped",
    ):
        require_exact_integer(
            forced_prefetch,
            name,
            0,
            "forced_eviction_generation.metrics.prefetch",
        )
    forced_accounted = _mapping(
        forced_metrics.get("accounted"),
        "forced_eviction_generation.metrics.accounted",
    )
    _require_exact_keys(
        forced_accounted,
        (
            "active_loads",
            "page_pool_bytes",
            "inflight_bytes",
            "resident_bytes",
            "retiring_bytes",
            "leases",
        ),
        "forced_eviction_generation.metrics.accounted",
    )
    expected_forced_accounted = {
        "active_loads": 0,
        "page_pool_bytes": 7_936,
        "inflight_bytes": 0,
        "resident_bytes": 7_936,
        "retiring_bytes": 0,
        "leases": 0,
    }
    for name, expected in expected_forced_accounted.items():
        require_exact_integer(
            forced_accounted,
            name,
            expected,
            "forced_eviction_generation.metrics.accounted",
        )

    forced_trace = _mapping(
        forced.get("trace"),
        "forced_eviction_generation.trace",
    )
    _require_exact_keys(
        forced_trace,
        ("access", "event_count", "outcomes", "events"),
        "forced_eviction_generation.trace",
    )
    if forced_trace.get("access") != "demand":
        raise EvidenceError("forced-eviction trace must contain demand accesses")
    require_exact_integer(
        forced_trace,
        "event_count",
        31,
        "forced_eviction_generation.trace",
    )
    forced_schedule = [
        ("miss", False, 7_904),
        ("load_started", False, 7_904),
        ("admitted", False, 7_904),
        ("miss", True, 65_536),
        ("evicted", False, 7_904),
        ("load_started", True, 65_536),
        ("admitted", True, 65_536),
        ("miss", False, 7_904),
        ("evicted", True, 65_536),
        ("load_started", False, 7_904),
        ("admitted", False, 7_904),
    ] + [("hit", False, 7_904)] * 20
    expected_forced_events = [
        {
            "sequence": sequence,
            "outcome": outcome,
            "reason": "demand",
            "object_digest": (
                multi_page["object_digest"] if uses_interference else tiny["object_digest"]
            ),
            "page_size": 65_536,
            "page_index": 0,
            "logical_bytes": logical_bytes,
        }
        for sequence, (outcome, uses_interference, logical_bytes) in enumerate(
            forced_schedule
        )
    ]
    forced_events = forced_trace.get("events")
    if not isinstance(forced_events, list) or len(forced_events) != len(
        expected_forced_events
    ):
        raise EvidenceError("forced-eviction normalized raw trace differs from the gate")
    for index, (event_value, expected_event) in enumerate(
        zip(forced_events, expected_forced_events, strict=True)
    ):
        event = _mapping(
            event_value,
            f"forced_eviction_generation.trace.events[{index}]",
        )
        _require_exact_keys(
            event,
            (
                "sequence",
                "outcome",
                "reason",
                "object_digest",
                "page_size",
                "page_index",
                "logical_bytes",
            ),
            f"forced_eviction_generation.trace.events[{index}]",
        )
        for name in ("outcome", "reason", "object_digest"):
            if event.get(name) != expected_event[name]:
                raise EvidenceError(
                    "forced-eviction normalized raw trace differs from the gate"
                )
        for name in ("sequence", "page_size", "page_index", "logical_bytes"):
            require_exact_integer(
                event,
                name,
                expected_event[name],
                f"forced_eviction_generation.trace.events[{index}]",
            )
    forced_outcomes = _mapping(
        forced_trace.get("outcomes"),
        "forced_eviction_generation.trace.outcomes",
    )
    expected_forced_outcomes = {
        "hit": 20,
        "miss": 3,
        "load_started": 3,
        "load_coalesced": 0,
        "late_prefetch": 0,
        "prefetch_coalesced": 0,
        "admitted": 3,
        "evicted": 2,
        "retired": 0,
        "load_failed": 0,
        "cancelled": 0,
        "prefetch_useful": 0,
        "prefetch_wasted": 0,
        "prefetch_redundant": 0,
        "prefetch_dropped": 0,
    }
    _require_exact_keys(
        forced_outcomes,
        tuple(expected_forced_outcomes),
        "forced_eviction_generation.trace.outcomes",
    )
    for name, expected in expected_forced_outcomes.items():
        require_exact_integer(
            forced_outcomes,
            name,
            expected,
            "forced_eviction_generation.trace.outcomes",
        )
    for metric_name, outcome_name in (
        ("hits", "hit"),
        ("misses", "miss"),
        ("admissions", "admitted"),
        ("evictions", "evicted"),
    ):
        if forced_metrics[metric_name] != forced_outcomes[outcome_name]:
            raise EvidenceError("forced-eviction trace and metric counts differ")

    trace = _mapping(projection.get("trace"), "trace")
    _require_exact_keys(
        trace,
        ("access", "page_indices", "page_lengths", "event_count", "outcomes", "events"),
        "trace",
    )
    if not isinstance(trace.get("access"), str) or trace["access"] != "demand":
        raise EvidenceError("M2 trace must contain demand accesses")
    require_exact_integer_list(
        trace.get("page_indices"), (0, 1, 0, 2, 2), "trace.page_indices"
    )
    require_exact_integer_list(
        trace.get("page_lengths"), (65_536, 65_536, 17), "trace.page_lengths"
    )
    require_exact_integer(trace, "event_count", 16, "trace")
    expected_events = [
        {
            "sequence": sequence,
            "outcome": outcome,
            "reason": "demand",
            "object_digest": multi_page["object_digest"],
            "page_size": 65_536,
            "page_index": page_index,
            "logical_bytes": logical_bytes,
        }
        for sequence, (outcome, page_index, logical_bytes) in enumerate(
            (
                ("miss", 0, 65_536),
                ("load_started", 0, 65_536),
                ("admitted", 0, 65_536),
                ("miss", 1, 65_536),
                ("evicted", 0, 65_536),
                ("load_started", 1, 65_536),
                ("admitted", 1, 65_536),
                ("miss", 0, 65_536),
                ("evicted", 1, 65_536),
                ("load_started", 0, 65_536),
                ("admitted", 0, 65_536),
                ("miss", 2, 17),
                ("evicted", 0, 65_536),
                ("load_started", 2, 17),
                ("admitted", 2, 17),
                ("hit", 2, 17),
            )
        )
    ]
    trace_events = trace.get("events")
    if not isinstance(trace_events, list) or len(trace_events) != len(expected_events):
        raise EvidenceError("M2 normalized raw trace differs from the pinned trace")
    for index, (event_value, expected_event) in enumerate(
        zip(trace_events, expected_events, strict=True)
    ):
        event = _mapping(event_value, f"trace.events[{index}]")
        _require_exact_keys(
            event,
            (
                "sequence",
                "outcome",
                "reason",
                "object_digest",
                "page_size",
                "page_index",
                "logical_bytes",
            ),
            f"trace.events[{index}]",
        )
        for name in ("outcome", "reason", "object_digest"):
            if not isinstance(event.get(name), str) or event[name] != expected_event[name]:
                raise EvidenceError("M2 normalized raw trace differs from the pinned trace")
        for name in ("sequence", "page_size", "page_index", "logical_bytes"):
            require_exact_integer(
                event,
                name,
                expected_event[name],
                f"trace.events[{index}]",
            )
    outcomes = _mapping(trace.get("outcomes"), "trace.outcomes")
    expected_outcomes = {
        "hit": 1,
        "miss": 4,
        "load_started": 4,
        "load_coalesced": 0,
        "late_prefetch": 0,
        "prefetch_coalesced": 0,
        "admitted": 4,
        "evicted": 3,
        "retired": 0,
        "load_failed": 0,
        "cancelled": 0,
        "prefetch_useful": 0,
        "prefetch_wasted": 0,
        "prefetch_redundant": 0,
        "prefetch_dropped": 0,
    }
    _require_exact_keys(outcomes, tuple(expected_outcomes), "trace.outcomes")
    for name, expected in expected_outcomes.items():
        require_exact_integer(outcomes, name, expected, "trace.outcomes")
    require_exact_integer(projection, "cache_capacity_bytes", 65_536, "root")

    metrics = _mapping(projection.get("metrics"), "metrics")
    expected_metrics = {
        "demand_bytes": 196_642,
        "physical_read_bytes": 196_625,
        "hits": 1,
        "misses": 4,
        "admissions": 4,
        "evictions": 3,
        "coalesced_demands": 0,
        "trace_events_dropped": 0,
    }
    _require_exact_keys(metrics, tuple(expected_metrics) + ("prefetch", "accounted"), "metrics")
    for name, expected in expected_metrics.items():
        require_exact_integer(metrics, name, expected, "metrics")
    prefetch = _mapping(metrics.get("prefetch"), "metrics.prefetch")
    prefetch_names = ("bytes", "coalesced", "late", "useful", "wasted", "redundant", "dropped")
    _require_exact_keys(prefetch, prefetch_names, "metrics.prefetch")
    for name in prefetch_names:
        require_exact_integer(prefetch, name, 0, "metrics.prefetch")
    accounted = _mapping(metrics.get("accounted"), "metrics.accounted")
    expected_accounted = {
        "active_loads": 0,
        "page_pool_bytes": 64,
        "inflight_bytes": 0,
        "resident_bytes": 64,
        "retiring_bytes": 0,
        "leases": 0,
    }
    _require_exact_keys(accounted, tuple(expected_accounted), "metrics.accounted")
    for name, expected in expected_accounted.items():
        require_exact_integer(accounted, name, expected, "metrics.accounted")


def fixture_and_trace_records(projection: Mapping[str, Any]) -> dict[str, Any]:
    fixtures = _copy_json(projection["fixtures"])
    trace = _mapping(projection["trace"], "trace")
    forced_eviction = _copy_json(projection["forced_eviction_generation"])
    trace_definition = {
        "access": _copy_json(trace["access"]),
        "page_indices": _copy_json(trace["page_indices"]),
        "page_lengths": _copy_json(trace["page_lengths"]),
        "events": _copy_json(trace["events"]),
        "cache_capacity_bytes": projection["cache_capacity_bytes"],
    }
    return {
        "fixture": {"digest": digest_json(fixtures), "projection": fixtures},
        "trace": {
            "digest": digest_json(trace_definition),
            "projection": trace_definition,
        },
        "forced_eviction": {
            "digest": digest_json(forced_eviction),
            "projection": forced_eviction,
        },
    }


def percentile(values: Sequence[int | float], probability: float) -> float:
    """Return the deterministic type-7 linearly interpolated percentile."""

    if not values:
        raise EvidenceError("cannot calculate a percentile of no samples")
    if not 0.0 <= probability <= 1.0:
        raise EvidenceError("percentile probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_median_interval(
    values: Sequence[int | float], *, seed: int, resamples: int
) -> dict[str, Any] | None:
    """Return a deterministic nonparametric 95% bootstrap median interval."""

    if not values:
        return None
    if resamples <= 0:
        raise EvidenceError("bootstrap resamples must be positive")
    samples = tuple(float(value) for value in values)
    generator = random.Random(seed)
    medians = []
    for _ in range(resamples):
        draw = [samples[generator.randrange(len(samples))] for _ in samples]
        medians.append(statistics.median(draw))
    return {
        "confidence_level": 0.95,
        "lower": percentile(medians, 0.025),
        "upper": percentile(medians, 0.975),
        "method": "seeded nonparametric percentile bootstrap of the median",
        "resamples": resamples,
        "seed": seed,
    }


def summarize_samples(
    values: Sequence[int | float], *, bootstrap_seed: int, bootstrap_resamples: int
) -> dict[str, Any] | None:
    """Return contract statistics, or ``None`` when no run succeeded."""

    if not values:
        return None
    numeric = tuple(float(value) for value in values)
    return {
        "sample_count": len(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
        "sample_standard_deviation": (
            statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        ),
        "median_bootstrap_95_percent_ci": bootstrap_median_interval(
            numeric, seed=bootstrap_seed, resamples=bootstrap_resamples
        ),
        "percentile_method": "type-7 linear interpolation",
    }


def _safe_version(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or len(value) != 1:
        return None
    line = value[0]
    if len(line) > 160 or not re.fullmatch(r"[A-Za-z0-9 ._+()/:=-]+", line):
        return None
    return line


def _read_text(path: Path, maximum: int = 16_384) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > maximum:
        return None
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None


def _cpu_metadata() -> dict[str, Any]:
    text = _read_text(Path("/proc/cpuinfo"), 2 * 1024 * 1024) or ""
    model = None
    flags: list[str] = []
    frequencies: list[float] = []
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if model is None and key in {"model name", "Hardware"}:
            model = value[:160]
        elif not flags and key in {"flags", "Features"}:
            flags = sorted(set(value.split()))
        elif key == "cpu MHz":
            try:
                frequencies.append(float(value))
            except ValueError:
                pass
    governors = set()
    for cpu in sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []:
        value = _read_text(
            Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"),
            128,
        )
        if value and re.fullmatch(r"[a-z0-9_-]+", value):
            governors.add(value)
    node_root = Path("/sys/devices/system/node")
    numa_nodes = 0
    try:
        numa_nodes = sum(
            1 for entry in node_root.iterdir() if re.fullmatch(r"node[0-9]+", entry.name)
        )
    except OSError:
        pass
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "logical_count": os.cpu_count(),
        "model": model,
        "flags": flags,
        "affinity_logical_cpus": affinity,
        "observed_mhz": {
            "minimum": min(frequencies) if frequencies else None,
            "maximum": max(frequencies) if frequencies else None,
        },
        "scaling_governors": sorted(governors),
        "numa_node_count": numa_nodes or None,
    }


def _memory_metadata() -> dict[str, int | None]:
    selected = {
        "MemTotal": "total_bytes",
        "MemAvailable": "available_bytes",
        "SwapTotal": "swap_total_bytes",
        "SwapFree": "swap_free_bytes",
    }
    result: dict[str, int | None] = {target: None for target in selected.values()}
    text = _read_text(Path("/proc/meminfo")) or ""
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in selected:
            continue
        match = re.fullmatch(r"\s*([0-9]+)\s+kB\s*", value)
        if match:
            result[selected[key]] = int(match.group(1)) * 1024
    return result


def _filesystem_type(path: Path) -> str | None:
    text = _read_text(Path("/proc/self/mountinfo"), 2 * 1024 * 1024) or ""
    target = str(path.resolve(strict=True))
    best: tuple[int, str] | None = None
    for line in text.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            mountpoint = fields[4]
            filesystem_type = fields[separator + 1]
        except (ValueError, IndexError):
            continue
        for escaped, character in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            mountpoint = mountpoint.replace(escaped, character)
        prefix = mountpoint.rstrip("/") + "/"
        if target == mountpoint or target.startswith(prefix):
            candidate = (len(mountpoint), filesystem_type)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best[1] if best else None


def _storage_class(path: Path) -> str:
    device = path.stat().st_dev
    sys_device = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}")
    try:
        resolved = sys_device.resolve(strict=True)
    except OSError:
        return "unknown"
    for candidate in (resolved, *resolved.parents):
        rotational = _read_text(candidate / "queue/rotational", 16)
        if rotational == "0":
            return "non-rotational block device"
        if rotational == "1":
            return "rotational block device"
    return "unknown"


def capture_environment(binary: Path, harness_digest: str) -> dict[str, Any]:
    stat = os.statvfs(ROOT)
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    os_release = platform.freedesktop_os_release()
    toolchain = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "rustc": _safe_version(("rustc", "--version")),
        "cargo": _safe_version(("cargo", "--version")),
        "runnel": _safe_version((str(binary), "--version")),
    }
    initial_child_major_faults = resource.getrusage(
        resource.RUSAGE_CHILDREN
    ).ru_majflt
    return {
        "schema_version": 1,
        "captured_at_utc": utc_now(),
        "harness": {
            "version": HARNESS_VERSION,
            "source_sha256": f"sha256:{harness_digest}",
        },
        "operating_system": {
            "id": os_release.get("ID"),
            "version_id": os_release.get("VERSION_ID"),
            "kernel_system": platform.system(),
            "kernel_release": platform.release(),
        },
        "architecture": platform.machine(),
        "cpu": _cpu_metadata(),
        "memory": _memory_metadata(),
        "filesystem": {
            "type": _filesystem_type(ROOT),
            "storage_class": _storage_class(ROOT),
            "block_size_bytes": stat.f_frsize,
            "total_bytes": stat.f_blocks * stat.f_frsize,
            "available_bytes": stat.f_bavail * stat.f_frsize,
        },
        "host_state": {
            "load_average_1_5_15": list(load) if load else None,
            "initial_child_major_page_faults": initial_child_major_faults,
        },
        "toolchain": toolchain,
        "build": {"profile": "release", "features": "workspace defaults"},
        "subprocess_environment": CONTROLLED_ENVIRONMENT,
    }


def build_release_binary(binary: Path) -> dict[str, Any]:
    """Build the measured release binary from the already-validated clean HEAD."""

    stat = os.statvfs(ROOT)
    available = stat.f_bavail * stat.f_frsize
    if available < MIN_BUILD_FREE_BYTES:
        raise EvidenceError("at least 2 GiB of free filesystem space is required to build")
    environment = {
        key: value
        for key in ("PATH", "HOME", "CARGO_HOME", "RUSTUP_HOME")
        if (value := os.environ.get(key)) is not None
    }
    environment.update(CONTROLLED_ENVIRONMENT)
    environment.update(BUILD_ENVIRONMENT)
    started = time.perf_counter_ns()
    completed = run_bounded_process(
        BUILD_COMMAND,
        cwd=ROOT,
        environment=environment,
        timeout_seconds=BUILD_TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter_ns() - started
    if completed.launch_error or completed.timed_out:
        raise EvidenceError("the locked offline release build could not complete")
    if completed.output_exceeded:
        raise EvidenceError("the locked offline release build emitted excessive output")
    if completed.return_code != 0:
        raise EvidenceError("the locked offline release build failed")
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise EvidenceError("the release build did not produce a regular executable")
    return {
        "command_argv": list(BUILD_COMMAND),
        "command_display": " ".join(BUILD_COMMAND),
        "environment": dict(sorted(BUILD_ENVIRONMENT.items())),
        "wall_time_nanoseconds": elapsed,
        "stdout_bytes": len(completed.stdout),
        "stderr_bytes": len(completed.stderr),
    }


def _contains_private_path(value: Any) -> bool:
    if isinstance(value, str):
        if (
            str(ROOT) in value
            or PurePosixPath(value).is_absolute()
            or re.search(r"(?:^|\s)/[^\s]+", value)
        ):
            return True
        return bool(PureWindowsPath(value).is_absolute())
    if isinstance(value, Mapping):
        return any(
            _contains_private_path(key) or _contains_private_path(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_path(item) for item in value)
    return False


def _usage_snapshot() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_CHILDREN)


def run_once(
    *,
    trial_index: int,
    timeout_seconds: float,
    expected_projection_digest: str | None,
) -> dict[str, Any]:
    """Run one bounded CLI process and return a complete raw observation."""

    before = _usage_snapshot()
    started_at = utc_now()
    start = time.perf_counter_ns()
    stdout = b""
    stderr = b""
    return_code: int | None = None
    status = "launch_error"
    demo: Mapping[str, Any] | None = None
    projection_digest: str | None = None
    matches_gate = False
    timed_out = False
    completed = run_bounded_process(
        COMMAND,
        cwd=ROOT,
        environment=CONTROLLED_ENVIRONMENT,
        timeout_seconds=timeout_seconds,
    )
    stdout = completed.stdout
    stderr = completed.stderr
    return_code = completed.return_code
    timed_out = completed.timed_out
    if completed.launch_error:
        status = "launch_error"
    elif timed_out:
        status = "timeout"
    elif completed.output_exceeded:
        status = "output_too_large"
    elif return_code != 0:
        status = "nonzero_exit"
    else:
        try:
            parsed = json.loads(stdout)
            demo = _mapping(parsed, "root")
            if _contains_private_path(demo):
                raise EvidenceError("data-plane output contains a private path")
            projection = project_correctness(demo)
            validate_correctness_projection(projection)
            projection_digest = digest_json(projection)
            matches_gate = (
                expected_projection_digest is None
                or projection_digest == expected_projection_digest
            )
            status = "ok" if matches_gate else "projection_mismatch"
        except (
            EvidenceError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            TypeError,
            ValueError,
        ):
            demo = None
            status = "invalid_or_incorrect_json"
    elapsed = time.perf_counter_ns() - start
    after = _usage_snapshot()
    row: dict[str, Any] = {
        "schema_version": 2,
        "trial_index": trial_index,
        "phase": "measured",
        "started_at_utc": started_at,
        "status": status,
        "timed_out": timed_out,
        "return_code": return_code,
        "wall_time_nanoseconds": elapsed,
        "child_page_faults": {
            "major": max(0, after.ru_majflt - before.ru_majflt),
            "minor": max(0, after.ru_minflt - before.ru_minflt),
        },
        "stdout": {"bytes": len(stdout), "sha256": f"sha256:{sha256_bytes(stdout)}"},
        "stderr": {"bytes": len(stderr), "sha256": f"sha256:{sha256_bytes(stderr)}"},
        "correctness_projection_digest": projection_digest,
        "correctness_projection_matches_gate": matches_gate,
    }
    if status == "ok" and demo is not None:
        row["demo"] = _copy_json(demo)
    return row


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(value, destination, allow_nan=False, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())


def build_summary(
    *,
    experiment_id: str,
    rows: Sequence[Mapping[str, Any]],
    gate_projection_digest: str,
    observations_digest: str,
    binary_unchanged: bool,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    status_counts = Counter(str(row["status"]) for row in rows)
    successful = [row for row in rows if row["status"] == "ok"]
    wall_times = [int(row["wall_time_nanoseconds"]) for row in successful]
    major_faults = [int(row["child_page_faults"]["major"]) for row in successful]
    minor_faults = [int(row["child_page_faults"]["minor"]) for row in successful]

    def demo_metric(path: Sequence[str]) -> list[int]:
        values: list[int] = []
        for row in successful:
            value: Any = row["demo"]
            for component in path:
                if not isinstance(value, Mapping) or component not in value:
                    value = None
                    break
                value = value[component]
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
        return values

    distribution_specs = {
        "cache_wait_nanoseconds": (("metrics", "wait_nanoseconds"), "nanoseconds"),
        "cache_io_nanoseconds": (("metrics", "io_nanoseconds"), "nanoseconds"),
        "observed_rss_resident_bytes": (
            ("metrics", "observed_rss", "resident_bytes"),
            "bytes",
        ),
        "observed_rss_peak_bytes": (
            ("metrics", "observed_rss", "peak_bytes"),
            "bytes",
        ),
    }
    distributions: dict[str, Any] = {}
    for name, (path, unit) in distribution_specs.items():
        values = demo_metric(path)
        distributions[name] = {
            "unit": unit,
            "observed_sample_count": len(values),
            "missing_sample_count": len(successful) - len(values),
            "statistics": summarize_samples(
                values,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            ),
        }

    deterministic_paths = {
        "demand_bytes": ("metrics", "demand_bytes"),
        "physical_read_bytes": ("metrics", "physical_read_bytes"),
        "hits": ("metrics", "hits"),
        "misses": ("metrics", "misses"),
        "admissions": ("metrics", "admissions"),
        "evictions": ("metrics", "evictions"),
    }
    deterministic_metrics = {}
    for name, path in deterministic_paths.items():
        values = demo_metric(path)
        unique = sorted(set(values))
        deterministic_metrics[name] = {
            "observed_sample_count": len(values),
            "all_successful_trials_equal": (
                bool(successful)
                and len(unique) == 1
                and len(values) == len(successful)
            ),
            "value": unique[0] if len(unique) == 1 else None,
        }

    forced_deterministic_paths = {
        "demand_bytes": (
            "forced_eviction_generation",
            "metrics",
            "demand_bytes",
        ),
        "physical_read_bytes": (
            "forced_eviction_generation",
            "metrics",
            "physical_read_bytes",
        ),
        "hits": ("forced_eviction_generation", "metrics", "hits"),
        "misses": ("forced_eviction_generation", "metrics", "misses"),
        "admissions": ("forced_eviction_generation", "metrics", "admissions"),
        "evictions": ("forced_eviction_generation", "metrics", "evictions"),
        "trace_event_count": ("forced_eviction_generation", "trace", "event_count"),
    }
    forced_deterministic_metrics = {}
    for name, path in forced_deterministic_paths.items():
        values = demo_metric(path)
        unique = sorted(set(values))
        forced_deterministic_metrics[name] = {
            "observed_sample_count": len(values),
            "all_successful_trials_equal": (
                bool(successful)
                and len(unique) == 1
                and len(values) == len(successful)
            ),
            "value": unique[0] if len(unique) == 1 else None,
        }

    matching_count = sum(
        row.get("correctness_projection_matches_gate") is True for row in rows
    )
    parity_count = sum(
        isinstance(row.get("demo"), Mapping) and row["demo"].get("parity") is True
        for row in rows
    )
    forced_parity_count = sum(
        isinstance(row.get("demo"), Mapping)
        and isinstance(row["demo"].get("forced_eviction_generation"), Mapping)
        and row["demo"]["forced_eviction_generation"].get("full_generation_parity")
        is True
        for row in rows
    )
    ratio_values: list[tuple[int, int]] = []
    for row in successful:
        ratio = row["demo"]["generation_cache"]["bytes_per_generated_token"]
        numerator = ratio["numerator_bytes"]
        denominator = ratio["denominator_tokens"]
        if (
            isinstance(numerator, int)
            and not isinstance(numerator, bool)
            and isinstance(denominator, int)
            and not isinstance(denominator, bool)
            and denominator > 0
        ):
            ratio_values.append((numerator, denominator))
    unique_ratios = sorted(set(ratio_values))
    exact_ratio = unique_ratios[0] if len(unique_ratios) == 1 else None
    observability_complete = bool(successful) and all(
        distribution["observed_sample_count"] == len(successful)
        for distribution in distributions.values()
    )
    revalidated_successful_count = 0
    for row in successful:
        try:
            demo = _mapping(row.get("demo"), "summary.demo")
            projection = project_correctness(demo)
            validate_correctness_projection(projection)
            digest = digest_json(projection)
            if (
                digest == gate_projection_digest
                and row.get("correctness_projection_digest") == digest
                and row.get("correctness_projection_matches_gate") is True
            ):
                revalidated_successful_count += 1
        except (EvidenceError, TypeError, ValueError):
            pass
    all_successful_revalidate = bool(successful) and revalidated_successful_count == len(
        successful
    )
    all_correct = (
        bool(rows)
        and len(successful) == len(rows)
        and all_successful_revalidate
    )
    all_ordinary_parity = bool(rows) and parity_count == len(rows)
    all_forced_parity = bool(rows) and forced_parity_count == len(rows)
    passed = (
        all_correct
        and all_ordinary_parity
        and all_forced_parity
        and observability_complete
        and binary_unchanged
    )
    return {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "source_observations_sha256": f"sha256:{observations_digest}",
        "trial_count": len(rows),
        "successful_trial_count": len(successful),
        "status_counts": dict(sorted(status_counts.items())),
        "correctness": {
            "gate_projection_digest": gate_projection_digest,
            "all_measured_trials_match_gate": all_correct,
            "all_successful_trials_revalidated": all_successful_revalidate,
            "projection_match_fraction": {
                "numerator": matching_count,
                "denominator": len(rows),
                "value": matching_count / len(rows) if rows else None,
            },
            "parity_pass_fraction": {
                "numerator": parity_count,
                "denominator": len(rows),
                "value": parity_count / len(rows) if rows else None,
            },
            "forced_eviction_full_generation_parity_fraction": {
                "numerator": forced_parity_count,
                "denominator": len(rows),
                "value": forced_parity_count / len(rows) if rows else None,
            },
            "exact_generation_bytes_per_token": {
                "numerator_physical_read_bytes": exact_ratio[0] if exact_ratio else None,
                "denominator_generated_tokens": exact_ratio[1] if exact_ratio else None,
                "bytes_per_generated_token": (
                    exact_ratio[0] / exact_ratio[1] if exact_ratio else None
                ),
                "all_successful_trials_equal": (
                    bool(successful)
                    and len(unique_ratios) == 1
                    and len(ratio_values) == len(successful)
                ),
                "definition": (
                    "one async cache-backed cold object-payload load divided by "
                    "subsequently generated tokens; excludes metadata, the synchronous "
                    "comparator, and the separate forced-eviction path"
                ),
            },
            "binary_unchanged_during_experiment": binary_unchanged,
            "all_successful_trials_have_observability": observability_complete,
            "outcome": "pass" if passed else "fail",
        },
        "wall_time_nanoseconds": summarize_samples(
            wall_times,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "cache_and_rss_distributions": distributions,
        "child_page_faults": {
            "major": summarize_samples(
                major_faults,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "minor": summarize_samples(
                minor_faults,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            ),
        },
        "deterministic_cache_metrics": deterministic_metrics,
        "deterministic_forced_eviction_metrics": forced_deterministic_metrics,
        "interpretation": {
            "scope": (
                "M2 full-generation parity under forced eviction, verified I/O "
                "accounting, and fixed-trace cache behavior"
            ),
            "performance_claim": "none; wall time characterizes this validation command only",
            "shared_host_caveat": "virtualized-host timing is noisy and uncontrolled",
        },
    }


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="new portable ID below benchmarks/raw")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parsed = parser.parse_args(arguments)
    validate_experiment_id(parsed.experiment_id)
    if parsed.warmups < 0 or parsed.warmups > 10_000:
        parser.error("--warmups must be between 0 and 10000")
    if parsed.repetitions < 30 or parsed.repetitions > 100_000:
        parser.error("--repetitions must be between 30 and 100000")
    if (
        not math.isfinite(parsed.timeout_seconds)
        or parsed.timeout_seconds <= 0
        or parsed.timeout_seconds > 3_600
    ):
        parser.error("--timeout-seconds must be finite and in (0, 3600]")
    if parsed.bootstrap_seed < 0 or parsed.bootstrap_seed >= 2**64:
        parser.error("--bootstrap-seed must be an unsigned 64-bit integer")
    if parsed.bootstrap_resamples < 1_000 or parsed.bootstrap_resamples > 1_000_000:
        parser.error("--bootstrap-resamples must be between 1000 and 1000000")
    return parsed


def effective_harness_invocation(arguments: argparse.Namespace) -> dict[str, Any]:
    """Return the canonical fully explicit reproduction command."""

    argv = [
        "python3",
        "scripts/run_m2_experiment.py",
        arguments.experiment_id,
        "--warmups",
        str(arguments.warmups),
        "--repetitions",
        str(arguments.repetitions),
        "--timeout-seconds",
        format(arguments.timeout_seconds, ".17g"),
        "--bootstrap-seed",
        str(arguments.bootstrap_seed),
        "--bootstrap-resamples",
        str(arguments.bootstrap_resamples),
    ]
    return {"argv": argv, "display": " ".join(argv)}


def run(arguments: argparse.Namespace) -> int:
    started_at = utc_now()
    output = ensure_new_output_destination(arguments.experiment_id)
    commit = require_clean_worktree()
    binary = resolve_repository_path(BINARY_RELATIVE)
    build = build_release_binary(binary)
    if require_clean_worktree() != commit:
        raise EvidenceError("Git HEAD changed during the release build")
    binary_digest = sha256_file(binary)
    harness_digest = sha256_file(Path(__file__).resolve(strict=True))

    gate = run_once(
        trial_index=-1,
        timeout_seconds=arguments.timeout_seconds,
        expected_projection_digest=None,
    )
    if gate["status"] != "ok" or "demo" not in gate:
        raise EvidenceError(f"correctness gate failed with status {gate['status']}")
    gate_projection = project_correctness(gate["demo"])
    validate_correctness_projection(gate_projection)
    gate_digest = digest_json(gate_projection)

    for warmup_index in range(arguments.warmups):
        warmup = run_once(
            trial_index=warmup_index,
            timeout_seconds=arguments.timeout_seconds,
            expected_projection_digest=gate_digest,
        )
        if warmup["status"] != "ok":
            raise EvidenceError(
                f"warmup {warmup_index} failed with status {warmup['status']}"
            )

    if require_clean_worktree() != commit:
        raise EvidenceError("Git HEAD changed while establishing the experiment gate")
    output = ensure_new_output_destination(arguments.experiment_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    fixture_trace = fixture_and_trace_records(gate_projection)
    harness_invocation = effective_harness_invocation(arguments)
    experiment = {
        "schema_version": 2,
        "experiment_id": arguments.experiment_id,
        "started_at_utc": started_at,
        "git": {"commit": commit, "dirty": False},
        "hypothesis": (
            "The asynchronous verified byte-aware cache reproduces the synchronous "
            "full generation while forced to evict and authenticate a tensor-page "
            "reload, and the one-page cache emits the pinned short-tail trace."
        ),
        "expected_mechanism": (
            "Pages are digest-verified before publication, byte-budgeted, and evicted "
            "by deterministic LRU state transitions."
        ),
        "baseline": "synchronous exact verified page reads",
        "candidate": (
            "asynchronous verified byte-aware one-page LRU cache with a "
            "forced-eviction tensor reload"
        ),
        "command_argv": list(COMMAND),
        "command_display": " ".join(COMMAND),
        "build": build,
        "harness_invocation_argv": harness_invocation["argv"],
        "harness_invocation_display": harness_invocation["display"],
        "binary_sha256": f"sha256:{binary_digest}",
        "harness": {
            "version": HARNESS_VERSION,
            "source": "scripts/run_m2_experiment.py",
            "source_sha256": f"sha256:{harness_digest}",
        },
        "fixture": fixture_trace["fixture"],
        "trace": fixture_trace["trace"],
        "forced_eviction": fixture_trace["forced_eviction"],
        "warmup_count": arguments.warmups,
        "measured_repetitions": arguments.repetitions,
        "seed_schedule": {
            "workload_randomness": "none; fixture and trace are deterministic",
            "trial_order": list(range(arguments.repetitions)),
            "bootstrap_seed": arguments.bootstrap_seed,
        },
        "timeout_seconds_per_process": arguments.timeout_seconds,
        "correctness_tolerance": {
            "tensor_bytes": "exact",
            "routing": "exact",
            "generated_tokens": "exact",
            "absolute": 0.0,
            "relative": 0.0,
        },
        "correctness_gate": {
            "passed": True,
            "parity": True,
            "forced_eviction_full_generation_parity": True,
            "projection_digest": gate_digest,
            "projection": gate_projection,
        },
        "primary_metric": {
            "name": "correctness_projection_match_fraction",
            "unit": "fraction",
            "direction": "higher",
        },
        "secondary_metrics": [
            {"name": "command_wall_time", "unit": "nanoseconds", "direction": "lower"},
            {"name": "cache_wait", "unit": "nanoseconds", "direction": "lower"},
            {"name": "cache_io", "unit": "nanoseconds", "direction": "lower"},
            {"name": "child_major_page_faults", "unit": "faults", "direction": "lower"},
            {"name": "physical_read_bytes", "unit": "bytes", "direction": "diagnostic"},
            {
                "name": "forced_eviction_physical_read_bytes",
                "unit": "bytes",
                "direction": "diagnostic",
            },
            {"name": "observed_peak_rss", "unit": "bytes", "direction": "diagnostic"},
        ],
        "bootstrap": {
            "seed": arguments.bootstrap_seed,
            "resamples": arguments.bootstrap_resamples,
            "confidence_level": 0.95,
        },
        "warmup_policy": (
            "Correctness gate first, then excluded warmups; process/file-cache effects "
            "after that are intentionally uncontrolled and captured as host noise."
        ),
    }
    environment = capture_environment(binary, harness_digest)
    if _contains_private_path(experiment) or _contains_private_path(environment):
        raise EvidenceError("refusing to write metadata containing a private path")
    write_json_exclusive(output / "experiment.json", experiment)
    write_json_exclusive(output / "environment.json", environment)

    rows: list[dict[str, Any]] = []
    observations = output / "observations.jsonl"
    with observations.open("x", encoding="utf-8", newline="\n") as destination:
        for trial_index in range(arguments.repetitions):
            row = run_once(
                trial_index=trial_index,
                timeout_seconds=arguments.timeout_seconds,
                expected_projection_digest=gate_digest,
            )
            rows.append(row)
            destination.write(canonical_json(row).decode("utf-8"))
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())

    observations_digest = sha256_file(observations)
    binary_unchanged = sha256_file(binary) == binary_digest
    summary = build_summary(
        experiment_id=arguments.experiment_id,
        rows=rows,
        gate_projection_digest=gate_digest,
        observations_digest=observations_digest,
        binary_unchanged=binary_unchanged,
        bootstrap_seed=arguments.bootstrap_seed,
        bootstrap_resamples=arguments.bootstrap_resamples,
    )
    write_json_exclusive(output / "summary.json", summary)
    relative_output = f"{OUTPUT_BASE_RELATIVE}/{arguments.experiment_id}"
    print(f"wrote M2 experiment: {relative_output} ({summary['correctness']['outcome']})")
    return 0 if summary["correctness"]["outcome"] == "pass" else 1


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        parsed = parse_arguments(arguments)
        return run(parsed)
    except EvidenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
