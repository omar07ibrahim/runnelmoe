#!/usr/bin/env python3
"""Capture and verify the deterministic M3 cache-policy evidence matrix.

This harness deliberately depends only on the Python 3.12 standard library.
It treats simulator output and committed evidence as hostile input: schemas are
closed, duplicate JSON keys and booleans masquerading as integers are rejected,
and every derived artifact is reproducible from the raw observation ledger.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HARNESS_SCHEMA = "runnel.m3-evidence/2"
MATRIX_SCHEMA = "runnel.cache-matrix/1"
RESULT_SCHEMA = "runnel.cache-result/1"
TRACE_ROW_SCHEMA = "runnel.m3-trace/2"
OBSERVATION_SCHEMA = "runnel.m3-observation/1"
SUMMARY_SCHEMA = "runnel.m3-summary/2"
ENVIRONMENT_SCHEMA = "runnel.m3-environment/2"
EXPERIMENT_SCHEMA = "runnel.m3-experiment/2"

FAMILIES = (
    "stationary_zipf",
    "scan_pollution",
    "phase_shift",
    "cyclic_pressure",
    "markov_clusters",
    "iid_uniform",
)
FAMILY_LABELS = {
    "stationary_zipf": "Stationary Zipf",
    "scan_pollution": "Scan pollution",
    "phase_shift": "Phase shift",
    "cyclic_pressure": "Cyclic pressure",
    "markov_clusters": "Markov clusters",
    "iid_uniform": "IID uniform",
}
FAMILY_TRACE_SLUGS = {
    "stationary_zipf": "stationary-zipf",
    "scan_pollution": "scan-pollution",
    "phase_shift": "phase-shift",
    "cyclic_pressure": "cyclic-pressure",
    "markov_clusters": "markov-clusters",
    "iid_uniform": "iid-uniform",
}
POLICIES = (
    "lru",
    "slru",
    "tiny-lfu",
    "router-admit",
    "router-prefetch",
    "belady",
)
POLICY_LABELS = {
    "lru": "LRU",
    "slru": "SLRU",
    "tiny-lfu": "TinyLFU",
    "router-admit": "Router admit",
    "router-prefetch": "Router prefetch",
    "belady": "Bélády/MIN",
}
CAPACITIES = (2_097_152, 4_194_304, 8_388_608)
REPLICATES = tuple(range(30))
PAGE_BYTES = 65_536
DEFAULT_MEASURED_STEPS = 4_096
MAX_MEASURED_STEPS = 4_096
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_STDOUT_BYTES = 16 * 1024 * 1024
MAX_GENERATED_TRACE_BYTES = 5 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024
MAX_DIRECTORY_BYTES = 16 * 1024 * 1024
MAX_GIT_OUTPUT_BYTES = 64 * 1024
MIN_FREE_BYTES_AFTER_CAPTURE = 2 * 1024 * 1024 * 1024
MIN_BUILD_ROOT_FREE_BYTES = 1024 * 1024 * 1024
BUILD_TIMEOUT_SECONDS = 15 * 60.0
BOOTSTRAP_RESAMPLES = 10_000
CONTROLLED_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
BUILD_RECORDED_ENVIRONMENT = {
    **CONTROLLED_ENVIRONMENT,
    "CARGO_BUILD_JOBS": "2",
    "CARGO_INCREMENTAL": "0",
    "CARGO_TARGET_DIR": "{private-tmpfs-target}",
    "CARGO_TERM_COLOR": "never",
}
BUILD_COMMAND = (
    "cargo",
    "build",
    "--release",
    "--locked",
    "--offline",
    "-p",
    "runnel-sim",
    "--bin",
    "runnel-cache-sim",
)
BINARY_LOGICAL_PATH = "{private-tmpfs-target}/release/runnel-cache-sim"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_FILES = frozenset(
    {
        "environment.json",
        "experiment.json",
        "traces.jsonl",
        "observations.jsonl",
        "summary.json",
        "figures/optimal-gap.svg",
        "figures/paired-change.svg",
        "figures/prefetch-accounting.svg",
    }
)

ANALYSIS_QUESTION = (
    "How do preregistered online policies change paired total physical bytes "
    "relative to byte-capacity LRU, and how large is each policy's traffic gap "
    "to exact uniform-page Belady/MIN, within each named family and capacity?"
)

ANALYSIS_CONTRACT = {
    "scope": "exploratory-per-cell",
    "families_pooled": False,
    "multiplicity_adjustment": "none",
    "omnibus_conclusion_permitted": False,
    "favorable_result_required": False,
    "paired_by": ["family", "replicate", "capacity_bytes"],
    "bootstrap_estimator": "median",
    "bootstrap_method": "deterministic SHA-256-seeded percentile",
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "confidence_level": 0.95,
    "timing_measured": False,
}

TRACE_CONTRACT = {
    "schema": "runnel.m3-generator-contract/1",
    "trace_schema": "runnel.cache-trace/1",
    "generator_revision": "runnel-m3-generator-v1",
    "expert_count": 128,
    "pages_per_expert": 3,
    "page_bytes": PAGE_BYTES,
    "layer": 0,
    "request": 0,
    "top_k_experts": 2,
    "burn_in_steps": 512,
    "measured_steps": DEFAULT_MEASURED_STEPS,
    "seed_domain": "runnel.m3-trace/v1\\u0000",
    "prng": "xoshiro256-star-star-from-sha256-little-endian",
    "expert_permutation": "fisher-yates-unbiased-rejection",
    "prefetch_model": "instant-between-events-v1",
    "measured_route_digest": {
        "domain": "runnel-m3-measured-routes-v1\\u0000",
        "count_encoding": "u64-little-endian",
        "route_encoding": "two-u32-little-endian-experts-in-demand-order",
    },
    "full_route_digest": {
        "domain": "runnel-m3-full-routes-v1\\u0000",
        "source": "simulator-attested; burn-in routes are not present in JSONL",
    },
    "predictor": {
        "age_interval_routes": 256,
        "age_rule": "ceil-half",
        "minimum_support": 8,
        "minimum_score_ppm": 100_000,
        "max_experts": 2,
        "score_scale": 1_000_000,
    },
    "families": {
        "stationary_zipf": {
            "kind": "harmonic-without-replacement",
            "expert_count": 128,
            "weight_numerator": 1_000_000,
        },
        "scan_pollution": {
            "epoch_routes": 64,
            "hot_routes": 48,
            "hot_experts": 16,
            "cold_experts": 112,
            "cold_experts_per_route": 2,
        },
        "phase_shift": {
            "phase_routes": 256,
            "group_count": 8,
            "experts_per_group": 16,
            "within_group": "harmonic-without-replacement",
        },
        "cyclic_pressure": {
            "expert_count": 128,
            "route_stride": 2,
            "pair_offset": 1,
        },
        "markov_clusters": {
            "cluster_count": 16,
            "experts_per_cluster": 8,
            "stay_percent": 85,
            "next_percent": 10,
            "other_percent": 5,
            "within_cluster": "harmonic-without-replacement",
        },
        "iid_uniform": {
            "expert_count": 128,
            "sampling": "uniform-without-replacement",
        },
    },
}

POLICY_CONTRACT = {
    "schema": "runnel.m3-policy-contract/1",
    "lru": {"capacity_unit": "bytes"},
    "slru": {
        "protected_fraction_ppm": 750_000,
        "new_fill_segment": "probation",
    },
    "tiny_lfu": {
        "sketch_depth": 4,
        "sketch_width": 2_048,
        "counter_max": 15,
        "doorkeeper": "one-bit-hash-per-row",
        "aging": "counter-right-shift-and-doorkeeper-clear",
        "sample_accesses": "10-times-capacity-pages",
        "hash_seeds_u64_hex": [
            "243f6a8885a308d3",
            "13198a2e03707344",
            "a4093822299f31d0",
            "082efa98ec4e6c89",
        ],
        "admission": "strict-candidate-density-over-aggregate-victim-density",
    },
    "router": {
        "protected_fraction_ppm": 750_000,
        "minimum_score_ppm": 100_000,
        "max_experts_per_signal": 2,
        "max_pages_per_signal": 6,
        "max_prefetch_bytes_per_signal": 393_216,
        "prefetch_group": "all-absent-expert-pages-or-none",
        "prefetch_victims": "probation-only",
    },
    "belady": {
        "geometry": "uniform-charge-and-logical-bytes-only",
        "miss_candidate_participates": True,
    },
}

METRIC_NAMES = (
    "demand_accesses",
    "demand_logical_bytes",
    "ordinary_demand_hits",
    "ordinary_demand_hit_bytes",
    "useful_prefetch_hits",
    "useful_prefetch_hit_bytes",
    "demand_misses",
    "demand_miss_bytes",
    "demand_load_bytes",
    "prefetch_offered",
    "prefetch_offered_bytes",
    "prefetch_admitted",
    "prefetch_load_bytes",
    "prefetch_useful",
    "prefetch_useful_bytes",
    "prefetch_wasted",
    "prefetch_wasted_bytes",
    "prefetch_redundant",
    "prefetch_redundant_bytes",
    "prefetch_dropped",
    "prefetch_dropped_bytes",
    "admissions",
    "bypasses",
    "evictions",
    "evicted_charge_bytes",
    "final_resident_charge_bytes",
    "peak_resident_charge_bytes",
    "policy_metadata_bytes",
    "policy_metadata_limit_bytes",
    "total_physical_load_bytes",
)


class EvidenceError(RuntimeError):
    """A fail-closed evidence contract violation."""


@dataclass(frozen=True)
class BoundedProcessResult:
    return_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_exceeded: bool
    stderr_exceeded: bool
    launch_error: str | None


@dataclass(frozen=True)
class GeneratedTraceEvidence:
    trace_sha256: str
    seed_sha256: str
    measured_route_sha256: str
    canonical_trace_bytes: int
    page_count: int
    event_count: int
    router_signal_events: int
    demand_events: int


@dataclass(frozen=True)
class PrivateBuild:
    root: Path
    binary: Path
    binary_sha256: str
    cargo_version: str
    rustc_version: str
    build_root_available_bytes: int


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_bounded_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    maximum_stdout_bytes: int = MAX_STDOUT_BYTES,
    maximum_stderr_bytes: int = MAX_STDERR_BYTES,
) -> BoundedProcessResult:
    """Run one matrix command with concurrent hard-bounded output capture."""

    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise EvidenceError("subprocess timeout must be finite and positive")
    if maximum_stdout_bytes < 1 or maximum_stderr_bytes < 1:
        raise EvidenceError("subprocess capture limits must be positive")
    try:
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            env=dict(CONTROLLED_ENVIRONMENT if environment is None else environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        return BoundedProcessResult(
            None, b"", b"", False, False, False, type(error).__name__
        )

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    exceeded = {"stdout": threading.Event(), "stderr": threading.Event()}
    limits = {"stdout": maximum_stdout_bytes, "stderr": maximum_stderr_bytes}

    def drain(stream: Any, name: str) -> None:
        try:
            while True:
                try:
                    chunk = stream.read(64 * 1024)
                except OSError:
                    break
                if not chunk:
                    break
                remaining = limits[name] + 1 - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(chunk[:remaining])
                if len(chunk) > remaining or len(buffers[name]) > limits[name]:
                    exceeded[name].set()
                    _kill_process_group(process)
                    break
        finally:
            stream.close()

    assert process.stdout is not None
    assert process.stderr is not None
    readers = (
        threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
    )
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
        raise EvidenceError("subprocess output reader did not terminate")
    return BoundedProcessResult(
        return_code,
        bytes(buffers["stdout"]),
        bytes(buffers["stderr"]),
        timed_out,
        exceeded["stdout"].is_set(),
        exceeded["stderr"].is_set(),
        None,
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_file_bytes(value: Any) -> bytes:
    return canonical_json(value) + b"\n"


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) + b"\n" for row in rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digest(data: bytes) -> str:
    return f"sha256:{sha256_bytes(data)}"


def _reject_constant(value: str) -> Any:
    raise EvidenceError(f"non-finite JSON number {value!r} is forbidden")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json_bytes(data: bytes, context: str) -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{context}: invalid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except EvidenceError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise EvidenceError(f"{context}: invalid JSON: {error}") from error


def parse_jsonl_bytes(data: bytes, context: str) -> list[dict[str, Any]]:
    if data and not data.endswith(b"\n"):
        raise EvidenceError(f"{context}: JSONL must end with LF")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(data.splitlines(), 1):
        if not line:
            raise EvidenceError(f"{context}:{number}: blank JSONL record")
        if len(line) > 256 * 1024:
            raise EvidenceError(f"{context}:{number}: record exceeds 256 KiB")
        row = parse_json_bytes(line, f"{context}:{number}")
        if type(row) is not dict:
            raise EvidenceError(f"{context}:{number}: record must be an object")
        rows.append(row)
    return rows


def _object(value: Any, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise EvidenceError(f"{context}: expected object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if type(value) is not list:
        raise EvidenceError(f"{context}: expected array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], context: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        unknown = sorted(actual - wanted)
        raise EvidenceError(
            f"{context}: closed schema mismatch; missing={missing}, unknown={unknown}"
        )


def _string(
    value: Any,
    context: str,
    *,
    expected: str | None = None,
    maximum: int = 512,
) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise EvidenceError(f"{context}: expected non-empty string of at most {maximum} characters")
    if expected is not None and value != expected:
        raise EvidenceError(f"{context}: expected {expected!r}, got {value!r}")
    return value


def _uint(value: Any, context: str, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise EvidenceError(f"{context}: expected unsigned integer <= {maximum}")
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise EvidenceError(f"{context}: expected boolean")
    return value


def _digest(value: Any, context: str) -> str:
    digest = _string(value, context, maximum=64)
    if not SHA256_RE.fullmatch(digest):
        raise EvidenceError(f"{context}: expected lowercase SHA-256 hex")
    return digest


def _prefixed_digest(value: Any, context: str) -> str:
    digest = _string(value, context, maximum=71)
    if not digest.startswith("sha256:") or not SHA256_RE.fullmatch(digest[7:]):
        raise EvidenceError(f"{context}: expected sha256:<lowercase-hex>")
    return digest


def _expected_trace_id(family: str, replicate: int, measured_steps: int) -> str:
    return f"m3-{FAMILY_TRACE_SLUGS[family]}-r{replicate:02}-n{measured_steps}"


def _expected_seed_sha256(family: str, replicate: int) -> str:
    material = (
        b"runnel.m3-trace/v1\0"
        + family.encode("ascii")
        + b"\0"
        + str(replicate).encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()


def _exact_key_order(value: Mapping[str, Any], expected: Sequence[str], context: str) -> None:
    _exact_keys(value, expected, context)
    if tuple(value) != tuple(expected):
        raise EvidenceError(
            f"{context}: noncanonical field order; expected {list(expected)!r}"
        )


def _canonical_ordered_record(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("ascii")


def validate_generated_trace(
    data: bytes,
    *,
    family: str,
    replicate: int,
    measured_steps: int,
) -> GeneratedTraceEvidence:
    """Independently validate canonical generator output and reconstruct routes."""

    context = f"generated trace {family} replicate {replicate}"
    if len(data) > MAX_GENERATED_TRACE_BYTES:
        raise EvidenceError(f"{context}: trace exceeds five MiB")
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise EvidenceError(f"{context}: canonical trace requires LF-only framing")
    try:
        data.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{context}: canonical trace must be ASCII") from error
    raw_lines = data[:-1].split(b"\n")
    if not raw_lines or any(not line for line in raw_lines):
        raise EvidenceError(f"{context}: canonical trace contains a blank record")

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw_lines, 1):
        if len(line) > 16 * 1024:
            raise EvidenceError(f"{context}:{number}: record exceeds 16 KiB")
        row = parse_json_bytes(line, f"{context}:{number}")
        if type(row) is not dict:
            raise EvidenceError(f"{context}:{number}: record must be an object")
        if _canonical_ordered_record(row) != line:
            raise EvidenceError(f"{context}:{number}: record is not compact canonical JSON")
        rows.append(row)

    header = rows[0]
    _exact_key_order(
        header,
        (
            "kind",
            "schema",
            "trace_id",
            "page_count",
            "event_count",
            "charge_quantum",
            "prefetch_model",
        ),
        f"{context}.header",
    )
    _string(header["kind"], f"{context}.header.kind", expected="header")
    _string(
        header["schema"],
        f"{context}.header.schema",
        expected="runnel.cache-trace/1",
    )
    _string(
        header["trace_id"],
        f"{context}.header.trace_id",
        expected=_expected_trace_id(family, replicate, measured_steps),
    )
    page_count = _uint(header["page_count"], f"{context}.header.page_count")
    if page_count != 384:
        raise EvidenceError(f"{context}: expected 384 generated pages")
    event_count = _uint(header["event_count"], f"{context}.header.event_count")
    if _uint(header["charge_quantum"], f"{context}.header.charge_quantum") != PAGE_BYTES:
        raise EvidenceError(f"{context}: unexpected charge quantum")
    _string(
        header["prefetch_model"],
        f"{context}.header.prefetch_model",
        expected="instant-between-events-v1",
    )
    if len(rows) != 1 + page_count + event_count:
        raise EvidenceError(f"{context}: header counts differ from JSONL records")

    for page_id, page in enumerate(rows[1 : 1 + page_count]):
        page_context = f"{context}.pages[{page_id}]"
        _exact_key_order(
            page,
            ("id", "logical_bytes", "charge_bytes", "class"),
            page_context,
        )
        if _uint(page["id"], f"{page_context}.id") != page_id:
            raise EvidenceError(f"{page_context}: page identifiers must be dense")
        if _uint(page["logical_bytes"], f"{page_context}.logical_bytes") != PAGE_BYTES:
            raise EvidenceError(f"{page_context}: unexpected logical bytes")
        if _uint(page["charge_bytes"], f"{page_context}.charge_bytes") != PAGE_BYTES:
            raise EvidenceError(f"{page_context}: unexpected charge bytes")
        page_class = _object(page["class"], f"{page_context}.class")
        _exact_key_order(
            page_class,
            ("kind", "layer", "expert", "ordinal"),
            f"{page_context}.class",
        )
        _string(page_class["kind"], f"{page_context}.class.kind", expected="expert")
        if _uint(page_class["layer"], f"{page_context}.class.layer") != 0:
            raise EvidenceError(f"{page_context}: generated layer must be zero")
        expected_expert, expected_ordinal = divmod(page_id, 3)
        if _uint(page_class["expert"], f"{page_context}.class.expert") != expected_expert:
            raise EvidenceError(f"{page_context}: expert catalog mapping changed")
        if _uint(page_class["ordinal"], f"{page_context}.class.ordinal") != expected_ordinal:
            raise EvidenceError(f"{page_context}: page ordinal mapping changed")

    events = rows[1 + page_count :]
    event_index = 0
    expected_sequence = 0
    signal_count = 0
    demand_count = 0
    routes: list[tuple[int, int]] = []
    for step in range(measured_steps):
        if event_index < len(events) and events[event_index].get("kind") == "router_signal":
            signal = events[event_index]
            signal_context = f"{context}.events[{event_index}]"
            _exact_key_order(
                signal,
                ("kind", "sequence", "request", "target_step", "layer", "predictions"),
                signal_context,
            )
            if _uint(signal["sequence"], f"{signal_context}.sequence") != expected_sequence:
                raise EvidenceError(f"{signal_context}: non-dense event sequence")
            if _uint(signal["request"], f"{signal_context}.request") != 0:
                raise EvidenceError(f"{signal_context}: generated request must be zero")
            if _uint(signal["target_step"], f"{signal_context}.target_step") != step:
                raise EvidenceError(f"{signal_context}: signal is not immediately causal")
            if _uint(signal["layer"], f"{signal_context}.layer") != 0:
                raise EvidenceError(f"{signal_context}: generated layer must be zero")
            predictions = _array(signal["predictions"], f"{signal_context}.predictions")
            if not 1 <= len(predictions) <= 2:
                raise EvidenceError(f"{signal_context}: expected one or two predictions")
            parsed_predictions: list[tuple[int, int]] = []
            for prediction_index, prediction_value in enumerate(predictions):
                prediction_context = f"{signal_context}.predictions[{prediction_index}]"
                prediction = _object(prediction_value, prediction_context)
                _exact_key_order(
                    prediction,
                    ("expert", "score_ppm"),
                    prediction_context,
                )
                expert = _uint(prediction["expert"], f"{prediction_context}.expert", 127)
                score = _uint(
                    prediction["score_ppm"],
                    f"{prediction_context}.score_ppm",
                    1_000_000,
                )
                if score < 100_000:
                    raise EvidenceError(f"{prediction_context}: score is below frozen minimum")
                parsed_predictions.append((score, expert))
            if len({expert for _, expert in parsed_predictions}) != len(parsed_predictions):
                raise EvidenceError(f"{signal_context}: duplicate predicted expert")
            if parsed_predictions != sorted(
                parsed_predictions, key=lambda item: (-item[0], item[1])
            ):
                raise EvidenceError(f"{signal_context}: predictions are not stably sorted")
            event_index += 1
            expected_sequence += 1
            signal_count += 1

        demanded_pages: list[int] = []
        for demand_offset in range(6):
            if event_index >= len(events):
                raise EvidenceError(f"{context}: event stream ended during step {step}")
            demand = events[event_index]
            demand_context = f"{context}.events[{event_index}]"
            _exact_key_order(
                demand,
                ("kind", "sequence", "request", "step", "page"),
                demand_context,
            )
            _string(demand["kind"], f"{demand_context}.kind", expected="demand")
            if _uint(demand["sequence"], f"{demand_context}.sequence") != expected_sequence:
                raise EvidenceError(f"{demand_context}: non-dense event sequence")
            if _uint(demand["request"], f"{demand_context}.request") != 0:
                raise EvidenceError(f"{demand_context}: generated request must be zero")
            if _uint(demand["step"], f"{demand_context}.step") != step:
                raise EvidenceError(f"{demand_context}: demand step is not dense")
            demanded_pages.append(_uint(demand["page"], f"{demand_context}.page", 383))
            event_index += 1
            expected_sequence += 1
            demand_count += 1
        first_expert, first_ordinal = divmod(demanded_pages[0], 3)
        second_expert, second_ordinal = divmod(demanded_pages[3], 3)
        if first_ordinal != 0 or second_ordinal != 0 or first_expert == second_expert:
            raise EvidenceError(f"{context}: step {step} is not a distinct top-2 route")
        expected_pages = [
            first_expert * 3,
            first_expert * 3 + 1,
            first_expert * 3 + 2,
            second_expert * 3,
            second_expert * 3 + 1,
            second_expert * 3 + 2,
        ]
        if demanded_pages != expected_pages:
            raise EvidenceError(f"{context}: step {step} page grouping changed")
        routes.append((first_expert, second_expert))

    if event_index != len(events):
        raise EvidenceError(f"{context}: records remain after measured routes")
    if demand_count != measured_steps * 6:
        raise EvidenceError(f"{context}: expected six demand pages per route")

    measured_digest = hashlib.sha256()
    measured_digest.update(b"runnel-m3-measured-routes-v1\0")
    measured_digest.update(struct.pack("<Q", measured_steps))
    for first_expert, second_expert in routes:
        measured_digest.update(struct.pack("<II", first_expert, second_expert))
    return GeneratedTraceEvidence(
        trace_sha256=sha256_bytes(data),
        seed_sha256=_expected_seed_sha256(family, replicate),
        measured_route_sha256=measured_digest.hexdigest(),
        canonical_trace_bytes=len(data),
        page_count=page_count,
        event_count=event_count,
        router_signal_events=signal_count,
        demand_events=demand_count,
    )


def _validate_policy_spec(value: Any, policy: str, capacity_bytes: int, context: str) -> dict[str, Any]:
    spec = _object(value, context)
    if policy in {"lru", "belady"}:
        _exact_keys(spec, ("name",), context)
    elif policy == "slru":
        _exact_keys(spec, ("name", "protected_fraction_ppm"), context)
        if _uint(spec["protected_fraction_ppm"], f"{context}.protected_fraction_ppm", (1 << 32) - 1) != 750_000:
            raise EvidenceError(f"{context}: unexpected SLRU protected fraction")
    elif policy == "tiny-lfu":
        _exact_keys(spec, ("name", "config"), context)
        config = _object(spec["config"], f"{context}.config")
        _exact_keys(config, ("sketch_depth", "sketch_width", "sample_accesses"), f"{context}.config")
        expected = {
            "sketch_depth": 4,
            "sketch_width": 2_048,
            "sample_accesses": (capacity_bytes // PAGE_BYTES) * 10,
        }
        for name, expected_value in expected.items():
            if _uint(config[name], f"{context}.config.{name}") != expected_value:
                raise EvidenceError(f"{context}.config.{name}: expected {expected_value}")
    elif policy in {"router-admit", "router-prefetch"}:
        _exact_keys(spec, ("name", "config"), context)
        config = _object(spec["config"], f"{context}.config")
        expected = {
            "protected_fraction_ppm": 750_000,
            "minimum_score_ppm": 100_000,
            "max_experts_per_signal": 2,
            "max_pages_per_signal": 6,
            "max_prefetch_bytes_per_signal": 393_216,
        }
        _exact_keys(config, tuple(expected), f"{context}.config")
        for name, expected_value in expected.items():
            if _uint(config[name], f"{context}.config.{name}") != expected_value:
                raise EvidenceError(f"{context}.config.{name}: expected {expected_value}")
    else:
        raise EvidenceError(f"{context}: unsupported policy {policy!r}")
    _string(spec["name"], f"{context}.name", expected=policy)
    return spec


def _validate_metrics(value: Any, capacity_bytes: int, policy: str, context: str) -> dict[str, int]:
    metrics = _object(value, context)
    _exact_keys(metrics, METRIC_NAMES, context)
    parsed = {name: _uint(metrics[name], f"{context}.{name}") for name in METRIC_NAMES}

    def identity(left: int, right: int, label: str) -> None:
        if left != right:
            raise EvidenceError(f"{context}: {label} identity failed ({left} != {right})")

    identity(
        parsed["demand_accesses"],
        parsed["ordinary_demand_hits"] + parsed["useful_prefetch_hits"] + parsed["demand_misses"],
        "demand count",
    )
    identity(
        parsed["demand_logical_bytes"],
        parsed["ordinary_demand_hit_bytes"] + parsed["useful_prefetch_hit_bytes"] + parsed["demand_miss_bytes"],
        "demand byte",
    )
    identity(
        parsed["total_physical_load_bytes"],
        parsed["demand_load_bytes"] + parsed["prefetch_load_bytes"],
        "physical-load byte",
    )
    identity(
        parsed["prefetch_admitted"],
        parsed["prefetch_useful"] + parsed["prefetch_wasted"],
        "prefetch classification count",
    )
    identity(
        parsed["prefetch_load_bytes"],
        parsed["prefetch_useful_bytes"] + parsed["prefetch_wasted_bytes"],
        "prefetch classification byte",
    )
    identity(
        parsed["prefetch_offered"],
        parsed["prefetch_admitted"] + parsed["prefetch_redundant"] + parsed["prefetch_dropped"],
        "prefetch offer count",
    )
    identity(
        parsed["prefetch_offered_bytes"],
        parsed["prefetch_load_bytes"] + parsed["prefetch_redundant_bytes"] + parsed["prefetch_dropped_bytes"],
        "prefetch offer byte",
    )
    identity(parsed["useful_prefetch_hits"], parsed["prefetch_useful"], "useful prefetch count")
    identity(parsed["useful_prefetch_hit_bytes"], parsed["prefetch_useful_bytes"], "useful prefetch byte")
    identity(parsed["demand_load_bytes"], parsed["demand_miss_bytes"], "fixed-page miss cost")
    identity(
        parsed["admissions"],
        parsed["demand_misses"] - parsed["bypasses"] + parsed["prefetch_admitted"],
        "admission count",
    )
    if parsed["bypasses"] > parsed["demand_misses"]:
        raise EvidenceError(f"{context}: bypasses exceed demand misses")
    if parsed["evictions"] > parsed["admissions"]:
        raise EvidenceError(f"{context}: evictions exceed admissions")

    count_byte_pairs = (
        ("demand_accesses", "demand_logical_bytes"),
        ("ordinary_demand_hits", "ordinary_demand_hit_bytes"),
        ("useful_prefetch_hits", "useful_prefetch_hit_bytes"),
        ("demand_misses", "demand_miss_bytes"),
        ("prefetch_offered", "prefetch_offered_bytes"),
        ("prefetch_admitted", "prefetch_load_bytes"),
        ("prefetch_useful", "prefetch_useful_bytes"),
        ("prefetch_wasted", "prefetch_wasted_bytes"),
        ("prefetch_redundant", "prefetch_redundant_bytes"),
        ("prefetch_dropped", "prefetch_dropped_bytes"),
        ("evictions", "evicted_charge_bytes"),
    )
    for count_name, byte_name in count_byte_pairs:
        identity(parsed[byte_name], parsed[count_name] * PAGE_BYTES, f"{byte_name} fixed-page")
    identity(
        parsed["final_resident_charge_bytes"],
        (parsed["admissions"] - parsed["evictions"]) * PAGE_BYTES,
        "final residency",
    )
    if parsed["final_resident_charge_bytes"] > capacity_bytes:
        raise EvidenceError(f"{context}: final residency exceeds capacity")
    if not parsed["final_resident_charge_bytes"] <= parsed["peak_resident_charge_bytes"] <= capacity_bytes:
        raise EvidenceError(f"{context}: peak residency is outside final..capacity")
    if parsed["peak_resident_charge_bytes"] % PAGE_BYTES:
        raise EvidenceError(f"{context}: peak residency is not page-aligned")
    if parsed["policy_metadata_bytes"] > parsed["policy_metadata_limit_bytes"]:
        raise EvidenceError(f"{context}: policy metadata exceeds its declared limit")
    if policy != "router-prefetch" and any(parsed[name] for name in METRIC_NAMES if name.startswith("prefetch_")):
        raise EvidenceError(f"{context}: only router-prefetch may report prefetch activity")
    return parsed


def validate_matrix(
    value: Any,
    *,
    expected_family: str,
    expected_replicate: int,
    expected_measured_steps: int,
    generated_trace: GeneratedTraceEvidence,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one CLI matrix and return its trace row and observations."""

    matrix = _object(value, "matrix")
    _exact_keys(
        matrix,
        (
            "schema",
            "family",
            "replicate",
            "measured_steps",
            "seed_sha256",
            "full_route_sha256",
            "measured_route_sha256",
            "trace_sha256",
            "results",
        ),
        "matrix",
    )
    _string(matrix["schema"], "matrix.schema", expected=MATRIX_SCHEMA)
    family = _string(matrix["family"], "matrix.family")
    if family not in FAMILIES or family != expected_family:
        raise EvidenceError(f"matrix.family: unexpected family {family!r}")
    replicate = _uint(matrix["replicate"], "matrix.replicate", (1 << 32) - 1)
    if replicate != expected_replicate:
        raise EvidenceError(f"matrix.replicate: expected {expected_replicate}, got {replicate}")
    measured_steps = _uint(matrix["measured_steps"], "matrix.measured_steps")
    if measured_steps != expected_measured_steps:
        raise EvidenceError(f"matrix.measured_steps: expected {expected_measured_steps}, got {measured_steps}")
    seed_sha256 = _digest(matrix["seed_sha256"], "matrix.seed_sha256")
    full_route_sha256 = _digest(matrix["full_route_sha256"], "matrix.full_route_sha256")
    measured_route_sha256 = _digest(matrix["measured_route_sha256"], "matrix.measured_route_sha256")
    trace_sha256 = _digest(matrix["trace_sha256"], "matrix.trace_sha256")
    if seed_sha256 != generated_trace.seed_sha256:
        raise EvidenceError("matrix.seed_sha256: differs from independent seed derivation")
    if measured_route_sha256 != generated_trace.measured_route_sha256:
        raise EvidenceError(
            "matrix.measured_route_sha256: differs from generated demand reconstruction"
        )
    if trace_sha256 != generated_trace.trace_sha256:
        raise EvidenceError("matrix.trace_sha256: differs from canonical generate output")
    trace_id = _expected_trace_id(family, replicate, measured_steps)
    results = _array(matrix["results"], "matrix.results")
    if len(results) != len(CAPACITIES) * len(POLICIES):
        raise EvidenceError(f"matrix.results: expected 18 results, got {len(results)}")

    observations: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    by_capacity: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, raw_result in enumerate(results):
        context = f"matrix.results[{index}]"
        result = _object(raw_result, context)
        _exact_keys(
            result,
            (
                "schema",
                "trace_id",
                "trace_sha256",
                "policy",
                "policy_spec",
                "capacity_bytes",
                "oracle_optimal",
                "metrics",
                "decision_sha256",
            ),
            context,
        )
        _string(result["schema"], f"{context}.schema", expected=RESULT_SCHEMA)
        _string(result["trace_id"], f"{context}.trace_id", expected=trace_id)
        if _digest(result["trace_sha256"], f"{context}.trace_sha256") != trace_sha256:
            raise EvidenceError(f"{context}: result trace hash differs from matrix")
        policy = _string(result["policy"], f"{context}.policy")
        if policy not in POLICIES:
            raise EvidenceError(f"{context}.policy: unsupported value {policy!r}")
        capacity_bytes = _uint(result["capacity_bytes"], f"{context}.capacity_bytes")
        if capacity_bytes not in CAPACITIES:
            raise EvidenceError(f"{context}.capacity_bytes: outside preregistered capacities")
        key = (capacity_bytes, policy)
        if key in seen:
            raise EvidenceError(f"{context}: duplicate result {key}")
        expected_key = (CAPACITIES[index // len(POLICIES)], POLICIES[index % len(POLICIES)])
        if key != expected_key:
            raise EvidenceError(f"{context}: unstable result order; expected {expected_key}, got {key}")
        seen.add(key)
        policy_spec = _validate_policy_spec(result["policy_spec"], policy, capacity_bytes, f"{context}.policy_spec")
        oracle_optimal = _boolean(result["oracle_optimal"], f"{context}.oracle_optimal")
        if oracle_optimal != (policy == "belady"):
            raise EvidenceError(f"{context}: oracle_optimal must be true only for belady")
        metrics = _validate_metrics(result["metrics"], capacity_bytes, policy, f"{context}.metrics")
        decision_sha256 = _digest(result["decision_sha256"], f"{context}.decision_sha256")
        observation = {
            "schema": OBSERVATION_SCHEMA,
            "family": family,
            "replicate": replicate,
            "measured_steps": measured_steps,
            "seed_sha256": seed_sha256,
            "full_route_sha256": full_route_sha256,
            "measured_route_sha256": measured_route_sha256,
            "trace_id": trace_id,
            "trace_sha256": trace_sha256,
            "capacity_bytes": capacity_bytes,
            "policy": policy,
            "policy_spec": policy_spec,
            "oracle_optimal": oracle_optimal,
            "metrics": metrics,
            "decision_sha256": decision_sha256,
        }
        observations.append(observation)
        by_capacity[capacity_bytes][policy] = observation

    expected_keys = {(capacity, policy) for capacity in CAPACITIES for policy in POLICIES}
    if seen != expected_keys:
        raise EvidenceError("matrix.results: missing preregistered result")
    expected_accesses = measured_steps * 6
    for capacity, policy_rows in by_capacity.items():
        belady_bytes = policy_rows["belady"]["metrics"]["total_physical_load_bytes"]
        if belady_bytes <= 0:
            raise EvidenceError(f"matrix: belady denominator is zero at capacity {capacity}")
        reference_accesses: int | None = None
        reference_logical: int | None = None
        for policy, observation in policy_rows.items():
            metrics = observation["metrics"]
            if metrics["demand_accesses"] != expected_accesses:
                raise EvidenceError("matrix: demand count does not match six pages per measured route")
            if reference_accesses is None:
                reference_accesses = metrics["demand_accesses"]
                reference_logical = metrics["demand_logical_bytes"]
            elif (metrics["demand_accesses"], metrics["demand_logical_bytes"]) != (reference_accesses, reference_logical):
                raise EvidenceError("matrix: policies observed inconsistent demand streams")
            if policy != "belady" and metrics["total_physical_load_bytes"] < belady_bytes:
                raise EvidenceError(f"matrix: {policy} beats exact belady at capacity {capacity}")

    trace_row = {
        "schema": TRACE_ROW_SCHEMA,
        "family": family,
        "replicate": replicate,
        "measured_steps": measured_steps,
        "seed_sha256": seed_sha256,
        "full_route_sha256": full_route_sha256,
        "measured_route_sha256": measured_route_sha256,
        "trace_id": trace_id,
        "trace_sha256": trace_sha256,
        "page_bytes": PAGE_BYTES,
        "canonical_trace_bytes": generated_trace.canonical_trace_bytes,
        "page_count": generated_trace.page_count,
        "event_count": generated_trace.event_count,
        "router_signal_events": generated_trace.router_signal_events,
        "demand_events": generated_trace.demand_events,
        "result_count": len(observations),
    }
    return trace_row, observations


def _validate_trace_row(value: Any, context: str) -> dict[str, Any]:
    row = _object(value, context)
    _exact_keys(
        row,
        (
            "schema",
            "family",
            "replicate",
            "measured_steps",
            "seed_sha256",
            "full_route_sha256",
            "measured_route_sha256",
            "trace_id",
            "trace_sha256",
            "page_bytes",
            "canonical_trace_bytes",
            "page_count",
            "event_count",
            "router_signal_events",
            "demand_events",
            "result_count",
        ),
        context,
    )
    _string(row["schema"], f"{context}.schema", expected=TRACE_ROW_SCHEMA)
    family = _string(row["family"], f"{context}.family")
    if family not in FAMILIES:
        raise EvidenceError(f"{context}.family: not preregistered")
    replicate = _uint(row["replicate"], f"{context}.replicate", (1 << 32) - 1)
    measured_steps = _uint(row["measured_steps"], f"{context}.measured_steps")
    if not 1 <= measured_steps <= MAX_MEASURED_STEPS:
        raise EvidenceError(f"{context}.measured_steps: outside supported range")
    for name in ("seed_sha256", "full_route_sha256", "measured_route_sha256", "trace_sha256"):
        _digest(row[name], f"{context}.{name}")
    if row["seed_sha256"] != _expected_seed_sha256(family, replicate):
        raise EvidenceError(f"{context}.seed_sha256: differs from frozen derivation")
    _string(
        row["trace_id"],
        f"{context}.trace_id",
        expected=_expected_trace_id(family, replicate, measured_steps),
    )
    if _uint(row["page_bytes"], f"{context}.page_bytes") != PAGE_BYTES:
        raise EvidenceError(f"{context}.page_bytes: fixed geometry changed")
    if _uint(row["canonical_trace_bytes"], f"{context}.canonical_trace_bytes") < 1:
        raise EvidenceError(f"{context}.canonical_trace_bytes: must be positive")
    if _uint(row["page_count"], f"{context}.page_count") != 384:
        raise EvidenceError(f"{context}.page_count: expected 384")
    event_count = _uint(row["event_count"], f"{context}.event_count")
    signal_events = _uint(
        row["router_signal_events"], f"{context}.router_signal_events"
    )
    demand_events = _uint(row["demand_events"], f"{context}.demand_events")
    if demand_events != measured_steps * 6:
        raise EvidenceError(f"{context}.demand_events: expected six pages per route")
    if event_count != signal_events + demand_events or signal_events > measured_steps:
        raise EvidenceError(f"{context}: event classification identity failed")
    if _uint(row["result_count"], f"{context}.result_count") != 18:
        raise EvidenceError(f"{context}.result_count: expected 18")
    return row


def _validate_observation(value: Any, context: str) -> dict[str, Any]:
    row = _object(value, context)
    _exact_keys(
        row,
        (
            "schema",
            "family",
            "replicate",
            "measured_steps",
            "seed_sha256",
            "full_route_sha256",
            "measured_route_sha256",
            "trace_id",
            "trace_sha256",
            "capacity_bytes",
            "policy",
            "policy_spec",
            "oracle_optimal",
            "metrics",
            "decision_sha256",
        ),
        context,
    )
    _string(row["schema"], f"{context}.schema", expected=OBSERVATION_SCHEMA)
    family = _string(row["family"], f"{context}.family")
    if family not in FAMILIES:
        raise EvidenceError(f"{context}.family: not preregistered")
    replicate = _uint(row["replicate"], f"{context}.replicate", (1 << 32) - 1)
    measured_steps = _uint(row["measured_steps"], f"{context}.measured_steps")
    if not 1 <= measured_steps <= MAX_MEASURED_STEPS:
        raise EvidenceError(f"{context}.measured_steps: outside supported range")
    for name in ("seed_sha256", "full_route_sha256", "measured_route_sha256", "trace_sha256", "decision_sha256"):
        _digest(row[name], f"{context}.{name}")
    if row["seed_sha256"] != _expected_seed_sha256(family, replicate):
        raise EvidenceError(f"{context}.seed_sha256: differs from frozen derivation")
    _string(
        row["trace_id"],
        f"{context}.trace_id",
        expected=_expected_trace_id(family, replicate, measured_steps),
    )
    capacity = _uint(row["capacity_bytes"], f"{context}.capacity_bytes")
    if capacity not in CAPACITIES:
        raise EvidenceError(f"{context}.capacity_bytes: not preregistered")
    policy = _string(row["policy"], f"{context}.policy")
    if policy not in POLICIES:
        raise EvidenceError(f"{context}.policy: not preregistered")
    _validate_policy_spec(row["policy_spec"], policy, capacity, f"{context}.policy_spec")
    oracle = _boolean(row["oracle_optimal"], f"{context}.oracle_optimal")
    if oracle != (policy == "belady"):
        raise EvidenceError(f"{context}.oracle_optimal: inconsistent with policy")
    metrics = _validate_metrics(row["metrics"], capacity, policy, f"{context}.metrics")
    if metrics["demand_accesses"] != measured_steps * 6:
        raise EvidenceError(f"{context}: expected six fixed pages per measured route")
    return row


def validate_dataset(
    traces: Sequence[dict[str, Any]],
    observations: Sequence[dict[str, Any]],
    *,
    measured_steps: int,
    require_full_matrix: bool = True,
) -> None:
    """Validate ledgers, pairing, trace identity, and exact-oracle ordering."""

    expected_trace_count = len(FAMILIES) * len(REPLICATES)
    expected_observation_count = expected_trace_count * len(CAPACITIES) * len(POLICIES)
    if require_full_matrix and len(traces) != expected_trace_count:
        raise EvidenceError(f"traces.jsonl: expected 180 rows, got {len(traces)}")
    if require_full_matrix and len(observations) != expected_observation_count:
        raise EvidenceError(f"observations.jsonl: expected 3240 rows, got {len(observations)}")

    trace_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    trace_hash_owners: dict[str, tuple[str, int]] = {}
    seed_owners: dict[str, tuple[str, int]] = {}
    full_route_owners: dict[str, tuple[str, int]] = {}
    measured_route_owners: dict[str, tuple[str, int]] = {}
    for index, raw in enumerate(traces):
        row = _validate_trace_row(raw, f"traces[{index}]")
        if row["measured_steps"] != measured_steps:
            raise EvidenceError(f"traces[{index}]: measured_steps differs from experiment")
        key = (row["family"], row["replicate"])
        if key in trace_by_key:
            raise EvidenceError(f"traces[{index}]: duplicate trace key {key}")
        if row["trace_sha256"] in trace_hash_owners:
            raise EvidenceError(f"traces[{index}]: trace digest reused by distinct trace")
        if row["seed_sha256"] in seed_owners:
            raise EvidenceError(f"traces[{index}]: seed digest reused by distinct trace")
        if row["full_route_sha256"] in full_route_owners:
            raise EvidenceError(f"traces[{index}]: full route digest reused by distinct trace")
        if row["measured_route_sha256"] in measured_route_owners:
            raise EvidenceError(
                f"traces[{index}]: measured route digest reused by distinct trace"
            )
        trace_by_key[key] = row
        trace_hash_owners[row["trace_sha256"]] = key
        seed_owners[row["seed_sha256"]] = key
        full_route_owners[row["full_route_sha256"]] = key
        measured_route_owners[row["measured_route_sha256"]] = key

    if require_full_matrix:
        expected_traces = {(family, replicate) for family in FAMILIES for replicate in REPLICATES}
        if set(trace_by_key) != expected_traces:
            raise EvidenceError("traces.jsonl: missing or unexpected family/replicate pair")

    observation_by_key: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    per_trace: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(observations):
        row = _validate_observation(raw, f"observations[{index}]")
        if row["measured_steps"] != measured_steps:
            raise EvidenceError(f"observations[{index}]: measured_steps differs from experiment")
        trace_key = (row["family"], row["replicate"])
        trace = trace_by_key.get(trace_key)
        if trace is None:
            raise EvidenceError(f"observations[{index}]: no matching trace ledger row")
        for name in (
            "measured_steps",
            "seed_sha256",
            "full_route_sha256",
            "measured_route_sha256",
            "trace_id",
            "trace_sha256",
        ):
            if row[name] != trace[name]:
                raise EvidenceError(f"observations[{index}].{name}: differs from trace ledger")
        key = (row["family"], row["replicate"], row["capacity_bytes"], row["policy"])
        if key in observation_by_key:
            raise EvidenceError(f"observations[{index}]: duplicate observation key {key}")
        observation_by_key[key] = row
        per_trace[trace_key].append(row)

    if require_full_matrix:
        expected_observations = {
            (family, replicate, capacity, policy)
            for family in FAMILIES
            for replicate in REPLICATES
            for capacity in CAPACITIES
            for policy in POLICIES
        }
        if set(observation_by_key) != expected_observations:
            raise EvidenceError("observations.jsonl: missing or unexpected matrix cell")

    for trace_key, rows in per_trace.items():
        if len(rows) != 18:
            raise EvidenceError(f"trace {trace_key}: expected exactly 18 observations")
        for capacity in CAPACITIES:
            at_capacity = {row["policy"]: row for row in rows if row["capacity_bytes"] == capacity}
            if set(at_capacity) != set(POLICIES):
                raise EvidenceError(f"trace {trace_key}: incomplete policy set at {capacity}")
            belady_bytes = at_capacity["belady"]["metrics"]["total_physical_load_bytes"]
            demand_identity = {
                (
                    row["metrics"]["demand_accesses"],
                    row["metrics"]["demand_logical_bytes"],
                )
                for row in at_capacity.values()
            }
            if len(demand_identity) != 1:
                raise EvidenceError(f"trace {trace_key}: policies saw different demand streams")
            for policy, row in at_capacity.items():
                if policy != "belady" and row["metrics"]["total_physical_load_bytes"] < belady_bytes:
                    raise EvidenceError(f"trace {trace_key}: {policy} beats exact belady")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise EvidenceError("cannot calculate percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise EvidenceError("percentile probability must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def descriptive_statistics(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise EvidenceError("cannot summarize an empty sample")
    checked: list[float] = []
    for value in values:
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise EvidenceError("summary samples must be finite numbers, not booleans")
        checked.append(float(value))
    ordered = sorted(checked)
    return {
        "n": len(checked),
        "mean": float(statistics.fmean(checked)),
        "median": float(statistics.median(checked)),
        "sample_stdev": float(statistics.stdev(checked)) if len(checked) > 1 else 0.0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def bootstrap_median_interval(
    values: Sequence[int | float],
    *,
    label: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Return a deterministic percentile CI using a SHA-256-derived PRNG seed."""

    if not values:
        raise EvidenceError("cannot bootstrap an empty sample")
    if type(resamples) is not int or resamples < 1:
        raise EvidenceError("bootstrap resamples must be a positive integer")
    checked = [float(value) for value in values]
    if any(not math.isfinite(value) for value in checked):
        raise EvidenceError("bootstrap samples must be finite")
    seed_material = b"runnel-m3-bootstrap-median-v1\0" + label.encode("utf-8")
    seed_digest = hashlib.sha256(seed_material).digest()
    seed_sha256 = seed_digest.hex()
    if all(value == checked[0] for value in checked):
        low = high = checked[0]
    else:
        generator = random.Random(int.from_bytes(seed_digest[:16], "big"))
        count = len(checked)
        medians: list[float] = []
        for _ in range(resamples):
            sample = [checked[generator.randrange(count)] for _ in range(count)]
            sample.sort()
            middle = count // 2
            median = sample[middle] if count % 2 else (sample[middle - 1] + sample[middle]) / 2.0
            medians.append(median)
        medians.sort()
        low = _percentile(medians, 0.025)
        high = _percentile(medians, 0.975)
    return {
        "estimator": "median",
        "method": "percentile-bootstrap",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed_sha256": seed_sha256,
        "low": float(low),
        "high": float(high),
    }


def _comparison(
    values: Sequence[float],
    *,
    label: str,
    unit: str,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    return {
        "unit": unit,
        "statistics": descriptive_statistics(values),
        "bootstrap_95": bootstrap_median_interval(values, label=label, resamples=bootstrap_resamples),
    }


def build_summary(
    observations: Sequence[dict[str, Any]],
    *,
    traces_digest: str,
    observations_digest: str,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Aggregate only within preregistered family/capacity/policy cells."""

    by_cell: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    lookup: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in observations:
        key = (row["family"], row["capacity_bytes"], row["policy"])
        by_cell[key].append(row)
        lookup[(row["family"], row["replicate"], row["capacity_bytes"], row["policy"])] = row

    cells: list[dict[str, Any]] = []
    for family in FAMILIES:
        for capacity in CAPACITIES:
            for policy in POLICIES:
                rows = sorted(by_cell.get((family, capacity, policy), []), key=lambda row: row["replicate"])
                if not rows:
                    continue
                metric_summary = {
                    name: descriptive_statistics([row["metrics"][name] for row in rows])
                    for name in METRIC_NAMES
                }
                ratios_lru: list[float] = []
                differences_lru: list[float] = []
                ratios_belady: list[float] = []
                demand_ratios_belady: list[float] = []
                for row in rows:
                    replicate = row["replicate"]
                    lru = lookup.get((family, replicate, capacity, "lru"))
                    belady = lookup.get((family, replicate, capacity, "belady"))
                    if lru is None or belady is None:
                        raise EvidenceError("summary pairing is incomplete")
                    candidate_bytes = row["metrics"]["total_physical_load_bytes"]
                    lru_bytes = lru["metrics"]["total_physical_load_bytes"]
                    belady_bytes = belady["metrics"]["total_physical_load_bytes"]
                    if lru_bytes <= 0 or belady_bytes <= 0:
                        raise EvidenceError("summary comparison denominator must be positive")
                    ratios_lru.append(candidate_bytes / lru_bytes)
                    differences_lru.append(float(candidate_bytes - lru_bytes))
                    ratios_belady.append(candidate_bytes / belady_bytes)
                    demand_ratios_belady.append(row["metrics"]["demand_load_bytes"] / belady_bytes)
                label_prefix = f"{family}\0{capacity}\0{policy}"
                ratio_lru = _comparison(
                    ratios_lru,
                    label=f"{label_prefix}\0total-physical-vs-lru-ratio",
                    unit="ratio; lower is better",
                    bootstrap_resamples=bootstrap_resamples,
                )
                difference_lru = _comparison(
                    differences_lru,
                    label=f"{label_prefix}\0total-physical-vs-lru-difference-bytes",
                    unit="bytes; lower is better",
                    bootstrap_resamples=bootstrap_resamples,
                )
                gap = _comparison(
                    ratios_belady,
                    label=f"{label_prefix}\0total-physical-over-belady",
                    unit="ratio; 1 is optimal",
                    bootstrap_resamples=bootstrap_resamples,
                )
                diagnostic_gap = _comparison(
                    demand_ratios_belady,
                    label=f"{label_prefix}\0demand-load-over-belady",
                    unit="diagnostic ratio; prefetch can shift I/O out of demand",
                    bootstrap_resamples=bootstrap_resamples,
                )
                ci = ratio_lru["bootstrap_95"]
                if policy == "lru":
                    comparison_role = "baseline"
                    interval_position = "not_applicable"
                elif policy == "belady":
                    comparison_role = "oracle"
                    interval_position = "not_applicable"
                elif ci["high"] < 1.0:
                    comparison_role = "online-candidate"
                    interval_position = "below_one"
                elif ci["low"] > 1.0:
                    comparison_role = "online-candidate"
                    interval_position = "above_one"
                else:
                    comparison_role = "online-candidate"
                    interval_position = "overlaps_one"
                cells.append(
                    {
                        "family": family,
                        "capacity_bytes": capacity,
                        "policy": policy,
                        "comparison_role": comparison_role,
                        "interpretation": "exploratory-only",
                        "n": len(rows),
                        "metrics": metric_summary,
                        "paired_vs_lru": {
                            "total_physical_ratio": ratio_lru,
                            "total_physical_difference_bytes": difference_lru,
                            "interval_position_vs_one": interval_position,
                        },
                        "online_to_optimal": {
                            "total_physical_over_belady": gap,
                            "demand_load_over_belady_diagnostic": diagnostic_gap,
                        },
                    }
                )
    return {
        "schema": SUMMARY_SCHEMA,
        "source": {
            "traces": traces_digest,
            "observations": observations_digest,
            "trace_rows": len({(row["family"], row["replicate"]) for row in observations}),
            "observation_rows": len(observations),
        },
        "methodology": {
            **ANALYSIS_CONTRACT,
            "bootstrap_resamples": bootstrap_resamples,
            "page_bytes": PAGE_BYTES,
        },
        "cells": cells,
    }


_PALETTE = {
    "lru": "#0072B2",
    "slru": "#E69F00",
    "tiny-lfu": "#009E73",
    "router-admit": "#CC79A7",
    "router-prefetch": "#D55E00",
    "belady": "#4D4D4D",
}
_MARKERS = {
    "lru": "circle",
    "slru": "square",
    "tiny-lfu": "triangle",
    "router-admit": "diamond",
    "router-prefetch": "cross",
    "belady": "plus",
}


def _svg_header(title: str, description: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{html.escape(title)}</title>",
        f"<desc id=\"desc\">{html.escape(description)}</desc>",
        "<style>text{font-family:system-ui,sans-serif;fill:#222}.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#444}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{font-size:12px}.small{font-size:10px;fill:#555}.ci{stroke-width:2}.mark{stroke:#222;stroke-width:1}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]


def _svg_marker(kind: str, x: float, y: float, color: str) -> str:
    if kind == "circle":
        return f'<circle class="mark" cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>'
    if kind == "square":
        return f'<rect class="mark" x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" fill="{color}"/>'
    if kind == "triangle":
        return f'<path class="mark" d="M{x:.2f},{y - 6:.2f} L{x - 6:.2f},{y + 5:.2f} L{x + 6:.2f},{y + 5:.2f} Z" fill="{color}"/>'
    if kind == "diamond":
        return f'<path class="mark" d="M{x:.2f},{y - 7:.2f} L{x - 7:.2f},{y:.2f} L{x:.2f},{y + 7:.2f} L{x + 7:.2f},{y:.2f} Z" fill="{color}"/>'
    if kind == "cross":
        return f'<path class="mark" d="M{x - 5:.2f},{y - 5:.2f} L{x + 5:.2f},{y + 5:.2f} M{x + 5:.2f},{y - 5:.2f} L{x - 5:.2f},{y + 5:.2f}" fill="none" stroke="{color}" stroke-width="3"/>'
    return f'<path class="mark" d="M{x - 6:.2f},{y:.2f} L{x + 6:.2f},{y:.2f} M{x:.2f},{y - 6:.2f} L{x:.2f},{y + 6:.2f}" fill="none" stroke="{color}" stroke-width="3"/>'


def _summary_cell_lookup(
    summary: Mapping[str, Any],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise EvidenceError("figure source must be an M3 summary")
    cells = _array(summary.get("cells"), "summary.cells")
    lookup: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for index, value in enumerate(cells):
        cell = _object(value, f"summary.cells[{index}]")
        key = (cell.get("family"), cell.get("capacity_bytes"), cell.get("policy"))
        if key in lookup:
            raise EvidenceError(f"summary.cells[{index}]: duplicate figure cell {key}")
        lookup[key] = cell
    return lookup


def _cell_comparisons(
    summary: Mapping[str, Any], capacity: int
) -> dict[tuple[str, str], tuple[float, float, float]]:
    lookup = _summary_cell_lookup(summary)
    output: dict[tuple[str, str], tuple[float, float, float]] = {}
    for family in FAMILIES:
        for policy in POLICIES:
            cell = lookup.get((family, capacity, policy))
            if cell is None:
                continue
            comparison = cell["online_to_optimal"]["total_physical_over_belady"]
            statistics_object = comparison["statistics"]
            interval = comparison["bootstrap_95"]
            output[(family, policy)] = (
                float(statistics_object["median"]),
                float(interval["low"]),
                float(interval["high"]),
            )
    return output


def render_optimal_gap_svg(
    summary: Mapping[str, Any],
) -> bytes:
    width, height = 1220, 650
    capacity = CAPACITIES[1]
    comparisons = _cell_comparisons(summary, capacity)
    maximum = max((high for _, _, high in comparisons.values()), default=1.0)
    maximum = max(1.25, maximum * 1.05)
    left, right, top = 190.0, 1175.0, 115.0
    plot_width = right - left
    lines = _svg_header(
        "Online traffic gap to Bélády/MIN",
        "Median total physical byte ratio and 95 percent bootstrap interval for each policy and workload family at four MiB capacity.",
        width,
        height,
    )
    lines += [
        '<text class="title" x="35" y="36">Online traffic gap to Bélády/MIN</text>',
        '<text class="subtitle" x="35" y="60">Total physical bytes / exact uniform-page optimum · 4 MiB capacity · lower is better</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        x = left + plot_width * value / maximum
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="92" x2="{x:.2f}" y2="565"/>')
        lines.append(f'<text class="small" x="{x:.2f}" y="585" text-anchor="middle">{value:.2f}×</text>')
    optimal_x = left + plot_width / maximum
    lines.append(f'<line x1="{optimal_x:.2f}" y1="92" x2="{optimal_x:.2f}" y2="565" stroke="#000" stroke-dasharray="5 4"/>')
    online = POLICIES[:-1]
    for family_index, family in enumerate(FAMILIES):
        base_y = top + family_index * 75
        lines.append(f'<text class="label" x="180" y="{base_y + 10:.2f}" text-anchor="end">{html.escape(FAMILY_LABELS[family])}</text>')
        for policy_index, policy in enumerate(online):
            result = comparisons.get((family, policy))
            if result is None:
                continue
            median, low, high = result
            y = base_y - 14 + policy_index * 12
            x1 = left + plot_width * low / maximum
            x2 = left + plot_width * high / maximum
            x = left + plot_width * median / maximum
            color = _PALETTE[policy]
            lines.append(f'<line class="ci" x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="{color}"/>')
            lines.append(_svg_marker(_MARKERS[policy], x, y, color))
    for index, policy in enumerate(online):
        x = 190 + index * 185
        lines.append(_svg_marker(_MARKERS[policy], x, 615, _PALETTE[policy]))
        lines.append(f'<text class="small" x="{x + 12}" y="619">{html.escape(POLICY_LABELS[policy])}</text>')
    lines.append('<text class="small" x="35" y="642">Source: summary.json regenerated from observations.jsonl · exploratory unadjusted cells · no omnibus claim or host timing</text>')
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_paired_change_svg(summary: Mapping[str, Any]) -> bytes:
    width, height = 1220, 1000
    lookup = _summary_cell_lookup(summary)
    policies = POLICIES[1:5]
    entries: list[tuple[str, int, str, float, float, float]] = []
    for family in FAMILIES:
        for capacity in CAPACITIES:
            for policy in policies:
                cell = lookup.get((family, capacity, policy))
                if cell is not None:
                    comparison = cell["paired_vs_lru"]["total_physical_ratio"]
                    interval = comparison["bootstrap_95"]
                    entries.append(
                        (
                            family,
                            capacity,
                            policy,
                            float(comparison["statistics"]["median"]),
                            float(interval["low"]),
                            float(interval["high"]),
                        )
                    )
    low_bound = min((low for _, _, _, _, low, _ in entries), default=0.8)
    high_bound = max((high for _, _, _, _, _, high in entries), default=1.2)
    span = max(0.1, high_bound - low_bound)
    lower = min(0.95, low_bound - span * 0.08)
    upper = max(1.05, high_bound + span * 0.08)
    left, right = 255.0, 1175.0
    plot_width = right - left
    lines = _svg_header(
        "Paired traffic change versus LRU",
        "Exploratory median paired total physical byte ratios with unadjusted 95 percent bootstrap intervals, grouped by workload and capacity; policy colors and markers are identified in the legend.",
        width,
        height,
    )
    lines += [
        '<text class="title" x="35" y="36">Paired traffic change versus LRU</text>',
        '<text class="subtitle" x="35" y="60">Candidate total physical bytes / LRU · exploratory unadjusted cells · lower is better</text>',
    ]
    for tick in range(6):
        value = lower + (upper - lower) * tick / 5
        x = left + plot_width * (value - lower) / (upper - lower)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="82" x2="{x:.2f}" y2="885"/>')
        lines.append(f'<text class="small" x="{x:.2f}" y="905" text-anchor="middle">{value:.2f}×</text>')
    baseline_x = left + plot_width * (1.0 - lower) / (upper - lower)
    lines.append(f'<line x1="{baseline_x:.2f}" y1="82" x2="{baseline_x:.2f}" y2="885" stroke="#000" stroke-dasharray="5 4"/>')
    entry_lookup = {(family, capacity, policy): (median, low, high) for family, capacity, policy, median, low, high in entries}
    for family_index, family in enumerate(FAMILIES):
        family_y = 105 + family_index * 130
        lines.append(f'<text class="label" x="150" y="{family_y + 54}" text-anchor="end">{html.escape(FAMILY_LABELS[family])}</text>')
        for capacity_index, capacity in enumerate(CAPACITIES):
            row_y = family_y + capacity_index * 42
            lines.append(
                f'<text class="small" x="240" y="{row_y + 4:.2f}" text-anchor="end">{capacity // (1024 * 1024)} MiB</text>'
            )
            for policy_index, policy in enumerate(policies):
                result = entry_lookup.get((family, capacity, policy))
                if result is None:
                    continue
                median, low, high = result
                y = row_y + (-15, -5, 5, 15)[policy_index]
                x1 = left + plot_width * (low - lower) / (upper - lower)
                x2 = left + plot_width * (high - lower) / (upper - lower)
                x = left + plot_width * (median - lower) / (upper - lower)
                color = _PALETTE[policy]
                lines.append(f'<line class="ci" x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="{color}"/>')
                lines.append(_svg_marker(_MARKERS[policy], x, y, color))
    for index, policy in enumerate(policies):
        x = 205 + index * 240
        lines.append(_svg_marker(_MARKERS[policy], x, 940, _PALETTE[policy]))
        lines.append(
            f'<text class="small" x="{x + 13}" y="944">{html.escape(POLICY_LABELS[policy])}</text>'
        )
    lines.append('<text class="small" x="35" y="982">Source: summary.json regenerated from observations.jsonl · exploratory unadjusted cells · no omnibus claim or host timing</text>')
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_prefetch_accounting_svg(summary: Mapping[str, Any]) -> bytes:
    width, height = 1220, 650
    capacity = CAPACITIES[1]
    categories = (
        ("prefetch_useful_bytes", "Useful", "#0072B2", "none"),
        ("prefetch_wasted_bytes", "Wasted", "#D55E00", "diag"),
        ("prefetch_redundant_bytes", "Redundant (no read)", "#009E73", "dots"),
        ("prefetch_dropped_bytes", "Dropped (no read)", "#CC79A7", "cross"),
    )
    medians: dict[tuple[str, str], float] = {}
    lookup = _summary_cell_lookup(summary)
    for family in FAMILIES:
        cell = lookup.get((family, capacity, "router-prefetch"))
        for metric, _, _, _ in categories:
            if cell is not None:
                medians[(family, metric)] = float(cell["metrics"][metric]["median"])
    maximum = max(medians.values(), default=1.0)
    maximum = max(maximum, 1.0)
    left, right, top, bottom = 120.0, 1170.0, 100.0, 545.0
    plot_height = bottom - top
    group_width = (right - left) / len(FAMILIES)
    bar_width = group_width / 6
    lines = _svg_header(
        "Router-prefetch byte accounting",
        "Median useful, wasted, redundant, and dropped prefetch bytes by workload at four MiB capacity.",
        width,
        height,
    )
    lines += [
        '<defs><pattern id="diag" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#fff" stroke-width="2"/></pattern><pattern id="dots" width="8" height="8" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.5" fill="#fff"/></pattern><pattern id="cross" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M1,1 L7,7 M7,1 L1,7" stroke="#fff" stroke-width="1.5"/></pattern></defs>',
        '<text class="title" x="35" y="36">Router-prefetch byte accounting</text>',
        '<text class="subtitle" x="35" y="60">Median classified bytes · 4 MiB capacity · redundant and dropped offers perform no physical read</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - plot_height * tick / 5
        lines.append(f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}"/>')
        lines.append(f'<text class="small" x="{left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end">{value / (1024 * 1024):.1f} MiB</text>')
    for family_index, family in enumerate(FAMILIES):
        center = left + group_width * (family_index + 0.5)
        for category_index, (metric, _, color, pattern) in enumerate(categories):
            value = medians.get((family, metric), 0.0)
            height_value = plot_height * value / maximum
            x = center + (category_index - 1.5) * bar_width
            y = bottom - height_value
            lines.append(f'<rect class="mark" x="{x - bar_width * 0.42:.2f}" y="{y:.2f}" width="{bar_width * 0.84:.2f}" height="{height_value:.2f}" fill="{color}"/>')
            if pattern != "none" and height_value > 0:
                lines.append(f'<rect x="{x - bar_width * 0.42:.2f}" y="{y:.2f}" width="{bar_width * 0.84:.2f}" height="{height_value:.2f}" fill="url(#{pattern})"/>')
        lines.append(f'<text class="small" x="{center:.2f}" y="{bottom + 19:.2f}" text-anchor="middle">{html.escape(FAMILY_LABELS[family])}</text>')
    for index, (_, label, color, pattern) in enumerate(categories):
        x = 145 + index * 250
        lines.append(f'<rect class="mark" x="{x}" y="585" width="18" height="12" fill="{color}"/>')
        if pattern != "none":
            lines.append(f'<rect x="{x}" y="585" width="18" height="12" fill="url(#{pattern})"/>')
        lines.append(f'<text class="small" x="{x + 25}" y="596">{html.escape(label)}</text>')
    lines.append('<text class="small" x="35" y="635">Source: summary.json regenerated from observations.jsonl · byte classifications are separate from cache hits · exploratory, no host timing</text>')
    lines.append("</svg>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_figures(summary: Mapping[str, Any]) -> dict[str, bytes]:
    """Generate all figures from the one summary derived from raw observations."""

    return {
        "figures/optimal-gap.svg": render_optimal_gap_svg(summary),
        "figures/paired-change.svg": render_paired_change_svg(summary),
        "figures/prefetch-accounting.svg": render_prefetch_accounting_svg(summary),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_timestamp(value: Any, context: str) -> str:
    timestamp = _string(value, context, maximum=32)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp):
        raise EvidenceError(f"{context}: expected second-resolution RFC 3339 UTC")
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise EvidenceError(f"{context}: invalid UTC timestamp") from error
    return timestamp


def _relative_path(value: Any, context: str) -> str:
    path = _string(value, context, maximum=256)
    if "\\" in path or path.startswith("/"):
        raise EvidenceError(f"{context}: expected repository-relative POSIX path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise EvidenceError(f"{context}: traversal or empty path component")
    return path


def _memory_snapshot() -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable"}:
                fields = remainder.split()
                if len(fields) == 2 and fields[1] == "kB" and fields[0].isdigit():
                    values[name] = int(fields[0]) * 1024
    except OSError:
        pass
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator and name.strip() in {"model name", "Hardware"} and value.strip():
                return value.strip()[:256]
    except OSError:
        pass
    return "unavailable"


def build_environment(
    *,
    binary_sha256: str,
    harness_sha256: str,
    captured_at: str,
    cargo_version: str,
    rustc_version: str,
    build_root_available_bytes: int,
) -> dict[str, Any]:
    stat = os.statvfs(ROOT)
    total_memory, available_memory = _memory_snapshot()
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "harness_schema": HARNESS_SCHEMA,
        "captured_at_utc": captured_at,
        "operating_system": {
            "system": platform.system() or "unavailable",
            "release": platform.release() or "unavailable",
            "machine": platform.machine() or "unavailable",
        },
        "cpu": {
            "logical_count": os.cpu_count() or 0,
            "model": _cpu_model(),
        },
        "memory": {
            "total_bytes": total_memory,
            "available_bytes": available_memory,
        },
        "filesystem": {
            "repository_free_bytes": stat.f_bavail * stat.f_frsize,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "toolchain": {
            "cargo": cargo_version,
            "rustc": rustc_version,
        },
        "build_filesystem": {
            "kind": "tmpfs",
            "available_bytes_before": build_root_available_bytes,
        },
        "controlled_environment": dict(CONTROLLED_ENVIRONMENT),
        "harness": {
            "path": "scripts/run_m3_experiment.py",
            "sha256": f"sha256:{harness_sha256}",
        },
        "binary": {
            "path": BINARY_LOGICAL_PATH,
            "sha256": f"sha256:{binary_sha256}",
        },
    }


def _validate_environment(value: Any) -> dict[str, Any]:
    env = _object(value, "environment")
    _exact_keys(
        env,
        (
            "schema",
            "harness_schema",
            "captured_at_utc",
            "operating_system",
            "cpu",
            "memory",
            "filesystem",
            "python",
            "toolchain",
            "build_filesystem",
            "controlled_environment",
            "harness",
            "binary",
        ),
        "environment",
    )
    _string(env["schema"], "environment.schema", expected=ENVIRONMENT_SCHEMA)
    _string(env["harness_schema"], "environment.harness_schema", expected=HARNESS_SCHEMA)
    _validate_timestamp(env["captured_at_utc"], "environment.captured_at_utc")
    operating_system = _object(env["operating_system"], "environment.operating_system")
    _exact_keys(operating_system, ("system", "release", "machine"), "environment.operating_system")
    for name in ("system", "release", "machine"):
        _string(operating_system[name], f"environment.operating_system.{name}", maximum=256)
    cpu = _object(env["cpu"], "environment.cpu")
    _exact_keys(cpu, ("logical_count", "model"), "environment.cpu")
    _uint(cpu["logical_count"], "environment.cpu.logical_count", 1 << 20)
    _string(cpu["model"], "environment.cpu.model", maximum=256)
    memory = _object(env["memory"], "environment.memory")
    _exact_keys(memory, ("total_bytes", "available_bytes"), "environment.memory")
    total = _uint(memory["total_bytes"], "environment.memory.total_bytes")
    available = _uint(memory["available_bytes"], "environment.memory.available_bytes")
    if total and available > total:
        raise EvidenceError("environment.memory: available bytes exceed total")
    filesystem = _object(env["filesystem"], "environment.filesystem")
    _exact_keys(filesystem, ("repository_free_bytes",), "environment.filesystem")
    _uint(filesystem["repository_free_bytes"], "environment.filesystem.repository_free_bytes")
    python = _object(env["python"], "environment.python")
    _exact_keys(python, ("implementation", "version"), "environment.python")
    _string(python["implementation"], "environment.python.implementation", maximum=64)
    _string(python["version"], "environment.python.version", maximum=64)
    toolchain = _object(env["toolchain"], "environment.toolchain")
    _exact_keys(toolchain, ("cargo", "rustc"), "environment.toolchain")
    _string(toolchain["cargo"], "environment.toolchain.cargo", maximum=256)
    _string(toolchain["rustc"], "environment.toolchain.rustc", maximum=256)
    build_filesystem = _object(env["build_filesystem"], "environment.build_filesystem")
    _exact_keys(
        build_filesystem,
        ("kind", "available_bytes_before"),
        "environment.build_filesystem",
    )
    _string(
        build_filesystem["kind"],
        "environment.build_filesystem.kind",
        expected="tmpfs",
    )
    available_bytes_before = _uint(
        build_filesystem["available_bytes_before"],
        "environment.build_filesystem.available_bytes_before",
    )
    if available_bytes_before < MIN_BUILD_ROOT_FREE_BYTES:
        raise EvidenceError(
            "environment.build_filesystem.available_bytes_before: "
            "expected at least one GiB"
        )
    controlled = _object(env["controlled_environment"], "environment.controlled_environment")
    _exact_keys(controlled, tuple(CONTROLLED_ENVIRONMENT), "environment.controlled_environment")
    for name, expected in CONTROLLED_ENVIRONMENT.items():
        _string(controlled[name], f"environment.controlled_environment.{name}", expected=expected)
    for field in ("harness", "binary"):
        item = _object(env[field], f"environment.{field}")
        _exact_keys(item, ("path", "sha256"), f"environment.{field}")
        _relative_path(item["path"], f"environment.{field}.path")
        _prefixed_digest(item["sha256"], f"environment.{field}.sha256")
    if env["harness"]["path"] != "scripts/run_m3_experiment.py":
        raise EvidenceError("environment.harness.path: unexpected harness path")
    if env["binary"]["path"] != BINARY_LOGICAL_PATH:
        raise EvidenceError("environment.binary.path: unexpected logical build path")
    return env


def build_experiment(
    *,
    captured_at: str,
    commit: str,
    binary_sha256: str,
    measured_steps: int,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_SCHEMA,
        "harness_schema": HARNESS_SCHEMA,
        "captured_at_utc": captured_at,
        "git_commit": commit,
        "tracked_worktree_clean": True,
        "analysis_question": ANALYSIS_QUESTION,
        "baseline": "byte-capacity-lru",
        "oracle": "exact-belady-min-uniform-65536-byte-pages",
        "build": {
            "argv": list(BUILD_COMMAND),
            "environment": dict(BUILD_RECORDED_ENVIRONMENT),
            "package": "runnel-sim",
            "binary": "runnel-cache-sim",
            "profile": "release",
            "features": "workspace-defaults",
            "timeout_seconds": BUILD_TIMEOUT_SECONDS,
            "binary_sha256": f"sha256:{binary_sha256}",
        },
        "commands": {
            "generate": [
                BINARY_LOGICAL_PATH,
                "generate",
                "--family",
                "{family}",
                "--replicate",
                "{replicate}",
                "--measured-steps",
                str(measured_steps),
            ],
            "matrix": [
                BINARY_LOGICAL_PATH,
                "matrix",
                "--family",
                "{family}",
                "--replicate",
                "{replicate}",
                "--measured-steps",
                str(measured_steps),
            ],
        },
        "trace_contract": copy.deepcopy(TRACE_CONTRACT),
        "policy_contract": copy.deepcopy(POLICY_CONTRACT),
        "matrix_contract": {
            "families": list(FAMILIES),
            "replicates": list(REPLICATES),
            "capacities_bytes": list(CAPACITIES),
            "policies": list(POLICIES),
            "measured_steps": measured_steps,
            "page_bytes": PAGE_BYTES,
            "trace_rows": 180,
            "observation_rows": 3_240,
            "generate_subprocesses": 180,
            "matrix_subprocesses": 180,
            "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            "generated_trace_stdout_limit_bytes": MAX_GENERATED_TRACE_BYTES,
            "matrix_stdout_limit_bytes": MAX_STDOUT_BYTES,
            "stderr_limit_bytes": MAX_STDERR_BYTES,
            "directory_limit_bytes": MAX_DIRECTORY_BYTES,
        },
        "analysis_contract": copy.deepcopy(ANALYSIS_CONTRACT),
        "artifacts": dict(artifacts),
    }


def _exact_scalar_array(value: Any, expected: Sequence[Any], context: str) -> None:
    array = _array(value, context)
    if len(array) != len(expected):
        raise EvidenceError(f"{context}: expected {len(expected)} elements")
    for index, (actual, wanted) in enumerate(zip(array, expected, strict=True)):
        if type(actual) is not type(wanted) or actual != wanted:
            raise EvidenceError(f"{context}[{index}]: expected {wanted!r}")


def _exact_value(value: Any, expected: Any, context: str) -> None:
    if type(value) is not type(expected):
        raise EvidenceError(f"{context}: expected {type(expected).__name__}")
    if type(expected) is dict:
        _exact_keys(value, tuple(expected), context)
        for name, expected_value in expected.items():
            _exact_value(value[name], expected_value, f"{context}.{name}")
    elif type(expected) is list:
        if len(value) != len(expected):
            raise EvidenceError(f"{context}: expected {len(expected)} elements")
        for index, (actual, expected_value) in enumerate(zip(value, expected, strict=True)):
            _exact_value(actual, expected_value, f"{context}[{index}]")
    elif value != expected:
        raise EvidenceError(f"{context}: expected {expected!r}")


def _validate_experiment(value: Any) -> dict[str, Any]:
    experiment = _object(value, "experiment")
    _exact_keys(
        experiment,
        (
            "schema",
            "harness_schema",
            "captured_at_utc",
            "git_commit",
            "tracked_worktree_clean",
            "analysis_question",
            "baseline",
            "oracle",
            "build",
            "commands",
            "trace_contract",
            "policy_contract",
            "matrix_contract",
            "analysis_contract",
            "artifacts",
        ),
        "experiment",
    )
    _string(experiment["schema"], "experiment.schema", expected=EXPERIMENT_SCHEMA)
    _string(experiment["harness_schema"], "experiment.harness_schema", expected=HARNESS_SCHEMA)
    _validate_timestamp(experiment["captured_at_utc"], "experiment.captured_at_utc")
    commit = _string(experiment["git_commit"], "experiment.git_commit", maximum=40)
    if not COMMIT_RE.fullmatch(commit):
        raise EvidenceError("experiment.git_commit: expected full lowercase commit ID")
    if not _boolean(experiment["tracked_worktree_clean"], "experiment.tracked_worktree_clean"):
        raise EvidenceError("experiment.tracked_worktree_clean: must be true")
    _string(
        experiment["analysis_question"],
        "experiment.analysis_question",
        expected=ANALYSIS_QUESTION,
        maximum=512,
    )
    _string(experiment["baseline"], "experiment.baseline", expected="byte-capacity-lru")
    _string(experiment["oracle"], "experiment.oracle", expected="exact-belady-min-uniform-65536-byte-pages")
    build = _object(experiment["build"], "experiment.build")
    expected_build = build_experiment(
        captured_at=experiment["captured_at_utc"],
        commit=commit,
        binary_sha256="0" * 64,
        measured_steps=DEFAULT_MEASURED_STEPS,
        artifacts={},
    )["build"]
    _exact_keys(build, tuple(expected_build), "experiment.build")
    binary_digest = _prefixed_digest(
        build["binary_sha256"], "experiment.build.binary_sha256"
    )
    expected_build["binary_sha256"] = binary_digest
    _exact_value(build, expected_build, "experiment.build")
    commands = _object(experiment["commands"], "experiment.commands")
    expected_commands = build_experiment(
        captured_at=experiment["captured_at_utc"],
        commit=commit,
        binary_sha256=binary_digest[7:],
        measured_steps=DEFAULT_MEASURED_STEPS,
        artifacts={},
    )["commands"]
    _exact_value(commands, expected_commands, "experiment.commands")
    _exact_value(experiment["trace_contract"], TRACE_CONTRACT, "experiment.trace_contract")
    _exact_value(experiment["policy_contract"], POLICY_CONTRACT, "experiment.policy_contract")
    contract = _object(experiment["matrix_contract"], "experiment.matrix_contract")
    _exact_keys(
        contract,
        (
            "families",
            "replicates",
            "capacities_bytes",
            "policies",
            "measured_steps",
            "page_bytes",
            "trace_rows",
            "observation_rows",
            "generate_subprocesses",
            "matrix_subprocesses",
            "timeout_seconds",
            "generated_trace_stdout_limit_bytes",
            "matrix_stdout_limit_bytes",
            "stderr_limit_bytes",
            "directory_limit_bytes",
        ),
        "experiment.matrix_contract",
    )
    _exact_scalar_array(contract["families"], FAMILIES, "experiment.matrix_contract.families")
    _exact_scalar_array(contract["replicates"], REPLICATES, "experiment.matrix_contract.replicates")
    _exact_scalar_array(contract["capacities_bytes"], CAPACITIES, "experiment.matrix_contract.capacities_bytes")
    _exact_scalar_array(contract["policies"], POLICIES, "experiment.matrix_contract.policies")
    measured_steps = _uint(contract["measured_steps"], "experiment.matrix_contract.measured_steps")
    if measured_steps != DEFAULT_MEASURED_STEPS:
        raise EvidenceError(
            f"experiment.matrix_contract.measured_steps: primary evidence requires {DEFAULT_MEASURED_STEPS}"
        )
    expected_contract = {
        "page_bytes": PAGE_BYTES,
        "trace_rows": 180,
        "observation_rows": 3_240,
        "generate_subprocesses": 180,
        "matrix_subprocesses": 180,
        "generated_trace_stdout_limit_bytes": MAX_GENERATED_TRACE_BYTES,
        "matrix_stdout_limit_bytes": MAX_STDOUT_BYTES,
        "stderr_limit_bytes": MAX_STDERR_BYTES,
        "directory_limit_bytes": MAX_DIRECTORY_BYTES,
    }
    for name, expected in expected_contract.items():
        if _uint(contract[name], f"experiment.matrix_contract.{name}") != expected:
            raise EvidenceError(f"experiment.matrix_contract.{name}: expected {expected}")
    timeout = contract["timeout_seconds"]
    if type(timeout) not in {int, float} or float(timeout) != DEFAULT_TIMEOUT_SECONDS:
        raise EvidenceError("experiment.matrix_contract.timeout_seconds: expected 30 seconds")
    _exact_value(
        experiment["analysis_contract"],
        ANALYSIS_CONTRACT,
        "experiment.analysis_contract",
    )
    artifacts = _object(experiment["artifacts"], "experiment.artifacts")
    artifact_names = sorted(EXPECTED_FILES - {"experiment.json"})
    _exact_keys(artifacts, artifact_names, "experiment.artifacts")
    for name in artifact_names:
        _prefixed_digest(artifacts[name], f"experiment.artifacts.{name}")
    return experiment


def _decode_mount_field(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _filesystem_type(path: Path) -> str:
    best_length = -1
    best_type = ""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvidenceError("cannot inspect build-root filesystem type") from error
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if len(fields) <= separator + 1 or len(fields) <= 4:
            continue
        mount_point = Path(_decode_mount_field(fields[4]))
        try:
            path.relative_to(mount_point)
        except ValueError:
            continue
        mount_length = len(mount_point.as_posix())
        if mount_length > best_length:
            best_length = mount_length
            best_type = fields[separator + 1]
    if not best_type:
        raise EvidenceError("cannot resolve build-root mount")
    return best_type


def _resolve_build_root(value: str) -> tuple[Path, int]:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise EvidenceError("--build-root must be an absolute tmpfs directory")
    try:
        metadata = os.lstat(candidate)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise EvidenceError("--build-root does not exist") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EvidenceError("--build-root must be a regular non-symlink directory")
    try:
        resolved.relative_to(ROOT.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise EvidenceError("--build-root must not be inside the repository")
    if _filesystem_type(resolved) != "tmpfs":
        raise EvidenceError("--build-root must reside on tmpfs")
    filesystem = os.statvfs(resolved)
    available = filesystem.f_bavail * filesystem.f_frsize
    if available < MIN_BUILD_ROOT_FREE_BYTES:
        raise EvidenceError("--build-root requires at least one GiB free")
    return resolved, available


def _build_environment(target: Path) -> dict[str, str]:
    environment = {
        name: value
        for name in ("PATH", "HOME", "CARGO_HOME", "RUSTUP_HOME")
        if (value := os.environ.get(name)) is not None
    }
    environment.update(CONTROLLED_ENVIRONMENT)
    environment.update(
        {
            "CARGO_BUILD_JOBS": "2",
            "CARGO_INCREMENTAL": "0",
            "CARGO_TARGET_DIR": str(target),
            "CARGO_TERM_COLOR": "never",
        }
    )
    return environment


def _tool_version(command: Sequence[str], environment: Mapping[str, str]) -> str:
    result = run_bounded_process(
        command,
        cwd=ROOT,
        environment=environment,
        timeout_seconds=30,
        maximum_stdout_bytes=4 * 1024,
        maximum_stderr_bytes=4 * 1024,
    )
    if (
        result.launch_error is not None
        or result.timed_out
        or result.stdout_exceeded
        or result.stderr_exceeded
        or result.return_code != 0
        or result.stderr
    ):
        raise EvidenceError(f"cannot record {' '.join(command)}")
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{' '.join(command)} returned non-ASCII output") from error
    if not value or "\n" in value or len(value) > 256:
        raise EvidenceError(f"{' '.join(command)} returned an invalid version")
    return value


def _build_release_binary(build_root_value: str) -> PrivateBuild:
    build_root, available = _resolve_build_root(build_root_value)
    private_root = Path(tempfile.mkdtemp(prefix=".runnel-m3-build-", dir=build_root))
    os.chmod(private_root, 0o700)
    target = private_root / "target"
    environment = _build_environment(target)
    try:
        cargo_version = _tool_version(("cargo", "--version"), environment)
        rustc_version = _tool_version(("rustc", "--version"), environment)
        result = run_bounded_process(
            BUILD_COMMAND,
            cwd=ROOT,
            environment=environment,
            timeout_seconds=BUILD_TIMEOUT_SECONDS,
            maximum_stdout_bytes=MAX_STDERR_BYTES,
            maximum_stderr_bytes=MAX_STDERR_BYTES,
        )
        if result.launch_error is not None or result.timed_out:
            raise EvidenceError("locked offline release build could not complete")
        if result.stdout_exceeded or result.stderr_exceeded:
            raise EvidenceError("locked offline release build output exceeded one MiB")
        if result.return_code != 0:
            raise EvidenceError("locked offline release build failed")
        binary = target / "release" / "runnel-cache-sim"
        metadata = os.lstat(binary)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError("release build did not produce a regular executable")
        if not os.access(binary, os.X_OK):
            raise EvidenceError("release build output is not executable")
        return PrivateBuild(
            root=private_root,
            binary=binary,
            binary_sha256=sha256_file(binary),
            cargo_version=cargo_version,
            rustc_version=rustc_version,
            build_root_available_bytes=available,
        )
    except BaseException:
        shutil.rmtree(private_root, ignore_errors=True)
        raise


def _resolve_new_output(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if os.path.lexists(candidate):
        raise EvidenceError("capture output must initially be absent")
    parent = candidate.parent.resolve(strict=True)
    try:
        parent.relative_to(ROOT.resolve(strict=True))
    except ValueError as error:
        raise EvidenceError("capture output must remain inside the repository") from error
    if not parent.is_dir() or parent.is_symlink():
        raise EvidenceError("capture output parent must be a regular repository directory")
    resolved = parent / candidate.name
    output_base = (ROOT / "benchmarks" / "raw").resolve(strict=True)
    if parent != output_base:
        raise EvidenceError("capture output must be one new directory directly under benchmarks/raw")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]{0,94}[a-z0-9])?", candidate.name) or ".." in candidate.name:
        raise EvidenceError("capture output requires a portable 1-96 character experiment ID")
    return resolved


def _require_capture_disk_reserve() -> None:
    stat = os.statvfs(ROOT)
    available = stat.f_bavail * stat.f_frsize
    required = MIN_FREE_BYTES_AFTER_CAPTURE + MAX_DIRECTORY_BYTES
    if available < required:
        raise EvidenceError(
            "capture requires at least 2 GiB free after reserving the 16 MiB evidence cap"
        )


def _git_output(arguments: Sequence[str]) -> str:
    result = run_bounded_process(
        ("git", *arguments),
        cwd=ROOT,
        environment=CONTROLLED_ENVIRONMENT,
        timeout_seconds=30,
        maximum_stdout_bytes=MAX_GIT_OUTPUT_BYTES,
        maximum_stderr_bytes=MAX_GIT_OUTPUT_BYTES,
    )
    if result.launch_error is not None or result.timed_out:
        raise EvidenceError(f"git {' '.join(arguments)} could not complete")
    if result.stdout_exceeded or result.stderr_exceeded:
        raise EvidenceError(f"git {' '.join(arguments)} exceeded bounded output")
    if result.return_code != 0 or result.stderr:
        raise EvidenceError(f"git {' '.join(arguments)} failed")
    try:
        return result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise EvidenceError("git returned non-ASCII repository metadata") from error


def _git_blob(commit: str, repository_path: str) -> bytes:
    if not COMMIT_RE.fullmatch(commit):
        raise EvidenceError("evidence commit must be full lowercase hexadecimal")
    result = run_bounded_process(
        ("git", "cat-file", "blob", f"{commit}:{repository_path}"),
        cwd=ROOT,
        timeout_seconds=30,
        maximum_stdout_bytes=2 * 1024 * 1024,
        maximum_stderr_bytes=64 * 1024,
    )
    if (
        result.launch_error is not None
        or result.timed_out
        or result.stdout_exceeded
        or result.stderr_exceeded
        or result.return_code != 0
        or result.stderr
    ):
        raise EvidenceError(f"cannot read {repository_path} from commit {commit}")
    return result.stdout


def _commit_harness_sha256(commit: str) -> str:
    return sha256_bytes(_git_blob(commit, "scripts/run_m3_experiment.py"))


def _verify_commit_harness_sha256(commit: str, recorded: str) -> None:
    expected = _prefixed_digest(recorded, "environment.harness.sha256")
    actual = f"sha256:{_commit_harness_sha256(commit)}"
    if expected != actual:
        raise EvidenceError("captured harness hash differs from named commit blob")


def _require_clean_commit(commit: str) -> str:
    if not COMMIT_RE.fullmatch(commit):
        raise EvidenceError("--commit must be a full lowercase 40-character commit ID")
    head = _git_output(("rev-parse", "--verify", "HEAD"))
    if head != commit:
        raise EvidenceError(f"--commit {commit} does not equal HEAD {head}")
    status = _git_output(("status", "--porcelain=v1", "--untracked-files=all"))
    if status:
        raise EvidenceError("capture requires a clean worktree and index")
    _git_output(("ls-files", "--error-unmatch", "scripts/run_m3_experiment.py"))
    blob_sha256 = _commit_harness_sha256(commit)
    current_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    if current_sha256 != blob_sha256:
        raise EvidenceError("working harness differs from the named commit blob")
    return blob_sha256


def _parse_matrix_stdout(data: bytes, context: str) -> Any:
    if len(data) > MAX_STDOUT_BYTES:
        raise EvidenceError(f"{context}: stdout exceeds 16 MiB")
    if not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise EvidenceError(f"{context}: expected exactly one LF-terminated JSON object")
    return parse_json_bytes(data[:-1], context)


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise EvidenceError(f"evidence contains forbidden symlink {item.name!r}")
        if item.is_file():
            total += item.stat().st_size
            if total > MAX_DIRECTORY_BYTES:
                raise EvidenceError("evidence directory exceeds the 16 MiB hard cap")
        elif not item.is_dir():
            raise EvidenceError("evidence contains a non-regular filesystem entry")
    return total


def _bounded_directory_entries(
    descriptor: int,
    expected: set[str],
    context: str,
) -> set[str]:
    """Enumerate one closed directory without materializing hostile extras."""

    actual: set[str] = set()
    with os.scandir(descriptor) as entries:
        for entry in entries:
            name = entry.name
            if name not in expected:
                raise EvidenceError(f"{context}: unknown entry {name!r}")
            actual.add(name)
    if actual != expected:
        raise EvidenceError(
            f"{context}: missing entries {sorted(expected - actual)}"
        )
    return actual


def _read_evidence_files(path: Path) -> dict[str, bytes]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = -1
    figures_fd = -1
    opened: list[tuple[str, int, os.stat_result]] = []
    try:
        root_fd = os.open(path, directory_flags)
        expected_root_files = {name for name in EXPECTED_FILES if "/" not in name}
        expected_root_entries = expected_root_files | {"figures"}
        _bounded_directory_entries(root_fd, expected_root_entries, "evidence root")
        figures_fd = os.open("figures", directory_flags, dir_fd=root_fd)
        expected_figures = {
            name.removeprefix("figures/")
            for name in EXPECTED_FILES
            if name.startswith("figures/")
        }
        _bounded_directory_entries(figures_fd, expected_figures, "evidence figures")

        targets = [
            (name, root_fd, name) for name in sorted(expected_root_files)
        ] + [
            (f"figures/{name}", figures_fd, name) for name in sorted(expected_figures)
        ]
        total = 0
        for logical_name, parent_fd, child_name in targets:
            descriptor = os.open(child_name, file_flags, dir_fd=parent_fd)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                os.close(descriptor)
                raise EvidenceError(f"evidence entry {logical_name!r} is not regular")
            total += metadata.st_size
            if total > MAX_DIRECTORY_BYTES:
                os.close(descriptor)
                raise EvidenceError("evidence directory exceeds the 16 MiB hard cap")
            opened.append((logical_name, descriptor, metadata))

        files: dict[str, bytes] = {}
        for logical_name, descriptor, before in opened:
            remaining = before.st_size
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    raise EvidenceError(f"evidence entry {logical_name!r} shrank while reading")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise EvidenceError(f"evidence entry {logical_name!r} grew while reading")
            after = os.fstat(descriptor)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if after_identity != before_identity:
                raise EvidenceError(f"evidence entry {logical_name!r} changed while reading")
            files[logical_name] = b"".join(chunks)
        return files
    except OSError as error:
        raise EvidenceError(f"cannot safely read evidence directory: {type(error).__name__}") from error
    finally:
        for _, descriptor, _ in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if figures_fd >= 0:
            try:
                os.close(figures_fd)
            except OSError:
                pass
        if root_fd >= 0:
            try:
                os.close(root_fd)
            except OSError:
                pass


def verify_directory(path: Path) -> dict[str, Any]:
    """Verify a complete artifact without writing any file."""

    files = _read_evidence_files(path)
    environment = _validate_environment(parse_json_bytes(files["environment.json"], "environment.json"))
    experiment = _validate_experiment(parse_json_bytes(files["experiment.json"], "experiment.json"))
    if files["environment.json"] != _json_file_bytes(environment):
        raise EvidenceError("environment.json is not canonical JSON")
    if files["experiment.json"] != _json_file_bytes(experiment):
        raise EvidenceError("experiment.json is not canonical JSON")
    if environment["captured_at_utc"] != experiment["captured_at_utc"]:
        raise EvidenceError("environment and experiment timestamps differ")
    if environment["binary"]["path"] != experiment["commands"]["matrix"][0]:
        raise EvidenceError("environment binary path differs from experiment command")
    if environment["binary"]["sha256"] != experiment["build"]["binary_sha256"]:
        raise EvidenceError("environment binary hash differs from experiment build")
    _verify_commit_harness_sha256(
        experiment["git_commit"], environment["harness"]["sha256"]
    )
    for name, expected_digest in experiment["artifacts"].items():
        if _artifact_digest(files[name]) != expected_digest:
            raise EvidenceError(f"{name}: content hash differs from experiment manifest")

    traces = parse_jsonl_bytes(files["traces.jsonl"], "traces.jsonl")
    observations = parse_jsonl_bytes(files["observations.jsonl"], "observations.jsonl")
    if files["traces.jsonl"] != _jsonl_bytes(traces):
        raise EvidenceError("traces.jsonl is not canonical JSONL")
    if files["observations.jsonl"] != _jsonl_bytes(observations):
        raise EvidenceError("observations.jsonl is not canonical JSONL")
    measured_steps = experiment["matrix_contract"]["measured_steps"]
    validate_dataset(traces, observations, measured_steps=measured_steps, require_full_matrix=True)

    traces_digest = _artifact_digest(files["traces.jsonl"])
    observations_digest = _artifact_digest(files["observations.jsonl"])
    expected_summary_object = build_summary(
        observations,
        traces_digest=traces_digest,
        observations_digest=observations_digest,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    expected_summary = _json_file_bytes(expected_summary_object)
    if files["summary.json"] != expected_summary:
        raise EvidenceError("summary.json differs from raw-observation regeneration")
    expected_figures = render_figures(expected_summary_object)
    for name, expected in expected_figures.items():
        if files[name] != expected:
            raise EvidenceError(f"{name} differs from raw-observation regeneration")
    return {
        "schema": HARNESS_SCHEMA,
        "status": "verified",
        "git_commit": experiment["git_commit"],
        "trace_rows": len(traces),
        "observation_rows": len(observations),
        "directory_bytes": sum(len(data) for data in files.values()),
    }


def _require_successful_simulator_process(
    result: BoundedProcessResult,
    *,
    context: str,
    stdout_limit_label: str,
) -> None:
    if result.launch_error is not None:
        raise EvidenceError(f"{context}: launch failed ({result.launch_error})")
    if result.timed_out:
        raise EvidenceError(f"{context}: exceeded 30-second timeout")
    if result.stdout_exceeded:
        raise EvidenceError(f"{context}: stdout exceeded {stdout_limit_label}")
    if result.stderr_exceeded:
        raise EvidenceError(f"{context}: stderr exceeded one MiB")
    if result.return_code != 0:
        raise EvidenceError(f"{context}: exited with status {result.return_code}")
    if result.stderr:
        raise EvidenceError(f"{context}: emitted forbidden stderr")


def capture(
    build_root_value: str,
    output_value: str,
    commit: str,
    measured_steps: int,
) -> dict[str, Any]:
    if type(measured_steps) is not int or measured_steps != DEFAULT_MEASURED_STEPS:
        raise EvidenceError(
            f"primary capture requires --measured-steps {DEFAULT_MEASURED_STEPS}"
        )
    harness_sha256 = _require_clean_commit(commit)
    output = _resolve_new_output(output_value)
    _require_capture_disk_reserve()
    build: PrivateBuild | None = None
    try:
        build = _build_release_binary(build_root_value)
        if _require_clean_commit(commit) != harness_sha256:
            raise EvidenceError("harness commit blob changed during the release build")
        captured_at = _utc_now()
        environment = build_environment(
            binary_sha256=build.binary_sha256,
            harness_sha256=harness_sha256,
            captured_at=captured_at,
            cargo_version=build.cargo_version,
            rustc_version=build.rustc_version,
            build_root_available_bytes=build.build_root_available_bytes,
        )

        traces: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        for family in FAMILIES:
            for replicate in REPLICATES:
                common = (
                    "--family",
                    family,
                    "--replicate",
                    str(replicate),
                    "--measured-steps",
                    str(measured_steps),
                )
                generate_context = f"generate {family} replicate {replicate}"
                generated_result = run_bounded_process(
                    (str(build.binary), "generate", *common),
                    cwd=ROOT,
                    maximum_stdout_bytes=MAX_GENERATED_TRACE_BYTES,
                )
                _require_successful_simulator_process(
                    generated_result,
                    context=generate_context,
                    stdout_limit_label="five MiB",
                )
                generated_trace = validate_generated_trace(
                    generated_result.stdout,
                    family=family,
                    replicate=replicate,
                    measured_steps=measured_steps,
                )

                matrix_context = f"matrix {family} replicate {replicate}"
                matrix_result = run_bounded_process(
                    (str(build.binary), "matrix", *common),
                    cwd=ROOT,
                    maximum_stdout_bytes=MAX_STDOUT_BYTES,
                )
                _require_successful_simulator_process(
                    matrix_result,
                    context=matrix_context,
                    stdout_limit_label="16 MiB",
                )
                matrix = _parse_matrix_stdout(matrix_result.stdout, matrix_context)
                trace, matrix_observations = validate_matrix(
                    matrix,
                    expected_family=family,
                    expected_replicate=replicate,
                    expected_measured_steps=measured_steps,
                    generated_trace=generated_trace,
                )
                traces.append(trace)
                observations.extend(matrix_observations)
            print(
                f"captured {family}: {len(REPLICATES)} paired traces",
                file=sys.stderr,
                flush=True,
            )

        validate_dataset(
            traces,
            observations,
            measured_steps=measured_steps,
            require_full_matrix=True,
        )
        if _require_clean_commit(commit) != harness_sha256:
            raise EvidenceError("harness commit blob changed during capture")
        if sha256_file(build.binary) != build.binary_sha256:
            raise EvidenceError("release binary changed during capture")

        traces_bytes = _jsonl_bytes(traces)
        observations_bytes = _jsonl_bytes(observations)
        summary_object = build_summary(
            observations,
            traces_digest=_artifact_digest(traces_bytes),
            observations_digest=_artifact_digest(observations_bytes),
        )
        summary_bytes = _json_file_bytes(summary_object)
        figure_bytes = render_figures(summary_object)
        artifact_bytes = {
            "environment.json": _json_file_bytes(environment),
            "traces.jsonl": traces_bytes,
            "observations.jsonl": observations_bytes,
            "summary.json": summary_bytes,
            **figure_bytes,
        }
        artifacts = {name: _artifact_digest(data) for name, data in artifact_bytes.items()}
        experiment = build_experiment(
            captured_at=captured_at,
            commit=commit,
            binary_sha256=build.binary_sha256,
            measured_steps=measured_steps,
            artifacts=artifacts,
        )
        artifact_bytes["experiment.json"] = _json_file_bytes(experiment)
        total_bytes = sum(len(data) for data in artifact_bytes.values())
        if total_bytes > MAX_DIRECTORY_BYTES:
            raise EvidenceError(
                f"generated evidence is {total_bytes} bytes, exceeding 16 MiB"
            )

        _require_capture_disk_reserve()
        staging = Path(tempfile.mkdtemp(prefix=".m3-evidence-staging-", dir=output.parent))
        try:
            (staging / "figures").mkdir(mode=0o755)
            for name, data in artifact_bytes.items():
                destination = staging / name
                destination.write_bytes(data)
            _directory_size(staging)
            status = verify_directory(staging)
            if os.path.lexists(output):
                raise EvidenceError("capture output appeared while staging evidence")
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return {
            **status,
            "status": "captured",
            "output": output.relative_to(ROOT).as_posix(),
        }
    finally:
        if build is not None:
            shutil.rmtree(build.root, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_parser = subparsers.add_parser("capture", help="capture a new append-only full matrix")
    capture_parser.add_argument(
        "--build-root",
        required=True,
        help="absolute tmpfs directory for one private release build",
    )
    capture_parser.add_argument("--output", required=True, help="new repository-relative evidence directory")
    capture_parser.add_argument("--commit", required=True, help="full clean HEAD commit ID")
    capture_parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    verify_parser = subparsers.add_parser("verify", help="verify an existing complete matrix")
    verify_parser.add_argument("--input", required=True, help="evidence directory")
    verify_parser.add_argument("--check", action="store_true", required=True, help="regenerate in memory and byte-compare without writes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "capture":
            status = capture(
                arguments.build_root,
                arguments.output,
                arguments.commit,
                arguments.measured_steps,
            )
        else:
            input_path = Path(arguments.input)
            if not input_path.is_absolute():
                input_path = ROOT / input_path
            status = verify_directory(input_path)
    except (EvidenceError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_json_file_bytes(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
