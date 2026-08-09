#!/usr/bin/env python3
"""Capture and verify the preregistered M4 BF16 GEMV evidence.

This standard-library-only harness treats both benchmark output and archived
evidence as hostile input. Schemas and file sets are closed, JSON duplicate
keys are rejected, every raw timing row is retained, and all summaries and
figures are byte-regenerated from the raw ledgers.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import re
import resource
import shutil
import signal
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HARNESS_SCHEMA = "runnel.m4-evidence/1"
CASE_SCHEMA = "runnel.m4-case/1"
CORRECTNESS_SCHEMA = "runnel.m4-correctness/1"
CELL_REQUEST_SCHEMA = "runnel.m4-cell-request/1"
WARMUP_SCHEMA = "runnel.m4-warmup/1"
OBSERVATION_SCHEMA = "runnel.m4-observation/1"
SUMMARY_SCHEMA = "runnel.m4-summary/1"
ENVIRONMENT_SCHEMA = "runnel.m4-environment/1"
EXPERIMENT_SCHEMA = "runnel.m4-experiment/1"

BASELINE_IMPLEMENTATION = "rust-scalar-bf16-gemv-v1"
AVX2_IMPLEMENTATION = "c-avx2-bf16-gemv-v1"
STAGED_IMPLEMENTATION = "rust-staged-f32-gemv-v1"
WARMUP_ORDERS = (
    "baseline-candidate",
    "candidate-baseline",
    "baseline-candidate",
    "candidate-baseline",
    "baseline-candidate",
)
MEASURED_PAIRS = 30
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_OBSERVATIONS = 13 * 2 * MEASURED_PAIRS

MAX_STDOUT_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024
MAX_JSONL_RECORD_BYTES = 256 * 1024
MAX_DIRECTORY_BYTES = 16 * 1024 * 1024
MAX_GIT_OUTPUT_BYTES = 64 * 1024
MAX_LIVE_BENCHMARK_BYTES = 256 * 1024 * 1024
MIN_EVIDENCE_FILESYSTEM_FREE_BYTES = 2 * MAX_DIRECTORY_BYTES
CELL_TIMEOUT_SECONDS = 180.0
FULL_CAPTURE_TIMEOUT_SECONDS = 15 * 60.0
BUILD_TIMEOUT_SECONDS = 15 * 60.0
MIN_FREE_BYTES_AFTER_CAPTURE = 2 * 1024 * 1024 * 1024
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
    "--jobs",
    "2",
    "-p",
    "runnel-kernels",
    "--bin",
    "runnel-kernel-bench",
    "-p",
    "runnel",
    "--bin",
    "runnel-m4-model-check",
)
NATIVE_CC_ARGV = (
    "cc",
    "-std=c11",
    "-O3",
    "-fPIC",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-Wconversion",
    "-Wshadow",
    "-Wstrict-prototypes",
    "-Wmissing-prototypes",
    "-Wformat=2",
    "-Wundef",
    "-Wwrite-strings",
    "-Iinclude",
    "-c",
    "native/bf16_gemv.c",
    "-o",
    "{private-cargo-out}/bf16_gemv.o",
)
NATIVE_AR_ARGV = (
    "ar",
    "crs",
    "{private-cargo-out}/librunnel_kernels.a",
    "{private-cargo-out}/bf16_gemv.o",
)
BINARY_LOGICAL_PATH = "{private-tmpfs-target}/release/runnel-kernel-bench"
MODEL_BINARY_LOGICAL_PATH = "{private-tmpfs-target}/release/runnel-m4-model-check"
HARNESS_PATH = "scripts/run_m4_experiment.py"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PORTABLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
EXPECTED_FILES = frozenset(
    {
        "environment.json",
        "experiment.json",
        "cases.jsonl",
        "correctness.jsonl",
        "observations.jsonl",
        "summary.json",
        "figures/elapsed-time.svg",
        "figures/paired-ratios.svg",
    }
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    rows: int
    columns: int
    calls_per_worker: int
    role: str

    @property
    def matrix_bytes(self) -> int:
        return self.rows * self.columns * 2

    @property
    def input_bytes(self) -> int:
        return self.columns * 4

    @property
    def output_bytes(self) -> int:
        return self.rows * 4

    @property
    def element_products_per_worker(self) -> int:
        return self.rows * self.columns * self.calls_per_worker


@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    case_id: str
    candidate: str
    allocation: str
    workers: int
    role: str

    def implementation(self, variant: str) -> str:
        if variant == "baseline":
            return BASELINE_IMPLEMENTATION
        if variant == "candidate":
            return self.candidate
        raise EvidenceError(f"unknown variant {variant!r}")


CASES = (
    CaseSpec("tail-257x513", 257, 513, 1_019, "descriptive"),
    CaseSpec("l2-512x512", 512, 512, 512, "descriptive"),
    CaseSpec("llc-2048x2048", 2_048, 2_048, 32, "descriptive"),
    CaseSpec("stream-expand-8192x2048", 8_192, 2_048, 8, "primary"),
    CaseSpec("stream-contract-2048x8192", 2_048, 8_192, 8, "primary"),
)
CASE_BY_ID = {case.case_id: case for case in CASES}

EXPECTED_CASE_DIGESTS = {
    "tail-257x513": (
        "sha256:f71a97cd58a4d41d072a30643f3b93d82351969a338e6acbf67310380af76573",
        "sha256:40cad436319e56ce852d9334c8c54a93fd6d04eaf530708722224c89829f6f5c",
    ),
    "l2-512x512": (
        "sha256:44a7d48fb4016ac0724b9b28aefe1898fbe5e1b62359cb546e17c3d53a68eb6c",
        "sha256:7c61014e62916ca3848dbad1904656a21f2745f88ecc38fe138f1b64b46c4179",
    ),
    "llc-2048x2048": (
        "sha256:da7bf0d37157514fd134e374cb55b3c281d38f97edc2372124d11e76c40515f5",
        "sha256:b04e28a0551f0c5fca3eb0f4be9fc5498bd6542a9bb0752fab263ff4ee168dbc",
    ),
    "stream-expand-8192x2048": (
        "sha256:6d17b58eaa4386cf23d8c240e120bf8e22cb0cf566960854da4f86d4616ca2e8",
        "sha256:e7e31b6745eb435841112d80958a2a388f47fbfbdb9936b6fb574f1aabdba375",
    ),
    "stream-contract-2048x8192": (
        "sha256:d9daa89e789a5dfff2ba007619f8636cf7ff9f03cee41f44aeaeac76e0dc6002",
        "sha256:2c0108bea6a9e2d6cb80f7a0503b1960ef5549e444f3553d4830e5ff50eb5f62",
    ),
}

CELLS = (
    CellSpec("avx-natural-tail", "tail-257x513", AVX2_IMPLEMENTATION, "natural", 1, "descriptive"),
    CellSpec("avx-natural-l2", "l2-512x512", AVX2_IMPLEMENTATION, "natural", 1, "descriptive"),
    CellSpec("avx-natural-llc", "llc-2048x2048", AVX2_IMPLEMENTATION, "natural", 1, "descriptive"),
    CellSpec(
        "avx-natural-stream-expand",
        "stream-expand-8192x2048",
        AVX2_IMPLEMENTATION,
        "natural",
        1,
        "primary",
    ),
    CellSpec(
        "avx-natural-stream-contract",
        "stream-contract-2048x8192",
        AVX2_IMPLEMENTATION,
        "natural",
        1,
        "primary",
    ),
    CellSpec("avx-offset-tail", "tail-257x513", AVX2_IMPLEMENTATION, "vector-offset", 1, "descriptive"),
    CellSpec(
        "avx-offset-stream-expand",
        "stream-expand-8192x2048",
        AVX2_IMPLEMENTATION,
        "vector-offset",
        1,
        "descriptive",
    ),
    CellSpec(
        "avx-offset-stream-contract",
        "stream-contract-2048x8192",
        AVX2_IMPLEMENTATION,
        "vector-offset",
        1,
        "descriptive",
    ),
    CellSpec(
        "avx-two-worker-stream-expand",
        "stream-expand-8192x2048",
        AVX2_IMPLEMENTATION,
        "natural",
        2,
        "descriptive",
    ),
    CellSpec(
        "avx-two-worker-stream-contract",
        "stream-contract-2048x8192",
        AVX2_IMPLEMENTATION,
        "natural",
        2,
        "descriptive",
    ),
    CellSpec(
        "staged-natural-llc",
        "llc-2048x2048",
        STAGED_IMPLEMENTATION,
        "natural",
        1,
        "secondary",
    ),
    CellSpec(
        "staged-natural-stream-expand",
        "stream-expand-8192x2048",
        STAGED_IMPLEMENTATION,
        "natural",
        1,
        "secondary",
    ),
    CellSpec(
        "staged-natural-stream-contract",
        "stream-contract-2048x8192",
        STAGED_IMPLEMENTATION,
        "natural",
        1,
        "secondary",
    ),
)
CELL_BY_ID = {cell.cell_id: cell for cell in CELLS}


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
class PrivateBuild:
    root: Path
    kernel_binary: Path
    kernel_binary_sha256: str
    model_binary: Path
    model_binary_sha256: str
    cargo_version: str
    rustc_version: str
    cc_path: str
    cc_sha256: str
    cc_version: str
    ar_path: str
    ar_sha256: str
    ar_version: str
    build_root_available_bytes: int


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
    except (ValueError, RecursionError) as error:
        raise EvidenceError(f"{context}: invalid JSON: {error}") from error


def parse_jsonl_bytes(data: bytes, context: str) -> list[dict[str, Any]]:
    if data and not data.endswith(b"\n"):
        raise EvidenceError(f"{context}: JSONL must end with LF")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(data.splitlines(), 1):
        if not line:
            raise EvidenceError(f"{context}:{number}: blank JSONL record")
        if len(line) > MAX_JSONL_RECORD_BYTES:
            raise EvidenceError(f"{context}:{number}: record exceeds 256 KiB")
        value = parse_json_bytes(line, f"{context}:{number}")
        if type(value) is not dict:
            raise EvidenceError(f"{context}:{number}: record must be an object")
        rows.append(value)
    return rows


def parse_jsonl_prefix_bytes(
    data: bytes, context: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Return the longest complete valid JSONL prefix and a bounded failure."""

    rows: list[dict[str, Any]] = []
    offset = 0
    number = 1
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline < 0:
            return rows, f"{context}:{number}: trailing partial JSONL record"
        line = data[offset:newline]
        if not line:
            return rows, f"{context}:{number}: blank JSONL record"
        if len(line) > MAX_JSONL_RECORD_BYTES:
            return rows, f"{context}:{number}: record exceeds 256 KiB"
        try:
            value = parse_json_bytes(line, f"{context}:{number}")
        except EvidenceError as error:
            return rows, str(error)[:512]
        if type(value) is not dict:
            return rows, f"{context}:{number}: record must be an object"
        rows.append(value)
        offset = newline + 1
        number += 1
    return rows, None


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
        raise EvidenceError(
            f"{context}: closed schema mismatch; "
            f"missing={sorted(wanted - actual)}, unknown={sorted(actual - wanted)}"
        )


def _string(
    value: Any,
    context: str,
    *,
    expected: str | None = None,
    maximum: int = 512,
) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise EvidenceError(
            f"{context}: expected non-empty string of at most {maximum} characters"
        )
    if expected is not None and value != expected:
        raise EvidenceError(f"{context}: expected {expected!r}, got {value!r}")
    return value


def _nullable_string(value: Any, context: str, maximum: int = 512) -> str | None:
    if value is None:
        return None
    return _string(value, context, maximum=maximum)


def _uint(value: Any, context: str, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise EvidenceError(f"{context}: expected unsigned integer <= {maximum}")
    return value


def _positive_uint(value: Any, context: str, maximum: int = (1 << 64) - 1) -> int:
    result = _uint(value, context, maximum)
    if result == 0:
        raise EvidenceError(f"{context}: expected positive integer")
    return result


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise EvidenceError(f"{context}: expected boolean")
    return value


def _number(value: Any, context: str, *, nonnegative: bool = False) -> float:
    if type(value) not in {int, float}:
        raise EvidenceError(f"{context}: expected finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise EvidenceError(f"{context}: expected finite{' nonnegative' if nonnegative else ''} number")
    return result


def _prefixed_digest(value: Any, context: str) -> str:
    digest = _string(value, context, maximum=71)
    if not digest.startswith("sha256:") or not SHA256_RE.fullmatch(digest[7:]):
        raise EvidenceError(f"{context}: expected sha256:<lowercase-hex>")
    return digest


def _raw_digest(value: Any, context: str) -> str:
    digest = _string(value, context, maximum=64)
    if not SHA256_RE.fullmatch(digest):
        raise EvidenceError(f"{context}: expected 64 lowercase hexadecimal characters")
    return digest


class Sha256U64Stream:
    """Domain-separated SHA-256 counter stream consumed as little-endian u64s."""

    def __init__(self, prefix: bytes) -> None:
        self.prefix = prefix
        self.counter = 0
        self.words: list[int] = []

    def next_u64(self) -> int:
        if not self.words:
            digest = hashlib.sha256(self.prefix + struct.pack("<Q", self.counter)).digest()
            self.counter += 1
            self.words.extend(struct.unpack("<QQQQ", digest))
        return self.words.pop(0)

    def below(self, bound: int) -> int:
        if bound <= 0 or bound > 1 << 64:
            raise EvidenceError("random bound must be in 1..=2^64")
        limit = ((1 << 64) // bound) * bound
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def _shuffle(values: Sequence[Any], stream: Sha256U64Stream) -> list[Any]:
    result = list(values)
    for index in range(len(result) - 1, 0, -1):
        other = stream.below(index + 1)
        result[index], result[other] = result[other], result[index]
    return result


def _cell_material(domain: bytes, cell_id: str) -> bytes:
    encoded = cell_id.encode("ascii")
    if len(encoded) > (1 << 16) - 1:
        raise EvidenceError("cell ID is too long")
    return domain + struct.pack("<H", len(encoded)) + encoded


def cell_order() -> list[str]:
    stream = Sha256U64Stream(b"runnel-m4-cell-order-v1\0")
    return _shuffle([cell.cell_id for cell in CELLS], stream)


def measured_pair_orders(cell_id: str) -> list[str]:
    if cell_id not in CELL_BY_ID:
        raise EvidenceError(f"unknown cell {cell_id!r}")
    initial = ["baseline-candidate"] * 15 + ["candidate-baseline"] * 15
    stream = Sha256U64Stream(_cell_material(b"runnel-m4-pair-order-v1\0", cell_id))
    return _shuffle(initial, stream)


def _variant_order(order: str) -> tuple[str, str]:
    if order == "baseline-candidate":
        return "baseline", "candidate"
    if order == "candidate-baseline":
        return "candidate", "baseline"
    raise EvidenceError(f"invalid pair order {order!r}")


def validate_cases(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(values) != len(CASES):
        raise EvidenceError(f"cases.jsonl: expected {len(CASES)} rows, got {len(values)}")
    normalized: list[dict[str, Any]] = []
    keys = (
        "schema",
        "case_id",
        "rows",
        "columns",
        "calls_per_worker",
        "matrix_bytes",
        "input_bytes",
        "output_bytes",
        "element_products_per_worker",
        "role",
        "weight_sha256",
        "input_sha256",
    )
    for index, (value, specification) in enumerate(zip(values, CASES, strict=True)):
        context = f"cases.jsonl:{index + 1}"
        row = _object(value, context)
        _exact_keys(row, keys, context)
        _string(row["schema"], f"{context}.schema", expected=CASE_SCHEMA)
        _string(row["case_id"], f"{context}.case_id", expected=specification.case_id)
        expected_numbers = {
            "rows": specification.rows,
            "columns": specification.columns,
            "calls_per_worker": specification.calls_per_worker,
            "matrix_bytes": specification.matrix_bytes,
            "input_bytes": specification.input_bytes,
            "output_bytes": specification.output_bytes,
            "element_products_per_worker": specification.element_products_per_worker,
        }
        for name, expected in expected_numbers.items():
            actual = _positive_uint(row[name], f"{context}.{name}")
            if actual != expected:
                raise EvidenceError(f"{context}.{name}: expected {expected}, got {actual}")
        _string(row["role"], f"{context}.role", expected=specification.role)
        weight_digest = _prefixed_digest(
            row["weight_sha256"], f"{context}.weight_sha256"
        )
        input_digest = _prefixed_digest(
            row["input_sha256"], f"{context}.input_sha256"
        )
        expected_weight, expected_input = EXPECTED_CASE_DIGESTS[specification.case_id]
        if (weight_digest, input_digest) != (expected_weight, expected_input):
            raise EvidenceError(f"{context}: fixture digest differs from the frozen case")
        normalized.append(dict(row))
    return normalized


KERNEL_METRIC_KEYS = (
    "components",
    "max_abs_error",
    "max_error_to_bound_ratio",
    "worst_index",
    "worst_abs_error",
    "worst_error_bound",
    "cross_scalar_max_abs",
    "cross_scalar_max_rel",
    "cross_scalar_max_ulp",
)
MODEL_METRIC_KEYS = (
    "adapter_version",
    "representation",
    "requested_backend",
    "selected_backend",
    "fixture",
    "goldens",
    "prompt",
    "input_ids",
    "full_ids",
    "positions",
    "repetitions",
    "tokens_exact",
    "expert_ids_exact",
    "deterministic",
    "tolerance",
    "logits_error",
    "router_score_error",
    "route_weight_error",
)
MODEL_FIXTURE_KEYS = (
    "artifact_id",
    "object_digest",
    "object_length",
    "page_table_digest",
    "page_table_length",
)
MODEL_GOLDEN_KEYS = (
    "logits_sha256",
    "routes_sha256",
    "tokens_sha256",
    "metadata_sha256",
)
MODEL_ERROR_KEYS = (
    "max_abs",
    "max_rel",
    "max_tolerance_ratio",
    "worst_position",
    "worst_index",
)
MODEL_CHECKS = (
    ("tiny-v1-preservation", "v1-f32"),
    ("tiny-v2-scalar", "scalar"),
    ("tiny-v2-avx2", "avx2"),
)
MODEL_CONTRACTS = {
    "tiny-v1-preservation": {
        "adapter_version": 1,
        "representation": "f32",
        "requested_backend": "f32-preservation",
        "fixture": {
            "artifact_id": "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3",
            "object_digest": "sha256:6b2b8a1bbb2854084b1e1fe1e5787a9cfdb021b397e774fc7dbef79ac9d24bf6",
            "object_length": 7_904,
            "page_table_digest": "sha256:29383b56a150f9e5705f3666ca7707f21bc3fbd663ffd7249f4ecb938da6a62d",
            "page_table_length": 96,
        },
        "goldens": {
            "logits_sha256": "0578cfbe25a8fffbc0bcf46dd02f13de70ad9cfb0611efddd7ab4124eb2e7ab1",
            "routes_sha256": "e1e2a2b4e209a57932f75898ad2b7e073a5cb9601f4e3375f2a6ad844aae78f1",
            "tokens_sha256": "81c168e3b067f86861babda66ee92a06e6f3a236d2d992751eb166eb45e9f6bf",
            "metadata_sha256": "d3758aebb051dbe3b415b0a6bb8e14482b412046dbbe9e6b949abfada6c07aa4",
        },
    },
    "tiny-v2-scalar": {
        "adapter_version": 2,
        "representation": "bf16-experts",
        "requested_backend": "forced-scalar",
        "fixture": {
            "artifact_id": "sha256:606baa0c1082b369632b5dd000d30dc51ae20321b2c032ef3395aaa0bfd7c76c",
            "object_digest": "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
            "object_length": 5_600,
            "page_table_digest": "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
            "page_table_length": 96,
        },
        "goldens": {
            "logits_sha256": "e06577a39618b0bdef46bb28c4a73d3f131aa86ea8a53c85727af8ed0cdf89f2",
            "routes_sha256": "2c048aeb63a8290e370750aabef79bc0b0c6fbc818e7beb1026b4c28ed63a73c",
            "tokens_sha256": "0a7368aa11bfe986afae53e3b72a644e48d4e7d9c3f8e3cefb247a8b6d4d521e",
            "metadata_sha256": "9997277ef66168f132621559a8cb17e55581715707a87e92906e5425b85bb4a3",
        },
    },
    "tiny-v2-avx2": {
        "adapter_version": 2,
        "representation": "bf16-experts",
        "requested_backend": "forced-avx2",
        "fixture": {
            "artifact_id": "sha256:606baa0c1082b369632b5dd000d30dc51ae20321b2c032ef3395aaa0bfd7c76c",
            "object_digest": "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
            "object_length": 5_600,
            "page_table_digest": "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
            "page_table_length": 96,
        },
        "goldens": {
            "logits_sha256": "e06577a39618b0bdef46bb28c4a73d3f131aa86ea8a53c85727af8ed0cdf89f2",
            "routes_sha256": "2c048aeb63a8290e370750aabef79bc0b0c6fbc818e7beb1026b4c28ed63a73c",
            "tokens_sha256": "0a7368aa11bfe986afae53e3b72a644e48d4e7d9c3f8e3cefb247a8b6d4d521e",
            "metadata_sha256": "9997277ef66168f132621559a8cb17e55581715707a87e92906e5425b85bb4a3",
        },
    },
}


CURRENT_MODEL_CONTRACT_COHORT = "pytorch-2.13.0+cpu"
M4_20260803_MODEL_CONTRACT_COHORT = "m4-20260803-pytorch-2.7.1+cpu"
M4_20260803_GIT_COMMIT = "035d217baf0901809fa02bf0a5c11c1a490198c2"
M4_20260803_MODEL_CONTRACTS = {
    "tiny-v1-preservation": {
        "adapter_version": 1,
        "representation": "f32",
        "requested_backend": "f32-preservation",
        "fixture": {
            "artifact_id": "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3",
            "object_digest": "sha256:6b2b8a1bbb2854084b1e1fe1e5787a9cfdb021b397e774fc7dbef79ac9d24bf6",
            "object_length": 7_904,
            "page_table_digest": "sha256:29383b56a150f9e5705f3666ca7707f21bc3fbd663ffd7249f4ecb938da6a62d",
            "page_table_length": 96,
        },
        "goldens": {
            "logits_sha256": "0578cfbe25a8fffbc0bcf46dd02f13de70ad9cfb0611efddd7ab4124eb2e7ab1",
            "routes_sha256": "e1e2a2b4e209a57932f75898ad2b7e073a5cb9601f4e3375f2a6ad844aae78f1",
            "tokens_sha256": "81c168e3b067f86861babda66ee92a06e6f3a236d2d992751eb166eb45e9f6bf",
            "metadata_sha256": "9c358629ccb1bdf704abf4625199cbde1afe1bf3108de01aef424e6643e1bec1",
        },
    },
    "tiny-v2-scalar": {
        "adapter_version": 2,
        "representation": "bf16-experts",
        "requested_backend": "forced-scalar",
        "fixture": {
            "artifact_id": "sha256:606baa0c1082b369632b5dd000d30dc51ae20321b2c032ef3395aaa0bfd7c76c",
            "object_digest": "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
            "object_length": 5_600,
            "page_table_digest": "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
            "page_table_length": 96,
        },
        "goldens": {
            "logits_sha256": "e06577a39618b0bdef46bb28c4a73d3f131aa86ea8a53c85727af8ed0cdf89f2",
            "routes_sha256": "2c048aeb63a8290e370750aabef79bc0b0c6fbc818e7beb1026b4c28ed63a73c",
            "tokens_sha256": "0a7368aa11bfe986afae53e3b72a644e48d4e7d9c3f8e3cefb247a8b6d4d521e",
            "metadata_sha256": "43d32b9bcfc00f0f34f04a99d930ca25edb4c1f4372064021439f35cbd211e3c",
        },
    },
    "tiny-v2-avx2": {
        "adapter_version": 2,
        "representation": "bf16-experts",
        "requested_backend": "forced-avx2",
        "fixture": {
            "artifact_id": "sha256:606baa0c1082b369632b5dd000d30dc51ae20321b2c032ef3395aaa0bfd7c76c",
            "object_digest": "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
            "object_length": 5_600,
            "page_table_digest": "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
            "page_table_length": 96,
        },
        "goldens": {
            "logits_sha256": "e06577a39618b0bdef46bb28c4a73d3f131aa86ea8a53c85727af8ed0cdf89f2",
            "routes_sha256": "2c048aeb63a8290e370750aabef79bc0b0c6fbc818e7beb1026b4c28ed63a73c",
            "tokens_sha256": "0a7368aa11bfe986afae53e3b72a644e48d4e7d9c3f8e3cefb247a8b6d4d521e",
            "metadata_sha256": "43d32b9bcfc00f0f34f04a99d930ca25edb4c1f4372064021439f35cbd211e3c",
        },
    },
}
MODEL_CONTRACTS_BY_COHORT = {
    CURRENT_MODEL_CONTRACT_COHORT: MODEL_CONTRACTS,
    M4_20260803_MODEL_CONTRACT_COHORT: M4_20260803_MODEL_CONTRACTS,
}


def _model_contract_cohort_for_commit(commit: str) -> str:
    if commit == M4_20260803_GIT_COMMIT:
        return M4_20260803_MODEL_CONTRACT_COHORT
    return CURRENT_MODEL_CONTRACT_COHORT


def _model_contracts_for_cohort(
    cohort: str,
) -> Mapping[str, Mapping[str, Any]]:
    try:
        return MODEL_CONTRACTS_BY_COHORT[cohort]
    except KeyError as error:
        raise EvidenceError("unknown model contract cohort") from error


def expected_correctness_checks() -> list[tuple[str, str, str | None, str]]:
    result: list[tuple[str, str, str | None, str]] = []
    for case in CASES:
        for backend in ("scalar", "avx2"):
            result.append((f"{case.case_id}-{backend}", "kernel", case.case_id, backend))
    for case_id in (
        "llc-2048x2048",
        "stream-expand-8192x2048",
        "stream-contract-2048x8192",
    ):
        result.append((f"{case_id}-staged", "kernel", case_id, "staged"))
    for check_id, backend in MODEL_CHECKS:
        result.append((check_id, "model", None, backend))
    return result


def _validate_model_error(
    value: Any,
    context: str,
    *,
    maximum_index: int,
) -> float:
    error = _object(value, context)
    _exact_keys(error, MODEL_ERROR_KEYS, context)
    _number(error["max_abs"], f"{context}.max_abs", nonnegative=True)
    _number(error["max_rel"], f"{context}.max_rel", nonnegative=True)
    ratio = _number(
        error["max_tolerance_ratio"],
        f"{context}.max_tolerance_ratio",
        nonnegative=True,
    )
    _uint(error["worst_position"], f"{context}.worst_position", 7)
    _uint(error["worst_index"], f"{context}.worst_index", maximum_index)
    return ratio


def _validate_kernel_diagnostics(
    metrics: Mapping[str, Any],
    context: str,
    *,
    components: int,
    cross_scalar_available: bool,
) -> float:
    max_error = _number(
        metrics["max_abs_error"],
        f"{context}.max_abs_error",
        nonnegative=True,
    )
    max_ratio = _number(
        metrics["max_error_to_bound_ratio"],
        f"{context}.max_error_to_bound_ratio",
        nonnegative=True,
    )
    _uint(metrics["worst_index"], f"{context}.worst_index", components - 1)
    worst_error = _number(
        metrics["worst_abs_error"],
        f"{context}.worst_abs_error",
        nonnegative=True,
    )
    worst_bound = _number(
        metrics["worst_error_bound"],
        f"{context}.worst_error_bound",
        nonnegative=True,
    )
    if worst_bound <= 0.0:
        raise EvidenceError(f"{context}.worst_error_bound: expected positive")
    if max_error < worst_error:
        raise EvidenceError(f"{context}: global maximum is below worst-ratio error")
    if not math.isclose(
        max_ratio,
        worst_error / worst_bound,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise EvidenceError(f"{context}: worst-component ratio fields disagree")
    cross_names = (
        "cross_scalar_max_abs",
        "cross_scalar_max_rel",
        "cross_scalar_max_ulp",
    )
    if not cross_scalar_available:
        if any(metrics[name] is not None for name in cross_names):
            raise EvidenceError(f"{context}: unavailable cross-scalar diagnostics must be null")
    else:
        _number(
            metrics["cross_scalar_max_abs"],
            f"{context}.cross_scalar_max_abs",
            nonnegative=True,
        )
        _number(
            metrics["cross_scalar_max_rel"],
            f"{context}.cross_scalar_max_rel",
            nonnegative=True,
        )
        _uint(
            metrics["cross_scalar_max_ulp"],
            f"{context}.cross_scalar_max_ulp",
            (1 << 32) - 1,
        )
    return max_ratio


def validate_correctness(
    values: Sequence[Mapping[str, Any]],
    *,
    model_contract_cohort: str = CURRENT_MODEL_CONTRACT_COHORT,
) -> list[dict[str, Any]]:
    model_contracts = _model_contracts_for_cohort(model_contract_cohort)
    expected = expected_correctness_checks()
    if len(values) != len(expected):
        raise EvidenceError(
            f"correctness.jsonl: expected {len(expected)} rows, got {len(values)}"
        )
    keys = (
        "schema",
        "check_id",
        "kind",
        "case_id",
        "backend",
        "status",
        "failure",
        "metrics",
    )
    normalized: list[dict[str, Any]] = []
    scalar_output_available: dict[str, bool] = {}
    for index, (value, expected_row) in enumerate(zip(values, expected, strict=True)):
        context = f"correctness.jsonl:{index + 1}"
        row = _object(value, context)
        _exact_keys(row, keys, context)
        check_id, kind, case_id, backend = expected_row
        _string(row["schema"], f"{context}.schema", expected=CORRECTNESS_SCHEMA)
        _string(row["check_id"], f"{context}.check_id", expected=check_id)
        _string(row["kind"], f"{context}.kind", expected=kind)
        if case_id is None:
            if row["case_id"] is not None:
                raise EvidenceError(f"{context}.case_id: expected null")
        else:
            _string(row["case_id"], f"{context}.case_id", expected=case_id)
        _string(row["backend"], f"{context}.backend", expected=backend)
        status = _string(row["status"], f"{context}.status", maximum=32)
        if status not in {"ok", "unsupported", "failed"}:
            raise EvidenceError(f"{context}.status: unexpected status {status!r}")
        failure = _nullable_string(row["failure"], f"{context}.failure", maximum=512)
        if (status == "ok") != (failure is None):
            raise EvidenceError(f"{context}: failure must be null exactly when status is ok")
        if status == "unsupported" and failure != "backend_unavailable":
            raise EvidenceError(
                f"{context}.failure: unsupported rows require 'backend_unavailable'"
            )
        metrics = _object(row["metrics"], f"{context}.metrics")
        if kind == "kernel":
            _exact_keys(metrics, KERNEL_METRIC_KEYS, f"{context}.metrics")
            components = _positive_uint(metrics["components"], f"{context}.metrics.components")
            if components != CASE_BY_ID[case_id].rows:
                raise EvidenceError(f"{context}.metrics.components: expected case row count")
            diagnostics = KERNEL_METRIC_KEYS[1:]
            diagnostics_are_null = all(metrics[name] is None for name in diagnostics)
            if status == "unsupported" and backend != "avx2":
                raise EvidenceError(f"{context}: only AVX2 may be unsupported")
            if status == "unsupported":
                if not diagnostics_are_null:
                    raise EvidenceError(
                        f"{context}: unsupported kernel diagnostics must be null"
                    )
            elif status == "failed" and failure == "kernel_execution_error":
                if not diagnostics_are_null:
                    raise EvidenceError(
                        f"{context}: execution-error kernel diagnostics must be null"
                    )
            else:
                if status == "failed" and failure != "numerical_bound_exceeded":
                    raise EvidenceError(
                        f"{context}.failure: failed kernel row has an unknown error code"
                    )
                max_ratio = _validate_kernel_diagnostics(
                    metrics,
                    f"{context}.metrics",
                    components=components,
                    cross_scalar_available=(
                        backend == "scalar"
                        or scalar_output_available.get(case_id, False)
                    ),
                )
                if status == "ok" and max_ratio > 1.0:
                    raise EvidenceError(
                        f"{context}: successful kernel row exceeds a componentwise f64 bound"
                    )
                if status == "failed" and max_ratio <= 1.0:
                    raise EvidenceError(
                        f"{context}: numerical failure does not exceed its componentwise bound"
                    )
            if status == "failed" and failure not in {
                "numerical_bound_exceeded",
                "kernel_execution_error",
            }:
                raise EvidenceError(
                    f"{context}.failure: failed kernel row has an unknown error code"
                )
            if status != "ok" and status != "unsupported" and failure is None:
                raise EvidenceError(f"{context}.failure: failed kernel row requires an error code")
            if backend == "scalar":
                scalar_output_available[case_id] = not (
                    status == "unsupported"
                    or (status == "failed" and failure == "kernel_execution_error")
                )
        else:
            _exact_keys(metrics, MODEL_METRIC_KEYS, f"{context}.metrics")
            contract = model_contracts[check_id]
            adapter_version = _positive_uint(
                metrics["adapter_version"], f"{context}.metrics.adapter_version", 2
            )
            if adapter_version != contract["adapter_version"]:
                raise EvidenceError(f"{context}.metrics.adapter_version: wrong fixture version")
            _string(
                metrics["representation"],
                f"{context}.metrics.representation",
                expected=contract["representation"],
            )
            _string(
                metrics["requested_backend"],
                f"{context}.metrics.requested_backend",
                expected=contract["requested_backend"],
            )
            selected = metrics["selected_backend"]
            expected_selected: str | None
            proof_absent = status == "unsupported" or (
                status == "failed" and failure == "model_execution_error"
            )
            if check_id == "tiny-v1-preservation" or proof_absent:
                expected_selected = None
            elif check_id == "tiny-v2-scalar":
                expected_selected = "scalar"
            else:
                expected_selected = "avx2"
            if selected != expected_selected:
                raise EvidenceError(
                    f"{context}.metrics.selected_backend: expected {expected_selected!r}"
                )
            if status == "unsupported" and check_id != "tiny-v2-avx2":
                raise EvidenceError(f"{context}: only forced AVX2 may be unsupported")

            fixture = _object(metrics["fixture"], f"{context}.metrics.fixture")
            _exact_keys(fixture, MODEL_FIXTURE_KEYS, f"{context}.metrics.fixture")
            for name in ("artifact_id", "object_digest", "page_table_digest"):
                _prefixed_digest(fixture[name], f"{context}.metrics.fixture.{name}")
            for name in ("object_length", "page_table_length"):
                _positive_uint(fixture[name], f"{context}.metrics.fixture.{name}")
            if fixture != contract["fixture"]:
                raise EvidenceError(f"{context}.metrics.fixture: wrong frozen identity")

            goldens = _object(metrics["goldens"], f"{context}.metrics.goldens")
            _exact_keys(goldens, MODEL_GOLDEN_KEYS, f"{context}.metrics.goldens")
            for name in MODEL_GOLDEN_KEYS:
                _raw_digest(goldens[name], f"{context}.metrics.goldens.{name}")
            if goldens != contract["goldens"]:
                raise EvidenceError(f"{context}.metrics.goldens: wrong oracle evidence")

            _string(metrics["prompt"], f"{context}.metrics.prompt", expected="moe")
            for name, expected_ids in (
                ("input_ids", [1, 14, 16, 6]),
                ("full_ids", [1, 14, 16, 6, 15, 11, 20, 9]),
                ("positions", list(range(8))),
            ):
                ids = _array(metrics[name], f"{context}.metrics.{name}")
                for item_index, item in enumerate(ids):
                    _uint(item, f"{context}.metrics.{name}[{item_index}]", 31)
                if ids != expected_ids:
                    raise EvidenceError(f"{context}.metrics.{name}: wrong golden vector")
            repetitions = _positive_uint(
                metrics["repetitions"], f"{context}.metrics.repetitions", 2
            )
            if repetitions != 2:
                raise EvidenceError(f"{context}.metrics.repetitions: expected 2")

            tokens_exact = _boolean(metrics["tokens_exact"], f"{context}.metrics.tokens_exact")
            expert_ids_exact = _boolean(
                metrics["expert_ids_exact"], f"{context}.metrics.expert_ids_exact"
            )
            deterministic = _boolean(
                metrics["deterministic"], f"{context}.metrics.deterministic"
            )
            tolerance = _object(metrics["tolerance"], f"{context}.metrics.tolerance")
            _exact_keys(tolerance, ("atol", "rtol"), f"{context}.metrics.tolerance")
            atol = _number(
                tolerance["atol"], f"{context}.metrics.tolerance.atol", nonnegative=True
            )
            rtol = _number(
                tolerance["rtol"], f"{context}.metrics.tolerance.rtol", nonnegative=True
            )
            if (atol, rtol) != (1e-5, 1e-4):
                raise EvidenceError(f"{context}.metrics.tolerance: wrong frozen tolerance")

            if proof_absent:
                if any(
                    metrics[name] is not None
                    for name in (
                        "logits_error",
                        "router_score_error",
                        "route_weight_error",
                    )
                ):
                    raise EvidenceError(f"{context}: unexecuted model checks require null proofs")
                if any((tokens_exact, expert_ids_exact, deterministic)):
                    raise EvidenceError(f"{context}: unexecuted model checks cannot claim parity")
                if status == "failed" and failure != "model_execution_error":
                    raise EvidenceError(
                        f"{context}.failure: failed model execution requires a closed error code"
                    )
            else:
                logits_ratio = _validate_model_error(
                    metrics["logits_error"],
                    f"{context}.metrics.logits_error",
                    maximum_index=31,
                )
                router_score_ratio = _validate_model_error(
                    metrics["router_score_error"],
                    f"{context}.metrics.router_score_error",
                    maximum_index=3,
                )
                route_ratio = _validate_model_error(
                    metrics["route_weight_error"],
                    f"{context}.metrics.route_weight_error",
                    maximum_index=1,
                )
                exact = all((tokens_exact, expert_ids_exact, deterministic))
                if status == "ok":
                    if (
                        not exact
                        or logits_ratio > 1.0
                        or router_score_ratio > 1.0
                        or route_ratio > 1.0
                    ):
                        raise EvidenceError(
                            f"{context}: successful model row lacks componentwise oracle parity"
                        )
                elif failure == "tolerance_exceeded":
                    if (
                        logits_ratio <= 1.0
                        and router_score_ratio <= 1.0
                        and route_ratio <= 1.0
                    ):
                        raise EvidenceError(
                            f"{context}: tolerance failure has no over-tolerance component"
                        )
                elif failure == "exactness_mismatch":
                    if not exact:
                        if (
                            logits_ratio > 1.0
                            or router_score_ratio > 1.0
                            or route_ratio > 1.0
                        ):
                            raise EvidenceError(
                                f"{context}: tolerance failure must take precedence"
                            )
                    else:
                        raise EvidenceError(
                            f"{context}: exactness failure has no false exactness flag"
                        )
                else:
                    raise EvidenceError(
                        f"{context}.failure: failed model row has an unknown error code"
                    )
        normalized.append(dict(row))
    return normalized


def correctness_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    return len(rows) == len(expected_correctness_checks()) and all(
        row.get("status") == "ok" for row in rows
    )


RESOURCE_KEYS = (
    "user_cpu_ns",
    "system_cpu_ns",
    "minor_page_faults",
    "major_page_faults",
    "voluntary_context_switches",
    "involuntary_context_switches",
)
WORKER_KEYS = (
    "worker_index",
    "cpu",
    "cpu_before",
    "cpu_after",
    "affinity",
    "mxcsr_before",
    "mxcsr_after",
)
OBSERVATION_KEYS = (
    "schema",
    "cell_id",
    "case_id",
    "child_sequence",
    "pair_sequence",
    "pair_order",
    "variant_sequence",
    "variant",
    "implementation",
    "status",
    "failure",
    "elapsed_ns",
    "expected_calls",
    "executed_calls",
    "sink_sha256",
    "output_sha256",
    "addresses_mod_64",
    "workers",
    "resource_usage",
)


def _validate_mxcsr(value: int, context: str) -> None:
    _uint(value, context, (1 << 32) - 1)
    forbidden = (0b11 << 13) | (1 << 15) | (1 << 6)
    if value & forbidden:
        raise EvidenceError(f"{context}: rounding mode, FTZ, or DAZ violates the contract")


def _validate_addresses(value: Any, context: str) -> None:
    addresses = _object(value, context)
    _exact_keys(addresses, ("weights", "input", "output"), context)
    for name in ("weights", "input", "output"):
        _uint(addresses[name], f"{context}.{name}", 63)
    if addresses["weights"] % 2 or addresses["input"] % 4 or addresses["output"] % 4:
        raise EvidenceError(f"{context}: buffer address violates natural alignment")


def _validate_workers(
    value: Any,
    cell: CellSpec,
    cpus: Sequence[int] | None,
    context: str,
    *,
    require_complete: bool,
    require_stable: bool,
) -> None:
    workers = _array(value, context)
    if require_complete and len(workers) != cell.workers:
        raise EvidenceError(f"{context}: expected {cell.workers} workers")
    if len(workers) > cell.workers:
        raise EvidenceError(f"{context}: too many worker records")
    if cpus is not None and len(cpus) != cell.workers:
        raise EvidenceError(f"{context}: expected {cell.workers} requested CPUs")
    seen_cpus: set[int] = set()
    seen_indices: set[int] = set()
    prior_index = -1
    for record_index, item in enumerate(workers):
        worker_context = f"{context}[{record_index}]"
        worker = _object(item, worker_context)
        _exact_keys(worker, WORKER_KEYS, worker_context)
        actual_index = _uint(
            worker["worker_index"],
            f"{worker_context}.worker_index",
            cell.workers - 1,
        )
        if actual_index in seen_indices or actual_index <= prior_index:
            raise EvidenceError(f"{worker_context}.worker_index: noncanonical order")
        seen_indices.add(actual_index)
        prior_index = actual_index
        cpu = _uint(worker["cpu"], f"{worker_context}.cpu", (1 << 31) - 1)
        before = _uint(worker["cpu_before"], f"{worker_context}.cpu_before", (1 << 31) - 1)
        after = _uint(worker["cpu_after"], f"{worker_context}.cpu_after", (1 << 31) - 1)
        affinity = _array(worker["affinity"], f"{worker_context}.affinity")
        if affinity != [cpu]:
            raise EvidenceError(f"{worker_context}: affinity does not match requested CPU")
        if require_stable and (before != cpu or after != cpu):
            raise EvidenceError(f"{worker_context}: CPU residency mismatch")
        if cpus is not None and cpu != cpus[actual_index]:
            raise EvidenceError(f"{worker_context}: worker used an unrequested CPU")
        if cpu in seen_cpus:
            raise EvidenceError(f"{worker_context}: workers share a CPU")
        seen_cpus.add(cpu)
        before_mxcsr = _uint(
            worker["mxcsr_before"], f"{worker_context}.mxcsr_before", (1 << 32) - 1
        )
        after_mxcsr = _uint(
            worker["mxcsr_after"], f"{worker_context}.mxcsr_after", (1 << 32) - 1
        )
        _validate_mxcsr(before_mxcsr, f"{worker_context}.mxcsr_before")
        if require_stable and after_mxcsr != before_mxcsr:
            raise EvidenceError(f"{worker_context}: MXCSR changed during timing")


def _validate_resource_usage(value: Any, context: str) -> None:
    usage = _object(value, context)
    _exact_keys(usage, RESOURCE_KEYS, context)
    for name in RESOURCE_KEYS:
        _uint(usage[name], f"{context}.{name}")


def _validate_success_metadata(
    row: Mapping[str, Any],
    cell: CellSpec,
    cpus: Sequence[int] | None,
    context: str,
) -> None:
    _validate_addresses(row["addresses_mod_64"], f"{context}.addresses_mod_64")
    _validate_workers(
        row["workers"],
        cell,
        cpus,
        f"{context}.workers",
        require_complete=True,
        require_stable=True,
    )
    _validate_resource_usage(row["resource_usage"], f"{context}.resource_usage")


def validate_observation(
    value: Mapping[str, Any],
    *,
    cell: CellSpec,
    child_sequence: int,
    pair_sequence: int,
    pair_order: str,
    variant_sequence: int,
    variant: str,
    cpus: Sequence[int] | None = None,
    context: str = "observation",
) -> dict[str, Any]:
    row = _object(value, context)
    _exact_keys(row, OBSERVATION_KEYS, context)
    _string(row["schema"], f"{context}.schema", expected=OBSERVATION_SCHEMA)
    _string(row["cell_id"], f"{context}.cell_id", expected=cell.cell_id)
    _string(row["case_id"], f"{context}.case_id", expected=cell.case_id)
    expected_uints = {
        "child_sequence": child_sequence,
        "pair_sequence": pair_sequence,
        "variant_sequence": variant_sequence,
    }
    for name, expected in expected_uints.items():
        actual = _uint(row[name], f"{context}.{name}", max(len(CELLS), MEASURED_PAIRS))
        if actual != expected:
            raise EvidenceError(f"{context}.{name}: expected {expected}, got {actual}")
    _string(row["pair_order"], f"{context}.pair_order", expected=pair_order)
    _string(row["variant"], f"{context}.variant", expected=variant)
    _string(
        row["implementation"],
        f"{context}.implementation",
        expected=cell.implementation(variant),
    )
    status = _string(row["status"], f"{context}.status", maximum=32)
    allowed_statuses = {
        "ok",
        "unsupported",
        "kernel_error",
        "protocol_error",
        "timeout",
        "launch_error",
        "output_limit",
        "nonzero_exit",
        "warmup_failure",
        "correctness_failure",
    }
    if status not in allowed_statuses:
        raise EvidenceError(f"{context}.status: unexpected status {status!r}")
    failure = _nullable_string(row["failure"], f"{context}.failure", maximum=512)
    if (status == "ok") != (failure is None):
        raise EvidenceError(f"{context}: failure must be null exactly when status is ok")
    case = CASE_BY_ID[cell.case_id]
    expected_calls = case.calls_per_worker * cell.workers
    actual_expected = _positive_uint(row["expected_calls"], f"{context}.expected_calls")
    if actual_expected != expected_calls:
        raise EvidenceError(f"{context}.expected_calls: expected {expected_calls}")
    executed = _uint(row["executed_calls"], f"{context}.executed_calls", expected_calls)

    if status == "ok":
        _positive_uint(row["elapsed_ns"], f"{context}.elapsed_ns")
        if executed != expected_calls:
            raise EvidenceError(f"{context}: successful row omitted a timed call")
        _prefixed_digest(row["sink_sha256"], f"{context}.sink_sha256")
        _prefixed_digest(row["output_sha256"], f"{context}.output_sha256")
        _validate_success_metadata(row, cell, cpus, context)
    else:
        if row["elapsed_ns"] is not None:
            _positive_uint(row["elapsed_ns"], f"{context}.elapsed_ns")
        for name in ("sink_sha256", "output_sha256"):
            if row[name] is not None:
                _prefixed_digest(row[name], f"{context}.{name}")
        if row["addresses_mod_64"] is not None:
            _validate_addresses(row["addresses_mod_64"], f"{context}.addresses_mod_64")
        _validate_workers(
            row["workers"],
            cell,
            cpus,
            f"{context}.workers",
            require_complete=False,
            require_stable=False,
        )
        if row["resource_usage"] is not None:
            _validate_resource_usage(row["resource_usage"], f"{context}.resource_usage")
    return dict(row)


def build_cell_request(cell: CellSpec, child_sequence: int, cpus: Sequence[int]) -> dict[str, Any]:
    if len(cpus) != cell.workers or len(set(cpus)) != len(cpus):
        raise EvidenceError(f"{cell.cell_id}: expected {cell.workers} distinct CPUs")
    return {
        "schema": CELL_REQUEST_SCHEMA,
        "cell_id": cell.cell_id,
        "case_id": cell.case_id,
        "child_sequence": child_sequence,
        "cpus": list(cpus),
        "warmup_orders": list(WARMUP_ORDERS),
        "measured_orders": measured_pair_orders(cell.cell_id),
    }


def validate_cell_stream_prefix(
    values: Sequence[Mapping[str, Any]],
    *,
    cell: CellSpec,
    child_sequence: int,
    cpus: Sequence[int],
) -> tuple[list[dict[str, Any]], bool]:
    """Validate an exact prefix of the five-warmup/sixty-observation stream."""

    if len(values) > len(WARMUP_ORDERS) + 2 * MEASURED_PAIRS:
        raise EvidenceError(f"cell stream {cell.cell_id}: more than 65 records")
    warmups_ok = True
    warmup_count = min(len(values), len(WARMUP_ORDERS))
    for pair_sequence in range(warmup_count):
        context = f"cell stream {cell.cell_id}.warmups[{pair_sequence}]"
        warmup = _object(values[pair_sequence], context)
        _exact_keys(
            warmup,
            (
                "schema",
                "cell_id",
                "case_id",
                "child_sequence",
                "pair_sequence",
                "order",
                "baseline_status",
                "candidate_status",
            ),
            context,
        )
        _string(warmup["schema"], f"{context}.schema", expected=WARMUP_SCHEMA)
        _string(warmup["cell_id"], f"{context}.cell_id", expected=cell.cell_id)
        _string(warmup["case_id"], f"{context}.case_id", expected=cell.case_id)
        if _uint(
            warmup["child_sequence"],
            f"{context}.child_sequence",
            len(CELLS) - 1,
        ) != child_sequence:
            raise EvidenceError(f"{context}: child sequence mismatch")
        if _uint(warmup["pair_sequence"], f"{context}.pair_sequence", 4) != pair_sequence:
            raise EvidenceError(f"{context}: pair sequence mismatch")
        _string(
            warmup["order"],
            f"{context}.order",
            expected=WARMUP_ORDERS[pair_sequence],
        )
        for name in ("baseline_status", "candidate_status"):
            status = _string(warmup[name], f"{context}.{name}", maximum=32)
            if status not in {"ok", "unsupported", "kernel_error"}:
                raise EvidenceError(f"{context}.{name}: unexpected status {status!r}")
            warmups_ok &= status == "ok"

    if len(values) <= len(WARMUP_ORDERS):
        return [], warmups_ok

    normalized: list[dict[str, Any]] = []
    for cursor, value in enumerate(values[len(WARMUP_ORDERS) :]):
        pair_sequence, variant_sequence = divmod(cursor, 2)
        order = measured_pair_orders(cell.cell_id)[pair_sequence]
        variant = _variant_order(order)[variant_sequence]
        normalized.append(
            validate_observation(
                value,
                cell=cell,
                child_sequence=child_sequence,
                pair_sequence=pair_sequence,
                pair_order=order,
                variant_sequence=variant_sequence,
                variant=variant,
                cpus=cpus,
                context=f"cell stream {cell.cell_id}.observations[{cursor}]",
            )
        )
    return normalized, warmups_ok


def synthesize_failure_rows(
    cell: CellSpec,
    child_sequence: int,
    status: str,
    failure: str,
    start_observation: int = 0,
) -> list[dict[str, Any]]:
    case = CASE_BY_ID[cell.case_id]
    rows: list[dict[str, Any]] = []
    cursor = 0
    for pair_sequence, order in enumerate(measured_pair_orders(cell.cell_id)):
        for variant_sequence, variant in enumerate(_variant_order(order)):
            if cursor < start_observation:
                cursor += 1
                continue
            rows.append(
                {
                    "schema": OBSERVATION_SCHEMA,
                    "cell_id": cell.cell_id,
                    "case_id": cell.case_id,
                    "child_sequence": child_sequence,
                    "pair_sequence": pair_sequence,
                    "pair_order": order,
                    "variant_sequence": variant_sequence,
                    "variant": variant,
                    "implementation": cell.implementation(variant),
                    "status": status,
                    "failure": failure[:512],
                    "elapsed_ns": None,
                    "expected_calls": case.calls_per_worker * cell.workers,
                    "executed_calls": 0,
                    "sink_sha256": None,
                    "output_sha256": None,
                    "addresses_mod_64": None,
                    "workers": [],
                    "resource_usage": None,
                }
            )
            cursor += 1
    return rows


def validate_dataset(
    values: Sequence[Mapping[str, Any]],
    *,
    benchmark_cpus: tuple[Sequence[int], Sequence[int] | None] | None = None,
) -> list[dict[str, Any]]:
    if len(values) != EXPECTED_OBSERVATIONS:
        raise EvidenceError(
            f"observations.jsonl: expected {EXPECTED_OBSERVATIONS} rows, got {len(values)}"
        )
    normalized: list[dict[str, Any]] = []
    cursor = 0
    for child_sequence, cell_id in enumerate(cell_order()):
        cell = CELL_BY_ID[cell_id]
        expected_cpus: Sequence[int] | None = None
        if benchmark_cpus is not None:
            expected_cpus = benchmark_cpus[0] if cell.workers == 1 else benchmark_cpus[1]
        cell_worker_cpus: tuple[int, ...] | None = None
        cell_variant_proofs: dict[str, tuple[str, str, bytes]] = {}
        for pair_sequence, order in enumerate(measured_pair_orders(cell_id)):
            pair_rows: list[dict[str, Any]] = []
            for variant_sequence, variant in enumerate(_variant_order(order)):
                row = validate_observation(
                    values[cursor],
                    cell=cell,
                    child_sequence=child_sequence,
                    pair_sequence=pair_sequence,
                    pair_order=order,
                    variant_sequence=variant_sequence,
                    variant=variant,
                    cpus=expected_cpus,
                    context=f"observations.jsonl:{cursor + 1}",
                )
                if benchmark_cpus is not None and expected_cpus is None and row["workers"]:
                    raise EvidenceError(
                        f"observations cell {cell_id}: recorded workers lack a valid frozen CPU set"
                    )
                if len(row["workers"]) == cell.workers:
                    worker_cpus = tuple(worker["cpu"] for worker in row["workers"])
                    if cell_worker_cpus is None:
                        cell_worker_cpus = worker_cpus
                    elif worker_cpus != cell_worker_cpus:
                        raise EvidenceError(
                            f"observations cell {cell_id}: rows used different worker CPUs"
                        )
                if row["status"] == "ok":
                    proof = (
                        row["sink_sha256"],
                        row["output_sha256"],
                        canonical_json(row["addresses_mod_64"]),
                    )
                    prior_proof = cell_variant_proofs.setdefault(variant, proof)
                    if proof != prior_proof:
                        raise EvidenceError(
                            f"observations cell {cell_id}/{variant}: repeated identical batches "
                            "changed sink, output, or buffer addresses"
                        )
                pair_rows.append(row)
                normalized.append(row)
                cursor += 1
            if all(row["status"] == "ok" for row in pair_rows):
                if pair_rows[0]["addresses_mod_64"] != pair_rows[1]["addresses_mod_64"]:
                    raise EvidenceError(
                        f"observations pair {cell_id}/{pair_sequence}: variants used different buffers"
                    )
                pair_cpus = [
                    tuple(worker["cpu"] for worker in row["workers"])
                    for row in pair_rows
                ]
                if pair_cpus[0] != pair_cpus[1]:
                    raise EvidenceError(
                        f"observations pair {cell_id}/{pair_sequence}: variants used different CPUs"
                    )
    return normalized


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise EvidenceError("median requires at least one value")
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def descriptive_statistics(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values or any(type(value) not in {int, float} for value in values):
        raise EvidenceError("descriptive statistics require nonempty finite numbers, not booleans")
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numbers):
        raise EvidenceError("descriptive statistics require finite values")
    ordered = sorted(numbers)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "sample_standard_deviation": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "minimum": ordered[0],
        "p50": _median(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def bootstrap_median_interval(
    values: Sequence[int | float],
    *,
    cell_id: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if len(values) != MEASURED_PAIRS:
        raise EvidenceError("M4 bootstrap requires exactly 30 complete paired ratios")
    if type(resamples) is not int or not 1 <= resamples <= BOOTSTRAP_RESAMPLES:
        raise EvidenceError("bootstrap resamples must be in 1..=10000")
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in numbers):
        raise EvidenceError("bootstrap ratios must be finite and nonnegative")
    prefix = _cell_material(b"runnel-m4-bootstrap-v1\0", cell_id)
    stream = Sha256U64Stream(prefix)
    replicate_medians: list[float] = []
    for _ in range(resamples):
        sample = [numbers[stream.below(MEASURED_PAIRS)] for _ in range(MEASURED_PAIRS)]
        replicate_medians.append(_median(sample))
    replicate_medians.sort()
    low_index = max(0, math.ceil(0.025 * resamples) - 1)
    high_index = max(0, math.ceil(0.975 * resamples) - 1)
    return {
        "method": "deterministic-sha256-percentile",
        "confidence_level": 0.95,
        "resamples": resamples,
        "low_index": low_index,
        "high_index": high_index,
        "low": replicate_medians[low_index],
        "high": replicate_medians[high_index],
        "stream_prefix_sha256": f"sha256:{sha256_bytes(prefix)}",
    }


def build_summary(
    observations: Sequence[Mapping[str, Any]],
    correctness: Sequence[Mapping[str, Any]],
    *,
    cases_digest: str,
    correctness_digest: str,
    observations_digest: str,
    model_contract_cohort: str = CURRENT_MODEL_CONTRACT_COHORT,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    rows = validate_dataset(observations)
    correctness_rows = validate_correctness(
        correctness,
        model_contract_cohort=model_contract_cohort,
    )
    all_correct = correctness_passed(correctness_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["cell_id"]].append(row)

    cells: list[dict[str, Any]] = []
    for cell in CELLS:
        cell_rows = grouped[cell.cell_id]
        statuses = Counter(row["status"] for row in cell_rows)
        complete = all_correct and len(cell_rows) == 60 and statuses == {"ok": 60}
        summary: dict[str, Any] = {
            "cell_id": cell.cell_id,
            "case_id": cell.case_id,
            "role": cell.role,
            "allocation": cell.allocation,
            "workers": cell.workers,
            "baseline": BASELINE_IMPLEMENTATION,
            "candidate": cell.candidate,
            "complete": complete,
            "complete_pairs": MEASURED_PAIRS if complete else 0,
            "status_counts": dict(sorted(statuses.items())),
            "baseline_elapsed_ns": None,
            "candidate_elapsed_ns": None,
            "paired_candidate_over_baseline": None,
            "paired_candidate_minus_baseline_ns": None,
            "baseline_element_products_per_second": None,
            "candidate_element_products_per_second": None,
            "baseline_logical_bf16_bytes_per_second": None,
            "candidate_logical_bf16_bytes_per_second": None,
            "exploratory_lower_elapsed_time": None,
        }
        if complete:
            pair_map: dict[int, dict[str, int]] = defaultdict(dict)
            for row in cell_rows:
                pair_map[row["pair_sequence"]][row["variant"]] = row["elapsed_ns"]
            if set(pair_map) != set(range(MEASURED_PAIRS)) or any(
                set(pair) != {"baseline", "candidate"} for pair in pair_map.values()
            ):
                raise EvidenceError(f"summary {cell.cell_id}: incomplete pair join")
            baseline = [pair_map[index]["baseline"] for index in range(MEASURED_PAIRS)]
            candidate = [pair_map[index]["candidate"] for index in range(MEASURED_PAIRS)]
            ratios = [right / left for left, right in zip(baseline, candidate, strict=True)]
            differences = [right - left for left, right in zip(baseline, candidate, strict=True)]
            interval = bootstrap_median_interval(
                ratios, cell_id=cell.cell_id, resamples=bootstrap_resamples
            )
            case = CASE_BY_ID[cell.case_id]
            products = case.element_products_per_worker * cell.workers
            logical_bytes = case.matrix_bytes * case.calls_per_worker * cell.workers
            baseline_products = [products * 1e9 / value for value in baseline]
            candidate_products = [products * 1e9 / value for value in candidate]
            baseline_bytes = [logical_bytes * 1e9 / value for value in baseline]
            candidate_bytes = [logical_bytes * 1e9 / value for value in candidate]
            summary.update(
                {
                    "baseline_elapsed_ns": descriptive_statistics(baseline),
                    "candidate_elapsed_ns": descriptive_statistics(candidate),
                    "paired_candidate_over_baseline": {
                        "descriptive": descriptive_statistics(ratios),
                        "median_95_percentile_bootstrap": interval,
                    },
                    "paired_candidate_minus_baseline_ns": descriptive_statistics(differences),
                    "baseline_element_products_per_second": descriptive_statistics(
                        baseline_products
                    ),
                    "candidate_element_products_per_second": descriptive_statistics(
                        candidate_products
                    ),
                    "baseline_logical_bf16_bytes_per_second": descriptive_statistics(
                        baseline_bytes
                    ),
                    "candidate_logical_bf16_bytes_per_second": descriptive_statistics(
                        candidate_bytes
                    ),
                    "exploratory_lower_elapsed_time": interval["high"] < 1.0,
                }
            )
        cells.append(summary)

    by_id = {cell["cell_id"]: cell for cell in cells}
    primary_ids = ("avx-natural-stream-expand", "avx-natural-stream-contract")
    primary = [by_id[cell_id] for cell_id in primary_ids]
    eligible = all(cell["complete"] for cell in primary)
    satisfied: bool | None = None
    if eligible:
        satisfied = all(
            cell["paired_candidate_over_baseline"]["descriptive"]["p50"] <= 0.95
            and cell["paired_candidate_over_baseline"]["median_95_percentile_bootstrap"][
                "high"
            ]
            < 1.0
            for cell in primary
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "sources": {
            "cases": cases_digest,
            "correctness": correctness_digest,
            "observations": observations_digest,
        },
        "correctness_passed": all_correct,
        "methodology": {
            "paired_by": ["cell_id", "pair_sequence"],
            "warmup_pairs_excluded": len(WARMUP_ORDERS),
            "measured_pairs_per_cell": MEASURED_PAIRS,
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": 0.95,
            "intervals_unadjusted": True,
            "cells_pooled": False,
            "outliers_removed": False,
            "favorable_result_required": False,
            "logical_rates_are_hardware_bandwidth": False,
        },
        "cells": cells,
        "general_claim_rule": {
            "cell_ids": list(primary_ids),
            "eligible": eligible,
            "satisfied": satisfied,
            "observed_median_ratio_maximum": 0.95,
            "requires_interval_upper_below_one": True,
            "confidence_bounded_minimum_effect_claimed": False,
        },
    }


def _svg_document(title: str, description: str, body: Sequence[str], *, height: int) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">'
        ),
        f"<title id=\"title\">{html.escape(title)}</title>",
        f"<desc id=\"desc\">{html.escape(description)}</desc>",
        "<style>text{font-family:system-ui,sans-serif;fill:#18222d}.title{font-size:22px;font-weight:700}.label{font-size:13px}.small{font-size:11px;fill:#4c5967}.axis{stroke:#778493;stroke-width:1}.baseline{fill:#506d85}.candidate{fill:#d26a3f}.interval{stroke:#202b36;stroke-width:2}.incomplete{fill:#8b949e}</style>",
        *body,
        "</svg>",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_paired_ratios_svg(summary: Mapping[str, Any]) -> bytes:
    cells = _array(summary["cells"], "summary.cells")
    height = 130 + len(cells) * 42
    left, right = 350.0, 1_140.0
    width = right - left
    maximum = 2.0
    lines = [
        '<text class="title" x="30" y="34">Paired candidate / baseline elapsed time</text>',
        '<text class="small" x="30" y="56">Lower is better · median and unadjusted 95% percentile-bootstrap interval · no pooling</text>',
    ]
    for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
        x = left + width * tick / maximum
        lines.append(f'<line class="axis" x1="{x:.2f}" y1="70" x2="{x:.2f}" y2="{height - 45}"/>')
        lines.append(f'<text class="small" x="{x:.2f}" y="{height - 27}" text-anchor="middle">{tick:.1f}</text>')
    for index, cell_value in enumerate(cells):
        cell = _object(cell_value, f"summary.cells[{index}]")
        y = 90 + index * 42
        label = html.escape(_string(cell["cell_id"], f"summary.cells[{index}].cell_id"))
        lines.append(f'<text class="label" x="30" y="{y + 4}">{label}</text>')
        if not _boolean(cell["complete"], f"summary.cells[{index}].complete"):
            lines.append(f'<text class="incomplete label" x="{left:.2f}" y="{y + 4}">incomplete</text>')
            continue
        paired = _object(
            cell["paired_candidate_over_baseline"],
            f"summary.cells[{index}].paired_candidate_over_baseline",
        )
        descriptive = _object(paired["descriptive"], "paired.descriptive")
        interval = _object(paired["median_95_percentile_bootstrap"], "paired.interval")
        median = min(max(_number(descriptive["p50"], "paired.p50"), 0.0), maximum)
        low = min(max(_number(interval["low"], "paired.low"), 0.0), maximum)
        high = min(max(_number(interval["high"], "paired.high"), 0.0), maximum)
        x_median = left + width * median / maximum
        x_low = left + width * low / maximum
        x_high = left + width * high / maximum
        lines.append(f'<line class="interval" x1="{x_low:.2f}" y1="{y}" x2="{x_high:.2f}" y2="{y}"/>')
        lines.append(f'<circle class="candidate" cx="{x_median:.2f}" cy="{y}" r="5"/>')
    lines.append(
        f'<text class="small" x="30" y="{height - 8}">Source: summary.json regenerated from observations.jsonl · ratios clipped at 2.0 for display</text>'
    )
    return _svg_document(
        "M4 paired elapsed-time ratios",
        "Median candidate divided by baseline elapsed time with unadjusted bootstrap intervals for each preregistered cell.",
        lines,
        height=height,
    )


def render_elapsed_time_svg(summary: Mapping[str, Any]) -> bytes:
    cells = _array(summary["cells"], "summary.cells")
    complete = [
        _object(cell, f"summary.cells[{index}]")
        for index, cell in enumerate(cells)
        if _object(cell, f"summary.cells[{index}]").get("complete") is True
    ]
    maxima = []
    for cell in complete:
        maxima.extend(
            [
                _number(cell["baseline_elapsed_ns"]["p50"], "baseline p50"),
                _number(cell["candidate_elapsed_ns"]["p50"], "candidate p50"),
            ]
        )
    scale_max = max(maxima, default=1.0)
    height = 130 + len(cells) * 48
    left, right = 350.0, 1_140.0
    width = right - left
    lines = [
        '<text class="title" x="30" y="34">Median batch elapsed time by implementation</text>',
        '<text class="small" x="30" y="56">Absolute times are cell-specific shared-host measurements; bar lengths are not comparable workloads</text>',
    ]
    for index, value in enumerate(cells):
        cell = _object(value, f"summary.cells[{index}]")
        y = 82 + index * 48
        label = html.escape(_string(cell["cell_id"], f"summary.cells[{index}].cell_id"))
        lines.append(f'<text class="label" x="30" y="{y + 14}">{label}</text>')
        if cell["complete"] is not True:
            lines.append(f'<text class="incomplete label" x="{left:.2f}" y="{y + 14}">incomplete</text>')
            continue
        baseline = _number(cell["baseline_elapsed_ns"]["p50"], "baseline p50")
        candidate = _number(cell["candidate_elapsed_ns"]["p50"], "candidate p50")
        baseline_width = width * baseline / scale_max
        candidate_width = width * candidate / scale_max
        lines.append(f'<rect class="baseline" x="{left:.2f}" y="{y}" width="{baseline_width:.2f}" height="12"/>')
        lines.append(f'<rect class="candidate" x="{left:.2f}" y="{y + 16}" width="{candidate_width:.2f}" height="12"/>')
    lines.append(
        f'<text class="small" x="30" y="{height - 8}">Source: summary.json regenerated from observations.jsonl · blue baseline · orange candidate</text>'
    )
    return _svg_document(
        "M4 median elapsed time",
        "Median baseline and candidate batch elapsed times for every complete preregistered cell.",
        lines,
        height=height,
    )


def render_figures(summary: Mapping[str, Any]) -> dict[str, bytes]:
    return {
        "figures/elapsed-time.svg": render_elapsed_time_svg(summary),
        "figures/paired-ratios.svg": render_paired_ratios_svg(summary),
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


def _read_small_text(path: Path, maximum: int = 64 * 1024) -> str | None:
    try:
        with path.open("rb") as source:
            data = source.read(maximum + 1)
    except OSError:
        return None
    if len(data) > maximum:
        return None
    try:
        return data.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return None


def _memory_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    text = _read_small_text(Path("/proc/meminfo"), 1024 * 1024)
    if text is not None:
        for line in text.splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                fields = remainder.split()
                if len(fields) == 2 and fields[1] == "kB" and fields[0].isdigit():
                    values[name] = int(fields[0]) * 1024
    return {
        "total_bytes": values.get("MemTotal", 0),
        "available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
    }


def _cpu_model_and_features() -> tuple[str, list[str]]:
    text = _read_small_text(Path("/proc/cpuinfo"), 4 * 1024 * 1024)
    model = "unavailable"
    features: list[str] = []
    if text is not None:
        for line in text.splitlines():
            name, separator, value = line.partition(":")
            if not separator:
                continue
            key = name.strip()
            value = value.strip()
            if model == "unavailable" and key in {"model name", "Hardware"} and value:
                model = value[:256]
            if not features and key in {"flags", "Features"}:
                features = sorted(set(value.split()))[:512]
    return model, features


def _parse_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    if not value:
        return result
    for field in value.split(","):
        if "-" in field:
            left, separator, right = field.partition("-")
            if not separator or not left.isdigit() or not right.isdigit():
                raise EvidenceError(f"invalid CPU list field {field!r}")
            start, end = int(left), int(right)
            if start > end or end > (1 << 31) - 1:
                raise EvidenceError(f"invalid CPU list range {field!r}")
            result.extend(range(start, end + 1))
        elif field.isdigit():
            result.append(int(field))
        else:
            raise EvidenceError(f"invalid CPU list field {field!r}")
    if result != sorted(set(result)):
        raise EvidenceError("CPU list is not sorted and unique")
    return result


def _cpu_topology(allowed_cpus: Sequence[int]) -> list[dict[str, Any]]:
    result = []
    for cpu in allowed_cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package_text = _read_small_text(base / "physical_package_id")
        core_text = _read_small_text(base / "core_id")
        siblings_text = _read_small_text(base / "thread_siblings_list")
        package = int(package_text) if package_text and package_text.isdigit() else 0
        core = int(core_text) if core_text and core_text.isdigit() else cpu
        try:
            siblings = _parse_cpu_list(siblings_text or str(cpu))
        except EvidenceError:
            siblings = [cpu]
        result.append(
            {
                "cpu": cpu,
                "physical_package_id": package,
                "core_id": core,
                "thread_siblings": siblings,
            }
        )
    return result


def _cache_topology(cpu: int) -> list[dict[str, Any]]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
    rows = []
    try:
        entries = sorted(base.glob("index[0-9]*"), key=lambda path: path.name)
    except OSError:
        entries = []
    for entry in entries[:32]:
        level = _read_small_text(entry / "level") or "unavailable"
        kind = _read_small_text(entry / "type") or "unavailable"
        size = _read_small_text(entry / "size") or "unavailable"
        shared = _read_small_text(entry / "shared_cpu_list") or str(cpu)
        rows.append(
            {
                "index": entry.name,
                "level": level[:32],
                "type": kind[:32],
                "size": size[:32],
                "shared_cpu_list": shared[:256],
            }
        )
    return rows


def _frequency_metadata(allowed_cpus: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for cpu in allowed_cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        rows.append(
            {
                "cpu": cpu,
                "governor": (_read_small_text(base / "scaling_governor") or "unavailable")[:64],
                "scaling_cur_freq_khz": (
                    _read_small_text(base / "scaling_cur_freq") or "unavailable"
                )[:64],
                "cpuinfo_max_freq_khz": (
                    _read_small_text(base / "cpuinfo_max_freq") or "unavailable"
                )[:64],
            }
        )
    return rows


def _derive_benchmark_cpus(
    allowed_cpus: Sequence[int], topology: Sequence[Mapping[str, Any]]
) -> tuple[list[int], list[int] | None]:
    """Derive the frozen lowest-CPU sets from recorded package/core identities."""

    if not allowed_cpus or len(topology) != len(allowed_cpus):
        raise EvidenceError("benchmark CPU selection requires complete recorded topology")
    if list(allowed_cpus) != sorted(set(allowed_cpus)):
        raise EvidenceError("benchmark CPU selection requires sorted unique allowed CPUs")
    for index, (cpu, row) in enumerate(zip(allowed_cpus, topology, strict=True)):
        if row.get("cpu") != cpu:
            raise EvidenceError(f"benchmark CPU topology row {index} is out of order")
    first = allowed_cpus[0]
    first_identity = (
        topology[0].get("physical_package_id"),
        topology[0].get("core_id"),
    )
    second = next(
        (
            row["cpu"]
            for row in topology[1:]
            if (row.get("physical_package_id"), row.get("core_id")) != first_identity
        ),
        None,
    )
    return [first], None if second is None else [first, second]


def _numa_nodes() -> list[int]:
    nodes: list[int] = []
    try:
        entries = Path("/sys/devices/system/node").glob("node[0-9]*")
        for path in entries:
            suffix = path.name[4:]
            if suffix.isdigit():
                nodes.append(int(suffix))
    except OSError:
        return []
    return sorted(set(nodes))


def _rusage_snapshot() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_cpu_ns": int(usage.ru_utime * 1_000_000_000),
        "system_cpu_ns": int(usage.ru_stime * 1_000_000_000),
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _filesystem_available(path: Path) -> int:
    filesystem = os.statvfs(path)
    return filesystem.f_bavail * filesystem.f_frsize


def build_environment(
    *,
    captured_at: str,
    harness_sha256: str,
    kernel_binary_sha256: str,
    model_binary_sha256: str,
    cargo_version: str,
    rustc_version: str,
    cc_path: str,
    cc_sha256: str,
    cc_version: str,
    ar_path: str,
    ar_sha256: str,
    ar_version: str,
    benchmark_cpus: tuple[Sequence[int], Sequence[int] | None],
    build_root_available_bytes: int,
    build_root_remaining_bytes: int,
    process_before: Mapping[str, int],
    process_after: Mapping[str, int],
) -> dict[str, Any]:
    allowed_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    model, features = _cpu_model_and_features()
    topology = _cpu_topology(allowed_cpus)
    derived_benchmark_cpus = _derive_benchmark_cpus(allowed_cpus, topology)
    normalized_benchmark_cpus = (
        list(benchmark_cpus[0]),
        None if benchmark_cpus[1] is None else list(benchmark_cpus[1]),
    )
    if normalized_benchmark_cpus != derived_benchmark_cpus:
        raise EvidenceError("selected benchmark CPUs differ from the recorded topology")
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
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
            "model": model,
            "features": features,
            "allowed_cpus": allowed_cpus,
            "topology": topology,
            "benchmark_sets": {
                "one_worker": normalized_benchmark_cpus[0],
                "two_worker": normalized_benchmark_cpus[1],
            },
            "caches": _cache_topology(allowed_cpus[0]) if allowed_cpus else [],
            "numa_nodes": _numa_nodes(),
            "frequency": _frequency_metadata(allowed_cpus),
        },
        "load_average": {"one": load[0], "five": load[1], "fifteen": load[2]},
        "memory": _memory_snapshot(),
        "filesystem": {"repository_available_bytes": _filesystem_available(ROOT)},
        "build_filesystem": {
            "kind": "tmpfs",
            "available_bytes_before": build_root_available_bytes,
            "available_bytes_after": build_root_remaining_bytes,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "toolchain": {
            "cargo": cargo_version,
            "rustc": rustc_version,
            "cc": {
                "path": cc_path,
                "sha256": f"sha256:{cc_sha256}",
                "version": cc_version,
            },
            "ar": {
                "path": ar_path,
                "sha256": f"sha256:{ar_sha256}",
                "version": ar_version,
            },
        },
        "controlled_environment": dict(CONTROLLED_ENVIRONMENT),
        "process_resources": {"before": dict(process_before), "after": dict(process_after)},
        "harness": {"path": HARNESS_PATH, "sha256": f"sha256:{harness_sha256}"},
        "binaries": {
            "kernel": {
                "path": BINARY_LOGICAL_PATH,
                "sha256": f"sha256:{kernel_binary_sha256}",
            },
            "model": {
                "path": MODEL_BINARY_LOGICAL_PATH,
                "sha256": f"sha256:{model_binary_sha256}",
            },
        },
    }


def _validate_environment(value: Any) -> dict[str, Any]:
    environment = _object(value, "environment")
    _exact_keys(
        environment,
        (
            "schema",
            "harness_schema",
            "captured_at_utc",
            "operating_system",
            "cpu",
            "load_average",
            "memory",
            "filesystem",
            "build_filesystem",
            "python",
            "toolchain",
            "controlled_environment",
            "process_resources",
            "harness",
            "binaries",
        ),
        "environment",
    )
    _string(environment["schema"], "environment.schema", expected=ENVIRONMENT_SCHEMA)
    _string(
        environment["harness_schema"],
        "environment.harness_schema",
        expected=HARNESS_SCHEMA,
    )
    _validate_timestamp(environment["captured_at_utc"], "environment.captured_at_utc")
    operating_system = _object(environment["operating_system"], "environment.operating_system")
    _exact_keys(operating_system, ("system", "release", "machine"), "environment.operating_system")
    for name in operating_system:
        _string(operating_system[name], f"environment.operating_system.{name}", maximum=256)

    cpu = _object(environment["cpu"], "environment.cpu")
    _exact_keys(
        cpu,
        (
            "logical_count",
            "model",
            "features",
            "allowed_cpus",
            "topology",
            "benchmark_sets",
            "caches",
            "numa_nodes",
            "frequency",
        ),
        "environment.cpu",
    )
    _uint(cpu["logical_count"], "environment.cpu.logical_count", 1 << 20)
    _string(cpu["model"], "environment.cpu.model", maximum=256)
    features = _array(cpu["features"], "environment.cpu.features")
    if len(features) > 512 or features != sorted(set(features)):
        raise EvidenceError("environment.cpu.features: expected sorted unique bounded features")
    for index, feature in enumerate(features):
        _string(feature, f"environment.cpu.features[{index}]", maximum=64)
    allowed = _array(cpu["allowed_cpus"], "environment.cpu.allowed_cpus")
    allowed_values = [_uint(item, "environment.cpu.allowed_cpus[]", (1 << 31) - 1) for item in allowed]
    if allowed_values != sorted(set(allowed_values)):
        raise EvidenceError("environment.cpu.allowed_cpus: expected sorted unique CPUs")
    topology = _array(cpu["topology"], "environment.cpu.topology")
    if len(topology) != len(allowed_values):
        raise EvidenceError("environment.cpu.topology: one row per allowed CPU is required")
    for index, value in enumerate(topology):
        context = f"environment.cpu.topology[{index}]"
        row = _object(value, context)
        _exact_keys(
            row,
            ("cpu", "physical_package_id", "core_id", "thread_siblings"),
            context,
        )
        if _uint(row["cpu"], f"{context}.cpu", (1 << 31) - 1) != allowed_values[index]:
            raise EvidenceError(f"{context}.cpu: topology order mismatch")
        _uint(row["physical_package_id"], f"{context}.physical_package_id", (1 << 31) - 1)
        _uint(row["core_id"], f"{context}.core_id", (1 << 31) - 1)
        siblings = _array(row["thread_siblings"], f"{context}.thread_siblings")
        sibling_values = [_uint(item, f"{context}.thread_siblings[]", (1 << 31) - 1) for item in siblings]
        if sibling_values != sorted(set(sibling_values)):
            raise EvidenceError(f"{context}.thread_siblings: expected sorted unique CPUs")
    benchmark_sets = _object(cpu["benchmark_sets"], "environment.cpu.benchmark_sets")
    _exact_keys(
        benchmark_sets,
        ("one_worker", "two_worker"),
        "environment.cpu.benchmark_sets",
    )
    one_worker = _array(
        benchmark_sets["one_worker"], "environment.cpu.benchmark_sets.one_worker"
    )
    one_worker_values = [
        _uint(item, "environment.cpu.benchmark_sets.one_worker[]", (1 << 31) - 1)
        for item in one_worker
    ]
    two_worker_value = benchmark_sets["two_worker"]
    if two_worker_value is None:
        two_worker_values = None
    else:
        two_worker = _array(
            two_worker_value, "environment.cpu.benchmark_sets.two_worker"
        )
        two_worker_values = [
            _uint(item, "environment.cpu.benchmark_sets.two_worker[]", (1 << 31) - 1)
            for item in two_worker
        ]
    if (one_worker_values, two_worker_values) != _derive_benchmark_cpus(
        allowed_values, topology
    ):
        raise EvidenceError(
            "environment.cpu.benchmark_sets: sets differ from the lowest eligible CPUs"
        )
    caches = _array(cpu["caches"], "environment.cpu.caches")
    if len(caches) > 32:
        raise EvidenceError("environment.cpu.caches: too many entries")
    for index, value in enumerate(caches):
        context = f"environment.cpu.caches[{index}]"
        row = _object(value, context)
        _exact_keys(row, ("index", "level", "type", "size", "shared_cpu_list"), context)
        for name in row:
            _string(row[name], f"{context}.{name}", maximum=256)
    nodes = _array(cpu["numa_nodes"], "environment.cpu.numa_nodes")
    node_values = [_uint(item, "environment.cpu.numa_nodes[]", (1 << 31) - 1) for item in nodes]
    if node_values != sorted(set(node_values)):
        raise EvidenceError("environment.cpu.numa_nodes: expected sorted unique nodes")
    frequency = _array(cpu["frequency"], "environment.cpu.frequency")
    if len(frequency) != len(allowed_values):
        raise EvidenceError("environment.cpu.frequency: one row per allowed CPU is required")
    for index, value in enumerate(frequency):
        context = f"environment.cpu.frequency[{index}]"
        row = _object(value, context)
        _exact_keys(
            row,
            ("cpu", "governor", "scaling_cur_freq_khz", "cpuinfo_max_freq_khz"),
            context,
        )
        if _uint(row["cpu"], f"{context}.cpu", (1 << 31) - 1) != allowed_values[index]:
            raise EvidenceError(f"{context}.cpu: frequency order mismatch")
        for name in ("governor", "scaling_cur_freq_khz", "cpuinfo_max_freq_khz"):
            _string(row[name], f"{context}.{name}", maximum=64)

    load = _object(environment["load_average"], "environment.load_average")
    _exact_keys(load, ("one", "five", "fifteen"), "environment.load_average")
    for name in load:
        _number(load[name], f"environment.load_average.{name}", nonnegative=True)
    memory = _object(environment["memory"], "environment.memory")
    _exact_keys(
        memory,
        ("total_bytes", "available_bytes", "swap_total_bytes", "swap_free_bytes"),
        "environment.memory",
    )
    for name in memory:
        _uint(memory[name], f"environment.memory.{name}")
    if memory["available_bytes"] > memory["total_bytes"]:
        raise EvidenceError("environment.memory: available memory exceeds total memory")
    if memory["swap_free_bytes"] > memory["swap_total_bytes"]:
        raise EvidenceError("environment.memory: free swap exceeds total swap")
    filesystem = _object(environment["filesystem"], "environment.filesystem")
    _exact_keys(filesystem, ("repository_available_bytes",), "environment.filesystem")
    _uint(filesystem["repository_available_bytes"], "environment.filesystem.repository_available_bytes")
    build_filesystem = _object(environment["build_filesystem"], "environment.build_filesystem")
    _exact_keys(
        build_filesystem,
        ("kind", "available_bytes_before", "available_bytes_after"),
        "environment.build_filesystem",
    )
    _string(build_filesystem["kind"], "environment.build_filesystem.kind", expected="tmpfs")
    before = _uint(
        build_filesystem["available_bytes_before"],
        "environment.build_filesystem.available_bytes_before",
    )
    after = _uint(
        build_filesystem["available_bytes_after"],
        "environment.build_filesystem.available_bytes_after",
    )
    if before < MIN_FREE_BYTES_AFTER_CAPTURE or after < MIN_FREE_BYTES_AFTER_CAPTURE:
        raise EvidenceError("environment.build_filesystem: tmpfs reserve was not maintained")
    python = _object(environment["python"], "environment.python")
    _exact_keys(python, ("implementation", "version"), "environment.python")
    _string(python["implementation"], "environment.python.implementation", maximum=64)
    _string(python["version"], "environment.python.version", maximum=64)
    toolchain = _object(environment["toolchain"], "environment.toolchain")
    _exact_keys(toolchain, ("cargo", "rustc", "cc", "ar"), "environment.toolchain")
    _string(toolchain["cargo"], "environment.toolchain.cargo", maximum=256)
    _string(toolchain["rustc"], "environment.toolchain.rustc", maximum=256)
    for name in ("cc", "ar"):
        identity = _object(toolchain[name], f"environment.toolchain.{name}")
        _exact_keys(identity, ("path", "sha256", "version"), f"environment.toolchain.{name}")
        path = _string(identity["path"], f"environment.toolchain.{name}.path", maximum=512)
        if not path.startswith("/"):
            raise EvidenceError(f"environment.toolchain.{name}.path: expected absolute path")
        _prefixed_digest(identity["sha256"], f"environment.toolchain.{name}.sha256")
        _string(identity["version"], f"environment.toolchain.{name}.version", maximum=256)
    controlled = _object(environment["controlled_environment"], "environment.controlled_environment")
    if controlled != CONTROLLED_ENVIRONMENT:
        raise EvidenceError("environment.controlled_environment: unexpected environment")
    resources = _object(environment["process_resources"], "environment.process_resources")
    _exact_keys(resources, ("before", "after"), "environment.process_resources")
    for phase in ("before", "after"):
        snapshot = _object(resources[phase], f"environment.process_resources.{phase}")
        _exact_keys(snapshot, RESOURCE_KEYS, f"environment.process_resources.{phase}")
        for name in RESOURCE_KEYS:
            _uint(snapshot[name], f"environment.process_resources.{phase}.{name}")
    for name in RESOURCE_KEYS:
        if resources["after"][name] < resources["before"][name]:
            raise EvidenceError(
                f"environment.process_resources.{name}: cumulative counter decreased"
            )
    harness = _object(environment["harness"], "environment.harness")
    _exact_keys(harness, ("path", "sha256"), "environment.harness")
    _string(harness["path"], "environment.harness.path", expected=HARNESS_PATH)
    _prefixed_digest(harness["sha256"], "environment.harness.sha256")
    binaries = _object(environment["binaries"], "environment.binaries")
    _exact_keys(binaries, ("kernel", "model"), "environment.binaries")
    for name, expected_path in (
        ("kernel", BINARY_LOGICAL_PATH),
        ("model", MODEL_BINARY_LOGICAL_PATH),
    ):
        item = _object(binaries[name], f"environment.binaries.{name}")
        _exact_keys(item, ("path", "sha256"), f"environment.binaries.{name}")
        _string(item["path"], f"environment.binaries.{name}.path", expected=expected_path)
        _prefixed_digest(item["sha256"], f"environment.binaries.{name}.sha256")
    return environment


def _case_contract() -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "rows": case.rows,
            "columns": case.columns,
            "calls_per_worker": case.calls_per_worker,
            "matrix_bytes": case.matrix_bytes,
            "role": case.role,
        }
        for case in CASES
    ]


def _cell_contract() -> list[dict[str, Any]]:
    return [
        {
            "cell_id": cell.cell_id,
            "case_id": cell.case_id,
            "baseline": BASELINE_IMPLEMENTATION,
            "candidate": cell.candidate,
            "allocation": cell.allocation,
            "workers": cell.workers,
            "role": cell.role,
        }
        for cell in CELLS
    ]


def build_experiment(
    *,
    captured_at: str,
    commit: str,
    kernel_binary_sha256: str,
    model_binary_sha256: str,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_SCHEMA,
        "harness_schema": HARNESS_SCHEMA,
        "captured_at_utc": captured_at,
        "git_commit": commit,
        "tracked_worktree_clean": True,
        "analysis_question": (
            "Within each preregistered cell, how does candidate batch elapsed time compare "
            "with the paired safe-Rust scalar BF16 baseline?"
        ),
        "baseline": BASELINE_IMPLEMENTATION,
        "candidate": AVX2_IMPLEMENTATION,
        "secondary_candidate": STAGED_IMPLEMENTATION,
        "build": {
            "argv": list(BUILD_COMMAND),
            "environment": dict(BUILD_RECORDED_ENVIRONMENT),
            "profile": "release",
            "features": "workspace-defaults",
            "timeout_seconds": BUILD_TIMEOUT_SECONDS,
            "native": {
                "cc_argv": list(NATIVE_CC_ARGV),
                "ar_argv": list(NATIVE_AR_ARGV),
            },
            "binaries": {
                "kernel": {
                    "package": "runnel-kernels",
                    "name": "runnel-kernel-bench",
                    "sha256": f"sha256:{kernel_binary_sha256}",
                },
                "model": {
                    "package": "runnel",
                    "name": "runnel-m4-model-check",
                    "sha256": f"sha256:{model_binary_sha256}",
                },
            },
        },
        "commands": {
            "cases": [BINARY_LOGICAL_PATH, "cases"],
            "kernel_correctness": [BINARY_LOGICAL_PATH, "correctness"],
            "model_correctness": [MODEL_BINARY_LOGICAL_PATH],
            "run_cell": [BINARY_LOGICAL_PATH, "run-cell"],
        },
        "fixture_contract": {
            "weight_domain": "runnel-m4-weight-v1\0",
            "input_domain": "runnel-m4-input-v1\0",
            "counter_encoding": "u64-little-endian",
            "case_id_length_encoding": "u16-little-endian",
            "digest_word_mapping": "q=i16(byte)-128",
            "weight_source": "q/128 converted f32-to-bf16-rne",
            "input_source": "q/64 f32 with element zero set to one",
        },
        "case_contract": _case_contract(),
        "cell_contract": _cell_contract(),
        "schedule_contract": {
            "cell_order_domain": "runnel-m4-cell-order-v1\0",
            "pair_order_domain": "runnel-m4-pair-order-v1\0",
            "cell_order": cell_order(),
            "warmup_orders": list(WARMUP_ORDERS),
            "warmup_pairs_per_cell": len(WARMUP_ORDERS),
            "measured_pairs_per_cell": MEASURED_PAIRS,
            "measured_rows": EXPECTED_OBSERVATIONS,
            "pair_orders": {
                cell.cell_id: measured_pair_orders(cell.cell_id) for cell in CELLS
            },
            "outliers_removed": False,
        },
        "correctness_contract": {
            "checks": [check[0] for check in expected_correctness_checks()],
            "required_before_timing": True,
            "f64_bound": "gamma(2*columns)*sum_abs+1e-7",
            "tiny_atol": 1e-5,
            "tiny_rtol": 1e-4,
        },
        "analysis_contract": {
            "paired_ratio": "candidate_elapsed_ns/baseline_elapsed_ns",
            "paired_difference": "candidate_elapsed_ns-baseline_elapsed_ns",
            "bootstrap_domain": "runnel-m4-bootstrap-v1\0",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "intervals_unadjusted": True,
            "cells_pooled": False,
            "omnibus_conclusion_permitted": False,
            "general_claim_cells": [
                "avx-natural-stream-expand",
                "avx-natural-stream-contract",
            ],
            "general_claim_observed_median_ratio_maximum": 0.95,
            "general_claim_interval_upper_below_one": True,
            "favorable_result_required": False,
        },
        "resource_contract": {
            "live_benchmark_bytes": MAX_LIVE_BENCHMARK_BYTES,
            "evidence_bytes": MAX_DIRECTORY_BYTES,
            "stdout_bytes_per_child": MAX_STDOUT_BYTES,
            "stderr_bytes_per_child": MAX_STDERR_BYTES,
            "cell_timeout_seconds": CELL_TIMEOUT_SECONDS,
            "full_capture_timeout_seconds": FULL_CAPTURE_TIMEOUT_SECONDS,
            "tmpfs_reserve_bytes": MIN_FREE_BYTES_AFTER_CAPTURE,
            "evidence_filesystem_reserve_bytes": MIN_EVIDENCE_FILESYSTEM_FREE_BYTES,
            "build_jobs": 2,
            "build_locked": True,
            "build_offline": True,
            "build_incremental": False,
        },
        "artifacts": dict(artifacts),
    }


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
            "candidate",
            "secondary_candidate",
            "build",
            "commands",
            "fixture_contract",
            "case_contract",
            "cell_contract",
            "schedule_contract",
            "correctness_contract",
            "analysis_contract",
            "resource_contract",
            "artifacts",
        ),
        "experiment",
    )
    _string(experiment["schema"], "experiment.schema", expected=EXPERIMENT_SCHEMA)
    _string(
        experiment["harness_schema"], "experiment.harness_schema", expected=HARNESS_SCHEMA
    )
    captured_at = _validate_timestamp(experiment["captured_at_utc"], "experiment.captured_at_utc")
    commit = _string(experiment["git_commit"], "experiment.git_commit", maximum=40)
    if not COMMIT_RE.fullmatch(commit):
        raise EvidenceError("experiment.git_commit: expected full lowercase commit ID")
    if not _boolean(experiment["tracked_worktree_clean"], "experiment.tracked_worktree_clean"):
        raise EvidenceError("experiment.tracked_worktree_clean: must be true")
    build = _object(experiment["build"], "experiment.build")
    binaries = _object(build.get("binaries"), "experiment.build.binaries")
    kernel = _object(binaries.get("kernel"), "experiment.build.binaries.kernel")
    model = _object(binaries.get("model"), "experiment.build.binaries.model")
    kernel_digest = _prefixed_digest(
        kernel.get("sha256"), "experiment.build.binaries.kernel.sha256"
    )
    model_digest = _prefixed_digest(
        model.get("sha256"), "experiment.build.binaries.model.sha256"
    )
    artifacts = _object(experiment["artifacts"], "experiment.artifacts")
    expected_artifacts = EXPECTED_FILES - {"experiment.json"}
    if set(artifacts) != expected_artifacts:
        raise EvidenceError("experiment.artifacts: expected the closed non-manifest file set")
    normalized_artifacts = {
        name: _prefixed_digest(artifacts[name], f"experiment.artifacts.{name}")
        for name in sorted(artifacts)
    }
    expected = build_experiment(
        captured_at=captured_at,
        commit=commit,
        kernel_binary_sha256=kernel_digest[7:],
        model_binary_sha256=model_digest[7:],
        artifacts=normalized_artifacts,
    )
    if experiment != expected:
        raise EvidenceError("experiment: value differs from the frozen M4 contract")
    return experiment


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
    input_bytes: bytes | None = None,
    affinity: Sequence[int] | None = None,
    address_space_bytes: int | None = None,
    timeout_seconds: float = CELL_TIMEOUT_SECONDS,
    maximum_stdout_bytes: int = MAX_STDOUT_BYTES,
    maximum_stderr_bytes: int = MAX_STDERR_BYTES,
    absolute_deadline: float | None = None,
) -> BoundedProcessResult:
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise EvidenceError("subprocess timeout must be finite and positive")
    if maximum_stdout_bytes < 1 or maximum_stderr_bytes < 1:
        raise EvidenceError("subprocess output limits must be positive")
    if absolute_deadline is not None:
        if not math.isfinite(absolute_deadline):
            raise EvidenceError("subprocess absolute deadline must be finite")
        timeout_seconds = min(
            timeout_seconds,
            _remaining_timeout(absolute_deadline, timeout_seconds, "subprocess launch"),
        )
    if address_space_bytes is not None and address_space_bytes < 1:
        raise EvidenceError("address-space limit must be positive")
    affinity_values = None if affinity is None else set(affinity)
    if affinity_values is not None and (not affinity_values or min(affinity_values) < 0):
        raise EvidenceError("subprocess affinity must contain nonnegative CPUs")

    def prepare_child() -> None:
        if affinity_values is not None:
            os.sched_setaffinity(0, affinity_values)
        if address_space_bytes is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (address_space_bytes, address_space_bytes),
            )

    try:
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            env=dict(CONTROLLED_ENVIRONMENT if environment is None else environment),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=prepare_child if affinity_values is not None or address_space_bytes else None,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return BoundedProcessResult(None, b"", b"", False, False, False, type(error).__name__)

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
    if input_bytes is not None:
        assert process.stdin is not None
        try:
            process.stdin.write(input_bytes)
            process.stdin.close()
        except BrokenPipeError:
            pass
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
        if absolute_deadline is None:
            return_code = process.wait()
        else:
            remaining = absolute_deadline - time.monotonic()
            if remaining <= 0:
                raise EvidenceError("subprocess did not terminate before the capture deadline")
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                _kill_process_group(process)
                raise EvidenceError(
                    "subprocess did not terminate before the capture deadline"
                ) from error
    for reader in readers:
        join_timeout = 5.0
        if absolute_deadline is not None:
            join_timeout = min(
                join_timeout,
                max(0.0, absolute_deadline - time.monotonic()),
            )
        reader.join(timeout=join_timeout)
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


def _decode_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _filesystem_type(path: Path) -> str:
    text = _read_small_text(Path("/proc/self/mountinfo"), 4 * 1024 * 1024)
    if text is None:
        raise EvidenceError("cannot inspect build-root filesystem type")
    best_length = -1
    best_type = ""
    for line in text.splitlines():
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
    available = _filesystem_available(resolved)
    required = MIN_FREE_BYTES_AFTER_CAPTURE + MAX_DIRECTORY_BYTES
    if available < required:
        raise EvidenceError("--build-root requires two GiB free after the evidence reserve")
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


def _remaining_timeout(deadline: float | None, maximum: float, context: str) -> float:
    if deadline is None:
        return maximum
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise EvidenceError(f"full capture deadline expired before {context}")
    return min(maximum, remaining)


def _tool_version(
    command: Sequence[str], environment: Mapping[str, str], deadline: float | None = None
) -> str:
    result = run_bounded_process(
        command,
        cwd=ROOT,
        environment=environment,
        timeout_seconds=_remaining_timeout(deadline, 30.0, "toolchain version probe"),
        maximum_stdout_bytes=4 * 1024,
        maximum_stderr_bytes=4 * 1024,
        absolute_deadline=deadline,
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


def _regular_executable(path: Path, context: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise EvidenceError(f"{context}: executable is absent") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise EvidenceError(f"{context}: expected regular non-symlink executable")
    if not os.access(path, os.X_OK):
        raise EvidenceError(f"{context}: build output is not executable")


def _native_tool_identity(
    name: str,
    environment: Mapping[str, str],
    deadline: float | None = None,
) -> tuple[str, str, str]:
    resolved_name = shutil.which(name, path=environment.get("PATH"))
    if resolved_name is None:
        raise EvidenceError(f"cannot resolve native tool {name}")
    try:
        resolved = Path(resolved_name).resolve(strict=True)
    except OSError as error:
        raise EvidenceError(f"cannot resolve native tool {name}") from error
    _regular_executable(resolved, f"native tool {name}")
    result = run_bounded_process(
        (name, "--version"),
        cwd=ROOT,
        environment=environment,
        timeout_seconds=_remaining_timeout(deadline, 30.0, f"native tool {name} probe"),
        maximum_stdout_bytes=16 * 1024,
        maximum_stderr_bytes=4 * 1024,
        absolute_deadline=deadline,
    )
    if (
        result.launch_error is not None
        or result.timed_out
        or result.stdout_exceeded
        or result.stderr_exceeded
        or result.return_code != 0
        or result.stderr
    ):
        raise EvidenceError(f"cannot record native tool {name}")
    try:
        lines = result.stdout.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise EvidenceError(f"native tool {name} returned non-ASCII output") from error
    if not lines or not lines[0] or len(lines[0]) > 256:
        raise EvidenceError(f"native tool {name} returned an invalid version")
    return str(resolved), sha256_file(resolved), lines[0]


def _native_argv_from_build_script(
    source: str,
    *,
    program: str,
    description: str,
) -> tuple[str, ...]:
    anchor = f'Command::new("{program}")'
    if source.count(anchor) != 1:
        raise EvidenceError(f"build.rs must contain exactly one {program} command")
    start = source.index(anchor) + len(anchor)
    end = source.find(f'"{description}"', start)
    if end < 0:
        raise EvidenceError(f"build.rs lacks the {description} command boundary")
    section = source[start:end]
    arguments = re.findall(r"\.arg\(([^()\n]+)\)", section)
    if ".args(" in section or section.count(".arg(") != len(arguments):
        raise EvidenceError(f"build.rs {program} command uses an unsupported argument form")
    replacements = {
        "SOURCE": "native/bf16_gemv.c",
        "&object": "{private-cargo-out}/bf16_gemv.o",
        "&archive": "{private-cargo-out}/librunnel_kernels.a",
    }
    result = [program]
    for expression in arguments:
        expression = expression.strip()
        if expression in replacements:
            result.append(replacements[expression])
            continue
        try:
            value = json.loads(expression)
        except (TypeError, ValueError) as error:
            raise EvidenceError(
                f"build.rs {program} command has an unknown argument expression"
            ) from error
        if type(value) is not str or not value:
            raise EvidenceError(f"build.rs {program} command has an invalid argument")
        result.append(value)
    return tuple(result)


def _verify_native_build_argv_contract(source: str | None = None) -> None:
    if source is None:
        source = _read_small_text(ROOT / "crates" / "runnel-kernels" / "build.rs")
    if source is None:
        raise EvidenceError("cannot read the native build contract")
    required_bindings = (
        'const SOURCE: &str = "native/bf16_gemv.c";',
        'let object = out_dir.join("bf16_gemv.o");',
        'let archive = out_dir.join("librunnel_kernels.a");',
    )
    if any(source.count(binding) != 1 for binding in required_bindings):
        raise EvidenceError("native build input/output bindings differ from the recorded argv")
    actual_cc = _native_argv_from_build_script(
        source,
        program="cc",
        description="C compiler",
    )
    actual_ar = _native_argv_from_build_script(
        source,
        program="ar",
        description="archiver",
    )
    if actual_cc != NATIVE_CC_ARGV or actual_ar != NATIVE_AR_ARGV:
        raise EvidenceError("recorded native argv differs from crates/runnel-kernels/build.rs")


def _build_release_binaries(
    build_root_value: str, deadline: float | None = None
) -> PrivateBuild:
    build_root, available = _resolve_build_root(build_root_value)
    private_root = Path(tempfile.mkdtemp(prefix=".runnel-m4-build-", dir=build_root))
    os.chmod(private_root, 0o700)
    target = private_root / "target"
    environment = _build_environment(target)
    try:
        _verify_native_build_argv_contract()
        cargo_version = _tool_version(("cargo", "--version"), environment, deadline)
        rustc_version = _tool_version(("rustc", "--version"), environment, deadline)
        cc_path, cc_sha256, cc_version = _native_tool_identity(
            "cc", environment, deadline
        )
        ar_path, ar_sha256, ar_version = _native_tool_identity(
            "ar", environment, deadline
        )
        result = run_bounded_process(
            BUILD_COMMAND,
            cwd=ROOT,
            environment=environment,
            timeout_seconds=_remaining_timeout(deadline, BUILD_TIMEOUT_SECONDS, "release build"),
            maximum_stdout_bytes=MAX_STDOUT_BYTES,
            maximum_stderr_bytes=MAX_STDERR_BYTES,
            absolute_deadline=deadline,
        )
        if result.launch_error is not None or result.timed_out:
            raise EvidenceError("locked offline release build could not complete")
        if result.stdout_exceeded or result.stderr_exceeded:
            raise EvidenceError("locked offline release build output exceeded one MiB")
        if result.return_code != 0:
            raise EvidenceError("locked offline release build failed")
        for name, expected_path, expected_sha256 in (
            ("cc", cc_path, cc_sha256),
            ("ar", ar_path, ar_sha256),
        ):
            resolved_name = shutil.which(name, path=environment.get("PATH"))
            if resolved_name is None:
                raise EvidenceError(f"native tool {name} disappeared during build")
            try:
                resolved = Path(resolved_name).resolve(strict=True)
            except OSError as error:
                raise EvidenceError(f"native tool {name} disappeared during build") from error
            if str(resolved) != expected_path or sha256_file(resolved) != expected_sha256:
                raise EvidenceError(f"native tool {name} changed during build")
        kernel_binary = target / "release" / "runnel-kernel-bench"
        model_binary = target / "release" / "runnel-m4-model-check"
        _regular_executable(kernel_binary, "kernel benchmark")
        _regular_executable(model_binary, "model correctness")
        return PrivateBuild(
            root=private_root,
            kernel_binary=kernel_binary,
            kernel_binary_sha256=sha256_file(kernel_binary),
            model_binary=model_binary,
            model_binary_sha256=sha256_file(model_binary),
            cargo_version=cargo_version,
            rustc_version=rustc_version,
            cc_path=cc_path,
            cc_sha256=cc_sha256,
            cc_version=cc_version,
            ar_path=ar_path,
            ar_sha256=ar_sha256,
            ar_version=ar_version,
            build_root_available_bytes=available,
        )
    except BaseException:
        shutil.rmtree(private_root, ignore_errors=True)
        raise


def _git_output(arguments: Sequence[str], deadline: float | None = None) -> str:
    environment = {
        name: value for name, value in os.environ.items() if name in {"PATH", "HOME"}
    }
    environment.update(CONTROLLED_ENVIRONMENT)
    result = run_bounded_process(
        ("git", *arguments),
        cwd=ROOT,
        environment=environment,
        timeout_seconds=_remaining_timeout(deadline, 30.0, "git metadata check"),
        maximum_stdout_bytes=MAX_GIT_OUTPUT_BYTES,
        maximum_stderr_bytes=MAX_GIT_OUTPUT_BYTES,
        absolute_deadline=deadline,
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


def _git_blob(
    commit: str, repository_path: str, deadline: float | None = None
) -> bytes:
    if not COMMIT_RE.fullmatch(commit):
        raise EvidenceError("evidence commit must be full lowercase hexadecimal")
    result = run_bounded_process(
        ("git", "cat-file", "blob", f"{commit}:{repository_path}"),
        cwd=ROOT,
        environment={**CONTROLLED_ENVIRONMENT, "PATH": os.environ.get("PATH", "")},
        timeout_seconds=_remaining_timeout(deadline, 30.0, "historical blob check"),
        maximum_stdout_bytes=2 * 1024 * 1024,
        maximum_stderr_bytes=MAX_GIT_OUTPUT_BYTES,
        absolute_deadline=deadline,
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


def _commit_harness_sha256(commit: str, deadline: float | None = None) -> str:
    return sha256_bytes(_git_blob(commit, HARNESS_PATH, deadline))


def _verify_commit_harness_sha256(
    commit: str, recorded: str, deadline: float | None = None
) -> None:
    expected = _prefixed_digest(recorded, "environment.harness.sha256")
    actual = f"sha256:{_commit_harness_sha256(commit, deadline)}"
    if expected != actual:
        raise EvidenceError("captured harness hash differs from named commit blob")


def _require_clean_commit(commit: str, deadline: float | None = None) -> str:
    if not COMMIT_RE.fullmatch(commit):
        raise EvidenceError("--commit must be a full lowercase 40-character commit ID")
    head = _git_output(("rev-parse", "--verify", "HEAD"), deadline)
    if head != commit:
        raise EvidenceError(f"--commit {commit} does not equal HEAD {head}")
    if _git_output(("status", "--porcelain=v1", "--untracked-files=all"), deadline):
        raise EvidenceError("capture requires a clean worktree and index")
    _git_output(("ls-files", "--error-unmatch", HARNESS_PATH), deadline)
    blob_sha256 = _commit_harness_sha256(commit, deadline)
    current_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    if current_sha256 != blob_sha256:
        raise EvidenceError("working harness differs from the named commit blob")
    return blob_sha256


def _bounded_directory_entries(
    descriptor: int,
    expected: set[str],
    context: str,
) -> None:
    actual: set[str] = set()
    with os.scandir(descriptor) as entries:
        for entry in entries:
            if entry.name not in expected:
                raise EvidenceError(f"{context}: unknown entry {entry.name!r}")
            actual.add(entry.name)
    if actual != expected:
        raise EvidenceError(f"{context}: missing entries {sorted(expected - actual)}")


def _read_evidence_files(
    path: Path, deadline: float | None = None
) -> dict[str, bytes]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = -1
    figures_fd = -1
    opened: list[tuple[str, int, os.stat_result]] = []
    try:
        root_fd = os.open(path, directory_flags)
        expected_root_files = {name for name in EXPECTED_FILES if "/" not in name}
        _bounded_directory_entries(root_fd, expected_root_files | {"figures"}, "evidence root")
        figures_fd = os.open("figures", directory_flags, dir_fd=root_fd)
        expected_figures = {
            name.removeprefix("figures/")
            for name in EXPECTED_FILES
            if name.startswith("figures/")
        }
        _bounded_directory_entries(figures_fd, expected_figures, "evidence figures")
        targets = [(name, root_fd, name) for name in sorted(expected_root_files)] + [
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
            _remaining_timeout(deadline, 1.0, f"reading evidence file {logical_name}")
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
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_before != identity_after:
                raise EvidenceError(f"evidence entry {logical_name!r} changed while reading")
            files[logical_name] = b"".join(chunks)
        _remaining_timeout(deadline, 1.0, "reading evidence directory")
        return files
    except OSError as error:
        raise EvidenceError(
            f"cannot safely read evidence directory: {type(error).__name__}"
        ) from error
    finally:
        for _, descriptor, _ in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass
        for descriptor in (figures_fd, root_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def verify_directory(path: Path, deadline: float | None = None) -> dict[str, Any]:
    """Verify a complete M4 artifact without writing any file."""

    files = _read_evidence_files(path, deadline)
    environment = _validate_environment(
        parse_json_bytes(files["environment.json"], "environment.json")
    )
    experiment = _validate_experiment(
        parse_json_bytes(files["experiment.json"], "experiment.json")
    )
    if files["environment.json"] != _json_file_bytes(environment):
        raise EvidenceError("environment.json is not canonical JSON")
    if files["experiment.json"] != _json_file_bytes(experiment):
        raise EvidenceError("experiment.json is not canonical JSON")
    if environment["captured_at_utc"] != experiment["captured_at_utc"]:
        raise EvidenceError("environment and experiment timestamps differ")
    if environment["binaries"]["kernel"]["sha256"] != experiment["build"]["binaries"][
        "kernel"
    ]["sha256"]:
        raise EvidenceError("kernel binary hashes differ between metadata files")
    if environment["binaries"]["model"]["sha256"] != experiment["build"]["binaries"][
        "model"
    ]["sha256"]:
        raise EvidenceError("model binary hashes differ between metadata files")
    _verify_commit_harness_sha256(
        experiment["git_commit"], environment["harness"]["sha256"], deadline
    )
    for name, expected_digest in experiment["artifacts"].items():
        actual = _artifact_digest(files[name])
        if actual != expected_digest:
            raise EvidenceError(f"{name}: digest differs from experiment manifest")

    case_values = parse_jsonl_bytes(files["cases.jsonl"], "cases.jsonl")
    cases = validate_cases(case_values)
    if files["cases.jsonl"] != _jsonl_bytes(cases):
        raise EvidenceError("cases.jsonl is not canonical JSONL")
    correctness_values = parse_jsonl_bytes(files["correctness.jsonl"], "correctness.jsonl")
    model_contract_cohort = _model_contract_cohort_for_commit(experiment["git_commit"])
    correctness = validate_correctness(
        correctness_values,
        model_contract_cohort=model_contract_cohort,
    )
    if files["correctness.jsonl"] != _jsonl_bytes(correctness):
        raise EvidenceError("correctness.jsonl is not canonical JSONL")
    observation_values = parse_jsonl_bytes(files["observations.jsonl"], "observations.jsonl")
    recorded_sets = environment["cpu"]["benchmark_sets"]
    observations = validate_dataset(
        observation_values,
        benchmark_cpus=(
            recorded_sets["one_worker"],
            recorded_sets["two_worker"],
        ),
    )
    if files["observations.jsonl"] != _jsonl_bytes(observations):
        raise EvidenceError("observations.jsonl is not canonical JSONL")

    expected_summary = build_summary(
        observations,
        correctness,
        cases_digest=_artifact_digest(files["cases.jsonl"]),
        correctness_digest=_artifact_digest(files["correctness.jsonl"]),
        observations_digest=_artifact_digest(files["observations.jsonl"]),
        model_contract_cohort=model_contract_cohort,
    )
    if files["summary.json"] != _json_file_bytes(expected_summary):
        raise EvidenceError("summary.json differs from raw-ledger regeneration")
    figures = render_figures(expected_summary)
    for name, expected in figures.items():
        if files[name] != expected:
            raise EvidenceError(f"{name} differs from summary regeneration")
    _remaining_timeout(deadline, 1.0, "evidence self-verification")
    return {
        "status": "verified",
        "git_commit": experiment["git_commit"],
        "correctness_passed": expected_summary["correctness_passed"],
        "complete_cells": sum(cell["complete"] for cell in expected_summary["cells"]),
        "general_claim_satisfied": expected_summary["general_claim_rule"]["satisfied"],
        "observation_rows": len(observations),
    }


def _require_successful_output(result: BoundedProcessResult, context: str) -> bytes:
    if result.launch_error is not None:
        raise EvidenceError(f"{context}: launch failed with {result.launch_error}")
    if result.timed_out:
        raise EvidenceError(f"{context}: timed out")
    if result.stdout_exceeded or result.stderr_exceeded:
        raise EvidenceError(f"{context}: bounded output was exceeded")
    if result.return_code != 0:
        raise EvidenceError(f"{context}: process exited with {result.return_code}")
    if result.stderr:
        raise EvidenceError(f"{context}: successful command wrote stderr")
    return result.stdout


def _run_jsonl_command(
    command: Sequence[str],
    *,
    expected_rows: int,
    context: str,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    result = run_bounded_process(
        command,
        cwd=ROOT,
        timeout_seconds=_remaining_timeout(deadline, CELL_TIMEOUT_SECONDS, context),
        address_space_bytes=MAX_LIVE_BENCHMARK_BYTES,
        absolute_deadline=deadline,
    )
    data = _require_successful_output(result, context)
    rows = parse_jsonl_bytes(data, context)
    if len(rows) != expected_rows:
        raise EvidenceError(f"{context}: expected {expected_rows} rows, got {len(rows)}")
    return rows


def _select_benchmark_cpus() -> tuple[list[int], list[int] | None]:
    if not hasattr(os, "sched_getaffinity"):
        raise EvidenceError("benchmark capture requires Linux CPU-affinity support")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise EvidenceError("benchmark process has an empty CPU affinity mask")
    topology = _cpu_topology(allowed)
    return _derive_benchmark_cpus(allowed, topology)


def _child_failure_status(result: BoundedProcessResult) -> tuple[str, str] | None:
    if result.launch_error is not None:
        return "launch_error", f"child launch failed: {result.launch_error}"
    if result.timed_out:
        return "timeout", "cell child exceeded its bounded timeout"
    if result.stdout_exceeded or result.stderr_exceeded:
        return "output_limit", "cell child exceeded a one-MiB output limit"
    if result.return_code != 0:
        return "nonzero_exit", f"cell child exited with status {result.return_code}"
    if result.stderr:
        return "protocol_error", "successful cell child wrote stderr"
    return None


def _run_cell(
    binary: Path,
    cell: CellSpec,
    child_sequence: int,
    cpus: Sequence[int],
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    request = build_cell_request(cell, child_sequence, cpus)
    result = run_bounded_process(
        (str(binary), "run-cell"),
        cwd=ROOT,
        input_bytes=_json_file_bytes(request),
        affinity=cpus,
        address_space_bytes=MAX_LIVE_BENCHMARK_BYTES,
        timeout_seconds=_remaining_timeout(deadline, CELL_TIMEOUT_SECONDS, cell.cell_id),
        absolute_deadline=deadline,
    )
    parsed, parse_failure = parse_jsonl_prefix_bytes(
        result.stdout, f"cell stream {cell.cell_id}"
    )
    accepted: list[dict[str, Any]] = []
    validation_failure: str | None = None
    for value in parsed:
        try:
            validate_cell_stream_prefix(
                [*accepted, value],
                cell=cell,
                child_sequence=child_sequence,
                cpus=cpus,
            )
        except EvidenceError as error:
            validation_failure = f"cell stream rejected: {error}"
            break
        accepted.append(value)
    observations, warmups_ok = validate_cell_stream_prefix(
        accepted,
        cell=cell,
        child_sequence=child_sequence,
        cpus=cpus,
    )

    process_failure = _child_failure_status(result)
    status = process_failure[0] if process_failure is not None else None
    failures = [process_failure[1]] if process_failure is not None else []
    if parse_failure is not None:
        status = status or "protocol_error"
        failures.append(parse_failure)
    if validation_failure is not None:
        status = status or "protocol_error"
        failures.append(validation_failure)
    expected_records = len(WARMUP_ORDERS) + 2 * MEASURED_PAIRS
    if len(accepted) != expected_records:
        status = status or "protocol_error"
        failures.append(
            f"cell stream retained {len(accepted)} of {expected_records} ordered records"
        )
    if len(accepted) >= len(WARMUP_ORDERS) and not warmups_ok:
        status = status or "warmup_failure"
        failures.append("one or more preregistered warmup variants failed")

    if status is None:
        return observations

    failure = "; ".join(failures)[:512]
    if len(observations) < 2 * MEASURED_PAIRS:
        return [
            *observations,
            *synthesize_failure_rows(
                cell,
                child_sequence,
                status,
                failure,
                start_observation=len(observations),
            ),
        ]

    # A process/warmup failure after all rows were durably flushed still makes
    # the cell ineligible. Attach it to the last row without discarding any
    # retained timing, resource, affinity, output, or sink evidence.
    last = dict(observations[-1])
    if last["status"] == "ok":
        last["status"] = status
        last["failure"] = failure
    else:
        last["failure"] = f"{last['failure']}; {failure}"[:512]
    observations[-1] = last
    return observations


def _resolve_new_output(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if os.path.lexists(candidate):
        raise EvidenceError("capture output must initially be absent")
    try:
        parent = candidate.parent.resolve(strict=True)
        repository = ROOT.resolve(strict=True)
    except OSError as error:
        raise EvidenceError("capture output parent is unavailable") from error
    try:
        parent.relative_to(repository)
    except ValueError as error:
        raise EvidenceError("capture output must remain inside the repository") from error
    output_base = (ROOT / "benchmarks" / "raw").resolve(strict=True)
    if parent != output_base:
        raise EvidenceError("capture output must be one new directory under benchmarks/raw")
    if not PORTABLE_ID_RE.fullmatch(candidate.name) or ".." in candidate.name:
        raise EvidenceError("capture output requires a portable 1-96 character experiment ID")
    return parent / candidate.name


def _require_capture_reserves(build_root: Path) -> None:
    tmpfs_required = MIN_FREE_BYTES_AFTER_CAPTURE + MAX_DIRECTORY_BYTES
    if _filesystem_available(build_root) < tmpfs_required:
        raise EvidenceError("capture requires two GiB of tmpfs reserve plus evidence capacity")
    if _filesystem_available(ROOT) < MIN_EVIDENCE_FILESYSTEM_FREE_BYTES:
        raise EvidenceError("evidence filesystem lacks the frozen 32-MiB reserve")


def _write_artifacts_exclusive(root: Path, artifacts: Mapping[str, bytes]) -> None:
    figures = root / "figures"
    figures.mkdir(mode=0o700)
    for name, data in artifacts.items():
        destination = root / name
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        with destination.open("xb") as output:
            output.write(data)


def capture(build_root_value: str, output_value: str, commit: str) -> dict[str, Any]:
    """Build once from clean HEAD, execute the frozen matrix, and publish atomically."""

    started = time.monotonic()
    deadline = started + FULL_CAPTURE_TIMEOUT_SECONDS
    harness_sha256 = _require_clean_commit(commit, deadline)
    output = _resolve_new_output(output_value)
    build_root, _ = _resolve_build_root(build_root_value)
    _require_capture_reserves(build_root)
    process_before = _rusage_snapshot()
    private = _build_release_binaries(str(build_root), deadline)
    try:
        if _require_clean_commit(commit, deadline) != harness_sha256:
            raise EvidenceError("harness commit blob changed during build")
        kernel_hash = private.kernel_binary_sha256
        model_hash = private.model_binary_sha256

        case_values = _run_jsonl_command(
            (str(private.kernel_binary), "cases"),
            expected_rows=len(CASES),
            context="kernel cases",
            deadline=deadline,
        )
        cases = validate_cases(case_values)
        kernel_correctness = _run_jsonl_command(
            (str(private.kernel_binary), "correctness"),
            expected_rows=sum(
                1 for _, kind, _, _ in expected_correctness_checks() if kind == "kernel"
            ),
            context="kernel correctness",
            deadline=deadline,
        )
        model_correctness = _run_jsonl_command(
            (str(private.model_binary),),
            expected_rows=len(MODEL_CHECKS),
            context="model correctness",
            deadline=deadline,
        )
        correctness = validate_correctness([*kernel_correctness, *model_correctness])

        one_cpu, two_cpus = _select_benchmark_cpus()
        observations: list[dict[str, Any]] = []
        all_correct = correctness_passed(correctness)
        for child_sequence, cell_id in enumerate(cell_order()):
            if time.monotonic() >= deadline:
                raise EvidenceError("full capture exceeded the fifteen-minute limit")
            cell = CELL_BY_ID[cell_id]
            if not all_correct:
                observations.extend(
                    synthesize_failure_rows(
                        cell,
                        child_sequence,
                        "correctness_failure",
                        "one or more preregistered correctness gates failed",
                    )
                )
            elif cell.workers == 2 and two_cpus is None:
                observations.extend(
                    synthesize_failure_rows(
                        cell,
                        child_sequence,
                        "unsupported",
                        "two distinct physical cores are unavailable",
                    )
                )
            else:
                cpus = one_cpu if cell.workers == 1 else two_cpus
                assert cpus is not None
                observations.extend(
                    _run_cell(
                        private.kernel_binary,
                        cell,
                        child_sequence,
                        cpus,
                        deadline,
                    )
                )
        observations = validate_dataset(
            observations,
            benchmark_cpus=(one_cpu, two_cpus),
        )

        if sha256_file(private.kernel_binary) != kernel_hash:
            raise EvidenceError("kernel benchmark executable changed during capture")
        if sha256_file(private.model_binary) != model_hash:
            raise EvidenceError("model correctness executable changed during capture")
        if _require_clean_commit(commit, deadline) != harness_sha256:
            raise EvidenceError("repository changed during capture")
        if time.monotonic() >= deadline:
            raise EvidenceError("full capture exceeded the fifteen-minute limit")

        captured_at = _utc_now()
        cases_bytes = _jsonl_bytes(cases)
        correctness_bytes = _jsonl_bytes(correctness)
        observations_bytes = _jsonl_bytes(observations)
        summary = build_summary(
            observations,
            correctness,
            cases_digest=_artifact_digest(cases_bytes),
            correctness_digest=_artifact_digest(correctness_bytes),
            observations_digest=_artifact_digest(observations_bytes),
        )
        process_after = _rusage_snapshot()
        environment = build_environment(
            captured_at=captured_at,
            harness_sha256=harness_sha256,
            kernel_binary_sha256=kernel_hash,
            model_binary_sha256=model_hash,
            cargo_version=private.cargo_version,
            rustc_version=private.rustc_version,
            cc_path=private.cc_path,
            cc_sha256=private.cc_sha256,
            cc_version=private.cc_version,
            ar_path=private.ar_path,
            ar_sha256=private.ar_sha256,
            ar_version=private.ar_version,
            benchmark_cpus=(one_cpu, two_cpus),
            build_root_available_bytes=private.build_root_available_bytes,
            build_root_remaining_bytes=_filesystem_available(build_root),
            process_before=process_before,
            process_after=process_after,
        )
        artifacts: dict[str, bytes] = {
            "environment.json": _json_file_bytes(environment),
            "cases.jsonl": cases_bytes,
            "correctness.jsonl": correctness_bytes,
            "observations.jsonl": observations_bytes,
            "summary.json": _json_file_bytes(summary),
            **render_figures(summary),
        }
        artifact_digests = {name: _artifact_digest(data) for name, data in artifacts.items()}
        experiment = build_experiment(
            captured_at=captured_at,
            commit=commit,
            kernel_binary_sha256=kernel_hash,
            model_binary_sha256=model_hash,
            artifacts=artifact_digests,
        )
        artifacts["experiment.json"] = _json_file_bytes(experiment)
        if set(artifacts) != EXPECTED_FILES:
            raise EvidenceError("internal artifact set differs from the closed contract")
        if sum(len(data) for data in artifacts.values()) > MAX_DIRECTORY_BYTES:
            raise EvidenceError("generated evidence exceeds the sixteen-MiB cap")
        _require_capture_reserves(build_root)
        _remaining_timeout(deadline, 1.0, "evidence publication")

        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        os.chmod(staging, 0o700)
        try:
            _write_artifacts_exclusive(staging, artifacts)
            status = verify_directory(staging, deadline)
            _remaining_timeout(deadline, 1.0, "evidence publication")
            if os.path.lexists(output):
                raise EvidenceError("capture output appeared while staging evidence")
            os.rename(staging, output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return {"status": "captured", "output": str(output.relative_to(ROOT)), **status}
    finally:
        shutil.rmtree(private.root, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_parser = subparsers.add_parser("capture", help="capture a new append-only M4 matrix")
    capture_parser.add_argument(
        "--build-root",
        required=True,
        help="absolute tmpfs directory for the private release build",
    )
    capture_parser.add_argument(
        "--output",
        required=True,
        help="new repository-relative directory directly under benchmarks/raw",
    )
    capture_parser.add_argument("--commit", required=True, help="full clean HEAD commit ID")
    verify_parser = subparsers.add_parser("verify", help="verify an existing complete matrix")
    verify_parser.add_argument("--input", required=True, help="evidence directory")
    verify_parser.add_argument(
        "--check",
        action="store_true",
        required=True,
        help="regenerate in memory and byte-compare without writes",
    )
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    try:
        if arguments.action == "capture":
            result = capture(arguments.build_root, arguments.output, arguments.commit)
        else:
            input_path = Path(arguments.input)
            if not input_path.is_absolute():
                input_path = ROOT / input_path
            result = verify_directory(input_path)
    except EvidenceError as error:
        print(f"M4 evidence error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
