#!/usr/bin/env python3
"""Independent stdlib oracle for RunnelMoE seeded sampling.

This module is deliberately separate from the Rust runtime.  Inputs which cross
the language boundary carry binary32 values as hexadecimal bit patterns; all
binary64 diagnostics use ``float.hex()`` strings.  The implementation follows
the arithmetic and visit order frozen by ADR 0007 and uses Python's IEEE-754
binary64 operations and :func:`math.exp`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Iterable, NoReturn, Sequence


SCHEMA = "runnel.sampling-vectors/1"
SPECIFICATION = "adr-0007-seeded-sampling-v1"
FLOAT_ENCODING = "f32-bits-hex-input_f64-float-hex-diagnostics"
MASK64 = (1 << 64) - 1
MAX_VECTOR_BYTES = 1024 * 1024
DEFAULT_VECTOR_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "scheduler" / "sampling-v1.json"
)


class SamplingError(ValueError):
    """A stable invalid-input or internal sampling failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> NoReturn:
    raise SamplingError(code, message)


def _require_plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("invalid_request", f"{label} must be an integer")
    return value


def _u64(value: Any, label: str) -> int:
    value = _require_plain_int(value, label)
    if value < 0 or value > MASK64:
        _fail("invalid_request", f"{label} is outside the unsigned 64-bit range")
    return value


def _u32(value: Any, label: str) -> int:
    value = _require_plain_int(value, label)
    if value < 0 or value > 0xFFFF_FFFF:
        _fail("invalid_request", f"{label} is outside the unsigned 32-bit range")
    return value


def _u64_hex(value: int) -> str:
    return f"0x{value:016x}"


def _u32_hex(value: int) -> str:
    return f"0x{value:08x}"


def _parse_fixed_hex(value: Any, digits: int, label: str) -> int:
    if not isinstance(value, str):
        _fail("invalid_vector", f"{label} must be a hexadecimal string")
    if len(value) != digits + 2 or not value.startswith("0x"):
        _fail("invalid_vector", f"{label} must use 0x followed by {digits} lowercase digits")
    tail = value[2:]
    if any(character not in "0123456789abcdef" for character in tail):
        _fail("invalid_vector", f"{label} is not canonical lowercase hexadecimal")
    return int(tail, 16)


def parse_u64_hex(value: Any, label: str = "u64") -> int:
    return _parse_fixed_hex(value, 16, label)


def parse_f32_bits(value: Any, label: str = "f32 bits") -> int:
    return _parse_fixed_hex(value, 8, label)


def f32_from_bits(bits: int) -> float:
    """Widen the binary32 encoded by *bits* exactly to Python binary64."""

    bits = _u32(bits, "binary32 bits")
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _parse_f64_hex(value: Any, label: str) -> float:
    if not isinstance(value, str):
        _fail("invalid_vector", f"{label} must be a float.hex() string")
    try:
        parsed = float.fromhex(value)
    except ValueError:
        _fail("invalid_vector", f"{label} is not a float.hex() string")
    if parsed.hex() != value:
        _fail("invalid_vector", f"{label} is not in canonical float.hex() form")
    return parsed


def _finite_hex(value: float, label: str) -> str:
    if not math.isfinite(value):
        _fail("internal", f"{label} is nonfinite")
    return value.hex()


@dataclass(frozen=True)
class RngPreview:
    next_state: int
    word: int
    unit: float


def splitmix64_preview(state: int) -> RngPreview:
    """Preview one SplitMix64-v1 transition without owning mutable state."""

    state = _u64(state, "RNG state")
    next_state = (state + 0x9E37_79B9_7F4A_7C15) & MASK64
    word = next_state
    word = ((word ^ (word >> 30)) * 0xBF58_476D_1CE4_E5B9) & MASK64
    word = ((word ^ (word >> 27)) * 0x94D0_49BB_1331_11EB) & MASK64
    word ^= word >> 31
    word &= MASK64
    unit = (word >> 11) * float.fromhex("0x1.0000000000000p-53")
    if not 0.0 <= unit < 1.0:
        _fail("internal", "SplitMix64 conversion escaped [0, 1)")
    return RngPreview(next_state=next_state, word=word, unit=unit)


@dataclass(frozen=True)
class CategoricalResult:
    token_id: int
    cumulative: tuple[float, ...]
    used_fallback: bool


def categorical_select(
    candidates: Sequence[tuple[int, float]], unit: float
) -> CategoricalResult:
    """Select in the supplied stable order using normalized binary64 weights."""

    if not candidates:
        _fail("invalid_request", "categorical candidates must not be empty")
    if not math.isfinite(unit) or unit < 0.0 or unit >= 1.0:
        _fail("invalid_request", "categorical unit value must be finite in [0, 1)")

    seen: set[int] = set()
    retained_sum = 0.0
    normalized: list[tuple[int, float]] = []
    for index, (token_id, weight) in enumerate(candidates):
        token_id = _require_plain_int(token_id, f"candidate {index} token ID")
        if token_id < 0:
            _fail("invalid_request", f"candidate {index} token ID must be nonnegative")
        if token_id in seen:
            _fail("invalid_request", f"candidate token ID {token_id} is duplicated")
        seen.add(token_id)
        if not isinstance(weight, float) or not math.isfinite(weight) or weight < 0.0:
            _fail("invalid_request", f"candidate {index} weight must be finite and nonnegative")
        retained_sum += weight
        if not math.isfinite(retained_sum):
            _fail("internal", "categorical retained sum is nonfinite")
        normalized.append((token_id, weight))
    if retained_sum <= 0.0:
        _fail("invalid_request", "categorical retained sum must be positive")

    cumulative = 0.0
    diagnostics: list[float] = []
    winner: int | None = None
    for token_id, weight in normalized:
        probability = weight / retained_sum
        if not math.isfinite(probability):
            _fail("internal", "categorical probability is nonfinite")
        cumulative += probability
        if not math.isfinite(cumulative):
            _fail("internal", "categorical cumulative probability is nonfinite")
        diagnostics.append(cumulative)
        if winner is None and cumulative > unit:
            winner = token_id

    used_fallback = winner is None
    if winner is None:
        winner = normalized[-1][0]
    return CategoricalResult(winner, tuple(diagnostics), used_fallback)


def sample_preview(
    *,
    state: int,
    logit_bits: Sequence[int],
    temperature_bits: int,
    top_k: int,
    top_p_bits: int,
) -> dict[str, Any]:
    """Compute one sampled-token preview using ADR 0007's exact visit order."""

    state = _u64(state, "RNG state")
    if not isinstance(logit_bits, Sequence) or isinstance(logit_bits, (str, bytes)):
        _fail("invalid_request", "logits must be a sequence of binary32 bit patterns")
    if not logit_bits:
        _fail("invalid_request", "logits must not be empty")

    logits: list[float] = []
    for token_id, bits in enumerate(logit_bits):
        value = f32_from_bits(_u32(bits, f"logit {token_id} bits"))
        if not math.isfinite(value):
            _fail("invalid_request", f"logit {token_id} must be finite")
        logits.append(value)

    temperature = f32_from_bits(_u32(temperature_bits, "temperature bits"))
    if not math.isfinite(temperature) or temperature <= 0.0:
        _fail("invalid_request", "temperature must be finite and greater than zero")
    top_k = _require_plain_int(top_k, "top_k")
    if top_k < 1 or top_k > len(logits):
        _fail("invalid_request", "top_k must be between one and the vocabulary size")
    top_p = f32_from_bits(_u32(top_p_bits, "top_p bits"))
    if not math.isfinite(top_p) or top_p <= 0.0 or top_p > 1.0:
        _fail("invalid_request", "top_p must be finite in (0, 1]")

    scaled: list[tuple[int, float]] = []
    for token_id, logit in enumerate(logits):
        value = logit / temperature
        if not math.isfinite(value):
            _fail("internal", f"scaled logit {token_id} is nonfinite")
        scaled.append((token_id, value))
    scaled.sort(key=lambda item: (-item[1], item[0]))
    ordered_token_ids = [token_id for token_id, _ in scaled]
    selected = scaled[:top_k]

    maximum = selected[0][1]
    weighted: list[tuple[int, float]] = []
    total_weight = 0.0
    for token_id, scaled_logit in selected:
        exponent = scaled_logit - maximum
        if not math.isfinite(exponent):
            _fail("internal", f"softmax exponent input for token {token_id} is nonfinite")
        weight = math.exp(exponent)
        if not math.isfinite(weight):
            _fail("internal", f"softmax weight for token {token_id} is nonfinite")
        total_weight += weight
        if not math.isfinite(total_weight):
            _fail("internal", "softmax total weight is nonfinite")
        weighted.append((token_id, weight))
    if total_weight <= 0.0:
        _fail("internal", "softmax total weight is not positive")

    threshold = top_p * total_weight
    if not math.isfinite(threshold):
        _fail("internal", "top-p threshold is nonfinite")
    prefix_weight = 0.0
    retained: list[tuple[int, float]] = []
    for token_id, weight in weighted:
        prefix_weight += weight
        if not math.isfinite(prefix_weight):
            _fail("internal", "top-p cumulative weight is nonfinite")
        retained.append((token_id, weight))
        if prefix_weight >= threshold:
            break

    retained_sum = 0.0
    for _, weight in retained:
        retained_sum += weight
        if not math.isfinite(retained_sum):
            _fail("internal", "retained weight sum is nonfinite")
    if retained_sum <= 0.0:
        _fail("internal", "retained weight sum is not positive")

    rng = splitmix64_preview(state)
    categorical = categorical_select(retained, rng.unit)
    return {
        "categorical_cumulative_f64_hex": [
            _finite_hex(value, "categorical cumulative")
            for value in categorical.cumulative
        ],
        "chosen_token_id": categorical.token_id,
        "ordered_token_ids": ordered_token_ids,
        "random_unit_f64_hex": _finite_hex(rng.unit, "random unit"),
        "random_word_hex": _u64_hex(rng.word),
        "retained_sum_f64_hex": _finite_hex(retained_sum, "retained sum"),
        "retained_token_ids": [token_id for token_id, _ in retained],
        "scaled_logits_f64_hex": [
            _finite_hex(value, "scaled logit") for _, value in selected
        ],
        "softmax_total_f64_hex": _finite_hex(total_weight, "softmax total"),
        "softmax_weights_f64_hex": [
            _finite_hex(weight, "softmax weight") for _, weight in weighted
        ],
        "state_after_hex": _u64_hex(rng.next_state),
        "top_k_token_ids": [token_id for token_id, _ in selected],
        "top_p_threshold_f64_hex": _finite_hex(threshold, "top-p threshold"),
        "used_rounding_fallback": categorical.used_fallback,
    }


def _rng_case(case_id: str, state: int) -> dict[str, Any]:
    preview = splitmix64_preview(state)
    return {
        "id": case_id,
        "state_before_hex": _u64_hex(state),
        "state_after_hex": _u64_hex(preview.next_state),
        "unit_f64_hex": _finite_hex(preview.unit, "random unit"),
        "word_hex": _u64_hex(preview.word),
    }


def _sample_input(
    case_id: str,
    state: int,
    logits: Iterable[int],
    temperature: int,
    top_k: int,
    top_p: int,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "logit_f32_bits": [_u32_hex(value) for value in logits],
        "state_before_hex": _u64_hex(state),
        "temperature_f32_bits": _u32_hex(temperature),
        "top_k": top_k,
        "top_p_f32_bits": _u32_hex(top_p),
    }


def _render_sample_case(inputs: dict[str, Any]) -> dict[str, Any]:
    output = sample_preview(
        state=parse_u64_hex(inputs["state_before_hex"], "sample state"),
        logit_bits=[
            parse_f32_bits(value, "sample logit bits")
            for value in inputs["logit_f32_bits"]
        ],
        temperature_bits=parse_f32_bits(
            inputs["temperature_f32_bits"], "sample temperature bits"
        ),
        top_k=inputs["top_k"],
        top_p_bits=parse_f32_bits(inputs["top_p_f32_bits"], "sample top_p bits"),
    )
    return {"id": inputs["id"], "input": inputs, "output": output}


def _categorical_case(
    case_id: str,
    unit: float,
    candidates: Sequence[tuple[int, float]] = ((7, 1.0), (3, 1.0)),
) -> dict[str, Any]:
    result = categorical_select(candidates, unit)
    return {
        "id": case_id,
        "input": {
            "candidates": [
                {"token_id": token_id, "weight_f64_hex": weight.hex()}
                for token_id, weight in candidates
            ],
            "unit_f64_hex": unit.hex(),
        },
        "output": {
            "chosen_token_id": result.token_id,
            "cumulative_f64_hex": [value.hex() for value in result.cumulative],
            "used_rounding_fallback": result.used_fallback,
        },
    }


def _invalid_case(inputs: dict[str, Any], expected_error: str) -> dict[str, Any]:
    try:
        _render_sample_case(inputs)
    except SamplingError as error:
        if error.code != expected_error:
            _fail(
                "internal",
                f"invalid vector {inputs['id']} raised {error.code}, expected {expected_error}",
            )
    else:
        _fail("internal", f"invalid vector {inputs['id']} unexpectedly succeeded")
    return {"expected_error": expected_error, "id": inputs["id"], "input": inputs}


def _identity_payload(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "vector_id"}


def canonical_bytes(document: Any) -> bytes:
    return (
        json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def vector_identity(document: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_bytes(_identity_payload(document))).hexdigest()
    return f"sha256:{digest}"


def build_vectors() -> dict[str, Any]:
    """Build the complete, deterministic version-1 golden-vector document."""

    rng_cases = [
        _rng_case("state-zero", 0),
        _rng_case("state-one", 1),
        _rng_case("state-representative", 0x0123_4567_89AB_CDEF),
        _rng_case("state-max", MASK64),
    ]

    sample_inputs = [
        _sample_input(
            "stable-logit-ties",
            0,
            (0x3F80_0000, 0x4000_0000, 0x4000_0000, 0x0000_0000),
            0x3F80_0000,
            4,
            0x3F80_0000,
        ),
        _sample_input(
            "top-k-one",
            1,
            (0xBF80_0000, 0x4040_0000, 0x4000_0000),
            0x4000_0000,
            1,
            0x3DCC_CCCD,
        ),
        _sample_input(
            "top-p-half-equality",
            0x0123_4567_89AB_CDEF,
            (0x0000_0000, 0x0000_0000),
            0x3F80_0000,
            2,
            0x3F00_0000,
        ),
        _sample_input(
            "top-p-half-neighbor-below",
            0x0123_4567_89AB_CDF0,
            (0x0000_0000, 0x0000_0000),
            0x3F80_0000,
            2,
            0x3EFF_FFFF,
        ),
        _sample_input(
            "top-p-half-neighbor-above",
            0x0123_4567_89AB_CDF1,
            (0x0000_0000, 0x0000_0000),
            0x3F80_0000,
            2,
            0x3F00_0001,
        ),
        _sample_input(
            "smallest-positive-temperature",
            MASK64,
            (0x0000_0000, 0x8000_0001, 0x8000_0002),
            0x0000_0001,
            3,
            0x3F80_0000,
        ),
        _sample_input(
            "underflowed-nonmaximum-weight",
            0xF0E1_D2C3_B4A5_9687,
            (0x0000_0000, 0xBF80_0000, 0xC47A_0000),
            0x3F80_0000,
            3,
            0x3F80_0000,
        ),
    ]
    sample_cases = [_render_sample_case(inputs) for inputs in sample_inputs]

    categorical_cases = [
        _categorical_case("unit-zero", 0.0),
        _categorical_case("unit-nextafter-zero", math.nextafter(0.0, 1.0)),
        _categorical_case("unit-half-strict-boundary", 0.5),
        _categorical_case("unit-nextafter-one", math.nextafter(1.0, 0.0)),
        _categorical_case(
            "final-sum-one-ulp-low-fallback",
            math.nextafter(1.0, 0.0),
            ((7, 8.0), (3, 9.0), (11, 9.0), (5, 9.0)),
        ),
    ]

    valid_base = _sample_input(
        "valid-base",
        0,
        (0x0000_0000, 0x3F80_0000),
        0x3F80_0000,
        2,
        0x3F80_0000,
    )

    def changed(case_id: str, **updates: Any) -> dict[str, Any]:
        result = dict(valid_base)
        result.update(updates)
        result["id"] = case_id
        return result

    invalid_cases = [
        _invalid_case(changed("empty-logits", logit_f32_bits=[]), "invalid_request"),
        _invalid_case(
            changed("logit-qnan", logit_f32_bits=["0x7fc00000"]),
            "invalid_request",
        ),
        _invalid_case(
            changed("logit-positive-infinity", logit_f32_bits=["0x7f800000"]),
            "invalid_request",
        ),
        _invalid_case(
            changed("logit-negative-infinity", logit_f32_bits=["0xff800000"]),
            "invalid_request",
        ),
        _invalid_case(
            changed("temperature-zero", temperature_f32_bits="0x00000000"),
            "invalid_request",
        ),
        _invalid_case(
            changed("temperature-negative", temperature_f32_bits="0xbf800000"),
            "invalid_request",
        ),
        _invalid_case(
            changed("temperature-qnan", temperature_f32_bits="0x7fc00000"),
            "invalid_request",
        ),
        _invalid_case(
            changed("temperature-infinity", temperature_f32_bits="0x7f800000"),
            "invalid_request",
        ),
        _invalid_case(changed("top-k-zero", top_k=0), "invalid_request"),
        _invalid_case(changed("top-k-above-vocab", top_k=3), "invalid_request"),
        _invalid_case(
            changed("top-p-zero", top_p_f32_bits="0x00000000"),
            "invalid_request",
        ),
        _invalid_case(
            changed("top-p-above-one", top_p_f32_bits="0x3f800001"),
            "invalid_request",
        ),
        _invalid_case(
            changed("top-p-qnan", top_p_f32_bits="0x7fc00000"),
            "invalid_request",
        ),
        _invalid_case(
            changed("top-p-infinity", top_p_f32_bits="0x7f800000"),
            "invalid_request",
        ),
    ]

    document: dict[str, Any] = {
        "categorical_cases": categorical_cases,
        "float_encoding": FLOAT_ENCODING,
        "invalid_cases": invalid_cases,
        "rng_cases": rng_cases,
        "sample_cases": sample_cases,
        "schema": SCHEMA,
        "specification": SPECIFICATION,
    }
    document["vector_id"] = vector_identity(document)
    return document


def generated_bytes() -> bytes:
    return canonical_bytes(build_vectors())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_vector", f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _expect_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        _fail("invalid_vector", f"{label} does not match the closed schema")
    return value


def _expect_case_ids(cases: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or not cases:
        _fail("invalid_vector", f"{label} must be a nonempty array")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            _fail("invalid_vector", f"{label}[{index}] must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            _fail("invalid_vector", f"{label}[{index}] has an invalid or duplicate ID")
        seen.add(case_id)
        checked.append(case)
    return checked


def _validate_sample_input(value: Any, label: str) -> dict[str, Any]:
    value = _expect_keys(
        value,
        {
            "id",
            "logit_f32_bits",
            "state_before_hex",
            "temperature_f32_bits",
            "top_k",
            "top_p_f32_bits",
        },
        label,
    )
    if not isinstance(value["id"], str) or not value["id"]:
        _fail("invalid_vector", f"{label} ID must be a nonempty string")
    logits = value["logit_f32_bits"]
    if not isinstance(logits, list):
        _fail("invalid_vector", f"{label} logits must be an array")
    for index, bits in enumerate(logits):
        parse_f32_bits(bits, f"{label} logit {index}")
    parse_u64_hex(value["state_before_hex"], f"{label} state")
    parse_f32_bits(value["temperature_f32_bits"], f"{label} temperature")
    _require_plain_int(value["top_k"], f"{label} top_k")
    parse_f32_bits(value["top_p_f32_bits"], f"{label} top_p")
    return value


def validate_document(document: Any) -> dict[str, Any]:
    """Validate the closed schema, identity, and every recomputable result."""

    document = _expect_keys(
        document,
        {
            "categorical_cases",
            "float_encoding",
            "invalid_cases",
            "rng_cases",
            "sample_cases",
            "schema",
            "specification",
            "vector_id",
        },
        "sampling vector document",
    )
    if document["schema"] != SCHEMA:
        _fail("invalid_vector", "sampling vector schema is unsupported")
    if document["specification"] != SPECIFICATION:
        _fail("invalid_vector", "sampling specification identity differs")
    if document["float_encoding"] != FLOAT_ENCODING:
        _fail("invalid_vector", "sampling float encoding differs")
    if document["vector_id"] != vector_identity(document):
        _fail("invalid_vector", "sampling vector identity digest differs")

    for case in _expect_case_ids(document["rng_cases"], "rng_cases"):
        _expect_keys(
            case,
            {"id", "state_before_hex", "state_after_hex", "unit_f64_hex", "word_hex"},
            f"RNG case {case['id']}",
        )
        state = parse_u64_hex(case["state_before_hex"], "RNG state")
        preview = splitmix64_preview(state)
        expected = _rng_case(case["id"], state)
        if case != expected:
            _fail("invalid_vector", f"RNG case {case['id']} result differs")
        _parse_f64_hex(case["unit_f64_hex"], "RNG unit")
        parse_u64_hex(case["state_after_hex"], "RNG next state")
        parse_u64_hex(case["word_hex"], "RNG word")
        if preview.unit.hex() != case["unit_f64_hex"]:
            _fail("invalid_vector", f"RNG case {case['id']} unit differs")

    for case in _expect_case_ids(document["sample_cases"], "sample_cases"):
        _expect_keys(case, {"id", "input", "output"}, f"sample case {case['id']}")
        inputs = _validate_sample_input(case["input"], f"sample case {case['id']} input")
        if inputs["id"] != case["id"]:
            _fail("invalid_vector", f"sample case {case['id']} IDs differ")
        if _render_sample_case(inputs)["output"] != case["output"]:
            _fail("invalid_vector", f"sample case {case['id']} result differs")

    for case in _expect_case_ids(document["categorical_cases"], "categorical_cases"):
        _expect_keys(case, {"id", "input", "output"}, f"categorical case {case['id']}")
        inputs = _expect_keys(
            case["input"], {"candidates", "unit_f64_hex"}, f"categorical case {case['id']} input"
        )
        if not isinstance(inputs["candidates"], list):
            _fail("invalid_vector", f"categorical case {case['id']} candidates must be an array")
        candidates: list[tuple[int, float]] = []
        for index, candidate in enumerate(inputs["candidates"]):
            candidate = _expect_keys(
                candidate,
                {"token_id", "weight_f64_hex"},
                f"categorical case {case['id']} candidate {index}",
            )
            token_id = _require_plain_int(candidate["token_id"], "categorical token ID")
            weight = _parse_f64_hex(candidate["weight_f64_hex"], "categorical weight")
            candidates.append((token_id, weight))
        unit = _parse_f64_hex(inputs["unit_f64_hex"], "categorical unit")
        result = categorical_select(candidates, unit)
        expected_output = {
            "chosen_token_id": result.token_id,
            "cumulative_f64_hex": [value.hex() for value in result.cumulative],
            "used_rounding_fallback": result.used_fallback,
        }
        if case["output"] != expected_output:
            _fail("invalid_vector", f"categorical case {case['id']} result differs")

    for case in _expect_case_ids(document["invalid_cases"], "invalid_cases"):
        _expect_keys(
            case,
            {"expected_error", "id", "input"},
            f"invalid case {case['id']}",
        )
        inputs = _validate_sample_input(case["input"], f"invalid case {case['id']} input")
        if inputs["id"] != case["id"]:
            _fail("invalid_vector", f"invalid case {case['id']} IDs differ")
        expected_error = case["expected_error"]
        if not isinstance(expected_error, str):
            _fail("invalid_vector", f"invalid case {case['id']} error must be a string")
        _invalid_case(inputs, expected_error)
    return document


def parse_document_bytes(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_VECTOR_BYTES:
        _fail("invalid_vector", "sampling vector file is empty or exceeds its byte limit")
    if not raw.endswith(b"\n") or b"\r" in raw or not raw.isascii():
        _fail("invalid_vector", "sampling vector file must be canonical ASCII ending in LF")
    try:
        document = json.loads(raw.decode("ascii"), object_pairs_hook=_unique_object)
    except SamplingError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        _fail("invalid_vector", f"sampling vector JSON is invalid: {error}")
    if canonical_bytes(document) != raw:
        _fail("invalid_vector", "sampling vector JSON is not in canonical formatting")
    return validate_document(document)


def check_path(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail("invalid_vector", f"cannot read sampling vectors: {error}")
    parse_document_bytes(raw)
    expected = generated_bytes()
    if raw != expected:
        _fail("invalid_vector", "sampling vectors differ from independent regeneration")
    return hashlib.sha256(raw).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify committed vectors")
    action.add_argument("--write", action="store_true", help="rewrite committed vectors")
    parser.add_argument("--path", type=Path, default=DEFAULT_VECTOR_PATH)
    arguments = parser.parse_args(argv)

    try:
        if arguments.write:
            payload = generated_bytes()
            _write_atomic(arguments.path, payload)
            digest = hashlib.sha256(payload).hexdigest()
            print(f"wrote {arguments.path} sha256:{digest}")
        else:
            digest = check_path(arguments.path)
            print(f"verified {arguments.path} sha256:{digest}")
    except (OSError, SamplingError) as error:
        print(f"sampling oracle: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
