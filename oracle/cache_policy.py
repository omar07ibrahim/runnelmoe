#!/usr/bin/env python3
"""Independent, deterministic reference for RunnelMoE online cache policies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, NoReturn


TRACE_SCHEMA = "runnel.cache-trace/1"
PREFETCH_MODEL = "instant-between-events-v1"
MAX_U64 = (1 << 64) - 1
MAX_U32 = (1 << 32) - 1
MAX_TRACE_BYTES = 5 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024
MAX_PAGES = 4_096
MAX_EVENTS = 100_000
MAX_RECORDS = 1 + MAX_PAGES + MAX_EVENTS
MAX_PREDICTIONS = 16
PPM = 1_000_000
SKETCH_COUNTER_MAX = 15
HASH_SEEDS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
    0xBE5466CF34E90C6C,
    0xC0AC29B7C97C50DD,
    0x3F84D5B5B5470917,
)
TRACE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class OracleError(ValueError):
    """A bounded-input, configuration, or accounting failure."""


def _fail(message: str) -> NoReturn:
    raise OracleError(message)


def _u64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    if value < 0 or value > MAX_U64:
        _fail(f"{label} is outside the unsigned 64-bit range")
    return value


def _u32(value: Any, label: str) -> int:
    value = _u64(value, label)
    if value > MAX_U32:
        _fail(f"{label} is outside the unsigned 32-bit range")
    return value


def _checked_add(left: int, right: int, label: str) -> int:
    if left < 0 or right < 0 or left > MAX_U64 - right:
        _fail(f"{label} overflows an unsigned 64-bit counter")
    return left + right


def _expect_keys(record: dict[str, Any], expected: tuple[str, ...], label: str) -> None:
    if tuple(record) != expected:
        _fail(f"{label} fields or field order differ from the canonical schema")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class Page:
    page_id: int
    logical_bytes: int
    charge_bytes: int
    layer: int | None
    expert: int | None
    ordinal: int | None


@dataclass(frozen=True)
class DemandEvent:
    sequence: int
    request: int
    step: int
    page_id: int


@dataclass(frozen=True)
class RouterSignalEvent:
    sequence: int
    request: int
    target_step: int
    layer: int
    predictions: tuple[tuple[int, int], ...]


TraceEvent = DemandEvent | RouterSignalEvent


@dataclass(frozen=True)
class Trace:
    pages: tuple[Page, ...]
    events: tuple[TraceEvent, ...]

    @property
    def demands(self) -> tuple[int, ...]:
        """Compatibility view used by demand-only oracle tests."""

        return tuple(
            event.page_id
            for event in self.events
            if isinstance(event, DemandEvent)
        )

    @property
    def minimum_charge(self) -> int:
        if not self.pages:
            _fail("the page catalog is empty")
        return min(page.charge_bytes for page in self.pages)


def _read_records(path: Path) -> list[dict[str, Any]]:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(f"cannot open trace without following links: {error}")
    try:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect opened trace: {error}")
        if not stat.S_ISREG(before.st_mode):
            _fail("trace path must name a regular file")
        if before.st_size > MAX_TRACE_BYTES:
            _fail(f"trace exceeds the {MAX_TRACE_BYTES}-byte limit")

        chunks: list[bytes] = []
        bytes_read = 0
        try:
            while bytes_read <= MAX_TRACE_BYTES:
                remaining = MAX_TRACE_BYTES + 1 - bytes_read
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
        except OSError as error:
            _fail(f"cannot read trace: {error}")
        raw = b"".join(chunks)
        if len(raw) > MAX_TRACE_BYTES:
            _fail(f"trace exceeds the {MAX_TRACE_BYTES}-byte limit")
        try:
            after = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect trace after reading: {error}")
        if (
            len(raw) != after.st_size
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            _fail("trace changed while it was being read")
    finally:
        os.close(descriptor)
    if not raw or not raw.endswith(b"\n"):
        _fail("canonical trace must end with LF")
    if b"\r" in raw:
        _fail("canonical trace must not contain CR bytes")
    if not raw.isascii():
        _fail("canonical trace must contain ASCII bytes only")
    record_count = raw.count(b"\n")
    if record_count > MAX_RECORDS:
        _fail(f"trace exceeds the {MAX_RECORDS}-record limit")

    records: list[dict[str, Any]] = []
    for line_number, encoded in enumerate(raw[:-1].split(b"\n"), start=1):
        if not encoded:
            _fail(f"line {line_number} is empty")
        if len(encoded) > MAX_LINE_BYTES:
            _fail(f"line {line_number} exceeds the {MAX_LINE_BYTES}-byte limit")
        text = encoded.decode("ascii")
        try:
            record = json.loads(text, object_pairs_hook=_unique_object)
        except OracleError:
            raise
        except json.JSONDecodeError as error:
            _fail(f"line {line_number} is invalid JSON: {error.msg}")
        except RecursionError:
            _fail(f"line {line_number} exceeds the JSON nesting limit")
        except ValueError as error:
            _fail(f"line {line_number} contains an invalid JSON value: {error}")
        if not isinstance(record, dict):
            _fail(f"line {line_number} must be a JSON object")
        try:
            canonical = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        except (ValueError, RecursionError) as error:
            _fail(f"line {line_number} cannot be encoded canonically: {error}")
        if canonical != text:
            _fail(f"line {line_number} is not compact canonical JSON")
        records.append(record)
    return records


def _validate_class(value: Any, page_id: int) -> tuple[int, int, int] | None:
    if not isinstance(value, dict):
        _fail(f"page {page_id} class must be an object")
    kind = value.get("kind")
    if kind == "shared":
        _expect_keys(value, ("kind",), f"page {page_id} shared class")
        return None
    if kind == "expert":
        _expect_keys(
            value,
            ("kind", "layer", "expert", "ordinal"),
            f"page {page_id} expert class",
        )
        layer = _u32(value["layer"], f"page {page_id} layer")
        expert = _u32(value["expert"], f"page {page_id} expert")
        ordinal = _u32(value["ordinal"], f"page {page_id} ordinal")
        return (layer, expert, ordinal)
    _fail(f"page {page_id} has an unknown class kind")


def parse_trace(path: Path) -> Trace:
    """Parse the closed canonical JSONL trace into independent Python records."""

    records = _read_records(path)
    if not records:
        _fail("trace is missing its header")
    header = records[0]
    _expect_keys(
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
        "header",
    )
    if header["kind"] != "header":
        _fail("header kind must be 'header'")
    if header["schema"] != TRACE_SCHEMA:
        _fail("unsupported trace schema")
    trace_id = header["trace_id"]
    if not isinstance(trace_id, str) or TRACE_ID.fullmatch(trace_id) is None:
        _fail("trace_id does not match the canonical identifier grammar")
    page_count = _u64(header["page_count"], "page_count")
    event_count = _u64(header["event_count"], "event_count")
    if page_count > MAX_PAGES:
        _fail(f"page_count exceeds the {MAX_PAGES}-page limit")
    if event_count > MAX_EVENTS:
        _fail(f"event_count exceeds the {MAX_EVENTS}-event limit")
    quantum = _u64(header["charge_quantum"], "charge_quantum")
    if quantum == 0:
        _fail("charge_quantum must be positive")
    if header["prefetch_model"] != PREFETCH_MODEL:
        _fail("unsupported prefetch timing model")
    declared_records = 1 + page_count + event_count
    if len(records) != declared_records:
        _fail("record count differs from the header declaration")

    pages: list[Page] = []
    expert_catalog: set[tuple[int, int]] = set()
    expert_tuples: set[tuple[int, int, int]] = set()
    for expected_id, record in enumerate(records[1 : 1 + page_count]):
        _expect_keys(
            record,
            ("id", "logical_bytes", "charge_bytes", "class"),
            f"page {expected_id}",
        )
        page_id = _u32(record["id"], f"page {expected_id} id")
        if page_id != expected_id:
            _fail(f"catalog page {expected_id} has non-dense id {page_id}")
        logical_bytes = _u64(record["logical_bytes"], f"page {page_id} logical_bytes")
        charge_bytes = _u64(record["charge_bytes"], f"page {page_id} charge_bytes")
        if logical_bytes == 0 or charge_bytes == 0:
            _fail(f"page {page_id} byte sizes must be positive")
        if logical_bytes > charge_bytes:
            _fail(f"page {page_id} logical bytes exceed charged bytes")
        if charge_bytes % quantum:
            _fail(f"page {page_id} charged bytes violate charge_quantum")
        expert_class = _validate_class(record["class"], page_id)
        if expert_class is not None:
            if expert_class in expert_tuples:
                _fail(
                    "duplicate expert page tuple "
                    f"({expert_class[0]}, {expert_class[1]}, {expert_class[2]})"
                )
            expert_tuples.add(expert_class)
            expert_catalog.add((expert_class[0], expert_class[1]))
        if expert_class is None:
            pages.append(Page(page_id, logical_bytes, charge_bytes, None, None, None))
        else:
            pages.append(
                Page(
                    page_id,
                    logical_bytes,
                    charge_bytes,
                    expert_class[0],
                    expert_class[1],
                    expert_class[2],
                )
            )

    parsed_events: list[TraceEvent] = []
    latest_step: dict[int, int] = {}
    realized_steps: set[tuple[int, int]] = set()
    signal_targets: set[tuple[int, int, int]] = set()
    events = records[1 + page_count :]
    for expected_sequence, record in enumerate(events):
        kind = record.get("kind")
        if kind == "demand":
            _expect_keys(
                record,
                ("kind", "sequence", "request", "step", "page"),
                f"event {expected_sequence}",
            )
            sequence = _u64(record["sequence"], f"event {expected_sequence} sequence")
            request = _u64(record["request"], f"event {expected_sequence} request")
            step = _u64(record["step"], f"event {expected_sequence} step")
            page_id = _u32(record["page"], f"event {expected_sequence} page")
            if sequence != expected_sequence:
                _fail(f"event {expected_sequence} has a non-dense sequence")
            if page_id >= len(pages):
                _fail(f"event {expected_sequence} references an unknown page")
            if request in latest_step and step < latest_step[request]:
                _fail(f"request {request} demand step regresses")
            latest_step[request] = step
            realized_steps.add((request, step))
            parsed_events.append(
                DemandEvent(
                    sequence=sequence,
                    request=request,
                    step=step,
                    page_id=page_id,
                )
            )
        elif kind == "router_signal":
            _expect_keys(
                record,
                (
                    "kind",
                    "sequence",
                    "request",
                    "target_step",
                    "layer",
                    "predictions",
                ),
                f"event {expected_sequence}",
            )
            sequence = _u64(record["sequence"], f"event {expected_sequence} sequence")
            request = _u64(record["request"], f"event {expected_sequence} request")
            target_step = _u64(
                record["target_step"], f"event {expected_sequence} target_step"
            )
            layer = _u32(record["layer"], f"event {expected_sequence} layer")
            if sequence != expected_sequence:
                _fail(f"event {expected_sequence} has a non-dense sequence")
            if request in latest_step and target_step <= latest_step[request]:
                _fail(f"event {expected_sequence} predicts a non-future step")
            target = (request, target_step, layer)
            if target in signal_targets:
                _fail(f"event {expected_sequence} repeats a router target")
            signal_targets.add(target)
            predictions = record["predictions"]
            if not isinstance(predictions, list) or len(predictions) > MAX_PREDICTIONS:
                _fail(f"event {expected_sequence} has an invalid prediction list")
            score_sum = 0
            seen_experts: set[int] = set()
            for prediction_index, prediction in enumerate(predictions):
                if not isinstance(prediction, dict):
                    _fail(f"event {expected_sequence} prediction must be an object")
                _expect_keys(
                    prediction,
                    ("expert", "score_ppm"),
                    f"event {expected_sequence} prediction {prediction_index}",
                )
                expert = _u32(
                    prediction["expert"],
                    f"event {expected_sequence} prediction expert",
                )
                score = _u32(
                    prediction["score_ppm"],
                    f"event {expected_sequence} prediction score",
                )
                if expert in seen_experts:
                    _fail(f"event {expected_sequence} repeats an expert")
                if (layer, expert) not in expert_catalog:
                    _fail(f"event {expected_sequence} references an absent expert")
                if score > PPM:
                    _fail(f"event {expected_sequence} prediction score exceeds one million")
                score_sum = _checked_add(score_sum, score, "router score sum")
                if score_sum > PPM:
                    _fail(f"event {expected_sequence} prediction scores exceed one million")
                seen_experts.add(expert)
            parsed_events.append(
                RouterSignalEvent(
                    sequence=sequence,
                    request=request,
                    target_step=target_step,
                    layer=layer,
                    predictions=tuple(
                        (prediction["expert"], prediction["score_ppm"])
                        for prediction in predictions
                    ),
                )
            )
        else:
            _fail(f"event {expected_sequence} has an unknown kind")

    for request, target_step, _layer in signal_targets:
        if (request, target_step) not in realized_steps:
            _fail(f"router target ({request}, {target_step}) is never realized")
    return Trace(tuple(pages), tuple(parsed_events))


@dataclass
class Resident:
    page_id: int
    charge_bytes: int
    last_touch: int
    segment: str


class TinyLfuSketch:
    """A bounded count-min estimate with a one-bit first-hit filter."""

    def __init__(self, depth: int, width: int, sample_accesses: int) -> None:
        if depth < 1 or depth > len(HASH_SEEDS):
            _fail("sketch_depth must be between 1 and 8")
        if width < 64 or width > 65_536:
            _fail("sketch_width must be between 64 and 65536")
        if sample_accesses < 1 or sample_accesses > MAX_U64:
            _fail("sample_accesses must be a positive unsigned 64-bit integer")
        self.depth = depth
        self.width = width
        self.sample_accesses = sample_accesses
        self.counters = [0] * (depth * width)
        self.door_bits = 0
        self.observed = 0

    @staticmethod
    def _splitmix(value: int) -> int:
        # Independent SplitMix64 finalizer encoding; source and reference
        # license provenance are recorded in docs/PRIOR_ART.md.
        value = (value + 0x9E3779B97F4A7C15) & MAX_U64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MAX_U64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MAX_U64
        return (value ^ (value >> 31)) & MAX_U64

    def _column(self, page_id: int, row: int) -> int:
        return self._splitmix(page_id ^ HASH_SEEDS[row]) % self.width

    def _door_contains(self, page_id: int) -> bool:
        return all(
            self.door_bits & (1 << self._column(page_id, row))
            for row in range(self.depth)
        )

    def estimate(self, page_id: int) -> int:
        minimum = min(
            self.counters[row * self.width + self._column(page_id, row)]
            for row in range(self.depth)
        )
        return minimum + int(self._door_contains(page_id))

    def observe(self, page_id: int) -> None:
        if self._door_contains(page_id):
            indices = [
                row * self.width + self._column(page_id, row)
                for row in range(self.depth)
            ]
            minimum = min(self.counters[index] for index in indices)
            for index in indices:
                if self.counters[index] == minimum:
                    self.counters[index] = min(
                        self.counters[index] + 1, SKETCH_COUNTER_MAX
                    )
        else:
            for row in range(self.depth):
                self.door_bits |= 1 << self._column(page_id, row)
        self.observed = _checked_add(self.observed, 1, "TinyLFU observations")
        if self.observed >= self.sample_accesses:
            self.counters = [counter >> 1 for counter in self.counters]
            self.door_bits = 0
            self.observed >>= 1


@dataclass
class Metrics:
    demand_accesses: int = 0
    ordinary_demand_hits: int = 0
    demand_misses: int = 0
    admissions: int = 0
    bypasses: int = 0
    evictions: int = 0
    demand_load_bytes: int = 0
    final_resident_charge_bytes: int = 0
    peak_resident_charge_bytes: int = 0

    def add(self, field: str, amount: int) -> None:
        current = getattr(self, field)
        setattr(self, field, _checked_add(current, amount, field))

    def as_dict(self) -> dict[str, int]:
        return {
            "demand_accesses": self.demand_accesses,
            "ordinary_demand_hits": self.ordinary_demand_hits,
            "demand_misses": self.demand_misses,
            "admissions": self.admissions,
            "bypasses": self.bypasses,
            "evictions": self.evictions,
            "demand_load_bytes": self.demand_load_bytes,
            "final_resident_charge_bytes": self.final_resident_charge_bytes,
            "peak_resident_charge_bytes": self.peak_resident_charge_bytes,
        }


def simulate(
    trace: Trace,
    policy: str,
    capacity_bytes: int,
    *,
    protected_fraction_ppm: int = 750_000,
    minimum_score_ppm: int = 100_000,
    max_experts_per_signal: int = 2,
    sketch_depth: int = 4,
    sketch_width: int = 2_048,
    sample_accesses: int = 100,
) -> dict[str, int]:
    """Replay ordered events with independent byte-capacity state."""

    capacity_bytes = _u64(capacity_bytes, "capacity_bytes")
    protected_fraction_ppm = _u32(
        protected_fraction_ppm, "protected_fraction_ppm"
    )
    if protected_fraction_ppm > PPM:
        _fail("protected_fraction_ppm exceeds one million")
    minimum_score_ppm = _u32(minimum_score_ppm, "minimum_score_ppm")
    if minimum_score_ppm > PPM:
        _fail("minimum_score_ppm exceeds one million")
    max_experts_per_signal = _u32(
        max_experts_per_signal, "max_experts_per_signal"
    )
    if max_experts_per_signal == 0:
        _fail("max_experts_per_signal must be positive")
    if policy not in {"lru", "slru", "tiny-lfu", "router-admit"}:
        _fail("unsupported policy")
    sketch = (
        TinyLfuSketch(sketch_depth, sketch_width, sample_accesses)
        if policy == "tiny-lfu"
        else None
    )
    protected_target = capacity_bytes * protected_fraction_ppm // PPM
    residents: dict[int, Resident] = {}
    active_scores: dict[tuple[int, int, int], dict[int, int]] = {}
    resident_bytes = 0
    touch = 0
    metrics = Metrics()
    segmented = policy in {"slru", "router-admit"}

    def active_score(request: int, step: int, page: Page) -> int:
        if policy != "router-admit" or page.layer is None or page.expert is None:
            return 0
        scores = active_scores.get((request, step, page.layer))
        if scores is None:
            return 0
        return scores.get(page.expert, 0)

    def assert_invariant() -> None:
        ledger = sum(entry.charge_bytes for entry in residents.values())
        if ledger != resident_bytes:
            _fail("resident byte ledger differs from resident entries")
        if resident_bytes < 0 or resident_bytes > capacity_bytes:
            _fail("resident bytes exceed the configured capacity")
        if any(key != entry.page_id for key, entry in residents.items()):
            _fail("resident map key differs from its entry")

    def rebalance_protected(demotion_touch: int) -> None:
        while True:
            protected_bytes = sum(
                entry.charge_bytes
                for entry in residents.values()
                if entry.segment == "protected"
            )
            if protected_bytes <= protected_target:
                return
            victim = min(
                (
                    entry
                    for entry in residents.values()
                    if entry.segment == "protected"
                ),
                key=lambda entry: (entry.last_touch, entry.page_id),
            )
            victim.segment = "probation"
            victim.last_touch = demotion_touch

    for event in trace.events:
        if isinstance(event, RouterSignalEvent):
            if policy == "router-admit":
                selected = sorted(
                    (
                        (expert, score)
                        for expert, score in event.predictions
                        if score >= minimum_score_ppm
                    ),
                    key=lambda prediction: (-prediction[1], prediction[0]),
                )[:max_experts_per_signal]
                key = (event.request, event.target_step, event.layer)
                if selected:
                    active_scores[key] = dict(selected)
                else:
                    active_scores.pop(key, None)
            continue

        request = event.request
        step = event.step
        page_id = event.page_id
        if policy == "router-admit":
            expired = [
                key
                for key in active_scores
                if key[0] == request and key[1] < step
            ]
            for key in expired:
                del active_scores[key]
        page = trace.pages[page_id]
        if sketch is not None:
            sketch.observe(page_id)
        metrics.add("demand_accesses", 1)
        touch = _checked_add(touch, 1, "policy touch clock")
        resident = residents.get(page_id)
        if resident is not None:
            metrics.add("ordinary_demand_hits", 1)
            resident.last_touch = touch
            if (
                segmented
                and resident.segment == "probation"
                and protected_target > 0
                and resident.charge_bytes <= protected_target
            ):
                resident.segment = "protected"
                rebalance_protected(touch)
            assert_invariant()
            continue

        metrics.add("demand_misses", 1)
        metrics.add("demand_load_bytes", page.logical_bytes)
        if page.charge_bytes > capacity_bytes:
            metrics.add("bypasses", 1)
            assert_invariant()
            continue

        available = capacity_bytes - resident_bytes
        victims: list[Resident] = []
        if page.charge_bytes > available:
            needed = page.charge_bytes - available

            def victim_key(entry: Resident) -> tuple[int, int, int, int]:
                segment_rank = (
                    0
                    if not segmented or entry.segment == "probation"
                    else 1
                )
                router_score = (
                    active_score(request, step, trace.pages[entry.page_id])
                    if policy == "router-admit"
                    else 0
                )
                return (
                    segment_rank,
                    router_score,
                    entry.last_touch,
                    entry.page_id,
                )

            released = 0
            for candidate in sorted(residents.values(), key=victim_key):
                victims.append(candidate)
                released = _checked_add(
                    released, candidate.charge_bytes, "planned victim bytes"
                )
                if released >= needed:
                    break
            if released < needed:
                _fail("resident plan cannot release enough bytes")

        if sketch is not None and victims:
            candidate_frequency = sketch.estimate(page_id)
            victim_charge = sum(victim.charge_bytes for victim in victims)
            victim_frequency = sum(
                sketch.estimate(victim.page_id) for victim in victims
            )
            if not (
                candidate_frequency * victim_charge
                > victim_frequency * page.charge_bytes
            ):
                metrics.add("bypasses", 1)
                assert_invariant()
                continue

        if policy == "router-admit" and victims and active_scores:
            candidate_score = active_score(request, step, page)
            victim_charge = sum(victim.charge_bytes for victim in victims)
            victim_score = sum(
                active_score(request, step, trace.pages[victim.page_id])
                for victim in victims
            )
            if not (
                (candidate_score == 0 and victim_score == 0)
                or candidate_score * victim_charge
                > victim_score * page.charge_bytes
            ):
                metrics.add("bypasses", 1)
                assert_invariant()
                continue

        for victim in victims:
            removed = residents.pop(victim.page_id)
            if removed is not victim:
                _fail("victim identity changed before atomic eviction")
            if removed.charge_bytes > resident_bytes:
                _fail("resident byte ledger underflow")
            resident_bytes -= removed.charge_bytes
            metrics.add("evictions", 1)

        segment = "probation" if segmented else "lru"
        if page_id in residents:
            _fail("attempted to admit an already resident page")
        new_resident_bytes = _checked_add(
            resident_bytes, page.charge_bytes, "resident charge bytes"
        )
        if new_resident_bytes > capacity_bytes:
            _fail("atomic victim plan did not free enough capacity")
        residents[page_id] = Resident(page_id, page.charge_bytes, touch, segment)
        resident_bytes = new_resident_bytes
        metrics.add("admissions", 1)
        metrics.peak_resident_charge_bytes = max(
            metrics.peak_resident_charge_bytes, resident_bytes
        )
        assert_invariant()

    metrics.final_resident_charge_bytes = resident_bytes
    if (
        _checked_add(
            metrics.ordinary_demand_hits,
            metrics.demand_misses,
            "demand accounting identity",
        )
        != metrics.demand_accesses
    ):
        _fail("demand hit/miss accounting identity failed")
    assert_invariant()
    return metrics.as_dict()


def _argument_u64(text: str) -> int:
    try:
        return _u64(int(text, 10), "argument")
    except (OracleError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _argument_u32(text: str) -> int:
    try:
        return _u32(int(text, 10), "argument")
    except (OracleError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument(
        "--policy",
        required=True,
        choices=("lru", "slru", "tiny-lfu", "router-admit"),
    )
    parser.add_argument("--capacity-bytes", required=True, type=_argument_u64)
    parser.add_argument(
        "--protected-fraction-ppm", type=_argument_u32, default=750_000
    )
    parser.add_argument("--minimum-score-ppm", type=_argument_u32, default=100_000)
    parser.add_argument("--max-experts-per-signal", type=_argument_u32, default=2)
    parser.add_argument("--sketch-depth", type=_argument_u32, default=4)
    parser.add_argument("--sketch-width", type=_argument_u32, default=2_048)
    parser.add_argument("--sample-accesses", type=_argument_u64)
    parsed = parser.parse_args(arguments)
    try:
        trace = parse_trace(parsed.trace)
        tiny_lfu_arguments: dict[str, int] = {}
        if parsed.policy == "tiny-lfu":
            sample_accesses = parsed.sample_accesses
            if sample_accesses is None:
                estimated_entries = (
                    parsed.capacity_bytes // trace.minimum_charge
                    if trace.pages
                    else 0
                )
                sample_accesses = max(estimated_entries, 1)
                if sample_accesses > MAX_U64 // 10:
                    _fail("default sample_accesses overflows an unsigned 64-bit counter")
                sample_accesses *= 10
            tiny_lfu_arguments["sample_accesses"] = sample_accesses
        result = simulate(
            trace,
            parsed.policy,
            parsed.capacity_bytes,
            protected_fraction_ppm=parsed.protected_fraction_ppm,
            minimum_score_ppm=parsed.minimum_score_ppm,
            max_experts_per_signal=parsed.max_experts_per_signal,
            sketch_depth=parsed.sketch_depth,
            sketch_width=parsed.sketch_width,
            **tiny_lfu_arguments,
        )
    except OracleError as error:
        print(f"cache_policy.py: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
