#!/usr/bin/env python3
"""Bounded Linux canonical-byte primitives for actor race-history captures.

This module deliberately stops below the semantic race model.  It provides a
fail-closed file reader and a small cursor for the capture's compact,
closed-schema JSON subset.  A later semantic checker can therefore consume
bytes directly without first handing attacker-controlled input to a general
JSON decoder.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import NoReturn


MAX_CAPTURE_BYTES = 32 * 1024 * 1024
MAX_NESTING_DEPTH = 16
MAX_STRING_BYTES = 128
MAX_U64 = (1 << 64) - 1
_READ_CHUNK_BYTES = 64 * 1024


class ActorRaceHistoryError(ValueError):
    """A custody, framing, or canonical-token failure."""


def _fail(message: str) -> NoReturn:
    raise ActorRaceHistoryError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the metadata fields that must remain stable during a read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_capture_bytes(path: str | Path) -> bytes:
    """Read and frame-check one bounded, immutable, regular-file capture.

    The descriptor is opened nonblocking without following a leaf symlink.
    The reader checks descriptor identity and size before and after the
    bounded read so descriptor-visible mutation is rejected.  An already-open
    descriptor deliberately remains authoritative if its pathname is later
    replaced.
    """

    required_flags = ("O_NONBLOCK", "O_CLOEXEC", "O_NOFOLLOW")
    _require(
        all(hasattr(os, name) for name in required_flags),
        "safe capture reads require Linux open flags",
    )
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as error:
        _fail(f"cannot safely open race-history capture: {error}")

    try:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect race-history capture: {error}")
        _require(stat.S_ISREG(before.st_mode), "capture must be a regular file")
        _require(before.st_size > 0, "capture must not be empty")
        _require(
            before.st_size <= MAX_CAPTURE_BYTES,
            "capture exceeds the 32 MiB byte limit",
        )

        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            remaining_probe = MAX_CAPTURE_BYTES + 1 - bytes_read
            if remaining_probe <= 0:
                _fail("capture exceeds the 32 MiB byte limit")
            try:
                chunk = os.read(
                    descriptor,
                    min(_READ_CHUNK_BYTES, remaining_probe),
                )
            except OSError as error:
                _fail(f"cannot read race-history capture: {error}")
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > MAX_CAPTURE_BYTES:
                _fail("capture exceeds the 32 MiB byte limit")

        try:
            after = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot re-inspect race-history capture: {error}")
        _require(
            _stat_identity(before) == _stat_identity(after),
            "capture changed while it was being read",
        )
        _require(
            bytes_read == after.st_size,
            "capture byte count does not match its file size",
        )
    finally:
        os.close(descriptor)

    raw = b"".join(chunks)
    CanonicalCursor(raw)
    return raw


@dataclass
class _Frame:
    kind: str
    item_count: int = 0
    last_key: bytes | None = None
    value_pending: bool = False


class CanonicalCursor:
    """Cursor over the actor capture's exact compact ASCII JSON subset."""

    def __init__(self, raw: bytes):
        _require(type(raw) is bytes, "capture must be supplied as exact bytes")
        self._validate_frame(raw)
        self.raw = raw
        self.position = 0
        self._frames: list[_Frame] = []
        self._root_started = False
        self._root_complete = False

    @staticmethod
    def _validate_frame(raw: bytes) -> None:
        _require(raw, "capture must not be empty")
        _require(
            len(raw) <= MAX_CAPTURE_BYTES,
            "capture exceeds the 32 MiB byte limit",
        )
        _require(raw.isascii(), "capture must contain only ASCII bytes")
        _require(raw.endswith(b"\n"), "capture must end with one LF byte")
        _require(
            b"\n" not in raw[:-1] and b"\r" not in raw,
            "capture must contain exactly one final LF and no CR bytes",
        )

        payload = raw[:-1]
        _require(
            payload.startswith(b"{") and payload.endswith(b"}"),
            "capture root must be an object",
        )

        stack: list[int] = []
        in_string = False
        string_bytes = 0
        pairs = {ord("}"): ord("{"), ord("]"): ord("[")}
        for index, byte in enumerate(payload):
            if in_string:
                if byte == ord('"'):
                    in_string = False
                    continue
                _require(byte != ord("\\"), "escape spellings are forbidden")
                _require(
                    0x20 <= byte <= 0x7E,
                    "string contains a non-printable ASCII byte",
                )
                string_bytes += 1
                _require(
                    string_bytes <= MAX_STRING_BYTES,
                    "string exceeds the 128-byte limit",
                )
                continue

            if byte == ord('"'):
                in_string = True
                string_bytes = 0
            elif byte in (ord("{"), ord("[")):
                stack.append(byte)
                _require(
                    len(stack) <= MAX_NESTING_DEPTH,
                    "capture exceeds the nesting-depth limit",
                )
            elif byte in pairs:
                _require(
                    bool(stack) and stack[-1] == pairs[byte],
                    "capture has mismatched structural punctuation",
                )
                stack.pop()
                _require(
                    bool(stack) or index == len(payload) - 1,
                    "capture contains bytes after its root object",
                )
            else:
                _require(byte != ord("\\"), "escape spellings are forbidden")
                _require(
                    0x21 <= byte <= 0x7E,
                    "whitespace and control bytes are forbidden outside strings",
                )

        _require(not in_string, "capture contains an unclosed string")
        _require(not stack, "capture contains unclosed structural punctuation")

    def _peek(self) -> int:
        _require(self.position < len(self.raw) - 1, "unexpected end of capture")
        return self.raw[self.position]

    def _expect_byte(self, expected: int, label: str) -> None:
        actual = self._peek()
        _require(actual == expected, f"expected {label} at byte {self.position}")
        self.position += 1

    def _push(self, kind: str) -> None:
        _require(
            len(self._frames) < MAX_NESTING_DEPTH,
            "capture exceeds the nesting-depth limit",
        )
        self._frames.append(_Frame(kind))

    def _begin_value(self) -> None:
        if not self._frames:
            _require(not self._root_started, "capture contains a second root value")
            self._root_started = True
            return
        frame = self._frames[-1]
        _require(frame.value_pending, f"{frame.kind} value was not requested")
        frame.value_pending = False

    def _finish_container(self) -> None:
        self._frames.pop()
        if not self._frames:
            self._root_complete = True

    def _top(self, kind: str) -> _Frame:
        _require(
            bool(self._frames) and self._frames[-1].kind == kind,
            f"cursor is not inside an {kind}",
        )
        return self._frames[-1]

    def _require_value_delimiter(self, label: str) -> None:
        _require(self.position < len(self.raw) - 1, f"{label} is not closed")
        _require(
            self.raw[self.position] in b",}]",
            f"{label} has a non-canonical token suffix",
        )

    def _parse_string_literal(self, label: str) -> tuple[str, bytes]:
        self._expect_byte(ord('"'), f'an opening quote for {label}')
        start = self.position
        while True:
            _require(self.position < len(self.raw) - 1, f"{label} is unclosed")
            byte = self.raw[self.position]
            if byte == ord('"'):
                value_bytes = self.raw[start : self.position]
                self.position += 1
                return value_bytes.decode("ascii"), value_bytes
            _require(byte != ord("\\"), f"{label} may not contain escapes")
            _require(
                0x20 <= byte <= 0x7E,
                f"{label} contains a non-printable ASCII byte",
            )
            _require(
                self.position - start < MAX_STRING_BYTES,
                f"{label} exceeds the 128-byte limit",
            )
            self.position += 1

    def begin_object(self) -> None:
        self._begin_value()
        self._expect_byte(ord("{"), "an object opening brace")
        self._push("object")

    def read_object_key(self) -> str:
        """Read the next colon-terminated key and enforce strict key order."""

        frame = self._top("object")
        _require(not frame.value_pending, "object key is missing its value")
        if frame.item_count:
            self._expect_byte(ord(","), "a comma between object members")
        key, key_bytes = self._parse_string_literal("object key")
        if frame.last_key is not None:
            _require(
                key_bytes > frame.last_key,
                "object keys must be unique and strictly ASCII-sorted",
            )
        self._expect_byte(ord(":"), "a colon after an object key")
        frame.item_count += 1
        frame.last_key = key_bytes
        frame.value_pending = True
        return key

    def object_key(self, expected: str) -> None:
        """Read one exact key in the caller's closed-schema field order."""

        try:
            expected_bytes = expected.encode("ascii")
        except UnicodeEncodeError:
            _fail("expected object key must be ASCII")
        _require(
            len(expected_bytes) <= MAX_STRING_BYTES,
            "expected object key exceeds the 128-byte limit",
        )
        actual = self.read_object_key()
        _require(actual == expected, f"expected object key {expected!r}")

    def end_object(self) -> None:
        frame = self._top("object")
        _require(not frame.value_pending, "object key is missing its value")
        self._expect_byte(ord("}"), "an object closing brace")
        self._finish_container()

    def begin_array(self) -> None:
        self._begin_value()
        self._expect_byte(ord("["), "an array opening bracket")
        self._push("array")

    def array_item(self) -> None:
        """Advance over the separator before the next array item."""

        frame = self._top("array")
        _require(not frame.value_pending, "previous array item is missing its value")
        if frame.item_count:
            self._expect_byte(ord(","), "a comma between array items")
        _require(self._peek() != ord("]"), "array item is missing")
        frame.item_count += 1
        frame.value_pending = True

    def at_array_end(self) -> bool:
        self._top("array")
        return self._peek() == ord("]")

    def end_array(self) -> None:
        frame = self._top("array")
        _require(not frame.value_pending, "array item is missing its value")
        self._expect_byte(ord("]"), "an array closing bracket")
        self._finish_container()

    def parse_u64(self, label: str) -> int:
        """Parse one canonical base-10 unsigned 64-bit integer token."""

        self._begin_value()
        first = self._peek()
        _require(ord("0") <= first <= ord("9"), f"{label} must be a u64")
        if first == ord("0"):
            self.position += 1
            _require(
                not (
                    self.position < len(self.raw) - 1
                    and ord("0") <= self.raw[self.position] <= ord("9")
                ),
                f"{label} has a leading zero",
            )
            self._require_value_delimiter(label)
            return 0

        value = 0
        while self.position < len(self.raw) - 1:
            byte = self.raw[self.position]
            if not ord("0") <= byte <= ord("9"):
                break
            digit = byte - ord("0")
            _require(
                value <= (MAX_U64 - digit) // 10,
                f"{label} exceeds the u64 range",
            )
            value = value * 10 + digit
            self.position += 1
        self._require_value_delimiter(label)
        return value

    def parse_raw_string(
        self,
        label: str,
        choices: Collection[str] | None = None,
    ) -> str:
        """Parse one bounded, closed, unescaped printable-ASCII string."""

        self._begin_value()
        value, _ = self._parse_string_literal(label)
        self._require_value_delimiter(label)
        if choices is not None:
            _require(
                not isinstance(choices, (str, bytes)),
                f"{label} choices must be a collection of complete strings",
            )
            _require(value in choices, f"{label} has an unsupported spelling")
        return value

    def parse_bool(self, label: str) -> bool:
        self._begin_value()
        for spelling, value in ((b"true", True), (b"false", False)):
            if self.raw.startswith(spelling, self.position):
                self.position += len(spelling)
                self._require_value_delimiter(label)
                return value
        _fail(f"{label} must be a boolean")

    def parse_null(self, label: str) -> None:
        self._begin_value()
        _require(
            self.raw.startswith(b"null", self.position),
            f"{label} must be null",
        )
        self.position += len(b"null")
        self._require_value_delimiter(label)

    def finish(self) -> None:
        """Require that the root value and sole final LF were consumed."""

        _require(not self._frames, "capture ended with an open container")
        _require(
            self._root_started and self._root_complete,
            "capture root object was not completely consumed",
        )
        _require(
            self.position == len(self.raw) - 1,
            "capture has unconsumed bytes after the root object",
        )
        _require(self.raw[self.position] == ord("\n"), "capture lacks final LF")
        self.position += 1
        _require(self.position == len(self.raw), "capture has trailing bytes")


UINT32_MAX = (1 << 32) - 1
MAX_CONTROL_GENERATION = (1 << 61) - 1
REPETITION_COUNT = 32
ACTION_COUNT = 1024
OBSERVATION_LIMIT = 675
MAX_NEW_TOKENS = 16
ACTOR_RACE_HISTORY_SCHEMA = "runnel.actor-race-history/1"

_LOWER_HEX = frozenset("0123456789abcdef")
_ACTION_KINDS = frozenset(
    ("submit", "cancel", "receiver_drop", "drain", "wake")
)
_ACTION_RESULTS = frozenset(
    (
        "submit_accepted",
        "submit_offer_exhausted",
        "cancel_requested",
        "cancel_already_requested",
        "cancel_already_terminal",
        "receiver_dropped",
        "drain_output",
        "drain_empty",
        "drain_eof",
        "wake_signaled",
        "target_unavailable",
        "error",
    )
)
_CONTROL_OPERATIONS = frozenset(("none", "cancel", "disconnect"))
_CONTROL_DISPOSITIONS = frozenset(
    ("requested", "already_requested", "already_terminal")
)
_OVERLAP_BOUNDARIES = frozenset(
    (
        "command",
        "control",
        "primary_endpoint",
        "opportunistic_endpoint",
        "wake",
    )
)
_TERMINAL_OUTCOMES = frozenset(("completed", "cancelled"))

_WORKLOAD_VALUES = (
    (
        "artifact_id",
        "sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c",
    ),
    (
        "artifact_object_sha256",
        "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
    ),
    (
        "artifact_page_table_sha256",
        "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
    ),
    (
        "model_spec_sha256",
        "sha256:ed57d7961e65c76223c169cabebaff9c02d8293da026abb0c0c0a22d38079845",
    ),
    ("specification", "runnel-m5-actor-stress-v1"),
    (
        "vector_file_sha256",
        "sha256:eca1faeee91a41d19d98be7ffdad6fc5cebb9027f3e7a634c01ea1cc394fb574",
    ),
    (
        "vector_id",
        "sha256:5010492fb74eda207511b26811992ed4779814185b9f184663b37a37747bd051",
    ),
    ("vector_schema", "runnel.actor-stress-vectors/2"),
)


@dataclass(frozen=True, slots=True)
class CapturedError:
    code: str
    category: str
    resource: str | None
    required: int | None
    limit: int | None


_STALE_ERROR = CapturedError(
    "request_not_found", "invalid_request", None, None, None
)
_SATURATED_ERROR = CapturedError(
    "resource_exhausted",
    "resource_exhausted",
    "request slot count",
    16,
    16,
)


@dataclass(frozen=True, slots=True)
class Output:
    request_id: int
    output_index: int
    token_id: int


@dataclass(frozen=True, slots=True)
class CommandWitness:
    boundary: bool
    slot: int
    ticket: int
    ready_sequence: int


@dataclass(frozen=True, slots=True)
class AcceptedWitness:
    boundary: bool
    request_id: int
    control_slot: int
    control_generation: int
    endpoint_slot: int
    endpoint_generation: int


@dataclass(frozen=True, slots=True)
class ControlWitness:
    operation: str
    boundary: bool
    slot: int
    expected_generation: int
    loaded_word: str
    resulting_word: str
    disposition: str | None


@dataclass(frozen=True, slots=True)
class PopWitness:
    kind: str
    boundary: bool
    slot: int
    generation: int
    drained_before: int
    drained_after: int
    output: Output | None


@dataclass(frozen=True, slots=True)
class WakeWitness:
    boundary: bool
    dirty_was_set: bool
    before_park_epoch: int
    after_park_epoch: int


@dataclass(frozen=True, slots=True)
class Action:
    ordinal: int
    producer: int
    kind: str
    submit_attempt: int | None
    client_index: int | None
    invocation: int
    response: int
    result: str
    error: CapturedError | None
    request_id: int | None
    output: Output | None
    command: CommandWitness
    accepted: AcceptedWitness
    control: ControlWitness
    primary_pop: PopWitness
    opportunistic_eof_pop: PopWitness
    cached_eof: bool
    wake: WakeWitness


@dataclass(frozen=True, slots=True)
class Terminal:
    request_id: int
    outcome: str
    committed_positions: int
    emitted_tokens: int


@dataclass(frozen=True, slots=True)
class ProbeSnapshot:
    command_in_flight: int
    command_ready: int
    command_reserved: int
    command_responded: int
    dirty: bool
    engine_steps: int
    outstanding_requests: int
    owner_done: bool
    park_epoch: int
    parked: bool
    pump_entries: int
    pump_hold_observed: int
    pump_hold_released: int
    pump_hold_requested: int
    pump_in_flight: bool
    request_bytes: int
    shared_bytes: int


@dataclass(frozen=True, slots=True)
class CleanupAuthority:
    invocation: int
    response: int
    client_index: int
    request_id: int
    error: CapturedError | None
    control: ControlWitness


@dataclass(frozen=True, slots=True)
class CleanupReceiver:
    invocation: int
    response: int
    client_index: int
    request_id: int
    terminal: Terminal
    outputs: tuple[Output, ...]
    eof_acknowledged: bool
    post_drop_quiescent: ProbeSnapshot


@dataclass(frozen=True, slots=True)
class Observation:
    kind: str
    request_id: int
    output_index: int | None
    token_id: int | None
    outcome: str | None
    committed_positions: int | None
    emitted_tokens: int | None


@dataclass(frozen=True, slots=True)
class RecorderStatus:
    allocated_capacity: int
    observation_count: int
    observation_limit: int
    overflowed: bool
    poisoned: bool


@dataclass(frozen=True, slots=True)
class Diagnostics:
    action_counter_final: int
    barrier_released: bool
    cleanup_counter_final: int
    engine_steps_delta: int
    final_engine_steps: int
    final_pump_entries: int
    initial_engine_steps: int
    initial_pump_entries: int
    overlap_pair: tuple[int, str, int, str]
    pump_entries_delta: int
    recorder_final: RecorderStatus
    recorder_initial: RecorderStatus


@dataclass(frozen=True, slots=True)
class Shutdown:
    accepted_submissions: int
    discarded_output_events: int
    engine_steps: int
    rejected_submissions: int
    released_request_bytes: int
    remaining_shared_bytes: int
    shutdown_cancellations: int
    terminated_requests: int


@dataclass(frozen=True, slots=True)
class Repetition:
    actions: tuple[Action, ...]
    cleanup_authorities: tuple[CleanupAuthority, ...]
    cleanup_receivers: tuple[CleanupReceiver, ...]
    diagnostics: Diagnostics
    observations: tuple[Observation, ...]
    post_shutdown: ProbeSnapshot
    pre_cleanup: ProbeSnapshot
    pre_shutdown: ProbeSnapshot
    repetition: int
    shutdown: Shutdown


@dataclass(frozen=True, slots=True)
class Workload:
    artifact_id: str
    artifact_object_sha256: str
    artifact_page_table_sha256: str
    model_spec_sha256: str
    specification: str
    vector_file_sha256: str
    vector_id: str
    vector_schema: str


@dataclass(frozen=True, slots=True)
class Capture:
    repetition_count: int
    repetitions: tuple[Repetition, ...]
    schema: str
    workload: Workload


def _bounded(value: int, minimum: int, maximum: int, label: str) -> int:
    _require(minimum <= value <= maximum, f"{label} is outside its allowed range")
    return value


def _nonzero(value: int, label: str) -> int:
    _require(value != 0, f"{label} must be nonzero")
    return value


def _parse_optional_u64(cursor: CanonicalCursor, label: str) -> int | None:
    if cursor._peek() == ord("n"):
        cursor.parse_null(label)
        return None
    return cursor.parse_u64(label)


def _parse_optional_string(
    cursor: CanonicalCursor,
    label: str,
    choices: Collection[str],
) -> str | None:
    if cursor._peek() == ord("n"):
        cursor.parse_null(label)
        return None
    return cursor.parse_raw_string(label, choices)


def _parse_hex64(cursor: CanonicalCursor, label: str) -> str:
    value = cursor.parse_raw_string(label)
    _require(
        len(value) == 16 and all(byte in _LOWER_HEX for byte in value),
        f"{label} must be exactly 16 lowercase hexadecimal digits",
    )
    return value


def _parse_sha256(cursor: CanonicalCursor, label: str) -> str:
    value = cursor.parse_raw_string(label)
    digest = value.removeprefix("sha256:")
    _require(
        value.startswith("sha256:")
        and len(digest) == 64
        and all(byte in _LOWER_HEX for byte in digest),
        f"{label} must be a canonical sha256 string",
    )
    return value


def _parse_error(cursor: CanonicalCursor, label: str) -> CapturedError:
    cursor.begin_array()
    cursor.array_item()
    code = cursor.parse_raw_string(f"{label} code", ("request_not_found", "resource_exhausted"))
    cursor.array_item()
    category = cursor.parse_raw_string(
        f"{label} category", ("invalid_request", "resource_exhausted")
    )
    cursor.array_item()
    resource = _parse_optional_string(cursor, f"{label} resource", ("request slot count",))
    cursor.array_item()
    required = _parse_optional_u64(cursor, f"{label} required")
    cursor.array_item()
    limit = _parse_optional_u64(cursor, f"{label} limit")
    cursor.end_array()
    error = CapturedError(code, category, resource, required, limit)
    _require(
        error
        in (
            _STALE_ERROR,
            _SATURATED_ERROR,
        ),
        f"{label} is not one of the two closed error spellings",
    )
    return error


def _parse_optional_error(
    cursor: CanonicalCursor, label: str
) -> CapturedError | None:
    if cursor._peek() == ord("n"):
        cursor.parse_null(label)
        return None
    return _parse_error(cursor, label)


def _parse_output(cursor: CanonicalCursor, label: str) -> Output:
    cursor.begin_array()
    cursor.array_item()
    request_id = _nonzero(cursor.parse_u64(f"{label} request_id"), f"{label} request_id")
    cursor.array_item()
    output_index = cursor.parse_u64(f"{label} output_index")
    cursor.array_item()
    token_id = _bounded(cursor.parse_u64(f"{label} token_id"), 0, UINT32_MAX, f"{label} token_id")
    cursor.end_array()
    return Output(request_id, output_index, token_id)


def _parse_optional_output(cursor: CanonicalCursor, label: str) -> Output | None:
    if cursor._peek() == ord("n"):
        cursor.parse_null(label)
        return None
    return _parse_output(cursor, label)


def _parse_command_witness(
    cursor: CanonicalCursor, label: str
) -> CommandWitness:
    cursor.begin_array()
    cursor.array_item()
    boundary = cursor.parse_bool(f"{label} boundary")
    cursor.array_item()
    slot = cursor.parse_u64(f"{label} slot")
    cursor.array_item()
    ticket = cursor.parse_u64(f"{label} ticket")
    cursor.array_item()
    ready_sequence = cursor.parse_u64(f"{label} ready_sequence")
    cursor.end_array()
    if boundary:
        _bounded(slot, 0, 7, f"{label} slot")
        _nonzero(ticket, f"{label} ticket")
        _nonzero(ready_sequence, f"{label} ready_sequence")
    else:
        _require(
            (slot, ticket, ready_sequence) == (0, 0, 0),
            f"{label} must use the command sentinel",
        )
    return CommandWitness(boundary, slot, ticket, ready_sequence)


def _parse_accepted_witness(
    cursor: CanonicalCursor, label: str
) -> AcceptedWitness:
    cursor.begin_array()
    cursor.array_item()
    boundary = cursor.parse_bool(f"{label} boundary")
    cursor.array_item()
    request_id = cursor.parse_u64(f"{label} request_id")
    cursor.array_item()
    control_slot = cursor.parse_u64(f"{label} control_slot")
    cursor.array_item()
    control_generation = cursor.parse_u64(f"{label} control_generation")
    cursor.array_item()
    endpoint_slot = cursor.parse_u64(f"{label} endpoint_slot")
    cursor.array_item()
    endpoint_generation = cursor.parse_u64(f"{label} endpoint_generation")
    cursor.end_array()
    if boundary:
        _nonzero(request_id, f"{label} request_id")
        _bounded(control_slot, 0, 15, f"{label} control_slot")
        _bounded(
            control_generation,
            1,
            MAX_CONTROL_GENERATION,
            f"{label} control_generation",
        )
        _bounded(endpoint_slot, 0, 15, f"{label} endpoint_slot")
        _nonzero(endpoint_generation, f"{label} endpoint_generation")
    else:
        _require(
            (request_id, control_slot, control_generation, endpoint_slot, endpoint_generation)
            == (0, 0, 0, 0, 0),
            f"{label} must use the accepted sentinel",
        )
    return AcceptedWitness(
        boundary,
        request_id,
        control_slot,
        control_generation,
        endpoint_slot,
        endpoint_generation,
    )


def _parse_control_witness(
    cursor: CanonicalCursor, label: str
) -> ControlWitness:
    cursor.begin_array()
    cursor.array_item()
    operation = cursor.parse_raw_string(f"{label} operation", _CONTROL_OPERATIONS)
    cursor.array_item()
    boundary = cursor.parse_bool(f"{label} boundary")
    cursor.array_item()
    slot = cursor.parse_u64(f"{label} slot")
    cursor.array_item()
    expected_generation = cursor.parse_u64(f"{label} expected_generation")
    cursor.array_item()
    loaded_word = _parse_hex64(cursor, f"{label} loaded_word")
    cursor.array_item()
    resulting_word = _parse_hex64(cursor, f"{label} resulting_word")
    cursor.array_item()
    disposition = _parse_optional_string(
        cursor, f"{label} disposition", _CONTROL_DISPOSITIONS
    )
    cursor.end_array()
    if operation == "none":
        _require(
            not boundary
            and slot == 0
            and expected_generation == 0
            and loaded_word == "0000000000000000"
            and resulting_word == "0000000000000000"
            and disposition is None,
            f"{label} must use the control sentinel",
        )
    else:
        _require(boundary, f"{label} reached control must set its boundary")
        _bounded(slot, 0, 15, f"{label} slot")
        _bounded(
            expected_generation,
            1,
            MAX_CONTROL_GENERATION,
            f"{label} expected_generation",
        )
        for word, word_label in (
            (loaded_word, "loaded_word"),
            (resulting_word, "resulting_word"),
        ):
            _bounded(
                int(word, 16) >> 3,
                1,
                MAX_CONTROL_GENERATION,
                f"{label} {word_label} generation",
            )
    return ControlWitness(
        operation,
        boundary,
        slot,
        expected_generation,
        loaded_word,
        resulting_word,
        disposition,
    )


def _parse_pop_witness(
    cursor: CanonicalCursor, label: str, expected_kind: str
) -> PopWitness:
    cursor.begin_array()
    cursor.array_item()
    kind = cursor.parse_raw_string(f"{label} kind", (expected_kind,))
    cursor.array_item()
    boundary = cursor.parse_bool(f"{label} boundary")
    cursor.array_item()
    slot = cursor.parse_u64(f"{label} slot")
    cursor.array_item()
    generation = cursor.parse_u64(f"{label} generation")
    cursor.array_item()
    drained_before = cursor.parse_u64(f"{label} drained_before")
    cursor.array_item()
    drained_after = cursor.parse_u64(f"{label} drained_after")
    cursor.array_item()
    output = _parse_optional_output(cursor, f"{label} output")
    cursor.end_array()
    if boundary:
        _bounded(slot, 0, 15, f"{label} slot")
        _nonzero(generation, f"{label} generation")
        if output is None:
            _require(
                drained_after == drained_before,
                f"{label} without output must not advance the drain count",
            )
        else:
            _require(
                drained_after == drained_before + 1,
                f"{label} output must advance the drain count once",
            )
    else:
        _require(
            slot == 0
            and generation == 0
            and drained_before == 0
            and drained_after == 0
            and output is None,
            f"{label} must use its exact pop sentinel",
        )
    if expected_kind == "opportunistic_eof":
        _require(output is None, f"{label} may not carry output")
    return PopWitness(
        kind,
        boundary,
        slot,
        generation,
        drained_before,
        drained_after,
        output,
    )


def _parse_wake_witness(cursor: CanonicalCursor, label: str) -> WakeWitness:
    cursor.begin_array()
    cursor.array_item()
    boundary = cursor.parse_bool(f"{label} boundary")
    cursor.array_item()
    dirty_was_set = cursor.parse_bool(f"{label} dirty_was_set")
    cursor.array_item()
    before_park_epoch = cursor.parse_u64(f"{label} before_park_epoch")
    cursor.array_item()
    after_park_epoch = cursor.parse_u64(f"{label} after_park_epoch")
    cursor.end_array()
    if not boundary:
        _require(
            not dirty_was_set
            and before_park_epoch == 0
            and after_park_epoch == 0,
            f"{label} must use the wake sentinel",
        )
    else:
        _require(
            after_park_epoch > before_park_epoch,
            f"{label} acknowledgement must advance the park epoch",
        )
    return WakeWitness(
        boundary, dirty_was_set, before_park_epoch, after_park_epoch
    )


def _parse_action(
    cursor: CanonicalCursor, expected_ordinal: int
) -> Action:
    label = f"action {expected_ordinal}"
    cursor.begin_array()
    cursor.array_item()
    ordinal = _bounded(cursor.parse_u64(f"{label} ordinal"), 0, 1023, f"{label} ordinal")
    _require(ordinal == expected_ordinal, f"{label} is out of ordinal order")
    cursor.array_item()
    producer = _bounded(cursor.parse_u64(f"{label} producer"), 0, 1, f"{label} producer")
    cursor.array_item()
    kind = cursor.parse_raw_string(f"{label} kind", _ACTION_KINDS)
    cursor.array_item()
    submit_attempt = _parse_optional_u64(cursor, f"{label} submit_attempt")
    if submit_attempt is not None:
        _bounded(submit_attempt, 0, 205, f"{label} submit_attempt")
    cursor.array_item()
    client_index = _parse_optional_u64(cursor, f"{label} client_index")
    if client_index is not None:
        _bounded(client_index, 0, 63, f"{label} client_index")
    cursor.array_item()
    invocation = _bounded(cursor.parse_u64(f"{label} invocation"), 1, 2047, f"{label} invocation")
    cursor.array_item()
    response = _bounded(cursor.parse_u64(f"{label} response"), 2, 2048, f"{label} response")
    _require(response > invocation, f"{label} response must follow invocation")
    cursor.array_item()
    result = cursor.parse_raw_string(f"{label} result", _ACTION_RESULTS)
    cursor.array_item()
    error = _parse_optional_error(cursor, f"{label} error")
    _require((error is not None) == (result == "error"), f"{label} error/result mismatch")
    cursor.array_item()
    request_id = _parse_optional_u64(cursor, f"{label} request_id")
    if request_id is not None:
        _nonzero(request_id, f"{label} request_id")
    cursor.array_item()
    output = _parse_optional_output(cursor, f"{label} output")
    _require((output is not None) == (result == "drain_output"), f"{label} output/result mismatch")
    cursor.array_item()
    command = _parse_command_witness(cursor, f"{label} command")
    cursor.array_item()
    accepted = _parse_accepted_witness(cursor, f"{label} accepted")
    cursor.array_item()
    control = _parse_control_witness(cursor, f"{label} control")
    cursor.array_item()
    primary_pop = _parse_pop_witness(cursor, f"{label} primary_pop", "primary")
    cursor.array_item()
    opportunistic_eof_pop = _parse_pop_witness(
        cursor, f"{label} opportunistic_eof_pop", "opportunistic_eof"
    )
    cursor.array_item()
    cached_eof = cursor.parse_bool(f"{label} cached_eof")
    cursor.array_item()
    wake = _parse_wake_witness(cursor, f"{label} wake")
    cursor.end_array()

    _require(
        (submit_attempt is not None) == (kind == "submit"),
        f"{label} submit_attempt applicability mismatch",
    )
    if kind == "submit":
        expected_client = (
            submit_attempt
            if submit_attempt is not None and submit_attempt < 64
            else None
        )
        _require(client_index == expected_client, f"{label} submit client_index mismatch")
        _require(
            command.boundary
            == (submit_attempt is not None and submit_attempt < 64),
            f"{label} command sentinel mismatch",
        )
    else:
        _require(client_index is not None, f"{label} targeted action requires client_index")
        _require(not command.boundary, f"{label} command witness is not applicable")
    _require(
        accepted.boundary == (result == "submit_accepted"),
        f"{label} accepted sentinel mismatch",
    )
    if accepted.boundary:
        _require(request_id == accepted.request_id, f"{label} accepted request ID mismatch")
    if kind not in ("cancel", "receiver_drop"):
        _require(control.operation == "none", f"{label} control witness is not applicable")
    elif control.operation != "none":
        expected_operation = "cancel" if kind == "cancel" else "disconnect"
        _require(control.operation == expected_operation, f"{label} control operation mismatch")
    if kind != "drain":
        _require(
            not primary_pop.boundary and not opportunistic_eof_pop.boundary,
            f"{label} pop witness is not applicable",
        )
    if output is not None:
        _require(
            output.request_id == request_id,
            f"{label} output request ID mismatch",
        )
        _require(primary_pop.output == output, f"{label} primary output witness mismatch")
    if opportunistic_eof_pop.boundary:
        _require(primary_pop.boundary, f"{label} opportunistic pop requires a primary pop")
        _require(
            opportunistic_eof_pop.slot == primary_pop.slot
            and opportunistic_eof_pop.generation == primary_pop.generation
            and opportunistic_eof_pop.drained_before == primary_pop.drained_after,
            f"{label} opportunistic pop endpoint chain mismatch",
        )
    if cached_eof:
        _require(
            result == "drain_eof"
            and not primary_pop.boundary
            and not opportunistic_eof_pop.boundary,
            f"{label} cached EOF sentinel mismatch",
        )
    _require(wake.boundary == (kind == "wake"), f"{label} wake sentinel mismatch")

    if kind == "submit":
        _require(
            control.operation == "none"
            and not primary_pop.boundary
            and not opportunistic_eof_pop.boundary
            and not cached_eof
            and not wake.boundary,
            f"{label} submit retained a nonapplicable witness",
        )
        if submit_attempt is not None and submit_attempt < 64:
            _require(
                result in ("submit_accepted", "error"),
                f"{label} in-range submit result is invalid",
            )
            if result == "submit_accepted":
                _require(
                    request_id is not None and error is None,
                    f"{label} accepted submit identity is invalid",
                )
            else:
                _require(
                    request_id is None and error == _SATURATED_ERROR,
                    f"{label} rejected submit spelling is invalid",
                )
        else:
            _require(
                result == "submit_offer_exhausted"
                and request_id is None
                and error is None,
                f"{label} exhausted submit spelling is invalid",
            )
    elif kind == "cancel":
        _require(
            result
            in (
                "cancel_requested",
                "cancel_already_requested",
                "cancel_already_terminal",
                "target_unavailable",
                "error",
            ),
            f"{label} cancel result is invalid",
        )
        if result == "target_unavailable":
            _require(
                request_id is None and control.operation == "none",
                f"{label} unavailable cancel retained target evidence",
            )
        else:
            dispositions = {
                "cancel_requested": "requested",
                "cancel_already_requested": "already_requested",
                "cancel_already_terminal": "already_terminal",
                "error": None,
            }
            _require(
                request_id is not None
                and control.operation == "cancel"
                and control.boundary
                and control.disposition == dispositions[result],
                f"{label} cancel control evidence is invalid",
            )
            expected_error = _STALE_ERROR if result == "error" else None
            _require(error == expected_error, f"{label} cancel error spelling is invalid")
        _require(
            not primary_pop.boundary
            and not opportunistic_eof_pop.boundary
            and not cached_eof
            and not wake.boundary,
            f"{label} cancel retained endpoint or wake evidence",
        )
    elif kind == "receiver_drop":
        _require(
            result in ("receiver_dropped", "target_unavailable"),
            f"{label} receiver-drop result is invalid",
        )
        if result == "target_unavailable":
            _require(
                control.operation == "none",
                f"{label} unavailable receiver drop retained control evidence",
            )
        else:
            _require(
                request_id is not None
                and control.operation == "disconnect"
                and control.boundary
                and control.disposition is not None,
                f"{label} receiver-drop control evidence is invalid",
            )
        _require(
            not primary_pop.boundary
            and not opportunistic_eof_pop.boundary
            and not cached_eof
            and not wake.boundary,
            f"{label} receiver drop retained endpoint or wake evidence",
        )
    elif kind == "drain":
        _require(
            result in ("drain_output", "drain_empty", "drain_eof", "target_unavailable"),
            f"{label} drain result is invalid",
        )
        _require(
            control.operation == "none" and not wake.boundary,
            f"{label} drain retained control or wake evidence",
        )
        if result == "target_unavailable":
            _require(
                not primary_pop.boundary
                and not opportunistic_eof_pop.boundary
                and not cached_eof,
                f"{label} unavailable drain retained endpoint evidence",
            )
        elif result == "drain_output":
            _require(
                request_id is not None
                and primary_pop.boundary
                and primary_pop.output == output
                and not cached_eof,
                f"{label} output drain evidence is invalid",
            )
        elif result == "drain_empty":
            _require(
                request_id is not None
                and primary_pop.boundary
                and primary_pop.output is None
                and not opportunistic_eof_pop.boundary
                and not cached_eof,
                f"{label} empty drain evidence is invalid",
            )
        else:
            direct = (
                request_id is not None
                and primary_pop.boundary
                and primary_pop.output is None
                and not opportunistic_eof_pop.boundary
                and not cached_eof
            )
            cached = (
                request_id is not None
                and not primary_pop.boundary
                and not opportunistic_eof_pop.boundary
                and cached_eof
            )
            _require(direct or cached, f"{label} EOF drain evidence is invalid")
    else:
        _require(
            result == "wake_signaled"
            and request_id is None
            and error is None
            and control.operation == "none"
            and not primary_pop.boundary
            and not opportunistic_eof_pop.boundary
            and not cached_eof
            and wake.boundary,
            f"{label} wake evidence is invalid",
        )
    return Action(
        ordinal,
        producer,
        kind,
        submit_attempt,
        client_index,
        invocation,
        response,
        result,
        error,
        request_id,
        output,
        command,
        accepted,
        control,
        primary_pop,
        opportunistic_eof_pop,
        cached_eof,
        wake,
    )


def _parse_terminal(cursor: CanonicalCursor, label: str) -> Terminal:
    cursor.begin_array()
    cursor.array_item()
    request_id = _nonzero(cursor.parse_u64(f"{label} request_id"), f"{label} request_id")
    cursor.array_item()
    outcome = cursor.parse_raw_string(f"{label} outcome", _TERMINAL_OUTCOMES)
    cursor.array_item()
    committed_positions = cursor.parse_u64(f"{label} committed_positions")
    cursor.array_item()
    emitted_tokens = cursor.parse_u64(f"{label} emitted_tokens")
    cursor.end_array()
    return Terminal(request_id, outcome, committed_positions, emitted_tokens)


def _parse_probe_snapshot(
    cursor: CanonicalCursor, label: str
) -> ProbeSnapshot:
    cursor.begin_object()
    values: list[int | bool] = []
    for key in (
        "command_in_flight",
        "command_ready",
        "command_reserved",
        "command_responded",
    ):
        cursor.object_key(key)
        values.append(cursor.parse_u64(f"{label} {key}"))
    cursor.object_key("dirty")
    dirty = cursor.parse_bool(f"{label} dirty")
    cursor.object_key("engine_steps")
    engine_steps = cursor.parse_u64(f"{label} engine_steps")
    cursor.object_key("outstanding_requests")
    outstanding_requests = cursor.parse_u64(f"{label} outstanding_requests")
    cursor.object_key("owner_done")
    owner_done = cursor.parse_bool(f"{label} owner_done")
    cursor.object_key("park_epoch")
    park_epoch = cursor.parse_u64(f"{label} park_epoch")
    cursor.object_key("parked")
    parked = cursor.parse_bool(f"{label} parked")
    cursor.object_key("pump_entries")
    pump_entries = cursor.parse_u64(f"{label} pump_entries")
    hold_values: list[int] = []
    for key in ("pump_hold_observed", "pump_hold_released", "pump_hold_requested"):
        cursor.object_key(key)
        hold_values.append(cursor.parse_u64(f"{label} {key}"))
    cursor.object_key("pump_in_flight")
    pump_in_flight = cursor.parse_bool(f"{label} pump_in_flight")
    cursor.object_key("request_bytes")
    request_bytes = cursor.parse_u64(f"{label} request_bytes")
    cursor.object_key("shared_bytes")
    shared_bytes = cursor.parse_u64(f"{label} shared_bytes")
    cursor.end_object()
    return ProbeSnapshot(
        values[0],
        values[1],
        values[2],
        values[3],
        dirty,
        engine_steps,
        outstanding_requests,
        owner_done,
        park_epoch,
        parked,
        pump_entries,
        hold_values[0],
        hold_values[1],
        hold_values[2],
        pump_in_flight,
        request_bytes,
        shared_bytes,
    )


def _require_quiescent(snapshot: ProbeSnapshot, label: str) -> None:
    _require(snapshot.parked, f"{label} must be parked")
    _require(not snapshot.dirty, f"{label} must not be dirty")
    _require(not snapshot.pump_in_flight, f"{label} pump must not be in flight")
    _require(not snapshot.owner_done, f"{label} owner must still be running")
    _require(
        snapshot.command_in_flight == 0
        and snapshot.command_ready == 0
        and snapshot.command_reserved == 0
        and snapshot.command_responded == 0,
        f"{label} must have zero command occupancy",
    )
    _require(
        snapshot.pump_hold_requested
        == snapshot.pump_hold_observed
        == snapshot.pump_hold_released,
        f"{label} must have no incomplete pump hold",
    )


def _require_request_ledger_presence(snapshot: ProbeSnapshot, label: str) -> None:
    _require(
        (snapshot.outstanding_requests == 0) == (snapshot.request_bytes == 0),
        f"{label} request count and ledger presence disagree",
    )


def _require_not_before(
    snapshot: ProbeSnapshot,
    previous: ProbeSnapshot,
    label: str,
) -> None:
    _require(
        snapshot.park_epoch >= previous.park_epoch
        and snapshot.pump_entries >= previous.pump_entries
        and snapshot.engine_steps >= previous.engine_steps
        and snapshot.pump_hold_requested >= previous.pump_hold_requested
        and snapshot.pump_hold_observed >= previous.pump_hold_observed
        and snapshot.pump_hold_released >= previous.pump_hold_released,
        f"{label} regressed a monotone actor counter",
    )


def _require_strictly_ascending(values: Collection[int], label: str) -> None:
    previous: int | None = None
    for value in values:
        if previous is not None:
            _require(value > previous, f"{label} must be strictly ascending")
        previous = value


def _require_pre_cleanup_not_before_initial(
    pre_cleanup: ProbeSnapshot,
    initial_engine_steps: int,
    initial_pump_entries: int,
    label: str,
) -> None:
    _require(
        pre_cleanup.engine_steps >= initial_engine_steps
        and pre_cleanup.pump_entries >= initial_pump_entries,
        f"{label} pre_cleanup predates the initial diagnostic endpoints",
    )


def _parse_cleanup_authority(
    cursor: CanonicalCursor, ordinal: int
) -> CleanupAuthority:
    label = f"cleanup authority {ordinal}"
    cursor.begin_array()
    cursor.array_item()
    invocation = cursor.parse_u64(f"{label} invocation")
    _require(
        invocation == ordinal * 2 + 1,
        f"{label} invocation is outside the exact cleanup counter sequence",
    )
    cursor.array_item()
    response = cursor.parse_u64(f"{label} response")
    _require(response == invocation + 1, f"{label} response must equal invocation plus one")
    cursor.array_item()
    client_index = _bounded(
        cursor.parse_u64(f"{label} client_index"),
        0,
        63,
        f"{label} client_index",
    )
    cursor.array_item()
    request_id = _nonzero(cursor.parse_u64(f"{label} request_id"), f"{label} request_id")
    cursor.array_item()
    error = _parse_optional_error(cursor, f"{label} error")
    cursor.array_item()
    control = _parse_control_witness(cursor, f"{label} control")
    cursor.end_array()
    _require(
        control.operation == "cancel" and control.boundary,
        f"{label} requires reached cancel control",
    )
    if error is None:
        _require(
            control.disposition is not None,
            f"{label} successful control requires disposition",
        )
    else:
        _require(
            error.code == "request_not_found" and control.disposition is None,
            f"{label} stale control spelling mismatch",
        )
    return CleanupAuthority(invocation, response, client_index, request_id, error, control)


def _parse_outputs(cursor: CanonicalCursor, label: str) -> tuple[Output, ...]:
    cursor.begin_array()
    outputs: list[Output] = []
    while not cursor.at_array_end():
        _require(
            len(outputs) < MAX_NEW_TOKENS,
            f"{label} exceeds the frozen generation ceiling",
        )
        cursor.array_item()
        outputs.append(_parse_output(cursor, f"{label} {len(outputs)}"))
    cursor.end_array()
    return tuple(outputs)


def _parse_cleanup_receiver(
    cursor: CanonicalCursor, sequence_ordinal: int
) -> CleanupReceiver:
    label = f"cleanup receiver {sequence_ordinal}"
    cursor.begin_array()
    cursor.array_item()
    invocation = cursor.parse_u64(f"{label} invocation")
    _require(
        invocation == sequence_ordinal * 2 + 1,
        f"{label} invocation is outside the exact cleanup counter sequence",
    )
    cursor.array_item()
    response = cursor.parse_u64(f"{label} response")
    _require(response == invocation + 1, f"{label} response must equal invocation plus one")
    cursor.array_item()
    client_index = _bounded(
        cursor.parse_u64(f"{label} client_index"),
        0,
        63,
        f"{label} client_index",
    )
    cursor.array_item()
    request_id = _nonzero(cursor.parse_u64(f"{label} request_id"), f"{label} request_id")
    cursor.array_item()
    terminal = _parse_terminal(cursor, f"{label} terminal")
    cursor.array_item()
    outputs = _parse_outputs(cursor, f"{label} outputs")
    cursor.array_item()
    eof_acknowledged = cursor.parse_bool(f"{label} eof_acknowledged")
    cursor.array_item()
    post_drop_quiescent = _parse_probe_snapshot(cursor, f"{label} post_drop_quiescent")
    cursor.end_array()
    _require(eof_acknowledged, f"{label} must acknowledge EOF")
    _require(terminal.request_id == request_id, f"{label} terminal request ID mismatch")
    _require(
        all(output.request_id == request_id for output in outputs),
        f"{label} output request ID mismatch",
    )
    _require(
        all(output.output_index < terminal.emitted_tokens for output in outputs),
        f"{label} output index exceeds terminal emissions",
    )
    for previous, current in zip(outputs, outputs[1:]):
        _require(
            current.output_index == previous.output_index + 1,
            f"{label} output suffix is not consecutive",
        )
    if outputs:
        _require(
            outputs[-1].output_index + 1 == terminal.emitted_tokens,
            f"{label} output suffix does not end at terminal emissions",
        )
    _require_quiescent(post_drop_quiescent, f"{label} post_drop_quiescent")
    return CleanupReceiver(
        invocation,
        response,
        client_index,
        request_id,
        terminal,
        outputs,
        eof_acknowledged,
        post_drop_quiescent,
    )


def _parse_observation(cursor: CanonicalCursor, ordinal: int) -> Observation:
    label = f"observation {ordinal}"
    cursor.begin_array()
    cursor.array_item()
    kind = cursor.parse_raw_string(label, ("output", "terminal", "output_eof"))
    cursor.array_item()
    request_id = _nonzero(cursor.parse_u64(f"{label} request_id"), f"{label} request_id")
    cursor.array_item()
    output_index = _parse_optional_u64(cursor, f"{label} output_index")
    cursor.array_item()
    token_id = _parse_optional_u64(cursor, f"{label} token_id")
    if token_id is not None:
        _bounded(token_id, 0, UINT32_MAX, f"{label} token_id")
    cursor.array_item()
    outcome = _parse_optional_string(cursor, f"{label} outcome", _TERMINAL_OUTCOMES)
    cursor.array_item()
    committed_positions = _parse_optional_u64(cursor, f"{label} committed_positions")
    cursor.array_item()
    emitted_tokens = _parse_optional_u64(cursor, f"{label} emitted_tokens")
    cursor.end_array()
    if kind == "output":
        _require(
            output_index is not None
            and token_id is not None
            and outcome is None
            and committed_positions is None
            and emitted_tokens is None,
            f"{label} output sentinels are invalid",
        )
    elif kind == "terminal":
        _require(
            output_index is None
            and token_id is None
            and outcome is not None
            and committed_positions is not None
            and emitted_tokens is not None,
            f"{label} terminal sentinels are invalid",
        )
    else:
        _require(
            output_index is None
            and token_id is None
            and outcome is None
            and committed_positions is None
            and emitted_tokens is None,
            f"{label} EOF sentinels are invalid",
        )
    return Observation(
        kind,
        request_id,
        output_index,
        token_id,
        outcome,
        committed_positions,
        emitted_tokens,
    )


def _parse_recorder_status(
    cursor: CanonicalCursor, label: str
) -> RecorderStatus:
    cursor.begin_object()
    cursor.object_key("allocated_capacity")
    allocated_capacity = cursor.parse_u64(f"{label} allocated_capacity")
    cursor.object_key("observation_count")
    observation_count = cursor.parse_u64(f"{label} observation_count")
    cursor.object_key("observation_limit")
    observation_limit = cursor.parse_u64(f"{label} observation_limit")
    _require(observation_limit == OBSERVATION_LIMIT, f"{label} observation_limit must be 675")
    cursor.object_key("overflowed")
    overflowed = cursor.parse_bool(f"{label} overflowed")
    cursor.object_key("poisoned")
    poisoned = cursor.parse_bool(f"{label} poisoned")
    cursor.end_object()
    return RecorderStatus(
        allocated_capacity,
        observation_count,
        observation_limit,
        overflowed,
        poisoned,
    )


def _validate_observation_order(
    observations: Collection[Observation], label: str
) -> None:
    phase = 0
    previous_output: tuple[int, int] | None = None
    previous_terminal: int | None = None
    previous_eof: int | None = None
    for observation in observations:
        if observation.kind == "output":
            _require(phase == 0, f"{label} output appears after a later phase")
            _require(
                observation.output_index is not None,
                f"{label} output omitted its index",
            )
            key = (observation.request_id, observation.output_index)
            if previous_output is not None:
                _require(key > previous_output, f"{label} outputs are not strictly sorted")
            previous_output = key
        elif observation.kind == "terminal":
            _require(phase <= 1, f"{label} terminal appears after the EOF phase")
            phase = 1
            if previous_terminal is not None:
                _require(
                    observation.request_id > previous_terminal,
                    f"{label} terminals are not strictly sorted",
                )
            previous_terminal = observation.request_id
        else:
            phase = 2
            if previous_eof is not None:
                _require(
                    observation.request_id > previous_eof,
                    f"{label} EOF observations are not strictly sorted",
                )
            previous_eof = observation.request_id


def _parse_overlap_pair(
    cursor: CanonicalCursor, label: str
) -> tuple[int, str, int, str]:
    cursor.begin_array()
    cursor.array_item()
    left = _bounded(cursor.parse_u64(f"{label} left ordinal"), 0, 1023, f"{label} left ordinal")
    cursor.array_item()
    left_boundary = cursor.parse_raw_string(f"{label} left boundary", _OVERLAP_BOUNDARIES)
    cursor.array_item()
    right = _bounded(cursor.parse_u64(f"{label} right ordinal"), 0, 1023, f"{label} right ordinal")
    cursor.array_item()
    right_boundary = cursor.parse_raw_string(f"{label} right boundary", _OVERLAP_BOUNDARIES)
    cursor.end_array()
    return left, left_boundary, right, right_boundary


def _parse_diagnostics(cursor: CanonicalCursor, label: str) -> Diagnostics:
    cursor.begin_object()
    cursor.object_key("action_counter_final")
    action_counter_final = cursor.parse_u64(f"{label} action_counter_final")
    _require(action_counter_final == 2048, f"{label} action_counter_final must be 2048")
    cursor.object_key("barrier_released")
    barrier_released = cursor.parse_bool(f"{label} barrier_released")
    _require(barrier_released, f"{label} barrier must be released")
    cursor.object_key("cleanup_counter_final")
    cleanup_counter_final = cursor.parse_u64(f"{label} cleanup_counter_final")
    cursor.object_key("engine_steps_delta")
    engine_steps_delta = cursor.parse_u64(f"{label} engine_steps_delta")
    cursor.object_key("final_engine_steps")
    final_engine_steps = cursor.parse_u64(f"{label} final_engine_steps")
    cursor.object_key("final_pump_entries")
    final_pump_entries = cursor.parse_u64(f"{label} final_pump_entries")
    cursor.object_key("initial_engine_steps")
    initial_engine_steps = cursor.parse_u64(f"{label} initial_engine_steps")
    cursor.object_key("initial_pump_entries")
    initial_pump_entries = cursor.parse_u64(f"{label} initial_pump_entries")
    cursor.object_key("overlap_pair")
    overlap_pair = _parse_overlap_pair(cursor, f"{label} overlap_pair")
    cursor.object_key("pump_entries_delta")
    pump_entries_delta = cursor.parse_u64(f"{label} pump_entries_delta")
    cursor.object_key("recorder_final")
    recorder_final = _parse_recorder_status(cursor, f"{label} recorder_final")
    cursor.object_key("recorder_initial")
    recorder_initial = _parse_recorder_status(cursor, f"{label} recorder_initial")
    cursor.end_object()
    _require(final_engine_steps >= initial_engine_steps, f"{label} engine endpoints regress")
    _require(final_pump_entries >= initial_pump_entries, f"{label} pump endpoints regress")
    _require(
        engine_steps_delta == final_engine_steps - initial_engine_steps,
        f"{label} engine delta mismatch",
    )
    _require(
        pump_entries_delta == final_pump_entries - initial_pump_entries,
        f"{label} pump delta mismatch",
    )
    _require(pump_entries_delta <= 4096, f"{label} pump delta exceeds 4096")
    _require(
        not recorder_initial.overflowed
        and not recorder_initial.poisoned
        and not recorder_final.overflowed
        and not recorder_final.poisoned,
        f"{label} recorder is unhealthy",
    )
    _require(
        recorder_initial.allocated_capacity == recorder_final.allocated_capacity,
        f"{label} recorder capacity changed",
    )
    _require(
        recorder_initial.allocated_capacity >= OBSERVATION_LIMIT,
        f"{label} recorder capacity is below its logical limit",
    )
    _require(
        recorder_initial.observation_count == 0,
        f"{label} initial recorder is not empty",
    )
    return Diagnostics(
        action_counter_final,
        barrier_released,
        cleanup_counter_final,
        engine_steps_delta,
        final_engine_steps,
        final_pump_entries,
        initial_engine_steps,
        initial_pump_entries,
        overlap_pair,
        pump_entries_delta,
        recorder_final,
        recorder_initial,
    )


def _parse_shutdown(cursor: CanonicalCursor, label: str) -> Shutdown:
    cursor.begin_object()
    values: list[int] = []
    for key in (
        "accepted_submissions",
        "discarded_output_events",
        "engine_steps",
        "rejected_submissions",
        "released_request_bytes",
        "remaining_shared_bytes",
        "shutdown_cancellations",
        "terminated_requests",
    ):
        cursor.object_key(key)
        values.append(cursor.parse_u64(f"{label} {key}"))
    cursor.end_object()
    shutdown = Shutdown(*values)
    _require(
        shutdown.accepted_submissions + shutdown.rejected_submissions == 64,
        f"{label} acceptance and rejection counts must sum to 64",
    )
    _require(
        shutdown.discarded_output_events == 0
        and shutdown.released_request_bytes == 0
        and shutdown.remaining_shared_bytes == 0
        and shutdown.shutdown_cancellations == 0
        and shutdown.terminated_requests == 0,
        f"{label} shutdown effects must all be zero",
    )
    return shutdown


def _parse_repetition(
    cursor: CanonicalCursor, expected_repetition: int
) -> Repetition:
    label = f"repetition {expected_repetition}"
    cursor.begin_object()
    cursor.object_key("actions")
    cursor.begin_array()
    actions: list[Action] = []
    for ordinal in range(ACTION_COUNT):
        cursor.array_item()
        actions.append(_parse_action(cursor, ordinal))
    cursor.end_array()
    cursor.object_key("cleanup_authorities")
    cursor.begin_array()
    cleanup_authorities: list[CleanupAuthority] = []
    while not cursor.at_array_end():
        _require(len(cleanup_authorities) < 64, f"{label} has too many cleanup authorities")
        cursor.array_item()
        cleanup_authorities.append(_parse_cleanup_authority(cursor, len(cleanup_authorities)))
    cursor.end_array()
    _require_strictly_ascending(
        tuple(record.client_index for record in cleanup_authorities),
        f"{label} cleanup authority client indexes",
    )
    cursor.object_key("cleanup_receivers")
    cursor.begin_array()
    cleanup_receivers: list[CleanupReceiver] = []
    while not cursor.at_array_end():
        _require(len(cleanup_receivers) < 64, f"{label} has too many cleanup receivers")
        cursor.array_item()
        cleanup_receivers.append(
            _parse_cleanup_receiver(
                cursor,
                len(cleanup_authorities) + len(cleanup_receivers),
            )
        )
    cursor.end_array()
    _require_strictly_ascending(
        tuple(record.client_index for record in cleanup_receivers),
        f"{label} cleanup receiver client indexes",
    )
    cursor.object_key("diagnostics")
    diagnostics = _parse_diagnostics(cursor, f"{label} diagnostics")
    cursor.object_key("observations")
    cursor.begin_array()
    observations: list[Observation] = []
    while not cursor.at_array_end():
        _require(len(observations) < OBSERVATION_LIMIT, f"{label} has too many observations")
        cursor.array_item()
        observations.append(_parse_observation(cursor, len(observations)))
    cursor.end_array()
    _validate_observation_order(observations, f"{label} observations")
    cursor.object_key("post_shutdown")
    post_shutdown = _parse_probe_snapshot(cursor, f"{label} post_shutdown")
    cursor.object_key("pre_cleanup")
    pre_cleanup = _parse_probe_snapshot(cursor, f"{label} pre_cleanup")
    cursor.object_key("pre_shutdown")
    pre_shutdown = _parse_probe_snapshot(cursor, f"{label} pre_shutdown")
    cursor.object_key("repetition")
    repetition = _bounded(cursor.parse_u64(f"{label} index"), 0, 31, f"{label} index")
    _require(repetition == expected_repetition, f"{label} index is out of order")
    cursor.object_key("shutdown")
    shutdown = _parse_shutdown(cursor, f"{label} shutdown")
    cursor.end_object()

    _require(
        diagnostics.cleanup_counter_final
        == 2 * (len(cleanup_authorities) + len(cleanup_receivers)),
        f"{label} cleanup counter does not match cleanup arrays",
    )
    _require(
        diagnostics.recorder_final.observation_count == len(observations),
        f"{label} recorder count does not match observations",
    )
    _require_quiescent(pre_cleanup, f"{label} pre_cleanup")
    _require_quiescent(pre_shutdown, f"{label} pre_shutdown")
    _require_pre_cleanup_not_before_initial(
        pre_cleanup,
        diagnostics.initial_engine_steps,
        diagnostics.initial_pump_entries,
        label,
    )
    _require_request_ledger_presence(pre_cleanup, f"{label} pre_cleanup")
    _require_request_ledger_presence(pre_shutdown, f"{label} pre_shutdown")
    _require(
        pre_cleanup.outstanding_requests == len(cleanup_receivers),
        f"{label} pre_cleanup request count differs from receiver custody",
    )
    _require(
        pre_cleanup.pump_hold_requested < MAX_U64,
        f"{label} cleanup pump-hold epoch overflowed",
    )
    expected_hold_epoch = pre_cleanup.pump_hold_requested + 1
    preceding_snapshot = pre_cleanup
    for receiver_ordinal, receiver in enumerate(cleanup_receivers):
        snapshot = receiver.post_drop_quiescent
        snapshot_label = f"{label} cleanup receiver {receiver_ordinal} snapshot"
        _require_request_ledger_presence(snapshot, snapshot_label)
        _require_not_before(snapshot, preceding_snapshot, snapshot_label)
        _require(
            snapshot.request_bytes <= preceding_snapshot.request_bytes,
            f"{snapshot_label} increased request ledger bytes",
        )
        _require(
            snapshot.pump_hold_requested
            == snapshot.pump_hold_observed
            == snapshot.pump_hold_released
            == expected_hold_epoch,
            f"{snapshot_label} does not conserve the cleanup pump hold",
        )
        _require(
            snapshot.shared_bytes == pre_cleanup.shared_bytes,
            f"{snapshot_label} changed static shared ledger bytes",
        )
        _require(
            snapshot.outstanding_requests
            == len(cleanup_receivers) - receiver_ordinal - 1,
            f"{snapshot_label} did not consume exactly one receiver",
        )
        preceding_snapshot = snapshot
    _require_not_before(pre_shutdown, preceding_snapshot, f"{label} pre_shutdown")
    _require(
        pre_shutdown.request_bytes <= preceding_snapshot.request_bytes,
        f"{label} pre_shutdown increased request ledger bytes",
    )
    _require(
        pre_shutdown.pump_hold_requested
        == pre_shutdown.pump_hold_observed
        == pre_shutdown.pump_hold_released
        == expected_hold_epoch,
        f"{label} pre_shutdown does not conserve the cleanup pump hold",
    )
    _require(
        pre_shutdown.shared_bytes == pre_cleanup.shared_bytes,
        f"{label} pre_shutdown changed static shared ledger bytes",
    )
    _require(
        pre_shutdown.outstanding_requests == 0
        and pre_shutdown.request_bytes == 0,
        f"{label} pre_shutdown retained request-owned state",
    )
    _require(
        post_shutdown.owner_done
        and post_shutdown.dirty
        and not post_shutdown.parked
        and not post_shutdown.pump_in_flight,
        f"{label} post_shutdown state flags are invalid",
    )
    _require(
        post_shutdown.command_in_flight == 0
        and post_shutdown.command_ready == 0
        and post_shutdown.command_reserved == 0
        and post_shutdown.command_responded == 0,
        f"{label} post_shutdown retained command occupancy",
    )
    _require(
        post_shutdown.outstanding_requests == 0
        and post_shutdown.request_bytes == 0
        and post_shutdown.shared_bytes == 0,
        f"{label} post_shutdown retained actor-owned state",
    )
    _require(
        post_shutdown.pump_hold_requested
        == post_shutdown.pump_hold_observed
        == post_shutdown.pump_hold_released,
        f"{label} post_shutdown retained an incomplete pump hold",
    )
    _require_not_before(post_shutdown, pre_shutdown, f"{label} post_shutdown")
    _require(
        post_shutdown.engine_steps == diagnostics.final_engine_steps
        and post_shutdown.pump_entries == diagnostics.final_pump_entries
        and post_shutdown.engine_steps == shutdown.engine_steps,
        f"{label} final diagnostic endpoints mismatch",
    )
    return Repetition(
        tuple(actions),
        tuple(cleanup_authorities),
        tuple(cleanup_receivers),
        diagnostics,
        tuple(observations),
        post_shutdown,
        pre_cleanup,
        pre_shutdown,
        repetition,
        shutdown,
    )


def _parse_workload(cursor: CanonicalCursor) -> Workload:
    cursor.begin_object()
    values: list[str] = []
    for key, expected in _WORKLOAD_VALUES:
        cursor.object_key(key)
        if expected.startswith("sha256:"):
            value = _parse_sha256(cursor, f"workload {key}")
        else:
            value = cursor.parse_raw_string(f"workload {key}")
        _require(value == expected, f"workload {key} is not the authenticated constant")
        values.append(value)
    cursor.end_object()
    return Workload(*values)


def decode_capture(raw: bytes) -> Capture:
    """Strictly decode the exact closed ``runnel.actor-race-history/1`` schema."""

    cursor = CanonicalCursor(raw)
    cursor.begin_object()
    cursor.object_key("repetition_count")
    repetition_count = cursor.parse_u64("repetition_count")
    _require(repetition_count == REPETITION_COUNT, "repetition_count must be 32")
    cursor.object_key("repetitions")
    cursor.begin_array()
    repetitions: list[Repetition] = []
    for repetition in range(REPETITION_COUNT):
        cursor.array_item()
        repetitions.append(_parse_repetition(cursor, repetition))
    cursor.end_array()
    cursor.object_key("schema")
    schema = cursor.parse_raw_string("schema", (ACTOR_RACE_HISTORY_SCHEMA,))
    cursor.object_key("workload")
    workload = _parse_workload(cursor)
    cursor.end_object()
    cursor.finish()
    return Capture(repetition_count, tuple(repetitions), schema, workload)


def load_capture(path: str | Path) -> Capture:
    """Safely read and strictly decode one race-history capture file."""

    return decode_capture(read_capture_bytes(path))
