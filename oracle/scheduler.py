#!/usr/bin/env python3
"""Independent stdlib generator for the M5 bounded-actor stress inputs.

This module freezes only deterministic request descriptors and coordinator
actions.  It does not model scheduler outcomes, inspect Rust implementation
details, or provide a golden terminal digest.  SHA-256 byte domains, integer
encodings, rejection sampling, and producer assignment are explicit so an
independent implementation can reproduce the same inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from typing import Any, Iterable, Iterator, NoReturn, Sequence


SCHEMA = "runnel.actor-stress-vectors/1"
SPECIFICATION = "runnel-m5-actor-stress-v1"
ACTION_DOMAIN = b"runnel-m5-actor-stress-v1\0"
REQUEST_DOMAIN = b"runnel-m5-actor-request-v1\0"
WORD_STREAM_ALGORITHM = "sha256-domain-u64le-counter-four-u64le-byte-order-v1"
UNBIASED_ALGORITHM = "reject-ge-floor-2^64-over-bound-times-bound-v1"
DESCRIPTOR_ALGORITHM = "sha256-domain-u32le-index-byte-formulas-v1"
ACTION_ALGORITHM = "unbiased-kind-then-selector-submit-cursor-v1"
PRODUCER_ALGORITHM = "submit-attempt-home-drop-drain-home-cancel-opposite-wake-ordinal-v1"
SEQUENCE_DIGEST_ALGORITHM = "sha256-canonical-json-ascii-v1"

REQUEST_COUNT = 64
ACTION_COUNT = 1_024
KIND_BOUND = 5
SELECTOR_BOUND = 64
WORDS_PER_DIGEST = 4
PIN_COUNT = 4
UINT64_RANGE = 1 << 64
MAX_U64 = UINT64_RANGE - 1
MAX_U32 = (1 << 32) - 1
MAX_FIXTURE_BYTES = 256 * 1024

ACTION_KINDS = ("submit", "cancel", "drop", "drain", "wake")
ACTOR_CONFIG = {
    "adapter": "tiny-v3",
    "admission_reserve_bytes": 1_048_576,
    "backend": "scalar",
    "batch_width": 8,
    "command_capacity": 8,
    "logical_memory_limit_bytes": 8_388_608,
    "max_active_requests": 8,
    "max_context_tokens": 19,
    "max_new_tokens": 16,
    "max_outstanding_requests": 16,
    "max_prompt_tokens": 4,
    "max_queued_requests": 16,
    "max_retained_terminal_results": 16,
    "output_capacity_per_request": 2,
    "page_pool_partition_bytes": 0,
    "state_page_tokens": 4,
    "trace_capacity": 1_024,
    "waves_per_step": 4,
    "worker_count": 1,
}
DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "scheduler"
    / "actor-stress-v1.json"
)


class SchedulerFixtureError(ValueError):
    """A stable generation or fixture-custody failure."""


def _fail(message: str) -> NoReturn:
    raise SchedulerFixtureError(message)


def _require_plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _u64(value: Any, label: str) -> int:
    value = _require_plain_int(value, label)
    if value < 0 or value > MAX_U64:
        _fail(f"{label} is outside the unsigned 64-bit range")
    return value


def _u32(value: Any, label: str) -> int:
    value = _require_plain_int(value, label)
    if value < 0 or value > MAX_U32:
        _fail(f"{label} is outside the unsigned 32-bit range")
    return value


def canonical_bytes(value: Any) -> bytes:
    """Encode one canonical ASCII JSON value with a final LF."""

    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as error:
        _fail(f"value cannot be encoded as canonical JSON: {error}")
    return (rendered + "\n").encode("ascii")


def sequence_identity(values: Sequence[dict[str, Any]]) -> str:
    """Return the frozen SHA-256 identity of a canonical sequence."""

    return f"sha256:{hashlib.sha256(canonical_bytes(values)).hexdigest()}"


def word_block(counter: int) -> tuple[int, int, int, int]:
    """Return digest *counter* as four consecutive little-endian u64 words."""

    counter = _u64(counter, "word-stream counter")
    digest = hashlib.sha256(ACTION_DOMAIN + struct.pack("<Q", counter)).digest()
    return struct.unpack("<QQQQ", digest)


def word_stream(start_counter: int = 0) -> Iterator[int]:
    """Yield SHA-256 word blocks in counter order and in digest byte order."""

    counter = _u64(start_counter, "word-stream start counter")
    while True:
        yield from word_block(counter)
        if counter == MAX_U64:
            _fail("word-stream counter is exhausted")
        counter += 1


def unbiased_value(word: int, bound: int) -> int | None:
    """Map one u64 without bias, returning ``None`` for the rejection tail."""

    word = _u64(word, "unbiased word")
    bound = _require_plain_int(bound, "unbiased bound")
    if bound < 1 or bound > UINT64_RANGE:
        _fail("unbiased bound must be in [1, 2^64]")
    limit = (UINT64_RANGE // bound) * bound
    if word >= limit:
        return None
    return word % bound


def take_unbiased(words: Iterator[int], bound: int) -> int:
    """Consume words until rejection sampling accepts one for *bound*."""

    for word in words:
        value = unbiased_value(word, bound)
        if value is not None:
            return value
    _fail("word stream ended before an unbiased value was available")


def request_descriptor(index: int) -> dict[str, Any]:
    """Generate one request descriptor from its domain-separated digest."""

    index = _u32(index, "request descriptor index")
    digest = hashlib.sha256(REQUEST_DOMAIN + struct.pack("<I", index)).digest()
    prompt_length = 1 + digest[0] % 4
    prompt = [1 + digest[1 + offset] % 31 for offset in range(prompt_length)]
    max_new_tokens = 1 + digest[5] % (17 - prompt_length)
    return {
        "deadline_ns": None,
        "index": index,
        "max_new_tokens": max_new_tokens,
        "prompt": prompt,
        "sampling": "greedy",
    }


def build_descriptors() -> list[dict[str, Any]]:
    """Build all 64 deterministic request descriptors."""

    return [request_descriptor(index) for index in range(REQUEST_COUNT)]


def _targeted_action(
    ordinal: int, kind: str, request_index: int
) -> dict[str, Any]:
    home = request_index % 2
    producer = 1 - home if kind == "cancel" else home
    return {
        "kind": kind,
        "ordinal": ordinal,
        "producer": producer,
        "request_index": request_index,
    }


def build_actions(
    count: int = ACTION_COUNT, *, words: Iterable[int] | None = None
) -> list[dict[str, Any]]:
    """Build the actor action script with exact stream-consumption semantics.

    Submit-kind attempts consume only the kind word.  Attempts 0 through 63
    name the corresponding descriptor; later attempts are explicit exhausted
    no-ops.  Their producer remains ``submit_attempt % 2`` and the attempt
    ordinal continues to advance.  Cancel, drop, drain, and wake each consume
    the next unbiased bound-64 selector.  Wake records that selector for input
    custody even though the global wake operation ignores request state.
    """

    count = _require_plain_int(count, "action count")
    if count < 0:
        _fail("action count must be nonnegative")
    source = iter(word_stream() if words is None else words)
    actions: list[dict[str, Any]] = []
    submit_attempt = 0
    for ordinal in range(count):
        kind = ACTION_KINDS[take_unbiased(source, KIND_BOUND)]
        if kind == "submit":
            exhausted = submit_attempt >= REQUEST_COUNT
            actions.append(
                {
                    "exhausted": exhausted,
                    "kind": kind,
                    "ordinal": ordinal,
                    "producer": submit_attempt % 2,
                    "request_index": None if exhausted else submit_attempt,
                    "submit_attempt": submit_attempt,
                }
            )
            submit_attempt += 1
            continue

        selector = take_unbiased(source, SELECTOR_BOUND)
        if kind == "wake":
            actions.append(
                {
                    "kind": kind,
                    "ordinal": ordinal,
                    "producer": ordinal % 2,
                    "selector_index": selector,
                }
            )
        else:
            actions.append(_targeted_action(ordinal, kind, selector))
    return actions


def _fixture_without_identity(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "fixture_id"}


def fixture_identity(document: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_bytes(_fixture_without_identity(document))).hexdigest()
    return f"sha256:{digest}"


def build_fixture() -> dict[str, Any]:
    """Build the closed custody fixture for descriptors and scripted actions."""

    descriptors = build_descriptors()
    actions = build_actions()
    by_kind = Counter(action["kind"] for action in actions)
    by_producer = Counter(action["producer"] for action in actions)
    exhausted_submits = sum(
        action["kind"] == "submit" and action["exhausted"] for action in actions
    )
    document: dict[str, Any] = {
        "actor_config": dict(ACTOR_CONFIG),
        "action_counts": {
            "by_kind": {kind: by_kind[kind] for kind in ACTION_KINDS},
            "exhausted_submits": exhausted_submits,
            "total": len(actions),
        },
        "action_vectors": {
            "count": len(actions),
            "digest": sequence_identity(actions),
            "first": actions[:PIN_COUNT],
            "last": actions[-PIN_COUNT:],
        },
        "algorithms": {
            "action_generation": ACTION_ALGORITHM,
            "descriptor_generation": DESCRIPTOR_ALGORITHM,
            "producer_assignment": PRODUCER_ALGORITHM,
            "sequence_digest": SEQUENCE_DIGEST_ALGORITHM,
            "unbiased_mapping": UNBIASED_ALGORITHM,
            "word_stream": WORD_STREAM_ALGORITHM,
        },
        "descriptor_vectors": {
            "count": len(descriptors),
            "digest": sequence_identity(descriptors),
            "first": descriptors[:PIN_COUNT],
            "last": descriptors[-PIN_COUNT:],
        },
        "domains": {
            "action_words": ACTION_DOMAIN.decode("ascii"),
            "request_descriptors": REQUEST_DOMAIN.decode("ascii"),
        },
        "kind_codes": {str(code): kind for code, kind in enumerate(ACTION_KINDS)},
        "parameters": {
            "action_count": ACTION_COUNT,
            "kind_bound": KIND_BOUND,
            "pin_count": PIN_COUNT,
            "request_count": REQUEST_COUNT,
            "selector_bound": SELECTOR_BOUND,
            "words_per_digest": WORDS_PER_DIGEST,
        },
        "producer_counts": {
            "producer_0": by_producer[0],
            "producer_1": by_producer[1],
        },
        "schema": SCHEMA,
        "specification": SPECIFICATION,
    }
    document["fixture_id"] = fixture_identity(document)
    return document


def generated_bytes() -> bytes:
    return canonical_bytes(build_fixture())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _expect_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        _fail(f"{label} does not match the closed schema")
    return value


def validate_document(document: Any) -> dict[str, Any]:
    """Validate closed identities, summaries, pins, and fresh regeneration."""

    document = _expect_keys(
        document,
        {
            "actor_config",
            "action_counts",
            "action_vectors",
            "algorithms",
            "descriptor_vectors",
            "domains",
            "fixture_id",
            "kind_codes",
            "parameters",
            "producer_counts",
            "schema",
            "specification",
        },
        "actor stress fixture",
    )
    if document["schema"] != SCHEMA:
        _fail("actor stress fixture schema is unsupported")
    if document["specification"] != SPECIFICATION:
        _fail("actor stress specification identity differs")
    if document["fixture_id"] != fixture_identity(document):
        _fail("actor stress fixture identity digest differs")

    _expect_keys(document["actor_config"], set(ACTOR_CONFIG), "actor config")
    _expect_keys(document["domains"], {"action_words", "request_descriptors"}, "domains")
    _expect_keys(
        document["algorithms"],
        {
            "action_generation",
            "descriptor_generation",
            "producer_assignment",
            "sequence_digest",
            "unbiased_mapping",
            "word_stream",
        },
        "algorithms",
    )
    _expect_keys(
        document["parameters"],
        {
            "action_count",
            "kind_bound",
            "pin_count",
            "request_count",
            "selector_bound",
            "words_per_digest",
        },
        "parameters",
    )
    _expect_keys(
        document["descriptor_vectors"],
        {"count", "digest", "first", "last"},
        "descriptor vectors",
    )
    _expect_keys(
        document["action_vectors"],
        {"count", "digest", "first", "last"},
        "action vectors",
    )
    action_counts = _expect_keys(
        document["action_counts"],
        {"by_kind", "exhausted_submits", "total"},
        "action counts",
    )
    _expect_keys(action_counts["by_kind"], set(ACTION_KINDS), "action kind counts")
    _expect_keys(
        document["producer_counts"],
        {"producer_0", "producer_1"},
        "producer counts",
    )
    _expect_keys(document["kind_codes"], {str(index) for index in range(KIND_BOUND)}, "kind codes")

    expected = build_fixture()
    # Compare canonical bytes, not Python container equality: ``bool`` is an
    # ``int`` subclass and would otherwise let JSON ``true`` equal integer 1.
    if canonical_bytes(document) != canonical_bytes(expected):
        _fail("actor stress fixture differs from independent regeneration")
    return document


def parse_document_bytes(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_FIXTURE_BYTES:
        _fail("actor stress fixture is empty or exceeds its byte limit")
    if not raw.endswith(b"\n") or b"\r" in raw or not raw.isascii():
        _fail("actor stress fixture must be canonical ASCII ending in LF")
    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: _fail(f"invalid JSON constant {value!r}"),
        )
    except SchedulerFixtureError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        _fail(f"actor stress fixture JSON is invalid: {error}")
    if canonical_bytes(document) != raw:
        _fail("actor stress fixture JSON is not in canonical formatting")
    return validate_document(document)


def _read_fixture(path: Path) -> bytes:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(f"cannot open actor stress fixture without following links: {error}")
    try:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect opened actor stress fixture: {error}")
        if not stat.S_ISREG(before.st_mode):
            _fail("actor stress fixture path must name a regular file")
        if before.st_size > MAX_FIXTURE_BYTES:
            _fail("actor stress fixture exceeds its byte limit")

        chunks: list[bytes] = []
        bytes_read = 0
        try:
            while bytes_read <= MAX_FIXTURE_BYTES:
                remaining = MAX_FIXTURE_BYTES + 1 - bytes_read
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
        except OSError as error:
            _fail(f"cannot read actor stress fixture: {error}")
        raw = b"".join(chunks)
        if len(raw) > MAX_FIXTURE_BYTES:
            _fail("actor stress fixture exceeds its byte limit")
        try:
            after = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect actor stress fixture after reading: {error}")
        if (
            len(raw) != after.st_size
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            _fail("actor stress fixture changed while it was being read")
        return raw
    finally:
        os.close(descriptor)


def check_path(path: Path) -> str:
    raw = _read_fixture(path)
    parse_document_bytes(raw)
    if raw != generated_bytes():
        _fail("actor stress fixture differs from independent regeneration")
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
    action.add_argument("--check", action="store_true", help="verify the committed fixture")
    action.add_argument("--write", action="store_true", help="rewrite the committed fixture")
    parser.add_argument("--path", type=Path, default=DEFAULT_FIXTURE_PATH)
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
    except (OSError, SchedulerFixtureError) as error:
        print(f"scheduler oracle: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
