#!/usr/bin/env python3
"""Independent verifier and serializer for the M5 actor semantic transcript.

The Rust harness supplies a bounded canonical *logical* capture containing
enum spellings and raw request IDs.  This module never accepts, parses, or
compares Rust-produced transcript bytes.  It independently regenerates the
frozen action stream, validates the capture's relational semantics, derives
client indices and numeric codes, and emits the binary layout frozen by ADR
0007.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, NoReturn

from oracle import scheduler


SCHEMA = "runnel.actor-semantic-capture/1"
TRANSCRIPT_DOMAIN = b"runnel-m5-actor-semantic-transcript-v2\0"
MAX_CAPTURE_BYTES = 1024 * 1024
MAX_TINY_SPEC_BYTES = 256 * 1024
EXPECTED_DIGEST_BYTES = 72
EXPECTED_DESCRIPTOR_VECTOR_ID = (
    "sha256:d902ecf3377310de99471f41287f62730671263b8e339ac03b87ee5d6edef42b"
)
MAX_U8 = (1 << 8) - 1
MAX_U32 = (1 << 32) - 1
MAX_U64 = (1 << 64) - 1
ACTION_COUNT = scheduler.ACTION_COUNT
REQUEST_COUNT = scheduler.REQUEST_COUNT
MAX_OUTSTANDING_REQUESTS = scheduler.ACTOR_CONFIG["max_outstanding_requests"]
OUTPUT_CAPACITY_PER_REQUEST = scheduler.ACTOR_CONFIG["output_capacity_per_request"]
OUTPUT_OBSERVATION_LIMIT = sum(
    descriptor["max_new_tokens"] for descriptor in scheduler.build_descriptors()
)
OBSERVATION_LIMIT = OUTPUT_OBSERVATION_LIMIT + 2 * REQUEST_COUNT
PUMP_ENTRY_LIMIT = 4_096
VOCABULARY_SIZE = 32
EOS_TOKEN_ID = 0
MAX_TRANSCRIPT_BYTES = (
    len(TRANSCRIPT_DOMAIN)
    + 16
    + ACTION_COUNT * 33
    + OUTPUT_OBSERVATION_LIMIT * 21
    + REQUEST_COUNT * 25
    + REQUEST_COUNT * 13
    + 57
)
TINY_V3_SPEC_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "tiny-v3" / "spec.json"
)

ROOT_FIELDS = {
    "schema",
    "workload",
    "action_results",
    "observations",
    "cleanup_cancellations",
    "shutdown",
    "diagnostics",
}
WORKLOAD_FIELDS = {
    "fixture_schema",
    "specification",
    "fixture_id",
    "fixture_file_sha256",
}
ACTION_FIELDS = {
    "ordinal",
    "kind",
    "producer",
    "submit_attempt",
    "client_index",
    "result",
    "error",
    "request_id",
    "output",
}
OUTPUT_FIELDS = {"output_index", "token_id"}
CLEANUP_CANCELLATION_FIELDS = {"client_index", "disposition"}
CLEANUP_DISPOSITIONS = {
    "requested",
    "already_requested",
    "already_terminal",
    "invalid_request",
}
SHUTDOWN_FIELDS = {
    "accepted_submissions",
    "rejected_submissions",
    "shutdown_cancellations",
    "terminated_requests",
    "discarded_output_events",
    "released_request_bytes",
    "remaining_shared_bytes",
    "final_request_bytes",
    "final_shared_bytes",
}
DIAGNOSTIC_FIELDS = {
    "initial_pump_entries",
    "final_pump_entries",
    "pump_entries_delta",
    "engine_steps",
    "observer_count",
    "observer_limit",
    "observer_initial_capacity",
    "observer_final_capacity",
    "observer_overflowed",
    "observer_poisoned",
}

KIND_CODES = {
    "submit": 0,
    "cancel": 1,
    "drop": 2,
    "drain": 3,
    "wake": 4,
}
RESULT_CODES = {
    "submit_accepted": 0,
    "submit_offer_exhausted": 1,
    "cancel_requested": 2,
    "cancel_already_requested": 3,
    "cancel_already_terminal": 4,
    "receiver_dropped": 5,
    "drain_output": 6,
    "drain_empty": 7,
    "drain_eof": 8,
    "wake_signaled": 9,
    "target_unavailable": 10,
    "error": 11,
}
ERROR_CODES = {
    "invalid_request": 1,
    "unsupported": 2,
    "resource_exhausted": 3,
    "cancelled": 4,
    "deadline_exceeded": 5,
    "internal": 6,
}
OUTCOME_CODES = {
    "completed": 0,
    "cancelled": 1,
    "deadline_exceeded": 2,
    "failed": 3,
}


class ActorTranscriptError(ValueError):
    """A closed-schema, semantic, custody, or serialization failure."""


def _fail(message: str) -> NoReturn:
    raise ActorTranscriptError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _unsigned(value: Any, maximum: int, label: str) -> int:
    value = _plain_int(value, label)
    if value < 0 or value > maximum:
        _fail(f"{label} is outside its unsigned integer range")
    return value


def _u8(value: Any, label: str) -> int:
    return _unsigned(value, MAX_U8, label)


def _u32(value: Any, label: str) -> int:
    return _unsigned(value, MAX_U32, label)


def _u64(value: Any, label: str) -> int:
    return _unsigned(value, MAX_U64, label)


def _optional_u32(value: Any, label: str) -> int | None:
    return None if value is None else _u32(value, label)


def _optional_u64(value: Any, label: str) -> int | None:
    if value is None:
        return None
    value = _u64(value, label)
    if value == 0:
        _fail(f"{label} must be nonzero when present")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be a boolean")
    return value


def _enum(value: Any, choices: dict[str, int], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail(f"{label} has an unsupported enum spelling")
    return value


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{label} does not match the closed schema")
    return value


def _array(
    value: Any, label: str, *, exact: int | None = None, maximum: int | None = None
) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    if exact is not None and len(value) != exact:
        _fail(f"{label} must contain exactly {exact} entries")
    if maximum is not None and len(value) > maximum:
        _fail(f"{label} exceeds its {maximum}-entry bound")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def canonical_bytes(value: Any) -> bytes:
    """Encode one value as sorted, pretty canonical ASCII JSON plus LF."""

    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as error:
        _fail(f"capture cannot be encoded as canonical JSON: {error}")
    return (rendered + "\n").encode("ascii")


@dataclass(frozen=True)
class ActionRecord:
    ordinal: int
    kind: str
    producer: int
    result: str
    error: str | None
    submit_attempt: int | None
    client_index: int | None
    request_id: int | None
    output_index: int | None
    token_id: int | None


@dataclass(frozen=True)
class OutputRecord:
    client_index: int
    request_id: int
    output_index: int
    token_id: int


@dataclass(frozen=True)
class TerminalRecord:
    client_index: int
    request_id: int
    outcome: str
    error: str | None
    committed_positions: int
    emitted_tokens: int


@dataclass(frozen=True)
class EofRecord:
    client_index: int
    request_id: int


@dataclass(frozen=True)
class ShutdownRecord:
    accepted_submissions: int
    rejected_submissions: int
    shutdown_cancellations: int
    terminated_requests: int
    discarded_output_events: int
    released_request_bytes: int
    remaining_shared_bytes: int
    final_request_bytes: int
    final_shared_bytes: int


_VALIDATED_CAPTURE_SEAL = object()


@dataclass(frozen=True, init=False)
class ValidatedCapture:
    """Opaque validated records ready for independent binary serialization."""

    actions: tuple[ActionRecord, ...]
    outputs: tuple[OutputRecord, ...]
    terminals: tuple[TerminalRecord, ...]
    eofs: tuple[EofRecord, ...]
    shutdown: ShutdownRecord
    _seal: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        _fail("ValidatedCapture values can only be created by capture validation")


def _make_validated_capture(
    *,
    actions: tuple[ActionRecord, ...],
    outputs: tuple[OutputRecord, ...],
    terminals: tuple[TerminalRecord, ...],
    eofs: tuple[EofRecord, ...],
    shutdown: ShutdownRecord,
) -> ValidatedCapture:
    capture = object.__new__(ValidatedCapture)
    object.__setattr__(capture, "actions", actions)
    object.__setattr__(capture, "outputs", outputs)
    object.__setattr__(capture, "terminals", terminals)
    object.__setattr__(capture, "eofs", eofs)
    object.__setattr__(capture, "shutdown", shutdown)
    object.__setattr__(capture, "_seal", _VALIDATED_CAPTURE_SEAL)
    return capture


def _validated(capture: ValidatedCapture | Any) -> ValidatedCapture:
    if not isinstance(capture, ValidatedCapture):
        return validate_capture(capture)
    _require(
        getattr(capture, "_seal", None) is _VALIDATED_CAPTURE_SEAL,
        "ValidatedCapture value was not created by capture validation",
    )
    return capture


def _expected_workload() -> dict[str, str]:
    try:
        fixture_file_digest = scheduler.check_path(scheduler.DEFAULT_FIXTURE_PATH)
    except scheduler.SchedulerFixtureError as error:
        _fail(f"committed actor workload failed independent custody checks: {error}")
    fixture = scheduler.build_fixture()
    return {
        "fixture_schema": scheduler.SCHEMA,
        "specification": scheduler.SPECIFICATION,
        "fixture_id": fixture["fixture_id"],
        "fixture_file_sha256": f"sha256:{fixture_file_digest}",
    }


def _expected_action_fields(action: dict[str, Any]) -> tuple[int | None, int | None]:
    if action["kind"] == "submit":
        return action["submit_attempt"], action["request_index"]
    if action["kind"] == "wake":
        return None, action["selector_index"]
    return None, action["request_index"]


def _parse_action(
    raw: Any,
    expected: dict[str, Any],
    accepted_by_client: dict[int, int],
    clients_by_id: dict[int, int],
    live_receivers: set[int],
    cancellation_requested: set[int],
    terminal_authorities: set[int],
    natural_terminal_proofs: set[int],
    stale_authorities: set[int],
    free_control_slots: list[int],
    control_slot_by_client: dict[int, int],
    control_owner_by_slot: dict[int, int],
    next_request_id: int,
    drained_counts: dict[int, int],
    drain_eof_seen: set[int],
    drain_eof_counts: list[tuple[int, int]],
) -> tuple[ActionRecord, int, bool]:
    ordinal = expected["ordinal"]
    label = f"action result {ordinal}"
    record = _object(raw, ACTION_FIELDS, label)
    captured_ordinal = _u32(record["ordinal"], f"{label} ordinal")
    _require(captured_ordinal == ordinal, f"{label} ordinal differs from the frozen action")
    kind = _enum(record["kind"], KIND_CODES, f"{label} kind")
    _require(kind == expected["kind"], f"{label} kind differs from the frozen action")
    producer = _u8(record["producer"], f"{label} producer")
    _require(
        producer == expected["producer"],
        f"{label} producer differs from the frozen assignment",
    )
    expected_attempt, expected_client = _expected_action_fields(expected)
    submit_attempt = _optional_u32(record["submit_attempt"], f"{label} submit attempt")
    client_index = _optional_u32(record["client_index"], f"{label} client index")
    _require(
        submit_attempt == expected_attempt,
        f"{label} submit attempt differs from independent regeneration",
    )
    _require(
        client_index == expected_client,
        f"{label} client index differs from independent regeneration",
    )
    result = _enum(record["result"], RESULT_CODES, f"{label} result")
    error = record["error"]
    if error is not None:
        error = _enum(error, ERROR_CODES, f"{label} error")
    _require(
        (result == "error") == (error is not None),
        f"{label} error presence disagrees with its result",
    )
    request_id = _optional_u64(record["request_id"], f"{label} request ID")

    output_index: int | None = None
    token_id: int | None = None
    output = record["output"]
    if result == "drain_output":
        output = _object(output, OUTPUT_FIELDS, f"{label} output")
        output_index = _u32(output["output_index"], f"{label} output index")
        token_id = _u32(output["token_id"], f"{label} token ID")
        _require(token_id < VOCABULARY_SIZE, f"{label} token ID exceeds tiny-v3 vocabulary")
    else:
        _require(output is None, f"{label} has output fields for a non-output result")

    rejected_submission = False
    if kind == "submit":
        if expected_client is None:
            _require(result == "submit_offer_exhausted", f"{label} exhausted offer result changed")
            _require(error is None and request_id is None, f"{label} exhausted offer has API fields")
        elif result == "submit_accepted":
            _require(
                len(live_receivers) < MAX_OUTSTANDING_REQUESTS,
                f"{label} exceeds the frozen outstanding-request cap",
            )
            _require(error is None and request_id is not None, f"{label} accepted result lacks an ID")
            _require(request_id == next_request_id, f"{label} request ID is not consecutive")
            _require(expected_client not in accepted_by_client, f"client {expected_client} was accepted twice")
            _require(request_id not in clients_by_id, f"request ID {request_id} was accepted twice")
            accepted_by_client[expected_client] = request_id
            clients_by_id[request_id] = expected_client
            live_receivers.add(expected_client)
            _require(free_control_slots, f"{label} has no free control slot")
            control_slot = free_control_slots.pop()
            previous_owner = control_owner_by_slot.get(control_slot)
            if previous_owner is not None:
                stale_authorities.add(previous_owner)
            control_owner_by_slot[control_slot] = expected_client
            control_slot_by_client[expected_client] = control_slot
            next_request_id += 1
        else:
            _require(
                result == "error" and error == "resource_exhausted" and request_id is None,
                f"{label} in-range rejection is not resource_exhausted",
            )
            _require(
                len(live_receivers) == MAX_OUTSTANDING_REQUESTS,
                f"{label} rejects below the frozen outstanding-request cap",
            )
            rejected_submission = True
    elif kind == "wake":
        _require(result == "wake_signaled", f"{label} wake result changed")
        _require(error is None and request_id is None, f"{label} global wake carries request state")
    else:
        _require(client_index is not None, f"{label} targeted action lacks a client index")
        assigned_id = accepted_by_client.get(client_index)
        _require(request_id == assigned_id, f"{label} request ID differs from its accepted client")
        if kind == "cancel":
            if assigned_id is None:
                _require(result == "target_unavailable", f"{label} called an unavailable authority")
            elif client_index in stale_authorities:
                _require(
                    result == "error" and error == "invalid_request",
                    f"{label} revived a stale cancellation authority",
                )
            else:
                allowed = {
                    "cancel_requested",
                    "cancel_already_requested",
                    "cancel_already_terminal",
                }
                if result == "error":
                    _fail(f"{label} reports a current authority as stale")
                else:
                    _require(result in allowed, f"{label} has an invalid cancellation disposition")
                if result == "cancel_requested":
                    _require(
                        client_index in live_receivers
                        and client_index not in cancellation_requested
                        and client_index not in terminal_authorities,
                        f"{label} requests cancellation more than once",
                    )
                    cancellation_requested.add(client_index)
                    terminal_authorities.add(client_index)
                elif result == "cancel_already_requested":
                    _fail(f"{label} remained requested after acknowledged quiescence")
                elif result == "cancel_already_terminal":
                    if client_index not in terminal_authorities:
                        natural_terminal_proofs.add(client_index)
                    terminal_authorities.add(client_index)
        elif kind == "drop":
            if client_index in live_receivers:
                _require(result == "receiver_dropped", f"{label} did not consume its live receiver")
                live_receivers.remove(client_index)
                terminal_authorities.add(client_index)
                control_slot = control_slot_by_client[client_index]
                _require(
                    control_slot not in free_control_slots,
                    f"{label} recycles a control slot twice",
                )
                free_control_slots.append(control_slot)
            else:
                _require(result == "target_unavailable", f"{label} called an absent receiver")
        elif kind == "drain":
            if client_index not in live_receivers:
                _require(result == "target_unavailable", f"{label} called an absent receiver")
            else:
                _require(
                    result in {"drain_output", "drain_empty", "drain_eof"},
                    f"{label} has an invalid nonblocking receive result",
                )
                if client_index in drain_eof_seen:
                    _require(result == "drain_eof", f"{label} regressed after acknowledged EOF")
                if result == "drain_output":
                    expected_output = drained_counts.get(client_index, 0)
                    _require(
                        output_index == expected_output,
                        f"{label} violates per-client FIFO output order",
                    )
                    drained_counts[client_index] = expected_output + 1
                elif result == "drain_empty":
                    _require(
                        client_index not in terminal_authorities,
                        f"{label} observed an empty queue after terminal quiescence",
                    )
                elif result == "drain_eof":
                    if client_index not in terminal_authorities:
                        natural_terminal_proofs.add(client_index)
                    terminal_authorities.add(client_index)
                    drain_eof_seen.add(client_index)
                    drain_eof_counts.append((client_index, drained_counts.get(client_index, 0)))

    return (
        ActionRecord(
            ordinal=captured_ordinal,
            kind=kind,
            producer=producer,
            result=result,
            error=error,
            submit_attempt=submit_attempt,
            client_index=client_index,
            request_id=request_id,
            output_index=output_index,
            token_id=token_id,
        ),
        next_request_id,
        rejected_submission,
    )


def _parse_observations(
    raw_observations: Any,
    clients_by_id: dict[int, int],
    accepted_by_client: dict[int, int],
) -> tuple[list[OutputRecord], list[TerminalRecord], list[EofRecord]]:
    observations = _array(
        raw_observations, "observations", maximum=OBSERVATION_LIMIT
    )
    outputs: list[OutputRecord] = []
    terminals: list[TerminalRecord] = []
    eofs: list[EofRecord] = []
    output_keys: set[tuple[int, int]] = set()
    terminal_clients: set[int] = set()
    eof_clients: set[int] = set()

    for index, raw in enumerate(observations):
        label = f"observation {index}"
        if not isinstance(raw, dict) or "kind" not in raw:
            _fail(f"{label} does not match a closed observation variant")
        kind = raw["kind"]
        if kind == "output":
            record = _object(
                raw,
                {"kind", "request_id", "output_index", "token_id"},
                label,
            )
        elif kind == "terminal":
            record = _object(
                raw,
                {
                    "kind",
                    "request_id",
                    "outcome",
                    "error",
                    "committed_positions",
                    "emitted_tokens",
                },
                label,
            )
        elif kind == "eof":
            record = _object(raw, {"kind", "request_id"}, label)
        else:
            _fail(f"{label} has an unsupported kind spelling")

        request_id = _u64(record["request_id"], f"{label} request ID")
        _require(request_id != 0, f"{label} request ID must be nonzero")
        client_index = clients_by_id.get(request_id)
        _require(client_index is not None, f"{label} refers to an unaccepted request")
        if kind == "output":
            output_index = _u32(record["output_index"], f"{label} output index")
            token_id = _u32(record["token_id"], f"{label} token ID")
            _require(token_id < VOCABULARY_SIZE, f"{label} token ID exceeds tiny-v3 vocabulary")
            _require(
                (client_index, output_index) not in output_keys,
                f"{label} duplicates a client output index",
            )
            output_keys.add((client_index, output_index))
            outputs.append(OutputRecord(client_index, request_id, output_index, token_id))
        elif kind == "terminal":
            _require(client_index not in terminal_clients, f"{label} duplicates a terminal")
            outcome = _enum(record["outcome"], OUTCOME_CODES, f"{label} outcome")
            error = record["error"]
            if error is not None:
                error = _enum(error, ERROR_CODES, f"{label} error")
            _require(
                (outcome == "failed") == (error is not None),
                f"{label} failed/error relationship is inconsistent",
            )
            _require(
                outcome not in {"deadline_exceeded", "failed"},
                f"{label} is not an allowed fault-free, deadline-free outcome",
            )
            terminals.append(
                TerminalRecord(
                    client_index=client_index,
                    request_id=request_id,
                    outcome=outcome,
                    error=error,
                    committed_positions=_u32(
                        record["committed_positions"], f"{label} committed positions"
                    ),
                    emitted_tokens=_u32(
                        record["emitted_tokens"], f"{label} emitted tokens"
                    ),
                )
            )
            terminal_clients.add(client_index)
        else:
            _require(client_index not in eof_clients, f"{label} duplicates an EOF acknowledgement")
            eofs.append(EofRecord(client_index, request_id))
            eof_clients.add(client_index)

    accepted_clients = set(accepted_by_client)
    _require(terminal_clients == accepted_clients, "terminal set differs from accepted clients")
    _require(eof_clients == accepted_clients, "EOF set differs from accepted clients")
    _require(len(outputs) <= OUTPUT_OBSERVATION_LIMIT, "output observations exceed their bound")
    outputs.sort(key=lambda record: (record.client_index, record.output_index))
    terminals.sort(key=lambda record: record.client_index)
    eofs.sort(key=lambda record: record.client_index)

    descriptors = scheduler.build_descriptors()
    output_by_client: dict[int, list[OutputRecord]] = {}
    for output in outputs:
        output_by_client.setdefault(output.client_index, []).append(output)
    terminal_by_client = {terminal.client_index: terminal for terminal in terminals}
    for client_index in sorted(accepted_clients):
        publications = output_by_client.get(client_index, [])
        _require(
            [record.output_index for record in publications] == list(range(len(publications))),
            f"client {client_index} output publications are not consecutive",
        )
        descriptor = descriptors[client_index]
        _require(
            len(publications) <= descriptor["max_new_tokens"],
            f"client {client_index} exceeds its output limit",
        )
        terminal = terminal_by_client[client_index]
        _require(
            terminal.emitted_tokens == len(publications),
            f"client {client_index} terminal/output counts differ",
        )
        prompt_length = len(descriptor["prompt"])
        maximum_positions = prompt_length + descriptor["max_new_tokens"] - 1
        _require(
            terminal.committed_positions <= maximum_positions,
            f"client {client_index} exceeds its committed-position envelope",
        )
        expected_emitted = max(0, terminal.committed_positions - (prompt_length - 1))
        _require(
            terminal.emitted_tokens == expected_emitted,
            f"client {client_index} committed/output progress is inconsistent",
        )
        if terminal.outcome == "completed":
            _require(publications, f"client {client_index} completed without an output")
            if len(publications) < descriptor["max_new_tokens"]:
                _require(
                    publications[-1].token_id == EOS_TOKEN_ID,
                    f"client {client_index} completed early without EOS",
                )
    return outputs, terminals, eofs


def _parse_shutdown(
    raw: Any, accepted_count: int, rejected_count: int
) -> ShutdownRecord:
    record = _object(raw, SHUTDOWN_FIELDS, "shutdown")
    shutdown = ShutdownRecord(
        accepted_submissions=_u32(record["accepted_submissions"], "accepted submissions"),
        rejected_submissions=_u32(record["rejected_submissions"], "rejected submissions"),
        shutdown_cancellations=_u32(record["shutdown_cancellations"], "shutdown cancellations"),
        terminated_requests=_u32(record["terminated_requests"], "shutdown terminated requests"),
        discarded_output_events=_u32(
            record["discarded_output_events"], "shutdown discarded output events"
        ),
        released_request_bytes=_u64(
            record["released_request_bytes"], "shutdown released request bytes"
        ),
        remaining_shared_bytes=_u64(
            record["remaining_shared_bytes"], "shutdown remaining shared bytes"
        ),
        final_request_bytes=_u64(record["final_request_bytes"], "final request bytes"),
        final_shared_bytes=_u64(record["final_shared_bytes"], "final shared bytes"),
    )
    _require(
        shutdown.accepted_submissions == accepted_count,
        "shutdown acceptance count differs from action results",
    )
    _require(
        shutdown.rejected_submissions == rejected_count,
        "shutdown rejection count differs from action results",
    )
    zero_fields = {
        "shutdown_cancellations": shutdown.shutdown_cancellations,
        "terminated_requests": shutdown.terminated_requests,
        "discarded_output_events": shutdown.discarded_output_events,
        "released_request_bytes": shutdown.released_request_bytes,
        "remaining_shared_bytes": shutdown.remaining_shared_bytes,
        "final_request_bytes": shutdown.final_request_bytes,
        "final_shared_bytes": shutdown.final_shared_bytes,
    }
    for field, value in zero_fields.items():
        _require(value == 0, f"pre-shutdown zero-ownership gate failed for {field}")
    return shutdown


def _validate_cleanup_cancellations(
    raw: Any,
    accepted_clients: set[int],
    live_receivers: set[int],
    terminal_authorities: set[int],
    stale_authorities: set[int],
    terminals: list[TerminalRecord],
) -> None:
    records = _array(
        raw,
        "cleanup cancellations",
        exact=len(accepted_clients),
    )
    terminal_by_client = {terminal.client_index: terminal for terminal in terminals}
    observed_clients: list[int] = []
    for index, raw_record in enumerate(records):
        label = f"cleanup cancellation {index}"
        record = _object(raw_record, CLEANUP_CANCELLATION_FIELDS, label)
        client_index = _u32(record["client_index"], f"{label} client index")
        disposition = record["disposition"]
        if not isinstance(disposition, str) or disposition not in CLEANUP_DISPOSITIONS:
            _fail(f"{label} has an unsupported disposition spelling")
        observed_clients.append(client_index)
        terminal = terminal_by_client.get(client_index)
        if client_index in stale_authorities:
            expected_disposition = "invalid_request"
        elif client_index in terminal_authorities or (
            terminal is not None and terminal.outcome == "completed"
        ):
            expected_disposition = "already_terminal"
        else:
            _require(
                client_index in live_receivers,
                f"{label} has a current nonterminal authority without a live receiver",
            )
            expected_disposition = "requested"
        _require(
            disposition == expected_disposition,
            f"{label} disposition differs from deterministic control-slot replay",
        )
        if disposition in {"requested", "already_requested"}:
            _require(
                terminal is not None and terminal.outcome == "cancelled",
                f"{label} requested cancellation without a cancelled terminal",
            )
    _require(
        observed_clients == sorted(accepted_clients),
        "cleanup cancellations do not exactly match accepted clients in client order",
    )


def _validate_diagnostics(raw: Any, observation_count: int) -> None:
    record = _object(raw, DIAGNOSTIC_FIELDS, "diagnostics")
    initial_pump = _u64(record["initial_pump_entries"], "initial pump entries")
    final_pump = _u64(record["final_pump_entries"], "final pump entries")
    pump_delta = _u64(record["pump_entries_delta"], "pump entry delta")
    _u64(record["engine_steps"], "engine steps")
    captured_count = _u64(record["observer_count"], "observer count")
    observer_limit = _u64(record["observer_limit"], "observer limit")
    initial_capacity = _u64(record["observer_initial_capacity"], "initial observer capacity")
    final_capacity = _u64(record["observer_final_capacity"], "final observer capacity")
    overflowed = _boolean(record["observer_overflowed"], "observer overflowed")
    poisoned = _boolean(record["observer_poisoned"], "observer poisoned")
    _require(final_pump >= initial_pump, "pump-entry counter moved backwards")
    _require(pump_delta == final_pump - initial_pump, "pump-entry delta is inconsistent")
    _require(pump_delta <= PUMP_ENTRY_LIMIT, "pump-entry delta exceeds the frozen cap")
    _require(captured_count == observation_count, "observer count differs from observations")
    _require(observer_limit == OBSERVATION_LIMIT, "observer logical limit changed")
    _require(initial_capacity >= observer_limit, "observer was not preallocated to its limit")
    _require(final_capacity == initial_capacity, "observer allocation grew during capture")
    _require(not overflowed and not poisoned, "observer overflowed or was poisoned")


def validate_capture(document: Any) -> ValidatedCapture:
    """Validate a logical capture and derive all transcript-level records."""

    root = _object(document, ROOT_FIELDS, "actor semantic capture")
    _require(root["schema"] == SCHEMA, "actor semantic capture schema is unsupported")
    workload = _object(root["workload"], WORKLOAD_FIELDS, "workload")
    _require(workload == _expected_workload(), "workload identity differs from regeneration")

    raw_actions = _array(root["action_results"], "action results", exact=ACTION_COUNT)
    expected_actions = scheduler.build_actions()
    accepted_by_client: dict[int, int] = {}
    clients_by_id: dict[int, int] = {}
    live_receivers: set[int] = set()
    cancellation_requested: set[int] = set()
    terminal_authorities: set[int] = set()
    natural_terminal_proofs: set[int] = set()
    stale_authorities: set[int] = set()
    free_control_slots = list(range(MAX_OUTSTANDING_REQUESTS - 1, -1, -1))
    control_slot_by_client: dict[int, int] = {}
    control_owner_by_slot: dict[int, int] = {}
    drained_counts: dict[int, int] = {}
    drain_eof_seen: set[int] = set()
    drain_eof_counts: list[tuple[int, int]] = []
    next_request_id = 1
    rejected_count = 0
    actions: list[ActionRecord] = []
    for raw, expected in zip(raw_actions, expected_actions, strict=True):
        action, next_request_id, rejected = _parse_action(
            raw,
            expected,
            accepted_by_client,
            clients_by_id,
            live_receivers,
            cancellation_requested,
            terminal_authorities,
            natural_terminal_proofs,
            stale_authorities,
            free_control_slots,
            control_slot_by_client,
            control_owner_by_slot,
            next_request_id,
            drained_counts,
            drain_eof_seen,
            drain_eof_counts,
        )
        actions.append(action)
        rejected_count += int(rejected)
    _require(accepted_by_client, "golden capture contains no accepted request")

    outputs, terminals, eofs = _parse_observations(
        root["observations"], clients_by_id, accepted_by_client
    )
    for terminal in terminals:
        _require(
            terminal.outcome == "cancelled"
            or terminal.client_index not in cancellation_requested,
            f"client {terminal.client_index} scripted cancellation lacks a cancelled terminal",
        )
        _require(
            terminal.outcome == "completed"
            or terminal.client_index not in natural_terminal_proofs,
            f"client {terminal.client_index} was already terminal without a cancellation cause",
        )
    _validate_cleanup_cancellations(
        root["cleanup_cancellations"],
        set(accepted_by_client),
        live_receivers,
        terminal_authorities,
        stale_authorities,
        terminals,
    )
    publications = {
        (record.request_id, record.output_index, record.token_id) for record in outputs
    }
    drained: set[tuple[int, int, int]] = set()
    for action in actions:
        if action.result != "drain_output":
            continue
        _require(
            action.request_id is not None
            and action.output_index is not None
            and action.token_id is not None,
            f"action {action.ordinal} drain output is incomplete",
        )
        identity = (action.request_id, action.output_index, action.token_id)
        _require(identity not in drained, f"action {action.ordinal} drains an output twice")
        _require(identity in publications, f"action {action.ordinal} has no output publication")
        drained.add(identity)
    output_counts = {
        client: sum(output.client_index == client for output in outputs)
        for client in accepted_by_client
    }
    for client_index, published_count in output_counts.items():
        queued_count = published_count - drained_counts.get(client_index, 0)
        _require(
            0 <= queued_count <= OUTPUT_CAPACITY_PER_REQUEST,
            f"client {client_index} exceeds the frozen output-channel capacity",
        )
    for client_index, drained_at_eof in drain_eof_counts:
        _require(
            drained_at_eof == output_counts[client_index],
            f"client {client_index} acknowledged EOF before conserving its publications",
        )

    shutdown = _parse_shutdown(
        root["shutdown"], len(accepted_by_client), rejected_count
    )
    _validate_diagnostics(root["diagnostics"], len(root["observations"]))
    return _make_validated_capture(
        actions=tuple(actions),
        outputs=tuple(outputs),
        terminals=tuple(terminals),
        eofs=tuple(eofs),
        shutdown=shutdown,
    )


def parse_capture_bytes(raw: bytes) -> ValidatedCapture:
    """Parse bounded canonical JSON bytes and validate their full semantics."""

    if not isinstance(raw, bytes):
        _fail("actor semantic capture must be supplied as bytes")
    if not raw or len(raw) > MAX_CAPTURE_BYTES:
        _fail("actor semantic capture is empty or exceeds its byte limit")
    if not raw.endswith(b"\n") or b"\r" in raw or not raw.isascii():
        _fail("actor semantic capture must be canonical ASCII ending in LF")
    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: _fail(f"invalid JSON constant {value!r}"),
        )
    except ActorTranscriptError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        _fail(f"actor semantic capture JSON is invalid: {error}")
    if canonical_bytes(document) != raw:
        _fail("actor semantic capture JSON is not in canonical formatting")
    return validate_capture(document)


def _read_capture(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        _fail(f"cannot open actor semantic capture without following links: {error}")
    try:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect opened actor semantic capture: {error}")
        if not stat.S_ISREG(before.st_mode):
            _fail("actor semantic capture path must name a regular file")
        if before.st_size > MAX_CAPTURE_BYTES:
            _fail("actor semantic capture exceeds its byte limit")
        chunks: list[bytes] = []
        bytes_read = 0
        try:
            while bytes_read <= MAX_CAPTURE_BYTES:
                remaining = MAX_CAPTURE_BYTES + 1 - bytes_read
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
        except OSError as error:
            _fail(f"cannot read actor semantic capture: {error}")
        raw = b"".join(chunks)
        if len(raw) > MAX_CAPTURE_BYTES:
            _fail("actor semantic capture exceeds its byte limit")
        try:
            after = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect actor semantic capture after reading: {error}")
        if (
            len(raw) != after.st_size
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            _fail("actor semantic capture changed while it was being read")
        return raw
    finally:
        os.close(descriptor)


def read_expected_digest_path(path: Path) -> str:
    """Read one canonical labeled SHA-256 line without following the leaf."""

    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as error:
        _fail(f"cannot open expected actor transcript digest without following links: {error}")
    try:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot inspect expected actor transcript digest: {error}")
        if not stat.S_ISREG(before.st_mode):
            _fail("expected actor transcript digest path must name a regular file")
        if before.st_size != EXPECTED_DIGEST_BYTES:
            _fail("expected actor transcript digest has a noncanonical length")
        chunks: list[bytes] = []
        bytes_read = 0
        try:
            while bytes_read <= EXPECTED_DIGEST_BYTES:
                remaining = EXPECTED_DIGEST_BYTES + 1 - bytes_read
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
        except OSError as error:
            _fail(f"cannot read expected actor transcript digest: {error}")
        raw = b"".join(chunks)
        try:
            after = os.fstat(descriptor)
        except OSError as error:
            _fail(f"cannot re-inspect expected actor transcript digest: {error}")
        if (
            len(raw) != EXPECTED_DIGEST_BYTES
            or len(raw) != after.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            _fail("expected actor transcript digest changed while it was being read")
        if raw[:7] != b"sha256:" or raw[-1:] != b"\n":
            _fail("expected actor transcript digest is not a canonical labeled line")
        hexadecimal = raw[7:-1]
        if len(hexadecimal) != 64 or any(
            byte not in b"0123456789abcdef" for byte in hexadecimal
        ):
            _fail("expected actor transcript digest is not lowercase SHA-256")
        return raw[:-1].decode("ascii")
    finally:
        os.close(descriptor)


def _read_authenticated_tiny_spec() -> bytes:
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(TINY_V3_SPEC_PATH, flags)
    except OSError as error:
        _fail(f"cannot open tiny-v3 spec without following links: {error}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            _fail("tiny-v3 spec path must name a nonempty regular file")
        if before.st_size > MAX_TINY_SPEC_BYTES:
            _fail("tiny-v3 spec exceeds its byte limit")
        chunks: list[bytes] = []
        bytes_read = 0
        while bytes_read <= MAX_TINY_SPEC_BYTES:
            remaining = MAX_TINY_SPEC_BYTES + 1 - bytes_read
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) != after.st_size
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            _fail("tiny-v3 spec changed while it was being authenticated")
        expected = scheduler.MODEL_IDENTITY["spec_file_sha256"]
        actual = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        _require(actual == expected, "tiny-v3 spec digest differs from the frozen workload")
        return raw
    except ActorTranscriptError:
        raise
    except OSError as error:
        _fail(f"cannot authenticate tiny-v3 spec: {error}")
    finally:
        os.close(descriptor)


def _load_authenticated_tiny_spec(load_fixture_spec: Any) -> Any:
    """Parse authenticated bytes from a sealed anonymous file.

    The oracle loader accepts a path, so a sealed memfd gives it an immutable
    snapshot rather than asking it to reopen the mutable repository inode.
    """

    raw = _read_authenticated_tiny_spec()
    try:
        flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        descriptor = os.memfd_create("runnel-tiny-v3-spec", flags)
    except (AttributeError, OSError) as error:
        _fail(f"cannot create sealed tiny-v3 spec snapshot: {error}")
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("cannot write sealed tiny-v3 spec snapshot")
            offset += written
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        return load_fixture_spec(f"/proc/self/fd/{descriptor}")
    except ActorTranscriptError:
        raise
    except (AttributeError, OSError) as error:
        _fail(f"cannot seal tiny-v3 spec snapshot: {error}")
    finally:
        os.close(descriptor)


def parse_capture_path(path: Path) -> ValidatedCapture:
    """Open one regular file without following links, then parse and validate it."""

    return parse_capture_bytes(_read_capture(Path(path)))


def serialize_transcript(capture: ValidatedCapture | Any) -> bytes:
    """Emit the exact ADR 0007 v2 transcript from validated logical fields."""

    validated = _validated(capture)
    parts = [
        TRANSCRIPT_DOMAIN,
        struct.pack(
            "<IIII",
            len(validated.actions),
            len(validated.outputs),
            len(validated.terminals),
            len(validated.eofs),
        ),
    ]
    for action in validated.actions:
        parts.append(
            struct.pack(
                "<BIBBBBIIQII",
                0x01,
                action.ordinal,
                KIND_CODES[action.kind],
                action.producer,
                RESULT_CODES[action.result],
                0 if action.error is None else ERROR_CODES[action.error],
                MAX_U32 if action.submit_attempt is None else action.submit_attempt,
                MAX_U32 if action.client_index is None else action.client_index,
                0 if action.request_id is None else action.request_id,
                0 if action.output_index is None else action.output_index,
                0 if action.token_id is None else action.token_id,
            )
        )
    for output in validated.outputs:
        parts.append(
            struct.pack(
                "<BIQII",
                0x02,
                output.client_index,
                output.request_id,
                output.output_index,
                output.token_id,
            )
        )
    for terminal in validated.terminals:
        parts.append(
            struct.pack(
                "<BIQBBHII",
                0x03,
                terminal.client_index,
                terminal.request_id,
                OUTCOME_CODES[terminal.outcome],
                0 if terminal.error is None else ERROR_CODES[terminal.error],
                0,
                terminal.committed_positions,
                terminal.emitted_tokens,
            )
        )
    for eof in validated.eofs:
        parts.append(struct.pack("<BIQ", 0x04, eof.client_index, eof.request_id))
    shutdown = validated.shutdown
    parts.append(
        struct.pack(
            "<BIIIIIIQQQQ",
            0x05,
            shutdown.accepted_submissions,
            shutdown.rejected_submissions,
            shutdown.shutdown_cancellations,
            shutdown.terminated_requests,
            shutdown.discarded_output_events,
            0,
            shutdown.released_request_bytes,
            shutdown.remaining_shared_bytes,
            shutdown.final_request_bytes,
            shutdown.final_shared_bytes,
        )
    )
    transcript = b"".join(parts)
    expected_length = (
        len(TRANSCRIPT_DOMAIN)
        + 16
        + 33 * len(validated.actions)
        + 21 * len(validated.outputs)
        + 25 * len(validated.terminals)
        + 13 * len(validated.eofs)
        + 57
    )
    _require(len(transcript) == expected_length, "transcript fixed-width plan changed")
    _require(
        len(transcript) <= MAX_TRANSCRIPT_BYTES,
        "transcript exceeds the independently derived worst-case bound",
    )
    return transcript


def transcript_digest(capture: ValidatedCapture | Any) -> str:
    """Return the labeled SHA-256 digest of the independently emitted bytes."""

    return f"sha256:{hashlib.sha256(serialize_transcript(capture)).hexdigest()}"


def _authenticated_model_descriptors() -> tuple[tuple[tuple[int, ...], int], ...]:
    """Regenerate, authenticate, and snapshot the exact 64 model requests."""

    try:
        descriptors = scheduler.build_descriptors()
    except (OverflowError, RuntimeError, TypeError, ValueError) as error:
        _fail(f"cannot regenerate model-prefix descriptors: {error}")
    _require(
        type(descriptors) is list and len(descriptors) == REQUEST_COUNT,
        "model-prefix descriptors must be an exact 64-entry list",
    )
    try:
        identity = scheduler.sequence_identity(descriptors)
    except scheduler.SchedulerFixtureError as error:
        _fail(f"cannot authenticate model-prefix descriptors: {error}")
    _require(
        identity == EXPECTED_DESCRIPTOR_VECTOR_ID,
        "model-prefix descriptor identity differs from the frozen corpus",
    )

    expected_keys = {
        "deadline_ns",
        "index",
        "max_new_tokens",
        "prompt",
        "sampling",
    }
    snapshots: list[tuple[tuple[int, ...], int]] = []
    for client_index, descriptor in enumerate(descriptors):
        label = f"model-prefix descriptor {client_index}"
        _require(
            type(descriptor) is dict and set(descriptor) == expected_keys,
            f"{label} does not use its closed schema",
        )
        index = _plain_int(descriptor["index"], f"{label} index")
        maximum = _plain_int(
            descriptor["max_new_tokens"],
            f"{label} max_new_tokens",
        )
        prompt = descriptor["prompt"]
        _require(index == client_index, f"{label} index is out of order")
        _require(
            descriptor["deadline_ns"] is None
            and type(descriptor["sampling"]) is str
            and descriptor["sampling"] == "greedy",
            f"{label} policy fields differ from the frozen workload",
        )
        _require(
            type(prompt) is list and 1 <= len(prompt) <= 4,
            f"{label} prompt must be an exact list with 1..4 tokens",
        )
        prompt_snapshot: list[int] = []
        for token_index, token in enumerate(prompt):
            token = _plain_int(token, f"{label} prompt token {token_index}")
            _require(
                1 <= token < VOCABULARY_SIZE,
                f"{label} prompt token is outside the tiny-v3 vocabulary",
            )
            prompt_snapshot.append(token)
        _require(1 <= maximum <= 16, f"{label} max_new_tokens is outside 1..16")
        _require(
            len(prompt_snapshot) + maximum - 1 <= 16,
            f"{label} exceeds the frozen 16-position model envelope",
        )
        snapshots.append((tuple(prompt_snapshot), maximum))
    return tuple(snapshots)


def _validated_generated_sequence(
    raw: Any,
    prompt: tuple[int, ...],
    maximum: int,
    client_index: int,
) -> tuple[int, ...]:
    """Validate one closed ``greedy_generate`` result before retaining tokens."""

    label = f"model-prefix sequence {client_index}"
    _require(
        type(raw) is tuple and len(raw) == 4,
        f"{label} generator result must be an exact four-tuple",
    )
    full_ids, generated, _logit_steps, stop_reason = raw
    _require(type(full_ids) is list, f"{label} full IDs must be an exact list")
    _require(type(generated) is list, f"{label} generated IDs must be an exact list")
    _require(
        type(stop_reason) is str,
        f"{label} stop reason must be an exact string",
    )
    _require(
        1 <= len(generated) <= maximum,
        f"{label} generated length is outside its descriptor bound",
    )
    tokens: list[int] = []
    for token_index, token in enumerate(generated):
        token = _plain_int(token, f"{label} token {token_index}")
        _require(
            0 <= token < VOCABULARY_SIZE,
            f"{label} token is outside the tiny-v3 vocabulary",
        )
        tokens.append(token)
    _require(
        EOS_TOKEN_ID not in tokens[:-1],
        f"{label} generated output after EOS",
    )
    full_snapshot: list[int] = []
    for token_index, token in enumerate(full_ids):
        token = _plain_int(token, f"{label} full token {token_index}")
        _require(
            0 <= token < VOCABULARY_SIZE,
            f"{label} full token is outside the tiny-v3 vocabulary",
        )
        full_snapshot.append(token)
    _require(
        tuple(full_snapshot) == (*prompt, *tokens),
        f"{label} full IDs do not equal prompt plus generated IDs",
    )
    if tokens[-1] == EOS_TOKEN_ID:
        _require(stop_reason == "eos", f"{label} EOS stop reason is inconsistent")
    else:
        _require(
            stop_reason == "max_new_tokens" and len(tokens) == maximum,
            f"{label} limit stop reason or length is inconsistent",
        )
    return tuple(tokens)


def build_authenticated_model_sequences() -> tuple[tuple[int, ...], ...]:
    """Build all 64 frozen tiny-v3 sequences with the independent oracle.

    Descriptor and spec authentication occur before any model result is
    accepted. PyTorch remains a lazy dependency, runs deterministically with one
    intra-op thread, and has its process-global settings restored on every exit.
    """

    descriptors = _authenticated_model_descriptors()
    try:
        import torch

        from oracle.generate import greedy_generate
        from oracle.runnel_oracle import TinyMoEOracle, load_fixture_spec
    except (ImportError, OSError, RuntimeError) as error:
        _fail(f"PyTorch model-prefix validation is unavailable: {error}")

    try:
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
    except (
        AttributeError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _fail(f"cannot inspect PyTorch model-prefix settings: {error}")

    sequences: tuple[tuple[int, ...], ...] | None = None
    failure: BaseException | None = None
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        spec = _load_authenticated_tiny_spec(load_fixture_spec)
        model = TinyMoEOracle(spec)
        generated_sequences: list[tuple[int, ...]] = []
        for client_index, (prompt, maximum) in enumerate(descriptors):
            raw = greedy_generate(model, list(prompt), maximum)
            generated_sequences.append(
                _validated_generated_sequence(
                    raw,
                    prompt,
                    maximum,
                    client_index,
                )
            )
        sequences = tuple(generated_sequences)
    except BaseException as error:
        failure = error

    restore_errors: list[tuple[str, BaseException]] = []
    try:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
    except BaseException as error:
        restore_errors.append(("deterministic algorithms", error))
    try:
        torch.set_num_threads(previous_threads)
    except BaseException as error:
        restore_errors.append(("thread count", error))
    restore_message = (
        "cannot restore PyTorch model-prefix settings: "
        + "; ".join(f"{label}: {error}" for label, error in restore_errors)
        if restore_errors
        else ""
    )

    fatal_restore = next(
        (
            error
            for _label, error in restore_errors
            if not isinstance(error, Exception)
        ),
        None,
    )
    if fatal_restore is not None:
        if failure is not None:
            fatal_restore.add_note(
                f"model-prefix generation also failed: {type(failure).__name__}: {failure}"
            )
        for label, error in restore_errors:
            if error is not fatal_restore:
                fatal_restore.add_note(
                    f"additional {label} restoration failure: "
                    f"{type(error).__name__}: {error}"
                )
        raise fatal_restore.with_traceback(fatal_restore.__traceback__)

    if failure is not None:
        if isinstance(failure, ActorTranscriptError):
            if restore_errors:
                raise ActorTranscriptError(f"{failure}; {restore_message}") from failure
            raise failure.with_traceback(failure.__traceback__)
        if isinstance(
            failure,
            (AttributeError, OSError, OverflowError, RuntimeError, TypeError, ValueError),
        ):
            message = f"independent tiny-v3 model-prefix generation failed: {failure}"
            if restore_errors:
                message = f"{message}; {restore_message}"
            raise ActorTranscriptError(message) from failure
        if restore_errors:
            failure.add_note(restore_message)
        raise failure.with_traceback(failure.__traceback__)
    if restore_errors:
        first_restore_error = restore_errors[0][1]
        for label, error in restore_errors[1:]:
            first_restore_error.add_note(f"additional {label} restoration failure: {error}")
        if isinstance(first_restore_error, Exception):
            raise ActorTranscriptError(restore_message) from first_restore_error
        raise first_restore_error.with_traceback(first_restore_error.__traceback__)
    _require(sequences is not None, "model-prefix generation produced no corpus")
    _require(
        type(sequences) is tuple
        and len(sequences) == REQUEST_COUNT
        and all(type(sequence) is tuple for sequence in sequences),
        "model-prefix corpus is not an immutable 64-sequence tuple",
    )
    _require(
        sum(len(sequence) for sequence in sequences)
        <= sum(maximum for _prompt, maximum in descriptors),
        "model-prefix corpus exceeds its authenticated token bound",
    )
    return sequences


def validate_model_output_prefixes(capture: ValidatedCapture | Any) -> None:
    """Check captured tokens against the independent tiny-v3 PyTorch oracle.

    PyTorch is imported only when this stronger CI gate is explicitly called;
    structural parsing and byte serialization remain stdlib-only.  Every one
    of the 64 frozen descriptors is regenerated, then each accepted request's
    publications must be an exact prefix.  A completed request must contain
    the oracle's entire stop-or-limit sequence.
    """

    validated = _validated(capture)
    expected_outputs = build_authenticated_model_sequences()

    observed: dict[int, list[int]] = {}
    for output in validated.outputs:
        observed.setdefault(output.client_index, []).append(output.token_id)
    terminals = {terminal.client_index: terminal for terminal in validated.terminals}
    for client_index, terminal in terminals.items():
        actual = tuple(observed.get(client_index, []))
        expected = expected_outputs[client_index]
        _require(
            actual == expected[: len(actual)],
            f"client {client_index} output is not an independent tiny-v3 prefix",
        )
        if terminal.outcome == "completed":
            _require(
                actual == expected,
                f"client {client_index} completed before the independent sequence ended",
            )
        else:
            _require(
                len(actual) < len(expected),
                f"client {client_index} was cancelled after its full sequence completed",
            )


def check_path(path: Path) -> str:
    """Validate a capture path and return its independently derived digest."""

    return transcript_digest(parse_capture_path(path))


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        _fail(f"cannot create new transcript output: {error}")
    try:
        written = 0
        while written < len(payload):
            try:
                count = os.write(descriptor, payload[written:])
            except OSError as error:
                _fail(f"cannot write transcript output: {error}")
            if count <= 0:
                _fail("transcript output write made no progress")
            written += count
        try:
            os.fsync(descriptor)
        except OSError as error:
            _fail(f"cannot synchronize transcript output: {error}")
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "verify a logical actor capture and optionally emit its independent "
            "ADR v2 transcript"
        )
    )
    parser.add_argument("capture", type=Path, help="canonical logical capture JSON")
    parser.add_argument(
        "--transcript",
        "--output",
        dest="transcript",
        type=Path,
        help="new binary transcript path (must not already exist)",
    )
    parser.add_argument(
        "--expected-digest",
        type=Path,
        help="canonical labeled semantic digest file to require",
    )
    parser.add_argument(
        "--validate-model",
        action="store_true",
        help="also require exact output prefixes from the tiny-v3 PyTorch oracle",
    )
    arguments = parser.parse_args(argv)
    if arguments.transcript is None and arguments.expected_digest is None:
        parser.error("at least one of --transcript or --expected-digest is required")
    try:
        capture = parse_capture_path(arguments.capture)
        if arguments.validate_model:
            validate_model_output_prefixes(capture)
        transcript = serialize_transcript(capture)
        digest = f"sha256:{hashlib.sha256(transcript).hexdigest()}"
        if arguments.expected_digest is not None:
            expected = read_expected_digest_path(arguments.expected_digest)
            _require(digest == expected, "actor semantic transcript digest is not accepted")
        if arguments.transcript is not None:
            _write_new(arguments.transcript, transcript)
    except ActorTranscriptError as error:
        print(f"actor transcript oracle: {error}", file=sys.stderr)
        return 1
    destination = (
        f" transcript={arguments.transcript}" if arguments.transcript is not None else ""
    )
    print(f"{digest} bytes={len(transcript)}{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
