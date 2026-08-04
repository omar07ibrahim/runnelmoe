from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import os
import struct
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from oracle import actor_transcript, scheduler


def _workload() -> dict[str, str]:
    fixture = scheduler.build_fixture()
    fixture_bytes = scheduler.canonical_bytes(fixture)
    return {
        "fixture_schema": scheduler.SCHEMA,
        "specification": scheduler.SPECIFICATION,
        "fixture_id": fixture["fixture_id"],
        "fixture_file_sha256": f"sha256:{hashlib.sha256(fixture_bytes).hexdigest()}",
    }


def _capture() -> dict[str, object]:
    assigned: dict[int, int] = {}
    live_receivers: set[int] = set()
    terminal_authorities: set[int] = set()
    stale_authorities: set[int] = set()
    free_control_slots = list(
        range(actor_transcript.MAX_OUTSTANDING_REQUESTS - 1, -1, -1)
    )
    control_slot_by_client: dict[int, int] = {}
    control_owner_by_slot: dict[int, int] = {}
    next_request_id = 1
    actions: list[dict[str, object]] = []
    rejected = 0
    for expected in scheduler.build_actions():
        kind = expected["kind"]
        if kind == "submit":
            submit_attempt: int | None = expected["submit_attempt"]
            client_index: int | None = expected["request_index"]
        elif kind == "wake":
            submit_attempt = None
            client_index = expected["selector_index"]
        else:
            submit_attempt = None
            client_index = expected["request_index"]
        record: dict[str, object] = {
            "client_index": client_index,
            "error": None,
            "kind": kind,
            "ordinal": expected["ordinal"],
            "output": None,
            "producer": expected["producer"],
            "request_id": None if client_index is None else assigned.get(client_index),
            "result": "target_unavailable",
            "submit_attempt": submit_attempt,
        }
        if kind == "submit":
            if client_index is None:
                record["result"] = "submit_offer_exhausted"
            elif len(live_receivers) < actor_transcript.MAX_OUTSTANDING_REQUESTS:
                assigned[client_index] = next_request_id
                live_receivers.add(client_index)
                record["request_id"] = next_request_id
                record["result"] = "submit_accepted"
                control_slot = free_control_slots.pop()
                previous_owner = control_owner_by_slot.get(control_slot)
                if previous_owner is not None:
                    stale_authorities.add(previous_owner)
                control_owner_by_slot[control_slot] = client_index
                control_slot_by_client[client_index] = control_slot
                next_request_id += 1
            else:
                record["result"] = "error"
                record["error"] = "resource_exhausted"
                rejected += 1
        elif kind == "wake":
            record["request_id"] = None
            record["result"] = "wake_signaled"
        elif client_index in assigned:
            record["request_id"] = assigned[client_index]
            if kind == "cancel":
                if client_index in stale_authorities:
                    record["result"] = "error"
                    record["error"] = "invalid_request"
                elif client_index in terminal_authorities:
                    record["result"] = "cancel_already_terminal"
                else:
                    record["result"] = "cancel_requested"
                    terminal_authorities.add(client_index)
            elif kind == "drop":
                if client_index in live_receivers:
                    record["result"] = "receiver_dropped"
                    live_receivers.remove(client_index)
                    terminal_authorities.add(client_index)
                    free_control_slots.append(control_slot_by_client[client_index])
            elif kind == "drain" and client_index in live_receivers:
                record["result"] = (
                    "drain_eof"
                    if client_index in terminal_authorities
                    else "drain_empty"
                )
        actions.append(record)

    observations: list[dict[str, object]] = []
    for client_index, request_id in assigned.items():
        observations.append(
            {
                "committed_positions": 0,
                "emitted_tokens": 0,
                "error": None,
                "kind": "terminal",
                "outcome": "cancelled",
                "request_id": request_id,
            }
        )
        observations.append({"kind": "eof", "request_id": request_id})
    observer_count = len(observations)
    cleanup_cancellations = [
        {
            "client_index": client_index,
            "disposition": (
                "invalid_request"
                if client_index in stale_authorities
                else "already_terminal"
                if client_index in terminal_authorities
                else "requested"
            ),
        }
        for client_index in sorted(assigned)
    ]
    return {
        "action_results": actions,
        "cleanup_cancellations": cleanup_cancellations,
        "diagnostics": {
            "engine_steps": 77,
            "final_pump_entries": 100,
            "initial_pump_entries": 10,
            "observer_count": observer_count,
            "observer_final_capacity": actor_transcript.OBSERVATION_LIMIT,
            "observer_initial_capacity": actor_transcript.OBSERVATION_LIMIT,
            "observer_limit": actor_transcript.OBSERVATION_LIMIT,
            "observer_overflowed": False,
            "observer_poisoned": False,
            "pump_entries_delta": 90,
        },
        "observations": observations,
        "schema": actor_transcript.SCHEMA,
        "shutdown": {
            "accepted_submissions": len(assigned),
            "discarded_output_events": 0,
            "final_request_bytes": 0,
            "final_shared_bytes": 0,
            "rejected_submissions": rejected,
            "released_request_bytes": 0,
            "remaining_shared_bytes": 0,
            "shutdown_cancellations": 0,
            "terminated_requests": 0,
        },
        "workload": _workload(),
    }


def _add_output(
    capture: dict[str, object], client_index: int, token_id: int = 7
) -> None:
    request_id = next(
        action["request_id"]
        for action in capture["action_results"]  # type: ignore[index]
        if action["kind"] == "submit"  # type: ignore[index]
        and action["client_index"] == client_index  # type: ignore[index]
    )
    observations = capture["observations"]  # type: ignore[assignment]
    output_index = sum(
        observation["kind"] == "output"
        and observation["request_id"] == request_id
        for observation in observations  # type: ignore[union-attr]
    )
    observations.insert(  # type: ignore[union-attr]
        0,
        {
            "kind": "output",
            "output_index": output_index,
            "request_id": request_id,
            "token_id": token_id,
        },
    )
    drain = next(
        (
            action
            for action in capture["action_results"]  # type: ignore[index]
            if action["kind"] == "drain"  # type: ignore[index]
            and action["client_index"] == client_index  # type: ignore[index]
            and action["result"] == "drain_empty"  # type: ignore[index]
        ),
        None,
    )
    if drain is not None:
        drain["result"] = "drain_output"
        drain["output"] = {"output_index": output_index, "token_id": token_id}
    descriptor = scheduler.build_descriptors()[client_index]
    for observation in observations:  # type: ignore[union-attr]
        if observation["kind"] == "terminal" and observation["request_id"] == request_id:
            observation["committed_positions"] = len(descriptor["prompt"]) + output_index
            observation["emitted_tokens"] = output_index + 1
            break
    capture["diagnostics"]["observer_count"] += 1  # type: ignore[index,operator]


class ActorTranscriptTests(unittest.TestCase):
    def test_validated_capture_cannot_bypass_validation(self) -> None:
        shutdown = actor_transcript.ShutdownRecord(0, 0, 0, 0, 0, 0, 0, 0, 0)
        with self.assertRaisesRegex(
            actor_transcript.ActorTranscriptError, "only be created"
        ):
            actor_transcript.ValidatedCapture((), (), (), (), shutdown)

        forged = object.__new__(actor_transcript.ValidatedCapture)
        for field, value in (
            ("actions", ()),
            ("outputs", ()),
            ("terminals", ()),
            ("eofs", ()),
            ("shutdown", shutdown),
        ):
            object.__setattr__(forged, field, value)
        with self.assertRaisesRegex(
            actor_transcript.ActorTranscriptError, "not created by capture validation"
        ):
            actor_transcript.serialize_transcript(forged)
        with self.assertRaisesRegex(
            actor_transcript.ActorTranscriptError, "not created by capture validation"
        ):
            actor_transcript.validate_model_output_prefixes(forged)

    def test_valid_capture_round_trips_and_emits_fixed_width_v2(self) -> None:
        document = _capture()
        raw = actor_transcript.canonical_bytes(document)
        validated = actor_transcript.parse_capture_bytes(raw)
        transcript = actor_transcript.serialize_transcript(validated)
        self.assertTrue(transcript.startswith(actor_transcript.TRANSCRIPT_DOMAIN))
        header_offset = len(actor_transcript.TRANSCRIPT_DOMAIN)
        self.assertEqual(
            struct.unpack_from("<IIII", transcript, header_offset),
            (scheduler.ACTION_COUNT, 0, 34, 34),
        )
        expected_length = (
            header_offset
            + 16
            + scheduler.ACTION_COUNT * 33
            + 34 * 25
            + 34 * 13
            + 57
        )
        self.assertEqual(len(transcript), expected_length)
        self.assertRegex(
            actor_transcript.transcript_digest(validated), r"^sha256:[0-9a-f]{64}$"
        )
        self.assertEqual(actor_transcript.MAX_TRANSCRIPT_BYTES, 47_823)
        self.assertLessEqual(len(transcript), actor_transcript.MAX_TRANSCRIPT_BYTES)

    def test_serializer_derives_codes_null_sentinels_and_reserved_zeroes(self) -> None:
        document = _capture()
        transcript = actor_transcript.serialize_transcript(document)
        action_base = len(actor_transcript.TRANSCRIPT_DOMAIN) + 16
        first = struct.unpack_from("<BIBBBBIIQII", transcript, action_base)
        self.assertEqual(
            first,
            (
                0x01,
                0,
                actor_transcript.KIND_CODES["drain"],
                0,
                actor_transcript.RESULT_CODES["target_unavailable"],
                0,
                actor_transcript.MAX_U32,
                42,
                0,
                0,
                0,
            ),
        )

        exhausted = next(
            action
            for action in document["action_results"]  # type: ignore[index]
            if action["result"] == "submit_offer_exhausted"  # type: ignore[index]
        )
        exhausted_offset = action_base + exhausted["ordinal"] * 33  # type: ignore[index,operator]
        unpacked = struct.unpack_from("<BIBBBBIIQII", transcript, exhausted_offset)
        self.assertEqual(unpacked[4], actor_transcript.RESULT_CODES["submit_offer_exhausted"])
        self.assertEqual(unpacked[7], actor_transcript.MAX_U32)
        self.assertEqual(unpacked[8], 0)

        terminal_offset = action_base + scheduler.ACTION_COUNT * 33
        terminal = struct.unpack_from("<BIQBBHII", transcript, terminal_offset)
        self.assertEqual(terminal, (0x03, 0, 1, 1, 0, 0, 0, 0))
        shutdown = struct.unpack_from("<BIIIIIIQQQQ", transcript, len(transcript) - 57)
        self.assertEqual(shutdown[0], 0x05)
        self.assertEqual(shutdown[6], 0)

    def test_observations_are_sorted_by_derived_client_and_output_index(self) -> None:
        document = _capture()
        _add_output(document, 0, 7)
        _add_output(document, 1, 9)
        document["observations"].reverse()  # type: ignore[union-attr]
        transcript = actor_transcript.serialize_transcript(document)
        output_base = (
            len(actor_transcript.TRANSCRIPT_DOMAIN)
            + 16
            + scheduler.ACTION_COUNT * 33
        )
        first = struct.unpack_from("<BIQII", transcript, output_base)
        second = struct.unpack_from("<BIQII", transcript, output_base + 21)
        self.assertEqual(first, (0x02, 0, 1, 0, 7))
        self.assertEqual(second, (0x02, 1, 2, 0, 9))

    def test_physical_diagnostics_do_not_enter_semantic_transcript(self) -> None:
        document = _capture()
        alternate = copy.deepcopy(document)
        alternate["diagnostics"].update(  # type: ignore[union-attr]
            {
                "engine_steps": 999,
                "final_pump_entries": 333,
                "initial_pump_entries": 222,
                "pump_entries_delta": 111,
            }
        )
        self.assertEqual(
            actor_transcript.serialize_transcript(document),
            actor_transcript.serialize_transcript(alternate),
        )

    def test_lazy_model_gate_checks_prefixes_and_completed_sequences(self) -> None:
        document = _capture()
        _add_output(document, 0, 7)
        validated = actor_transcript.validate_capture(document)

        fake_torch = types.ModuleType("torch")
        fake_torch.get_num_threads = lambda: 4  # type: ignore[attr-defined]
        fake_torch.set_num_threads = lambda _threads: None  # type: ignore[attr-defined]
        fake_torch.are_deterministic_algorithms_enabled = lambda: False  # type: ignore[attr-defined]
        fake_torch.use_deterministic_algorithms = lambda _enabled: None  # type: ignore[attr-defined]
        fake_generate = types.ModuleType("oracle.generate")
        fake_generate.greedy_generate = (  # type: ignore[attr-defined]
            lambda _model, prompt, _maximum: (prompt + [7, 8], [7, 8], [], "limit")
        )
        fake_oracle = types.ModuleType("oracle.runnel_oracle")
        fake_oracle.TinyMoEOracle = lambda _spec: object()  # type: ignore[attr-defined]
        fake_oracle.load_fixture_spec = lambda _path: object()  # type: ignore[attr-defined]
        with mock.patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "oracle.generate": fake_generate,
                "oracle.runnel_oracle": fake_oracle,
            },
        ):
            actor_transcript.validate_model_output_prefixes(validated)
            wrong = copy.deepcopy(document)
            wrong["observations"][0]["token_id"] = 9  # type: ignore[index]
            with self.assertRaises(actor_transcript.ActorTranscriptError):
                actor_transcript.validate_model_output_prefixes(wrong)

            full_but_cancelled = copy.deepcopy(document)
            full_but_cancelled["observations"].insert(  # type: ignore[union-attr]
                1,
                {
                    "kind": "output",
                    "output_index": 1,
                    "request_id": 1,
                    "token_id": 8,
                },
            )
            terminal = next(
                observation
                for observation in full_but_cancelled["observations"]  # type: ignore[index]
                if observation["kind"] == "terminal"  # type: ignore[index]
            )
            terminal["committed_positions"] += 1  # type: ignore[index,operator]
            terminal["emitted_tokens"] = 2  # type: ignore[index]
            full_but_cancelled["diagnostics"]["observer_count"] += 1  # type: ignore[index,operator]
            with self.assertRaises(actor_transcript.ActorTranscriptError):
                actor_transcript.validate_model_output_prefixes(full_but_cancelled)

            with tempfile.TemporaryDirectory() as directory:
                stale_spec = Path(directory) / "spec.json"
                stale_spec.write_bytes(b"{}\n")
                with mock.patch.object(
                    actor_transcript, "TINY_V3_SPEC_PATH", stale_spec
                ):
                    with self.assertRaisesRegex(
                        actor_transcript.ActorTranscriptError, "digest differs"
                    ):
                        actor_transcript.validate_model_output_prefixes(validated)

    def test_capture_is_bound_to_independently_regenerated_actions(self) -> None:
        mutations = []
        for field, value in (
            ("kind", "wake"),
            ("producer", 1),
            ("client_index", 41),
            ("submit_attempt", 0),
        ):
            mutated = _capture()
            mutated["action_results"][0][field] = value  # type: ignore[index]
            mutations.append((field, mutated))
        for field, mutated in mutations:
            with self.subTest(field=field):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.validate_capture(mutated)

    def test_request_ids_are_raw_nonzero_consecutive_and_client_bound(self) -> None:
        for label, mutate in (
            (
                "nonconsecutive",
                lambda document: document["action_results"][1].update(request_id=2),
            ),
            (
                "target mismatch",
                lambda document: next(
                    action
                    for action in document["action_results"]
                    if action["client_index"] == 0 and action["kind"] == "drain"
                ).update(request_id=2),
            ),
            (
                "zero",
                lambda document: document["action_results"][1].update(request_id=0),
            ),
        ):
            document = _capture()
            mutate(document)  # type: ignore[arg-type]
            with self.subTest(label=label):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.validate_capture(document)

        duplicate = _capture()
        duplicate["action_results"][2]["request_id"] = 1  # type: ignore[index]
        with self.assertRaises(actor_transcript.ActorTranscriptError):
            actor_transcript.validate_capture(duplicate)

    def test_admission_replay_enforces_the_frozen_outstanding_cap(self) -> None:
        rejected_below_cap = _capture()
        first_accepted = next(
            action
            for action in rejected_below_cap["action_results"]  # type: ignore[index]
            if action["result"] == "submit_accepted"  # type: ignore[index]
        )
        first_accepted.update(  # type: ignore[union-attr]
            result="error", error="resource_exhausted", request_id=None
        )
        with self.assertRaisesRegex(
            actor_transcript.ActorTranscriptError, "rejects below"
        ):
            actor_transcript.validate_capture(rejected_below_cap)

        accepted_above_cap = _capture()
        first_rejected = next(
            action
            for action in accepted_above_cap["action_results"]  # type: ignore[index]
            if action["kind"] == "submit"  # type: ignore[index]
            and action["error"] == "resource_exhausted"  # type: ignore[index]
        )
        accepted_before = sum(
            action["kind"] == "submit"  # type: ignore[index]
            and action["result"] == "submit_accepted"  # type: ignore[index]
            and action["ordinal"] < first_rejected["ordinal"]  # type: ignore[index,operator]
            for action in accepted_above_cap["action_results"]  # type: ignore[index]
        )
        first_rejected.update(  # type: ignore[union-attr]
            result="submit_accepted",
            error=None,
            request_id=accepted_before + 1,
        )
        with self.assertRaisesRegex(
            actor_transcript.ActorTranscriptError, "exceeds the frozen"
        ):
            actor_transcript.validate_capture(accepted_above_cap)

    def test_action_error_output_and_fifo_relationships_fail_closed(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []
        wrong_error = _capture()
        wrong_error["action_results"][1]["error"] = "internal"  # type: ignore[index]
        cases.append(("error relationship", wrong_error))

        reserved_output = _capture()
        reserved_output["action_results"][0]["output"] = {  # type: ignore[index]
            "output_index": 0,
            "token_id": 1,
        }
        cases.append(("reserved output", reserved_output))

        unknown_drain = _capture()
        drain = next(
            action
            for action in unknown_drain["action_results"]  # type: ignore[index]
            if action["ordinal"] == 43  # type: ignore[index]
        )
        drain["result"] = "drain_output"  # type: ignore[index]
        drain["output"] = {"output_index": 0, "token_id": 7}  # type: ignore[index]
        cases.append(("unpublished drain", unknown_drain))

        for label, document in cases:
            with self.subTest(label=label):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.validate_capture(document)

    def test_cancellation_state_rejects_unrequested_and_revived_authorities(self) -> None:
        unrequested = _capture()
        first_cancel = next(
            action
            for action in unrequested["action_results"]  # type: ignore[index]
            if action["client_index"] == 0 and action["kind"] == "cancel"  # type: ignore[index]
        )
        first_cancel["result"] = "cancel_already_requested"  # type: ignore[index]
        with self.assertRaises(actor_transcript.ActorTranscriptError):
            actor_transcript.validate_capture(unrequested)

        terminal_without_cause = _capture()
        first_cancel = next(
            action
            for action in terminal_without_cause["action_results"]  # type: ignore[index]
            if action["client_index"] == 0  # type: ignore[index]
            and action["result"] == "cancel_requested"  # type: ignore[index]
        )
        first_cancel["result"] = "cancel_already_terminal"  # type: ignore[index]
        with self.assertRaises(actor_transcript.ActorTranscriptError):
            actor_transcript.validate_capture(terminal_without_cause)

        empty_after_quiescence = _capture()
        first_eof = next(
            action
            for action in empty_after_quiescence["action_results"]  # type: ignore[index]
            if action["client_index"] == 0  # type: ignore[index]
            and action["result"] == "drain_eof"  # type: ignore[index]
        )
        first_eof["result"] = "drain_empty"  # type: ignore[index]
        with self.assertRaisesRegex(
            actor_transcript.ActorTranscriptError, "after terminal quiescence"
        ):
            actor_transcript.validate_capture(empty_after_quiescence)

        revived = _capture()
        revived_cancel = next(
            action
            for action in revived["action_results"]  # type: ignore[index]
            if action["kind"] == "cancel"  # type: ignore[index]
            and action["result"] == "error"  # type: ignore[index]
            and action["error"] == "invalid_request"  # type: ignore[index]
        )
        revived_cancel.update(result="cancel_already_terminal", error=None)  # type: ignore[union-attr]
        with self.assertRaises(actor_transcript.ActorTranscriptError):
            actor_transcript.validate_capture(revived)

    def test_cleanup_authorities_are_complete_sorted_and_semantically_excluded(self) -> None:
        document = _capture()
        accepted_clients = sorted(
            action["client_index"]
            for action in document["action_results"]  # type: ignore[index]
            if action["result"] == "submit_accepted"  # type: ignore[index]
        )
        self.assertEqual(
            [record["client_index"] for record in document["cleanup_cancellations"]],  # type: ignore[index]
            accepted_clients,
        )
        transcript = actor_transcript.serialize_transcript(document)
        for spelling in actor_transcript.CLEANUP_DISPOSITIONS:
            self.assertNotIn(spelling.encode("ascii"), transcript)

        for label, mutate in (
            (
                "missing",
                lambda capture: capture["cleanup_cancellations"].pop(),
            ),
            (
                "unsorted",
                lambda capture: capture["cleanup_cancellations"].reverse(),
            ),
            (
                "spelling",
                lambda capture: capture["cleanup_cancellations"][0].update(
                    disposition="stale"
                ),
            ),
            (
                "wrong disposition",
                lambda capture: capture["cleanup_cancellations"][0].update(
                    disposition="already_terminal"
                ),
            ),
            (
                "wrong current disposition",
                lambda capture: capture["cleanup_cancellations"][1].update(
                    disposition="invalid_request"
                ),
            ),
        ):
            malformed = copy.deepcopy(document)
            mutate(malformed)  # type: ignore[arg-type]
            with self.subTest(label=label):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.validate_capture(malformed)

    def test_repeated_eof_is_sticky_for_a_live_receiver(self) -> None:
        document = _capture()
        _add_output(document, 0, 7)
        actor_transcript.validate_capture(document)

        regression = copy.deepcopy(document)
        next(
            action
            for action in regression["action_results"]  # type: ignore[index]
            if action["ordinal"] == 124  # type: ignore[index]
        )["result"] = "drain_empty"
        with self.assertRaises(actor_transcript.ActorTranscriptError):
            actor_transcript.validate_capture(regression)

    def test_observation_conservation_progress_and_terminal_sets_are_checked(self) -> None:
        missing_eof = _capture()
        missing_eof["observations"] = [  # type: ignore[assignment]
            observation
            for observation in missing_eof["observations"]  # type: ignore[index]
            if observation["kind"] != "eof"
        ]
        missing_eof["diagnostics"]["observer_count"] -= 1  # type: ignore[index,operator]

        gap = _capture()
        _add_output(gap, 0)
        gap["observations"][0]["output_index"] = 1  # type: ignore[index]

        count_mismatch = _capture()
        _add_output(count_mismatch, 0)
        terminal = next(
            observation
            for observation in count_mismatch["observations"]  # type: ignore[index]
            if observation["kind"] == "terminal"  # type: ignore[index]
        )
        terminal["emitted_tokens"] = 0  # type: ignore[index]

        foreign = _capture()
        foreign["observations"][0]["request_id"] = 999  # type: ignore[index]

        for label, document in (
            ("missing EOF", missing_eof),
            ("output gap", gap),
            ("terminal count", count_mismatch),
            ("foreign request", foreign),
        ):
            with self.subTest(label=label):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.validate_capture(document)

    def test_undrained_publications_cannot_exceed_output_channel_capacity(self) -> None:
        document = _capture()
        _add_output(document, 1, 7)
        _add_output(document, 1, 8)
        actor_transcript.validate_capture(document)
        _add_output(document, 1, 9)
        with self.assertRaisesRegex(
            actor_transcript.ActorTranscriptError, "output-channel capacity"
        ):
            actor_transcript.validate_capture(document)

    def test_early_completion_requires_eos_and_fault_outcomes_are_invalid(self) -> None:
        completed = _capture()
        _add_output(completed, 0, actor_transcript.EOS_TOKEN_ID)
        terminal = next(
            observation
            for observation in completed["observations"]  # type: ignore[index]
            if observation["kind"] == "terminal"  # type: ignore[index]
        )
        terminal["outcome"] = "completed"  # type: ignore[index]
        for action in completed["action_results"]:  # type: ignore[index]
            if action["client_index"] == 0 and action["result"] == "cancel_requested":
                action["result"] = "cancel_already_terminal"
        actor_transcript.validate_capture(completed)

        no_eos = copy.deepcopy(completed)
        no_eos["observations"][0]["token_id"] = 2  # type: ignore[index]
        with self.assertRaises(actor_transcript.ActorTranscriptError):
            actor_transcript.validate_capture(no_eos)

        failed = _capture()
        terminal = next(
            observation
            for observation in failed["observations"]  # type: ignore[index]
            if observation["kind"] == "terminal"  # type: ignore[index]
        )
        terminal.update(outcome="failed", error="internal")  # type: ignore[union-attr]
        with self.assertRaises(actor_transcript.ActorTranscriptError):
            actor_transcript.validate_capture(failed)

    def test_shutdown_and_diagnostic_guards_are_enforced(self) -> None:
        cases = []
        nonzero_shutdown = _capture()
        nonzero_shutdown["shutdown"]["released_request_bytes"] = 1  # type: ignore[index]
        cases.append(("shutdown", nonzero_shutdown))
        wrong_delta = _capture()
        wrong_delta["diagnostics"]["pump_entries_delta"] = 89  # type: ignore[index]
        cases.append(("pump delta", wrong_delta))
        grown = _capture()
        grown["diagnostics"]["observer_final_capacity"] += 1  # type: ignore[index,operator]
        cases.append(("observer growth", grown))
        poisoned = _capture()
        poisoned["diagnostics"]["observer_poisoned"] = True  # type: ignore[index]
        cases.append(("poisoned", poisoned))
        pump_overflow = _capture()
        pump_overflow["diagnostics"].update(  # type: ignore[union-attr]
            {"final_pump_entries": 5_000, "pump_entries_delta": 4_990}
        )
        cases.append(("pump overflow", pump_overflow))
        for label, document in cases:
            with self.subTest(label=label):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.validate_capture(document)

    def test_closed_schema_and_plain_integer_rules_reject_mutations(self) -> None:
        cases = []
        unknown_root = _capture()
        unknown_root["unknown"] = None
        cases.append(("root", unknown_root))
        unknown_action = _capture()
        unknown_action["action_results"][0]["unknown"] = None  # type: ignore[index]
        cases.append(("action", unknown_action))
        boolean_integer = _capture()
        boolean_integer["action_results"][0]["ordinal"] = False  # type: ignore[index]
        cases.append(("boolean", boolean_integer))
        over_width = _capture()
        over_width["action_results"][0]["ordinal"] = actor_transcript.MAX_U32 + 1  # type: ignore[index]
        cases.append(("over width", over_width))
        stale_workload = _capture()
        stale_workload["workload"]["fixture_id"] = "sha256:" + "0" * 64  # type: ignore[index]
        cases.append(("workload", stale_workload))
        unknown_observation = _capture()
        unknown_observation["observations"][0]["unknown"] = None  # type: ignore[index]
        cases.append(("observation", unknown_observation))
        for label, document in cases:
            with self.subTest(label=label):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.validate_capture(document)

    def test_parser_rejects_duplicate_noncanonical_and_unbounded_json(self) -> None:
        raw = actor_transcript.canonical_bytes(_capture())
        duplicate = raw.replace(
            b"{\n",
            b'{\n  "schema": "runnel.actor-semantic-capture/1",\n',
            1,
        )
        cases = (
            raw[:-1],
            raw.replace(b"\n", b" \n", 1),
            duplicate,
            raw.replace(
                b'"schema": "runnel.actor-semantic-capture/1"', b'"schema": NaN'
            ),
            b"x" * (actor_transcript.MAX_CAPTURE_BYTES + 1),
        )
        for payload in cases:
            with self.subTest(payload=payload[:40]):
                with self.assertRaises(actor_transcript.ActorTranscriptError):
                    actor_transcript.parse_capture_bytes(payload)

    def test_path_reader_and_cli_create_new_binary_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_path = root / "capture.json"
            capture_path.write_bytes(actor_transcript.canonical_bytes(_capture()))
            transcript_path = root / "transcript.bin"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = actor_transcript.main(
                    [str(capture_path), "--transcript", str(transcript_path)]
                )
            self.assertEqual(result, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("sha256:", stdout.getvalue())
            self.assertEqual(
                transcript_path.read_bytes(),
                actor_transcript.serialize_transcript(_capture()),
            )

            before = transcript_path.read_bytes()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    actor_transcript.main(
                        [str(capture_path), "--output", str(transcript_path)]
                    ),
                    1,
                )
            self.assertEqual(transcript_path.read_bytes(), before)

            link = root / "capture-link.json"
            link.symlink_to(capture_path)
            with self.assertRaises(actor_transcript.ActorTranscriptError):
                actor_transcript.parse_capture_path(link)

            fifo = root / "capture.fifo"
            os.mkfifo(fifo)
            with self.assertRaises(actor_transcript.ActorTranscriptError):
                actor_transcript.parse_capture_path(fifo)

            oversized = root / "oversized.json"
            oversized.write_bytes(b"x" * (actor_transcript.MAX_CAPTURE_BYTES + 1))
            with self.assertRaises(actor_transcript.ActorTranscriptError):
                actor_transcript.parse_capture_path(oversized)


if __name__ == "__main__":
    unittest.main()
