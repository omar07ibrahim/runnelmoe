from __future__ import annotations

from dataclasses import dataclass, replace
import gc
from itertools import permutations
import random
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

from oracle import actor_race_history, actor_race_verify


class ReasonedDAGTests(unittest.TestCase):
    def test_deduplicates_edges_and_orders_reasons_deterministically(self) -> None:
        constraints = (
            actor_race_verify.EdgeConstraint(1, 3, "second"),
            actor_race_verify.EdgeConstraint(0, 3, "only"),
            actor_race_verify.EdgeConstraint(1, 3, "first"),
            actor_race_verify.EdgeConstraint(1, 3, "second"),
        )
        graph = actor_race_verify.ReasonedDAG.build(4, reversed(constraints))
        self.assertEqual(graph.topological_order, (0, 1, 2, 3))
        self.assertEqual(graph.reasons(1, 3), ("first", "second"))
        self.assertEqual(len(graph.edges), 2)

    def test_reverse_topological_bitsets_compute_transitive_reachability(self) -> None:
        graph = actor_race_verify.ReasonedDAG.build(
            5,
            (
                actor_race_verify.EdgeConstraint(0, 1, "left"),
                actor_race_verify.EdgeConstraint(0, 2, "right"),
                actor_race_verify.EdgeConstraint(1, 3, "join-left"),
                actor_race_verify.EdgeConstraint(2, 3, "join-right"),
                actor_race_verify.EdgeConstraint(3, 4, "tail"),
            ),
        )
        self.assertEqual(graph.topological_order, (0, 1, 2, 3, 4))
        self.assertTrue(graph.precedes(0, 4))
        self.assertTrue(graph.precedes(2, 4))
        self.assertFalse(graph.precedes(1, 2))
        self.assertFalse(graph.precedes(4, 4))
        self.assertEqual(graph.reachable[0], 0b1_1110)

    def test_cycle_diagnostic_is_deterministic_reasoned_and_short(self) -> None:
        constraints = [
            actor_race_verify.EdgeConstraint(
                node,
                (node + 1) % 12,
                f"edge-{node}",
            )
            for node in range(12)
        ]
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            r"cycle detected \(12 edges\): 0 -\[edge-0\]-> 1.* -> \.\.\.$",
        ):
            actor_race_verify.ReasonedDAG.build(12, reversed(constraints))

    def test_self_cycle_and_invalid_bounds_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            r"cycle detected \(1 edges\): 0 -\[self\]-> 0",
        ):
            actor_race_verify.ReasonedDAG.build(
                1, (actor_race_verify.EdgeConstraint(0, 0, "self"),)
            )
        with self.assertRaises(actor_race_verify.ActorRaceVerificationError):
            actor_race_verify.ReasonedDAG.build(
                1, (actor_race_verify.EdgeConstraint(0, 1, "outside"),)
            )
        with self.assertRaises(actor_race_verify.ActorRaceVerificationError):
            actor_race_verify.ReasonedDAG.build(
                1, (actor_race_verify.EdgeConstraint(0, 0, "café"),)
            )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "must be an integer",
        ):
            actor_race_verify.ReasonedDAG.build(
                1,
                (actor_race_verify.EdgeConstraint(False, 0, "boolean"),),
            )

    def test_allocation_and_reason_caps_fail_before_unbounded_growth(self) -> None:
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            rf"{actor_race_verify.MAX_DAG_NODES}-node limit",
        ):
            actor_race_verify.ReasonedDAG.build(
                actor_race_verify.MAX_DAG_NODES + 1,
                (),
            )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "eight-reason limit",
        ):
            actor_race_verify.ReasonedDAG.build(
                2,
                tuple(
                    actor_race_verify.EdgeConstraint(0, 1, f"reason-{index}")
                    for index in range(actor_race_verify.MAX_REASONS_PER_EDGE + 1)
                ),
            )

    def test_exact_graph_limit_and_distinct_input_caps(self) -> None:
        graph = actor_race_verify.ReasonedDAG.build(
            actor_race_verify.MAX_DAG_NODES,
            (
                actor_race_verify.EdgeConstraint(node, node + 1, "chain")
                for node in range(actor_race_verify.MAX_DAG_NODES - 1)
            ),
        )
        self.assertEqual(
            len(graph.topological_order),
            actor_race_verify.MAX_DAG_NODES,
        )
        self.assertEqual(
            graph.reachable[0].bit_count(),
            actor_race_verify.MAX_DAG_NODES - 1,
        )

        with mock.patch.object(actor_race_verify, "MAX_DAG_EDGES", 1):
            with self.assertRaisesRegex(
                actor_race_verify.ActorRaceVerificationError,
                "edge limit",
            ):
                actor_race_verify.ReasonedDAG.build(
                    3,
                    (
                        actor_race_verify.EdgeConstraint(0, 1, "first"),
                        actor_race_verify.EdgeConstraint(1, 2, "second"),
                    ),
                )

        with mock.patch.object(actor_race_verify, "MAX_EDGE_INPUTS", 2):
            with self.assertRaisesRegex(
                actor_race_verify.ActorRaceVerificationError,
                "constraint input limit",
            ):
                actor_race_verify.ReasonedDAG.build(
                    2,
                    (
                        actor_race_verify.EdgeConstraint(0, 1, "first"),
                        actor_race_verify.EdgeConstraint(0, 1, "second"),
                        actor_race_verify.EdgeConstraint(0, 1, "third"),
                    ),
                )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "1..128 bytes",
        ):
            actor_race_verify.ReasonedDAG.build(
                2,
                (
                    actor_race_verify.EdgeConstraint(
                        0,
                        1,
                        "x" * (actor_race_verify.MAX_REASON_BYTES + 1),
                    ),
                ),
            )


def _action(
    ordinal: int,
    producer: int,
    kind: str,
    submit_attempt: int | None = None,
    client_index: int | None = 7,
) -> actor_race_verify.StaticAction:
    return actor_race_verify.StaticAction(
        ordinal,
        producer,
        kind,
        submit_attempt,
        client_index,
    )


def _program(
    actions: tuple[actor_race_verify.StaticAction, ...],
) -> actor_race_verify.ActionProgram:
    kind_counts = tuple(
        sum(action.kind == kind for action in actions)
        for kind in actor_race_verify.CAPTURE_ACTION_KINDS
    )
    producer_counts = (
        sum(action.producer == 0 for action in actions),
        sum(action.producer == 1 for action in actions),
    )
    return actor_race_verify.ActionProgram(
        actions,
        kind_counts,
        producer_counts,
        "unit-fixture",
        "sha256:unit",
    )


@dataclass(frozen=True, slots=True)
class _ObservedAction:
    ordinal: int
    producer: int
    kind: str
    submit_attempt: int | None
    client_index: int | None
    invocation: int = 0
    response: int = 0


def _observe(action: actor_race_verify.StaticAction) -> _ObservedAction:
    return _ObservedAction(
        action.ordinal,
        action.producer,
        action.kind,
        action.submit_attempt,
        action.client_index,
    )


def _small_interval_order(
    actions: tuple[_ObservedAction, ...],
) -> actor_race_verify.ActionIntervalOrder:
    return actor_race_verify._build_action_interval_order(
        actions,
        len(actions) * 2,
        expected_action_count=len(actions),
    )


def _random_valid_history(
    random_source: random.Random,
    action_count: int,
) -> tuple[_ObservedAction, ...]:
    producers = tuple(random_source.randrange(2) for _ in range(action_count))
    producer_events: list[list[tuple[int, str]]] = [[], []]
    for ordinal, producer in enumerate(producers):
        producer_events[producer].extend(
            ((ordinal, "invoke"), (ordinal, "respond"))
        )

    positions = [0, 0]
    event_order: list[tuple[int, str]] = []
    while len(event_order) < action_count * 2:
        available = [
            producer
            for producer in (0, 1)
            if positions[producer] < len(producer_events[producer])
        ]
        producer = random_source.choice(available)
        event_order.append(producer_events[producer][positions[producer]])
        positions[producer] += 1

    counters = {
        event: counter
        for counter, event in enumerate(event_order, start=1)
    }
    return tuple(
        _ObservedAction(
            ordinal,
            producers[ordinal],
            "drain",
            None,
            ordinal,
            counters[(ordinal, "invoke")],
            counters[(ordinal, "respond")],
        )
        for ordinal in range(action_count)
    )


def _exhaustive_interval_graph(
    actions: tuple[_ObservedAction, ...],
) -> actor_race_verify.ReasonedDAG:
    constraints: list[actor_race_verify.EdgeConstraint] = []
    previous_by_producer: list[int | None] = [None, None]
    for ordinal, action in enumerate(actions):
        invocation_node = ordinal * 2
        response_node = invocation_node + 1
        constraints.append(
            actor_race_verify.EdgeConstraint(
                invocation_node,
                response_node,
                "action",
            )
        )
        previous = previous_by_producer[action.producer]
        if previous is not None:
            constraints.append(
                actor_race_verify.EdgeConstraint(
                    previous * 2 + 1,
                    invocation_node,
                    "program",
                )
            )
        previous_by_producer[action.producer] = ordinal

    for response_ordinal, response_action in enumerate(actions):
        for invocation_ordinal, invocation_action in enumerate(actions):
            if response_action.response < invocation_action.invocation:
                constraints.append(
                    actor_race_verify.EdgeConstraint(
                        response_ordinal * 2 + 1,
                        invocation_ordinal * 2,
                        "real-time",
                    )
                )
    return actor_race_verify.ReasonedDAG.build(len(actions) * 2, constraints)


def _exact_interval_repetition() -> SimpleNamespace:
    return _exact_submission_repetition()


_COMMAND_SENTINEL = actor_race_history.CommandWitness(False, 0, 0, 0)
_ACCEPTED_SENTINEL = actor_race_history.AcceptedWitness(
    False, 0, 0, 0, 0, 0
)
_CONTROL_SENTINEL = actor_race_history.ControlWitness(
    "none",
    False,
    0,
    0,
    "0000000000000000",
    "0000000000000000",
    None,
)
_PRIMARY_POP_SENTINEL = actor_race_history.PopWitness(
    "primary", False, 0, 0, 0, 0, None
)
_OPPORTUNISTIC_POP_SENTINEL = actor_race_history.PopWitness(
    "opportunistic_eof", False, 0, 0, 0, 0, None
)
_WAKE_SENTINEL = actor_race_history.WakeWitness(False, False, 0, 0)
_SATURATED_ERROR = actor_race_history.CapturedError(
    "resource_exhausted",
    "resource_exhausted",
    "request slot count",
    16,
    16,
)
_STALE_ERROR = actor_race_history.CapturedError(
    "request_not_found",
    "invalid_request",
    None,
    None,
    None,
)


def _exact_submission_repetition(
    accepted_attempts: frozenset[int] | None = None,
) -> SimpleNamespace:
    program = actor_race_verify.regenerate_authenticated_action_program()
    if accepted_attempts is None:
        accepted_attempts = frozenset(range(0, 64, 2))
    actions: list[actor_race_history.Action] = []
    request_id = 0
    control_generations = [0] * 16
    endpoint_generations = [0] * 16
    for static in program.actions:
        command = _COMMAND_SENTINEL
        accepted = _ACCEPTED_SENTINEL
        result = "irrelevant"
        error = None
        action_request_id = None
        attempt = static.submit_attempt
        if static.kind == "submit" and attempt is not None:
            if attempt < actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT:
                command = actor_race_history.CommandWitness(
                    True,
                    attempt % 8,
                    attempt // 8 + 1,
                    attempt + 1,
                )
                if attempt in accepted_attempts:
                    result = "submit_accepted"
                    request_id += 1
                    action_request_id = request_id
                    control_slot = (request_id - 1) % 16
                    endpoint_slot = (request_id + 4) % 16
                    control_generations[control_slot] += 1
                    endpoint_generations[endpoint_slot] += 1
                    accepted = actor_race_history.AcceptedWitness(
                        True,
                        request_id,
                        control_slot,
                        control_generations[control_slot],
                        endpoint_slot,
                        endpoint_generations[endpoint_slot],
                    )
                else:
                    result = "error"
                    error = _SATURATED_ERROR
            else:
                result = "submit_offer_exhausted"

        actions.append(
            actor_race_history.Action(
                ordinal=static.ordinal,
                producer=static.producer,
                kind=static.kind,
                submit_attempt=static.submit_attempt,
                client_index=static.client_index,
                invocation=static.ordinal * 2 + 1,
                response=static.ordinal * 2 + 2,
                result=result,
                error=error,
                request_id=action_request_id,
                output=None,
                command=command,
                accepted=accepted,
                control=_CONTROL_SENTINEL,
                primary_pop=_PRIMARY_POP_SENTINEL,
                opportunistic_eof_pop=_OPPORTUNISTIC_POP_SENTINEL,
                cached_eof=False,
                wake=_WAKE_SENTINEL,
            )
        )

    return SimpleNamespace(
        actions=tuple(actions),
        diagnostics=SimpleNamespace(
            action_counter_final=(
                actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL
            )
        ),
        shutdown=actor_race_history.Shutdown(
            accepted_submissions=request_id,
            discarded_output_events=0,
            engine_steps=1,
            rejected_submissions=(
                actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT - request_id
            ),
            released_request_bytes=0,
            remaining_shared_bytes=0,
            shutdown_cancellations=0,
            terminated_requests=0,
        ),
    )


def _exact_target_repetition() -> SimpleNamespace:
    repetition = _exact_submission_repetition()
    actions = list(repetition.actions)
    accepted: dict[int, actor_race_history.AcceptedWitness] = {}
    receiver_state = ["absent"] * actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT
    cancel_requested: set[int] = set()
    cached_eof_added = False

    for ordinal, action in enumerate(actions):
        if action.kind == "submit" and action.result == "submit_accepted":
            assert action.client_index is not None
            accepted[action.client_index] = action.accepted
            receiver_state[action.client_index] = "owned"
            continue
        if action.kind not in ("cancel", "receiver_drop", "drain"):
            continue

        assert action.client_index is not None
        client_index = action.client_index
        witness = accepted.get(client_index)
        if action.kind == "cancel":
            if witness is None:
                actions[ordinal] = replace(
                    action,
                    result="target_unavailable",
                )
                continue
            base_word = witness.control_generation << 3
            if client_index in cancel_requested:
                result = "cancel_already_requested"
                loaded_word = base_word | 1
                resulting_word = loaded_word
                disposition = "already_requested"
            else:
                result = "cancel_requested"
                loaded_word = base_word
                resulting_word = base_word | 1
                disposition = "requested"
                cancel_requested.add(client_index)
            actions[ordinal] = replace(
                action,
                result=result,
                request_id=witness.request_id,
                control=actor_race_history.ControlWitness(
                    "cancel",
                    True,
                    witness.control_slot,
                    witness.control_generation,
                    f"{loaded_word:016x}",
                    f"{resulting_word:016x}",
                    disposition,
                ),
            )
            continue

        state = receiver_state[client_index]
        if state == "absent":
            actions[ordinal] = replace(
                action,
                result="target_unavailable",
            )
            continue
        assert witness is not None
        if state == "consumed":
            actions[ordinal] = replace(
                action,
                result="target_unavailable",
                request_id=witness.request_id,
            )
            continue

        if action.kind == "receiver_drop":
            base_word = witness.control_generation << 3
            actions[ordinal] = replace(
                action,
                result="receiver_dropped",
                request_id=witness.request_id,
                control=actor_race_history.ControlWitness(
                    "disconnect",
                    True,
                    witness.control_slot,
                    witness.control_generation,
                    f"{base_word:016x}",
                    f"{base_word | 3:016x}",
                    "requested",
                ),
            )
            receiver_state[client_index] = "consumed"
            continue

        if not cached_eof_added:
            actions[ordinal] = replace(
                action,
                result="drain_eof",
                request_id=witness.request_id,
                cached_eof=True,
            )
            cached_eof_added = True
        else:
            actions[ordinal] = replace(
                action,
                result="drain_empty",
                request_id=witness.request_id,
                primary_pop=actor_race_history.PopWitness(
                    "primary",
                    True,
                    witness.endpoint_slot,
                    witness.endpoint_generation,
                    0,
                    0,
                    None,
                ),
            )

    assert cached_eof_added
    return SimpleNamespace(
        actions=tuple(actions),
        diagnostics=repetition.diagnostics,
        shutdown=repetition.shutdown,
    )


def _probe_snapshot(outstanding_requests: int) -> actor_race_history.ProbeSnapshot:
    return actor_race_history.ProbeSnapshot(
        command_in_flight=0,
        command_ready=0,
        command_reserved=0,
        command_responded=0,
        dirty=False,
        engine_steps=1,
        outstanding_requests=outstanding_requests,
        owner_done=False,
        park_epoch=0,
        parked=True,
        pump_entries=1,
        pump_hold_observed=0,
        pump_hold_released=0,
        pump_hold_requested=0,
        pump_in_flight=False,
        request_bytes=outstanding_requests,
        shared_bytes=1,
    )


def _exact_control_repetition() -> SimpleNamespace:
    """Build an exact synthetic word history with one real control rebind."""

    accepted_attempts = frozenset((*range(4, 20), 50))
    repetition = _exact_submission_repetition(accepted_attempts)
    actions = list(repetition.actions)
    accepted: dict[int, actor_race_history.AcceptedWitness] = {}
    current_by_slot: dict[int, actor_race_history.AcceptedWitness] = {}
    flags: dict[tuple[int, int], int] = {}
    receiver_state = ["absent"] * actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT
    cached_eof_added = False

    for ordinal, action in enumerate(actions):
        if action.kind == "submit" and action.result == "submit_accepted":
            assert action.client_index is not None
            accepted[action.client_index] = action.accepted
            current_by_slot[action.accepted.control_slot] = action.accepted
            flags[
                (
                    action.accepted.control_slot,
                    action.accepted.control_generation,
                )
            ] = 0
            receiver_state[action.client_index] = "owned"
            continue
        if action.kind not in ("cancel", "receiver_drop", "drain"):
            continue

        assert action.client_index is not None
        client_index = action.client_index
        witness = accepted.get(client_index)
        if action.kind == "cancel":
            if witness is None:
                actions[ordinal] = replace(action, result="target_unavailable")
                continue
            current = current_by_slot[witness.control_slot]
            current_key = (current.control_slot, current.control_generation)
            loaded_flags = flags[current_key]
            loaded_word = current.control_generation << 3 | loaded_flags
            if current.control_generation > witness.control_generation:
                result = "error"
                error = _STALE_ERROR
                resulting_word = loaded_word
                disposition = None
            elif loaded_flags & 4:
                result = "cancel_already_terminal"
                error = None
                resulting_word = loaded_word
                disposition = "already_terminal"
            elif loaded_flags & 1:
                result = "cancel_already_requested"
                error = None
                resulting_word = loaded_word
                disposition = "already_requested"
            else:
                result = "cancel_requested"
                error = None
                resulting_word = loaded_word | 1
                disposition = "requested"
                flags[current_key] = loaded_flags | 1
            actions[ordinal] = replace(
                action,
                result=result,
                error=error,
                request_id=witness.request_id,
                control=actor_race_history.ControlWitness(
                    "cancel",
                    True,
                    witness.control_slot,
                    witness.control_generation,
                    f"{loaded_word:016x}",
                    f"{resulting_word:016x}",
                    disposition,
                ),
            )
            continue

        state = receiver_state[client_index]
        if state == "absent":
            actions[ordinal] = replace(action, result="target_unavailable")
            continue
        assert witness is not None
        if state == "consumed":
            actions[ordinal] = replace(
                action,
                result="target_unavailable",
                request_id=witness.request_id,
            )
            continue

        if action.kind == "receiver_drop":
            key = (witness.control_slot, witness.control_generation)
            loaded_flags = flags[key]
            loaded_word = witness.control_generation << 3 | loaded_flags
            if loaded_flags & 4:
                disposition = "already_terminal"
            elif loaded_flags & 2:
                disposition = "already_requested"
            else:
                disposition = "requested"
            resulting_flags = loaded_flags | 3
            resulting_word = witness.control_generation << 3 | resulting_flags
            actions[ordinal] = replace(
                action,
                result="receiver_dropped",
                request_id=witness.request_id,
                control=actor_race_history.ControlWitness(
                    "disconnect",
                    True,
                    witness.control_slot,
                    witness.control_generation,
                    f"{loaded_word:016x}",
                    f"{resulting_word:016x}",
                    disposition,
                ),
            )
            # The synthetic engine terminalizes immediately after the reached
            # destructor boundary, so all later readers observe the tombstone.
            flags[key] = resulting_flags | 4
            receiver_state[client_index] = "consumed"
            continue

        if not cached_eof_added:
            actions[ordinal] = replace(
                action,
                result="drain_eof",
                request_id=witness.request_id,
                cached_eof=True,
            )
            cached_eof_added = True
        else:
            actions[ordinal] = replace(
                action,
                result="drain_empty",
                request_id=witness.request_id,
                primary_pop=actor_race_history.PopWitness(
                    "primary",
                    True,
                    witness.endpoint_slot,
                    witness.endpoint_generation,
                    0,
                    0,
                    None,
                ),
            )

    assert cached_eof_added
    live_clients = tuple(
        sorted(
            client_index
            for client_index in accepted
            if receiver_state[client_index] == "owned"
        )
    )
    for client_index in live_clients:
        witness = accepted[client_index]
        key = (witness.control_slot, witness.control_generation)
        flags[key] |= 4

    authorities: list[actor_race_history.CleanupAuthority] = []
    for ordinal, client_index in enumerate(sorted(accepted)):
        witness = accepted[client_index]
        current = current_by_slot[witness.control_slot]
        key = (current.control_slot, current.control_generation)
        word = current.control_generation << 3 | flags[key]
        if current.control_generation > witness.control_generation:
            error = _STALE_ERROR
            disposition = None
        else:
            error = None
            disposition = "already_terminal"
        authorities.append(
            actor_race_history.CleanupAuthority(
                ordinal * 2 + 1,
                ordinal * 2 + 2,
                client_index,
                witness.request_id,
                error,
                actor_race_history.ControlWitness(
                    "cancel",
                    True,
                    witness.control_slot,
                    witness.control_generation,
                    f"{word:016x}",
                    f"{word:016x}",
                    disposition,
                ),
            )
        )

    receivers: list[actor_race_history.CleanupReceiver] = []
    for receiver_ordinal, client_index in enumerate(live_clients):
        witness = accepted[client_index]
        sequence_ordinal = len(authorities) + receiver_ordinal
        receivers.append(
            actor_race_history.CleanupReceiver(
                sequence_ordinal * 2 + 1,
                sequence_ordinal * 2 + 2,
                client_index,
                witness.request_id,
                actor_race_history.Terminal(
                    witness.request_id,
                    "cancelled",
                    0,
                    0,
                ),
                (),
                True,
                _probe_snapshot(len(live_clients) - receiver_ordinal - 1),
            )
        )

    return SimpleNamespace(
        actions=tuple(actions),
        cleanup_authorities=tuple(authorities),
        cleanup_receivers=tuple(receivers),
        diagnostics=SimpleNamespace(
            action_counter_final=(
                actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL
            ),
            cleanup_counter_final=2 * (len(authorities) + len(receivers)),
        ),
        pre_cleanup=_probe_snapshot(len(receivers)),
        shutdown=repetition.shutdown,
    )


def _recorder_status(observation_count: int) -> actor_race_history.RecorderStatus:
    return actor_race_history.RecorderStatus(
        actor_race_history.OBSERVATION_LIMIT,
        observation_count,
        actor_race_history.OBSERVATION_LIMIT,
        False,
        False,
    )


def _held_probe_snapshot(
    outstanding_requests: int,
    reap_progress: int,
) -> actor_race_history.ProbeSnapshot:
    return replace(
        _probe_snapshot(outstanding_requests),
        engine_steps=1 + reap_progress,
        park_epoch=reap_progress,
        pump_entries=1 + reap_progress,
        pump_hold_observed=1,
        pump_hold_released=1,
        pump_hold_requested=1,
    )


def _exact_endpoint_repetition() -> SimpleNamespace:
    """Extend the exact control fixture with grounded endpoint semantics."""

    repetition = _exact_control_repetition()
    actions = list(repetition.actions)
    accepted = {
        action.client_index: action.accepted
        for action in actions
        if action.result == "submit_accepted"
    }
    cached_ordinal = next(
        action.ordinal for action in actions if action.cached_eof
    )
    cached_action = actions[cached_ordinal]
    assert cached_action.client_index is not None
    identity = accepted[cached_action.client_index]
    actions[cached_ordinal] = replace(
        cached_action,
        result="drain_empty",
        cached_eof=False,
        primary_pop=actor_race_history.PopWitness(
            "primary",
            True,
            identity.endpoint_slot,
            identity.endpoint_generation,
            0,
            0,
            None,
        ),
    )

    descriptors = actor_race_verify.scheduler.build_descriptors()
    terminals = {
        witness.request_id: actor_race_history.Terminal(
            witness.request_id,
            "cancelled",
            len(descriptors[client_index]["prompt"]) - 1,
            0,
        )
        for client_index, witness in accepted.items()
    }
    receivers = tuple(
        replace(
            receiver,
            terminal=terminals[receiver.request_id],
            post_drop_quiescent=_held_probe_snapshot(
                len(repetition.cleanup_receivers) - ordinal - 1,
                ordinal + 1,
            ),
        )
        for ordinal, receiver in enumerate(repetition.cleanup_receivers)
    )
    observations = tuple(
        actor_race_history.Observation(
            "terminal",
            request_id,
            None,
            None,
            terminal.outcome,
            terminal.committed_positions,
            terminal.emitted_tokens,
        )
        for request_id, terminal in sorted(terminals.items())
    ) + tuple(
        actor_race_history.Observation(
            "output_eof",
            request_id,
            None,
            None,
            None,
            None,
            None,
        )
        for request_id in sorted(terminals)
    )
    return SimpleNamespace(
        actions=tuple(actions),
        cleanup_authorities=repetition.cleanup_authorities,
        cleanup_receivers=receivers,
        diagnostics=SimpleNamespace(
            action_counter_final=repetition.diagnostics.action_counter_final,
            cleanup_counter_final=repetition.diagnostics.cleanup_counter_final,
            recorder_initial=_recorder_status(0),
            recorder_final=_recorder_status(len(observations)),
        ),
        observations=observations,
        pre_cleanup=repetition.pre_cleanup,
        pre_shutdown=_held_probe_snapshot(
            0,
            len(repetition.cleanup_receivers),
        ),
        shutdown=repetition.shutdown,
    )


def _reached_overlap_boundary(action: actor_race_history.Action) -> str | None:
    if action.command.boundary:
        return "command"
    if action.control.boundary:
        return "control"
    if action.primary_pop.boundary:
        return "primary_endpoint"
    if action.opportunistic_eof_pop.boundary:
        return "opportunistic_endpoint"
    if action.wake.boundary:
        return "wake"
    return None


def _exact_wake_shutdown_repetition(
    repetition_index: int = 0,
) -> actor_race_history.Repetition:
    """Close the endpoint fixture with all wake and owner lifecycle evidence."""

    endpoint = _exact_endpoint_repetition()
    actions = list(endpoint.actions)
    wake_epoch = 1
    for ordinal, action in enumerate(actions):
        if action.kind != "wake":
            continue
        actions[ordinal] = replace(
            action,
            result="wake_signaled",
            wake=actor_race_history.WakeWitness(
                True,
                bool(wake_epoch % 2),
                wake_epoch,
                wake_epoch + 1,
            ),
        )
        wake_epoch += 1
    assert wake_epoch == actor_race_verify.EXPECTED_KIND_COUNTS[4] + 1

    overlap: tuple[int, str, int, str] | None = None
    for left_ordinal in range(len(actions) - 1):
        right_ordinal = left_ordinal + 1
        left = actions[left_ordinal]
        right = actions[right_ordinal]
        left_boundary = _reached_overlap_boundary(left)
        right_boundary = _reached_overlap_boundary(right)
        if (
            left.producer != right.producer
            and left_boundary is not None
            and right_boundary is not None
        ):
            original_left_response = left.response
            original_right_invocation = right.invocation
            actions[left_ordinal] = replace(
                left,
                response=original_right_invocation,
            )
            actions[right_ordinal] = replace(
                right,
                invocation=original_left_response,
            )
            overlap = (
                left_ordinal,
                left_boundary,
                right_ordinal,
                right_boundary,
            )
            break
    assert overlap is not None

    live = len(endpoint.cleanup_receivers)
    pre_cleanup = replace(endpoint.pre_cleanup, park_epoch=wake_epoch)
    receivers = tuple(
        replace(
            receiver,
            post_drop_quiescent=replace(
                receiver.post_drop_quiescent,
                engine_steps=ordinal + 2,
                park_epoch=wake_epoch + ordinal + 1,
                pump_entries=ordinal + 2,
            ),
        )
        for ordinal, receiver in enumerate(endpoint.cleanup_receivers)
    )
    pre_shutdown = replace(
        endpoint.pre_shutdown,
        engine_steps=live + 1,
        park_epoch=wake_epoch + live,
        pump_entries=live + 1,
    )
    post_shutdown = replace(
        pre_shutdown,
        dirty=True,
        owner_done=True,
        parked=False,
        pump_entries=pre_shutdown.pump_entries + 1,
        shared_bytes=0,
    )
    shutdown = replace(endpoint.shutdown, engine_steps=post_shutdown.engine_steps)
    diagnostics = actor_race_history.Diagnostics(
        endpoint.diagnostics.action_counter_final,
        True,
        endpoint.diagnostics.cleanup_counter_final,
        post_shutdown.engine_steps - 1,
        post_shutdown.engine_steps,
        post_shutdown.pump_entries,
        1,
        1,
        overlap,
        post_shutdown.pump_entries - 1,
        endpoint.diagnostics.recorder_final,
        endpoint.diagnostics.recorder_initial,
    )
    return actor_race_history.Repetition(
        tuple(actions),
        endpoint.cleanup_authorities,
        receivers,
        diagnostics,
        endpoint.observations,
        post_shutdown,
        pre_cleanup,
        pre_shutdown,
        repetition_index,
        shutdown,
    )


def _exact_cleanup_requested_endpoint_repetition() -> SimpleNamespace:
    """Ground the one-request cleanup-Requested control fixture."""

    repetition = _live_cleanup_requested_repetition()
    receiver = repetition.cleanup_receivers[0]
    receiver = replace(
        receiver,
        post_drop_quiescent=_held_probe_snapshot(0, 1),
    )
    observations = (
        actor_race_history.Observation(
            "terminal",
            receiver.request_id,
            None,
            None,
            receiver.terminal.outcome,
            receiver.terminal.committed_positions,
            receiver.terminal.emitted_tokens,
        ),
        actor_race_history.Observation(
            "output_eof",
            receiver.request_id,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    return SimpleNamespace(
        actions=repetition.actions,
        cleanup_authorities=repetition.cleanup_authorities,
        cleanup_receivers=(receiver,),
        diagnostics=SimpleNamespace(
            action_counter_final=repetition.diagnostics.action_counter_final,
            cleanup_counter_final=repetition.diagnostics.cleanup_counter_final,
            recorder_initial=_recorder_status(0),
            recorder_final=_recorder_status(len(observations)),
        ),
        observations=observations,
        pre_cleanup=repetition.pre_cleanup,
        pre_shutdown=_held_probe_snapshot(0, 1),
        shutdown=repetition.shutdown,
    )


def _replace_endpoint_repetition(
    repetition: SimpleNamespace,
    **changes: object,
) -> SimpleNamespace:
    values = {
        "actions": repetition.actions,
        "cleanup_authorities": repetition.cleanup_authorities,
        "cleanup_receivers": repetition.cleanup_receivers,
        "diagnostics": repetition.diagnostics,
        "observations": repetition.observations,
        "pre_cleanup": repetition.pre_cleanup,
        "pre_shutdown": repetition.pre_shutdown,
        "shutdown": repetition.shutdown,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _with_endpoint_publications(
    repetition: SimpleNamespace,
    client_index: int,
    outputs: tuple[actor_race_history.Output, ...],
    cleanup_outputs: tuple[actor_race_history.Output, ...],
    *,
    terminal_outcome: str = "cancelled",
) -> SimpleNamespace:
    submit = next(
        action
        for action in repetition.actions
        if action.result == "submit_accepted" and action.client_index == client_index
    )
    request_id = submit.request_id
    assert request_id is not None
    descriptor = actor_race_verify.scheduler.build_descriptors()[client_index]
    terminal = actor_race_history.Terminal(
        request_id,
        terminal_outcome,
        len(descriptor["prompt"]) - 1 + len(outputs),
        len(outputs),
    )
    output_observations = tuple(
        actor_race_history.Observation(
            "output",
            output.request_id,
            output.output_index,
            output.token_id,
            None,
            None,
            None,
        )
        for output in outputs
    )
    later_observations = tuple(
        replace(
            observation,
            outcome=terminal.outcome,
            committed_positions=terminal.committed_positions,
            emitted_tokens=terminal.emitted_tokens,
        )
        if observation.kind == "terminal" and observation.request_id == request_id
        else observation
        for observation in repetition.observations
    )
    observations = output_observations + later_observations
    receivers = tuple(
        replace(
            receiver,
            terminal=terminal,
            outputs=cleanup_outputs,
        )
        if receiver.client_index == client_index
        else receiver
        for receiver in repetition.cleanup_receivers
    )
    diagnostics = SimpleNamespace(
        action_counter_final=repetition.diagnostics.action_counter_final,
        cleanup_counter_final=repetition.diagnostics.cleanup_counter_final,
        recorder_initial=repetition.diagnostics.recorder_initial,
        recorder_final=_recorder_status(len(observations)),
    )
    return _replace_endpoint_repetition(
        repetition,
        cleanup_receivers=receivers,
        diagnostics=diagnostics,
        observations=observations,
    )


def _live_cleanup_requested_repetition() -> SimpleNamespace:
    """Keep one submit unpublished until all of its script cancels miss."""

    client_index = 16
    repetition = _exact_submission_repetition(frozenset((client_index,)))
    actions = list(repetition.actions)
    submit_ordinal = next(
        action.ordinal
        for action in actions
        if action.kind == "submit" and action.client_index == client_index
    )
    submit = actions[submit_ordinal]
    release_after = max(
        action.ordinal
        for action in actions
        if action.kind == "cancel" and action.client_index == client_index
    )

    boundaries: dict[tuple[int, str], int] = {}
    counter = 1

    def issue(ordinal: int, boundary: str) -> None:
        nonlocal counter
        boundaries[ordinal, boundary] = counter
        counter += 1

    for action in actions[:submit_ordinal]:
        issue(action.ordinal, "invoke")
        issue(action.ordinal, "respond")
    issue(submit_ordinal, "invoke")
    for action in actions[submit_ordinal + 1 : release_after + 1]:
        if action.producer != submit.producer:
            issue(action.ordinal, "invoke")
            issue(action.ordinal, "respond")
    issue(submit_ordinal, "respond")
    for action in actions[submit_ordinal + 1 :]:
        if (action.ordinal, "invoke") not in boundaries:
            issue(action.ordinal, "invoke")
            issue(action.ordinal, "respond")
    assert counter == actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL + 1

    actions = [
        replace(
            action,
            invocation=boundaries[action.ordinal, "invoke"],
            response=boundaries[action.ordinal, "respond"],
        )
        for action in actions
    ]
    in_range_submits = sorted(
        (
            action
            for action in actions
            if action.kind == "submit"
            and action.submit_attempt is not None
            and action.submit_attempt
            < actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT
        ),
        key=lambda action: action.invocation,
    )
    ready_by_ordinal = {
        action.ordinal: ready
        for ready, action in enumerate(in_range_submits, start=1)
    }
    actions = [
        replace(
            action,
            command=replace(
                action.command,
                ready_sequence=ready_by_ordinal[action.ordinal],
            ),
        )
        if action.ordinal in ready_by_ordinal
        else action
        for action in actions
    ]

    submit = actions[submit_ordinal]
    identity = submit.accepted
    for ordinal, action in enumerate(actions):
        if action.kind not in ("cancel", "receiver_drop", "drain"):
            continue
        if (
            action.client_index == client_index
            and submit.response < action.invocation
        ):
            assert action.kind == "drain"
            actions[ordinal] = replace(
                action,
                result="drain_empty",
                request_id=identity.request_id,
                primary_pop=actor_race_history.PopWitness(
                    "primary",
                    True,
                    identity.endpoint_slot,
                    identity.endpoint_generation,
                    0,
                    0,
                    None,
                ),
            )
        else:
            actions[ordinal] = replace(action, result="target_unavailable")

    base_word = identity.control_generation << 3
    authority = actor_race_history.CleanupAuthority(
        1,
        2,
        client_index,
        identity.request_id,
        None,
        actor_race_history.ControlWitness(
            "cancel",
            True,
            identity.control_slot,
            identity.control_generation,
            f"{base_word:016x}",
            f"{base_word | 1:016x}",
            "requested",
        ),
    )
    receiver = actor_race_history.CleanupReceiver(
        3,
        4,
        client_index,
        identity.request_id,
        actor_race_history.Terminal(identity.request_id, "cancelled", 0, 0),
        (),
        True,
        _probe_snapshot(0),
    )
    return SimpleNamespace(
        actions=tuple(actions),
        cleanup_authorities=(authority,),
        cleanup_receivers=(receiver,),
        diagnostics=SimpleNamespace(
            action_counter_final=actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL,
            cleanup_counter_final=4,
        ),
        pre_cleanup=_probe_snapshot(1),
        shutdown=repetition.shutdown,
    )


def _submit_by_attempt(repetition: SimpleNamespace, attempt: int) -> int:
    return next(
        action.ordinal
        for action in repetition.actions
        if action.kind == "submit" and action.submit_attempt == attempt
    )


def _replace_repetition_action(
    repetition: SimpleNamespace,
    ordinal: int,
    action: actor_race_history.Action,
) -> SimpleNamespace:
    actions = list(repetition.actions)
    actions[ordinal] = action
    return SimpleNamespace(
        actions=tuple(actions),
        diagnostics=repetition.diagnostics,
        shutdown=repetition.shutdown,
    )


def _replace_control_repetition(
    repetition: SimpleNamespace,
    **changes: object,
) -> SimpleNamespace:
    values = {
        "actions": repetition.actions,
        "cleanup_authorities": repetition.cleanup_authorities,
        "cleanup_receivers": repetition.cleanup_receivers,
        "diagnostics": repetition.diagnostics,
        "pre_cleanup": repetition.pre_cleanup,
        "shutdown": repetition.shutdown,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _replace_control_action(
    repetition: SimpleNamespace,
    ordinal: int,
    action: actor_race_history.Action,
) -> SimpleNamespace:
    actions = list(repetition.actions)
    actions[ordinal] = action
    return _replace_control_repetition(repetition, actions=tuple(actions))


def _replace_cleanup_authority(
    repetition: SimpleNamespace,
    ordinal: int,
    authority: actor_race_history.CleanupAuthority,
) -> SimpleNamespace:
    authorities = list(repetition.cleanup_authorities)
    authorities[ordinal] = authority
    return _replace_control_repetition(
        repetition,
        cleanup_authorities=tuple(authorities),
    )


class TargetAccessOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repetition = _exact_target_repetition()
        cls.order = actor_race_verify.build_target_access_order(cls.repetition)

    def _replace_target(
        self,
        target: actor_race_verify.TargetActionEvent,
        **changes: object,
    ) -> SimpleNamespace:
        action = self.repetition.actions[target.action_ordinal]
        return _replace_repetition_action(
            self.repetition,
            target.action_ordinal,
            replace(action, **changes),
        )

    def _one_target(
        self,
        *,
        kind: str | None = None,
        resolution: str | None = None,
        event_kind: str | None = None,
        identity: bool | None = None,
    ) -> actor_race_verify.TargetActionEvent:
        return next(
            target
            for target in self.order.targets.by_action
            if target is not None
            and (kind is None or target.action_kind == kind)
            and (resolution is None or target.resolution == resolution)
            and (event_kind is None or target.event.kind == event_kind)
            and (identity is None or (target.identity is not None) == identity)
        )

    def test_exact_immutable_identity_projection_and_event_budget(self) -> None:
        order = self.order
        identities = order.targets.identities
        self.assertEqual(
            len(order.targets.target_events),
            actor_race_verify.EXPECTED_TARGET_ACTION_COUNT,
        )
        self.assertEqual(actor_race_verify.EXPECTED_TARGET_ACTION_COUNT, 679)
        self.assertEqual(
            order.graph.node_count,
            order.submission_order.graph.node_count
            + actor_race_verify.EXPECTED_TARGET_ACTION_COUNT,
        )
        self.assertLessEqual(
            order.graph.node_count,
            actor_race_verify.MAX_PROTOCOL_NODES,
        )
        self.assertEqual(
            len({event.node for event in order.targets.target_events}),
            actor_race_verify.EXPECTED_TARGET_ACTION_COUNT,
        )
        self.assertEqual(
            sum(target is not None for target in order.targets.by_action),
            actor_race_verify.EXPECTED_TARGET_ACTION_COUNT,
        )
        self.assertEqual(len(identities.by_client_index), 64)
        self.assertEqual(len(identities.by_request_id), 32)
        self.assertEqual(
            tuple(identity.request_id for identity in identities.by_request_id),
            tuple(range(1, 33)),
        )
        for identity in identities.by_request_id:
            self.assertIs(
                identities.by_client_index[identity.client_index],
                identity,
            )
            self.assertIs(
                identities.by_control_slot[identity.control_slot][
                    identity.control_generation - 1
                ],
                identity,
            )
            self.assertIs(
                identities.by_endpoint_slot[identity.endpoint_slot][
                    identity.endpoint_generation - 1
                ],
                identity,
            )
        for client_index, targets in enumerate(
            order.targets.by_client_index
        ):
            self.assertEqual(
                tuple(target.action_ordinal for target in targets),
                tuple(
                    target.action_ordinal
                    for target in order.targets.by_action
                    if target is not None and target.client_index == client_index
                ),
            )

        event_kinds = {event.kind for event in order.targets.target_events}
        self.assertEqual(
            event_kinds,
            {
                "TargetLookup",
                "ControlCancel",
                "ControlDisconnect",
                "PrimaryEndpointPop",
                "CachedEofRead",
            },
        )

    def test_main_event_is_distinct_and_has_only_sound_publication_edges(self) -> None:
        order = self.order
        absent = self._one_target(resolution="absent", identity=True)
        found = self._one_target(resolution="cancel_authority")
        consumed = self._one_target(resolution="receiver_consumed")
        for target in (absent, found, consumed):
            endpoints = order.submission_order.interval_order.endpoints.by_action[
                target.action_ordinal
            ]
            self.assertNotIn(
                target.event.node,
                (endpoints.invocation.node, endpoints.response.node),
            )
            self.assertEqual(
                order.graph.reasons(endpoints.invocation.node, target.event.node),
                ("target-action-invocation-before-main-event",),
            )
            self.assertEqual(
                order.graph.reasons(target.event.node, endpoints.response.node),
                ("target-main-event-before-action-response",),
            )

        assert absent.identity is not None
        absent_publish = absent.identity.submission.registry_publish
        assert absent_publish is not None
        self.assertEqual(absent.event.kind, "TargetLookup")
        self.assertEqual(
            order.graph.reasons(absent.event.node, absent_publish.node),
            ("absent-target-lookup-before-registry-publication",),
        )

        assert found.identity is not None
        found_publish = found.identity.submission.registry_publish
        assert found_publish is not None
        self.assertEqual(found.event.kind, "ControlCancel")
        self.assertEqual(
            order.graph.reasons(found_publish.node, found.event.node),
            ("accepted-registry-publication-before-target-main-event",),
        )
        # A found lookup does not add the stronger publication-before-Invoke
        # edge; the object boundary may follow a publication that raced after
        # the action invocation.
        found_invoke = order.submission_order.interval_order.endpoints.by_action[
            found.action_ordinal
        ].invocation
        self.assertEqual(order.graph.reasons(found_publish.node, found_invoke.node), ())

        assert consumed.prior_receiver_drop_action is not None
        prior_response = order.submission_order.interval_order.endpoints.by_action[
            consumed.prior_receiver_drop_action
        ].response
        self.assertEqual(
            order.graph.reasons(prior_response.node, consumed.event.node),
            ("receiver-consumption-before-unavailable-lookup",),
        )

    def test_absent_and_found_results_choose_opposite_publication_order(self) -> None:
        absent = self._one_target(
            kind="cancel",
            resolution="absent",
            identity=True,
        )
        assert absent.identity is not None
        witness = absent.identity
        base_word = witness.control_generation << 3
        forged_found = self._replace_target(
            absent,
            result="cancel_requested",
            request_id=witness.request_id,
            control=actor_race_history.ControlWitness(
                "cancel",
                True,
                witness.control_slot,
                witness.control_generation,
                f"{base_word:016x}",
                f"{base_word | 1:016x}",
                "requested",
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            actor_race_verify.build_target_access_order(forged_found)

        found = self._one_target(kind="cancel", resolution="cancel_authority")
        forged_absent = self._replace_target(
            found,
            result="target_unavailable",
            request_id=None,
            error=None,
            control=_CONTROL_SENTINEL,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            actor_race_verify.build_target_access_order(forged_absent)

    def test_receiver_ownership_and_retained_identity_fail_closed(self) -> None:
        consumed = self._one_target(
            kind="drain",
            resolution="receiver_consumed",
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "request_id must be an integer",
        ):
            actor_race_verify.build_target_access_order(
                self._replace_target(consumed, request_id=None)
            )

        owned_drop = next(
            target
            for target in self.order.targets.by_action
            if target is not None
            and target.action_kind == "receiver_drop"
            and target.resolution == "receiver_owned"
            and any(
                later.action_ordinal > target.action_ordinal
                and later.action_kind == "receiver_drop"
                and later.resolution == "receiver_consumed"
                for later in self.order.targets.by_client_index[
                    target.client_index
                ]
            )
        )
        assert owned_drop.identity is not None
        later_consumed = next(
            target
            for target in self.order.targets.by_client_index[
                owned_drop.client_index
            ]
            if target.action_ordinal > owned_drop.action_ordinal
            and target.action_kind == "receiver_drop"
            and target.resolution == "receiver_consumed"
        )
        identity = owned_drop.identity
        base_word = identity.control_generation << 3
        forged_second_drop = self._replace_target(
            later_consumed,
            result="receiver_dropped",
            control=actor_race_history.ControlWitness(
                "disconnect",
                True,
                identity.control_slot,
                identity.control_generation,
                f"{base_word:016x}",
                f"{base_word | 3:016x}",
                "requested",
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "called a receiver that is consumed",
        ):
            actor_race_verify.build_target_access_order(forged_second_drop)

        absent_receiver = self._one_target(
            kind="drain",
            resolution="absent",
            identity=True,
        )
        assert absent_receiver.identity is not None
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "absent lookup retained a request ID",
        ):
            actor_race_verify.build_target_access_order(
                self._replace_target(
                    absent_receiver,
                    request_id=absent_receiver.identity.request_id,
                )
            )

    def test_target_identity_and_exact_types_fail_closed(self) -> None:
        class IntSubclass(int):
            pass

        found = self._one_target(kind="cancel", resolution="cancel_authority")
        action = self.repetition.actions[found.action_ordinal]
        assert found.identity is not None
        cases = (
            (
                "request_id must be an integer",
                replace(action, request_id=False),
            ),
            (
                "request_id must be an integer",
                replace(action, request_id=IntSubclass(action.request_id)),
            ),
            (
                "must be a boolean",
                replace(
                    action,
                    control=replace(action.control, boundary=1),
                ),
            ),
            (
                "accepted control identity",
                replace(
                    action,
                    control=replace(
                        action.control,
                        slot=(found.identity.control_slot + 1) % 16,
                    ),
                ),
            ),
            (
                "slot must be an integer",
                replace(
                    action,
                    control=replace(
                        action.control,
                        slot=IntSubclass(found.identity.control_slot),
                    ),
                ),
            ),
            (
                "16 lowercase hexadecimal",
                replace(
                    action,
                    control=replace(action.control, loaded_word="0" * 15),
                ),
            ),
        )
        for diagnostic, forged_action in cases:
            with self.subTest(diagnostic=diagnostic):
                forged = _replace_repetition_action(
                    self.repetition,
                    found.action_ordinal,
                    forged_action,
                )
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_target_access_order(forged)

        absent = self._one_target(resolution="absent")
        absent_action = self.repetition.actions[absent.action_ordinal]
        forged_sentinel = _replace_repetition_action(
            self.repetition,
            absent.action_ordinal,
            replace(
                absent_action,
                accepted=replace(_ACCEPTED_SENTINEL, boundary=0),
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "exact typed sentinel",
        ):
            actor_race_verify.build_target_access_order(forged_sentinel)

    def test_full_entry_point_rejects_tuple_and_action_subclasses_before_graph(
        self,
    ) -> None:
        class TupleSubclass(tuple):
            pass

        class ActionSubclass(actor_race_history.Action):
            pass

        tuple_forged = SimpleNamespace(
            actions=TupleSubclass(self.repetition.actions),
            diagnostics=self.repetition.diagnostics,
            shutdown=self.repetition.shutdown,
        )
        action_values = tuple(
            getattr(self.repetition.actions[0], field)
            for field in self.repetition.actions[0].__dataclass_fields__
        )
        subclass_action = ActionSubclass(*action_values)
        subclass_actions = list(self.repetition.actions)
        subclass_actions[0] = subclass_action
        action_forged = SimpleNamespace(
            actions=tuple(subclass_actions),
            diagnostics=self.repetition.diagnostics,
            shutdown=self.repetition.shutdown,
        )
        for diagnostic, forged in (
            ("exact immutable tuple", tuple_forged),
            ("invalid exact type", action_forged),
        ):
            for builder in (
                actor_race_verify.build_action_interval_order,
                actor_race_verify.build_submission_protocol_order,
                actor_race_verify.build_target_access_order,
            ):
                with self.subTest(diagnostic=diagnostic, builder=builder.__name__):
                    with mock.patch.object(
                        actor_race_verify.ReasonedDAG,
                        "build",
                    ) as graph_build:
                        with self.assertRaisesRegex(
                            actor_race_verify.ActorRaceVerificationError,
                            diagnostic,
                        ):
                            builder(forged)
                    graph_build.assert_not_called()

    def test_public_builders_snapshot_repetition_properties_exactly_once(self) -> None:
        clean_actions = self.repetition.actions
        forged_actions = list(clean_actions)
        forged_actions[0] = replace(
            forged_actions[0],
            producer=1 - forged_actions[0].producer,
        )

        class TogglingRepetition:
            def __init__(self) -> None:
                self.action_reads = 0
                self.diagnostic_reads = 0
                self.shutdown_reads = 0

            @property
            def actions(self) -> tuple[actor_race_history.Action, ...]:
                self.action_reads += 1
                if self.action_reads == 1:
                    return clean_actions
                return tuple(forged_actions)

            @property
            def diagnostics(self) -> object:
                self.diagnostic_reads += 1
                if self.diagnostic_reads == 1:
                    return self_outer.repetition.diagnostics
                return SimpleNamespace(action_counter_final=0)

            @property
            def shutdown(self) -> object:
                self.shutdown_reads += 1
                if self.shutdown_reads == 1:
                    return self_outer.repetition.shutdown
                return replace(
                    self_outer.repetition.shutdown,
                    accepted_submissions=0,
                )

        self_outer = self
        for builder, expected_shutdown_reads in (
            (actor_race_verify.build_action_interval_order, 0),
            (actor_race_verify.build_submission_protocol_order, 1),
            (actor_race_verify.build_target_access_order, 1),
        ):
            with self.subTest(builder=builder.__name__):
                toggling = TogglingRepetition()
                builder(toggling)
                self.assertEqual(toggling.action_reads, 1)
                self.assertEqual(toggling.diagnostic_reads, 1)
                self.assertEqual(
                    toggling.shutdown_reads,
                    expected_shutdown_reads,
                )

    def test_public_entry_point_authenticates_locally_and_is_not_injectable(self) -> None:
        authenticate = actor_race_verify.regenerate_authenticated_action_program
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
            wraps=authenticate,
        ) as authenticate_once:
            actor_race_verify.build_target_access_order(self.repetition)
        authenticate_once.assert_called_once_with()
        with self.assertRaises(TypeError):
            actor_race_verify.build_target_access_order(
                self.repetition,
                program=authenticate(),
            )


class ControlLifecycleOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repetition = _exact_control_repetition()
        cls.order = actor_race_verify.build_control_lifecycle_order(
            cls.repetition
        )

    def _script_target(
        self,
        observation: actor_race_verify.ControlWordObservation,
    ) -> actor_race_verify.TargetActionEvent:
        self.assertEqual(observation.source_kind, "script")
        return next(
            target
            for target in self.order.target_order.targets.by_action
            if target is not None and target.event.node == observation.node
        )

    def _replace_script_observation(
        self,
        observation: actor_race_verify.ControlWordObservation,
        *,
        loaded_word: int,
        resulting_word: int,
        disposition: str | None,
        result: str,
        error: actor_race_history.CapturedError | None,
    ) -> SimpleNamespace:
        target = self._script_target(observation)
        action = self.repetition.actions[target.action_ordinal]
        return _replace_control_action(
            self.repetition,
            target.action_ordinal,
            replace(
                action,
                result=result,
                error=error,
                control=replace(
                    action.control,
                    loaded_word=f"{loaded_word:016x}",
                    resulting_word=f"{resulting_word:016x}",
                    disposition=disposition,
                ),
            ),
        )

    def test_exact_typed_mapping_budget_and_lifecycle_edges(self) -> None:
        order = self.order
        lifecycle = order.lifecycle
        accepted_count = len(lifecycle.identities.by_request_id)
        self.assertEqual(accepted_count, 17)
        self.assertEqual(
            order.graph.node_count,
            order.target_order.graph.node_count + 5 * accepted_count + 1,
        )
        self.assertLessEqual(
            order.graph.node_count,
            actor_race_verify.MAX_CONTROL_LIFECYCLE_NODES,
        )
        self.assertEqual(
            actor_race_verify.MAX_CONTROL_LIFECYCLE_NODES,
            sum(
                count
                for _, count in actor_race_verify._CONTROL_LIFECYCLE_NODE_BUDGET
            ),
        )
        self.assertEqual(
            actor_race_verify.MAX_CONTROL_LIFECYCLE_EDGE_INPUTS,
            10_048,
        )
        self.assertLessEqual(
            actor_race_verify.MAX_CONTROL_LIFECYCLE_EDGE_INPUTS,
            actor_race_verify.MAX_PROTOCOL_EDGE_INPUTS,
        )
        self.assertEqual(len(lifecycle.request_events), 2 * accepted_count)
        self.assertEqual(len(lifecycle.cleanup_events), 3 * accepted_count)
        self.assertEqual(len(lifecycle.phase_events), 1)
        self.assertTrue(
            all(
                type(event) is actor_race_verify.RequestEvent
                and not hasattr(event, "action_ordinal")
                for event in lifecycle.request_events
            )
        )
        self.assertTrue(
            all(
                type(event) is actor_race_verify.CleanupEvent
                and not hasattr(event, "action_ordinal")
                for event in lifecycle.cleanup_events
            )
        )
        self.assertIs(
            lifecycle.phase_events[0],
            lifecycle.pre_cleanup,
        )

        for request in lifecycle.requests_by_request_id:
            identity = request.identity
            control_bind = identity.submission.control_bind
            assert control_bind is not None
            self.assertTrue(
                order.graph.precedes(
                    control_bind.node,
                    request.control_terminal_publish.node,
                )
            )
            self.assertTrue(
                order.graph.precedes(
                    request.control_terminal_publish.node,
                    request.request_reap.node,
                )
            )
            self.assertIs(
                lifecycle.requests_by_client_index[identity.client_index],
                request,
            )

        reused = lifecycle.control_by_slot[0]
        self.assertEqual(len(reused), 2)
        previous, current = reused
        next_bind = current.identity.submission.control_bind
        assert next_bind is not None
        self.assertEqual(
            order.graph.reasons(
                previous.request.request_reap.node,
                next_bind.node,
            ),
            ("control-slot-0-reap-before-rebind",),
        )
        for observation in previous.observations:
            self.assertTrue(order.graph.precedes(observation.node, next_bind.node))

    def test_script_nodes_are_reused_and_stale_maps_to_real_later_bind(self) -> None:
        target_nodes = {
            event.node for event in self.order.target_order.targets.target_events
        }
        stale_count = 0
        for observation in self.order.lifecycle.observations:
            if observation.source_kind == "script":
                self.assertIn(observation.node, target_nodes)
            if not observation.stale:
                continue
            stale_count += 1
            self.assertGreater(
                observation.observed_identity.control_generation,
                observation.expected_generation,
            )
            slot = observation.expected_slot
            generation = observation.loaded_word >> 3
            self.assertIs(
                self.order.lifecycle.identities.by_control_slot[slot][
                    generation - 1
                ],
                observation.observed_identity,
            )
            bind = observation.observed_identity.submission.control_bind
            assert bind is not None
            self.assertTrue(self.order.graph.precedes(bind.node, observation.node))
        self.assertGreater(stale_count, 0)

    def test_pre_cleanup_partition_and_authority_sequence_are_exact(self) -> None:
        lifecycle = self.order.lifecycle
        gate = lifecycle.pre_cleanup
        endpoints = self.order.target_order.submission_order.interval_order.endpoints
        for producer in (0, 1):
            tail = next(
                item.response
                for item in reversed(endpoints.by_action)
                if item.producer == producer
            )
            self.assertTrue(self.order.graph.precedes(tail.node, gate.node))

        cleanup = lifecycle.cleanup_in_order
        self.assertEqual(len(cleanup), len(lifecycle.requests_by_request_id))
        self.assertEqual(
            self.order.graph.reasons(gate.node, cleanup[0].invocation.node),
            ("pre-cleanup-before-first-authority",),
        )
        for previous, current in zip(cleanup, cleanup[1:]):
            self.assertEqual(
                self.order.graph.reasons(
                    previous.response.node,
                    current.invocation.node,
                ),
                ("cleanup-authority-counter-order",),
            )
        for events in cleanup:
            self.assertTrue(
                self.order.graph.precedes(
                    events.invocation.node,
                    events.control_decision.node,
                )
            )
            self.assertTrue(
                self.order.graph.precedes(
                    events.control_decision.node,
                    events.response.node,
                )
            )

        for request in lifecycle.requests_by_request_id:
            if request.receiver_state == "consumed":
                self.assertTrue(
                    self.order.graph.precedes(request.request_reap.node, gate.node)
                )
                assert request.successful_drop_action is not None
                drop = self.order.target_order.targets.by_action[
                    request.successful_drop_action
                ]
                assert drop is not None
                self.assertTrue(
                    self.order.graph.precedes(
                        drop.event.node,
                        request.request_reap.node,
                    )
                )
            else:
                self.assertTrue(
                    self.order.graph.precedes(gate.node, request.request_reap.node)
                )
                authority = lifecycle.cleanup_by_client_index[
                    request.identity.client_index
                ]
                assert authority is not None
                self.assertTrue(
                    self.order.graph.precedes(
                        authority.response.node,
                        request.request_reap.node,
                    )
                )

    def test_cleanup_t_present_precedes_the_entire_authority_hold(self) -> None:
        first_invocation = self.order.lifecycle.cleanup_in_order[0].invocation
        bracketed = 0
        for slot in self.order.lifecycle.control_by_slot:
            for state in slot:
                cleanup_observations = tuple(
                    observation
                    for observation in state.observations
                    if observation.source_kind == "cleanup"
                )
                if not cleanup_observations or not any(
                    (observation.loaded_word | observation.resulting_word) & 4
                    for observation in cleanup_observations
                ):
                    continue
                bracketed += 1
                self.assertEqual(
                    self.order.graph.reasons(
                        state.request.control_terminal_publish.node,
                        first_invocation.node,
                    ),
                    ("control-terminal-before-cleanup-hold",),
                )
        self.assertGreater(bracketed, 0)

    def test_cleanup_t_absent_follows_the_entire_authority_hold(self) -> None:
        identity = self.order.lifecycle.identities.by_request_id[0]
        planner = actor_race_verify._ProtocolGraphPlanner(0)
        terminal = planner.allocate_request(
            identity.client_index,
            identity.request_id,
            "ControlTerminalPublish",
        )
        first = planner.allocate_cleanup(
            0,
            identity.client_index,
            identity.request_id,
            "CleanupInvoke",
        )
        last = planner.allocate_cleanup(
            0,
            identity.client_index,
            identity.request_id,
            "CleanupRespond",
        )
        word = identity.control_generation << 3 | 1
        observation = actor_race_verify.ControlWordObservation(
            first.node,
            "cleanup",
            identity.client_index,
            identity.request_id,
            "cancel",
            identity.control_slot,
            identity.control_generation,
            identity,
            word,
            word,
            "already_requested",
            False,
        )
        actor_race_verify._add_cleanup_terminal_hold_brackets(
            planner,
            (observation,),
            terminal.node,
            first.node,
            last.node,
        )
        graph = planner.build()
        self.assertEqual(
            graph.reasons(last.node, terminal.node),
            ("cleanup-hold-before-control-terminal",),
        )
        self.assertEqual(graph.reasons(terminal.node, first.node), ())

    def test_mixed_cleanup_t_state_cycles_across_the_pump_hold(self) -> None:
        reused = self.order.lifecycle.control_by_slot[0][1]
        stale = next(
            observation
            for observation in reused.observations
            if observation.source_kind == "cleanup" and observation.stale
        )
        current = next(
            observation
            for observation in reused.observations
            if observation.source_kind == "cleanup" and not observation.stale
        )
        self.assertTrue(stale.loaded_word & 4)
        self.assertTrue(current.loaded_word & 4)
        events = self.order.lifecycle.cleanup_by_client_index[
            stale.owner_client_index
        ]
        assert events is not None
        authority = self.repetition.cleanup_authorities[
            events.cleanup_ordinal
        ]
        t_absent_word = reused.identity.control_generation << 3 | 1
        forged = _replace_cleanup_authority(
            self.repetition,
            events.cleanup_ordinal,
            replace(
                authority,
                control=replace(
                    authority.control,
                    loaded_word=f"{t_absent_word:016x}",
                    resulting_word=f"{t_absent_word:016x}",
                ),
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            actor_race_verify.build_control_lifecycle_order(forged)

    def test_cleanup_hold_brackets_add_at_most_two_edges_per_generation(self) -> None:
        identity = self.order.lifecycle.identities.by_request_id[0]
        planner = actor_race_verify._ProtocolGraphPlanner(0)
        terminal = planner.allocate_request(
            identity.client_index,
            identity.request_id,
            "ControlTerminalPublish",
        )
        first = planner.allocate_cleanup(
            0,
            identity.client_index,
            identity.request_id,
            "CleanupInvoke",
        )
        last = planner.allocate_cleanup(
            0,
            identity.client_index,
            identity.request_id,
            "CleanupRespond",
        )
        planner.edge(first.node, last.node, "authority-hold")
        words = (1, 1, 5, 5)
        observations = tuple(
            actor_race_verify.ControlWordObservation(
                first.node,
                "cleanup",
                identity.client_index,
                identity.request_id,
                "cancel",
                identity.control_slot,
                identity.control_generation,
                identity,
                identity.control_generation << 3 | flags,
                identity.control_generation << 3 | flags,
                "already_terminal" if flags & 4 else "already_requested",
                False,
            )
            for flags in words
        )
        actor_race_verify._add_cleanup_terminal_hold_brackets(
            planner,
            observations,
            terminal.node,
            first.node,
            last.node,
        )
        self.assertEqual(planner.edge_input_count, 3)
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            planner.build()

    def test_control_word_algebra_rejects_invalid_flags_and_transitions(self) -> None:
        requested_cancel = next(
            observation
            for observation in self.order.lifecycle.observations
            if observation.source_kind == "script"
            and observation.operation == "cancel"
            and observation.disposition == "requested"
        )
        requested_disconnect = next(
            observation
            for observation in self.order.lifecycle.observations
            if observation.source_kind == "script"
            and observation.operation == "disconnect"
            and observation.disposition == "requested"
        )
        stale = next(
            observation
            for observation in self.order.lifecycle.observations
            if observation.source_kind == "script" and observation.stale
        )
        cancel_generation = requested_cancel.expected_generation
        disconnect_generation = requested_disconnect.expected_generation
        cases = (
            (
                "D-implies-C",
                self._replace_script_observation(
                    requested_cancel,
                    loaded_word=cancel_generation << 3 | 2,
                    resulting_word=cancel_generation << 3 | 3,
                    disposition="requested",
                    result="cancel_requested",
                    error=None,
                ),
            ),
            (
                "changed the loaded control generation",
                self._replace_script_observation(
                    requested_cancel,
                    loaded_word=cancel_generation << 3,
                    resulting_word=(cancel_generation + 1) << 3 | 1,
                    disposition="requested",
                    result="cancel_requested",
                    error=None,
                ),
            ),
            (
                "disposition disagrees",
                self._replace_script_observation(
                    requested_cancel,
                    loaded_word=cancel_generation << 3 | 1,
                    resulting_word=cancel_generation << 3 | 1,
                    disposition="requested",
                    result="cancel_requested",
                    error=None,
                ),
            ),
            (
                "operation algebra",
                self._replace_script_observation(
                    requested_disconnect,
                    loaded_word=disconnect_generation << 3 | 4,
                    resulting_word=disconnect_generation << 3 | 4,
                    disposition="already_terminal",
                    result="receiver_dropped",
                    error=None,
                ),
            ),
            (
                "stale control changed",
                self._replace_script_observation(
                    stale,
                    loaded_word=stale.loaded_word,
                    resulting_word=stale.loaded_word ^ 4,
                    disposition=None,
                    result="error",
                    error=_STALE_ERROR,
                ),
            ),
            (
                "no accepted control bind",
                self._replace_script_observation(
                    stale,
                    loaded_word=64 << 3,
                    resulting_word=64 << 3,
                    disposition=None,
                    result="error",
                    error=_STALE_ERROR,
                ),
            ),
        )
        for diagnostic, forged in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_control_lifecycle_order(forged)

    def test_stale_spelling_cannot_hide_a_late_old_generation_observation(self) -> None:
        stale = next(
            observation
            for observation in self.order.lifecycle.observations
            if observation.source_kind == "script"
            and observation.stale
            and observation.expected_slot == 0
        )
        old_word = stale.expected_generation << 3 | 7
        forged = self._replace_script_observation(
            stale,
            loaded_word=old_word,
            resulting_word=old_word,
            disposition="already_terminal",
            result="cancel_already_terminal",
            error=None,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            actor_race_verify.build_control_lifecycle_order(forged)

    def test_unique_publishers_and_word_lattice_fail_closed(self) -> None:
        state = self.order.lifecycle.control_by_slot[0][0]
        already_requested = next(
            observation
            for observation in state.observations
            if observation.source_kind == "script"
            and observation.disposition == "already_requested"
        )
        generation = state.identity.control_generation
        duplicate = self._replace_script_observation(
            already_requested,
            loaded_word=generation << 3,
            resulting_word=generation << 3 | 1,
            disposition="requested",
            result="cancel_requested",
            error=None,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "exactly one captured C publisher",
        ):
            actor_race_verify.build_control_lifecycle_order(duplicate)

        state_without_prior_cancel = next(
            state
            for slot in self.order.lifecycle.control_by_slot
            for state in slot
            if state.cancel_publisher is state.disconnect_publisher
            and state.disconnect_publisher is not None
        )
        publisher = state_without_prior_cancel.disconnect_publisher
        assert publisher is not None
        generation = state_without_prior_cancel.identity.control_generation
        missing = self._replace_script_observation(
            publisher,
            loaded_word=generation << 3 | 3,
            resulting_word=generation << 3 | 3,
            disposition="already_requested",
            result="receiver_dropped",
            error=None,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "exactly one captured C publisher",
        ):
            actor_race_verify.build_control_lifecycle_order(missing)

        terminal_reader = next(
            observation
            for observation in state.observations
            if observation.source_kind == "script"
            and observation.disposition == "already_terminal"
        )
        impossible = self._replace_script_observation(
            terminal_reader,
            loaded_word=state.identity.control_generation << 3 | 4,
            resulting_word=state.identity.control_generation << 3 | 4,
            disposition="already_terminal",
            result="cancel_already_terminal",
            error=None,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            actor_race_verify.build_control_lifecycle_order(impossible)

    def test_equal_state_readers_remain_unordered_without_external_evidence(
        self,
    ) -> None:
        identity = self.order.lifecycle.identities.by_request_id[0]
        planner = actor_race_verify._ProtocolGraphPlanner(0)
        publisher = planner.allocate(0, "ControlCancel")
        left = planner.allocate(1, "ControlCancel")
        right = planner.allocate(2, "ControlCancel")
        for event in (publisher, left, right):
            loaded = 0 if event is publisher else 1
            observation = actor_race_verify.ControlWordObservation(
                event.node,
                "script",
                identity.client_index,
                identity.request_id,
                "cancel",
                identity.control_slot,
                identity.control_generation,
                identity,
                identity.control_generation << 3 | loaded,
                identity.control_generation << 3 | 1,
                "requested" if event is publisher else "already_requested",
                False,
            )
            actor_race_verify._word_bit_relation(
                planner,
                observation,
                publisher.node,
                1,
                "C",
            )
        graph = planner.build()
        self.assertTrue(graph.precedes(publisher.node, left.node))
        self.assertTrue(graph.precedes(publisher.node, right.node))
        self.assertFalse(graph.precedes(left.node, right.node))
        self.assertFalse(graph.precedes(right.node, left.node))

    def test_control_lattice_matches_every_bruteforce_must_order(self) -> None:
        labels = ("C0", "C1", "D0", "D1", "T")

        def transition(flags: int, label: str) -> tuple[int, int, str | None]:
            if label.startswith("C"):
                if flags & 4:
                    return flags, flags, "already_terminal"
                if flags & 1:
                    return flags, flags, "already_requested"
                return flags, flags | 1, "requested"
            if label.startswith("D"):
                disposition = (
                    "already_terminal"
                    if flags & 4
                    else "already_requested" if flags & 2 else "requested"
                )
                return flags, flags if flags & 2 else flags | 3, disposition
            return flags, flags | 4, None

        histories = []
        for schedule in permutations(labels):
            flags = 0
            witnesses = {}
            for label in schedule:
                loaded, flags, disposition = transition(flags, label)
                if label != "T":
                    witnesses[label] = (loaded, flags, disposition)
            histories.append((schedule, witnesses))

        identity = self.order.lifecycle.identities.by_request_id[0]
        signatures = set()
        for _, witnesses in histories:
            signature = tuple(witnesses[label] for label in labels[:-1])
            if signature in signatures:
                continue
            signatures.add(signature)
            valid = [
                schedule
                for schedule, candidate in histories
                if candidate == witnesses
            ]
            planner = actor_race_verify._ProtocolGraphPlanner(0)
            events = {
                "C0": planner.allocate(0, "ControlCancel"),
                "C1": planner.allocate(1, "ControlCancel"),
                "D0": planner.allocate(2, "ControlDisconnect"),
                "D1": planner.allocate(3, "ControlDisconnect"),
                "T": planner.allocate_request(
                    identity.client_index,
                    identity.request_id,
                    "ControlTerminalPublish",
                ),
            }
            base_word = identity.control_generation << 3
            observations = {}
            for label in labels[:-1]:
                loaded, resulting, disposition = witnesses[label]
                observations[label] = actor_race_verify.ControlWordObservation(
                    events[label].node,
                    "script",
                    identity.client_index,
                    identity.request_id,
                    "cancel" if label.startswith("C") else "disconnect",
                    identity.control_slot,
                    identity.control_generation,
                    identity,
                    base_word | loaded,
                    base_word | resulting,
                    disposition,
                    False,
                )
            cancel_publisher = next(
                label
                for label, observation in observations.items()
                if not observation.loaded_word & 1 and observation.resulting_word & 1
            )
            disconnect_publisher = next(
                label
                for label, observation in observations.items()
                if not observation.loaded_word & 2 and observation.resulting_word & 2
            )
            for observation in observations.values():
                for publisher, bit, name in (
                    ("T", 4, "T"),
                    (cancel_publisher, 1, "C"),
                    (disconnect_publisher, 2, "D"),
                ):
                    actor_race_verify._word_bit_relation(
                        planner,
                        observation,
                        events[publisher].node,
                        bit,
                        name,
                    )
            graph = planner.build()
            for before, after in permutations(labels, 2):
                expected = all(
                    schedule.index(before) < schedule.index(after)
                    for schedule in valid
                )
                self.assertEqual(
                    graph.precedes(events[before].node, events[after].node),
                    expected,
                    (signature, before, after),
                )
        self.assertEqual(len(histories), 120)
        self.assertEqual(len(signatures), 60)

    def test_live_cleanup_requested_waits_for_last_authority_response(self) -> None:
        order = actor_race_verify.build_control_lifecycle_order(
            _live_cleanup_requested_repetition()
        )
        request = order.lifecycle.requests_by_request_id[0]
        cleanup = order.lifecycle.cleanup_in_order[0]
        self.assertEqual(request.receiver_state, "live")
        self.assertEqual(cleanup.observation.disposition, "requested")
        self.assertFalse(
            any(
                observation.source_kind == "script"
                for observation in order.lifecycle.observations
            )
        )
        self.assertEqual(
            order.graph.reasons(
                cleanup.response.node,
                request.control_terminal_publish.node,
            ),
            ("cleanup-hold-before-control-terminal",),
        )

    def test_stale_reader_cannot_invent_an_unreached_later_state(self) -> None:
        later = self.order.lifecycle.control_by_slot[0][1]
        stale = next(
            observation
            for observation in later.observations
            if observation.source_kind == "script"
            and observation.stale
            and observation.loaded_word & 7 == 1
        )
        base_word = later.identity.control_generation << 3
        for impossible_flags in (0, 5):
            with self.subTest(impossible_flags=impossible_flags):
                forged = self._replace_script_observation(
                    stale,
                    loaded_word=base_word | impossible_flags,
                    resulting_word=base_word | impossible_flags,
                    disposition=None,
                    result="error",
                    error=_STALE_ERROR,
                )
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    "cycle detected",
                ):
                    actor_race_verify.build_control_lifecycle_order(forged)

    def test_live_old_generation_cannot_survive_rebind(self) -> None:
        actions = []
        for action in self.repetition.actions:
            if action.result == "submit_accepted" and action.client_index in (
                4,
                16,
            ):
                action = replace(
                    action,
                    accepted=replace(
                        action.accepted,
                        control_slot=12 if action.client_index == 4 else 0,
                    ),
                )
            if action.request_id in (1, 13) and action.control.boundary:
                control_slot = 12 if action.request_id == 1 else 0
                control = replace(action.control, slot=control_slot)
                if action.request_id == 1 and action.error == _STALE_ERROR:
                    word = action.control.expected_generation << 3 | 7
                    action = replace(
                        action,
                        result="cancel_already_terminal",
                        error=None,
                        control=replace(
                            control,
                            loaded_word=f"{word:016x}",
                            resulting_word=f"{word:016x}",
                            disposition="already_terminal",
                        ),
                    )
                else:
                    action = replace(action, control=control)
            actions.append(action)

        cleanup_authorities = []
        for authority in self.repetition.cleanup_authorities:
            if authority.request_id == 1:
                word = authority.control.expected_generation << 3 | 7
                authority = replace(
                    authority,
                    error=None,
                    control=replace(
                        authority.control,
                        slot=12,
                        loaded_word=f"{word:016x}",
                        resulting_word=f"{word:016x}",
                        disposition="already_terminal",
                    ),
                )
            elif authority.request_id == 13:
                authority = replace(
                    authority,
                    control=replace(authority.control, slot=0),
                )
            cleanup_authorities.append(authority)

        forged = _replace_control_repetition(
            self.repetition,
            actions=tuple(actions),
            cleanup_authorities=tuple(cleanup_authorities),
        )
        actor_race_verify.build_target_access_order(forged)
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            actor_race_verify.build_control_lifecycle_order(forged)

    def test_malformed_cleanup_shells_fail_before_auth_or_graph_build(self) -> None:
        class TupleSubclass(tuple):
            pass

        class CleanupAuthoritySubclass(actor_race_history.CleanupAuthority):
            pass

        class CleanupReceiverSubclass(actor_race_history.CleanupReceiver):
            pass

        authorities = self.repetition.cleanup_authorities
        receivers = self.repetition.cleanup_receivers
        authority_values = tuple(
            getattr(authorities[0], field)
            for field in authorities[0].__dataclass_fields__
        )
        receiver_values = tuple(
            getattr(receivers[0], field)
            for field in receivers[0].__dataclass_fields__
        )
        authority_items = list(authorities)
        authority_items[0] = CleanupAuthoritySubclass(*authority_values)
        receiver_items = list(receivers)
        receiver_items[0] = CleanupReceiverSubclass(*receiver_values)
        cases = (
            _replace_control_repetition(
                self.repetition,
                cleanup_authorities=TupleSubclass(authorities),
            ),
            _replace_control_repetition(
                self.repetition,
                cleanup_authorities=tuple(authority_items),
            ),
            _replace_control_repetition(
                self.repetition,
                cleanup_receivers=TupleSubclass(receivers),
            ),
            _replace_control_repetition(
                self.repetition,
                cleanup_receivers=tuple(receiver_items),
            ),
            _replace_control_repetition(
                self.repetition,
                pre_cleanup=SimpleNamespace(
                    outstanding_requests=len(receivers)
                ),
            ),
            _replace_control_repetition(
                self.repetition,
                diagnostics=SimpleNamespace(
                    action_counter_final=2_048,
                    cleanup_counter_final=False,
                ),
            ),
            _replace_control_repetition(
                self.repetition,
                cleanup_authorities=authorities
                + (authorities[0],)
                * (
                    actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT
                    + 1
                    - len(authorities)
                ),
            ),
        )
        for forged in cases:
            with self.subTest(
                authority_type=type(forged.cleanup_authorities).__name__,
                receiver_type=type(forged.cleanup_receivers).__name__,
            ):
                with mock.patch.object(
                    actor_race_verify.ReasonedDAG,
                    "build",
                ) as graph_build, mock.patch.object(
                    actor_race_verify,
                    "regenerate_authenticated_action_program",
                ) as authenticate:
                    with self.assertRaises(
                        actor_race_verify.ActorRaceVerificationError
                    ):
                        actor_race_verify.build_control_lifecycle_order(forged)
                graph_build.assert_not_called()
                authenticate.assert_not_called()

    def test_cleanup_identity_counter_and_partition_fail_closed(self) -> None:
        class TupleSubclass(tuple):
            pass

        class CleanupAuthoritySubclass(actor_race_history.CleanupAuthority):
            pass

        authorities = self.repetition.cleanup_authorities
        receivers = self.repetition.cleanup_receivers
        first = authorities[0]
        second = authorities[1]
        subclass_values = tuple(
            getattr(first, field)
            for field in first.__dataclass_fields__
        )
        subclass_authorities = list(authorities)
        subclass_authorities[0] = CleanupAuthoritySubclass(*subclass_values)
        cases = (
            (
                "exact immutable tuple",
                _replace_control_repetition(
                    self.repetition,
                    cleanup_authorities=TupleSubclass(authorities),
                ),
            ),
            (
                "invalid exact type",
                _replace_control_repetition(
                    self.repetition,
                    cleanup_authorities=tuple(subclass_authorities),
                ),
            ),
            (
                "exactly cover",
                _replace_control_repetition(
                    self.repetition,
                    cleanup_authorities=authorities[:-1],
                    diagnostics=SimpleNamespace(
                        action_counter_final=2_048,
                        cleanup_counter_final=2
                        * (len(authorities) - 1 + len(receivers)),
                    ),
                ),
            ),
            (
                "invocation must be an integer",
                _replace_cleanup_authority(
                    self.repetition,
                    0,
                    replace(first, invocation=False),
                ),
            ),
            (
                "accepted request identity",
                _replace_cleanup_authority(
                    self.repetition,
                    0,
                    replace(
                        first,
                        client_index=second.client_index,
                        request_id=second.request_id,
                    ),
                ),
            ),
            (
                "cleanup_counter_final must be an integer",
                _replace_control_repetition(
                    self.repetition,
                    diagnostics=SimpleNamespace(
                        action_counter_final=2_048,
                        cleanup_counter_final=False,
                    ),
                ),
            ),
            (
                "pre_cleanup has an invalid exact type",
                _replace_control_repetition(
                    self.repetition,
                    pre_cleanup=SimpleNamespace(
                        outstanding_requests=len(receivers)
                    ),
                ),
            ),
        )
        for diagnostic, forged in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_control_lifecycle_order(forged)

        remaining = receivers[:-1]
        missing_live = _replace_control_repetition(
            self.repetition,
            cleanup_receivers=remaining,
            diagnostics=SimpleNamespace(
                action_counter_final=2_048,
                cleanup_counter_final=2 * (len(authorities) + len(remaining)),
            ),
            pre_cleanup=replace(
                self.repetition.pre_cleanup,
                outstanding_requests=len(remaining),
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "do not partition",
        ):
            actor_race_verify.build_control_lifecycle_order(missing_live)

        consumed = next(
            request
            for request in self.order.lifecycle.requests_by_request_id
            if request.receiver_state == "consumed"
            and request.identity.client_index < receivers[1].client_index
        )
        overlap_receivers = list(receivers)
        overlap_receivers[0] = replace(
            overlap_receivers[0],
            client_index=consumed.identity.client_index,
            request_id=consumed.identity.request_id,
        )
        overlap = _replace_control_repetition(
            self.repetition,
            cleanup_receivers=tuple(overlap_receivers),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "ownership overlap",
        ):
            actor_race_verify.build_control_lifecycle_order(overlap)

    def test_cleanup_result_matrix_rejects_invalid_live_and_consumed_words(self) -> None:
        lifecycle = self.order.lifecycle
        consumed_events = next(
            events
            for events in lifecycle.cleanup_in_order
            if lifecycle.requests_by_client_index[
                events.identity.client_index
            ].receiver_state
            == "consumed"
            and not events.observation.stale
        )
        live_events = next(
            events
            for events in lifecycle.cleanup_in_order
            if lifecycle.requests_by_client_index[
                events.identity.client_index
            ].receiver_state
            == "live"
        )
        consumed = self.repetition.cleanup_authorities[
            consumed_events.cleanup_ordinal
        ]
        live = self.repetition.cleanup_authorities[live_events.cleanup_ordinal]
        consumed_word = consumed.control.expected_generation << 3 | 5
        forged_consumed = _replace_cleanup_authority(
            self.repetition,
            consumed_events.cleanup_ordinal,
            replace(
                consumed,
                control=replace(
                    consumed.control,
                    loaded_word=f"{consumed_word:016x}",
                    resulting_word=f"{consumed_word:016x}",
                ),
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "consumed tombstone must be flags 7",
        ):
            actor_race_verify.build_control_lifecycle_order(forged_consumed)

        live_word = live.control.expected_generation << 3 | 1
        forged_live = _replace_cleanup_authority(
            self.repetition,
            live_events.cleanup_ordinal,
            replace(
                live,
                control=replace(
                    live.control,
                    loaded_word=f"{live_word:016x}",
                    resulting_word=f"{live_word:016x}",
                    disposition="already_requested",
                ),
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "invalid live-request result",
        ):
            actor_race_verify.build_control_lifecycle_order(forged_live)

    def test_non_action_allocator_validates_ownership_before_reserving(self) -> None:
        planner = actor_race_verify._ProtocolGraphPlanner(0)
        request = planner.allocate_request(0, 1, "ControlTerminalPublish")
        cleanup = planner.allocate_cleanup(0, 0, 1, "CleanupInvoke")
        phase = planner.allocate_phase("PreCleanupGate")
        self.assertIs(type(request), actor_race_verify.RequestEvent)
        self.assertIs(type(cleanup), actor_race_verify.CleanupEvent)
        self.assertIs(type(phase), actor_race_verify.PhaseEvent)
        node_count = planner.node_count
        for call in (
            lambda: planner.allocate_request(False, 1, "RequestReap"),
            lambda: planner.allocate_request(0, 1, "Fake"),
            lambda: planner.allocate_cleanup(False, 0, 1, "CleanupRespond"),
            lambda: planner.allocate_phase("Fake"),
        ):
            with self.subTest(call=call):
                with self.assertRaises(actor_race_verify.ActorRaceVerificationError):
                    call()
                self.assertEqual(planner.node_count, node_count)

    def test_public_builder_snapshots_once_and_authenticates_locally(self) -> None:
        source = self.repetition

        class CountingRepetition:
            def __init__(self) -> None:
                self.reads: dict[str, int] = {}

            def _read(self, name: str) -> object:
                self.reads[name] = self.reads.get(name, 0) + 1
                return getattr(source, name)

            actions = property(lambda self: self._read("actions"))
            cleanup_authorities = property(
                lambda self: self._read("cleanup_authorities")
            )
            cleanup_receivers = property(
                lambda self: self._read("cleanup_receivers")
            )
            diagnostics = property(lambda self: self._read("diagnostics"))
            pre_cleanup = property(lambda self: self._read("pre_cleanup"))
            shutdown = property(lambda self: self._read("shutdown"))

        counting = CountingRepetition()
        authenticate = actor_race_verify.regenerate_authenticated_action_program
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
            wraps=authenticate,
        ) as authenticate_once:
            actor_race_verify.build_control_lifecycle_order(counting)
        authenticate_once.assert_called_once_with()
        self.assertEqual(
            counting.reads,
            {
                "actions": 1,
                "cleanup_authorities": 1,
                "cleanup_receivers": 1,
                "diagnostics": 1,
                "pre_cleanup": 1,
                "shutdown": 1,
            },
        )
        with self.assertRaises(TypeError):
            actor_race_verify.build_control_lifecycle_order(
                source,
                program=authenticate(),
            )


class EndpointObservationOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repetition = _exact_endpoint_repetition()
        cls.order = actor_race_verify.build_endpoint_observation_order(
            cls.repetition
        )

    def test_exact_mapping_has_frozen_node_arithmetic_and_non_action_owners(
        self,
    ) -> None:
        accepted = len(self.order.endpoint.requests_by_request_id)
        live = len(self.order.endpoint.cleanup_in_order)
        opportunistic = sum(
            event is not None
            for event in self.order.endpoint.opportunistic_by_action
        )
        self.assertEqual(
            self.order.graph.node_count,
            self.order.control_order.graph.node_count
            + 2 * accepted
            + 3 * live
            + opportunistic
            + 1,
        )
        self.assertLessEqual(
            self.order.graph.node_count,
            actor_race_verify.MAX_ENDPOINT_OBSERVATION_NODES,
        )
        self.assertEqual(
            actor_race_verify.MAX_ENDPOINT_OBSERVATION_NODES,
            4_118,
        )
        self.assertEqual(
            actor_race_verify.MAX_ENDPOINT_OBSERVATION_EDGE_INPUTS,
            12_976,
        )
        self.assertEqual(
            tuple(event.kind for event in self.order.endpoint.phase_events),
            ("PreCleanupGate", "PreShutdownGate"),
        )
        eof_sources = {
            lifecycle.first_eof_source
            for lifecycle in self.order.endpoint.requests_by_request_id
        }
        self.assertIn("receiver_drop", eof_sources)
        self.assertIn("cleanup_receiver", eof_sources)
        for receiver_ordinal, events in enumerate(
            self.order.endpoint.cleanup_in_order
        ):
            self.assertEqual(events.receiver_ordinal, receiver_ordinal)
            self.assertEqual(
                events.cleanup_sequence_ordinal,
                accepted + receiver_ordinal,
            )
            for event in (
                events.invocation,
                events.terminal_acknowledgement,
                events.response,
            ):
                self.assertIs(type(event), actor_race_verify.CleanupReceiverEvent)
                self.assertEqual(event.receiver_ordinal, receiver_ordinal)

    def test_endpoint_terminal_eof_reap_rebind_and_cleanup_edges(self) -> None:
        graph = self.order.graph
        for lifecycle in self.order.endpoint.requests_by_request_id:
            bind = lifecycle.identity.submission.endpoint_bind
            assert bind is not None
            self.assertTrue(
                graph.precedes(bind.node, lifecycle.endpoint_terminal_publish.node)
            )
            self.assertTrue(
                graph.precedes(
                    lifecycle.request.control_terminal_publish.node,
                    lifecycle.endpoint_terminal_publish.node,
                )
            )
            self.assertTrue(
                graph.precedes(
                    lifecycle.endpoint_terminal_publish.node,
                    lifecycle.first_eof_acknowledgement.node,
                )
            )
            self.assertTrue(
                graph.precedes(
                    lifecycle.first_eof_acknowledgement.node,
                    lifecycle.request.request_reap.node,
                )
            )
            self.assertTrue(
                graph.precedes(
                    lifecycle.request.request_reap.node,
                    self.order.endpoint.pre_shutdown.node,
                )
            )
        for slot in self.order.endpoint.identities.by_endpoint_slot:
            for previous, current in zip(slot, slot[1:]):
                previous_lifecycle = self.order.endpoint.requests_by_client_index[
                    previous.client_index
                ]
                next_bind = current.submission.endpoint_bind
                assert previous_lifecycle is not None and next_bind is not None
                self.assertTrue(
                    graph.precedes(
                        previous_lifecycle.request.request_reap.node,
                        next_bind.node,
                    )
                )
        for events in self.order.endpoint.cleanup_in_order:
            lifecycle = self.order.endpoint.requests_by_client_index[
                events.identity.client_index
            ]
            assert lifecycle is not None
            self.assertTrue(
                graph.precedes(
                    events.invocation.node,
                    events.terminal_acknowledgement.node,
                )
            )
            self.assertTrue(
                graph.precedes(
                    lifecycle.endpoint_terminal_publish.node,
                    events.terminal_acknowledgement.node,
                )
            )
            self.assertTrue(
                graph.precedes(
                    events.terminal_acknowledgement.node,
                    lifecycle.request.request_reap.node,
                )
            )
            self.assertTrue(
                graph.precedes(
                    lifecycle.request.request_reap.node,
                    events.response.node,
                )
            )

    def test_ungrounded_cached_eof_fails_closed(self) -> None:
        ordinal = next(
            action.ordinal
            for action in self.repetition.actions
            if action.result == "drain_empty"
        )
        action = self.repetition.actions[ordinal]
        forged = _replace_endpoint_repetition(
            self.repetition,
            actions=tuple(
                replace(
                    candidate,
                    result="drain_eof",
                    cached_eof=True,
                    primary_pop=_PRIMARY_POP_SENTINEL,
                )
                if candidate.ordinal == ordinal
                else candidate
                for candidate in self.repetition.actions
            ),
        )
        self.assertFalse(action.cached_eof)
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cached EOF before its first acknowledgement",
        ):
            actor_race_verify.build_endpoint_observation_order(forged)

    def test_direct_then_cached_eof_has_exact_response_custody(self) -> None:
        client_index = 50
        actions = list(self.repetition.actions)
        drains = [
            action
            for action in actions
            if action.client_index == client_index
            and action.kind == "drain"
            and action.result == "drain_empty"
        ]
        direct = drains[-2]
        cached = drains[-1]
        identity = next(
            action.accepted
            for action in actions
            if action.result == "submit_accepted"
            and action.client_index == client_index
        )
        actions[direct.ordinal] = replace(
            direct,
            result="drain_eof",
            primary_pop=actor_race_history.PopWitness(
                "primary",
                True,
                identity.endpoint_slot,
                identity.endpoint_generation,
                0,
                0,
                None,
            ),
        )
        actions[cached.ordinal] = replace(
            cached,
            result="drain_eof",
            cached_eof=True,
            primary_pop=_PRIMARY_POP_SENTINEL,
        )
        later_readers = []
        for candidate in actions[direct.ordinal + 1 :]:
            if not candidate.control.boundary:
                continue
            loaded_word = int(candidate.control.loaded_word, 16)
            if (
                candidate.control.slot != identity.control_slot
                or loaded_word >> 3 != identity.control_generation
                or loaded_word & 4
            ):
                continue
            self.assertNotEqual(candidate.client_index, client_index)
            self.assertEqual(candidate.result, "error")
            self.assertIsNone(candidate.control.disposition)
            later_readers.append(candidate.ordinal)
            resulting_word = int(candidate.control.resulting_word, 16)
            actions[candidate.ordinal] = replace(
                candidate,
                control=replace(
                    candidate.control,
                    loaded_word=f"{loaded_word | 4:016x}",
                    resulting_word=f"{resulting_word | 4:016x}",
                ),
            )
        self.assertEqual(later_readers, [924])
        order = actor_race_verify.build_endpoint_observation_order(
            _replace_endpoint_repetition(self.repetition, actions=tuple(actions))
        )
        lifecycle = order.endpoint.requests_by_client_index[client_index]
        direct_target = order.control_order.target_order.targets.by_action[
            direct.ordinal
        ]
        cached_target = order.control_order.target_order.targets.by_action[
            cached.ordinal
        ]
        assert lifecycle is not None
        assert direct_target is not None and cached_target is not None
        endpoints = (
            order.control_order.target_order.submission_order.interval_order
            .endpoints
        )
        response = endpoints.by_action[direct.ordinal].response
        self.assertEqual(lifecycle.first_eof_source, "script_direct")
        self.assertTrue(
            order.graph.precedes(
                direct_target.event.node,
                lifecycle.first_eof_acknowledgement.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                lifecycle.first_eof_acknowledgement.node,
                response.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                lifecycle.first_eof_acknowledgement.node,
                cached_target.event.node,
            )
        )

    def test_requested_terminal_correlation_uses_only_causal_winners(self) -> None:
        client_index = 50
        request_id = next(
            action.request_id
            for action in self.repetition.actions
            if action.result == "submit_accepted"
            and action.client_index == client_index
        )
        assert request_id is not None
        output = actor_race_history.Output(request_id, 0, 0)
        late_script = _with_endpoint_publications(
            self.repetition,
            client_index,
            (output,),
            (output,),
            terminal_outcome="completed",
        )
        order = actor_race_verify.build_endpoint_observation_order(late_script)
        publication = order.endpoint.requests_by_client_index[client_index]
        assert publication is not None
        self.assertEqual(publication.publications.terminal.outcome, "completed")
        self.assertTrue(
            any(
                observation.source_kind == "script"
                and observation.disposition == "requested"
                for observation in order.control_order.lifecycle.observations
                if observation.owner_client_index == client_index
            )
        )

        cleanup = _exact_cleanup_requested_endpoint_repetition()
        cleanup_receiver = cleanup.cleanup_receivers[0]
        cleanup_output = actor_race_history.Output(
            cleanup_receiver.request_id,
            0,
            0,
        )
        lost_cleanup = _with_endpoint_publications(
            cleanup,
            cleanup_receiver.client_index,
            (cleanup_output,),
            (cleanup_output,),
            terminal_outcome="completed",
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "requested control disposition did not reach a cancelled terminal",
        ):
            actor_race_verify.build_endpoint_observation_order(lost_cleanup)

    def test_cancelled_terminal_requires_a_preterminal_C_publisher(self) -> None:
        client_index = 5
        drop = next(
            action
            for action in self.repetition.actions
            if action.client_index == client_index
            and action.result == "receiver_dropped"
        )
        loaded_word = int(drop.control.loaded_word, 16) | 4
        resulting_word = int(drop.control.resulting_word, 16) | 4
        actions = tuple(
            replace(
                action,
                control=replace(
                    action.control,
                    loaded_word=f"{loaded_word:016x}",
                    resulting_word=f"{resulting_word:016x}",
                    disposition="already_terminal",
                ),
            )
            if action.ordinal == drop.ordinal
            else action
            for action in self.repetition.actions
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cancelled terminal lacks a preterminal C publisher",
        ):
            actor_race_verify.build_endpoint_observation_order(
                _replace_endpoint_repetition(self.repetition, actions=actions)
            )

    def test_opportunistic_eof_is_distinct_and_precedes_response(self) -> None:
        client_index = 16
        ordinal = max(
            action.ordinal
            for action in self.repetition.actions
            if action.client_index == client_index
            and action.kind == "drain"
            and action.result == "drain_empty"
        )
        action = self.repetition.actions[ordinal]
        identity = next(
            candidate.accepted
            for candidate in self.repetition.actions
            if candidate.result == "submit_accepted"
            and candidate.client_index == client_index
        )
        output = actor_race_history.Output(identity.request_id, 0, 7)
        actions = list(self.repetition.actions)
        actions[ordinal] = replace(
            action,
            result="drain_output",
            output=output,
            primary_pop=actor_race_history.PopWitness(
                "primary",
                True,
                identity.endpoint_slot,
                identity.endpoint_generation,
                0,
                1,
                output,
            ),
            opportunistic_eof_pop=actor_race_history.PopWitness(
                "opportunistic_eof",
                True,
                identity.endpoint_slot,
                identity.endpoint_generation,
                1,
                1,
                None,
            ),
        )
        repetition = _with_endpoint_publications(
            _replace_endpoint_repetition(self.repetition, actions=tuple(actions)),
            client_index,
            (output,),
            (),
        )
        order = actor_race_verify.build_endpoint_observation_order(repetition)
        lifecycle = order.endpoint.requests_by_client_index[client_index]
        primary = order.control_order.target_order.targets.by_action[ordinal]
        opportunistic = order.endpoint.opportunistic_by_action[ordinal]
        endpoints = (
            order.control_order.target_order.submission_order.interval_order
            .endpoints
        )
        response = endpoints.by_action[ordinal].response
        assert lifecycle is not None
        assert primary is not None and opportunistic is not None
        self.assertEqual(lifecycle.first_eof_source, "script_opportunistic")
        self.assertTrue(order.graph.precedes(primary.event.node, opportunistic.node))
        self.assertTrue(
            order.graph.precedes(
                lifecycle.endpoint_terminal_publish.node,
                opportunistic.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                opportunistic.node,
                lifecycle.first_eof_acknowledgement.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                lifecycle.first_eof_acknowledgement.node,
                response.node,
            )
        )
        self.assertEqual(
            order.graph.reasons(
                lifecycle.endpoint_terminal_publish.node,
                primary.event.node,
            ),
            (),
        )

    def test_nonempty_cleanup_suffix_is_conserved_without_output_nodes(self) -> None:
        client_index = self.repetition.cleanup_receivers[0].client_index
        request_id = self.repetition.cleanup_receivers[0].request_id
        output = actor_race_history.Output(request_id, 0, 9)
        repetition = _with_endpoint_publications(
            self.repetition,
            client_index,
            (output,),
            (output,),
        )
        order = actor_race_verify.build_endpoint_observation_order(repetition)
        lifecycle = order.endpoint.requests_by_client_index[client_index]
        cleanup = order.endpoint.cleanup_by_client_index[client_index]
        assert lifecycle is not None and cleanup is not None
        self.assertEqual(lifecycle.first_eof_source, "cleanup_receiver")
        self.assertEqual(lifecycle.publications.outputs, (output,))
        self.assertEqual(order.graph.node_count, self.order.graph.node_count)
        self.assertTrue(
            order.graph.precedes(
                cleanup.terminal_acknowledgement.node,
                lifecycle.first_eof_acknowledgement.node,
            )
        )

    def test_undrained_capacity_is_bounded_for_live_and_consumed_requests(
        self,
    ) -> None:
        live_clients = {
            receiver.client_index for receiver in self.repetition.cleanup_receivers
        }
        for client_index in (50, 5):
            with self.subTest(client_index=client_index):
                request_id = next(
                    action.request_id
                    for action in self.repetition.actions
                    if action.result == "submit_accepted"
                    and action.client_index == client_index
                )
                assert request_id is not None
                outputs = tuple(
                    actor_race_history.Output(request_id, index, index + 1)
                    for index in range(3)
                )
                cleanup_outputs = (
                    outputs if client_index in live_clients else ()
                )
                forged = _with_endpoint_publications(
                    self.repetition,
                    client_index,
                    outputs,
                    cleanup_outputs,
                )
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    "retained more than two undrained outputs",
                ):
                    actor_race_verify.build_endpoint_observation_order(forged)

    def test_terminal_stop_rules_reject_impossible_output_prefixes(self) -> None:
        client_index = 50
        request_id = next(
            action.request_id
            for action in self.repetition.actions
            if action.result == "submit_accepted"
            and action.client_index == client_index
        )
        assert request_id is not None
        maximum = actor_race_verify.scheduler.build_descriptors()[client_index][
            "max_new_tokens"
        ]
        cases = (
            (
                "cancelled-eos",
                (actor_race_history.Output(request_id, 0, 0),),
                "cancelled",
                "cancelled after EOS",
            ),
            (
                "cancelled-limit",
                tuple(
                    actor_race_history.Output(request_id, index, 1)
                    for index in range(maximum)
                ),
                "cancelled",
                "cancelled at its generation limit",
            ),
            (
                "continued-after-eos",
                (
                    actor_race_history.Output(request_id, 0, 0),
                    actor_race_history.Output(request_id, 1, 1),
                ),
                "completed",
                "published output after EOS",
            ),
            (
                "completed-early-without-eos",
                (actor_race_history.Output(request_id, 0, 1),),
                "completed",
                "completed early without EOS",
            ),
        )
        for label, outputs, outcome, error in cases:
            with self.subTest(label=label):
                forged = _with_endpoint_publications(
                    self.repetition,
                    client_index,
                    outputs,
                    outputs,
                    terminal_outcome=outcome,
                )
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    error,
                ):
                    actor_race_verify.build_endpoint_observation_order(forged)

    def test_canonical_observation_phases_are_value_only_but_enforced(self) -> None:
        eof_index = next(
            index
            for index, observation in enumerate(self.repetition.observations)
            if observation.kind == "output_eof"
        )
        observations = list(self.repetition.observations)
        eof = observations.pop(eof_index)
        observations.insert(0, eof)
        forged = _replace_endpoint_repetition(
            self.repetition,
            observations=tuple(observations),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "terminal appears after the EOF phase",
        ):
            actor_race_verify.build_endpoint_observation_order(forged)

    def test_observation_and_shell_mutations_fail_closed(self) -> None:
        client_index = 50
        request_id = next(
            action.request_id
            for action in self.repetition.actions
            if action.result == "submit_accepted"
            and action.client_index == client_index
        )
        assert request_id is not None
        outputs = (
            actor_race_history.Output(request_id, 0, 7),
            actor_race_history.Output(request_id, 1, 8),
        )
        valid = _with_endpoint_publications(
            self.repetition,
            client_index,
            outputs,
            outputs,
        )

        def with_observations(
            repetition: SimpleNamespace,
            observations: tuple[actor_race_history.Observation, ...],
        ) -> SimpleNamespace:
            diagnostics = SimpleNamespace(
                action_counter_final=(
                    repetition.diagnostics.action_counter_final
                ),
                cleanup_counter_final=(
                    repetition.diagnostics.cleanup_counter_final
                ),
                recorder_initial=repetition.diagnostics.recorder_initial,
                recorder_final=_recorder_status(len(observations)),
            )
            return _replace_endpoint_repetition(
                repetition,
                diagnostics=diagnostics,
                observations=observations,
            )

        swapped = list(valid.observations)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        one_output = _with_endpoint_publications(
            self.repetition,
            client_index,
            outputs[:1],
            outputs[:1],
        )
        gap = tuple(
            replace(observation, output_index=1)
            if observation.kind == "output"
            else observation
            for observation in one_output.observations
        )
        vocabulary = tuple(
            replace(observation, token_id=32)
            if observation.kind == "output"
            else observation
            for observation in one_output.observations
        )
        missing_terminal = tuple(
            observation
            for observation in self.repetition.observations
            if not (
                observation.kind == "terminal"
                and observation.request_id == request_id
            )
        )
        bad_progress = tuple(
            replace(
                observation,
                committed_positions=observation.committed_positions + 1,
                emitted_tokens=observation.emitted_tokens + 1,
            )
            if observation.kind == "terminal"
            and observation.request_id == request_id
            else observation
            for observation in one_output.observations
        )
        cases = (
            (
                "output-sort",
                with_observations(valid, tuple(swapped)),
                "output observations are not strictly sorted",
            ),
            (
                "output-gap",
                with_observations(one_output, gap),
                "output publications are not consecutive",
            ),
            (
                "vocabulary",
                with_observations(one_output, vocabulary),
                "token is outside the tiny-v3 vocabulary",
            ),
            (
                "missing-terminal",
                with_observations(self.repetition, missing_terminal),
                "has no terminal observation",
            ),
            (
                "terminal-progress",
                with_observations(one_output, bad_progress),
                "terminal/output counts differ",
            ),
        )
        for label, forged, error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    error,
                ):
                    actor_race_verify.build_endpoint_observation_order(forged)

        bool_index = tuple(
            replace(observation, output_index=True)
            if observation.kind == "output"
            else observation
            for observation in one_output.observations
        )
        receiver = one_output.cleanup_receivers[-1]
        shell_cases = (
            (
                "bool-output-index",
                with_observations(one_output, bool_index),
                "output_index must be an integer",
            ),
            (
                "mutable-cleanup-outputs",
                _replace_endpoint_repetition(
                    one_output,
                    cleanup_receivers=tuple(
                        replace(candidate, outputs=list(candidate.outputs))
                        if candidate.client_index == client_index
                        else candidate
                        for candidate in one_output.cleanup_receivers
                    ),
                ),
                "outputs must be an exact immutable tuple",
            ),
            (
                "bool-snapshot-counter",
                _replace_endpoint_repetition(
                    one_output,
                    cleanup_receivers=tuple(
                        replace(
                            candidate,
                            post_drop_quiescent=replace(
                                candidate.post_drop_quiescent,
                                pump_entries=True,
                            ),
                        )
                        if candidate == receiver
                        else candidate
                        for candidate in one_output.cleanup_receivers
                    ),
                ),
                "pump_entries must be an integer",
            ),
        )
        for label, forged, error in shell_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    error,
                ):
                    actor_race_verify.build_endpoint_observation_order(forged)

    def test_missing_eof_and_wrong_cleanup_suffix_fail_conservation(self) -> None:
        missing_observations = self.repetition.observations[:-1]
        missing = _replace_endpoint_repetition(
            self.repetition,
            observations=missing_observations,
            diagnostics=SimpleNamespace(
                action_counter_final=self.repetition.diagnostics.action_counter_final,
                cleanup_counter_final=self.repetition.diagnostics.cleanup_counter_final,
                recorder_initial=self.repetition.diagnostics.recorder_initial,
                recorder_final=_recorder_status(len(missing_observations)),
            ),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "has no first-EOF observation",
        ):
            actor_race_verify.build_endpoint_observation_order(missing)

        receiver = self.repetition.cleanup_receivers[0]
        output = actor_race_history.Output(receiver.request_id, 0, 4)
        valid = _with_endpoint_publications(
            self.repetition,
            receiver.client_index,
            (output,),
            (output,),
        )
        forged_receivers = tuple(
            replace(candidate, outputs=())
            if candidate.client_index == receiver.client_index
            else candidate
            for candidate in valid.cleanup_receivers
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "not the exact FIFO suffix",
        ):
            actor_race_verify.build_endpoint_observation_order(
                _replace_endpoint_repetition(
                    valid,
                    cleanup_receivers=forged_receivers,
                )
            )

    def test_drain_frontier_mutations_fail_closed(self) -> None:
        client_index = 16
        drains = [
            action
            for action in self.repetition.actions
            if action.client_index == client_index
            and action.kind == "drain"
            and action.result == "drain_empty"
        ]
        first = drains[0]
        identity = next(
            action.accepted
            for action in self.repetition.actions
            if action.result == "submit_accepted"
            and action.client_index == client_index
        )

        bad_empty = tuple(
            replace(
                action,
                primary_pop=replace(action.primary_pop, drained_after=1),
            )
            if action.ordinal == first.ordinal
            else action
            for action in self.repetition.actions
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "empty pop has invalid count algebra",
        ):
            actor_race_verify.build_endpoint_observation_order(
                _replace_endpoint_repetition(
                    self.repetition,
                    actions=bad_empty,
                )
            )

        popped = actor_race_history.Output(identity.request_id, 0, 7)
        published = actor_race_history.Output(identity.request_id, 0, 8)
        mismatch_actions = tuple(
            replace(
                action,
                result="drain_output",
                output=popped,
                primary_pop=actor_race_history.PopWitness(
                    "primary",
                    True,
                    identity.endpoint_slot,
                    identity.endpoint_generation,
                    0,
                    1,
                    popped,
                ),
            )
            if action.ordinal == first.ordinal
            else action
            for action in self.repetition.actions
        )
        mismatch = _with_endpoint_publications(
            _replace_endpoint_repetition(
                self.repetition,
                actions=mismatch_actions,
            ),
            client_index,
            (published,),
            (),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "output has no exact publication",
        ):
            actor_race_verify.build_endpoint_observation_order(mismatch)

        last = drains[-1]
        direct_actions = tuple(
            replace(action, result="drain_eof")
            if action.ordinal == last.ordinal
            else action
            for action in self.repetition.actions
        )
        premature = _with_endpoint_publications(
            _replace_endpoint_repetition(
                self.repetition,
                actions=direct_actions,
            ),
            client_index,
            (published,),
            (published,),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "acknowledged EOF before conserving publications",
        ):
            actor_race_verify.build_endpoint_observation_order(premature)

        reentry_action = drains[-2]
        reentry_actions = tuple(
            replace(action, result="drain_eof")
            if action.ordinal == reentry_action.ordinal
            else action
            for action in self.repetition.actions
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "re-entered an endpoint after EOF",
        ):
            actor_race_verify.build_endpoint_observation_order(
                _replace_endpoint_repetition(
                    self.repetition,
                    actions=reentry_actions,
                )
            )

    def test_cleanup_snapshot_reap_progress_is_strict(self) -> None:
        first = self.repetition.cleanup_receivers[0]
        snapshot = first.post_drop_quiescent
        previous = self.repetition.pre_cleanup
        cases = (
            (
                "request-bytes",
                replace(snapshot, request_bytes=previous.request_bytes),
                "did not release request ledger bytes",
            ),
            (
                "pump-entry",
                replace(snapshot, pump_entries=previous.pump_entries),
                "did not advance actor reap progress",
            ),
            (
                "park-epoch",
                replace(snapshot, park_epoch=previous.park_epoch),
                "did not advance actor reap progress",
            ),
            (
                "engine-step",
                replace(snapshot, engine_steps=previous.engine_steps),
                "did not advance actor reap progress",
            ),
            (
                "outstanding",
                replace(snapshot, outstanding_requests=previous.outstanding_requests),
                "did not reap exactly one receiver",
            ),
            (
                "shared-ledger",
                replace(snapshot, shared_bytes=previous.shared_bytes + 1),
                "changed static shared ledger bytes",
            ),
            (
                "pump-hold",
                replace(
                    snapshot,
                    pump_hold_requested=2,
                    pump_hold_observed=2,
                    pump_hold_released=2,
                ),
                "does not conserve the cleanup pump hold",
            ),
        )
        for label, forged_snapshot, error in cases:
            with self.subTest(label=label):
                receivers = (
                    replace(first, post_drop_quiescent=forged_snapshot),
                    *self.repetition.cleanup_receivers[1:],
                )
                forged = _replace_endpoint_repetition(
                    self.repetition,
                    cleanup_receivers=receivers,
                )
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    error,
                ):
                    actor_race_verify.build_endpoint_observation_order(forged)

        retained = replace(
            self.repetition.pre_shutdown,
            outstanding_requests=1,
            request_bytes=1,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "pre_shutdown increased request ledger bytes",
        ):
            actor_race_verify.build_endpoint_observation_order(
                _replace_endpoint_repetition(
                    self.repetition,
                    pre_shutdown=retained,
                )
            )

    def test_preflight_rejects_before_auth_and_snapshots_properties_once(self) -> None:
        source = self.repetition

        class CountingRepetition:
            def __init__(self) -> None:
                self.reads: dict[str, int] = {}

            def _read(self, name: str) -> object:
                self.reads[name] = self.reads.get(name, 0) + 1
                return getattr(source, name)

            actions = property(lambda self: self._read("actions"))
            cleanup_authorities = property(
                lambda self: self._read("cleanup_authorities")
            )
            cleanup_receivers = property(
                lambda self: self._read("cleanup_receivers")
            )
            observations = property(lambda self: self._read("observations"))
            diagnostics = property(lambda self: self._read("diagnostics"))
            pre_cleanup = property(lambda self: self._read("pre_cleanup"))
            pre_shutdown = property(lambda self: self._read("pre_shutdown"))
            shutdown = property(lambda self: self._read("shutdown"))

        counting = CountingRepetition()
        authenticate = actor_race_verify.regenerate_authenticated_action_program
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
            wraps=authenticate,
        ) as authenticate_once:
            actor_race_verify.build_endpoint_observation_order(counting)
        authenticate_once.assert_called_once_with()
        self.assertEqual(
            counting.reads,
            {
                "actions": 1,
                "cleanup_authorities": 1,
                "cleanup_receivers": 1,
                "observations": 1,
                "diagnostics": 1,
                "pre_cleanup": 1,
                "pre_shutdown": 1,
                "shutdown": 1,
            },
        )

        malformed = _replace_endpoint_repetition(
            self.repetition,
            observations=list(self.repetition.observations),
        )
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
        ) as authenticate_never:
            with mock.patch.object(
                actor_race_verify.ReasonedDAG,
                "build",
            ) as graph_never:
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    "exact immutable tuple",
                ):
                    actor_race_verify.build_endpoint_observation_order(malformed)
        authenticate_never.assert_not_called()
        graph_never.assert_not_called()

    def test_descriptor_and_recorder_preflight_are_noninjectable(self) -> None:
        descriptors = actor_race_verify.scheduler.build_descriptors()
        stateful_first = [
            {**descriptor, "prompt": list(descriptor["prompt"])}
            for descriptor in descriptors
        ]
        original_token = stateful_first[0]["prompt"][0]
        stateful_first[0]["prompt"][0] = 1 if original_token != 1 else 2
        with mock.patch.object(
            actor_race_verify.scheduler,
            "build_descriptors",
            side_effect=(stateful_first, descriptors),
        ):
            with mock.patch.object(
                actor_race_verify,
                "regenerate_authenticated_action_program",
            ) as authenticate:
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    "descriptor identity differs from the frozen corpus",
                ):
                    actor_race_verify.build_endpoint_observation_order(
                        self.repetition
                    )
        authenticate.assert_not_called()

        with mock.patch.object(
            actor_race_verify.scheduler,
            "build_descriptors",
            return_value=(),
        ):
            with mock.patch.object(
                actor_race_verify,
                "regenerate_authenticated_action_program",
            ) as authenticate:
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    "exact 64-entry list",
                ):
                    actor_race_verify.build_endpoint_observation_order(
                        self.repetition
                    )
        authenticate.assert_not_called()

        unhealthy = replace(
            self.repetition.diagnostics.recorder_final,
            overflowed=True,
        )
        forged = _replace_endpoint_repetition(
            self.repetition,
            diagnostics=SimpleNamespace(
                action_counter_final=self.repetition.diagnostics.action_counter_final,
                cleanup_counter_final=self.repetition.diagnostics.cleanup_counter_final,
                recorder_initial=self.repetition.diagnostics.recorder_initial,
                recorder_final=unhealthy,
            ),
        )
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
        ) as authenticate:
            with self.assertRaisesRegex(
                actor_race_verify.ActorRaceVerificationError,
                "unhealthy",
            ):
                actor_race_verify.build_endpoint_observation_order(forged)
        authenticate.assert_not_called()


class WakeShutdownOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repetition = _exact_wake_shutdown_repetition()
        cls.order = actor_race_verify.build_wake_shutdown_order(cls.repetition)

    @staticmethod
    def _replace_action(
        repetition: actor_race_history.Repetition,
        ordinal: int,
        **changes: object,
    ) -> actor_race_history.Repetition:
        actions = list(repetition.actions)
        actions[ordinal] = replace(actions[ordinal], **changes)
        return replace(repetition, actions=tuple(actions))

    def test_exact_remaining_budget_and_event_custody(self) -> None:
        lifecycle = self.order.lifecycle
        self.assertEqual(len(lifecycle.in_action_order), 139)
        self.assertEqual(
            self.order.graph.node_count,
            self.order.endpoint_order.graph.node_count + 139 + 1,
        )
        self.assertLessEqual(
            self.order.graph.node_count,
            actor_race_verify.MAX_WAKE_SHUTDOWN_NODES,
        )
        self.assertEqual(actor_race_verify.MAX_WAKE_SHUTDOWN_NODES, 4_258)
        self.assertEqual(actor_race_verify.MAX_WAKE_SHUTDOWN_EDGE_INPUTS, 22_846)
        self.assertEqual(
            tuple(event.kind for event in lifecycle.phase_events),
            ("PreCleanupGate", "PreShutdownGate", "OwnerDonePublish"),
        )
        self.assertTrue(
            self.order.graph.precedes(
                self.order.endpoint_order.endpoint.pre_shutdown.node,
                lifecycle.owner_done_publish.node,
            )
        )
        for wake in lifecycle.in_action_order:
            action_endpoints = (
                self.order.endpoint_order.control_order.target_order.submission_order
                .interval_order.endpoints.by_action[wake.action_ordinal]
            )
            self.assertEqual(wake.signal.kind, "WakeSignal")
            self.assertTrue(
                self.order.graph.precedes(
                    action_endpoints.invocation.node,
                    wake.signal.node,
                )
            )
            self.assertTrue(
                self.order.graph.precedes(
                    wake.signal.node,
                    action_endpoints.response.node,
                )
            )
        self.assertEqual(
            {wake.dirty_was_set for wake in lifecycle.in_action_order},
            {False, True},
        )
        self.assertTrue(
            self.order.graph.precedes(
                lifecycle.in_action_order[0].signal.node,
                lifecycle.in_action_order[-1].signal.node,
            )
        )

    def test_overlapping_coalesced_wakes_remain_unordered(self) -> None:
        left_ordinal, right_ordinal = 94, 95
        actions = list(self.repetition.actions)
        left = actions[left_ordinal]
        right = actions[right_ordinal]
        self.assertEqual((left.kind, right.kind), ("wake", "wake"))
        actions[left_ordinal] = replace(
            left,
            response=right.invocation,
            wake=replace(
                left.wake,
                dirty_was_set=True,
                before_park_epoch=12,
                after_park_epoch=14,
            ),
        )
        actions[right_ordinal] = replace(
            right,
            invocation=left.response,
            wake=replace(
                right.wake,
                dirty_was_set=False,
                before_park_epoch=12,
                after_park_epoch=14,
            ),
        )
        order = actor_race_verify.build_wake_shutdown_order(
            replace(self.repetition, actions=tuple(actions))
        )
        left_wake = order.lifecycle.by_action[left_ordinal]
        right_wake = order.lifecycle.by_action[right_ordinal]
        assert left_wake is not None and right_wake is not None
        self.assertEqual((left_wake.dirty_was_set, right_wake.dirty_was_set), (True, False))
        self.assertFalse(order.graph.precedes(left_wake.signal.node, right_wake.signal.node))
        self.assertFalse(order.graph.precedes(right_wake.signal.node, left_wake.signal.node))

    def test_reverse_epoch_bracket_orders_overlapping_wake_signals(self) -> None:
        left_ordinal, right_ordinal = 94, 95
        actions = list(self.repetition.actions)
        left = actions[left_ordinal]
        right = actions[right_ordinal]
        self.assertEqual((left.kind, right.kind), ("wake", "wake"))
        actions[left_ordinal] = replace(
            left,
            response=right.invocation,
            wake=replace(
                left.wake,
                before_park_epoch=13,
                after_park_epoch=14,
            ),
        )
        actions[right_ordinal] = replace(
            right,
            invocation=left.response,
            wake=replace(
                right.wake,
                before_park_epoch=12,
                after_park_epoch=13,
            ),
        )
        order = actor_race_verify.build_wake_shutdown_order(
            replace(self.repetition, actions=tuple(actions))
        )
        left_wake = order.lifecycle.by_action[left_ordinal]
        right_wake = order.lifecycle.by_action[right_ordinal]
        assert left_wake is not None and right_wake is not None
        self.assertTrue(
            order.graph.precedes(right_wake.signal.node, left_wake.signal.node)
        )
        self.assertFalse(
            order.graph.precedes(left_wake.signal.node, right_wake.signal.node)
        )

    def test_wake_bracket_and_real_time_mutations_fail_closed(self) -> None:
        wakes = [action for action in self.repetition.actions if action.kind == "wake"]
        first, second = wakes[:2]
        cases = (
            (
                "missing-boundary",
                self._replace_action(
                    self.repetition,
                    first.ordinal,
                    wake=replace(first.wake, boundary=False),
                ),
                "omitted its reached boundary",
            ),
            (
                "nonadvancing",
                self._replace_action(
                    self.repetition,
                    first.ordinal,
                    wake=replace(
                        first.wake,
                        after_park_epoch=first.wake.before_park_epoch,
                    ),
                ),
                "did not acknowledge a later park epoch",
            ),
            (
                "after-cleanup",
                self._replace_action(
                    self.repetition,
                    first.ordinal,
                    wake=replace(
                        first.wake,
                        after_park_epoch=self.repetition.pre_cleanup.park_epoch + 1,
                    ),
                ),
                "acknowledgement is after pre_cleanup",
            ),
            (
                "real-time-overlap",
                self._replace_action(
                    self.repetition,
                    first.ordinal,
                    wake=replace(
                        first.wake,
                        after_park_epoch=second.wake.before_park_epoch + 1,
                    ),
                ),
                "acknowledgement overlaps later wake",
            ),
        )
        for label, forged, error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    error,
                ):
                    actor_race_verify.build_wake_shutdown_order(forged)

        nonwake = next(action for action in self.repetition.actions if action.kind != "wake")
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "exact typed sentinel",
        ):
            actor_race_verify.build_wake_shutdown_order(
                self._replace_action(
                    self.repetition,
                    nonwake.ordinal,
                    wake=wakes[0].wake,
                )
            )

    def test_wake_exact_types_and_bool_integer_boundaries_fail_closed(self) -> None:
        wake_action = next(
            action for action in self.repetition.actions if action.kind == "wake"
        )
        nonwake = next(
            action for action in self.repetition.actions if action.kind != "wake"
        )
        cases = (
            (
                "wrong-witness-type",
                self._replace_action(
                    self.repetition,
                    wake_action.ordinal,
                    wake=SimpleNamespace(
                        boundary=True,
                        dirty_was_set=False,
                        before_park_epoch=1,
                        after_park_epoch=2,
                    ),
                ),
                "invalid exact type",
            ),
            (
                "integer-boundary",
                self._replace_action(
                    self.repetition,
                    wake_action.ordinal,
                    wake=replace(wake_action.wake, boundary=1),
                ),
                "boundary must be a boolean",
            ),
            (
                "integer-dirty",
                self._replace_action(
                    self.repetition,
                    wake_action.ordinal,
                    wake=replace(wake_action.wake, dirty_was_set=1),
                ),
                "dirty_was_set must be a boolean",
            ),
            (
                "boolean-epoch",
                self._replace_action(
                    self.repetition,
                    wake_action.ordinal,
                    wake=replace(wake_action.wake, before_park_epoch=False),
                ),
                "before_park_epoch must be an integer",
            ),
            (
                "nonwake-boolean-sentinel",
                self._replace_action(
                    self.repetition,
                    nonwake.ordinal,
                    wake=actor_race_history.WakeWitness(False, False, False, 0),
                ),
                "exact typed sentinel",
            ),
        )
        for label, forged, error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    error,
                ):
                    actor_race_verify.build_wake_shutdown_order(forged)

    def test_final_snapshot_diagnostics_and_overlap_mutations_fail_closed(self) -> None:
        post = self.repetition.post_shutdown
        diagnostics = self.repetition.diagnostics
        overlap = diagnostics.overlap_pair
        large_pump_endpoint = diagnostics.initial_pump_entries + 4_097
        cases = (
            (
                "barrier-held",
                replace(
                    self.repetition,
                    diagnostics=replace(diagnostics, barrier_released=False),
                ),
                "barrier was not released",
            ),
            (
                "engine-delta",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        engine_steps_delta=diagnostics.engine_steps_delta + 1,
                    ),
                ),
                "engine delta mismatch",
            ),
            (
                "pump-delta",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        pump_entries_delta=diagnostics.pump_entries_delta + 1,
                    ),
                ),
                "pump delta mismatch",
            ),
            (
                "pump-delta-cap",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        final_pump_entries=large_pump_endpoint,
                        pump_entries_delta=4_097,
                    ),
                    post_shutdown=replace(
                        post,
                        pump_entries=large_pump_endpoint,
                    ),
                ),
                "pump delta exceeds 4096",
            ),
            (
                "owner-running",
                replace(self.repetition, post_shutdown=replace(post, owner_done=False)),
                "owner state is invalid",
            ),
            (
                "shared-residue",
                replace(self.repetition, post_shutdown=replace(post, shared_bytes=1)),
                "retained actor-owned state",
            ),
            (
                "hold-residue",
                replace(
                    self.repetition,
                    post_shutdown=replace(post, pump_hold_released=0),
                ),
                "incomplete pump hold",
            ),
            (
                "final-endpoint",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        final_pump_entries=diagnostics.final_pump_entries + 1,
                        pump_entries_delta=diagnostics.pump_entries_delta + 1,
                    ),
                ),
                "final diagnostic endpoints mismatch",
            ),
            (
                "overlap-list",
                replace(
                    self.repetition,
                    diagnostics=replace(diagnostics, overlap_pair=list(overlap)),
                ),
                "exact four-position tuple",
            ),
            (
                "repeated-overlap-action",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        overlap_pair=(overlap[0], overlap[1], overlap[0], overlap[1]),
                    ),
                ),
                "repeats one action",
            ),
            (
                "same-producer-overlap",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        overlap_pair=(1, "command", 3, "command"),
                    ),
                ),
                "same producer",
            ),
            (
                "nonoverlapping-actions",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        overlap_pair=(3, "command", 4, "command"),
                    ),
                ),
                "intervals do not overlap",
            ),
            (
                "unsupported-overlap-boundary",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        overlap_pair=(1, "foreign", 2, "command"),
                    ),
                ),
                "unsupported boundary",
            ),
            (
                "unreached-overlap-boundary",
                replace(
                    self.repetition,
                    diagnostics=replace(
                        diagnostics,
                        overlap_pair=(1, "wake", 2, "command"),
                    ),
                ),
                "did not reach its named boundary",
            ),
            (
                "post-park-regression",
                replace(
                    self.repetition,
                    post_shutdown=replace(
                        post,
                        park_epoch=self.repetition.pre_shutdown.park_epoch - 1,
                    ),
                ),
                "regressed a monotone actor counter",
            ),
            (
                "post-engine-regression",
                replace(
                    self.repetition,
                    post_shutdown=replace(
                        post,
                        engine_steps=self.repetition.pre_shutdown.engine_steps - 1,
                    ),
                ),
                "regressed a monotone actor counter",
            ),
            (
                "post-pump-regression",
                replace(
                    self.repetition,
                    post_shutdown=replace(
                        post,
                        pump_entries=self.repetition.pre_shutdown.pump_entries - 1,
                    ),
                ),
                "regressed a monotone actor counter",
            ),
            (
                "post-hold-regression",
                replace(
                    self.repetition,
                    post_shutdown=replace(
                        post,
                        pump_hold_requested=0,
                        pump_hold_observed=0,
                        pump_hold_released=0,
                    ),
                ),
                "regressed a monotone actor counter",
            ),
            (
                "shutdown-residue",
                replace(
                    self.repetition,
                    shutdown=replace(
                        self.repetition.shutdown,
                        discarded_output_events=1,
                    ),
                ),
                "must be exactly zero",
            ),
        )
        for label, forged, error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    error,
                ):
                    actor_race_verify.build_wake_shutdown_order(forged)

        self.assertEqual(post.park_epoch, self.repetition.pre_shutdown.park_epoch)
        self.assertEqual(post.engine_steps, self.repetition.pre_shutdown.engine_steps)

    def test_post_shutdown_may_advance_park_and_engine_counters(self) -> None:
        post = replace(
            self.repetition.post_shutdown,
            park_epoch=self.repetition.post_shutdown.park_epoch + 1,
            engine_steps=self.repetition.post_shutdown.engine_steps + 1,
        )
        diagnostics = replace(
            self.repetition.diagnostics,
            final_engine_steps=post.engine_steps,
            engine_steps_delta=(
                post.engine_steps - self.repetition.diagnostics.initial_engine_steps
            ),
        )
        shutdown = replace(self.repetition.shutdown, engine_steps=post.engine_steps)
        order = actor_race_verify.build_wake_shutdown_order(
            replace(
                self.repetition,
                diagnostics=diagnostics,
                post_shutdown=post,
                shutdown=shutdown,
            )
        )
        self.assertEqual(order.lifecycle.post_shutdown.park_epoch, post.park_epoch)
        self.assertEqual(order.lifecycle.post_shutdown.engine_steps, post.engine_steps)

    def test_full_preflight_reads_root_once_and_fails_before_auth_or_graph(self) -> None:
        source = self.repetition

        class CountingRepetition:
            def __init__(self) -> None:
                self.reads: dict[str, int] = {}

            def _read(self, name: str) -> object:
                self.reads[name] = self.reads.get(name, 0) + 1
                return getattr(source, name)

            actions = property(lambda self: self._read("actions"))
            cleanup_authorities = property(lambda self: self._read("cleanup_authorities"))
            cleanup_receivers = property(lambda self: self._read("cleanup_receivers"))
            diagnostics = property(lambda self: self._read("diagnostics"))
            observations = property(lambda self: self._read("observations"))
            post_shutdown = property(lambda self: self._read("post_shutdown"))
            pre_cleanup = property(lambda self: self._read("pre_cleanup"))
            pre_shutdown = property(lambda self: self._read("pre_shutdown"))
            repetition = property(lambda self: self._read("repetition"))
            shutdown = property(lambda self: self._read("shutdown"))

        counting = CountingRepetition()
        actor_race_verify.build_wake_shutdown_order(counting)
        self.assertEqual(
            counting.reads,
            {
                "actions": 1,
                "cleanup_authorities": 1,
                "cleanup_receivers": 1,
                "diagnostics": 1,
                "observations": 1,
                "post_shutdown": 1,
                "pre_cleanup": 1,
                "pre_shutdown": 1,
                "repetition": 1,
                "shutdown": 1,
            },
        )

        malformed = replace(
            self.repetition,
            post_shutdown=replace(self.repetition.post_shutdown, owner_done=False),
        )
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
        ) as authenticate_never:
            with mock.patch.object(actor_race_verify.ReasonedDAG, "build") as graph_never:
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    "owner state is invalid",
                ):
                    actor_race_verify.build_wake_shutdown_order(malformed)
        authenticate_never.assert_not_called()
        graph_never.assert_not_called()

    def test_streaming_capture_entry_has_exact_root_gates(self) -> None:
        repetitions = tuple(
            replace(self.repetition, repetition=index)
            for index in range(actor_race_verify.EXPECTED_REPETITION_COUNT)
        )
        capture = actor_race_history.Capture(
            actor_race_verify.EXPECTED_REPETITION_COUNT,
            repetitions,
            actor_race_verify._EXPECTED_CAPTURE_SCHEMA,
            actor_race_verify._EXPECTED_WORKLOAD,
        )
        with mock.patch.object(
            actor_race_verify,
            "build_wake_shutdown_order",
            return_value=self.order,
        ) as build:
            report = actor_race_verify.verify_capture_structure(capture)
        self.assertEqual(build.call_count, 32)
        self.assertEqual(report.repetition_count, 32)
        self.assertEqual(len(report.repetitions), 32)
        self.assertTrue(
            all(
                type(item) is actor_race_verify.StructuralRepetitionReport
                for item in report.repetitions
            )
        )

        invalid = (
            replace(capture, schema="runnel.actor-race-history/0"),
            replace(
                capture,
                workload=replace(capture.workload, specification="foreign"),
            ),
            replace(
                capture,
                repetitions=(replace(repetitions[0], repetition=1), *repetitions[1:]),
            ),
        )
        for forged in invalid:
            with self.assertRaises(actor_race_verify.ActorRaceVerificationError):
                actor_race_verify.verify_capture_structure(forged)

    def test_streaming_capture_releases_each_graph_before_the_next_build(self) -> None:
        repetitions = tuple(
            replace(self.repetition, repetition=index)
            for index in range(actor_race_verify.EXPECTED_REPETITION_COUNT)
        )
        capture = actor_race_history.Capture(
            actor_race_verify.EXPECTED_REPETITION_COUNT,
            repetitions,
            actor_race_verify._EXPECTED_CAPTURE_SCHEMA,
            actor_race_verify._EXPECTED_WORKLOAD,
        )

        class EphemeralOrder:
            def __init__(self) -> None:
                self.graph = SimpleNamespace(node_count=4_000, edges=())
                self.lifecycle = SimpleNamespace(in_action_order=(None,) * 139)
                self.endpoint_order = SimpleNamespace(
                    endpoint=SimpleNamespace(requests_by_request_id=())
                )

        live: weakref.WeakSet[EphemeralOrder] = weakref.WeakSet()

        def build(_: actor_race_history.Repetition) -> EphemeralOrder:
            gc.collect()
            self.assertEqual(
                len(live),
                0,
                "the preceding complete DAG survived into the next build",
            )
            order = EphemeralOrder()
            live.add(order)
            return order

        with mock.patch.object(
            actor_race_verify,
            "build_wake_shutdown_order",
            side_effect=build,
        ) as build_mock:
            report = actor_race_verify.verify_capture_structure(capture)
        gc.collect()
        self.assertEqual(build_mock.call_count, 32)
        self.assertEqual(len(report.repetitions), 32)
        self.assertEqual(len(live), 0)


class SubmissionProtocolOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repetition = _exact_submission_repetition()

    def _replace_attempt(
        self,
        attempt: int,
        **changes: object,
    ) -> SimpleNamespace:
        ordinal = _submit_by_attempt(self.repetition, attempt)
        action = self.repetition.actions[ordinal]
        return _replace_repetition_action(
            self.repetition,
            ordinal,
            replace(action, **changes),
        )

    def test_exact_protocol_graph_has_distinct_reasoned_event_custody(self) -> None:
        order = actor_race_verify.build_submission_protocol_order(
            self.repetition
        )
        self.assertEqual(order.accepted_count, 32)
        self.assertEqual(order.rejected_count, 32)
        self.assertEqual(
            order.graph.node_count,
            actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL
            + 5 * actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT
            + 3 * order.accepted_count,
        )
        self.assertEqual(
            len(order.submissions.protocol_events),
            order.graph.node_count
            - actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL,
        )
        self.assertEqual(
            len({event.node for event in order.submissions.protocol_events}),
            len(order.submissions.protocol_events),
        )
        event_kind_counts = {
            kind: sum(
                event.kind == kind
                for event in order.submissions.protocol_events
            )
            for kind in (
                "CommandReserve",
                "ReadyCommit",
                "ActorCommandClaim",
                "ActorCommandRespond",
                "CommandRelease",
                "ControlBind",
                "EndpointBind",
                "RegistryPublish",
            )
        }
        for kind in (
            "CommandReserve",
            "ReadyCommit",
            "ActorCommandClaim",
            "ActorCommandRespond",
            "CommandRelease",
        ):
            self.assertEqual(
                event_kind_counts[kind],
                actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT,
            )
        for kind in ("ControlBind", "EndpointBind", "RegistryPublish"):
            self.assertEqual(event_kind_counts[kind], order.accepted_count)
        self.assertEqual(
            sum(nodes is not None for nodes in order.submissions.by_action),
            actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT,
        )
        self.assertEqual(
            tuple(
                nodes.ready_sequence
                for nodes in order.submissions.by_ready_sequence
            ),
            tuple(
                range(
                    1,
                    actor_race_verify.EXPECTED_IN_RANGE_SUBMIT_COUNT + 1,
                )
            ),
        )

        for edge in order.interval_order.graph.edges:
            self.assertEqual(
                order.graph.reasons(edge.before, edge.after),
                edge.reasons,
            )

        accepted = order.submissions.by_ready_sequence[0]
        accepted_endpoints = order.interval_order.endpoints.by_action[
            accepted.action_ordinal
        ]
        self.assertTrue(accepted.accepted)
        self.assertIsNotNone(accepted.control_bind)
        self.assertIsNotNone(accepted.endpoint_bind)
        self.assertIsNotNone(accepted.registry_publish)
        self.assertNotEqual(
            accepted.command_release.node,
            accepted_endpoints.response.node,
        )
        self.assertEqual(
            len(
                {
                    accepted.command_reserve.node,
                    accepted.ready_commit.node,
                    accepted.actor_command_claim.node,
                    accepted.control_bind.node,
                    accepted.endpoint_bind.node,
                    accepted.actor_command_respond.node,
                    accepted.command_release.node,
                    accepted.registry_publish.node,
                    accepted_endpoints.response.node,
                }
            ),
            9,
        )
        self.assertTrue(
            order.graph.precedes(
                accepted_endpoints.invocation.node,
                accepted.command_reserve.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.command_reserve.node,
                accepted.ready_commit.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.ready_commit.node,
                accepted.actor_command_claim.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.actor_command_claim.node,
                accepted.control_bind.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.actor_command_claim.node,
                accepted.endpoint_bind.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.control_bind.node,
                accepted.actor_command_respond.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.endpoint_bind.node,
                accepted.actor_command_respond.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.actor_command_respond.node,
                accepted.command_release.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.command_release.node,
                accepted.registry_publish.node,
            )
        )
        self.assertTrue(
            order.graph.precedes(
                accepted.registry_publish.node,
                accepted_endpoints.response.node,
            )
        )
        self.assertEqual(
            order.graph.reasons(
                accepted.registry_publish.node,
                accepted_endpoints.response.node,
            ),
            ("registry-publish-before-submit-action-response",),
        )
        self.assertFalse(
            order.graph.precedes(
                accepted.control_bind.node,
                accepted.endpoint_bind.node,
            )
        )
        self.assertFalse(
            order.graph.precedes(
                accepted.endpoint_bind.node,
                accepted.control_bind.node,
            )
        )

        rejected = order.submissions.by_ready_sequence[1]
        rejected_endpoints = order.interval_order.endpoints.by_action[
            rejected.action_ordinal
        ]
        self.assertFalse(rejected.accepted)
        self.assertIsNone(rejected.control_bind)
        self.assertIsNone(rejected.endpoint_bind)
        self.assertIsNone(rejected.registry_publish)
        self.assertEqual(
            order.graph.reasons(
                rejected.actor_command_claim.node,
                rejected.actor_command_respond.node,
            ),
            ("rejected-actor-command-claim-before-respond",),
        )
        self.assertEqual(
            order.graph.reasons(
                rejected.actor_command_respond.node,
                rejected.command_release.node,
            ),
            ("actor-command-respond-before-command-release",),
        )
        self.assertEqual(
            order.graph.reasons(
                rejected.command_release.node,
                rejected_endpoints.response.node,
            ),
            ("rejected-command-release-before-submit-action-response",),
        )

    def test_ready_actor_fifo_and_per_slot_ticket_chains_are_explicit(self) -> None:
        order = actor_race_verify.build_submission_protocol_order(
            self.repetition
        )
        ready = order.submissions.by_ready_sequence
        for previous, current in zip(ready, ready[1:]):
            self.assertEqual(
                order.graph.reasons(
                    previous.ready_commit.node,
                    current.ready_commit.node,
                ),
                ("command-ready-sequence-order",),
            )
            self.assertEqual(
                order.graph.reasons(
                    previous.actor_command_respond.node,
                    current.actor_command_claim.node,
                ),
                ("actor-command-ready-sequence-order",),
            )
            self.assertTrue(
                order.graph.precedes(
                    previous.actor_command_respond.node,
                    current.actor_command_claim.node,
                )
            )
            if current.accepted:
                self.assertTrue(
                    order.graph.precedes(
                        previous.actor_command_respond.node,
                        current.control_bind.node,
                    )
                )
                self.assertTrue(
                    order.graph.precedes(
                        previous.actor_command_respond.node,
                        current.endpoint_bind.node,
                    )
                )
        for slot in range(8):
            slot_nodes = tuple(
                nodes for nodes in ready if nodes.command_slot == slot
            )
            self.assertEqual(
                tuple(nodes.command_ticket for nodes in slot_nodes),
                tuple(range(1, len(slot_nodes) + 1)),
            )
            for previous, current in zip(slot_nodes, slot_nodes[1:]):
                self.assertEqual(
                    order.graph.reasons(
                        previous.command_release.node,
                        current.command_reserve.node,
                    ),
                    (f"command-slot-{slot}-ticket-order",),
                )

    def test_public_entry_point_authenticates_locally_and_accepts_no_program(self) -> None:
        authenticate = actor_race_verify.regenerate_authenticated_action_program
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
            wraps=authenticate,
        ) as authenticate_once:
            actor_race_verify.build_submission_protocol_order(self.repetition)
        authenticate_once.assert_called_once_with()
        with self.assertRaises(TypeError):
            actor_race_verify.build_submission_protocol_order(
                self.repetition,
                program=authenticate(),
            )

    def test_command_coverage_ready_sequence_and_tickets_fail_closed(self) -> None:
        in_range_zero = _submit_by_attempt(self.repetition, 0)
        exhausted = _submit_by_attempt(self.repetition, 64)
        targeted = next(
            action.ordinal
            for action in self.repetition.actions
            if action.kind != "submit"
        )
        cases: tuple[tuple[str, SimpleNamespace], ...] = (
            (
                "invalid decoded type",
                self._replace_attempt(
                    0,
                    command=SimpleNamespace(
                        boundary=True,
                        slot=0,
                        ticket=1,
                        ready_sequence=1,
                    ),
                ),
            ),
            (
                "omitted",
                self._replace_attempt(
                    0,
                    command=replace(
                        self.repetition.actions[in_range_zero].command,
                        boundary=False,
                    ),
                ),
            ),
            (
                "duplicated",
                self._replace_attempt(
                    1,
                    command=replace(
                        self.repetition.actions[
                            _submit_by_attempt(self.repetition, 1)
                        ].command,
                        ready_sequence=1,
                    ),
                ),
            ),
            (
                "tickets are not exactly",
                self._replace_attempt(
                    8,
                    command=replace(
                        self.repetition.actions[
                            _submit_by_attempt(self.repetition, 8)
                        ].command,
                        ticket=3,
                    ),
                ),
            ),
            (
                "typed sentinel",
                _replace_repetition_action(
                    self.repetition,
                    exhausted,
                    replace(
                        self.repetition.actions[exhausted],
                        command=actor_race_history.CommandWitness(
                            True, 0, 1, 1
                        ),
                    ),
                ),
            ),
            (
                "typed sentinel",
                _replace_repetition_action(
                    self.repetition,
                    targeted,
                    replace(
                        self.repetition.actions[targeted],
                        command=actor_race_history.CommandWitness(
                            True, 0, 1, 1
                        ),
                    ),
                ),
            ),
            (
                "must be an integer",
                self._replace_attempt(
                    0,
                    command=replace(
                        self.repetition.actions[in_range_zero].command,
                        ready_sequence=False,
                    ),
                ),
            ),
        )
        for diagnostic, forged in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_submission_protocol_order(forged)

    def test_result_error_identity_and_generation_gates_fail_closed(self) -> None:
        accepted_zero = _submit_by_attempt(self.repetition, 0)
        accepted_two = _submit_by_attempt(self.repetition, 2)
        accepted_action = self.repetition.actions[accepted_zero]
        second_accepted = self.repetition.actions[accepted_two]
        rejected_one = _submit_by_attempt(self.repetition, 1)
        rejected_action = self.repetition.actions[rejected_one]
        cases: tuple[tuple[str, SimpleNamespace], ...] = (
            (
                "invalid decoded type",
                self._replace_attempt(
                    0,
                    accepted=SimpleNamespace(
                        boundary=True,
                        request_id=1,
                        control_slot=0,
                        control_generation=1,
                        endpoint_slot=5,
                        endpoint_generation=1,
                    ),
                ),
            ),
            (
                "exact typed sentinel",
                self._replace_attempt(
                    1,
                    error=SimpleNamespace(
                        code="resource_exhausted",
                        category="resource_exhausted",
                        resource="request slot count",
                        required=16,
                        limit=16,
                    ),
                ),
            ),
            (
                "exact typed sentinel",
                self._replace_attempt(
                    1,
                    error=replace(rejected_action.error, required=17),
                ),
            ),
            (
                "invalid in-range result spelling",
                self._replace_attempt(
                    1,
                    result="submit_offer_exhausted",
                    error=None,
                ),
            ),
            (
                "does not match the action request identity",
                self._replace_attempt(
                    0,
                    accepted=replace(
                        accepted_action.accepted,
                        request_id=2,
                    ),
                ),
            ),
            (
                "exactly 1..N",
                self._replace_attempt(
                    2,
                    request_id=3,
                    accepted=replace(
                        second_accepted.accepted,
                        request_id=3,
                    ),
                ),
            ),
            (
                "control slot 0 generations",
                self._replace_attempt(
                    0,
                    accepted=replace(
                        accepted_action.accepted,
                        control_generation=2,
                    ),
                ),
            ),
            (
                "endpoint slot 5 generations",
                self._replace_attempt(
                    0,
                    accepted=replace(
                        accepted_action.accepted,
                        endpoint_generation=2,
                    ),
                ),
            ),
        )
        for diagnostic, forged in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_submission_protocol_order(forged)

    def test_submit_zero_effect_and_shutdown_accounting_gates_fail_closed(self) -> None:
        accepted_zero = _submit_by_attempt(self.repetition, 0)
        action = self.repetition.actions[accepted_zero]
        action_cases: tuple[tuple[str, actor_race_history.Action], ...] = (
            (
                "retained an output",
                replace(
                    action,
                    output=actor_race_history.Output(1, 0, 1),
                ),
            ),
            (
                "control witness",
                replace(
                    action,
                    control=replace(_CONTROL_SENTINEL, boundary=True),
                ),
            ),
            (
                "primary-pop witness",
                replace(
                    action,
                    primary_pop=replace(
                        _PRIMARY_POP_SENTINEL,
                        boundary=True,
                    ),
                ),
            ),
            (
                "opportunistic-pop witness",
                replace(
                    action,
                    opportunistic_eof_pop=replace(
                        _OPPORTUNISTIC_POP_SENTINEL,
                        boundary=True,
                    ),
                ),
            ),
            ("cached EOF", replace(action, cached_eof=True)),
            (
                "wake witness",
                replace(action, wake=replace(_WAKE_SENTINEL, boundary=True)),
            ),
        )
        for diagnostic, forged_action in action_cases:
            with self.subTest(action_effect=diagnostic):
                forged = _replace_repetition_action(
                    self.repetition,
                    accepted_zero,
                    forged_action,
                )
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_submission_protocol_order(forged)

    def test_exact_sentinels_reject_numeric_type_confusion(self) -> None:
        accepted_zero = _submit_by_attempt(self.repetition, 0)
        rejected_one = _submit_by_attempt(self.repetition, 1)
        exhausted = _submit_by_attempt(self.repetition, 64)
        cases = (
            (
                "exact typed sentinel",
                self._replace_attempt(
                    1,
                    accepted=replace(_ACCEPTED_SENTINEL, request_id=False),
                ),
            ),
            (
                "exact typed sentinel",
                self._replace_attempt(
                    0,
                    control=replace(_CONTROL_SENTINEL, boundary=0),
                ),
            ),
            (
                "exact typed sentinel",
                self._replace_attempt(
                    0,
                    primary_pop=replace(
                        _PRIMARY_POP_SENTINEL,
                        boundary=0,
                    ),
                ),
            ),
            (
                "exact typed sentinel",
                self._replace_attempt(
                    0,
                    opportunistic_eof_pop=replace(
                        _OPPORTUNISTIC_POP_SENTINEL,
                        boundary=0,
                    ),
                ),
            ),
            (
                "exact typed sentinel",
                self._replace_attempt(
                    0,
                    wake=replace(_WAKE_SENTINEL, boundary=0),
                ),
            ),
            (
                "exact typed sentinel",
                self._replace_attempt(
                    1,
                    error=replace(
                        self.repetition.actions[rejected_one].error,
                        required=16.0,
                    ),
                ),
            ),
            (
                "must be an integer",
                _replace_repetition_action(
                    self.repetition,
                    exhausted,
                    replace(
                        self.repetition.actions[exhausted],
                        command=replace(_COMMAND_SENTINEL, slot=False),
                    ),
                ),
            ),
        )
        for diagnostic, forged in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_submission_protocol_order(forged)

        # Keep these ordinals live in the fixture contract: the test above
        # must exercise accepted, rejected, and exhausted submit spellings.
        self.assertNotEqual(accepted_zero, rejected_one)
        self.assertNotEqual(rejected_one, exhausted)

        shutdown_cases = (
            (
                "invalid decoded type",
                SimpleNamespace(
                    **{
                        field: getattr(self.repetition.shutdown, field)
                        for field in (
                            "accepted_submissions",
                            "discarded_output_events",
                            "engine_steps",
                            "rejected_submissions",
                            "released_request_bytes",
                            "remaining_shared_bytes",
                            "shutdown_cancellations",
                            "terminated_requests",
                        )
                    }
                ),
            ),
            (
                "accepted_submissions",
                replace(
                    self.repetition.shutdown,
                    accepted_submissions=31,
                ),
            ),
            (
                "rejected_submissions",
                replace(
                    self.repetition.shutdown,
                    rejected_submissions=31,
                ),
            ),
            (
                "discarded_output_events",
                replace(
                    self.repetition.shutdown,
                    discarded_output_events=1,
                ),
            ),
            (
                "released_request_bytes",
                replace(
                    self.repetition.shutdown,
                    released_request_bytes=1,
                ),
            ),
            (
                "remaining_shared_bytes",
                replace(
                    self.repetition.shutdown,
                    remaining_shared_bytes=1,
                ),
            ),
            (
                "shutdown_cancellations",
                replace(
                    self.repetition.shutdown,
                    shutdown_cancellations=1,
                ),
            ),
            (
                "terminated_requests",
                replace(
                    self.repetition.shutdown,
                    terminated_requests=1,
                ),
            ),
        )
        for diagnostic, shutdown in shutdown_cases:
            with self.subTest(shutdown=diagnostic):
                forged = SimpleNamespace(
                    actions=self.repetition.actions,
                    diagnostics=self.repetition.diagnostics,
                    shutdown=shutdown,
                )
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    actor_race_verify.build_submission_protocol_order(forged)

    def test_contradictory_ready_and_ticket_orders_form_a_reasoned_cycle(self) -> None:
        first_ordinal = _submit_by_attempt(self.repetition, 1)
        second_ordinal = _submit_by_attempt(self.repetition, 9)
        actions = list(self.repetition.actions)
        first = actions[first_ordinal]
        second = actions[second_ordinal]
        actions[first_ordinal] = replace(
            first,
            command=replace(
                first.command,
                ready_sequence=second.command.ready_sequence,
            ),
        )
        actions[second_ordinal] = replace(
            second,
            command=replace(
                second.command,
                ready_sequence=first.command.ready_sequence,
            ),
        )
        forged = SimpleNamespace(
            actions=tuple(actions),
            diagnostics=self.repetition.diagnostics,
            shutdown=self.repetition.shutdown,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "cycle detected",
        ):
            actor_race_verify.build_submission_protocol_order(forged)

    def test_protocol_planner_exact_caps_fail_before_retaining_excess(self) -> None:
        self.assertEqual(actor_race_verify.MAX_PROTOCOL_NODES, 4_258)
        self.assertEqual(
            sum(
                count
                for _, count in actor_race_verify._PROTOCOL_NODE_BUDGET
            ),
            actor_race_verify.MAX_PROTOCOL_NODES,
        )
        planner = actor_race_verify._ProtocolGraphPlanner(
            actor_race_verify.MAX_PROTOCOL_NODES - 1
        )
        planner.allocate(0, "CommandReserve")
        self.assertEqual(
            planner.node_count,
            actor_race_verify.MAX_PROTOCOL_NODES,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            rf"{actor_race_verify.MAX_PROTOCOL_NODES}-node limit",
        ):
            planner.allocate(0, "ReadyCommit")
        self.assertEqual(
            planner.node_count,
            actor_race_verify.MAX_PROTOCOL_NODES,
        )
        self.assertEqual(len(planner.protocol_events), 1)

        planner = actor_race_verify._ProtocolGraphPlanner(2)
        for _ in range(actor_race_verify.MAX_PROTOCOL_EDGE_INPUTS):
            planner.edge(0, 1, "bounded-duplicate")
        self.assertEqual(
            planner.edge_input_count,
            actor_race_verify.MAX_PROTOCOL_EDGE_INPUTS,
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            rf"{actor_race_verify.MAX_PROTOCOL_EDGE_INPUTS}-constraint input limit",
        ):
            planner.edge(0, 1, "first-excess")
        self.assertEqual(
            planner.edge_input_count,
            actor_race_verify.MAX_PROTOCOL_EDGE_INPUTS,
        )

    def test_protocol_planner_layers_every_reason_without_loss(self) -> None:
        base = actor_race_verify.ReasonedDAG.build(
            2,
            (
                actor_race_verify.EdgeConstraint(0, 1, "second"),
                actor_race_verify.EdgeConstraint(0, 1, "first"),
            ),
        )
        planner = actor_race_verify._ProtocolGraphPlanner(2)
        planner.layer(base)
        layered = planner.build()
        self.assertEqual(layered.reasons(0, 1), ("first", "second"))


class ActionIntervalOrderTests(unittest.TestCase):
    def test_exact_builder_returns_immutable_explicit_endpoint_mapping(self) -> None:
        order = actor_race_verify.build_action_interval_order(
            _exact_interval_repetition()
        )
        self.assertEqual(
            order.graph.node_count,
            actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL,
        )
        self.assertEqual(
            len(order.endpoints.by_action),
            actor_race_verify.EXPECTED_ACTION_COUNT,
        )
        self.assertLessEqual(
            len(order.graph.edges),
            actor_race_verify.EXPECTED_ACTION_COUNT * 3,
        )
        self.assertEqual(
            tuple(endpoint.counter for endpoint in order.endpoints.by_counter),
            tuple(
                range(1, actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL + 1)
            ),
        )
        for ordinal, endpoints in enumerate(order.endpoints.by_action):
            self.assertEqual(endpoints.invocation.kind, "invoke")
            self.assertEqual(endpoints.response.kind, "respond")
            self.assertIs(
                order.endpoints.by_node[ordinal * 2],
                endpoints.invocation,
            )
            self.assertIs(
                order.endpoints.by_node[ordinal * 2 + 1],
                endpoints.response,
            )
            self.assertIs(
                order.endpoints.by_counter[
                    endpoints.invocation.counter - 1
                ],
                endpoints.invocation,
            )
            self.assertIs(
                order.endpoints.by_counter[endpoints.response.counter - 1],
                endpoints.response,
            )

    def test_compressed_graph_matches_exhaustive_random_small_histories(self) -> None:
        random_source = random.Random(0x5EED_C0DE)
        for case in range(160):
            action_count = random_source.randint(1, 18)
            actions = _random_valid_history(random_source, action_count)
            with self.subTest(case=case, action_count=action_count):
                compressed = _small_interval_order(actions)
                exhaustive = _exhaustive_interval_graph(actions)
                self.assertEqual(
                    compressed.graph.reachable,
                    exhaustive.reachable,
                )
                cross_edges = sum(
                    any("latest-response" in reason for reason in edge.reasons)
                    for edge in compressed.graph.edges
                )
                self.assertLessEqual(cross_edges, action_count)

                for ordinal, action in enumerate(actions):
                    endpoints = compressed.endpoints.by_action[ordinal]
                    self.assertEqual(endpoints.ordinal, ordinal)
                    self.assertEqual(endpoints.producer, action.producer)
                    self.assertEqual(
                        endpoints.invocation.counter,
                        action.invocation,
                    )
                    self.assertEqual(
                        endpoints.response.counter,
                        action.response,
                    )

    def test_overlapping_cross_producer_intervals_remain_unordered(self) -> None:
        actions = (
            _ObservedAction(0, 0, "drain", None, 0, 1, 4),
            _ObservedAction(1, 1, "wake", None, 1, 2, 3),
        )
        graph = _small_interval_order(actions).graph
        for left in (0, 1):
            for right in (2, 3):
                self.assertFalse(graph.precedes(left, right))
                self.assertFalse(graph.precedes(right, left))

    def test_plain_integer_endpoint_and_identity_fields_fail_closed(self) -> None:
        actions = (
            _ObservedAction(0, 0, "drain", None, 0, 1, 2),
            _ObservedAction(1, 1, "wake", None, 1, 3, 4),
        )
        mutations = {
            "ordinal": (replace(actions[0], ordinal=False), actions[1]),
            "producer": (replace(actions[0], producer=False), actions[1]),
            "invocation": (
                replace(actions[0], invocation=False),
                actions[1],
            ),
            "response": (replace(actions[0], response=False), actions[1]),
        }
        for field, changed in mutations.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    "must be an integer",
                ):
                    _small_interval_order(changed)

        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "action_counter_final must be an integer",
        ):
            actor_race_verify._build_action_interval_order(
                actions,
                False,
                expected_action_count=2,
            )

    def test_endpoint_multiset_interval_and_program_order_fail_closed(self) -> None:
        cases = (
            (
                "outside the exact counter range",
                (
                    _ObservedAction(0, 0, "drain", None, 0, 1, 2),
                    _ObservedAction(1, 1, "wake", None, 1, 3, 5),
                ),
            ),
            (
                "counter 2 is duplicated",
                (
                    _ObservedAction(0, 0, "drain", None, 0, 1, 2),
                    _ObservedAction(1, 1, "wake", None, 1, 2, 4),
                ),
            ),
            (
                "response must follow",
                (
                    _ObservedAction(0, 0, "drain", None, 0, 2, 1),
                    _ObservedAction(1, 1, "wake", None, 1, 3, 4),
                ),
            ),
            (
                "violate program order",
                (
                    _ObservedAction(0, 0, "drain", None, 0, 1, 4),
                    _ObservedAction(1, 0, "wake", None, 1, 2, 3),
                ),
            ),
        )
        for diagnostic, actions in cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    diagnostic,
                ):
                    _small_interval_order(actions)

    def test_exact_count_and_final_counter_gates_fail_before_graph_build(self) -> None:
        repetition = _exact_interval_repetition()
        cases = (
            SimpleNamespace(
                actions=repetition.actions[:-1],
                diagnostics=repetition.diagnostics,
            ),
            SimpleNamespace(
                actions=repetition.actions,
                diagnostics=SimpleNamespace(action_counter_final=2_047),
            ),
            SimpleNamespace(
                actions=repetition.actions,
                diagnostics=SimpleNamespace(action_counter_final=False),
            ),
        )
        for forged in cases:
            with self.subTest(
                action_count=len(forged.actions),
                counter=forged.diagnostics.action_counter_final,
            ):
                with mock.patch.object(
                    actor_race_verify.ReasonedDAG,
                    "build",
                ) as graph_build:
                    with self.assertRaises(
                        actor_race_verify.ActorRaceVerificationError
                    ):
                        actor_race_verify.build_action_interval_order(forged)
                graph_build.assert_not_called()

    def test_public_builder_rejects_forged_static_producer_before_graph(self) -> None:
        repetition = _exact_interval_repetition()
        forged = SimpleNamespace(
            actions=tuple(
                replace(action, producer=0) for action in repetition.actions
            ),
            diagnostics=repetition.diagnostics,
        )
        with mock.patch.object(
            actor_race_verify.ReasonedDAG,
            "build",
        ) as graph_build:
            with self.assertRaisesRegex(
                actor_race_verify.ActorRaceVerificationError,
                "producer mismatch",
            ):
                actor_race_verify.build_action_interval_order(forged)
        graph_build.assert_not_called()


class StaticActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_actions = (
            _action(0, 0, "drain", client_index=2),
            _action(1, 1, "submit", submit_attempt=1, client_index=1),
            _action(2, 0, "wake", client_index=9),
        )
        self.program = _program(self.expected_actions)

    def test_small_static_program_accepts_exact_fields_only(self) -> None:
        observed = tuple(_observe(action) for action in self.expected_actions)
        actor_race_verify._verify_static_actions(observed, self.program)

        # Dynamic interval values are outside static custody and cannot create
        # an accidental counter-derived total order in this layer.
        counter_scrambled = (
            replace(observed[0], invocation=900, response=901),
            replace(observed[1], invocation=3, response=4),
            replace(observed[2], invocation=50, response=51),
        )
        actor_race_verify._verify_static_actions(counter_scrambled, self.program)

    def test_each_static_field_mismatch_fails_at_its_exact_action(self) -> None:
        observed = [_observe(action) for action in self.expected_actions]
        mutations = {
            "ordinal": replace(observed[1], ordinal=9),
            "producer": replace(observed[1], producer=0),
            "kind": replace(observed[1], kind="drain"),
            "submit_attempt": replace(observed[1], submit_attempt=2),
            "client_index": replace(observed[1], client_index=2),
        }
        for field, mutation in mutations.items():
            with self.subTest(field=field):
                changed = list(observed)
                changed[1] = mutation
                with self.assertRaisesRegex(
                    actor_race_verify.ActorRaceVerificationError,
                    rf"action 1 {field} mismatch",
                ):
                    actor_race_verify._verify_static_actions(changed, self.program)

    def test_count_gates_are_checked_independently(self) -> None:
        observed = tuple(_observe(action) for action in self.expected_actions)
        bad_kinds = replace(self.program, kind_counts=(0, 0, 0, 0, 0))
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "inconsistent kind-count gates",
        ):
            actor_race_verify._verify_static_actions(observed, bad_kinds)
        bad_producers = replace(self.program, producer_counts=(3, 0))
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "inconsistent producer-count gates",
        ):
            actor_race_verify._verify_static_actions(observed, bad_producers)

    def test_boolean_ordinals_and_producers_fail_closed(self) -> None:
        observed = tuple(_observe(action) for action in self.expected_actions)
        forged = replace(
            self.program,
            actions=(replace(self.expected_actions[0], producer=False),)
            + self.expected_actions[1:],
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "producer must be an integer",
        ):
            actor_race_verify._verify_static_actions(observed, forged)

    def test_program_order_has_no_cross_producer_ordinal_edges(self) -> None:
        actions = (
            _ObservedAction(0, 0, "drain", None, 0, 100, 110),
            _ObservedAction(1, 1, "drain", None, 1, 1, 5),
            _ObservedAction(2, 0, "wake", None, 8, 120, 130),
            _ObservedAction(3, 1, "wake", None, 9, 6, 7),
        )
        edges = actor_race_verify.producer_action_interval_edges(actions)
        graph = actor_race_verify.ReasonedDAG.build(len(actions), edges)
        self.assertEqual(
            edges,
            (
                actor_race_verify.EdgeConstraint(
                    0, 2, "producer-0-response-before-invocation"
                ),
                actor_race_verify.EdgeConstraint(
                    1, 3, "producer-1-response-before-invocation"
                ),
            ),
        )
        self.assertTrue(graph.precedes(0, 2))
        self.assertTrue(graph.precedes(1, 3))
        self.assertFalse(graph.precedes(0, 1))
        self.assertFalse(graph.precedes(1, 2))

    def test_program_order_requires_strict_witnessed_intervals(self) -> None:
        actions = (
            _ObservedAction(0, 0, "drain", None, 0, 10, 20),
            _ObservedAction(1, 0, "wake", None, 1, 20, 30),
        )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "violate program order",
        ):
            actor_race_verify.producer_action_interval_edges(actions)
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "producer must be an integer",
        ):
            actor_race_verify.producer_action_interval_edges(
                (replace(actions[0], producer=False),)
            )
        with self.assertRaisesRegex(
            actor_race_verify.ActorRaceVerificationError,
            "lacks interval fields",
        ):
            actor_race_verify.producer_action_interval_edges(
                (SimpleNamespace(ordinal=0, producer=0, invocation=1),)
            )

    def test_full_program_is_authenticated_and_has_frozen_gates(self) -> None:
        program = actor_race_verify.regenerate_authenticated_action_program()
        self.assertEqual(len(program.actions), 1_024)
        self.assertEqual(
            program.kind_counts,
            actor_race_verify.EXPECTED_KIND_COUNTS,
        )
        self.assertEqual(
            program.producer_counts,
            actor_race_verify.EXPECTED_PRODUCER_COUNTS,
        )
        self.assertEqual(
            program.fixture_id,
            actor_race_verify.EXPECTED_FIXTURE_ID,
        )
        self.assertEqual(
            program.fixture_file_sha256,
            actor_race_verify.EXPECTED_FIXTURE_FILE_SHA256,
        )
        actor_race_verify._verify_static_actions(program.actions, program)

    def test_capture_entry_point_authenticates_locally_once(self) -> None:
        repetition = SimpleNamespace(actions=self.program.actions)
        # A forged program cannot be supplied to either public entry point.
        with self.assertRaises(TypeError):
            actor_race_verify.verify_repetition_static_actions(
                repetition,
                program=self.program,
            )

        full_program = actor_race_verify.regenerate_authenticated_action_program()
        decoded = SimpleNamespace(
            repetition_count=32,
            repetitions=tuple(
                SimpleNamespace(actions=full_program.actions) for _ in range(32)
            ),
        )
        authenticate = actor_race_verify.regenerate_authenticated_action_program
        with mock.patch.object(
            actor_race_verify,
            "regenerate_authenticated_action_program",
            wraps=authenticate,
        ) as authenticate_once:
            authenticated = actor_race_verify.verify_capture_static_actions(decoded)
        authenticate_once.assert_called_once_with()
        self.assertEqual(authenticated.fixture_id, full_program.fixture_id)

    def test_capture_entry_point_rejects_nonexact_custody_before_auth(self) -> None:
        cases = ((0, 0), (31, 31), (32, 31), (33, 33), (False, 32))
        for declared, actual in cases:
            with self.subTest(declared=declared, actual=actual):
                decoded = SimpleNamespace(
                    repetition_count=declared,
                    repetitions=tuple(SimpleNamespace() for _ in range(actual)),
                )
                with mock.patch.object(
                    actor_race_verify,
                    "regenerate_authenticated_action_program",
                ) as authenticate:
                    with self.assertRaises(
                        actor_race_verify.ActorRaceVerificationError
                    ):
                        actor_race_verify.verify_capture_static_actions(decoded)
                authenticate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
