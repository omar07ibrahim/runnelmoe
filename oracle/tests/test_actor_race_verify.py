from __future__ import annotations

from dataclasses import dataclass, replace
import random
from types import SimpleNamespace
import unittest
from unittest import mock

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
    program = actor_race_verify.regenerate_authenticated_action_program()
    actions = tuple(
        _ObservedAction(
            action.ordinal,
            action.producer,
            action.kind,
            action.submit_attempt,
            action.client_index,
            action.ordinal * 2 + 1,
            action.ordinal * 2 + 2,
        )
        for action in program.actions
    )
    return SimpleNamespace(
        actions=actions,
        diagnostics=SimpleNamespace(
            action_counter_final=(
                actor_race_verify.EXPECTED_ACTION_COUNTER_FINAL
            )
        ),
    )


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


def _exact_submission_repetition() -> SimpleNamespace:
    program = actor_race_verify.regenerate_authenticated_action_program()
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
                if attempt % 2 == 0:
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
