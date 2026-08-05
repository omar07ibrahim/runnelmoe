from __future__ import annotations

from dataclasses import dataclass, replace
import random
from types import SimpleNamespace
import unittest
from unittest import mock

from oracle import actor_race_verify


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
