#!/usr/bin/env python3
"""Independent bounded semantic primitives for actor race-history captures.

This layer owns three deliberately narrow responsibilities:

* a deterministic, reason-labelled partial-order DAG with bitset reachability;
* authentication and exact comparison of the frozen scheduler action program.
* exact action invocation/response custody and its witnessed partial order.

It does not interpret command, control-word, object-boundary, or terminal
witnesses.  Action interval counters are mapped to explicit Invoke and Respond
nodes; they are never promoted into a counter-derived global execution order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import heapq
from typing import Any, NoReturn

from oracle import scheduler


MAX_DAG_NODES = 8_192
MAX_DAG_EDGES = 65_536
MAX_EDGE_INPUTS = 262_144
MAX_REASONS_PER_EDGE = 8
MAX_REASON_BYTES = 128
MAX_CYCLE_DIAGNOSTIC_EDGES = 8

CAPTURE_ACTION_KINDS = (
    "submit",
    "cancel",
    "receiver_drop",
    "drain",
    "wake",
)
EXPECTED_ACTION_VECTOR_ID = (
    "sha256:430810784f31659367ecc4fecb5cf8693b4758176debe4a4769dc7cc62611b73"
)
EXPECTED_FIXTURE_ID = (
    "sha256:5010492fb74eda207511b26811992ed4779814185b9f184663b37a37747bd051"
)
EXPECTED_FIXTURE_FILE_SHA256 = (
    "sha256:eca1faeee91a41d19d98be7ffdad6fc5cebb9027f3e7a634c01ea1cc394fb574"
)
EXPECTED_KIND_COUNTS = (206, 220, 222, 237, 139)
EXPECTED_PRODUCER_COUNTS = (518, 506)
EXPECTED_REPETITION_COUNT = 32
EXPECTED_ACTION_COUNT = 1_024
EXPECTED_ACTION_COUNTER_FINAL = 2_048
_SCHEDULER_TO_CAPTURE_KIND = {
    "submit": "submit",
    "cancel": "cancel",
    "drop": "receiver_drop",
    "drain": "drain",
    "wake": "wake",
}


class ActorRaceVerificationError(ValueError):
    """A stable action-custody or semantic-graph verification failure."""


def _fail(message: str) -> NoReturn:
    raise ActorRaceVerificationError(message)


def _plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _bounded_node(value: Any, node_count: int, label: str) -> int:
    value = _plain_int(value, label)
    if value < 0 or value >= node_count:
        _fail(f"{label} is outside the graph")
    return value


def _reason(value: Any) -> str:
    if not isinstance(value, str):
        _fail("edge reason must be a string")
    # Reject attacker-sized values before any operation that duplicates their
    # storage.  For accepted ASCII strings, code-point and byte lengths match.
    if not value or len(value) > MAX_REASON_BYTES:
        _fail("edge reason must contain 1..128 bytes")
    if not value.isascii():
        _fail("edge reason must contain only ASCII")
    if any(
        ord(character) < 0x20 or ord(character) > 0x7E
        for character in value
    ):
        _fail("edge reason must contain only printable ASCII")
    return value


@dataclass(frozen=True, slots=True)
class EdgeConstraint:
    """One labelled requirement that ``before`` happens before ``after``."""

    before: int
    after: int
    reason: str


@dataclass(frozen=True, slots=True)
class DagEdge:
    """One deduplicated directed edge and all its unique sorted reasons."""

    before: int
    after: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReasonedDAG:
    """An immutable DAG with deterministic order and descendant bitsets."""

    node_count: int
    edges: tuple[DagEdge, ...]
    successors: tuple[tuple[int, ...], ...]
    topological_order: tuple[int, ...]
    reachable: tuple[int, ...]

    @classmethod
    def build(
        cls,
        node_count: int,
        constraints: Iterable[EdgeConstraint],
    ) -> ReasonedDAG:
        """Validate, deduplicate, and close one bounded partial-order graph."""

        node_count = _plain_int(node_count, "graph node count")
        if node_count < 0 or node_count > MAX_DAG_NODES:
            _fail(
                f"graph node count exceeds the {MAX_DAG_NODES}-node limit"
            )

        reason_sets: dict[tuple[int, int], set[str]] = {}
        for input_index, constraint in enumerate(constraints):
            if input_index >= MAX_EDGE_INPUTS:
                _fail("graph exceeds the 262144-constraint input limit")
            if not isinstance(constraint, EdgeConstraint):
                _fail("graph constraints must be EdgeConstraint values")
            before = _bounded_node(
                constraint.before, node_count, "edge source node"
            )
            after = _bounded_node(
                constraint.after, node_count, "edge destination node"
            )
            reason = _reason(constraint.reason)
            key = (before, after)
            reasons = reason_sets.get(key)
            if reasons is None:
                if len(reason_sets) >= MAX_DAG_EDGES:
                    _fail("graph exceeds the 65536-edge limit")
                reasons = set()
                reason_sets[key] = reasons
            reasons.add(reason)
            if len(reasons) > MAX_REASONS_PER_EDGE:
                _fail("one graph edge exceeds the eight-reason limit")

        edges = tuple(
            DagEdge(before, after, tuple(sorted(reasons)))
            for (before, after), reasons in sorted(reason_sets.items())
        )
        adjacency: list[list[int]] = [[] for _ in range(node_count)]
        indegree = [0] * node_count
        edge_reasons: dict[tuple[int, int], tuple[str, ...]] = {}
        for edge in edges:
            adjacency[edge.before].append(edge.after)
            indegree[edge.after] += 1
            edge_reasons[(edge.before, edge.after)] = edge.reasons
        for targets in adjacency:
            targets.sort()

        ready = [node for node, count in enumerate(indegree) if count == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            node = heapq.heappop(ready)
            order.append(node)
            for successor in adjacency[node]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, successor)

        if len(order) != node_count:
            cycle = _find_cycle(adjacency, indegree)
            _fail(_format_cycle(cycle, edge_reasons))

        reachable = [0] * node_count
        for node in reversed(order):
            descendants = 0
            for successor in adjacency[node]:
                descendants |= (1 << successor) | reachable[successor]
            reachable[node] = descendants

        return cls(
            node_count,
            edges,
            tuple(tuple(targets) for targets in adjacency),
            tuple(order),
            tuple(reachable),
        )

    def precedes(self, before: int, after: int) -> bool:
        """Return whether the graph forces ``before`` before ``after``."""

        before = _bounded_node(before, self.node_count, "source node")
        after = _bounded_node(after, self.node_count, "destination node")
        return bool(self.reachable[before] & (1 << after))

    def reasons(self, before: int, after: int) -> tuple[str, ...]:
        """Return the labels on a direct edge, or an empty tuple."""

        before = _bounded_node(before, self.node_count, "source node")
        after = _bounded_node(after, self.node_count, "destination node")
        for edge in self.edges:
            if edge.before == before and edge.after == after:
                return edge.reasons
            if (edge.before, edge.after) > (before, after):
                break
        return ()


def _find_cycle(
    adjacency: Sequence[Sequence[int]],
    residual_indegree: Sequence[int],
) -> tuple[int, ...]:
    """Return one deterministic closed cycle from Kahn's residual graph."""

    node_count = len(adjacency)
    color = bytearray(node_count)
    parent = [-1] * node_count
    residual = tuple(count > 0 for count in residual_indegree)

    for start in range(node_count):
        if not residual[start] or color[start] != 0:
            continue
        color[start] = 1
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            node, successor_index = stack[-1]
            targets = adjacency[node]
            while (
                successor_index < len(targets)
                and not residual[targets[successor_index]]
            ):
                successor_index += 1
            if successor_index == len(targets):
                color[node] = 2
                stack.pop()
                continue

            successor = targets[successor_index]
            stack[-1] = (node, successor_index + 1)
            if color[successor] == 0:
                parent[successor] = node
                color[successor] = 1
                stack.append((successor, 0))
                continue
            if color[successor] == 1:
                path = [node]
                while path[-1] != successor:
                    predecessor = parent[path[-1]]
                    if predecessor < 0:
                        _fail(
                            "graph is cyclic but its cycle could not be reconstructed"
                        )
                    path.append(predecessor)
                path.reverse()
                path.append(successor)
                return tuple(path)

    _fail("graph is cyclic but its cycle could not be located")


def _format_cycle(
    cycle: Sequence[int],
    edge_reasons: dict[tuple[int, int], tuple[str, ...]],
) -> str:
    edge_count = len(cycle) - 1
    shown = min(edge_count, MAX_CYCLE_DIAGNOSTIC_EDGES)
    pieces = [f"cycle detected ({edge_count} edges): {cycle[0]}"]
    for index in range(shown):
        before = cycle[index]
        after = cycle[index + 1]
        reasons = edge_reasons[(before, after)]
        pieces.append(f" -[{reasons[0]}]-> {after}")
    if shown < edge_count:
        pieces.append(" -> ...")
    return "".join(pieces)


@dataclass(frozen=True, slots=True)
class StaticAction:
    """The scheduler-generated fields that are invariant across race runs."""

    ordinal: int
    producer: int
    kind: str
    submit_attempt: int | None
    client_index: int | None


@dataclass(frozen=True, slots=True)
class ActionProgram:
    """An immutable regenerated program plus independently checked count gates."""

    actions: tuple[StaticAction, ...]
    kind_counts: tuple[int, int, int, int, int]
    producer_counts: tuple[int, int]
    fixture_id: str
    fixture_file_sha256: str


@dataclass(frozen=True, slots=True)
class ActionEndpoint:
    """One explicitly typed action boundary and its graph-node identity."""

    counter: int
    node: int
    action_ordinal: int
    producer: int
    kind: str


@dataclass(frozen=True, slots=True)
class ActionEndpoints:
    """The immutable Invoke/Respond node pair for one action."""

    ordinal: int
    producer: int
    invocation: ActionEndpoint
    response: ActionEndpoint


@dataclass(frozen=True, slots=True)
class ActionEndpointMapping:
    """Exact endpoint lookup by action, counter, and graph node."""

    by_action: tuple[ActionEndpoints, ...]
    by_counter: tuple[ActionEndpoint, ...]
    by_node: tuple[ActionEndpoint, ...]


@dataclass(frozen=True, slots=True)
class ActionIntervalOrder:
    """Validated endpoint custody and its reason-labelled partial order."""

    endpoints: ActionEndpointMapping
    graph: ReasonedDAG


def _normalize_generated_action(raw: Any, expected_ordinal: int) -> StaticAction:
    if not isinstance(raw, dict):
        _fail(f"regenerated action {expected_ordinal} is not an object")
    kind = raw.get("kind")
    capture_kind = _SCHEDULER_TO_CAPTURE_KIND.get(kind)
    if capture_kind is None:
        _fail(f"regenerated action {expected_ordinal} has an unknown kind")

    ordinal = _plain_int(
        raw.get("ordinal"),
        f"regenerated action {expected_ordinal} ordinal",
    )
    if ordinal != expected_ordinal:
        _fail(f"regenerated action {expected_ordinal} is out of ordinal order")
    producer = _plain_int(raw.get("producer"), f"regenerated action {ordinal} producer")
    if producer not in (0, 1):
        _fail(f"regenerated action {ordinal} has an invalid producer")

    if kind == "submit":
        expected_keys = {
            "exhausted",
            "kind",
            "ordinal",
            "producer",
            "request_index",
            "submit_attempt",
        }
        if set(raw) != expected_keys:
            _fail(f"regenerated submit action {ordinal} has an open schema")
        submit_attempt = _plain_int(
            raw["submit_attempt"], f"regenerated action {ordinal} submit attempt"
        )
        if submit_attempt < 0:
            _fail(f"regenerated action {ordinal} has a negative submit attempt")
        exhausted = raw["exhausted"]
        if not isinstance(exhausted, bool):
            _fail(f"regenerated action {ordinal} exhausted flag is not boolean")
        client_index = raw["request_index"]
        expected_client = (
            None if submit_attempt >= scheduler.REQUEST_COUNT else submit_attempt
        )
        if client_index is not None:
            client_index = _plain_int(
                client_index,
                f"regenerated action {ordinal} client index",
            )
        if exhausted != (expected_client is None) or client_index != expected_client:
            _fail(f"regenerated action {ordinal} has an invalid submit mapping")
        if producer != submit_attempt % 2:
            _fail(f"regenerated action {ordinal} has an invalid submit producer")
    else:
        submit_attempt = None
        selector_key = "selector_index" if kind == "wake" else "request_index"
        expected_keys = {"kind", "ordinal", "producer", selector_key}
        if set(raw) != expected_keys:
            _fail(f"regenerated targeted action {ordinal} has an open schema")
        client_index = _plain_int(
            raw[selector_key], f"regenerated action {ordinal} client index"
        )
        if client_index < 0 or client_index >= scheduler.REQUEST_COUNT:
            _fail(f"regenerated action {ordinal} client index is out of range")
        if kind == "wake":
            expected_producer = ordinal % 2
        else:
            home = client_index % 2
            expected_producer = 1 - home if kind == "cancel" else home
        if producer != expected_producer:
            _fail(f"regenerated action {ordinal} has an invalid producer mapping")

    return StaticAction(
        ordinal,
        producer,
        capture_kind,
        submit_attempt,
        client_index,
    )


def _count_program(
    actions: Sequence[StaticAction],
) -> tuple[tuple[int, int, int, int, int], tuple[int, int]]:
    kinds = [0] * len(CAPTURE_ACTION_KINDS)
    producers = [0, 0]
    kind_indexes = {kind: index for index, kind in enumerate(CAPTURE_ACTION_KINDS)}
    for action in actions:
        if not isinstance(action, StaticAction):
            _fail("action program must contain StaticAction values")
        ordinal = _plain_int(action.ordinal, "static action ordinal")
        if ordinal < 0:
            _fail("static action ordinal must be nonnegative")
        kind_index = kind_indexes.get(action.kind)
        if kind_index is None:
            _fail(f"action {ordinal} has an unknown static kind")
        producer = _plain_int(action.producer, f"action {ordinal} producer")
        if producer not in (0, 1):
            _fail(f"action {ordinal} has an invalid static producer")
        kinds[kind_index] += 1
        producers[producer] += 1
    return (
        (kinds[0], kinds[1], kinds[2], kinds[3], kinds[4]),
        (producers[0], producers[1]),
    )


def regenerate_authenticated_action_program() -> ActionProgram:
    """Authenticate the committed fixture and freshly regenerate 1,024 actions."""

    try:
        fixture_file_sha256 = scheduler.check_path(scheduler.DEFAULT_FIXTURE_PATH)
        fixture = scheduler.validate_document(scheduler.build_fixture())
        generated = scheduler.build_actions(scheduler.ACTION_COUNT)
    except (OSError, scheduler.SchedulerFixtureError) as error:
        _fail(f"scheduler action authentication failed: {error}")

    if scheduler.ACTION_KINDS != ("submit", "cancel", "drop", "drain", "wake"):
        _fail("scheduler action kind codebook differs from the frozen codebook")
    if (
        len(generated) != scheduler.ACTION_COUNT
        or scheduler.ACTION_COUNT != EXPECTED_ACTION_COUNT
    ):
        _fail("regenerated scheduler action count is not 1024")
    generated_identity = scheduler.sequence_identity(generated)
    if generated_identity != fixture["action_vectors"]["digest"]:
        _fail("regenerated scheduler action identity differs from its fixture")
    if generated_identity != EXPECTED_ACTION_VECTOR_ID:
        _fail("regenerated scheduler action identity differs from the frozen corpus")
    if fixture["fixture_id"] != EXPECTED_FIXTURE_ID:
        _fail("scheduler fixture identity differs from the frozen corpus")
    if f"sha256:{fixture_file_sha256}" != EXPECTED_FIXTURE_FILE_SHA256:
        _fail("scheduler fixture file identity differs from the frozen corpus")

    actions = tuple(
        _normalize_generated_action(raw, ordinal)
        for ordinal, raw in enumerate(generated)
    )
    kind_counts, producer_counts = _count_program(actions)
    expected_kind_counts = tuple(
        fixture["action_counts"]["by_kind"][kind]
        for kind in scheduler.ACTION_KINDS
    )
    expected_producer_counts = (
        fixture["producer_counts"]["producer_0"],
        fixture["producer_counts"]["producer_1"],
    )
    if kind_counts != expected_kind_counts:
        _fail("regenerated scheduler action kind counts differ from the fixture")
    if producer_counts != expected_producer_counts:
        _fail("regenerated scheduler producer counts differ from the fixture")
    if kind_counts != EXPECTED_KIND_COUNTS:
        _fail("regenerated scheduler action kind counts differ from frozen gates")
    if producer_counts != EXPECTED_PRODUCER_COUNTS:
        _fail("regenerated scheduler producer counts differ from frozen gates")
    if fixture["action_counts"]["total"] != len(actions):
        _fail("fixture action total differs from regenerated actions")

    return ActionProgram(
        actions,
        kind_counts,
        producer_counts,
        fixture["fixture_id"],
        f"sha256:{fixture_file_sha256}",
    )


def _observed_static_action(action: Any, ordinal: int) -> StaticAction:
    fields: list[Any] = []
    for field in ("ordinal", "producer", "kind", "submit_attempt", "client_index"):
        try:
            fields.append(getattr(action, field))
        except AttributeError:
            _fail(f"observed action {ordinal} lacks static field {field}")
    return StaticAction(fields[0], fields[1], fields[2], fields[3], fields[4])


def _verify_static_actions(
    observed_actions: Sequence[Any],
    program: ActionProgram,
) -> None:
    """Compare one decoded action sequence with an immutable expected program."""

    if not isinstance(program, ActionProgram):
        _fail("expected action program has an invalid type")
    expected_kind_counts, expected_producer_counts = _count_program(program.actions)
    if program.kind_counts != expected_kind_counts:
        _fail("expected action program has inconsistent kind-count gates")
    if program.producer_counts != expected_producer_counts:
        _fail("expected action program has inconsistent producer-count gates")
    if len(observed_actions) != len(program.actions):
        _fail(
            "observed action count differs: "
            f"expected {len(program.actions)}, observed {len(observed_actions)}"
        )

    observed: list[StaticAction] = []
    for ordinal, (actual_value, expected) in enumerate(
        zip(observed_actions, program.actions, strict=True)
    ):
        actual = _observed_static_action(actual_value, ordinal)
        observed.append(actual)
        for field in ("ordinal", "producer", "kind", "submit_attempt", "client_index"):
            expected_value = getattr(expected, field)
            actual_field = getattr(actual, field)
            if (
                actual_field != expected_value
                or type(actual_field) is not type(expected_value)
            ):
                _fail(
                    f"action {ordinal} {field} mismatch: "
                    f"expected {expected_value!r}, observed {actual_field!r}"
                )

    observed_kind_counts, observed_producer_counts = _count_program(observed)
    if observed_kind_counts != program.kind_counts:
        _fail("observed action kind counts differ from the authenticated gates")
    if observed_producer_counts != program.producer_counts:
        _fail("observed producer counts differ from the authenticated gates")


def _verify_repetition_static_actions(
    repetition: Any,
    program: ActionProgram,
) -> None:
    try:
        actions = repetition.actions
    except AttributeError:
        _fail("decoded repetition lacks its action sequence")
    _verify_static_actions(actions, program)


def verify_repetition_static_actions(repetition: Any) -> ActionProgram:
    """Authenticate locally and verify one repetition's static actions."""

    program = regenerate_authenticated_action_program()
    _verify_repetition_static_actions(repetition, program)
    return program


def verify_capture_static_actions(capture: Any) -> ActionProgram:
    """Authenticate once locally and verify every decoded repetition."""

    try:
        repetition_count = _plain_int(
            capture.repetition_count,
            "decoded capture repetition count",
        )
        repetitions = capture.repetitions
    except AttributeError:
        _fail("decoded capture lacks its repetition custody fields")
    if repetition_count != EXPECTED_REPETITION_COUNT:
        _fail("decoded capture repetition count is not exactly 32")
    if not isinstance(repetitions, tuple):
        _fail("decoded capture repetitions must be an immutable tuple")
    if len(repetitions) != EXPECTED_REPETITION_COUNT:
        _fail(
            "decoded capture repetition sequence does not contain exactly 32 entries"
        )
    program = regenerate_authenticated_action_program()
    for repetition in repetitions:
        _verify_repetition_static_actions(repetition, program)
    return program


def _build_action_interval_order(
    actions: Sequence[Any],
    action_counter_final: Any,
    *,
    expected_action_count: int,
) -> ActionIntervalOrder:
    """Build the exact endpoint graph for one bounded action sequence.

    The configurable count exists so tests can differentially exercise this
    production construction on small histories.  The public entry point below
    always supplies the frozen 1,024-action corpus gate.
    """

    expected_action_count = _plain_int(
        expected_action_count,
        "expected action count",
    )
    if (
        expected_action_count < 1
        or expected_action_count * 2 > MAX_DAG_NODES
    ):
        _fail("expected action count exceeds the bounded endpoint graph")
    if not isinstance(actions, tuple):
        _fail("decoded actions must be an immutable tuple")
    if len(actions) != expected_action_count:
        _fail(
            "observed action count differs: "
            f"expected {expected_action_count}, observed {len(actions)}"
        )

    expected_counter_final = expected_action_count * 2
    action_counter_final = _plain_int(
        action_counter_final,
        "action_counter_final",
    )
    if action_counter_final != expected_counter_final:
        _fail(f"action_counter_final must be exactly {expected_counter_final}")

    endpoint_slots: list[ActionEndpoint | None] = [
        None
    ] * action_counter_final
    by_action: list[ActionEndpoints] = []
    by_node: list[ActionEndpoint] = []
    constraints: list[EdgeConstraint] = []
    last_response_by_producer: list[ActionEndpoint | None] = [None, None]

    for position, action in enumerate(actions):
        try:
            ordinal = _plain_int(action.ordinal, f"action {position} ordinal")
            producer = _plain_int(
                action.producer,
                f"action {position} producer",
            )
            invocation_counter = _plain_int(
                action.invocation,
                f"action {position} invocation",
            )
            response_counter = _plain_int(
                action.response,
                f"action {position} response",
            )
        except AttributeError:
            _fail(f"action {position} lacks interval fields")

        if ordinal != position:
            _fail(f"action {position} is out of ordinal order")
        if producer not in (0, 1):
            _fail(f"action {position} has an invalid producer")
        if not 1 <= invocation_counter <= action_counter_final:
            _fail(f"action {position} invocation is outside the exact counter range")
        if not 1 <= response_counter <= action_counter_final:
            _fail(f"action {position} response is outside the exact counter range")
        if invocation_counter >= response_counter:
            _fail(f"action {position} response must follow its invocation")

        invocation = ActionEndpoint(
            invocation_counter,
            position * 2,
            ordinal,
            producer,
            "invoke",
        )
        response = ActionEndpoint(
            response_counter,
            position * 2 + 1,
            ordinal,
            producer,
            "respond",
        )
        for endpoint in (invocation, response):
            slot = endpoint.counter - 1
            previous_endpoint = endpoint_slots[slot]
            if previous_endpoint is not None:
                _fail(
                    f"action endpoint counter {endpoint.counter} is duplicated"
                )
            endpoint_slots[slot] = endpoint

        previous_response = last_response_by_producer[producer]
        if (
            previous_response is not None
            and previous_response.counter >= invocation.counter
        ):
            _fail(
                f"producer {producer} action intervals violate program order"
            )

        interval = ActionEndpoints(ordinal, producer, invocation, response)
        by_action.append(interval)
        by_node.extend((invocation, response))
        constraints.append(
            EdgeConstraint(
                invocation.node,
                response.node,
                "action-invocation-before-response",
            )
        )
        if previous_response is not None:
            constraints.append(
                EdgeConstraint(
                    previous_response.node,
                    invocation.node,
                    f"producer-{producer}-program-order",
                )
            )
        last_response_by_producer[producer] = response

    missing_counter = next(
        (
            counter
            for counter, endpoint in enumerate(endpoint_slots, start=1)
            if endpoint is None
        ),
        None,
    )
    if missing_counter is not None:
        _fail(f"action endpoint counter {missing_counter} is missing")
    by_counter = tuple(
        endpoint for endpoint in endpoint_slots if endpoint is not None
    )

    # Scan the exact counter sequence once.  For each invocation, only the
    # latest earlier response from the other producer is needed: producer
    # program order makes every earlier response from that producer reach it.
    # This is O(actions * producer_count), not O(actions^2).
    latest_response_by_producer: list[ActionEndpoint | None] = [None, None]
    for endpoint in by_counter:
        if endpoint.kind == "respond":
            latest_response_by_producer[endpoint.producer] = endpoint
            continue
        for producer, latest_response in enumerate(
            latest_response_by_producer
        ):
            if producer == endpoint.producer or latest_response is None:
                continue
            constraints.append(
                EdgeConstraint(
                    latest_response.node,
                    endpoint.node,
                    f"producer-{producer}-latest-response-before-invocation",
                )
            )

    mapping = ActionEndpointMapping(
        tuple(by_action),
        by_counter,
        tuple(by_node),
    )
    graph = ReasonedDAG.build(action_counter_final, constraints)
    return ActionIntervalOrder(mapping, graph)


def build_action_interval_order(repetition: Any) -> ActionIntervalOrder:
    """Authenticate, validate, and order one exact action counter history."""

    program = regenerate_authenticated_action_program()
    _verify_repetition_static_actions(repetition, program)
    try:
        actions = repetition.actions
        diagnostics = repetition.diagnostics
    except AttributeError:
        _fail("decoded repetition lacks action interval custody fields")
    try:
        action_counter_final = diagnostics.action_counter_final
    except AttributeError:
        _fail("decoded repetition lacks action_counter_final")
    return _build_action_interval_order(
        actions,
        action_counter_final,
        expected_action_count=EXPECTED_ACTION_COUNT,
    )


def producer_action_interval_edges(
    actions: Sequence[Any],
) -> tuple[EdgeConstraint, ...]:
    """Return consecutive per-producer whole-action interval constraints.

    Each edge means that the previous action's witnessed response precedes the
    next action's witnessed invocation.  These nodes therefore represent whole
    action intervals only.  Command, control, endpoint, and wake boundaries
    require distinct event nodes and must never be collapsed onto this graph.
    Ordinal adjacency across different producers is intentionally absent.
    """

    if len(actions) > MAX_DAG_NODES:
        _fail("action program exceeds the graph node limit")
    last: list[tuple[int, int] | None] = [None, None]
    constraints: list[EdgeConstraint] = []
    for position, action in enumerate(actions):
        try:
            ordinal = _plain_int(action.ordinal, f"action {position} ordinal")
            producer = _plain_int(action.producer, f"action {position} producer")
            invocation = _plain_int(
                action.invocation, f"action {position} invocation"
            )
            response = _plain_int(action.response, f"action {position} response")
        except AttributeError:
            _fail(f"action {position} lacks interval fields")
        if ordinal != position:
            _fail(f"static action {position} is out of ordinal order")
        if producer not in (0, 1):
            _fail(f"static action {position} has an invalid producer")
        if invocation < 1 or response <= invocation:
            _fail(f"action {position} has an invalid interval")
        previous = last[producer]
        if previous is not None:
            previous_position, previous_response = previous
            if previous_response >= invocation:
                _fail(
                    f"producer {producer} action intervals violate program order"
                )
            constraints.append(
                EdgeConstraint(
                    previous_position,
                    position,
                    f"producer-{producer}-response-before-invocation",
                )
            )
        last[producer] = (position, response)
    return tuple(constraints)
