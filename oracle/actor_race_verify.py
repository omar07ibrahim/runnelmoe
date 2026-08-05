#!/usr/bin/env python3
"""Independent bounded semantic primitives for actor race-history captures.

This layer owns five deliberately narrow responsibilities:

* a deterministic, reason-labelled partial-order DAG with bitset reachability;
* authentication and exact comparison of the frozen scheduler action program;
* exact action invocation/response custody and its witnessed partial order;
* submission-command custody and an explicit event graph for admission results;
* immutable accepted-identity projection and target lookup/access custody.

It does not yet interpret packed control-word transitions, output drain-count
chains, terminal witnesses, or observation conservation.  Action interval
counters are mapped to explicit Invoke and Respond nodes; they are never
promoted into a counter-derived global execution order.  In particular,
consuming an actor command response is a separate CommandRelease event, never
an alias for either ActorCommandRespond publication or the action's final
Respond counter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, is_dataclass
import heapq
from typing import Any, NoReturn

from oracle import actor_race_history, scheduler


MAX_DAG_NODES = 8_192
MAX_DAG_EDGES = 65_536
MAX_EDGE_INPUTS = 262_144
MAX_REASONS_PER_EDGE = 8
MAX_REASON_BYTES = 128
MAX_CYCLE_DIAGNOSTIC_EDGES = 8

# The protocol edge planner is intentionally tighter than the reusable DAG.
MAX_PROTOCOL_EDGE_INPUTS = 32_768

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
EXPECTED_SUBMIT_COUNT = 206
EXPECTED_IN_RANGE_SUBMIT_COUNT = 64
EXPECTED_EXHAUSTED_SUBMIT_COUNT = 142
EXPECTED_TARGET_ACTION_COUNT = sum(EXPECTED_KIND_COUNTS[1:4])

# Exact schema maximum for the finished event model.  Several classes are not
# allocated by the submission-only layer yet, but reserving and checking their
# budget now prevents a later integration from silently crossing the generic
# DAG cap.  The component counts follow ADR-0007's frozen cardinalities.
_PROTOCOL_NODE_BUDGET = (
    ("script action endpoints", 2 * EXPECTED_ACTION_COUNT),
    (
        "main non-exhausted operations",
        EXPECTED_ACTION_COUNT - EXPECTED_EXHAUSTED_SUBMIT_COUNT,
    ),
    (
        "submit ready, actor claim, actor response, and release events",
        4 * EXPECTED_IN_RANGE_SUBMIT_COUNT,
    ),
    ("accepted bind and publish events", 3 * EXPECTED_IN_RANGE_SUBMIT_COUNT),
    ("opportunistic endpoint pops", EXPECTED_KIND_COUNTS[3]),
    (
        "per-request terminal EOF and reap events",
        4 * EXPECTED_IN_RANGE_SUBMIT_COUNT,
    ),
    (
        "cleanup invocation and response events",
        4 * EXPECTED_IN_RANGE_SUBMIT_COUNT,
    ),
    ("cleanup authority control events", EXPECTED_IN_RANGE_SUBMIT_COUNT),
    (
        "cleanup receiver terminal acknowledgements",
        EXPECTED_IN_RANGE_SUBMIT_COUNT,
    ),
    ("lifecycle phase gates", 3),
)
MAX_PROTOCOL_NODES = 4_258
if sum(count for _, count in _PROTOCOL_NODE_BUDGET) != MAX_PROTOCOL_NODES:
    raise RuntimeError("actor race protocol node budget does not total 4258")
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
    if type(value) is not int:
        _fail(f"{label} must be an integer")
    return value


def _plain_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be a boolean")
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


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    """One explicitly allocated submission-protocol event node."""

    node: int
    action_ordinal: int
    kind: str


@dataclass(frozen=True, slots=True)
class SubmitEventNodes:
    """All protocol nodes allocated for one authenticated in-range submit."""

    action_ordinal: int
    submit_attempt: int
    command_slot: int
    command_ticket: int
    ready_sequence: int
    accepted: bool
    command_reserve: ProtocolEvent
    ready_commit: ProtocolEvent
    actor_command_claim: ProtocolEvent
    actor_command_respond: ProtocolEvent
    command_release: ProtocolEvent
    control_bind: ProtocolEvent | None
    endpoint_bind: ProtocolEvent | None
    registry_publish: ProtocolEvent | None


@dataclass(frozen=True, slots=True)
class SubmissionProtocolMapping:
    """Exact submit-event lookup by action, ready sequence, and added node."""

    by_action: tuple[SubmitEventNodes | None, ...]
    by_ready_sequence: tuple[SubmitEventNodes, ...]
    protocol_events: tuple[ProtocolEvent, ...]


@dataclass(frozen=True, slots=True)
class SubmissionProtocolOrder:
    """Authenticated action intervals plus the submission protocol event DAG."""

    interval_order: ActionIntervalOrder
    submissions: SubmissionProtocolMapping
    graph: ReasonedDAG
    accepted_count: int
    rejected_count: int


@dataclass(frozen=True, slots=True)
class AcceptedIdentityProjection:
    """One accepted client and its complete immutable published identity."""

    client_index: int
    request_id: int
    control_slot: int
    control_generation: int
    endpoint_slot: int
    endpoint_generation: int
    submission: SubmitEventNodes


@dataclass(frozen=True, slots=True)
class AcceptedIdentityMapping:
    """Exact accepted-identity lookup without exposing mutable dictionaries."""

    by_client_index: tuple[AcceptedIdentityProjection | None, ...]
    by_request_id: tuple[AcceptedIdentityProjection, ...]
    by_control_slot: tuple[tuple[AcceptedIdentityProjection, ...], ...]
    by_endpoint_slot: tuple[tuple[AcceptedIdentityProjection, ...], ...]


@dataclass(frozen=True, slots=True)
class TargetActionEvent:
    """The one concrete main event reconstructed for a targeted action."""

    action_ordinal: int
    client_index: int
    action_kind: str
    resolution: str
    identity: AcceptedIdentityProjection | None
    prior_receiver_drop_action: int | None
    event: ProtocolEvent


@dataclass(frozen=True, slots=True)
class TargetAccessMapping:
    """Immutable target-event lookup by action and accepted client."""

    identities: AcceptedIdentityMapping
    by_action: tuple[TargetActionEvent | None, ...]
    by_client_index: tuple[tuple[TargetActionEvent, ...], ...]
    target_events: tuple[ProtocolEvent, ...]


@dataclass(frozen=True, slots=True)
class TargetAccessOrder:
    """Submission custody extended through target lookup or object access."""

    submission_order: SubmissionProtocolOrder
    targets: TargetAccessMapping
    graph: ReasonedDAG


class _ProtocolGraphPlanner:
    """Bound protocol node and edge allocation before retaining each input."""

    __slots__ = (
        "_base_node_count",
        "_constraints",
        "_events",
        "_next_node",
    )

    def __init__(self, base_node_count: int):
        base_node_count = _plain_int(
            base_node_count,
            "protocol base node count",
        )
        if base_node_count < 0 or base_node_count > MAX_PROTOCOL_NODES:
            _fail(
                "protocol base node count exceeds the "
                f"{MAX_PROTOCOL_NODES}-node limit"
            )
        self._base_node_count = base_node_count
        self._next_node = base_node_count
        self._constraints: list[EdgeConstraint] = []
        self._events: list[ProtocolEvent] = []

    @property
    def node_count(self) -> int:
        return self._next_node

    @property
    def edge_input_count(self) -> int:
        return len(self._constraints)

    @property
    def protocol_events(self) -> tuple[ProtocolEvent, ...]:
        return tuple(self._events)

    def allocate(self, action_ordinal: int, kind: str) -> ProtocolEvent:
        """Allocate one new node, rejecting the first over-limit request."""

        action_ordinal = _plain_int(
            action_ordinal,
            "protocol event action ordinal",
        )
        if action_ordinal < 0 or action_ordinal >= EXPECTED_ACTION_COUNT:
            _fail("protocol event action ordinal is outside the action corpus")
        if kind not in {
            "CommandReserve",
            "ReadyCommit",
            "ActorCommandClaim",
            "ActorCommandRespond",
            "CommandRelease",
            "ControlBind",
            "EndpointBind",
            "RegistryPublish",
            "TargetLookup",
            "ControlCancel",
            "ControlDisconnect",
            "PrimaryEndpointPop",
            "CachedEofRead",
        }:
            _fail("protocol event kind is unsupported")
        if self._next_node >= MAX_PROTOCOL_NODES:
            _fail(
                f"protocol graph exceeds the {MAX_PROTOCOL_NODES}-node limit"
            )
        event = ProtocolEvent(self._next_node, action_ordinal, kind)
        self._next_node += 1
        self._events.append(event)
        return event

    def edge(self, before: int, after: int, reason: str) -> None:
        """Retain one reasoned input, rejecting before the list can exceed its cap."""

        if len(self._constraints) >= MAX_PROTOCOL_EDGE_INPUTS:
            _fail(
                "protocol graph exceeds the "
                f"{MAX_PROTOCOL_EDGE_INPUTS}-constraint input limit"
            )
        # Reason and endpoint validation remain centralized in ReasonedDAG;
        # the planner's responsibility is to cap retained attacker influence.
        self._constraints.append(EdgeConstraint(before, after, reason))

    def layer(self, graph: ReasonedDAG) -> None:
        """Copy every base edge reason without collapsing its provenance."""

        if not isinstance(graph, ReasonedDAG):
            _fail("protocol base graph has an invalid type")
        if graph.node_count != self._base_node_count:
            _fail("protocol base graph node count changed")
        for edge in graph.edges:
            for reason in edge.reasons:
                self.edge(edge.before, edge.after, reason)

    def build(self) -> ReasonedDAG:
        """Build the immutable graph from the already bounded input set."""

        return ReasonedDAG.build(self._next_node, tuple(self._constraints))


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

    submit_attempts = tuple(
        action.submit_attempt
        for action in actions
        if action.kind == "submit"
    )
    in_range_attempts = tuple(
        attempt
        for attempt in submit_attempts
        if attempt is not None and attempt < scheduler.REQUEST_COUNT
    )
    exhausted_attempts = tuple(
        attempt
        for attempt in submit_attempts
        if attempt is not None and attempt >= scheduler.REQUEST_COUNT
    )
    if (
        scheduler.REQUEST_COUNT != EXPECTED_IN_RANGE_SUBMIT_COUNT
        or len(submit_attempts) != EXPECTED_SUBMIT_COUNT
        or len(in_range_attempts) != EXPECTED_IN_RANGE_SUBMIT_COUNT
        or len(exhausted_attempts) != EXPECTED_EXHAUSTED_SUBMIT_COUNT
    ):
        _fail("regenerated scheduler submit counts differ from frozen gates")
    if tuple(sorted(in_range_attempts)) != tuple(
        range(EXPECTED_IN_RANGE_SUBMIT_COUNT)
    ):
        _fail("regenerated in-range submit attempts lack exact coverage")
    if tuple(sorted(exhausted_attempts)) != tuple(
        range(EXPECTED_IN_RANGE_SUBMIT_COUNT, EXPECTED_SUBMIT_COUNT)
    ):
        _fail("regenerated exhausted submit attempts lack exact coverage")

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

    try:
        actions = repetition.actions
        diagnostics = repetition.diagnostics
    except AttributeError:
        _fail("decoded repetition lacks action interval custody fields")
    try:
        action_counter_final = diagnostics.action_counter_final
    except AttributeError:
        _fail("decoded repetition lacks action_counter_final")
    _require_exact_decoded_actions(actions)
    program = regenerate_authenticated_action_program()
    _verify_static_actions(actions, program)
    return _build_action_interval_order(
        actions,
        action_counter_final,
        expected_action_count=EXPECTED_ACTION_COUNT,
    )


_COMMAND_SENTINEL = actor_race_history.CommandWitness(False, 0, 0, 0)
_ACCEPTED_SENTINEL = actor_race_history.AcceptedWitness(
    False,
    0,
    0,
    0,
    0,
    0,
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
    "primary",
    False,
    0,
    0,
    0,
    0,
    None,
)
_OPPORTUNISTIC_POP_SENTINEL = actor_race_history.PopWitness(
    "opportunistic_eof",
    False,
    0,
    0,
    0,
    0,
    None,
)
_WAKE_SENTINEL = actor_race_history.WakeWitness(False, False, 0, 0)
_SATURATED_SUBMIT_ERROR = actor_race_history.CapturedError(
    "resource_exhausted",
    "resource_exhausted",
    "request slot count",
    16,
    16,
)
_STALE_TARGET_ERROR = actor_race_history.CapturedError(
    "request_not_found",
    "invalid_request",
    None,
    None,
    None,
)


@dataclass(frozen=True, slots=True)
class _ValidatedSubmit:
    action_ordinal: int
    submit_attempt: int
    command_slot: int
    command_ticket: int
    ready_sequence: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class _ValidatedTarget:
    action_ordinal: int
    client_index: int
    action_kind: str
    resolution: str
    event_kind: str
    identity: AcceptedIdentityProjection | None
    prior_receiver_drop_action: int | None


def _field(value: Any, name: str, label: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError:
        _fail(f"{label} lacks field {name}")


def _require_exact_decoded_actions(actions: Any) -> tuple[Any, ...]:
    """Seal full-history entry points against sequence/action TOCTOU values."""

    if type(actions) is not tuple:
        _fail("decoded actions must be an exact immutable tuple")
    if len(actions) != EXPECTED_ACTION_COUNT:
        _fail("decoded action corpus must contain exactly 1024 entries")
    for ordinal, action in enumerate(actions):
        if type(action) is not actor_race_history.Action:
            _fail(f"decoded action {ordinal} has an invalid exact type")
    return actions


def _exact_typed_value(value: Any, expected: Any, label: str) -> None:
    """Require one closed decoded dataclass spelling and all of its values."""

    if not _same_exact_value(value, expected):
        _fail(f"{label} does not use its exact typed sentinel")


def _same_exact_value(value: Any, expected: Any) -> bool:
    """Compare a bounded frozen value without Python bool/int coercion."""

    if type(value) is not type(expected):
        return False
    if is_dataclass(expected) and not isinstance(expected, type):
        return all(
            _same_exact_value(
                getattr(value, field.name),
                getattr(expected, field.name),
            )
            for field in fields(expected)
        )
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(
            _same_exact_value(actual, wanted)
            for actual, wanted in zip(value, expected, strict=True)
        )
    return bool(value == expected)


def _validate_submit_zero_effects(action: Any, ordinal: int) -> None:
    """Reject non-submission effects on every submit result spelling."""

    label = f"submit action {ordinal}"
    if _field(action, "output", label) is not None:
        _fail(f"{label} retained an output")
    _exact_typed_value(
        _field(action, "control", label),
        _CONTROL_SENTINEL,
        f"{label} control witness",
    )
    _exact_typed_value(
        _field(action, "primary_pop", label),
        _PRIMARY_POP_SENTINEL,
        f"{label} primary-pop witness",
    )
    _exact_typed_value(
        _field(action, "opportunistic_eof_pop", label),
        _OPPORTUNISTIC_POP_SENTINEL,
        f"{label} opportunistic-pop witness",
    )
    if _plain_bool(_field(action, "cached_eof", label), f"{label} cached_eof"):
        _fail(f"{label} retained cached EOF state")
    _exact_typed_value(
        _field(action, "wake", label),
        _WAKE_SENTINEL,
        f"{label} wake witness",
    )


def _validate_command_witness(
    action: Any,
    ordinal: int,
    *,
    applicable: bool,
) -> tuple[int, int, int] | None:
    label = f"action {ordinal} command witness"
    command = _field(action, "command", label)
    if type(command) is not actor_race_history.CommandWitness:
        _fail(f"{label} has an invalid decoded type")
    boundary = _plain_bool(command.boundary, f"{label} boundary")
    slot = _plain_int(command.slot, f"{label} slot")
    ticket = _plain_int(command.ticket, f"{label} ticket")
    ready_sequence = _plain_int(
        command.ready_sequence,
        f"{label} ready_sequence",
    )
    if not applicable:
        _exact_typed_value(command, _COMMAND_SENTINEL, label)
        return None
    if not boundary:
        _fail(f"{label} omitted its in-range submit boundary")
    if slot < 0 or slot >= 8:
        _fail(f"{label} slot is outside 0..7")
    if ticket < 1 or ticket > EXPECTED_IN_RANGE_SUBMIT_COUNT:
        _fail(f"{label} ticket is outside 1..64")
    if ready_sequence < 1 or ready_sequence > EXPECTED_IN_RANGE_SUBMIT_COUNT:
        _fail(f"{label} ready_sequence is outside 1..64")
    return slot, ticket, ready_sequence


def _validate_accepted_witness(
    action: Any,
    ordinal: int,
) -> actor_race_history.AcceptedWitness:
    label = f"submit action {ordinal} accepted witness"
    accepted = _field(action, "accepted", label)
    if type(accepted) is not actor_race_history.AcceptedWitness:
        _fail(f"{label} has an invalid decoded type")
    if not _plain_bool(accepted.boundary, f"{label} boundary"):
        _fail(f"{label} omitted its boundary")
    request_id = _plain_int(accepted.request_id, f"{label} request_id")
    control_slot = _plain_int(accepted.control_slot, f"{label} control_slot")
    control_generation = _plain_int(
        accepted.control_generation,
        f"{label} control_generation",
    )
    endpoint_slot = _plain_int(
        accepted.endpoint_slot,
        f"{label} endpoint_slot",
    )
    endpoint_generation = _plain_int(
        accepted.endpoint_generation,
        f"{label} endpoint_generation",
    )
    if request_id < 1 or request_id > EXPECTED_IN_RANGE_SUBMIT_COUNT:
        _fail(f"{label} request_id is outside 1..64")
    if control_slot < 0 or control_slot >= 16:
        _fail(f"{label} control_slot is outside 0..15")
    if endpoint_slot < 0 or endpoint_slot >= 16:
        _fail(f"{label} endpoint_slot is outside 0..15")
    if control_generation < 1 or control_generation > EXPECTED_IN_RANGE_SUBMIT_COUNT:
        _fail(f"{label} control_generation is outside 1..64")
    if endpoint_generation < 1 or endpoint_generation > EXPECTED_IN_RANGE_SUBMIT_COUNT:
        _fail(f"{label} endpoint_generation is outside 1..64")
    action_request_id = _plain_int(
        _field(action, "request_id", label),
        f"submit action {ordinal} request_id",
    )
    if action_request_id != request_id:
        _fail(f"{label} does not match the action request identity")
    return accepted


def _validate_shutdown(
    shutdown: Any,
    accepted_count: int,
    rejected_count: int,
) -> None:
    if type(shutdown) is not actor_race_history.Shutdown:
        _fail("shutdown accounting has an invalid decoded type")

    values = {
        name: _plain_int(getattr(shutdown, name), f"shutdown {name}")
        for name in (
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
    if values["accepted_submissions"] != accepted_count:
        _fail("shutdown accepted_submissions differs from accepted submit evidence")
    if values["rejected_submissions"] != rejected_count:
        _fail("shutdown rejected_submissions differs from rejected submit evidence")
    if values["engine_steps"] < 0:
        _fail("shutdown engine_steps must be nonnegative")
    for field_name in (
        "discarded_output_events",
        "released_request_bytes",
        "remaining_shared_bytes",
        "shutdown_cancellations",
        "terminated_requests",
    ):
        if values[field_name] != 0:
            _fail(f"shutdown {field_name} must be exactly zero")


def _validate_submission_custody(
    actions: tuple[Any, ...],
    shutdown: Any,
    program: ActionProgram,
) -> tuple[tuple[_ValidatedSubmit, ...], int, int]:
    """Validate exact command coverage and admission-result identities."""

    _require_exact_decoded_actions(actions)

    expected_in_range = {
        action.ordinal
        for action in program.actions
        if action.kind == "submit"
        and action.submit_attempt is not None
        and action.submit_attempt < EXPECTED_IN_RANGE_SUBMIT_COUNT
    }
    expected_exhausted = {
        action.ordinal
        for action in program.actions
        if action.kind == "submit"
        and action.submit_attempt is not None
        and action.submit_attempt >= EXPECTED_IN_RANGE_SUBMIT_COUNT
    }
    if len(expected_in_range) != EXPECTED_IN_RANGE_SUBMIT_COUNT:
        _fail("authenticated action program does not contain exactly 64 in-range submits")
    if len(expected_exhausted) != EXPECTED_EXHAUSTED_SUBMIT_COUNT:
        _fail("authenticated action program does not contain exactly 142 exhausted submits")

    records: list[_ValidatedSubmit] = []
    by_ready: list[_ValidatedSubmit | None] = [
        None
    ] * EXPECTED_IN_RANGE_SUBMIT_COUNT
    tickets_by_slot: list[list[_ValidatedSubmit]] = [[] for _ in range(8)]
    accepted_by_ordinal: dict[int, actor_race_history.AcceptedWitness] = {}

    for ordinal, (action, expected) in enumerate(
        zip(actions, program.actions, strict=True)
    ):
        in_range = ordinal in expected_in_range
        command_fields = _validate_command_witness(
            action,
            ordinal,
            applicable=in_range,
        )
        if expected.kind != "submit":
            continue

        _validate_submit_zero_effects(action, ordinal)
        result = _field(action, "result", f"submit action {ordinal}")
        if type(result) is not str:
            _fail(f"submit action {ordinal} result must be a string")
        error = _field(action, "error", f"submit action {ordinal}")
        request_id = _field(action, "request_id", f"submit action {ordinal}")
        accepted_witness = _field(
            action,
            "accepted",
            f"submit action {ordinal}",
        )

        if ordinal in expected_exhausted:
            if (
                result != "submit_offer_exhausted"
                or error is not None
                or request_id is not None
            ):
                _fail(f"submit action {ordinal} has an invalid exhausted spelling")
            _exact_typed_value(
                accepted_witness,
                _ACCEPTED_SENTINEL,
                f"submit action {ordinal} accepted witness",
            )
            continue

        if command_fields is None:
            _fail(f"submit action {ordinal} lacks command custody")
        slot, ticket, ready_sequence = command_fields
        if result == "submit_accepted":
            if error is not None or request_id is None:
                _fail(f"submit action {ordinal} has an invalid accepted spelling")
            accepted = True
            accepted_by_ordinal[ordinal] = _validate_accepted_witness(
                action,
                ordinal,
            )
        elif result == "error":
            # This gate proves only the exact typed rejection returned by the
            # actor.  It intentionally makes no claim about admission-time
            # live occupancy, for which the capture has no atomic witness.
            if request_id is not None:
                _fail(f"submit action {ordinal} rejected with a request identity")
            _exact_typed_value(
                error,
                _SATURATED_SUBMIT_ERROR,
                f"submit action {ordinal} rejection error",
            )
            _exact_typed_value(
                accepted_witness,
                _ACCEPTED_SENTINEL,
                f"submit action {ordinal} accepted witness",
            )
            accepted = False
        else:
            _fail(f"submit action {ordinal} has an invalid in-range result spelling")

        submit_attempt = _plain_int(
            expected.submit_attempt,
            f"authenticated submit action {ordinal} attempt",
        )
        record = _ValidatedSubmit(
            ordinal,
            submit_attempt,
            slot,
            ticket,
            ready_sequence,
            accepted,
        )
        ready_slot = ready_sequence - 1
        if by_ready[ready_slot] is not None:
            _fail(f"command ready_sequence {ready_sequence} is duplicated")
        by_ready[ready_slot] = record
        tickets_by_slot[slot].append(record)
        records.append(record)

    if len(records) != EXPECTED_IN_RANGE_SUBMIT_COUNT:
        _fail("observed command custody does not cover exactly 64 in-range submits")
    missing_ready = next(
        (
            index + 1
            for index, record in enumerate(by_ready)
            if record is None
        ),
        None,
    )
    if missing_ready is not None:
        _fail(f"command ready_sequence {missing_ready} is missing")

    for slot, slot_records in enumerate(tickets_by_slot):
        tickets = tuple(sorted(record.command_ticket for record in slot_records))
        if tickets != tuple(range(1, len(slot_records) + 1)):
            _fail(f"command slot {slot} tickets are not exactly 1..k")

    ready_records = tuple(record for record in by_ready if record is not None)
    next_request_id = 1
    next_control_generation = [1] * 16
    next_endpoint_generation = [1] * 16
    for record in ready_records:
        if not record.accepted:
            continue
        witness = accepted_by_ordinal[record.action_ordinal]
        if witness.request_id != next_request_id:
            _fail(
                "accepted request IDs are not exactly 1..N in command-ready order"
            )
        expected_control_generation = next_control_generation[witness.control_slot]
        if witness.control_generation != expected_control_generation:
            _fail(
                f"control slot {witness.control_slot} generations do not start at 1 "
                "and remain contiguous in command-ready order"
            )
        next_control_generation[witness.control_slot] += 1
        expected_endpoint_generation = next_endpoint_generation[witness.endpoint_slot]
        if witness.endpoint_generation != expected_endpoint_generation:
            _fail(
                f"endpoint slot {witness.endpoint_slot} generations do not start at 1 "
                "and remain contiguous in command-ready order"
            )
        next_endpoint_generation[witness.endpoint_slot] += 1
        next_request_id += 1

    accepted_count = next_request_id - 1
    rejected_count = EXPECTED_IN_RANGE_SUBMIT_COUNT - accepted_count
    _validate_shutdown(shutdown, accepted_count, rejected_count)
    return tuple(records), accepted_count, rejected_count


def _build_submission_protocol_order(
    actions: tuple[Any, ...],
    shutdown: Any,
    program: ActionProgram,
    interval_order: ActionIntervalOrder,
) -> SubmissionProtocolOrder:
    records, accepted_count, rejected_count = _validate_submission_custody(
        actions,
        shutdown,
        program,
    )
    planner = _ProtocolGraphPlanner(interval_order.graph.node_count)
    planner.layer(interval_order.graph)

    by_action: list[SubmitEventNodes | None] = [None] * EXPECTED_ACTION_COUNT
    for record in records:
        ordinal = record.action_ordinal
        reserve = planner.allocate(ordinal, "CommandReserve")
        ready = planner.allocate(ordinal, "ReadyCommit")
        actor_claim = planner.allocate(ordinal, "ActorCommandClaim")
        actor_respond = planner.allocate(ordinal, "ActorCommandRespond")
        release = planner.allocate(ordinal, "CommandRelease")
        control_bind: ProtocolEvent | None = None
        endpoint_bind: ProtocolEvent | None = None
        registry_publish: ProtocolEvent | None = None
        if record.accepted:
            control_bind = planner.allocate(ordinal, "ControlBind")
            endpoint_bind = planner.allocate(ordinal, "EndpointBind")
            registry_publish = planner.allocate(ordinal, "RegistryPublish")

        nodes = SubmitEventNodes(
            ordinal,
            record.submit_attempt,
            record.command_slot,
            record.command_ticket,
            record.ready_sequence,
            record.accepted,
            reserve,
            ready,
            actor_claim,
            actor_respond,
            release,
            control_bind,
            endpoint_bind,
            registry_publish,
        )
        by_action[ordinal] = nodes
        endpoints = interval_order.endpoints.by_action[ordinal]
        if endpoints.response.node == release.node:
            _fail("command release was aliased to the action response endpoint")
        planner.edge(
            endpoints.invocation.node,
            reserve.node,
            "submit-invocation-before-command-reserve",
        )
        planner.edge(
            reserve.node,
            ready.node,
            "command-reserve-before-ready-commit",
        )
        planner.edge(
            ready.node,
            actor_claim.node,
            "ready-commit-before-actor-command-claim",
        )
        if record.accepted:
            if (
                control_bind is None
                or endpoint_bind is None
                or registry_publish is None
            ):
                _fail("accepted submit event allocation is incomplete")
            planner.edge(
                actor_claim.node,
                control_bind.node,
                "actor-command-claim-before-control-bind",
            )
            planner.edge(
                actor_claim.node,
                endpoint_bind.node,
                "actor-command-claim-before-endpoint-bind",
            )
            planner.edge(
                control_bind.node,
                actor_respond.node,
                "control-bind-before-actor-command-respond",
            )
            planner.edge(
                endpoint_bind.node,
                actor_respond.node,
                "endpoint-bind-before-actor-command-respond",
            )
            planner.edge(
                actor_respond.node,
                release.node,
                "actor-command-respond-before-command-release",
            )
            planner.edge(
                release.node,
                registry_publish.node,
                "command-release-before-registry-publish",
            )
            planner.edge(
                registry_publish.node,
                endpoints.response.node,
                "registry-publish-before-submit-action-response",
            )
        else:
            planner.edge(
                actor_claim.node,
                actor_respond.node,
                "rejected-actor-command-claim-before-respond",
            )
            planner.edge(
                actor_respond.node,
                release.node,
                "actor-command-respond-before-command-release",
            )
            planner.edge(
                release.node,
                endpoints.response.node,
                "rejected-command-release-before-submit-action-response",
            )

    ready_nodes = tuple(
        sorted(
            (nodes for nodes in by_action if nodes is not None),
            key=lambda nodes: nodes.ready_sequence,
        )
    )
    if tuple(nodes.ready_sequence for nodes in ready_nodes) != tuple(
        range(1, EXPECTED_IN_RANGE_SUBMIT_COUNT + 1)
    ):
        _fail("internal ready-sequence mapping is incomplete")
    for previous, current in zip(ready_nodes, ready_nodes[1:]):
        planner.edge(
            previous.ready_commit.node,
            current.ready_commit.node,
            "command-ready-sequence-order",
        )
        planner.edge(
            previous.actor_command_respond.node,
            current.actor_command_claim.node,
            "actor-command-ready-sequence-order",
        )

    for slot in range(8):
        slot_nodes = tuple(
            sorted(
                (
                    nodes
                    for nodes in ready_nodes
                    if nodes.command_slot == slot
                ),
                key=lambda nodes: nodes.command_ticket,
            )
        )
        for previous, current in zip(slot_nodes, slot_nodes[1:]):
            planner.edge(
                previous.command_release.node,
                current.command_reserve.node,
                f"command-slot-{slot}-ticket-order",
            )

    graph = planner.build()
    for nodes in ready_nodes:
        if not nodes.accepted:
            continue
        control_bind = nodes.control_bind
        endpoint_bind = nodes.endpoint_bind
        if control_bind is None or endpoint_bind is None:
            _fail("accepted submit lacks its bind nodes")
        if graph.precedes(control_bind.node, endpoint_bind.node) or graph.precedes(
            endpoint_bind.node,
            control_bind.node,
        ):
            _fail("accepted control and endpoint binds became ordered")

    return SubmissionProtocolOrder(
        interval_order,
        SubmissionProtocolMapping(
            tuple(by_action),
            ready_nodes,
            planner.protocol_events,
        ),
        graph,
        accepted_count,
        rejected_count,
    )


def build_submission_protocol_order(repetition: Any) -> SubmissionProtocolOrder:
    """Authenticate and reconstruct one exact submission-command event DAG.

    Authentication is deliberately local: callers cannot inject an action
    program or bypass frozen fixture and count gates.
    """

    try:
        actions = repetition.actions
        diagnostics = repetition.diagnostics
        shutdown = repetition.shutdown
        action_counter_final = diagnostics.action_counter_final
    except AttributeError:
        _fail("decoded repetition lacks submission protocol custody fields")
    _require_exact_decoded_actions(actions)
    program = regenerate_authenticated_action_program()
    _verify_static_actions(actions, program)
    interval_order = _build_action_interval_order(
        actions,
        action_counter_final,
        expected_action_count=EXPECTED_ACTION_COUNT,
    )
    return _build_submission_protocol_order(
        actions,
        shutdown,
        program,
        interval_order,
    )


def _build_accepted_identity_mapping(
    actions: tuple[Any, ...],
    submission_order: SubmissionProtocolOrder,
) -> AcceptedIdentityMapping:
    """Project every accepted submit onto immutable client/object indexes."""

    _require_exact_decoded_actions(actions)

    by_client: list[AcceptedIdentityProjection | None] = [
        None
    ] * EXPECTED_IN_RANGE_SUBMIT_COUNT
    projections: list[AcceptedIdentityProjection] = []
    control_slots: list[list[AcceptedIdentityProjection]] = [
        [] for _ in range(16)
    ]
    endpoint_slots: list[list[AcceptedIdentityProjection]] = [
        [] for _ in range(16)
    ]

    for submission in submission_order.submissions.by_ready_sequence:
        if not submission.accepted:
            continue
        action = actions[submission.action_ordinal]
        label = f"accepted submit action {submission.action_ordinal}"
        client_index = _plain_int(
            _field(action, "client_index", label),
            f"{label} client_index",
        )
        if client_index != submission.submit_attempt:
            _fail(f"{label} client identity differs from its submit attempt")
        if client_index < 0 or client_index >= EXPECTED_IN_RANGE_SUBMIT_COUNT:
            _fail(f"{label} client_index is outside 0..63")
        witness = _field(action, "accepted", label)
        if type(witness) is not actor_race_history.AcceptedWitness:
            _fail(f"{label} accepted witness has an invalid decoded type")
        if (
            submission.control_bind is None
            or submission.endpoint_bind is None
            or submission.registry_publish is None
        ):
            _fail(f"{label} submission event projection is incomplete")
        projection = AcceptedIdentityProjection(
            client_index,
            _plain_int(witness.request_id, f"{label} request_id"),
            _plain_int(witness.control_slot, f"{label} control_slot"),
            _plain_int(
                witness.control_generation,
                f"{label} control_generation",
            ),
            _plain_int(witness.endpoint_slot, f"{label} endpoint_slot"),
            _plain_int(
                witness.endpoint_generation,
                f"{label} endpoint_generation",
            ),
            submission,
        )
        if by_client[client_index] is not None:
            _fail(f"accepted client {client_index} is duplicated")
        by_client[client_index] = projection
        projections.append(projection)
        control_slots[projection.control_slot].append(projection)
        endpoint_slots[projection.endpoint_slot].append(projection)

    by_request_id = tuple(sorted(projections, key=lambda item: item.request_id))
    if tuple(item.request_id for item in by_request_id) != tuple(
        range(1, submission_order.accepted_count + 1)
    ):
        _fail("accepted identity projection is not exactly indexed by request ID")
    if len(by_request_id) != submission_order.accepted_count:
        _fail("accepted identity projection count differs from submissions")

    by_control_slot = tuple(
        tuple(sorted(slot, key=lambda item: item.control_generation))
        for slot in control_slots
    )
    by_endpoint_slot = tuple(
        tuple(sorted(slot, key=lambda item: item.endpoint_generation))
        for slot in endpoint_slots
    )
    for slot_index, slot in enumerate(by_control_slot):
        if tuple(item.control_generation for item in slot) != tuple(
            range(1, len(slot) + 1)
        ):
            _fail(
                f"accepted control slot {slot_index} projection is not contiguous"
            )
    for slot_index, slot in enumerate(by_endpoint_slot):
        if tuple(item.endpoint_generation for item in slot) != tuple(
            range(1, len(slot) + 1)
        ):
            _fail(
                f"accepted endpoint slot {slot_index} projection is not contiguous"
            )

    return AcceptedIdentityMapping(
        tuple(by_client),
        by_request_id,
        by_control_slot,
        by_endpoint_slot,
    )


def _validate_hex64_word(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be exactly 16 lowercase hexadecimal digits")
    return value


def _validate_output_identity(
    output: Any,
    identity: AcceptedIdentityProjection,
    label: str,
) -> actor_race_history.Output:
    if type(output) is not actor_race_history.Output:
        _fail(f"{label} has an invalid decoded type")
    request_id = _plain_int(output.request_id, f"{label} request_id")
    output_index = _plain_int(output.output_index, f"{label} output_index")
    token_id = _plain_int(output.token_id, f"{label} token_id")
    if request_id != identity.request_id:
        _fail(f"{label} differs from its accepted request identity")
    if output_index < 0:
        _fail(f"{label} output_index must be nonnegative")
    if token_id < 0 or token_id > actor_race_history.UINT32_MAX:
        _fail(f"{label} token_id is outside the u32 range")
    return output


def _validate_control_target(
    action: Any,
    ordinal: int,
    identity: AcceptedIdentityProjection,
    operation: str,
) -> actor_race_history.ControlWitness:
    label = f"{operation} action {ordinal} control witness"
    control = _field(action, "control", label)
    if type(control) is not actor_race_history.ControlWitness:
        _fail(f"{label} has an invalid decoded type")
    if type(control.operation) is not str or control.operation != operation:
        _fail(f"{label} has the wrong operation")
    if not _plain_bool(control.boundary, f"{label} boundary"):
        _fail(f"{label} omitted its reached boundary")
    slot = _plain_int(control.slot, f"{label} slot")
    generation = _plain_int(
        control.expected_generation,
        f"{label} expected_generation",
    )
    if (slot, generation) != (
        identity.control_slot,
        identity.control_generation,
    ):
        _fail(f"{label} differs from its accepted control identity")
    _validate_hex64_word(control.loaded_word, f"{label} loaded_word")
    _validate_hex64_word(control.resulting_word, f"{label} resulting_word")
    if control.disposition is not None and type(control.disposition) is not str:
        _fail(f"{label} disposition must be a string or null")
    return control


def _validate_pop_target(
    pop: Any,
    identity: AcceptedIdentityProjection,
    expected_kind: str,
    label: str,
) -> actor_race_history.PopWitness:
    if type(pop) is not actor_race_history.PopWitness:
        _fail(f"{label} has an invalid decoded type")
    if type(pop.kind) is not str or pop.kind != expected_kind:
        _fail(f"{label} has the wrong endpoint-pop kind")
    if not _plain_bool(pop.boundary, f"{label} boundary"):
        _fail(f"{label} omitted its reached boundary")
    slot = _plain_int(pop.slot, f"{label} slot")
    generation = _plain_int(pop.generation, f"{label} generation")
    if (slot, generation) != (
        identity.endpoint_slot,
        identity.endpoint_generation,
    ):
        _fail(f"{label} differs from its accepted endpoint identity")
    drained_before = _plain_int(
        pop.drained_before,
        f"{label} drained_before",
    )
    drained_after = _plain_int(
        pop.drained_after,
        f"{label} drained_after",
    )
    if drained_before < 0 or drained_after < 0:
        _fail(f"{label} drain counts must be nonnegative")
    if pop.output is not None:
        _validate_output_identity(pop.output, identity, f"{label} output")
    return pop


def _validate_target_common_sentinels(action: Any, ordinal: int) -> None:
    label = f"target action {ordinal}"
    _exact_typed_value(
        _field(action, "command", label),
        _COMMAND_SENTINEL,
        f"{label} command witness",
    )
    _exact_typed_value(
        _field(action, "accepted", label),
        _ACCEPTED_SENTINEL,
        f"{label} accepted witness",
    )
    _exact_typed_value(
        _field(action, "wake", label),
        _WAKE_SENTINEL,
        f"{label} wake witness",
    )


def _require_target_request_id(
    action: Any,
    ordinal: int,
    identity: AcceptedIdentityProjection,
) -> int:
    request_id = _plain_int(
        _field(action, "request_id", f"target action {ordinal}"),
        f"target action {ordinal} request_id",
    )
    if request_id != identity.request_id:
        _fail(
            f"target action {ordinal} request ID differs from its accepted client"
        )
    return request_id


def _validate_cancel_target(
    action: Any,
    ordinal: int,
    client_index: int,
    identity: AcceptedIdentityProjection | None,
) -> _ValidatedTarget:
    label = f"cancel action {ordinal}"
    _validate_target_common_sentinels(action, ordinal)
    if _field(action, "output", label) is not None:
        _fail(f"{label} retained an output")
    _exact_typed_value(
        _field(action, "primary_pop", label),
        _PRIMARY_POP_SENTINEL,
        f"{label} primary-pop witness",
    )
    _exact_typed_value(
        _field(action, "opportunistic_eof_pop", label),
        _OPPORTUNISTIC_POP_SENTINEL,
        f"{label} opportunistic-pop witness",
    )
    if _plain_bool(_field(action, "cached_eof", label), f"{label} cached_eof"):
        _fail(f"{label} retained cached EOF state")

    result = _field(action, "result", label)
    if type(result) is not str:
        _fail(f"{label} result must be a string")
    error = _field(action, "error", label)
    if result == "target_unavailable":
        if error is not None or _field(action, "request_id", label) is not None:
            _fail(f"{label} unavailable spelling retained target state")
        _exact_typed_value(
            _field(action, "control", label),
            _CONTROL_SENTINEL,
            f"{label} control witness",
        )
        return _ValidatedTarget(
            ordinal,
            client_index,
            "cancel",
            "absent",
            "TargetLookup",
            identity,
            None,
        )

    if identity is None:
        _fail(f"{label} called an unaccepted client")
    _require_target_request_id(action, ordinal, identity)
    control = _validate_control_target(action, ordinal, identity, "cancel")
    dispositions = {
        "cancel_requested": "requested",
        "cancel_already_requested": "already_requested",
        "cancel_already_terminal": "already_terminal",
    }
    if result == "error":
        _exact_typed_value(error, _STALE_TARGET_ERROR, f"{label} stale error")
        if control.disposition is not None:
            _fail(f"{label} stale control retained a disposition")
    else:
        expected_disposition = dispositions.get(result)
        if expected_disposition is None:
            _fail(f"{label} has an invalid found-target result")
        if error is not None or control.disposition != expected_disposition:
            _fail(f"{label} result and control disposition disagree")
    return _ValidatedTarget(
        ordinal,
        client_index,
        "cancel",
        "cancel_authority",
        "ControlCancel",
        identity,
        None,
    )


def _validate_receiver_drop_target(
    action: Any,
    ordinal: int,
    client_index: int,
    state: str,
    identity: AcceptedIdentityProjection | None,
    prior_drop: int | None,
) -> _ValidatedTarget:
    label = f"receiver-drop action {ordinal}"
    _validate_target_common_sentinels(action, ordinal)
    if _field(action, "error", label) is not None:
        _fail(f"{label} retained an error")
    if _field(action, "output", label) is not None:
        _fail(f"{label} retained an output")
    _exact_typed_value(
        _field(action, "primary_pop", label),
        _PRIMARY_POP_SENTINEL,
        f"{label} primary-pop witness",
    )
    _exact_typed_value(
        _field(action, "opportunistic_eof_pop", label),
        _OPPORTUNISTIC_POP_SENTINEL,
        f"{label} opportunistic-pop witness",
    )
    if _plain_bool(_field(action, "cached_eof", label), f"{label} cached_eof"):
        _fail(f"{label} retained cached EOF state")
    result = _field(action, "result", label)
    if type(result) is not str:
        _fail(f"{label} result must be a string")

    if state in ("absent", "consumed"):
        if result != "target_unavailable":
            _fail(f"{label} called a receiver that is {state}")
        _exact_typed_value(
            _field(action, "control", label),
            _CONTROL_SENTINEL,
            f"{label} control witness",
        )
        if state == "absent":
            if _field(action, "request_id", label) is not None:
                _fail(f"{label} absent lookup retained a request ID")
            resolution = "absent"
            resolved_identity = identity
            consumed_by = None
        else:
            if identity is None or prior_drop is None:
                _fail(f"{label} consumed receiver lacks its prior ownership proof")
            _require_target_request_id(action, ordinal, identity)
            resolution = "receiver_consumed"
            resolved_identity = identity
            consumed_by = prior_drop
        return _ValidatedTarget(
            ordinal,
            client_index,
            "receiver_drop",
            resolution,
            "TargetLookup",
            resolved_identity,
            consumed_by,
        )

    if state != "owned" or identity is None:
        _fail(f"{label} has an invalid receiver state")
    if result != "receiver_dropped":
        _fail(f"{label} did not consume its owned receiver")
    _require_target_request_id(action, ordinal, identity)
    control = _validate_control_target(action, ordinal, identity, "disconnect")
    if control.disposition not in {
        "requested",
        "already_requested",
        "already_terminal",
    }:
        _fail(f"{label} omitted its disconnect disposition")
    return _ValidatedTarget(
        ordinal,
        client_index,
        "receiver_drop",
        "receiver_owned",
        "ControlDisconnect",
        identity,
        None,
    )


def _validate_drain_target(
    action: Any,
    ordinal: int,
    client_index: int,
    state: str,
    identity: AcceptedIdentityProjection | None,
    prior_drop: int | None,
) -> _ValidatedTarget:
    label = f"drain action {ordinal}"
    _validate_target_common_sentinels(action, ordinal)
    if _field(action, "error", label) is not None:
        _fail(f"{label} retained an error")
    _exact_typed_value(
        _field(action, "control", label),
        _CONTROL_SENTINEL,
        f"{label} control witness",
    )
    result = _field(action, "result", label)
    if type(result) is not str:
        _fail(f"{label} result must be a string")

    if state in ("absent", "consumed"):
        if result != "target_unavailable":
            _fail(f"{label} called a receiver that is {state}")
        if _field(action, "output", label) is not None:
            _fail(f"{label} unavailable lookup retained an output")
        _exact_typed_value(
            _field(action, "primary_pop", label),
            _PRIMARY_POP_SENTINEL,
            f"{label} primary-pop witness",
        )
        _exact_typed_value(
            _field(action, "opportunistic_eof_pop", label),
            _OPPORTUNISTIC_POP_SENTINEL,
            f"{label} opportunistic-pop witness",
        )
        if _plain_bool(
            _field(action, "cached_eof", label),
            f"{label} cached_eof",
        ):
            _fail(f"{label} unavailable lookup retained cached EOF")
        if state == "absent":
            if _field(action, "request_id", label) is not None:
                _fail(f"{label} absent lookup retained a request ID")
            resolution = "absent"
            resolved_identity = identity
            consumed_by = None
        else:
            if identity is None or prior_drop is None:
                _fail(f"{label} consumed receiver lacks its prior ownership proof")
            _require_target_request_id(action, ordinal, identity)
            resolution = "receiver_consumed"
            resolved_identity = identity
            consumed_by = prior_drop
        return _ValidatedTarget(
            ordinal,
            client_index,
            "drain",
            resolution,
            "TargetLookup",
            resolved_identity,
            consumed_by,
        )

    if state != "owned" or identity is None:
        _fail(f"{label} has an invalid receiver state")
    _require_target_request_id(action, ordinal, identity)
    cached_eof = _plain_bool(
        _field(action, "cached_eof", label),
        f"{label} cached_eof",
    )
    primary = _field(action, "primary_pop", label)
    opportunistic = _field(action, "opportunistic_eof_pop", label)
    output = _field(action, "output", label)

    if result == "drain_eof" and cached_eof:
        if output is not None:
            _fail(f"{label} cached EOF retained an output")
        _exact_typed_value(
            primary,
            _PRIMARY_POP_SENTINEL,
            f"{label} primary-pop witness",
        )
        _exact_typed_value(
            opportunistic,
            _OPPORTUNISTIC_POP_SENTINEL,
            f"{label} opportunistic-pop witness",
        )
        event_kind = "CachedEofRead"
    else:
        if cached_eof:
            _fail(f"{label} non-cached result retained cached EOF")
        primary = _validate_pop_target(
            primary,
            identity,
            "primary",
            f"{label} primary-pop witness",
        )
        if result == "drain_output":
            output = _validate_output_identity(output, identity, f"{label} output")
            if not _same_exact_value(primary.output, output):
                _fail(f"{label} output differs from its primary-pop witness")
            if getattr(opportunistic, "boundary", None) is True:
                opportunistic = _validate_pop_target(
                    opportunistic,
                    identity,
                    "opportunistic_eof",
                    f"{label} opportunistic-pop witness",
                )
                if opportunistic.output is not None:
                    _fail(f"{label} opportunistic EOF retained an output")
            else:
                _exact_typed_value(
                    opportunistic,
                    _OPPORTUNISTIC_POP_SENTINEL,
                    f"{label} opportunistic-pop witness",
                )
        elif result in ("drain_empty", "drain_eof"):
            if output is not None or primary.output is not None:
                _fail(f"{label} non-output result retained an output")
            _exact_typed_value(
                opportunistic,
                _OPPORTUNISTIC_POP_SENTINEL,
                f"{label} opportunistic-pop witness",
            )
        else:
            _fail(f"{label} has an invalid owned-receiver result")
        event_kind = "PrimaryEndpointPop"

    return _ValidatedTarget(
        ordinal,
        client_index,
        "drain",
        "receiver_owned",
        event_kind,
        identity,
        None,
    )


def _validate_target_custody(
    actions: tuple[Any, ...],
    program: ActionProgram,
    identities: AcceptedIdentityMapping,
) -> tuple[_ValidatedTarget, ...]:
    """Validate exact target resolution without total-ordering the race."""

    _require_exact_decoded_actions(actions)

    receiver_drop_by_client: list[int | None] = [
        None
    ] * EXPECTED_IN_RANGE_SUBMIT_COUNT
    records: list[_ValidatedTarget] = []
    for ordinal, (action, expected) in enumerate(
        zip(actions, program.actions, strict=True)
    ):
        if expected.kind not in ("cancel", "receiver_drop", "drain"):
            continue
        client_index = _plain_int(
            _field(action, "client_index", f"target action {ordinal}"),
            f"target action {ordinal} client_index",
        )
        if client_index != expected.client_index:
            _fail(f"target action {ordinal} client index changed")
        if client_index < 0 or client_index >= EXPECTED_IN_RANGE_SUBMIT_COUNT:
            _fail(f"target action {ordinal} client_index is outside 0..63")
        identity = identities.by_client_index[client_index]

        if expected.kind == "cancel":
            record = _validate_cancel_target(
                action,
                ordinal,
                client_index,
                identity,
            )
        else:
            if identity is None or identity.submission.action_ordinal > ordinal:
                receiver_state = "absent"
            elif receiver_drop_by_client[client_index] is None:
                receiver_state = "owned"
            else:
                receiver_state = "consumed"
            prior_drop = receiver_drop_by_client[client_index]
            if expected.kind == "receiver_drop":
                record = _validate_receiver_drop_target(
                    action,
                    ordinal,
                    client_index,
                    receiver_state,
                    identity,
                    prior_drop,
                )
                if record.event_kind == "ControlDisconnect":
                    if prior_drop is not None:
                        _fail(f"receiver-drop action {ordinal} consumed twice")
                    receiver_drop_by_client[client_index] = ordinal
            else:
                record = _validate_drain_target(
                    action,
                    ordinal,
                    client_index,
                    receiver_state,
                    identity,
                    prior_drop,
                )
        records.append(record)

    if len(records) != EXPECTED_TARGET_ACTION_COUNT:
        _fail(
            "target action custody does not cover exactly "
            f"{EXPECTED_TARGET_ACTION_COUNT} operations"
        )
    return tuple(records)


def _build_target_access_order(
    actions: tuple[Any, ...],
    program: ActionProgram,
    submission_order: SubmissionProtocolOrder,
) -> TargetAccessOrder:
    identities = _build_accepted_identity_mapping(actions, submission_order)
    records = _validate_target_custody(actions, program, identities)
    planner = _ProtocolGraphPlanner(submission_order.graph.node_count)
    planner.layer(submission_order.graph)

    by_action: list[TargetActionEvent | None] = [None] * EXPECTED_ACTION_COUNT
    by_client: list[list[TargetActionEvent]] = [
        [] for _ in range(EXPECTED_IN_RANGE_SUBMIT_COUNT)
    ]
    for record in records:
        event = planner.allocate(record.action_ordinal, record.event_kind)
        endpoints = submission_order.interval_order.endpoints.by_action[
            record.action_ordinal
        ]
        planner.edge(
            endpoints.invocation.node,
            event.node,
            "target-action-invocation-before-main-event",
        )
        planner.edge(
            event.node,
            endpoints.response.node,
            "target-main-event-before-action-response",
        )

        identity = record.identity
        if record.resolution == "absent":
            if identity is not None:
                registry_publish = identity.submission.registry_publish
                if registry_publish is None:
                    _fail("accepted identity omitted its registry publication")
                planner.edge(
                    event.node,
                    registry_publish.node,
                    "absent-target-lookup-before-registry-publication",
                )
        else:
            if identity is None:
                _fail("found target omitted its accepted identity")
            registry_publish = identity.submission.registry_publish
            if registry_publish is None:
                _fail("accepted identity omitted its registry publication")
            planner.edge(
                registry_publish.node,
                event.node,
                "accepted-registry-publication-before-target-main-event",
            )
        if record.resolution == "receiver_consumed":
            prior_drop = record.prior_receiver_drop_action
            if prior_drop is None:
                _fail("consumed receiver lookup omitted its prior drop")
            prior_response = submission_order.interval_order.endpoints.by_action[
                prior_drop
            ].response
            planner.edge(
                prior_response.node,
                event.node,
                "receiver-consumption-before-unavailable-lookup",
            )

        target = TargetActionEvent(
            record.action_ordinal,
            record.client_index,
            record.action_kind,
            record.resolution,
            identity,
            record.prior_receiver_drop_action,
            event,
        )
        if by_action[record.action_ordinal] is not None:
            _fail(f"target action {record.action_ordinal} was allocated twice")
        by_action[record.action_ordinal] = target
        by_client[record.client_index].append(target)

    graph = planner.build()
    return TargetAccessOrder(
        submission_order,
        TargetAccessMapping(
            identities,
            tuple(by_action),
            tuple(tuple(client) for client in by_client),
            planner.protocol_events,
        ),
        graph,
    )


def build_target_access_order(repetition: Any) -> TargetAccessOrder:
    """Authenticate and reconstruct accepted identities and target access."""

    try:
        actions = repetition.actions
        diagnostics = repetition.diagnostics
        shutdown = repetition.shutdown
        action_counter_final = diagnostics.action_counter_final
    except AttributeError:
        _fail("decoded repetition lacks target protocol custody fields")
    _require_exact_decoded_actions(actions)
    program = regenerate_authenticated_action_program()
    _verify_static_actions(actions, program)
    interval_order = _build_action_interval_order(
        actions,
        action_counter_final,
        expected_action_count=EXPECTED_ACTION_COUNT,
    )
    submission_order = _build_submission_protocol_order(
        actions,
        shutdown,
        program,
        interval_order,
    )
    return _build_target_access_order(actions, program, submission_order)


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
