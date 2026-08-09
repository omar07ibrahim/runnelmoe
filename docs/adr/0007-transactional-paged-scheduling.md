# ADR 0007: transactional paged state and deterministic continuous scheduling

- Status: accepted; implementation in progress; measurement pending
- Date: 2026-08-03
- Last amended: 2026-08-06 (preemption, lifecycle, and run-observer contracts)
- Milestone: M5

## Context

The M1/M4 tiny runtime is an intentionally direct numerical reference. Its
current token operation clones the complete sequence state, materializes an
attention-score vector proportional to context length, allocates expert
temporaries inside the operation, and commits routing, expert execution, KV
state, and logits in one function. Those properties make the reference easy to
inspect, but wrapping that function in a request queue would not provide
bounded long-prefill state, transactional cancellation, complete scheduler
accounting, or real cross-request expert coalescing.

M5 must establish those boundaries without weakening the independently tested
scalar semantics. It also needs evidence that separates correctness and
resource gates from timing outcomes. The existing tiny-v2 adapter has a
16-position context, so it cannot by itself exercise a meaningful number of
state pages or distinguish streaming from materialized attention. A new open,
synthetic adapter revision is needed for mechanics evidence; it is not a claim
about language quality or production-scale model behavior.

This decision was written before M5 implementation or timing. It is an
independent design based on this repository's existing contracts and primary
language/runtime specifications. No implementation, fixture, prose, or result
from the credited Kimi K3 C prior-art repository is used.

## Decision

### Scope and explicit non-claims

M5 adds:

- a 1,024-position revision of the generated tiny adapter;
- token-major paged K/V state and stable streaming causal attention;
- a transactional prepare, grouped-expert, finish, and commit pipeline;
- a deterministic synchronous scheduler core with a bounded Tokio owner;
- chunked prefill, continuous batching, equal-weight deficit round robin,
  expert-aware grouping, bounded output queues, cancellation, deadlines, and
  seeded sampling; and
- correctness, accounting, fairness, latency, throughput, and RSS evidence.

The tiny model's weights remain eagerly resident during M5 execution. The M2
store can authenticate and bound the bytes used to construct that model, but
M5 does not claim that cache leases remain live across every expert call. M5
therefore does not establish arbitrary-model performance, language quality,
weighted or multi-tenant fairness, storage-device bandwidth, constant RSS,
or production long-context behavior. The adapter-v3 results demonstrate
bounded multi-page mechanics on an openly generated one-layer model with
hidden width 8.

### Synthetic adapter v3

`runnel.tiny-causal-moe` adapter version 3 has the same tensor equations,
tokenizer, topology, deterministic tensor recipe, and compact BF16 expert
representation as adapter v2. Its sole model-semantic change is a frozen
`context_length` of 1,024 rather than 16. Versions 1 and 2 remain unchanged and
retain their existing identities and golden vectors.

The v3 fixture receives its own canonical manifest identity, specification,
short golden route/logit/token vectors, and independent PyTorch validation.
Long tests use direct valid token IDs generated as follows:

```text
P(n)[0] = 1
P(n)[i] = [14, 16, 6][(i - 1) mod 3], for 1 <= i < n
```

The repeated synthetic input has no quality interpretation. Page and
attention tests cover positions 1, 15, 16, 17, 255, 256, 257, 1,023, and
1,024. The fixture object remains kilobyte-scale; neither fixture generation
nor any M5 command downloads model weights.

### Runtime boundary

`runnel-runtime` remains synchronous and Tokio-free. A model-agnostic
`DecoderAdapter` trait owns these associated types and phases:

```text
vocabulary_size() -> fixed token count
is_stop_token(token) -> terminal policy
state_layout(limits, page_tokens) -> StateLayout
new_state(layout) -> State
prepare_token(state, token, workspace) -> PreparedToken
expert_tasks(prepared) -> ordered tasks
execute_expert(task, workspace) -> ExpertContribution
finish_token(prepared, contributions, workspace) -> PendingStateCommit
pending_logits(pending) -> logits
with_validated_state_commit(state, pending, scoped_apply) -> result
apply_state_commit(permit)
```

`TinyModel` implements the trait. Existing `forward_token`, `run_tokens`, and
`generate_greedy` remain compatibility compositions of the same phases, so
the original public path and scheduler path cannot silently acquire different
model equations. The trait is an externally implementable, trusted extension
boundary: implementations own model work, identities, and permits, while the
scheduler validates every observable envelope. Adapter identities have a
checked public constructor and are validation tags rather than capabilities.
Their fields remain private; conforming adapters keep the tag binding inside
model-derived payloads opaque, and the scheduler never treats possession of a
tag as authorization.

`PreparedToken` is immutable and contains the pending key/value vectors,
normalized expert input, stable router result, model identity, state identity,
state revision, position, and adapter transaction identity. Adapter expert
tasks and contributions carry that adapter identity, router rank, and expert
ID. The scheduler wraps them in an engine-owned envelope containing:

```text
(request_id, request_slot, slot_generation, engine_transaction_id,
 adapter_transaction_id, state_id, state_revision, position,
 router_rank, expert_id)
```

Runtime adapter types never depend on scheduler request slots. Both layers
validate their complete identity before expert execution and again before
scatter. Every reusable numeric scatter lane is prearmed with the complete
scheduler envelope above; the eventual single-use completion must match that
authorization field-for-field before it can occupy the lane. A mismatch leaves
the expected authorization and contribution payload unchanged. Rollback and
wave reset clear both fields only after proving that no service reservation is
discarded. Request IDs, request slots, slot generations, engine/adapter
transaction IDs, model/state IDs, revisions, and positions use checked
increments; exhaustion fails before work selection and never wraps or reuses
an identity. Adapter transaction identities are observed against one
engine-lifetime high-water mark and must increase strictly across requests as
well as within one request.

`finish_token` rejects a missing, duplicate, foreign, stale, wrong-rank,
wrong-expert, nonfinite, or dimension-mismatched contribution before it
creates a pending commit. Contributions may execute in expert-ID order, but
the mixture is always reduced in router-rank order. A completion permutation
cannot alter the numerical accumulation order. Unit tests mutate each slot,
request, engine transaction, adapter transaction/model/state/revision/position,
selection, router-rank, expert, and scatter component independently, and
reuse one physical lane in a later wave; every stale write is rejected without
damaging the current lane.

Model state carries a nonzero identity and monotonically increasing revision.
A pending state commit can apply only to the state/revision from which it was
prepared and can apply at most once. Adapter `with_validated_state_commit`
performs fallible model/state identity, revision, position, length, and
finiteness checks, then invokes a higher-ranked synchronous callback with a
single-use `StateCommitPermit` holding exclusive state access. The callback's
result cannot depend on the fresh permit lifetime, so safe code cannot return
the permit in a future or retain it across an outer asynchronous yield.
Adapter `apply_state_commit` is allocation-free, has no public error path, and
only copies into already validated slices before updating revision. It returns
no position or status that a caller could discover was inconsistent only after
the state linearization point.

The scheduler separately validates control state, sampler preview, output
reservation, and request phase before entering the scoped callback. A held
generation-tagged endpoint guard excludes both receiver mutation and endpoint
recycle from reservation through publication. Inside the callback the final
live clock/control snapshot gates the infallible adapter apply, DRR debit, RNG,
phase, and append-only trace update without further fallible work, panic, or
yield. The endpoint guard remains held after the callback so an adapter error returned
after applying can be classified as terminal before the optional output and
terminal payload become visible together. Runtime types never validate
scheduler-owned capacity. This two-layer fallible-permit/infallible-apply split
is what lets stale/ABA rejection coexist with atomic K/V, RNG, and result
publication.

### Paged K/V state

State is token-major. The scheduler configuration fixes 16 tokens per page for
the v3 experiment. Each page contains contiguous key and value payloads:

```text
page_payload = 2 * page_tokens * hidden_size * sizeof(f32)
page_count   = ceil(max_state_tokens / page_tokens)
state_payload = page_count * round_up_64(page_payload)
state_charge  = state_payload + page_count * 64
```

The final logical page is charged at full page capacity. All arithmetic is
checked before request bytes are copied. The scheduler reserves the complete
state charge before construction, and explicit scheduler states allocate
every payload page before model work starts. This makes commit allocation-free
and prevents a later page boundary from violating admission. The compatibility
constructor may bind lazily to a model, but it performs the same full bounded
allocation before its first token.

State exposes logical length, context limit, page geometry, state identity,
revision, and accounted payload bytes, but not mutable page storage. Failed
preparation, expert execution, scatter, sampling, output reservation,
cancellation, or deadline checks do not change logical length, page contents,
or revision.

### Streaming attention

Tiny attention uses a stable three-pass causal reduction over the paged state
plus the pending current key/value:

1. visit positions in ascending order to find the finite maximum score;
2. revisit them in ascending order to sum `exp(score - maximum)`; and
3. revisit them in ascending order to accumulate values and divide by the
   positive finite denominator.

No vector proportional to context length is allocated. Query, score, and
value dimensions use checked indexing, and every intermediate required by the
adapter contract must be finite. Tests retain a separately implemented
materialized f64 reference and the independent PyTorch oracle. Routes and
tokens remain exact; continuous outputs retain the existing componentwise
`atol = 1e-5`, `rtol = 1e-4` gate. A streaming-rounding change that violates
that gate blocks M5 rather than changing the tolerance after measurement.

### Reusable expert workspace and coalescing

Adapter construction reports a checked `WorkspaceLayout`. The scheduler owns
the fixed number of worker workspaces for its lifetime and charges their
declared buffer capacities. No token operation grows a workspace. Within one
token wave, work is placed in a flat task array fallibly preallocated to the
configured maximum. An allocation-free unstable sort uses the complete unique
key `(expert_id, request_id, position, router_rank, transaction_id)`, producing
the same total order as a stable grouping without allocating merge scratch. A
contiguous expert group reuses one expert workspace while its weights are hot.
This is expert-aware coalescing, not a claim of a batched GEMM kernel.

At most one dependent token from a sequence can be in a wave. Batch and
contribution caps are validated and partitioned before allocation. Canceling
one member leaves stable task identities; it cannot shift a sibling's scatter
slot. Worker-owned scratch remains globally charged until the worker returns
it, even if every interested request cancels.

After grouped execution and validation, ready tokens are considered for commit
in the original ring-selection order. Expert ID, task completion order, and
request ID sorting inside a group never redefine the service sequence.

### Deterministic scheduler core

A new `runnel-scheduler` crate contains a pure, synchronously step-able
`SchedulerEngine`. Its external operations are conceptually:

```text
try_submit(request)
cancel(request_id)
advance_clock(monotonic_ns)
step()
drain_events(request_id, limit)
take_terminal(request_id)
snapshot()
shutdown()
```

Accepted requests receive monotonically increasing internal IDs and
generation-tagged slots. Stable accepted ingress order is part of the retained
input trace. Wall-clock timestamps never break a scheduling tie.

Admission is bounded FIFO. A head request that cannot acquire active-state
bytes is not bypassed by a later request. Active requests occupy a cyclic ring
in accepted-ID order. The ring retains a cursor and round epoch. A request
admitted during an open round is appended but has `join_epoch = current + 1`,
so it cannot jump ahead of members already awaiting their visit.

M5 uses the equal-weight, unit-cost specialization of deficit round robin. A
round visits exactly the membership snapshot eligible for that epoch once,
starting at the retained cursor. On a runnable member's visit, its deficit is
increased by one quantum and capped at one. If it has one credit, it reserves
one model-position service and the cursor advances to the next ring member.
The reservation is debited only by a successful commit. A request that blocks
before selection is visited without service and its credit is reset to zero;
reactivation begins with zero credit in the next eligible round. Cancellation
or terminal failure removes the member and its reservation, with the cursor
continuing at the removed member's successor. A recoverable failure before
commit restores the one reserved credit but cannot select that request twice in
the same round. When every epoch member has been visited, the round closes and
the next scan starts at the first cyclic survivor from that closed membership
snapshot. A request appended during the round cannot itself occupy the
rollover cursor while such a survivor remains. After that first due survivor,
the scan follows ordinary cyclic membership order and need not place every
survivor before every arrival. The closed-snapshot cursor is re-normalized
after terminal or cancellation removal until the next round successfully
opens. If no snapshot member survives, the ordinary physical successor remains
authoritative. These rules, rather than a fresh request-ID sort per wave, are
the independently replayed policy.

One token wave selects at most eight distinct sequences and at most one
position from each while continuing the current ring scan. A scheduler `step`
may execute up to four waves, which is the frozen prefill chunk bound; the
cursor and round state persist across waves and steps. New requests may be
accepted between steps and join their first eligible round; an earlier request
need not finish first. Decode work is one model position per visit. Thus
chunking bounds actor monopolization without turning a four-token chunk into an
indivisible fairness unit. Blocked-on-output, terminal, canceled, expired, and
otherwise non-runnable requests consume no service.

The named baseline uses the identical v3 scalar model, paged state, streaming
attention, sampling, configured batch width eight, preallocated workspace,
memory budget, and output sink. Its one-member policy rounds make effective
wave occupancy one and run the accepted FIFO head to completion, so no
cross-request expert grouping occurs. The named candidate uses the same
configured geometry with the DRR/coalesced policy above. The comparison
isolates scheduling and grouping rather than numeric representation, scratch
capacity, ISA, or storage.

Implementation mapping (2026-08-06): `SchedulerConfig` now binds one immutable
versioned policy without changing validated geometry or shared charges. The
continuous default retains the original full eligible-member DRR rounds. The
FIFO baseline opens one-member rounds for the oldest active member while the
same maximal active FIFO prefix remains promoted and charged; repeated rounds
therefore target that head until removal and never bypass it while blocked,
without a second execution or commit path. This note records implementation of
the frozen comparison and does not amend its workload, metrics, or claim rules.
The continuous implementation also retains a bounded closed-snapshot epoch
marker so mid-round arrivals cannot add a second rotation to a survivor's
service gap. The real tiny-v3 scalar integration independently replays the
first 16 events and exact integer fairness metrics, then verifies the complete
`continuous-arrival-1000` event formula and cleanup contract. No timing enters
either gate and no performance result is implied.

Pre-measurement fairness clarification (2026-08-06): a raw physical cursor at
round closure allowed members appended during that round to run before the
first-due surviving snapshot member. Under the already frozen
continuous-arrival input, the anchor's first recurrence would then be turn 31,
with 30 intervening commits, contradicting the hard maximum of 15 below. The
closed-snapshot normalization above prevents an arrival from occupying that
rollover cursor and resolves the contradiction before any M5 timing, service
golden, or performance conclusion. It changes no workload, bound, numeric
path, FIFO baseline, or claim rule.

### Request state machine and token transaction

The request lifecycle is:

```text
validated -> queued -> admitted -> preparing -> expert_owned
          -> ready_to_commit -> ready | preempted | output_blocked | terminal
preempted -> control scan -> ready | terminal
output_blocked -> receiver progress/control scan -> ready | terminal
```

Before the linearization point, the actor must hold:

- an admitted, preallocated state slot;
- complete identity-checked finite expert contributions;
- finite logits and, when this position emits a token, a sampler preview;
- a reserved non-terminal output slot when a token will be emitted; and
- all required request and shared ledger reservations; and
- a single-use scheduler `TransactionCommitPermit` wrapping the adapter state
  permit, sampler preview, reserved output slot, and phase transition.

The single actor obtains that permit, then rechecks cancellation and deadline
without yielding. If control still permits the token, it performs an
infallible, non-yielding apply of pending K/V, optional next RNG state,
optional output event, and request phase. Intermediate prefill positions commit
state only. The final prompt position and subsequent decode positions also
commit one sampled output; generation evaluates exactly `prompt_length +
max(max_new_tokens - 1, 0)` model positions. Cancellation or expiry observed
before the final check wins and publishes none of that position's fields. A
signal observed after it loses to that one committed position and becomes
terminal at the next boundary. An uninterruptible expert call may finish after
cancellation, but run cleanup is not complete until its globally charged
workspace returns.

Cancellation is idempotent. If cancellation and deadline expiry are both
already visible at a boundary, `cancelled` wins. Cancellation checks occur
before preparation, after routing and before expert execution, after expert
execution, before publication planning, and at the final snapshot immediately
before commit. No retry, wake-up, or stale completion may advance state or RNG
twice. A request suppressed at the post-router boundary contributes no expert
call, while the already charged shared wave scratch is still drained normally.

Four cancellation timestamps are distinct. `cancel_linearized` is the atomic
control transition. `terminal_decided` is the actor's irrevocable outcome after
no request-owned work can commit. `request_owned_zero` is when prompt, record,
slot, state, pending, output, and terminal charges have been drained, discarded,
or reaped. `worker_quiescent` is when any shared workspace carrying canceled
work has returned; run cleanup completes only after every request is zero and
all shared workers are quiescent. A terminal result may become visible after
`terminal_decided` even while already charged shared scratch drains, but the
run and shutdown cannot report complete reclamation then. Evidence reports
both `request_owned_zero - cancel_linearized` and
`worker_quiescent - cancel_linearized`.

### Deterministic cooperative preemption

`waves_per_step` is the closed scheduler-pump budget. A successful,
non-terminal position committed in the final configured wave enters
`preempted` atomically with its adapter-state, RNG, output, service-credit, and
service-trace commit. The engine then returns to its owner, allowing the actor
to service already bounded command/control work before the next pump. A
reported survivor selects an explicit Tokio cooperative yield before that
next blocking pump. A completion, full output queue, cancellation, deadline,
or contained adapter failure takes precedence over `preempted`.

This is resident cooperative preemption, not state eviction. The request keeps
its exact ring member, cursor/epoch relationship, zero post-commit deficit,
adapter state and binding, prompt and active reservations, output endpoint,
sampling state, and ledger ownership. It is never removed and reinserted and
does not pass through the admission FIFO. At the next pump the engine first
resolves cancellation and inclusive deadlines while the request is still
`preempted`; only surviving requests make one allocation-free transition to
`ready` before a ring visit. The two-pass resume audit validates all resident
owners and the complete ring/queue relationship before changing any phase, so
an invariant failure cannot partially resume a batch.

The stable mechanism identifier is
`resident-step-budget-preemption-v1`. `batch_width <= 8` and
`waves_per_step <= 4` bound one pump to 32 selected positions, four positions
per request, and at most eight preempted survivors.

`StepReport.preempted_requests` counts requests that remain `preempted` after
the final wave's post-commit control scan, not transient planned transitions;
these are precisely the survivors that force an actor cooperative yield.
`StepReport.resumed_requests` counts the control-cleared transitions at the
next opening boundary.
`EngineSnapshot.preempted_requests` is the current resident count and those
requests also contribute to `active_requests`, preserving actor liveness. A
post-final-snapshot cancellation can therefore commit one position and report
zero preemptions when the same step immediately terminalizes it. Preemption
and resume add no service event, ledger mutation, RNG transition, output, or
allocation. Step-partition tests require identical service traces, terminals,
and greedy and seeded tokens for one versus four waves under both policies.

The FIFO comparison policy deliberately retains its run-to-completion ring
order across a pump yield; the continuous policy retains its DRR cursor and can
serve the next due member after the yield. Reclaiming opaque adapter state or
swapping it to another tier would require a separately authenticated
snapshot/restore ABI and explicit storage accounting. That is not claimed by
this M5 mechanism.

### Backpressure and bounded Tokio owner

The production owner is a thin Tokio actor around the deterministic engine.
Its command channel, accepted queue, active slots, retained terminal results,
per-request output events, trace events, expert tasks, and workspaces all have
hard count and byte caps. A full command or admission queue fails immediately
with `resource_exhausted`; repeated rejection cannot grow capacity or change
the accepted schedule.

The ordinary bounded command lane carries submissions only. Each accepted
handle owns an atomic cancellation/disconnect flag inside its already charged
request control record; setting it is wait-free with respect to ordinary
command capacity and wakes the actor through a pre-existing notification.
Output draining reads the per-request receiver directly and likewise consumes
no command slot. Shutdown sets one actor-wide atomic closed flag and uses a
separately reserved wake notification. Tests saturate the ordinary lane before
cancel, receiver drop, output drain, and shutdown, and require all four control
paths to progress.

Each request has a bounded output queue and a separately reserved terminal
slot. A full output queue blocks only that request; siblings continue. Draining
an output queue wakes the actor. Dropping a receiver atomically requests
cancellation and discards any committed but undrained output after recording
its already-published event identities. With a live receiver, committed output
preceding cancellation remains drainable and charged until consumed; the
terminal slot cannot be blocked by it. Shutdown rejects new submissions,
resolves every admitted handle, and does not report final reclamation until all
worker ownership has returned or transferred to an explicitly charged
actor-owned residue.

Deadlines use an injected monotonic `u64` nanosecond clock in the core and Tokio
monotonic time only in the owner. `advance_clock` rejects a value below the
current value; `u64::MAX` is a valid terminal clock value after which no larger
advance exists. `now >= deadline` is expired. Duration and absolute-time
arithmetic is checked. Tests use a manual clock and deterministic gates rather
than sleeps.

The deadline lifecycle matrix is explicit and prefix-preserving:

| observed state | deterministic gate |
| --- | --- |
| queued | inclusive expiry before promotion in `cancellation_and_deadlines_are_inclusive_with_cancellation_precedence` |
| active/ready | non-head expiry without service in `fifo_non_head_deadline_terminalizes_without_receiving_service` |
| expert-owned | `PostRouterPreExpert` expiry in `cancellation_and_deadline_actions_suppress_each_preapply_checkpoint_exactly` |
| ready-to-commit | `ReadyToCommitPrePlan` and `CompositePermitPreFinalSnapshot` expiry in the same checkpoint matrix |
| output-blocked | committed-prefix expiry in `output_blocked_cancellation_and_deadline_preserve_the_committed_prefix` |
| preempted | control-first expiry and cancellation precedence in `preempted_controls_win_before_resume_and_preserve_the_committed_prefix` |

The manual clock is inclusive in every row. No suppressed position changes
state, RNG, output, service credit, or trace; any earlier committed prefix
remains drainable and is reported by the terminal result.

### Sealed deterministic checkpoint instrumentation

The opt-in Cargo feature `deterministic-checkpoint-instrumentation` exports a
hidden concrete `CheckpointPlan` for integration tests. It is not a production
control surface. Scheduler code never invokes a caller callback and never
retains a plan. Preparation accepts at most 64 directives, performs the plan's
sole allocation, sorts targets canonically, rejects duplicate boundaries, and
binds every target to the engine's weak control-table domain, exact slot
generation, request ID, position, and optional deadline. A plan therefore
cannot keep an engine alive, cross engines, or redirect through slot reuse.
Debug output redacts request identity.

The closed boundary set is:

1. `PostRouterPreExpert`, after authenticated task construction and sorting but
   before any expert invocation;
2. `ReadyToCommitPrePlan`, after validated contributions and pending state but
   before output/RNG publication planning;
3. `CompositePermitPreFinalSnapshot`, inside the validated adapter/service
   permit composition immediately before the final live clock and control
   read; and
4. `PostFinalSnapshot`, after those values are fixed for the current position
   and before infallible apply.

The closed actions are observe-only, generation-checked cancellation,
inclusive deadline expiry, and cancellation followed by expiry. The combined
action always publishes cancellation first, with no scheduler observation
between the two mutations. The final-snapshot action intentionally loses to
exactly that already-snapshotted position; every earlier action suppresses all
state, RNG, output, service-credit, and service-trace publication for its
target position. Cancellation remains the terminal outcome when cancellation
and expiry are both visible at the next boundary.

Effects and fire ordinals are written into preallocated plan entries. Execution
does not allocate for instrumentation. The ordinary feature-off path uses the
sealed `NoCheckpoints` generic driver, so it has no callback, dynamic dispatch,
plan scan, or instrumentation allocation. Plan preparation and effect
inspection are excluded from every M5 timing region.

The deterministic matrix covers every boundary with cancellation, expiry, and
their ordered combination; exact seeded RNG/state/output/service prefixes;
output-blocked cleanup; completing-position precedence; foreign engines; stale
slot generations; duplicate/limit/deadline validation; and balanced request
ownership. A separate scalar tiny-v3 integration workload freezes 12 `P(896)`
requests with queued cancellation plus post-final/pre-apply position 0,
post-router/pre-expert position 8, and composite-permit position 895 actions.
This closes the transaction-boundary injection mechanism. Together with the
public queued, ready, output-blocked, and preempted manual-clock tests, it also
closes the lifecycle-state deadline matrix and resident cooperative-preemption
correctness gate. The run-level timing observer and accepted M5 evidence remain
explicit gates.

### Sealed bounded run observer

Pre-measurement amendment (2026-08-06): the timing observer contract below was
closed before an M5 timing run, correctness capture, raw timing row, summary,
or figure existed. Earlier correctness executions are not timing evidence and
will not be relabeled as the preregistered capture.

The semantic service trace deliberately contains no clock values, and output
queues are drained only after the primary timing endpoint. Therefore neither
endpoint response order nor step return time can reconstruct TTFT, ITL,
first-work queue delay, phase boundaries, or terminal-decision time. The
opt-in Cargo feature `m5-run-observer-instrumentation` exports one concrete,
sealed `RunObserver`; it exports no clock source, callback trait, trait object,
or arbitrary event sink. Each observer owns an unexported, nonallocating
`Instant`-origin `RunClock`, and every production observed API samples only
that source. An external clock cannot be mixed into a run. The ordinary
feature-off path is monomorphized with a zero-sized null observer and performs
no observer allocation, clock read, branch, lock, or event write.

The observer is harness-owned rather than scheduler-owned. Its allocations are
not included in the scheduler logical ledger, are constructed before release,
are identical for baseline and candidate, and remain included in fresh-child
`VmHWM`. Evidence reports their requested bytes separately. No observer cost is
subtracted from elapsed time. This preserves the closed scheduler-owned ledger
identity while making the instrumentation overhead visible.

An observer is weakly authenticated to one engine control-table domain, owns
one clock origin captured before its fallible reservations, and accepts one
atomic evidence admission. Binding before release checks the domain, every
mutable field is pristine, accepted-count capacity, maximum output count, and
every checked size product. Runtime callbacks before release poison the
observer and prevent later publication. Successful binding preallocates all
storage; a binding or allocation failure precedes publication and leaves the
observer and engine reusable. The clock is an inline scalar and performs no
allocation or reference-count operation. The evidence configuration allocates
exactly:

- 32 accepted-request summary slots, matching the normal total-outstanding
  limit;
- 256 optional output-commit timestamps, equal to 32 requests times eight
  maximum outputs;
- a fixed nine-bin nonempty-wave occupancy histogram for widths zero through
  eight, where bin zero must remain zero; and
- scalar checked totals for steps, waves, selections, expert contributions,
  expert groups, commits, terminal decisions, resident preemptions, resumes,
  timestamp-bearing milestone observations, release, final emission, final
  terminal decision, observer requested bytes, and the three checked
  state-utilization sampling integers.

Rejected offers never receive an internal request ID and remain harness rows,
not observer slots. Each accepted summary retains only:

```text
offered_index, opaque request_id
prompt_len, max_new_tokens, total_positions, resolved_deadline_ns
admitted_ns, first_work_start_ns, first_decode_start_ns
prefill_complete_ns, output_commit_ns[0..emitted_tokens]
cancel_linearized_ns, terminal_decided_ns, terminal_outcome
request_owned_zero_ns, worker_quiescent_ns
committed_positions, emitted_tokens
```

Prompt tokens, output token values, sampling seeds and state, logits, router
scores, expert IDs, adapter identities, and deadline contents outside the
synthetic contract are never retained by this observer. The semantic service
trace remains the authority for request/position/phase order, and the ledger
trace remains the authority for ownership and peaks.

The timestamp boundaries are closed:

- `admitted_ns` is the batch `release_ns` installed by the allocation-free
  publication boundary;
- `first_work_start_ns` is sampled immediately before the first
  `prepare_token` call for that request;
- `first_decode_start_ns` is sampled at the same boundary for position
  `prompt_len`, if that position exists;
- `prefill_complete_ns` and each emitted-output timestamp are sampled
  immediately after the infallible state/service/endpoint publication for the
  relevant position;
- `cancel_linearized_ns` is sampled immediately after the successful atomic
  cancellation transition, including a deterministic checkpoint action;
- `terminal_decided_ns` is sampled immediately after terminal publication;
- `worker_quiescent_ns` is sampled when synchronous adapter work carrying that
  request has returned and the reusable wave no longer carries its work; and
- `request_owned_zero_ns` is sampled immediately after terminal endpoint,
  record, slot, prompt, state, and retained-reservation reap completes.

The direct evidence core has one synchronous worker. For a cancellation
resolved before work, or after all selected adapter calls have returned,
`worker_quiescent_ns` may equal `terminal_decided_ns`; the two named fields are
still retained and verified independently. The later actor/serving surface may
have a distinct asynchronous quiescence boundary and cannot infer it from this
direct-core result.

Every nonempty wave increments exactly one occupancy bin after its selections
are fixed. Completed step reports feed the checked aggregate counters. Observer
request counts, committed positions, emitted indices, terminal values, and
global totals must reconcile exactly with the accepted batch, terminal
results, step reports, and complete service trace. Both trace capacities remain
exactly 8,192 and their sticky overflow states remain independent of observer
health.

Clock regression or saturation, a pre-release callback, duplicate admission
or lifecycle milestone, unknown request, noncontiguous
position or output index, impossible phase or terminal boundary, an
unobserved cancellation source, counter overflow, observer capacity breach,
or a callback after finish sets one sticky closed failure code. Runtime
observer failure retains the longest valid prefix, never returns into or
changes the scheduler transaction, and makes that run ineligible at final
verification. Formatting, serialization, hashing, cloning, and JSON writing
occur only after the observed interval. The observer path performs no dynamic
allocation, mutex operation, atomic reference-count churn, or user callback.
Foreign-engine attempts instead fail the observed API preflight without
mutating either engine or observer, so a pristine observer remains reusable.
Test-only manual-clock hooks are compiled solely under `cfg(test)` to freeze
exact boundaries and sticky regression/saturation behavior; they are not part
of the feature's production surface.

State-page utilization is a time-sampled logical slot diagnostic, not physical
residency. Immediately after every successful position commit, including a
completing position before its conceptual state is removed, the observer adds
the current committed-position count across all active state to one numerator
and the current admitted full-page slot count across that same state to one
denominator:

```text
live_sample              = sum(active committed_positions)
allocated_sample         = sum(ceil(active total_positions /
                                   state_page_tokens) * state_page_tokens)
live_token_sample_sum   += live_sample
allocated_slot_sum      += allocated_sample
state_sample_count      += 1
utilization              = live_token_sample_sum / allocated_slot_sum
```

Promotion adds one request's rounded slot count to the live observer state;
terminal cleanup removes its slot and committed-token counts only after any
same-position sample. Cancellation without a commit adds no sample. The raw
evidence retains both integer sums and the sample count; it never averages
per-request ratios. This definition does not imply that pages were faulted in,
cache-resident, compressed, or backed by storage.

Observer acceptance requires exact manual-clock TTFT/ITL/queue/prefill/decode/
terminal cases; observer-on/off token, terminal, service-trace, ledger-trace,
and seeded-RNG parity under both policies and one/four-wave partitioning;
exact-capacity and sticky-failure neutrality; foreign-domain and clock-failure
tests; cancellation/deadline/reap lifecycle coverage; a no-growth allocation
fingerprint across the observed interval; and no-default, observer-only,
checkpoint-plus-observer, actor-stress-plus-observer, and all-feature builds.
Only after those gates and the ordered 26-row correctness producer pass may
the five-cell timing capture begin.

### Seeded sampling

Sampling policies are:

```text
Greedy
Sample { seed: u64, temperature: finite f32 > 0,
         top_k: 1..=vocab_size, top_p: finite f32 in (0, 1] }
```

The reusable sampling workspace has one target-stable 32-byte candidate slot
per vocabulary item. Its checked preallocation contract is
`payload = vocab_size * 32` and `charge = round_up(payload, 64)`; multiplication,
rounding, and the host `isize::MAX` allocation bound are validated without
allocating. The workspace reserves exactly that candidate count once and does
not grow during preview.

Greedy chooses the lowest token ID at the greatest finite logit and does not
advance RNG. Sampled candidates are ordered by descending scaled logit then
ascending token ID. Top-k is applied first. Stable softmax probabilities are
then computed, and the shortest prefix whose cumulative probability is at
least top-p is retained. The retained probabilities are renormalized.

Sampling arithmetic is frozen independently of model arithmetic. Each finite
f32 logit and the positive finite f32 temperature are converted exactly to f64;
division, maximum subtraction, `exp`, weight sums, top-p accumulation,
renormalization, and categorical accumulation then use f64. Candidates and all
sums are visited in the stable candidate order. A nonfinite scaled value,
exponential, sum, or probability is an internal failure before preview. At
least the maximum candidate has exponential weight one, so an all-zero sum is
also a failure. Top-p retains the first prefix satisfying
`cumulative_weight >= f64(top_p) * total_weight`. Its retained sum is recomputed
in candidate order; categorical selection accumulates `weight / retained_sum`
in that order and chooses the first cumulative value strictly greater than
`u`. f64 intermediates are never narrowed before the token is chosen.

The independent Python oracle uses IEEE binary64 `math.exp`, the same visit
order, and the same comparisons. Golden cases include logit ties, a one-token
top-k, top-p equality and just-below/above boundaries, `u = 0`-adjacent and
one-adjacent values, the smallest positive f32 temperature, underflowed
non-maximum weights, and invalid/nonfinite parameters. Cross-language golden
tokens and retained candidate IDs are exact; finite probability diagnostics
use a maximum four-ULP binary64 tolerance rather than a false bitwise-libm
claim. This diagnostic tolerance is fixed before M5 timing begins.

The version-1 RNG is SplitMix64. Given state `s`, one preview computes:

```text
s' = s + 0x9e3779b97f4a7c15 (mod 2^64)
z  = s'
z  = (z xor (z >> 30)) * 0xbf58476d1ce4e5b9 (mod 2^64)
z  = (z xor (z >> 27)) * 0x94d049bb133111eb (mod 2^64)
z  = z xor (z >> 31)
u  = (z >> 11) * 2^-53
```

The first cumulative probability strictly greater than `u` wins; the last
candidate is a rounding fallback. Sampling returns a token and preview state,
but the state advances only in the token transaction. Planning, rejection,
retry, cancellation, deadline expiry, or failed output reservation consumes no
random value. A separately structured Python implementation freezes golden RNG
and categorical vectors. Identical request, seed, artifact, and backend must
produce identical tokens under every batch width, chunk size, trace setting,
and eligible completion permutation.

### Error taxonomy

The scheduler exposes stable categories compatible with the cross-system
error model: `invalid_request`, `unsupported`, `resource_exhausted`,
`cancelled`, `deadline_exceeded`, and `internal`. Adapter errors retain their
stable classification while potentially sensitive raw source values are
discarded at the public scheduler boundary. Invalid lengths, tokens,
sampling floats, limits, count products, byte products, deadlines, and
configuration are rejected before request payload allocation or schedule
mutation.

### Logical memory ledger

Pre-measurement amendment (2026-08-04): actor integration review found that
the original formulas charged only command metadata and one global control
wake while omitting the bounded offered-prompt copies and accepted-control /
deadline table. Before any M5 capture, the formulas were corrected to charge
the maximum offered prompt in every command slot, one actor-control slot per
maximum outstanding request, and the separate global wake/lifecycle slot. The
shared offered prompt and newly admitted request prompt intentionally overlap
until the command response is published. No earlier number is retained as M5
evidence.

The scheduler ledger is exact for declared semantic payload capacities and
fixed logical metadata charges. It is not an allocator or RSS estimator. Every
scheduler-owned payload is charged before allocation, every owner releases
once, and every count is independently capped. All charges round upward to
64 bytes. Pending-token buffers, output storage, worker workspaces, batch task
and contribution arrays, command slots, and trace storage are allocated or
logically partitioned at bounded construction/admission points rather than
grown per token.

Every owned `Vec`, page, queue, and workspace uses fallible reservation
(`try_reserve`/`try_reserve_exact` or an equivalent fallible builder) before it
becomes visible. A reservation error maps to `resource_exhausted`, rolls back
the logical charge, and leaves queue order, state, RNG, and high-water counters
unchanged. No production request path relies on the global allocator's
abort-on-OOM behavior for an expected capacity decision. Tests inject failure
after logical reservation at each constructor and request promotion, and also
exercise an impossible host reservation. The flat coalescing array and expert
workspaces are built fallibly once and reused without growth.

Per-request identity:

```text
request_used = prompt_storage + request_record + request_slot
             + active_state + pending_transaction_capacity
             + output_queue_capacity + terminal_slot
```

Shared identity:

```text
shared_used = ordinary_command_capacity + actor_control_capacity
            + worker_scratch + sampling_scratch
            + coalesced_batch_capacity
            + model_resident_partition + page_pool_partition
            + trace_capacity + admission_reserve
total_used  = sum(request_used) + shared_used
```

The production actor statically charges every ordinary command slot for
`round_up_64(64 + max_prompt_tokens * 4)` bytes from construction through
shutdown. A submitter claims a vacant slot before fallibly copying the offered
prompt, and a ready command keeps that shared capacity through engine admission
and response publication. Successful engine admission separately acquires the
request-owned prompt charge before copying into the engine record. The shared
offered-prompt charge and request prompt charge therefore intentionally overlap
until the command payload is destroyed after the admission response; this is
not a zero-delta owner relabel. A rejected or abandoned command destroys its
offered payload exactly once while the statically reserved command partition
remains unchanged.

The owner/lifetime transitions are closed:

| Charge | Owner and lifetime |
| --- | --- |
| ordinary command capacity | shared static partition from actor construction through shutdown; each slot includes its fixed metadata and maximum offered prompt payload, while occupied/responded states obey the count cap |
| actor control capacity | shared static partition through shutdown: 64 bytes of global actor lifecycle/wake metadata plus 64 bytes for each bounded accepted-control/deadline slot; never consumed by ordinary submissions |
| prompt storage | request, from successful submission reservation through terminal cleanup; the original allocation remains charged after tokens are consumed |
| request record and request slot | request, from successful submission through result reap/drop; a slot changes queued/ready/active status without a second charge |
| active state and pending transaction capacity | request, reserved before promotion and retained until numerical state is destroyed; the pending buffers change only an in-use count per token |
| output queue capacity and terminal slot | request, reserved before admission; committed events occupy bounded slots until drain/discard, and capacity releases at result reap/drop |
| worker scratch | shared static partition through scheduler shutdown; worker ownership changes do not remove its charge |
| sampling scratch | shared static partition through scheduler shutdown; logits and sampling workspace occupancy change without changing its reserved capacity |
| coalesced batch capacity | shared static partition through shutdown; includes 64 bytes per task envelope, a separate fixed 128-byte full scatter-authorization slot, and adapter task/contribution buffers; charged once, never per participating request |
| trace capacity | shared static partition through shutdown at 128 bytes per configured paired index: one independent 64-byte service-event slot and one independent 64-byte ledger-event slot; borrowed cursor reads leave both append-only occupancies and the combined capacity charge unchanged, and successful shutdown discards both arrays |
| model resident partition | validated static partition present when the scheduler is constructed |
| page-pool partition | exact configured M2 cache payload capacity when a cache is attached, otherwise zero |
| admission reserve | shared semantic headroom through scheduler lifetime; typed construction requires the maximum requested Vec-payload peak across direct batch preparation phases, while allocator overhead and RSS remain separate observations |

Fixed logical metadata charges are 64 bytes per command, actor-control,
request, output-event, terminal, expert-task envelope, state-page,
service-event, and ledger-event slot; each full task scatter authorization is
128 bytes, each paired trace index remains 128 bytes, and each retained request
record is 512 bytes. Each actor command also
includes the configured maximum offered-prompt payload before its per-slot
charge is rounded. Actor control adds one 64-byte accepted-control/deadline slot
per maximum outstanding request to a separate 64-byte global wake/lifecycle
charge. Payload charges use checked buffer capacities: four bytes per prompt
token, four bytes per f32, two bytes per BF16, and adapter-reported
state/pending/workspace layouts.

The model and optional M2 cache can pre-exist the scheduler, so their static
partitions are not described as scheduler reserve-before-allocation. Scheduler
construction instead requires `model_resident_partition` to equal the
adapter's authenticated artifact-payload plus declared owned-buffer charge and,
when attached, requires `page_pool_partition` to equal the cache's configured
payload capacity from its invariant snapshot. A caller-supplied estimate is
rejected. The ledger begins with these partitions charged before it accepts a
request.

Allocator control blocks, Tokio internals, executable text, and system page
cache are outside this logical ledger. Evidence therefore reports exact peak
ledger use and fresh-child Linux `VmHWM` separately and never relabels either
one as the other.

Implementation-wide configuration ceilings are 64 workers, 65,536 ordinary
command slots, 65,536 total outstanding request slots, 4,096 active requests,
65,536 retained terminal results, 8 sequences per batch, 4 waves per step,
65,536 output events per request, 1,048,576 paired trace indices, 65,536 tokens
per state page, 262,144 expert tasks per wave, and 1 TiB of logical ledger
capacity. Adapter context and vocabulary limits remain independently enforced.
Zero, exact-ceiling, ceiling-plus-one, multiplication overflow, and host-`usize`
conversion tests gate every field; a valid configured value can still fail the
minimum-operation fit check.

The frozen evidence configuration has an 8 MiB scheduler-accounted ceiling,
a 256 MiB address-space ceiling, one worker, ordinary command capacity 32,
normal total-outstanding/active/retained caps 32/16/32, and pressure-cell caps
16/16/16. It uses batch width 8, four waves per step, 16-token state pages,
64 output events per normal request, and 8,192 paired trace indices, providing
independent capacities of 8,192 service events and 8,192 ledger events.
Configuration
sets `model_resident_partition = 5,632` bytes (the 5,600-byte v3 semantic tensor
payload rounded once to 64), `page_pool_partition = 0`, and
`admission_reserve = 1,048,576` bytes. The eager authenticated `Artifact` and
encoded object are dropped after model construction and before release; only
the model's declared tensor-buffer payload remains in the logical partition.
Fresh-child `VmHWM` still records any larger initialization peak. Configuration
fails if one minimum executable request plus all fixed shared partitions cannot
fit.

#### Bounded ledger evidence clarification (2026-08-06)

This clarification was fixed before the ledger-recorder implementation and any
M5 capture. It does not change the logical-memory formula, evidence
configuration, workload, or claim rule. The existing `trace_capacity * 128`
shared charge is a paired static allocation: one pre-reserved service-event
array and one pre-reserved ledger-event array each have exactly
`trace_capacity` elements and a 64-byte logical payload charge per element.
Their cursors, retained lengths, and sticky overflow states are independent;
unused capacity in one array cannot be transferred to the other. Both concrete
event types must satisfy `size_of(event) <= 64`.

Every ledger row starts from the immutable `LedgerSnapshot` taken immediately
after successful engine construction and static shared acquisition, before any
request admission. That snapshot is sequence zero. The stable category IDs are
the following closed mapping:

```text
0  prompt_storage       1  request_record
2  request_slot         3  active_state
4  pending_transaction  5  output
6  terminal             7  worker_scratch
8  sampling_scratch     9  coalesced_batch
10 model_resident       11 page_pool
12 trace                13 admission_reserve
14 actor_command        15 actor_control
```

Owner ID zero is the shared scheduler owner. A nonzero owner ID is exactly the
opaque engine-local `RequestId::get()` value and is valid only for a
request-owned category. Each retained delta is a nonzero `i64` multiple of 64
bytes. Durable post-construction mutations receive contiguous sequence numbers
starting at one. Every `(owner_id, category_id, signed_delta)` belonging to one
atomic ledger mutation repeats that sequence number and is stored in canonical
`(owner_id, category_id)` order. Replay applies a complete sequence as one
unit. If every delta in the next mutation does not fit, the recorder appends
none of it, marks overflow sticky, and never changes scheduling or accounting.
Sequence exhaustion has the same evidence-only overflow result.

Only durable request-owner byte changes enter the stream. Successful single
or batch publication and active promotion append positive deltas; terminal
resource release and retained-result reap append negative deltas. Atomic batch
publication uses one sequence across every accepted owner. Fit projections,
reservation splits, owner transfers, prepare-only permits, provisional
attempts, dropped permits, allocation rollbacks, and per-token buffer occupancy
append nothing. A provisional charge linearizes in evidence only when it
becomes durable; an exact rollback restores usage and peaks and remains absent
from the stream. This makes independent replay reproduce both current and peak
category, request, shared, and aggregate totals.

The observable ledger interval ends only after request-owned usage has returned
to zero and immediately before successful scheduler teardown. Static shared
bytes exist in sequence zero and do not change during that interval. Successful
shutdown destroys both trace arrays before applying the final static shared
release, so that all-negative teardown is deliberately outside `ledger.jsonl`;
the pre-shutdown replay must equal the live shared-only snapshot, and the
`ShutdownReport` plus post-shutdown ledger snapshot separately prove shared and
aggregate zero. Excluding a final all-negative transition cannot change a
peak. Any fallible shutdown path precedes trace destruction and retains both
readable prefixes.

Service and ledger completeness are separate gates. Every measured M5 run
requires both 8,192-capacity streams to remain healthy. The standalone
`continuous-arrival-1000` correctness check retains its frozen capacity of
1,024 and requires only its exactly 1,000-event service stream to be healthy;
its more than 14,000 category deltas intentionally overflow the independent
ledger recorder. That overflow must be sticky and behavior-neutral and does
not create a measured ledger row.

Implementation mapping (2026-08-06): `CapacityLedger` owns the preallocated
recorder and emits through request-bound acquisition/release permits;
`SchedulerEngine::ledger_trace_since` exposes only borrowed complete-mutation
suffixes and rejects an ordinal that bisects a mutation. Constructor and
shutdown paths destroy physical payloads before rolling back or releasing
their logical owners. Public integration tests independently replay category,
request-owner, shared, and aggregate current/peak totals; exercise atomic batch
ordering, rollback neutrality, exact-full/overflow stream independence, foreign
cursors, and shutdown; the frozen continuous-arrival test separately proves
its expected ledger overflow alongside a healthy service stream.

### Fairness contract

One committed model position is one equal-cost service quantum. In the
homogeneous fairness cell, 16 requests are continuously runnable through the
fixed horizon `H = 16`, so each is entitled to one quantum. The horizon ends
before even the FIFO baseline can finish its first request.

For committed prefix `k` and request `i`:

```text
q_i(k)       = committed service quanta for i
lag_num_i(k) = max(0, k - 16 * q_i(k))
lag_i(k)     = lag_num_i(k) / 16
x_i          = q_i(16)
Jain         = (sum(x_i))^2 / (16 * sum(x_i^2))
```

The hard candidate gate is a maximum runnable gap of at most 15 other commits,
`lag_num < 16` across every prefix using integer arithmetic, exactly 16
eligible service events, no blocking before the horizon, and nonzero service
for every request. Jain's index is a mandatory diagnostic, not the starvation
proof. Runnable gap includes commits before a request's first service, between
successive services, and after its last service through the horizon; only
commits by other requests count. Lag, gap, Jain, and the baseline difference
are exact replayed properties, not bootstrap estimates. All 30 timing
repetitions must have identical eligible service-sequence digests.

The `continuous-arrival-1000` check is a deterministic candidate-core
correctness test, not a timing cell. It uses the evidence configuration with
these exact overrides: one worker, batch width one, one wave per step, maximum
active requests 16, outstanding/queued/retained-terminal caps 64/64/64, output
capacity two, and trace capacity 1,024. Before turn zero it admits anchor
offered index zero as `P(16)` with eight output tokens, followed by churner
indices 1 through 15 as `P(1)` with one output token each. All requests are
greedy and have no deadline.

For each zero-based turn `t` from zero through 999, the harness first offers
churner index `16 + t`, again as `P(1)` with one output token, and requires
acceptance. It then calls exactly one bounded scheduler step and requires that
the step commit exactly one model position and append exactly one service
event. Before the next turn, it visits retained requests in ascending request
ID, acknowledges any terminal result, nonblocking-drains every available output
event, and acknowledges EOF in the same pass. The second direct-engine
acknowledgement synchronously reaps a terminal record; the following turn's
bounded step observes that freed slot before committing its sole service event.
If the anchor is terminal, the same ordered pass records and reaps it. After
the final turn, cleanup performs the corresponding acknowledgements before
shutdown. No sink operation may execute model work, and lifecycle reaps are not
service turns.

The valid trace has exactly 1,000 service events. Anchor positions zero through
22 each appear exactly once, its first service occurs no later than turn 15,
at most 15 other commits separate successive anchor services while it remains
runnable, and it reaches `Completed` no later than turn 367. No accepted
runnable request may report output or memory blocking in this test. After turn
999, the harness requests cancellation of every outstanding request in
ascending request ID, takes at most 64 further bounded cleanup steps, drains
and reaps in ascending request ID, and shuts down. Cleanup service is excluded
from the 1,000-event trace and must not commit another position; final request
and shared ledger use are zero.

### Preregistered evidence workloads

The experiment compares:

- baseline `fifo-single-request-run-to-completion-v1`; and
- candidate `deficit-continuous-expert-coalesce-v1`.

Both force the tiny-v3 scalar BF16 backend, one pinned compute CPU, identical
state/workspace/output configurations, and identical accepted request/seed
schedules. Before timing, `prepare_submit_batch` validates every offered
description, performs all fallible allocations, and holds provisional charges
in an unpublished `PreparedAdmission`; it assigns no request ID and cannot be
scheduled. Preparation latency is recorded separately. The harness then records
`release_ns` and calls allocation-free `commit_prepared_batch` under the engine
lock. That single boundary converts provisional charges, assigns accepted IDs,
publishes the accepted set in offered-index order, and records
`admitted_ns = release_ns` for every accepted request before promotion or model
work. Rejected offered indices are published at the same boundary. Thus the
reported TTFT is the repository-wide admission-to-first-token metric without
sequential staging delay. This evidence excludes pre-admission validation and
Tokio command-delivery latency; both are recorded/tested separately and M6
measures the complete surface. `P(n)` is the direct-token pattern defined above.
Normal requests are greedy.

The accepted set is a strict FIFO prefix. Complete-slice intrinsic validation
precedes pressure selection, and the first pressure error describes the whole
suffix; smaller later offers cannot bypass it. Commit rechecks absolute and
release-relative deadlines for every offer, including that suffix, before its
mutation boundary. Prepared request and endpoint payloads drop before an armed
aggregate ledger permit rolls back. The compact result stores the first
accepted ID, accepted count, and one suffix error, then synthesizes exact-size
iterators without an escaping heap allocation.

Typed engine construction computes the maximum semantic Vec-payload footprint
across payload preparation, control-permit construction, and endpoint-permit
construction. It rejects a raw `admission_reserve_bytes` value below that
checked peak. The formula includes both `A::StateLayout` copies and all
coexisting control/endpoint transaction buffers, while request-owned prompt and
output buffers remain under the provisional request reservation. As elsewhere
in the ledger, this is requested semantic capacity; allocator metadata,
possible `try_reserve_exact` over-allocation, and RSS are not inferred.

| Cell | Exact workload | Role |
| --- | --- | --- |
| `single-long-control` | 1 request: `P(896)`, 8 output tokens | descriptive long-prefill overhead and parity |
| `homogeneous-burst-16` | 16 requests: `P(16)`, 8 output tokens; horizon 16 | primary end-to-end output throughput and exact service lag |
| `mixed-prefill-burst-16` | prompt lengths by request ID: `896,16,512,64, 64,896,16,512, 512,64,896,16, 16,512,64,896`; 8 output tokens each | primary p95 TTFT and prefill-completion rate |
| `deadline-pressure-24` | 24 requests: `P(16)`, 8 output tokens; total-outstanding cap 16; deadline 20,000,000 ns after release | goodput, deadline, exact accept/reject set |
| `cancellation-pressure-12` | 12 requests: `P(896)`, 8 output tokens; cancel IDs 2, 5, 8, 11 at frozen queued, position-0 post-final-snapshot/pre-apply, position-8 post-router/pre-expert, and position-895 post-scatter/permit/preapply hooks | transactional cleanup and ownership |

The batch call prevents promotion while all 24 offers linearize, so offered
indices 0 through 15 are accepted and indices 16 through 23 must fail
`resource_exhausted`; rejected offers receive no internal request ID.
The deadline duration is a fixed workload parameter, not a success threshold.
Its real-time expiry boundary and terminal order are observed results and are
excluded from cross-repetition trace-digest equality; manual-clock deadline
tests retain deterministic reference digests.

The evidence sink does not drain normal, mixed, deadline, or long-control
output queues before all admitted requests reach `terminal_decided`; capacity
64 exceeds each request's eight committed outputs, so this policy cannot block
model work. It then drains and reaps requests in ascending accepted-ID order.
Cancellation hooks drain only after their terminal decision. Primary throughput
ends at the last emitted-token commit, and cleanup/quiescence intervals are
reported separately, so sink drain/reap work cannot manipulate that endpoint.

Cancellation hooks are atomic deterministic harness barriers, never sleeps.
Accepted request ID 2 cancels before any promotion. ID 5 cancels after the
final clock/control snapshot for prompt position 0 but before infallible apply;
the signal therefore loses to exactly that commit. ID 8 cancels before expert
execution for prompt position 8. ID 11 cancels after prompt position 895 has a
validated composite permit but before the final control check/apply, so that
position, its RNG preview, and its first output must not commit. The
post-router cancellation retains shared batch scratch until worker return
while releasing request ownership exactly once.

### KPI hierarchy and definitions

Only three outcomes are primary:

1. `homogeneous-burst-16` end-to-end emitted-token-throughput
   candidate/baseline ratio, higher is better;
2. `mixed-prefill-burst-16` within-run request p95 TTFT
   candidate/baseline ratio, lower is better; and
3. `homogeneous-burst-16` exact maximum-service-lag
   candidate-minus-baseline difference plus the candidate's hard starvation
   gate, lower is better.

Drivers are p50/p95 inter-token latency, p50/p95 queue delay, the
single-long-control prefill and decode throughputs,
mixed-burst prefill-completion rate, mean/p95 batch occupancy, logical expert
contributions, expert-group calls, coalescing factor, preemption count, maximum
runnable gap, timed application-read bytes per committed position, and
state-page utilization. Guardrails are exact token/route/stop parity,
tolerance-bound logits and route weights, deterministic sampling/trace
digests, balanced ledgers, cancellation cleanup, backpressure identity,
deadline handling, the fairness bounds, CPU-affinity eligibility, the logical
ledger ceiling, the address-space limit, and separate `VmHWM` observation.

For request `i`:

```text
TTFT_i        = first emitted-token commit - admitted_ns
queue_delay_i = first model work start - admitted_ns
ITL_i,j       = emit_i,j - emit_i,j-1, for j >= 2
```

The single-long-control prefill throughput is its 896 committed prompt
positions divided by final-prompt commit minus first-prefill start. Its decode
throughput is the subsequent seven committed decode-input positions divided by
last-decode commit minus first-decode start; the first emission produced by the
final prompt position is excluded. No other work can contaminate either
single-request phase. The mixed-burst prefill-completion rate is explicitly an
end-to-end scheduler diagnostic: completed prompt positions divided by last
prefill commit minus release, including any intervening short-request decode
work. It is not used for a kernel or phase-isolated speed claim.

Homogeneous end-to-end output throughput is all successfully emitted tokens
divided by `last_emitted_commit - release_ns`. Raw ITL intervals form the
percentile population. Timed application-read bytes/token is zero for the
preregistered eager v3 model; initialization reads and their interval are
recorded separately. That zero is not an out-of-core or storage result.

Nearest-rank percentiles use rank `ceil(p * n)`. A run's p95 is one
experimental-unit value; requests inside a run are not treated as independent
replicates. With 16 mixed requests, nearest-rank p95 is the maximum observed
request TTFT and is disclosed as such. Deadline goodput is successful admitted
requests with `terminal_decided <= deadline` divided by
`last_terminal_decided - release_ns`; offered, admitted, and rejected counts
are reported separately. Peak accounted bytes are replayed from the complete
ledger event stream. Peak RSS is fresh-child Linux `VmHWM`; sampled `VmRSS` is
a separate diagnostic.

### Repetitions and analysis

The experiment ID is `m5-scheduler-20260803`; its only capture command is:

```text
python3 scripts/run_m5_experiment.py capture \
  --output benchmarks/raw/m5-scheduler-20260803
```

Each cell has five excluded warmup pairs in fixed
baseline/candidate, candidate/baseline, baseline/candidate,
candidate/baseline, baseline/candidate order, followed by 30 measured pairs.
Measured order contains fifteen baseline/candidate pairs followed by fifteen
candidate/baseline pairs before shuffling.

Cell order uses SHA-256 counter stream
`"runnel-m5-cell-order-v1\0" || u64_le(counter)`. Pair order uses
`"runnel-m5-pair-order-v1\0" || u16_le(case_id_length) || case_id ||
u64_le(counter)`. Counters begin at zero and each digest is consumed as four
little-endian u64 words. Fisher-Yates visits indices `length - 1` through 1;
for bound `b = index + 1`, a word is accepted only below
`floor(2^64 / b) * b`, then `word mod b` selects the swap index. Warmups are
not shuffled. No outlier is removed or replaced. Every timing interval requires
all 30 complete pairs.

Summaries report count, mean, sample standard deviation with denominator
`n - 1`, minimum, ordinary median (mean of the two central values for even
`n`), nearest-rank p50/p95, and maximum. Positive timing metrics use paired
ratios only when both values are finite and the denominator is positive.

Each timing metric's deterministic 10,000-resample paired-median stream is
`SHA-256("runnel-m5-bootstrap-v1\0" || u16_le(case_id_length) || case_id ||
u16_le(metric_id_length) || metric_id || u64_le(counter))`, consumed four
little-endian u64 words per digest. Index draws reject at or above
`floor(2^64 / 30) * 30`, then take remainder 30. Each replicate resamples the
30 paired ratios with replacement and takes the ordinary even-sample median.
Sorted zero-based endpoints 249 and 9,749 form the unadjusted exploratory 95%
percentile interval. Cells are never pooled. Service lag, runnable gap, and
Jain are exact schedule properties with no bootstrap interval.

The only bootstrap `metric_id` values are
`homogeneous-output-throughput-ratio`, `mixed-p95-ttft-ratio`,
`single-prefill-throughput-ratio`, `single-decode-throughput-ratio`,
`mixed-prefill-completion-rate-ratio`, `itl-p50-ratio`, `itl-p95-ratio`,
`queue-delay-p50-ratio`, `queue-delay-p95-ratio`, and
`deadline-goodput-difference`. Ratio IDs use the procedure above. The goodput
ID instead forms 30 paired values `candidate_goodput - baseline_goodput`, uses
the same paired resample indices and ordinary median, and applies the same
sorted endpoints. No other metric receives an inferential interval.

Request sampling seeds, used by correctness cells even though timed requests
are greedy, are the first little-endian u64 of
`SHA-256("runnel-m5-request-seed-v1\0" || u16_le(case_id_length) || case_id ||
u32_le(offered_request_index))`.

The closed evidence set is `experiment.json`, `environment.json`,
`capture.json`, five rows in `cases.jsonl`, 26 fixed check IDs in
`correctness.jsonl`, 50 rows in `warmups.jsonl`, 300 rows in `runs.jsonl`,
4,140 rows in `requests.jsonl`, 300 rows in `service.jsonl`, 300 rows in
`ledger.jsonl`, `summary.json`, and summary-sourced
`figures/m5-throughput.svg`, `figures/m5-latency.svg`, and
`figures/m5-fairness-memory.svg`. Measured-run
primary keys are `(case_id, pair_index, variant)`; request keys append the
fixed offered index. Warmup keys use `(case_id, warmup_index, variant)`.

The 26 correctness IDs are closed before implementation timing:

```text
fixture-custody-v1             fixture-custody-v2
fixture-custody-v3             oracle-scalar-v1
oracle-scalar-v2               oracle-scalar-v3
oracle-avx2-v2                 oracle-avx2-v3
streaming-materialized-short   streaming-materialized-long
chunk-partitions-short         chunk-partitions-long
direct-fifo-parity             direct-continuous-parity
splitmix-python                categorical-python
sampling-schedule-permutation  contribution-identity
scatter-permutation            cancellation-checkpoint-matrix
ledger-boundary-overflow       backpressure-control-saturation
deadline-manual-clock          drr-reference-fairness
continuous-arrival-1000        actor-concurrency-stress
```

`correctness.jsonl` is canonical UTF-8 JSONL with one LF-terminated row in
exactly that order. Its closed top-level schema is:

```text
schema                   "runnel.m5-correctness/1"
check_index              integer 0..25
check_id                 the identifier at that index above
kind                     closed proof-kind identifier
backend                  null | scalar | avx2 | python-pytorch
status                   ok | failed | unsupported
failure_code             null or one closed failure identifier
proof                    one closed kind-specific object
```

Unknown or duplicate keys, duplicate IDs, a missing or reordered row,
noncanonical JSON, a nonfinite number, or a value of the wrong exact JSON type
rejects the file. The interchange schema retains `unsupported`, but every row
must be `ok` for the accepted capture; the recorded AVX2 host therefore cannot
accept either AVX2 row as unsupported. A separate Python verifier validates
the schema and reconstructs hashes, counts, comparisons, and digests rather
than trusting `status`. The producer invokes reusable typed checks directly;
human `cargo test` output is not a proof payload.

The proof objects retain raw counts and error maxima, not boolean-only
attestations. Their closed requirements are:

| Check group | Required proof material |
| --- | --- |
| fixture custody | fixture ID/version/representation/context, spec/object/page-table/golden file sizes and SHA-256 values, tensor payload bytes, rounded resident charge |
| scalar/AVX2 oracle | requested/selected backend, fixture/oracle hashes, positions, token/route/stop digests, logit/router-score/route-weight absolute and relative maxima, tolerances, mismatch counts |
| streaming/materialized | exact positions and page-boundary suite, input/reference/actual digests, Rust-f64 and independent-PyTorch comparisons, error maxima and tolerance ratios, state-capacity and pointer-stability witnesses |
| chunk partitions | frozen partition suite, suite digest and count, token-at-a-time reference digest, per-partition state/token/route/stop digests, mismatch count |
| direct policy parity | all five workload IDs, direct-oracle and scheduler digests, offered/accepted/rejected/terminal and committed-position counts, final request ownership zero |
| RNG and categorical | committed fixture/oracle hashes, exact case counts and state/word/unit/token/order mismatches, ULP diagnostics, exact invalid classifications |
| sampling schedule | batch/chunk/completion/preemption permutations, output/RNG/service digests, failed-preview RNG neutrality, preemption/resume counts |
| contribution/scatter | closed malformed/permutation/lane-reuse cases, rejection stage, canonical reduction digest, zero mutation-before-reject and stale-overwrite counts |
| cancellation matrix | four boundaries by three actions, fire ordinals, committed prefixes, state/RNG/output/service digests, skipped expert calls, request-zero and worker-quiescent witnesses |
| ledger boundary | category-map digest, 64-byte quantum, fit/one-short/rollback/batch/full/overflow cases, replayed current/peak snapshots, shutdown zero |
| backpressure/control | command/output saturation cases, sibling progress, cancel/drop/drain/shutdown dispositions, zero busy-spin, output identity, ownership zero |
| deadline clock | every lifecycle state, inclusive and cancellation-precedence cases, service-prefix digest, terminal results, ownership zero |
| DRR fairness | exact service digest, horizon 16, integer lag numerator, maximum gap, Jain numerator/denominator, nonzero service for all 16 requests |
| continuous arrival | exact overrides, 1,000 healthy service events, expected independent ledger overflow, anchor positions/turn bounds/gap, zero cleanup commits |
| actor concurrency | action/golden/custody hashes, accepted semantic digest, exactly 32 independently verified fresh race repetitions and overlap witnesses, pump cap, healthy 675-record observer, final ledger zero |

The correctness producer hashes its executable, source closure, fixtures,
goldens, Python oracle, commit, build command, and controlled environment
before and after capture. Any failed correctness row prevents measured child
execution; the harness retains the failure and synthesizes the closed missing
timing grid without retrying or replacing a variant.

`runs` and `requests` are rectangular. Each `service` row contains the complete
valid prefix as compact arrays of request index and phase bit, rather than one
JSON object per token. The phase mapping is frozen as `0=prefill` and
`1=decode`; a zero-based position is prefill exactly when it is less than the
prompt length, so the final prompt position remains prefill when it publishes
the first output. The direct engine exposes this append-only prefix through a
borrowed cursor view. Exact capacity is healthy until another successful commit
sets sticky overflow; reads never drain or reset it. Each `ledger` row follows
the bounded ledger-evidence contract above: it contains sequence-zero category
totals and compact arrays of
`(sequence, category_id, owner_id, signed_delta)`; static
preallocated token buffers avoid per-token byte transitions. Parent and child
flush framed canonical records after every request or bounded event chunk. A
crash, timeout, signal, malformed output, migration, invariant failure, or cap
violation retains the longest valid prefix and fills only missing rectangular
keys with null measurements and a closed failure code. Variable service/ledger
rows retain their authenticated prefix and failure status. A surviving variant
is never paired with a replacement run.

The serializer must prove a test-derived worst-case bound of at most 24 MiB for
this exact grid; capture must reject a larger staged set, and the publication
directory cap remains 32 MiB. The verifier independently reconstructs that
bound and every compact event stream before accepting summary or figure bytes.

### Correctness and evidence gates

M5 does not close until all of these pass:

- v1/v2/v3 scalar and eligible AVX2 route/token/logit oracle gates;
- paged streaming attention against materialized f64 and independent PyTorch
  references at partial tails and many page boundaries;
- every small prompt chunk partition and fixed long partitions against
  token-at-a-time execution;
- direct, FIFO, and continuous per-request greedy parity;
- Rust SplitMix64/sampling vectors against a separately implemented Python
  oracle and deterministic output under different batch/chunk settings;
- missing/duplicate/foreign/stale contribution and router-rank scatter tests;
- cancellation/failure injection before and after every token transaction
  phase, including stale slot-generation/ABA completion;
- exact fit, one-byte-short, overflow, queue/output saturation, repeated
  rejection, shutdown, and complete ledger-release tests;
- manual-clock deadline tests at queued, active, expert-owned,
  ready-to-commit, output-blocked, and preempted states;
- deterministic scheduler trace against an independent reference, the hard
  service-lag/gap bound, and continuous-arrival starvation stress;
- bounded concurrent actor submit/cancel/disconnect stress; and
- committed raw evidence whose verifier recomputes schema, hashes, arithmetic,
  percentiles, fairness, accounting, parity, and capture provenance.

Pre-measurement protocol amendment (2026-08-04): the actor workload, producer
assignment, cleanup, witnesses, and semantic transcript below were made
explicit before generating a golden digest or running any M5 measurement or
capture. Neither this actor-stress protocol nor the `continuous-arrival-1000`
protocol had been run, no semantic golden digest had been generated, and no M5
measurement, capture, or timing result existed when this amendment was
accepted. Earlier unit and integration correctness results are not relabeled as
this protocol's evidence. The amendment changes no model or scheduler policy
and uses no implementation, fixture, prose, or result from the credited
prior-art repository.

Pre-measurement protocol correction (2026-08-06): the
`cancellation-pressure-12` ID 5 hook was changed from the earlier phrase
"immediately after committing prompt position 0" to the exact
post-final-snapshot/pre-apply boundary. The earlier phrase did not distinguish
the transaction's commit linearization point from its subsequent infallible
physical apply. The corrected hook publishes cancellation after the final
clock/control snapshot, so it loses to exactly position 0, while still making
the signal and cleanup boundary deterministic. This moves the measured
`cancel_linearized` timestamp to before physical apply rather than after it;
the committed service/output prefix is unchanged. No M5 timing run, capture,
raw evidence row, cancellation golden digest, or accepted M5 result existed
when this correction was adopted. Earlier correctness runs are not timing
evidence and are not relabeled as a preregistered capture. The table and hook
description above contain the corrected protocol; published Git history
retains the earlier wording.

Golden custody acceptance (2026-08-04): after three byte-identical local
captures, independent Rust/Python byte agreement, PyTorch prefix validation,
two independent reviews, and green hosted CI run
[`30951994706`](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30951994706)
for pre-freeze commit `9dfee37`, the v2 semantic transcript was accepted at
36,561 bytes with digest
`sha256:2711f6b6b28849dd9cb9692f75d97f09d24520645d7b0ccaf7c4c2fd023ddd7a`.
The canonical historical capture is `actor-golden-v1.json`; its JSON file
SHA-256 is `d070c5158ae6ba3ae36553604fb98b231482db5333df545d75572716c2f67705`.
The semantic digest excludes physical diagnostics and recorder append order.
Future live captures must satisfy every diagnostic bound and semantic check,
but their JSON bytes and physical counts need not equal the historical capture.
No binary transcript is committed.

#### Actor request and action streams

Actor validation has a deterministic interleaving test and a separate genuine
race stress. Both run the tiny-v3 model on the scalar backend through two Tokio
producer tasks and one scheduler actor. The exact limits are one compute worker,
ordinary-command capacity eight, outstanding/active/queued/retained-terminal
caps 16/8/16/16, prompt/generation/context ceilings 4/16/19, four-token state
pages, two output events per request, batch width eight, four waves per step,
1,024 paired trace indices, an 8,388,608-byte logical-memory limit, no page-pool
partition, and a 1,048,576-byte admission reserve. The global context envelope
is 19 because configuration validates `4 + 16 - 1`; every derived request below
still has at most 16 model positions.

Before actor construction, the harness authenticates the canonical
`fixtures/tiny-v3/spec.json` bytes as
`sha256:ed57d7961e65c76223c169cabebaff9c02d8293da026abb0c0c0a22d38079845`.
The generated artifact must then validate as artifact ID
`sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c`,
one 5,600-byte object with digest
`sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab`,
and a 96-byte page table with digest
`sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c`.
The scalar backend is mandatory. A model/spec identity mismatch invalidates
the run; it is never an alternate golden outcome.

The committed custody document uses closed schema
`runnel.actor-stress-vectors/2`. Its pre-result `/1` predecessor froze the same
descriptor and action sequences but did not bind these model inputs, so it is
superseded and is not admissible as golden evidence.

The 64 offered descriptions are derived independently of the action stream.
For zero-based request index `i`, let `d` be the 32 bytes of:

```text
SHA-256("runnel-m5-actor-request-v1\0" || u32_le(i))
```

Bytes are numbered from zero in digest byte order. The descriptor is:

```text
prompt_len      = 1 + d[0] mod 4
prompt[j]       = 1 + d[1 + j] mod 31, for 0 <= j < prompt_len
max_new_tokens  = 1 + d[5] mod (17 - prompt_len)
sampling        = Greedy
deadline        = None
```

Thus `prompt_len + max_new_tokens - 1` is at most 16. No tokenizer or model
output is fed back into descriptor generation.

The action word stream concatenates, in increasing counter order, the four
little-endian `u64` values represented by digest byte ranges 0..8, 8..16,
16..24, and 24..32 of:

```text
SHA-256("runnel-m5-actor-stress-v1\0" || u64_le(counter))
```

Counters start at zero. To draw uniformly below bound `b`, words are consumed
until one is below `floor(2^64 / b) * b`, and its remainder modulo `b` is used.
Exactly 1,024 accepted bound-five draws define action kinds
`0=submit`, `1=cancel`, `2=receiver drop`, `3=drain one event`, and
`4=actor wake`. Kinds one through four consume a subsequent unbiased bound-64
draw. A wake consumes and records that request selector to preserve stream
custody, but the selector is ignored by the global wake operation.

Every submit-kind action has its own zero-based submit-attempt ordinal. Attempts
zero through 63 map one-to-one to the same request index; the cursor advances
even if actor submission or engine admission rejects that offer. Attempts 64
and later are `offer_exhausted` no-ops with no request index and do not call the
actor. All submit-kind actions are assigned to producer
`submit_attempt_ordinal mod 2`. Request index `i` has home producer `i mod 2`:
drain and receiver-drop actions run on that producer, cancellation runs on the
opposite producer, and wake action ordinal `a` runs on producer `a mod 2`.

The fault-free request-ID contract starts at one. Each successful engine
admission consumes exactly the next consecutive ID in owner-FIFO order; a
rejected offer consumes no ID. The golden's serialized IDs must therefore be
`1..N` in its successful-admission order. For every fresh actor, the witnessed
`ready_commit_sequence` starts at one and is unique and contiguous across all
64 in-range offers. It orders transitions into the owner's ready FIFO; it is
structural FIFO evidence, not an API or engine-admission linearization point.
The 142 exhausted offers do not call the actor and have zero command witnesses.
A race history sorts accepted submissions by ready sequence and requires their
request IDs to be exactly `1..N`; it never validates IDs in producer response
order. Identity exhaustion or a gap, duplicate, or out-of-order assignment
invalidates the run rather than defining another transcript.

An accepted request stores its cancellation authority separately from its sole
receiver. Dropping the receiver does not drop that authority; it remains usable
through terminal publication. Recycling returns the control slot to its free
list but deliberately leaves the terminal generation word intact until a later
bind replaces it, so a retained authority may continue to observe
`already_terminal` after its request record has reaped; after rebind it reports
the typed stale-generation error. A targeted operation whose needed authority
or receiver is absent, not yet accepted, rejected, or already consumed is an
`unavailable` no-op and makes no actor API call. A stored authority that has
since become stale is still called and records its typed API error. A drain
action performs exactly one nonblocking receive attempt: it consumes at most
one FIFO output event, observes empty, or acknowledges EOF. A receiver-drop
action consumes the sole handle; that handle's destructor publishes disconnect
exactly once before relinquishing the receiver, and the harness does not make a
second explicit-disconnect call. Wake is global and never inspects the selected
request.

For this bounded corpus, the 16 control slots use a frozen deterministic reuse
rule. Initial binds consume slot indices `0..15` in ascending order. Recycling
appends the freed index to the free stack, and the next successful admission
binds the most recently recycled index (LIFO), incrementing its generation and
making every older authority for that slot stale. A rejected admission consumes
no slot and does not change this stack. The serialized golden independently
replays this exact rule because every action is followed by acknowledged
quiescence. A concurrent race history instead validates accepted control
identities, monotone generation changes, stale-generation protection, and
witnessed control-word chains; it does not infer recycle order from destructor
or semantic-recorder append order. Focused control-registry tests freeze exact
LIFO allocation. This deliberately narrows the race claim without weakening the
allocator contract or inventing an unwitnessed owner order. The reuse rule
affects structural capture custody but not semantic transcript bytes.

#### Golden execution, cleanup, and race witnesses

The golden actor first reaches acknowledged quiescence. For action ordinals in
increasing order, a coordinator releases exactly one action to its assigned
producer, waits for the operation response, and then waits for explicit actor
quiescence before releasing the next action. An in-range submit waits for its
admission response, including a typed rejection. Quiescence means the actor is
parked, no pump is in flight, and its dirty flag is false, observed through a
lost-wake-safe acknowledgement. A pump turn is one entry into the blocking
owner's `pump` function, including command-only and no-work entries; engine
step count is not a substitute. The cap is a delta of 4,096 pump entries from
the initial acknowledged quiescence through completed shutdown.

The accounting fields returned with acknowledged quiescence are the owner's
publication from the completed pump preceding that park. After every scripted
receiver drop, the golden additionally requires the exact live request-record
count to equal the number of receivers still owned by the two producers. Script
actions never consume a terminal result, so a still-owned receiver prevents its
generation from reaping; this equality therefore proves every script-dropped
receiver's request record has reaped. It does not infer cancellation-authority
staleness: the control registry retains a terminal generation word after
recycle until that slot is rebound.

Cleanup begins from quiescence. While the pump is held, the coordinator visits
every accepted request index in ascending order and probes its stored
cancellation authority. Because every action is followed by acknowledged
quiescence and cleanup probes each authority exactly once, a live receiver
permits only `requested` or `already_terminal`; `already_requested` is
unreachable and invalid in this protocol. A script-dropped and proven-reaped
record permits only the retained `already_terminal` tombstone or the typed
stale-generation error after slot reuse. Every probe must witness a nonzero
generation at its control-word boundary. A `requested` result from this
pump-held cleanup probe must correlate with a later `cancelled` terminal. A
scripted cancel or disconnect that returns `requested` proves the control-word
mutation, but it may race after the actor has irrevocably selected a successful
completion and therefore may end at `completed`. Conversely, every `cancelled`
terminal requires exactly one captured preterminal control-word transition from
`C=0` to `C=1`. A `requested` disposition alone is insufficient: disconnect
may return `requested` while changing only `D` when `C` was already set.
Duplicate `C=0 -> C=1` publishers are invalid, `already_requested` is only a
reader, and an `already_terminal` disconnect that adds `C` is too late. The
harness does not infer `control-cancel publish < terminal publish` if and only
if the semantic outcome is `cancelled`. It records all cleanup
dispositions in the logical cross-language capture but excludes them from the
semantic transcript bytes.

The coordinator then releases the pump and waits for quiescence. In ascending
request-index order, each remaining receiver first consumes its terminal result,
then drains the exact FIFO suffix of committed output through EOF, and is
dropped; a receiver already dropped by the script is skipped. After each such
receiver is dropped, the harness waits for acknowledged actor quiescence and
requires exactly one fewer live request record. Relative to the preceding
authoritative snapshot, that post-drop snapshot must also strictly decrease
`request_bytes` and strictly increase `pump_entries`, `park_epoch`, and
`engine_steps`; equality is not accepted as cleanup evidence. The terminal
control tombstone must remain generation-valid because no new bind occurs
during cleanup. The remaining semantic output suffix after all scripted drains
is bounded by the actor's authenticated `output_capacity_per_request` of two
for every accepted request, whether its receiver was dropped by the script or
retained for cleanup. Cleanup preallocates its collection for that capacity and
enforces the same logical limit on every push, instead of using a descriptor's
lifetime emission bound as queue capacity. The harness then drops the stored
cancellation authorities, waits for quiescence, and requires zero outstanding
requests and zero request-owned ledger bytes before cooperative actor shutdown.
Consequently shutdown itself cancels and terminalizes no request, discards no
output, and releases no request-owned bytes. Terminal and output publication
observers retain semantic records even when a receiver was
previously dropped, so discarded client delivery cannot erase a committed
event from the golden transcript. Observer-vector order is
retention order only: endpoint mutation unlocks before recorder append, so the
golden treats observations as identity-keyed effects and applies the explicit
sorting rules below rather than interpreting append order as a linearization
order. Before actor construction,
the observer pre-reserves a logical lifetime capacity of exactly 675 records:
the frozen descriptors permit at most 547 output publications, plus at most 64
terminal publications and 64 first-EOF acknowledgements. Its allocation may
reserve more storage but must never grow after construction; the retained
logical limit remains 675 and overflow or poison invalidates the run.

Only deterministic request-level outcomes allowed by the frozen actions are
serialized. A host allocation failure, authenticated-artifact failure, worker
or adapter failure, actor panic, timeout, observer overflow/poison, unexpected
closed state, or internal invariant error invalidates the complete run and is
not an action-record alternative. In particular, allocation failure is
distinguished from the expected bounded-capacity `resource_exhausted`
admission result even though both share the public error category.

The genuine race uses the identical actions and producer assignment for exactly
32 independent repetitions. Each repetition constructs a fresh authenticated
model, actor, recorder, accepted-request registry, interval counter, and
preallocated history. After acknowledged initial quiescence, both producers
wait behind one three-party Tokio barrier and then execute only their own
518/506-action subsequences in original action-ordinal order. There are no
cross-producer gates, retries, sleeps, or scheduler hints after barrier release.
The coordinator joins both tasks, reaches acknowledged quiescence, and performs
the golden's ordered cleanup. One repetition has a 20-second hard timeout and
the complete 32-repetition command has a 15-minute hard timeout. Timeout,
panic, missing or duplicate record, recorder failure, unresolved action, or a
diagnostic bound violation invalidates the run; the harness never retries a
failed repetition. The 4,096 pump-entry cap is the delta from barrier release
through completed shutdown.

Every scripted action reserves two values from fresh shared-counter storage
initialized to zero; each reservation records `fetch_add(1) + 1`. The
invocation value is taken immediately before target lookup or scheduler API
entry. An exhausted submit has neither boundary, so it takes its invocation
value immediately before evaluating the authenticated exhausted no-op. The
response value is taken only after the result, all structural witnesses, and
any complete accepted-state entry have been published. Exactly 1,024 actions
therefore produce exactly the unique values `1..2048`. Cleanup uses a separate
counter domain. Per-producer program order is binding, and
`A.response < B.invocation` creates a cross-thread real-time edge. Invocation
adjacency, response adjacency, the numeric order of overlapping intervals, and
recorder append order are never linearization points. Every repetition must
contain at least one pair of cross-producer intervals that overlap
(`A.invocation < B.response` and `B.invocation < A.response`) and in which both
actions reach a non-sentinel shared command, control, endpoint, or wake
boundary. Exhausted-submit and unavailable-target no-ops cannot satisfy this
requirement; releasing the start barrier alone is not accepted as proof of a
genuine race.

The accepted-request registry publishes one mutex-protected entry containing
request ID, cancellation authority, accepted identity witness, and receiver
ownership before publishing the submit response counter. It never publishes
those fields piecemeal. This submit response is the harness action-response
counter. It is distinct from the actor's command-response publication and the
producer's subsequent consumption of that response. The checker reconstructs
the complete accepted path as distinct existential events:

```text
Invoke -> CommandReserve -> ReadyCommit -> ActorCommandClaim
       -> {ControlBind, EndpointBind} -> ActorCommandRespond
       -> CommandRelease -> RegistryPublish -> action Respond
```

The two bind events are unordered siblings. A rejected in-range offer follows
the same path through `ActorCommandClaim`, then goes directly through
`ActorCommandRespond` and `CommandRelease` to the action `Respond`, without
bind or registry-publication events. `ActorCommandRespond` moves the command
slot to `Responded`; `CommandRelease` is the producer's later consumption and
slot release. For consecutive ready-sequence entries, the previous
`ActorCommandRespond` precedes the next `ActorCommandClaim`, making the single
owner's FIFO processing explicit rather than inferring it from action-counter
order. Ready commits are also chained in witnessed sequence order, and reuse
of one command slot requires the previous ticket's `CommandRelease` before the
next ticket's `CommandReserve`. The command, control, and endpoint slot numbers
are three distinct namespaces and are never compared as though they shared an
allocator. Their complete identities are respectively `(slot, ticket)`,
`(slot, generation)`, and `(slot, generation)`. An accepted response carries
its request ID plus the exact control and endpoint identities returned by the
accepted handle; a response-order observer cannot reconstruct these values.

The checker projects each complete accepted identity into immutable indexes by
client, request ID, control slot/generation, and endpoint slot/generation. For
each cancel, receiver-drop, or drain action it then allocates exactly one main
event between that action's `Invoke` and `Respond`. A found target uses the
captured object boundary: `ControlCancel`, `ControlDisconnect`,
`PrimaryEndpointPop`, or `CachedEofRead`. The earlier mutex lookup is omitted,
not aliased to that boundary, and `RegistryPublish` precedes the captured event.
For `target_unavailable`, the main event is instead the actual `TargetLookup`.
For an absent lookup whose client is eventually accepted, `TargetLookup`
precedes its `RegistryPublish`; the resulting cycle check rejects a claimed
absence when publication was already forced before invocation. An unavailable
receiver lookup retaining a request ID must instead match the accepted identity
and follow both its `RegistryPublish` and the unique prior successful
home-producer drop response. These constraints preserve the exact one-main-event
budget without inventing an unobserved API boundary.

The packed control word is frozen as bit 0 cancellation, bit 1 receiver
disconnection, bit 2 terminal, and bits 3 through 63 generation. Valid live
generations are `1..=2^61-1`. Flags are monotone within one generation; a bind
increments that slot's generation and clears all three flags. Cancellation
sets bit 0, disconnection sets bits 0 and 1 atomically, and terminal publication
sets bit 2. Cancellation's `already_requested` and `already_terminal` results
leave the word unchanged. Disconnection leaves the word unchanged only when bit
1 was already set; otherwise it sets bits 0 and 1 even if bit 2 was already set,
in which case its disposition is `already_terminal`. A stale-generation
decision cannot modify the observed word. Serialized words are exactly 16
lowercase hexadecimal digits so JSON number precision is never part of the
contract.

Each operation records its interval plus a closed structural witness:

- a submit records command slot, nonzero ticket, and nonzero ready sequence if
  it reached ready commit; engine acceptance additionally records the request,
  control, and endpoint identities;
- cancel and destructor disconnect record the control slot, expected
  generation, and packed word loaded before and resulting after their final
  load/CAS decision;
- drain records a primary endpoint pop and the optional opportunistic EOF pop,
  each with endpoint slot/generation, drained count before and after, and any
  consumed output identity; a separate boolean identifies an already-cached
  EOF that entered neither endpoint boundary; and
- wake records whether dirty was already set and the park epochs observed
  before the signal and after acknowledgement.

An unreached boundary has `boundary=false` and all remaining raw fields zero or
null. A primary cached-EOF result has both endpoint witnesses at that sentinel
and `cached_eof=true`; every other result has `cached_eof=false`. A successful
wake requires `after_park_epoch > before_park_epoch`. Concurrent wakes may
coalesce on one later park epoch, and `dirty_was_set` does not attribute a pump
entry to a unique signal.

The compact capture is one JSON document with closed schema
`runnel.actor-race-history/1`, exact `repetition_count=32`, and repetitions
indexed `0..31`. It is UTF-8 without a BOM, contains only the ASCII string
literals or pattern-constrained strings defined below, uses lexicographically
sorted object keys, compact `,`/`:` separators with no other whitespace, and
ends in exactly one LF byte. Every permitted string is emitted literally;
backslash escape spellings, including equivalent `\/` or `\u` forms, are
noncanonical and rejected. Integers are unsigned canonical decimal JSON
integers: no sign, leading zero (except
zero itself), fraction, or exponent, and at most `u64::MAX`; `UInt` below means
exactly this type. `Hex64` is exactly
16 lowercase hexadecimal digits; `Sha256` is `sha256:` followed by exactly 64
lowercase hexadecimal digits. Parsers reject duplicate or unknown keys, an
array of the wrong length, nesting deeper than 16, a string longer than 128
bytes, or a string that matches neither an enumerated literal nor its stated
pattern. The complete file is bounded to 33,554,432
bytes before parsing and after encoding. A writer that exceeds the bound fails
the run; it never truncates, compresses, retries, or changes representation.

The exact top-level objects and key sets are:

```text
Capture = {
  "repetition_count": 32,
  "repetitions": [Repetition; 32],
  "schema": "runnel.actor-race-history/1",
  "workload": Workload
}

Workload = {
  "artifact_id":
    "sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c",
  "artifact_object_sha256":
    "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab",
  "artifact_page_table_sha256":
    "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c",
  "model_spec_sha256":
    "sha256:ed57d7961e65c76223c169cabebaff9c02d8293da026abb0c0c0a22d38079845",
  "specification": "runnel-m5-actor-stress-v1",
  "vector_file_sha256":
    "sha256:eca1faeee91a41d19d98be7ffdad6fc5cebb9027f3e7a634c01ea1cc394fb574",
  "vector_id":
    "sha256:5010492fb74eda207511b26811992ed4779814185b9f184663b37a37747bd051",
  "vector_schema": "runnel.actor-stress-vectors/2"
}

Repetition = {
  "actions": [Action; 1024],
  "cleanup_authorities": [CleanupAuthority; accepted_count],
  "cleanup_receivers": [CleanupReceiver; live_receiver_count],
  "diagnostics": Diagnostics,
  "observations": [Observation; observation_count],
  "post_shutdown": ProbeSnapshot,
  "pre_cleanup": ProbeSnapshot,
  "pre_shutdown": ProbeSnapshot,
  "repetition": UInt in 0..31,
  "shutdown": Shutdown
}
```

High-cardinality records are fixed-length tuples to keep 32 raw histories
bounded. Their positions, types, and sentinels are part of the schema:

```text
Action = [
  ordinal, producer, kind, submit_attempt, client_index,
  invocation, response, result, error, request_id, output,
  command, accepted, control, primary_pop, opportunistic_eof_pop,
  cached_eof, wake
]

ordinal        = UInt in 0..1023; tuple order equals ordinal
producer       = UInt in 0..1
kind           = "submit" | "cancel" | "receiver_drop" | "drain" | "wake"
submit_attempt = UInt in 0..205 for submit; null otherwise
client_index   = UInt in 0..63 for targeted actions, wake's consumed selector,
                 and submit attempts 0..63; null for submit attempts 64..205
invocation     = UInt in 1..2047
response       = UInt in 2..2048 and strictly greater than invocation
result         = "submit_accepted" | "submit_offer_exhausted" |
                 "cancel_requested" | "cancel_already_requested" |
                 "cancel_already_terminal" | "receiver_dropped" |
                 "drain_output" | "drain_empty" | "drain_eof" |
                 "wake_signaled" | "target_unavailable" | "error"
error          = Error or null; nonnull if and only if result is "error"
request_id     = nonzero UInt for an accepted submit or a target lookup that
                 found a completely published accepted entry; null for wake,
                 exhausted submit, rejection, or a target not yet published
output         = Output for "drain_output"; null otherwise
command        = CommandWitness
accepted       = AcceptedWitness
control        = ControlWitness
primary_pop    = PopWitness whose kind is "primary"
opportunistic_eof_pop = PopWitness whose kind is "opportunistic_eof"
cached_eof     = Boolean
wake           = WakeWitness

Error = [code, category, resource, required, limit]
  stale authority =
    ["request_not_found", "invalid_request", null, null, null]
  saturated engine admission =
    ["resource_exhausted", "resource_exhausted",
     "request slot count", 16, 16]
No other error is a valid captured outcome.

Output = [request_id: nonzero UInt, output_index: UInt, token_id: UInt <= 2^32-1]

CommandWitness = [boundary, slot, ticket, ready_sequence]
  sentinel = [false, 0, 0, 0]
  reached  = [true, slot in 0..7, nonzero UInt, nonzero UInt]

AcceptedWitness = [
  boundary, request_id, control_slot, control_generation,
  endpoint_slot, endpoint_generation
]
  sentinel = [false, 0, 0, 0, 0, 0]
  reached  = [true, nonzero UInt, slot in 0..15, nonzero UInt,
              slot in 0..15, nonzero UInt]

ControlWitness = [
  operation, boundary, slot, expected_generation,
  loaded_word, resulting_word, disposition
]
  sentinel = ["none", false, 0, 0,
              "0000000000000000", "0000000000000000", null]
  operation   = "none" | "cancel" | "disconnect"
  disposition = "requested" | "already_requested" |
                "already_terminal" | null
  reached     = operation != "none", boundary = true, slot in 0..15,
                expected_generation != 0, loaded_word/resulting_word = Hex64

PopWitness = [
  kind, boundary, slot, generation, drained_before, drained_after, output
]
  sentinel(primary) =
    ["primary", false, 0, 0, 0, 0, null]
  sentinel(opportunistic) =
    ["opportunistic_eof", false, 0, 0, 0, 0, null]
  reached = [kind, true, slot in 0..15, nonzero UInt,
             UInt, UInt, Output or null]

WakeWitness = [boundary, dirty_was_set, before_park_epoch, after_park_epoch]
  sentinel = [false, false, 0, 0]
  reached  = [true, Boolean, UInt, UInt]
```

Every nonapplicable witness is its exact sentinel. All 64 in-range submits have
reached command witnesses; their ready sequences are exactly `1..64` and their
accepted witnesses are reached if and only if result is `submit_accepted`. Per
command slot, tickets start at one and increase by one without gaps. Exhausted
submits have the command sentinel. The accepted tuple's request ID equals the
action request ID. Within each control and endpoint slot, accepted generations
start at one and each later accepted generation increases by one; the two slot
namespaces remain independent. A stale control error still has a reached
witness and null disposition. `cached_eof=true` is valid only for `drain_eof`
with both pop sentinels; every other action has `cached_eof=false`. The primary
output tuple equals the action output. Direct EOF has a reached primary pop and
opportunistic sentinel; last-output EOF acknowledgement has both pops reached;
empty and ordinary-output reads have a reached primary pop and opportunistic
sentinel. A primary output advances `drained_after` by exactly one; primary
empty or EOF leaves it unchanged. A reached opportunistic EOF always has null
output and unchanged drain counts, uses the same endpoint identity as its
primary, and has `drained_before` equal to the primary's `drained_after`.
The authenticated vector kind named `drop` is serialized as race-history kind
`receiver_drop`; every other kind retains the vector spelling.

Cleanup is serialized after the two producers and uses fresh counter storage
initialized to zero. Each boundary records `fetch_add(1) + 1`; every authority
record and then every receiver record receives one invocation/response pair.
`cleanup_counter_final` is the last issued value, so the exact issued range is
`1..2*(accepted_count + live_receiver_count)` and the stored final is
`2*(accepted_count + live_receiver_count)`:

```text
CleanupAuthority = [
  invocation, response, client_index, request_id, error, control
]
  invocation/response = UInt from the cleanup counter; response = invocation + 1
  client_index        = UInt in 0..63
  request_id          = nonzero UInt
  error               = Error or null
  control             = ControlWitness
  control.operation = "cancel" and control.boundary = true
  error is null with a nonnull disposition, or the stale-authority Error with
  a null disposition

CleanupReceiver = [
  invocation, response, client_index, request_id,
  terminal, outputs, eof_acknowledged, post_drop_quiescent
]
  invocation/response = UInt from the cleanup counter; response = invocation + 1
  client_index        = UInt in 0..63
  request_id          = nonzero UInt
  terminal         = Terminal
  outputs          = [Output; remaining FIFO suffix length <=
                      output_capacity_per_request = 2]
  eof_acknowledged = true
  post_drop_quiescent = ProbeSnapshot satisfying quiescence

Terminal = [
  request_id: nonzero UInt, outcome,
  committed_positions: UInt, emitted_tokens: UInt
]
  outcome = "completed" | "cancelled"
```

Authority records include every accepted request in ascending client-index
order. Receiver records include exactly the receivers still owned after the
script, also in ascending client-index order. A receiver interval begins before
terminal consumption and ends only after FIFO/EOF consumption, handle drop,
and the acknowledged post-drop snapshot. Its request count is exactly one less
than the preceding authoritative snapshot. Its `request_bytes` is strictly
less, while its `pump_entries`, `park_epoch`, and `engine_steps` are each
strictly greater, than that preceding snapshot.

Semantic recorder order is discarded. Observations are serialized as all
outputs sorted by `(request_id, output_index)`, then terminals by request ID,
then first-EOF observations by request ID:

```text
Observation = [
  kind, request_id, output_index, token_id,
  outcome, committed_positions, emitted_tokens
]
  id        = nonzero UInt
  index     = UInt
  token     = UInt <= 2^32-1
  positions = UInt
  tokens    = UInt
  output    = ["output", id, index, token, null, null, null]
  terminal  = ["terminal", id, null, null,
               "completed" | "cancelled", positions, tokens]
  EOF       = ["output_eof", id, null, null, null, null, null]
```

Each accepted request's output indexes are consecutive from zero and its
terminal emitted count equals that output length. EOS may appear only as the
last output. A `completed` sequence is nonempty and either reaches its
descriptor's maximum generation length or ends in EOS; a `cancelled` sequence
is a strict prefix of that maximum and contains no EOS.

The three lifecycle snapshots are deliberately distinct: `pre_cleanup` is the
acknowledged quiescent snapshot after producer join; `pre_shutdown` is the
acknowledged quiescent snapshot after all cleanup and authority drops; and
`post_shutdown` is the non-quiescent owner-done snapshot after cooperative
shutdown. Each object has exactly the following keys:

```text
ProbeSnapshot = {
  "command_in_flight": UInt,
  "command_ready": UInt,
  "command_reserved": UInt,
  "command_responded": UInt,
  "dirty": Boolean,
  "engine_steps": UInt,
  "outstanding_requests": UInt,
  "owner_done": Boolean,
  "park_epoch": UInt,
  "parked": Boolean,
  "pump_entries": UInt,
  "pump_hold_observed": UInt,
  "pump_hold_released": UInt,
  "pump_hold_requested": UInt,
  "pump_in_flight": Boolean,
  "request_bytes": UInt,
  "shared_bytes": UInt
}

RecorderStatus = {
  "allocated_capacity": UInt,
  "observation_count": UInt,
  "observation_limit": 675,
  "overflowed": Boolean,
  "poisoned": Boolean
}

Diagnostics = {
  "action_counter_final": 2048,
  "barrier_released": true,
  "cleanup_counter_final": UInt,
  "engine_steps_delta": UInt,
  "final_engine_steps": UInt,
  "final_pump_entries": UInt,
  "initial_engine_steps": UInt,
  "initial_pump_entries": UInt,
  "overlap_pair": [left_ordinal, left_boundary,
                    right_ordinal, right_boundary],
  "pump_entries_delta": UInt,
  "recorder_final": RecorderStatus,
  "recorder_initial": RecorderStatus
}
  left/right_boundary = "command" | "control" | "primary_endpoint" |
                        "opportunistic_endpoint" | "wake"

Shutdown = {
  "accepted_submissions": UInt,
  "discarded_output_events": UInt,
  "engine_steps": UInt,
  "rejected_submissions": UInt,
  "released_request_bytes": UInt,
  "remaining_shared_bytes": UInt,
  "shutdown_cancellations": UInt,
  "terminated_requests": UInt
}
```

Both recorder snapshots must be healthy, their allocated capacity must be
equal, and the final observation count must equal the observation array length.
The overlap ordinals identify opposite producers, their intervals overlap, and
the named boundary is reached in each corresponding action. Diagnostic deltas
are exact subtractions of their endpoints; pump delta is at most 4,096. Every
quiescent snapshot has `parked=true`, `dirty=false`,
`pump_in_flight=false`, and `owner_done=false`. `post_shutdown` has
`owner_done=true`, `dirty=true`, `parked=false`, and
`pump_in_flight=false`. Its pump/engine endpoints equal the diagnostic final
values, and its engine count also equals `shutdown.engine_steps`.
`cleanup_counter_final` equals twice the combined cleanup-array lengths.
Shutdown accepted/rejected counts equal the corresponding in-range action
results and sum to 64; all five shutdown effect/remaining fields are zero.

After the producers join, only an acknowledged stable-quiescence snapshot is
authoritative for accounting. It must report zero command occupancy and an
outstanding-request count equal to the still-owned receiver set, proving all
script-dropped records have reaped. While the pump is held, cleanup probes every
accepted authority in ascending offered-index order and retains each exact
control witness. After release and re-quiescence, remaining receivers are
consumed in ascending order: terminal first, then the exact FIFO output suffix
through EOF. For every accepted request, including script-dropped receivers,
the semantic publications not consumed by scripted drains are at most the
authenticated endpoint capacity of two. The harness re-quiesces after every
cleanup drop and requires the live count and request ledger bytes to decrease
strictly while pump entries, park epoch, and engine steps each advance
strictly. Authorities are then dropped, pre-shutdown quiescence must show zero
outstanding requests and zero request-owned ledger use. Cooperative shutdown
must have zero request effects. The post-shutdown owner-done snapshot must show
zero request and shared-ledger use.

The independent checker regenerates all descriptors and actions and constructs
a partial-order DAG from producer order and strict response-before-invocation
edges. It combines that DAG with object-local
command tickets, ready FIFO sequence, control generation/word chains, endpoint
drain chains, and the existential actor terminal/rebind events required by
observed dispositions. It checks topological satisfiability rather than sorting
counters into a total execution. A cross-producer target may be unavailable
only when its complete acceptance publication is not forced before that
action's invocation. The checker requires consecutive accepted IDs in ready
sequence order, exact typed saturation errors, valid sentinel use, output
identity and drain algebra, monotone control transitions, complete terminal and
first-EOF conservation, completed full sequences, cancelled strict prefixes,
cleanup conservation, and zero final ledgers. Each repetition is verified on
its own. No expected race digest, accepted set, physical count, or
cross-repetition byte equality is committed. Authoritative acceptance and CI
always enable the separate PyTorch gate for all 64 model sequences; a
`--no-model` checker mode may diagnose structural history only and is never
publishable evidence because it cannot independently validate token values or
the completed/cancelled prefix claims.

For the genuine race, an exact typed saturation error requires the closed
five-field engine result, a reached command witness, and no accepted witness or
request ID. Accepted request IDs must still be consecutive in ready-sequence
order, so a rejection followed by a later acceptance cannot leave an observable
ID gap. The capture has neither an admission-linearization occupancy witness
nor a request-ID high-water witness, so the checker does not reinterpret that
result as independent evidence that exactly 16 request records were live at the
rejection instant or that no otherwise-unobservable trailing ID was issued. The
action-by-action golden retains its stronger quiescent live-count proof.

#### Canonical actor semantic transcript

The golden digest is SHA-256 over one canonical binary transcript; it is
reported as `sha256:` followed by 64 lowercase hexadecimal digits. Every
integer is unsigned little-endian, reserved fields must be zero, and no native
`usize`, enum representation, JSON spelling, timestamp, interval counter, or
structural witness enters this semantic digest. The byte stream is:

```text
"runnel-m5-actor-semantic-transcript-v2\0"
u32(1024 action records)
u32(output publication count)
u32(terminal publication count)
u32(EOF acknowledgement count)
1024 action records in action-ordinal order
output records sorted by (client_index, output_index)
terminal records sorted by client_index
EOF records sorted by client_index
one shutdown record
```

An action record has tag `0x01`, followed by `ordinal:u32`, `kind:u8`,
`producer:u8`, `result:u8`, `error:u8`, `submit_attempt:u32`,
`client_index:u32`, `request_id:u64`, `value0:u32`, and `value1:u32`.
`submit_attempt` is `0xffffffff` for non-submit actions. `client_index` is
`0xffffffff` for an exhausted submit, the offered index for an in-range submit,
and the consumed selector for every targeted action including wake. For an
accepted submit, `request_id` is its assigned ID; an exhausted or rejected
submit uses zero. For another targeted action, it is the ID already assigned to
that client index when the action begins, even if the operation is now
unavailable; it is zero if that index has not yet been accepted. A wake always
uses zero because it is global. For `drain_output`, `value0` and `value1` are
output index and token ID; otherwise both are zero.

Action result codes are `0=submit_accepted`, `1=submit_offer_exhausted`,
`2=cancel_requested`, `3=cancel_already_requested`,
`4=cancel_already_terminal`, `5=receiver_dropped`, `6=drain_output`,
`7=drain_empty`, `8=drain_eof`, `9=wake_signaled`,
`10=target_unavailable`, and `11=error`. Error codes are
`0=none`, `1=invalid_request`, `2=unsupported`, `3=resource_exhausted`,
`4=cancelled`, `5=deadline_exceeded`, and `6=internal`, matching the public
scheduler category; `error` is nonzero exactly for result 11.

An output-publication record has tag `0x02`, followed by `client_index:u32`,
`request_id:u64`, `output_index:u32`, and `token_id:u32`. It records every
committed output exactly once whether later drained or discarded. A terminal
record has tag `0x03`, followed by `client_index:u32`, `request_id:u64`,
`outcome:u8`, `error:u8`, `reserved:u16`, `committed_positions:u32`, and
`emitted_tokens:u32`. Outcome codes are `0=completed`, `1=cancelled`,
`2=deadline_exceeded`, and `3=failed`; `error` uses the action-record error
table and is nonzero exactly for failed.
An EOF record has tag `0x04`, followed by `client_index:u32` and
`request_id:u64`, and appears exactly when endpoint output-EOF acknowledgement
first changes from false to true, whether by client EOF consumption or the
equivalent disconnect/discard settlement. Repeated `drain_eof` action results
add no further EOF record.

The final record has tag `0x05`, followed by
`accepted_submissions:u32`, `rejected_submissions:u32`,
`shutdown_cancellations:u32`, `terminated_requests:u32`,
`discarded_output_events:u32`, `reserved:u32`,
`released_request_bytes:u64`, `remaining_shared_bytes:u64`,
`final_request_bytes:u64`, and `final_shared_bytes:u64`. Actor and engine
report fields are copied exactly after checked conversion to the declared
widths. `rejected_submissions` excludes exhausted offers and unavailable
operations, because neither calls the actor. The pre-shutdown zero-ownership
gate requires `shutdown_cancellations`, `terminated_requests`,
`discarded_output_events`, `released_request_bytes`, and all three remaining or
final byte fields to be zero. Cleanup operations are fixed by the preceding
contract and are not additional action records. Their authority dispositions
are retained only in the logical cross-language capture, while their semantic
effects appear in output, terminal, EOF, and shutdown records.

Physical `engine_steps` and `pump_entries` are captured alongside the golden
result as observed bounded diagnostics, but neither enters this semantic byte
stream or its digest. Wake coalescing may change no-work owner re-entry counts
without changing any request semantics, so binding those counts into the
golden would reject a valid implementation-preserving scheduling change. The
4,096-entry pump cap remains a required guardrail, measured as the previously
defined delta, and the capture retains both diagnostic counts. Independently
structured Rust and Python generators must agree on every semantic transcript
byte and on the accepted digest in every gated live run.

Removing those two physical counters changes the binary record layout, so this
corrected pre-result format uses the `v2` domain above. The never-populated `v1`
domain from the preceding draft is superseded and must not be accepted as M5
evidence.

The `actor-concurrency-stress` correctness row aggregates the golden and race
subtests. Only the gated golden execution has a committed semantic digest; race
histories retain their interval and structural-witness corpus for independent
verification but have no expected event digest or accepted set.

Timing eligibility additionally requires exact accepted/rejected ID sets, no
lost or duplicate events, balanced ledgers, final request-owned bytes zero,
shared worker residue zero, both peak-ledger and `VmHWM` fields present, no
affinity violation or CPU migration during a timed interval, a 30-second child
timeout, a 15-minute full-capture deadline, 1 MiB stdout/stderr caps, a 32 MiB
evidence-directory cap, private tmpfs build/capture, offline locked builds, and
the existing 2 GiB tmpfs reserve. `VmHWM` is an observation under the 256 MiB
address-space hard limit, not a claimed RSS ceiling.

### Claim rules

A favorable timing is not an M5 acceptance gate. Cell-specific improvement
language requires every correctness/resource gate, 30 complete pairs, and a
95% paired-ratio interval wholly below one for lower-is-better metrics or
wholly above one for higher-is-better metrics. Goodput uses a paired-difference
interval wholly above zero. Exact fairness language requires identical
candidate service digests in all eligible repetitions, strictly smaller exact
maximum lag than baseline, Jain no lower than baseline, and the hard candidate
lag/gap gate; it receives no confidence interval.

The optional joint wording is:

> On the recorded shared host and fixed tiny-v3 synthetic workloads, the
> preregistered continuous scheduler improved the joint M5 outcome.

It is permitted only if homogeneous emitted-token-throughput median ratio is
at least 1.05 with lower interval bound above one, mixed-burst p95-TTFT median ratio is
at most 0.95 with upper interval bound below one, candidate maximum service lag
is strictly lower, candidate Jain is no lower, maximum runnable gap is at most
15, and every guardrail passes. Otherwise the report publishes positive,
negative, or ambiguous cell-specific outcomes.

No M5 result may be generalized to production traffic, natural language,
weighted fairness, arbitrary models, large-model long contexts, storage-device
bandwidth, constant RSS, other hosts, or an overall “faster and fairer” claim.

## Consequences

The scalar runtime acquires more internal structure, but its old API becomes a
composition of the same transaction used by scheduling. Paged preallocation
uses full admitted state capacity rather than growing with logical length,
trading a predictable bounded charge for unused tail space. Three-pass
attention performs more score computation to eliminate context-proportional
score storage.

The deterministic core permits exhaustive/property testing and exact trace
replay without timing races. The Tokio layer remains small and owns only
wake-up, channel, and lifetime mechanics. Stable grouping improves expert
locality but is not a matrix-batching claim. The v3 fixture gives page and
long-prefill mechanics a credible bounded exercise while keeping all evidence
synthetic and offline.

The most important residual system gap is live cache-backed expert execution:
M5's adapter/expert boundary makes that integration possible, but M5 does not
mislabel eager tiny weights as end-to-end out-of-core inference. The M5 review
must keep that residual explicit; scoped disclosure permits M5 closure. A
scheduler/store expert-execution path must resolve it before M7, however, or the
persistent mission Goal cannot be marked complete. Disclosure alone is not
sufficient for the flagship outcome.
