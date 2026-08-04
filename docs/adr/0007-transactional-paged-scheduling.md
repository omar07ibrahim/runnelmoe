# ADR 0007: transactional paged state and deterministic continuous scheduling

- Status: accepted; implementation in progress; measurement pending
- Date: 2026-08-03
- Last amended: 2026-08-04 (pre-measurement stress-protocol clarification)
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
validate their complete identity before scatter. Request IDs, request slots,
slot generations, engine/adapter transaction IDs, model/state IDs, revisions,
and positions use checked increments; exhaustion fails before work selection
and never wraps or reuses an identity. Adapter transaction identities are
observed against one engine-lifetime high-water mark and must increase
strictly across requests as well as within one request.

`finish_token` rejects a missing, duplicate, foreign, stale, wrong-rank,
wrong-expert, nonfinite, or dimension-mismatched contribution before it
creates a pending commit. Contributions may execute in expert-ID order, but
the mixture is always reduced in router-rank order. A completion permutation
cannot alter the numerical accumulation order.

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
phase, and trace update without further fallible work, panic, or yield. The
endpoint guard remains held after the callback so an adapter error returned
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
the next scan starts at the retained cursor. These rules, rather than a fresh
request-ID sort per wave, are the independently replayed policy.

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
attention, sampling, workspace, memory budget, and output sink but runs one
request to completion in accepted FIFO order with batch width one and no
cross-request expert grouping. The named candidate uses the DRR/coalesced
policy above. The comparison isolates scheduling and grouping rather than
numeric representation, ISA, or storage.

### Request state machine and token transaction

The request lifecycle is:

```text
validated -> queued -> admitted -> preparing -> expert_owned
          -> ready_to_commit -> ready | output_blocked | terminal
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
before preparation, after expert execution, and immediately before commit.
No retry, wake-up, or stale completion may advance state or RNG twice.

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
| coalesced batch capacity | shared static partition through shutdown; includes 64 bytes per task identity plus adapter contribution buffers and is charged once, never per participating request |
| trace capacity | shared static partition through shutdown at 128 bytes per bounded slot; draining changes occupancy, not capacity charge |
| model resident partition | validated static partition present when the scheduler is constructed |
| page-pool partition | exact configured M2 cache payload capacity when a cache is attached, otherwise zero |
| admission reserve | shared unavailable headroom through scheduler lifetime |

Fixed logical metadata charges are 64 bytes per command, actor-control,
request, output-event, terminal, expert-task, and state-page slot; 128 bytes per
trace slot; and 512 bytes per retained request record. Each actor command also
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
65,536 output events per request, 1,048,576 trace events, 65,536 tokens
per state page, 262,144 expert tasks per wave, and 1 TiB of logical ledger
capacity. Adapter context and vocabulary limits remain independently enforced.
Zero, exact-ceiling, ceiling-plus-one, multiplication overflow, and host-`usize`
conversion tests gate every field; a valid configured value can still fail the
minimum-operation fit check.

The frozen evidence configuration has an 8 MiB scheduler-accounted ceiling,
a 256 MiB address-space ceiling, one worker, ordinary command capacity 32,
normal total-outstanding/active/retained caps 32/16/32, and pressure-cell caps
16/16/16. It uses batch width 8, four waves per step, 16-token state pages,
64 output events per normal request, and 8,192 trace events. Configuration
sets `model_resident_partition = 5,632` bytes (the 5,600-byte v3 semantic tensor
payload rounded once to 64), `page_pool_partition = 0`, and
`admission_reserve = 1,048,576` bytes. The eager authenticated `Artifact` and
encoded object are dropped after model construction and before release; only
the model's declared tensor-buffer payload remains in the logical partition.
Fresh-child `VmHWM` still records any larger initialization peak. Configuration
fails if one minimum executable request plus all fixed shared partitions cannot
fit.

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
ID, nonblocking-drains every available output event, and acknowledges every
terminal churner; if the anchor is terminal, it records its result, drains it,
and acknowledges EOF in the same pass. The following turn's bounded step reaps
those ready records during its control prefix before committing its sole
service event. After the final turn, cleanup performs the corresponding reap
before shutdown. No sink operation may execute model work, and lifecycle reaps
are not service turns.

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

| Cell | Exact workload | Role |
| --- | --- | --- |
| `single-long-control` | 1 request: `P(896)`, 8 output tokens | descriptive long-prefill overhead and parity |
| `homogeneous-burst-16` | 16 requests: `P(16)`, 8 output tokens; horizon 16 | primary end-to-end output throughput and exact service lag |
| `mixed-prefill-burst-16` | prompt lengths by request ID: `896,16,512,64, 64,896,16,512, 512,64,896,16, 16,512,64,896`; 8 output tokens each | primary p95 TTFT and prefill-completion rate |
| `deadline-pressure-24` | 24 requests: `P(16)`, 8 output tokens; total-outstanding cap 16; deadline 20,000,000 ns after release | goodput, deadline, exact accept/reject set |
| `cancellation-pressure-12` | 12 requests: `P(896)`, 8 output tokens; cancel IDs 2, 5, 8, 11 at frozen queued, after position-0 commit, position-8 post-router/pre-expert, and position-895 post-scatter/permit/preapply hooks | transactional cleanup and ownership |

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
Accepted request ID 2 cancels before any promotion. ID 5 cancels immediately
after committing prompt position 0. ID 8 cancels before expert execution for
prompt position 8. ID 11 cancels after prompt position 895 has a validated
composite permit but before the final control check/apply, so that position,
its RNG preview, and its first output must not commit. The post-router
cancellation retains shared batch scratch until worker return while releasing
request ownership exactly once.

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

`runs` and `requests` are rectangular. Each `service` row contains the complete
valid prefix as compact arrays of request index and phase bit, rather than one
JSON object per token. Each `ledger` row contains initial category totals and
compact arrays of `(sequence, category_id, owner_id, signed_delta)`; static
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

#### Actor request and action streams

Actor validation has a deterministic interleaving test and a separate genuine
race stress. Both run the tiny-v3 model on the scalar backend through two Tokio
producer tasks and one scheduler actor. The exact limits are one compute worker,
ordinary-command capacity eight, outstanding/active/queued/retained-terminal
caps 16/8/16/16, prompt/generation/context ceilings 4/16/19, four-token state
pages, two output events per request, batch width eight, four waves per step,
1,024 trace events, an 8,388,608-byte logical-memory limit, no page-pool
partition, and a 1,048,576-byte admission reserve. The global context envelope
is 19 because configuration validates `4 + 16 - 1`; every derived request below
still has at most 16 model positions.

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

An accepted request stores its cancellation authority separately from its sole
receiver. Dropping the receiver does not drop that authority; it remains usable
until its generation becomes terminal and is recycled. A targeted operation
whose needed authority or receiver is absent, not yet accepted, rejected, or
already consumed is an `unavailable` no-op and makes no actor API call. A
stored authority that has since become stale is still called and records its
typed API error. A drain action performs exactly one nonblocking receive
attempt: it consumes at most one FIFO output event, observes empty, or
acknowledges EOF. Receiver drop explicitly publishes disconnect before
consuming the handle. Wake is global and never inspects the selected request.

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

Cleanup begins from quiescence. While the pump is held, the coordinator visits
request indices in ascending order and requests cancellation through every
still-generation-valid stored authority. It then releases the pump and waits
for quiescence. In ascending request-index order, each remaining receiver first
consumes its terminal result, then drains committed output in FIFO order through
EOF, and is dropped; a receiver already dropped by the script is skipped. The
harness drops the stored cancellation authorities, waits for quiescence, and
performs cooperative actor shutdown. Terminal and output publication observers
retain semantic records even when a receiver was previously dropped, so
discarded client delivery cannot erase a committed event from the golden
transcript.

The genuine race uses the identical actions and producer assignment for 32
repetitions. After initial actor quiescence, both producers wait behind one
start barrier and then execute their own action subsequence in original action
ordinal order without cross-producer gates. After both producer tasks join, the
coordinator waits for quiescence and performs the identical ordered cleanup
used by the golden execution. The 4,096 pump-entry cap is the delta from barrier
release through completed shutdown. Each operation records invocation and
response interval counters plus a structural witness:

- a submit records command-slot index and ticket and, if it reached ready
  commit, its ready-commit sequence; an accepted response also records the
  assigned request ID;
- cancel and disconnect record control-slot identity and the generation-bound
  packed control word before and after their load/CAS decision;
- drain records endpoint slot/generation, drained-event count before and after,
  and any consumed output identity; and
- wake records whether dirty was already set and the park epoch observed before
  the wake and after its acknowledgement.

Zero/sentinel witness fields are required for an operation that never reaches
the corresponding boundary. Shared interval counters establish only invocation,
response, per-producer, and response-before-later-invocation constraints. The
independent checker must choose a linearization consistent with those edges and
the structural witnesses; it must not use an adjacent counter value as the
linearization point or infer a total order from it. It then matches every
acceptance, typed rejection, cancellation, drop, drain, output, terminal, and
cleanup effect. Every repetition resolves each accepted request exactly once,
preserves committed-token semantics, resolves or deliberately drops every
receiver, and ends with zero request and shared ledger use.

#### Canonical actor semantic transcript

The golden digest is SHA-256 over one canonical binary transcript; it is
reported as `sha256:` followed by 64 lowercase hexadecimal digits. Every
integer is unsigned little-endian, reserved fields must be zero, and no native
`usize`, enum representation, JSON spelling, timestamp, interval counter, or
structural witness enters this semantic digest. The byte stream is:

```text
"runnel-m5-actor-semantic-transcript-v1\0"
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
`discarded_output_events:u32`, `reserved:u32`, `engine_steps:u64`,
`pump_entries:u64`, `released_request_bytes:u64`,
`remaining_shared_bytes:u64`, `final_request_bytes:u64`, and
`final_shared_bytes:u64`. Actor and engine report fields are copied exactly
after checked conversion to the declared widths; `pump_entries` is the
previously defined cap delta, and the final byte fields are the post-shutdown
ledger snapshot. `rejected_submissions` therefore excludes exhausted offers
and unavailable operations, because neither calls the actor. Both final byte
fields and `remaining_shared_bytes` must be zero. Cleanup operations are fixed
by the preceding contract and are not additional action records; their effects
appear in output, terminal, EOF, and shutdown records. Independently structured
Rust and Python generators must agree on every transcript byte before the
expected golden digest is committed.

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
