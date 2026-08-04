# System design

## Scope

RunnelMoE is a local inference runtime and research harness that jointly
schedules sparse-model tokens, expert pages, resident memory, and storage I/O
under a declared memory budget. “Model-agnostic” means the storage, cache,
scheduler, and serving layers depend on a versioned adapter interface; it does
not mean arbitrary checkpoints work without adapter code.

The first supported adapter is deliberately tiny and synthetic. It exists to
exercise a complete decoder path:

1. a fixed deterministic tokenizer boundary;
2. token embedding and RMS normalization;
3. two-head causal attention with explicit prefill/decode state;
4. stable top-2 routing with deterministic tie-breaking;
5. independently stored gated-MLP experts and weighted dispatch;
6. residual composition, final normalization, and LM projection; and
7. deterministic greedy generation.

Default fixture dimensions are intentionally hand-inspectable: vocabulary 32,
hidden width 8, one decoder block, four experts, top two selected, expert width
12, and a short context cap. Weights are deterministically formula-generated
from tensor ID and row-major index; there is no seed and no downloaded data.
The M5 runtime provides deterministic seeded sampling primitives; scheduler
integration and the serving policy surface remain milestone work.

Tiny-v1 deliberately has no positional encoding. In this one-layer fixture,
the final-token result is therefore invariant to permutations of the preceding
token multiset. It still exercises causal state, parser, routing, expert, and
generation boundaries, but it is not evidence of language quality or a
production model architecture.

## Architecture

The source for the architecture figure is
`docs/diagrams/runtime.dot`; generated images are disposable outputs.

    adapter plan
         |
    request scheduler ---- policy / trace
         |                     |
    tensor leases <------ verified page cache
         |                     |
    scalar or C-ABI kernel   bounded I/O pool
                               |
                         immutable RMOA objects

The M3 cache simulator sits beside this production path rather than inside it:

    causal trace generator ----> strict JSONL validator
                                      |
                     +----------------+----------------+
                     |                |                |
                online policies   uniform MIN    tiny exact-byte DP
                     |                |                |
                     +-------- checked byte ledger ----+

This separation keeps workloads policy-neutral. A production-cache trace
records decisions already made and therefore cannot validate a replacement
policy independently. A shared golden schedule instead cross-checks the
simulator and production cache as two implementations.

### Boundaries

- **Adapter:** validates topology, maps semantic tensor roles, and owns
  sequence-state equations. Adapters are compiled versioned code, never
  untrusted dynamic plugins.
- **Scheduler:** admits requests against a complete memory ledger, chooses
  deterministic token-boundary work, groups expert calls, and scatters results
  without changing per-request order or RNG streams.
- **Tensor store:** resolves semantic tensor IDs to verified immutable objects.
  It never receives a filesystem path from a manifest.
- **Cache:** provides byte-accounted leases over verified fixed-size pages.
  Its explicit states are `absent`, `loading`, `resident`, and
  `retiring`. A leased page cannot be evicted.
- **I/O:** a synchronous positional-read implementation is the reference. The
  asynchronous implementation uses the same bounded buffer ownership and
  completion contract.
- **Compute:** scalar Rust is authoritative. Optimized code crosses a narrow C
  ABI only after Rust validates shape, dtype, extent, alignment, aliasing, and
  runtime ISA.
- **Oracle:** Python/PyTorch evaluates the documented model equations with a
  different module structure and control flow. It cannot consume production
  routing decisions or intermediate outputs as inputs.
- **Cache-policy simulator:** validates bounded canonical traces, applies one
  complete victim plan atomically, and reports demand, speculative, resident,
  and normalized metadata-payload bytes separately. Router state uses exact
  request/target-step/layer identity; same-step scores survive every page
  demand and older targets retire deterministically. Scores protect pages only
  for the request and target being evaluated—cross-request lookahead and
  competing-horizon arbitration are deferred to M5. It cannot alter numerical
  routes.
- **Serving:** translates a documented HTTP subset into bounded runtime
  requests. It has no direct tensor, cache-policy, or filesystem access.

## Interfaces

The conceptual adapter interface is intentionally small:

    validate(model_spec, tensor_catalog) -> validated_model
    new_sequence(prompt_tokens, limits, rng_seed) -> sequence_state
    plan_prefill(sequence_state, token_chunk) -> tensor_requirements
    plan_decode(sequence_state, token) -> tensor_requirements
    apply(sequence_state, leased_tensors, backend) -> logits

Tensor requirements identify semantic IDs, byte ranges, access reason
(`demand` or `prefetch`), deadline, and priority. Cache policy sees only
immutable trace events and capacity; it cannot alter numerical execution.

Storage has two equivalent entry points:

    read_verified_page(object_id, page_index, owned_buffer, cancellation)
    read_verified_page_async(object_id, page_index, owned_buffer, cancellation)

Both read and authenticate one complete logical page, including the short final
page, or return a typed error. A tensor slice is exposed only after every
intersecting whole page verifies; physical read amplification is traced.
Partial unverified bytes never become cache-resident.

## Runtime invariants

These are end-state system invariants. M1 implements the numerical contract,
byte identity/integrity, schema failure behavior, and an eager artifact budget
for a trusted local directory. Descriptor-relative path safety, the complete
memory ledger, publication, cache leases, and scheduling close in M2 and M5.

1. **Numerical semantics:** stable top-k orders by descending score and then
   ascending expert ID. The scalar reference accumulates dot-product and RMS
   input dimensions in ascending index order, attention values in ascending
   causal-position order, and selected experts in router-rank order. Optimized
   paths meet the declared tolerance; greedy token IDs match exactly.
2. **Identity:** every consumed byte belongs to the manifest's root identity
   and a length-checked SHA-256 object. No mutable path is trusted after open.
3. **Bounded resources:** admission counts resident pages, in-flight buffers,
   sequence state, kernel scratch, queued work, and bounded metadata. Cache
   capacity is never oversubscribed; observed RSS is reported separately.
4. **Publication:** only complete verified objects are atomically published.
   Cancellation leaves no addressable partial object.
5. **Lease safety:** resident storage outlives all consumers. Eviction removes
   eligibility before reclaiming bytes.
6. **Determinism:** with identical artifact, request, seed, and backend,
   scheduling and batching cannot change tokens.
7. **Fail closed:** unknown versions, required fields, dtypes, flags, adapters,
   or numeric policies are rejected.
8. **Evidence:** tracing is observational; disabling it does not change policy
   decisions or numerical order.

## Artifact contract

RMOA version 1 is specified by [the format contract](FORMAT.md), ADR-0002,
[the M1 runtime decision](adr/0003-tiny-reference-runtime.md), and
[the verified data-plane decision](adr/0004-verified-data-plane.md).
The bounded JSON manifest points only to lowercase SHA-256 object IDs in a
derived `objects/sha256/` namespace. Tensor descriptors have a semantic role,
dtype, shape, logical length, object digest, and a required digest-bound page
table. All arithmetic is checked before allocation.

Version 1 excludes compression, sparse aliases, executable metadata, remote
URLs, manifest paths, host-endian values, implicit strides, and overlapping
ranges. The artifact ID is the SHA-256 of the exact canonical manifest bytes.

## Memory scheduling

The configured runtime budget is partitioned explicitly:

    total = object_and_policy_metadata + waiter_and_lease_metadata
          + sequence_state + kernel_scratch + page_pool_capacity
          + request_output_trace_queues
          + admission_reserve

    page_pool_used = loading_bytes + resident_bytes + retiring_bytes
    loading_bytes <= max_in_flight_bytes <= page_pool_capacity

Capacity, including the configured capacity-quantum padding, is reserved before
every allocation or I/O submission and released exactly once by its owner. The
in-flight limit is a subset cap within the page pool, not a second allocation
pool. A shared
physical page buffer is charged once as it moves from loading through resident
or retiring; each waiter and lease is charged separately.

The data plane charges page payload capacity at an explicit 64-byte quantum and
moves the reader's `Vec` allocation into an `Arc`-owned control block without a
second payload copy. Entry/control-block and allocator bookkeeping are bounded
or observed separately; process RSS remains the authoritative whole-process
observation rather than an inferred allocator total.
Configuration is rejected if the minimum executable operation cannot fit.
The production cache operates on fixed-size pages plus an aligned short tail.
The research simulator permits unequal logical and charge bytes so online
policies and accounting can be tested against those tails. Offline Bélády/MIN
is called optimal only for uniform charge and uniform miss cost; a bounded
exponential dynamic program supplies tiny variable-byte correctness cases.
The data-plane API exposes large tensors as ordered pages; the M2 tiny-runtime
adapter still reconstructs complete verified tensors before compute. Tensors
that share a physical page share its buffer charge.

The I/O backend owns a submitted buffer until completion or acknowledged
cancellation. Request cancellation becomes terminal only after buffer
ownership returns. A completed page may publish at exactly one atomic
state transition from `loading` to `resident`; cancellation that linearizes
first prevents publication. One in-flight owner serves duplicate waiters, and
each waiter independently releases its metadata reservation.

That terminal-ownership statement applies to direct I/O submissions. A cache
waiter owns only its waiter and prospective-lease reservations, so it may
return after withdrawing them; any now-unwanted cache-owned physical load stays
charged until its worker completion releases the page-pool reservation.

The M5 scheduler contract is frozen in
[ADR-0007](adr/0007-transactional-paged-scheduling.md) and is being implemented
in independently testable vertical slices. The completed state slice supports
adapter v3, checked request-bounded page layouts, eager fallible page
allocation, nonwrapping model/state/transaction identities, and allocation-free
K/V commit permits. Compatibility calls compute against an unpublished bound
candidate, so a failed first token leaves the caller's unbound shell unchanged.

Tiny-model construction validates the complete closed 22-tensor role, shape,
and dtype table before requesting payload bytes. The direct verified-artifact
path decodes borrowed slices without transient tensor copies; ordered staging,
f32/BF16 decode buffers, shapes, and expert arrays reserve fallibly. Model
identity is assigned only after every weight has validated and all owned
storage exists, so injected failure at any reservation cannot publish or
consume an identity. The callback-based store boundary remains responsible for
fallibly producing its owned byte vector before returning it to the runtime.

Tiny adapter v3 retains the generated tiny equations and compact BF16 experts
but raises the frozen synthetic context cap to 1,024. Its token-major K/V state
uses 16-token pages charged at full admitted capacity. Stable three-pass
streaming attention allocates no score vector proportional to context. The
fixture and boundary tests exercise multi-page mechanics only; this is not a
large-model performance claim. A fallibly preallocated Rust sampler now
implements the frozen SplitMix64/top-k/top-p arithmetic and is checked against
an independently generated Python vector set; preview does not publish RNG
state. Its public layout reports exact candidate payload and one rounded
ledger charge without exposing the private candidate representation. The
public `DecoderAdapter` is a trusted model-extension boundary that splits token
work into prepare, owned expert tasks, validated contributions, deterministic
rank-order finish, and a single-use state commit. It exposes fixed vocabulary
and stop-token policy, and its task iterator contract requires contiguous
router ranks. Complete model/state/revision/position/transaction identity
follows every phase; external implementations construct checked, nonzero
identity tags, but those tags are not capabilities. Conforming adapters keep
their binding inside model-derived task and contribution payloads opaque, and
the scheduler still validates the complete envelope at every boundary. A
higher-ranked synchronous callback prevents a validated commit capability from
escaping into an outer future; all fallible work precedes its allocation-free,
return-free apply. The fixed 192-byte tiny-adapter scratch is reused through
the full 1,024-position context without growth. The synchronous scheduler now
composes adapter state, DRR credit, RNG, output, phase, and trace publication at
one non-yielding boundary. It uses bounded FIFO admission, retained-round DRR,
expert-sorted waves, per-request output backpressure, exact category ownership,
generation-bound atomic request controls, and prevalidated release permits for
terminal/reap/shutdown cleanup. A live monotonic-clock and control snapshot is
taken inside the adapter's validated commit callback; suppression drops both
unapplied permits before terminal cleanup, so that position publishes no state,
RNG, output, trace, or service-credit debit. A generation-tagged endpoint lock
is acquired before sampling and remains held across that callback. After the
adapter returns, the scheduler classifies any post-callback failure and then
infallibly publishes the optional output and independent terminal result from
the same guard. Endpoint and control lifecycle guards make admission and reap
cross-registry transitions rollback-safe. Deterministic gated tests exercise
late cancellation, inclusive expiry, precedence, credit recovery, stale slot
reuse, endpoint saturation, disconnect, and lifecycle rollback. The concurrent
actor, full adversarial interleaving matrix, independent trace/fairness replay,
and accepted evidence remain incomplete until the rest of M5 lands. Weights
remain eagerly resident in M5, so live cache-leased expert execution remains an
explicit system gap.

## Error model

Public operations return stable categories. Internal source chains are retained
only when they are bounded and content-safe; the scheduler deliberately
discards raw adapter and sampler sources at its public boundary:

| Category | Examples | Retry |
| --- | --- | --- |
| `invalid_artifact` | schema, shape, overflow, unknown required value | no |
| `integrity_failure` | length or digest mismatch, page reordering | only after replacing source |
| `unsupported` | adapter, dtype, format version, ISA-only request | no |
| `resource_exhausted` | memory admission, queue, body, context cap | after reducing demand |
| `io` | open, positional read, sync failure | policy-dependent |
| `cancelled` | caller cancellation or disconnect | no |
| `deadline_exceeded` | queue, I/O, or generation deadline | caller choice |
| `internal` | violated invariant or unexpected backend failure | no automatic retry |

Errors never contain tensor bytes, prompt text, credentials, or unrestricted
host paths. HTTP mapping will be documented with M6.

## Observability

Trace events use monotonic timestamps, stable request pseudonyms, page/object
IDs, byte counts, reason, queue/wait/compute durations, cache decision, and
outcome. Prompt and generated content are absent by default. Metrics use
bounded labels; object, expert, and request IDs remain in sampled traces rather
than Prometheus labels.

Required data-plane counters include demand bytes, physical bytes read, cache
hits/misses/admissions/evictions, in-flight and resident bytes, useful and
wasted prefetches, I/O wait, compute time, and process RSS samples.

## Non-goals

- training, fine-tuning, distributed inference, GPUs, or public hosting;
- downloading, redistributing, or initially running Kimi K3 weights;
- accepting arbitrary code or tokenizer implementations from checkpoints;
- promising constant RSS independent of allocator/runtime overhead;
- claiming every MoE family is supported by one adapter;
- encrypted artifacts, authenticity signatures, or hostile multi-tenant
  isolation in version 1; and
- speed claims before repeatable measurements on named hardware.
