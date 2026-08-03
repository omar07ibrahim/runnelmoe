# Threat model

## Assets and trust boundaries

The completed RunnelMoE system is intended to protect host availability,
bounded memory/disk use, numerical integrity, artifact identity, request
isolation, and confidentiality of local prompts/tokens. Controls become claims
only at their roadmap gate. The runtime binary and compiled adapters are
trusted. Model artifacts and manifests are untrusted; later HTTP requests,
client disconnect timing, and storage failures join that boundary. The
operating system, local administrator, compiler, and cryptographic
implementation are outside this model.

Version 1 is a single-user localhost service, not a hostile multi-tenant
sandbox. SHA-256 proves byte identity against a trusted manifest; it does not
prove publisher authenticity. Users must obtain expected root digests through
a trusted channel.

## Security invariants

M1 enforces bounded canonical parsing, digest-derived leaf names, exact
length/SHA/page verification, finite tensors and numerical outputs, a fixed
adapter schema, an eager retained-byte budget, and content-redacted external
errors. Its local convenience reader rejects Unix leaf symlinks and special
files, but assumes no concurrent mutation or ancestor-directory substitution.

The cross-milestone system invariants are:

- Parse limits are enforced before allocation or multiplication.
- Manifest data never supplies an absolute/relative object path or URL.
- An object is used only after exact length and digest verification.
- Unverified bytes cannot enter the resident cache or compute backend.
- Runtime-accounted memory and queue counts cannot exceed configured caps.
- Cancellation releases ownership without publishing partial state.
- Logs/metrics omit request content and secrets by default.
- The eventual server binds only to loopback unless an explicit operator
  override is provided; no public deployment is part of this project.
- SIMD entry is impossible until dtype, dimensions, bounds, alignment,
  aliasing, pointer lifetime, and runtime ISA are validated.

## Threats and mitigations

| Threat | Controls | Verification | Gate |
| --- | --- | --- | --- |
| Path traversal or symlink substitution | digest-derived names; retained directory FDs plus one-time-selected `openat2` beneath/no-symlink or component-wise `openat(O_NOFOLLOW)` | leaf FIFO/symlink, ancestor symlink, retained-rename, and CAS-layout tests | M1/M2 |
| Manifest allocation bomb | byte/count/rank/string caps and checked arithmetic before reserve | boundary tests now; fuzzing next | M1/M2 |
| Truncation, mutation, or reordered pages | exact lengths, retained handles, bounded positional reads, and whole/page SHA before exposure | truncation, extension, corruption, reordering, unlink/rename retention, and cancellation checkpoints | M1/M2 |
| Cross-artifact object replay | trusted expected manifest ID plus object digest, page geometry, tensor role/shape binding | expected-ID and corrupt-preexisting-object rejection | M1/M2 |
| Partial-object publication | exclusive staging, hash and `fsync`, no-replace publication, manifest last | cancellation/crash at each transition | M2 |
| Duplicate concurrent load | one cache-owned in-flight generation; bounded waiters share completion/error | deterministic worker-gate and concurrent fan-out tests | M2 |
| Use-after-evict | explicit page leases; retire before reclaim; rounded payload charge retained through final lease | eviction/invalidation/shutdown state tests | M2 |
| Integer overflow/shape confusion | checked arithmetic and exact dtype/byte/shape validation | generated mutation corpus | M1 |
| Valid model construction allocation failure or partial publication | closed tensor table; borrowed verified bytes; fallible owned-buffer reservations; model identity assigned last | every-reservation injection, callback failure, impossible-host allocation, and retry parity tests | M1/M5 |
| NaN/Inf or unstable routing | finite checks and specified score/tie policy | special-value and tie tests | M1 |
| Compression bomb | compression absent from and rejected by RMOA v1 | exact-schema tests | M1 |
| Memory/queue denial of service | eager metadata caps; 64-byte-quantized page-pool ledger; bounded entries, loads, workers, queues, waiters, leases, and traces | exact budget, queue saturation, hostile configuration, and failed-admission tests | M1/M2/M5 |
| Cross-request state or stale completion | generation-tagged request/state identities; contribution identity includes position, rank, expert, revision, and transaction | missing/duplicate/foreign contribution, slot-reuse ABA, completion-permutation, and sibling-cancellation tests | M5 |
| Malformed external decoder adapter | trusted extension boundary; nonzero checked identities with private fields; contiguous task-rank contract; complete scheduler envelope validation; return-free commit apply | external mock implementation plus malformed rank/identity/completion tests | M5 |
| Cancellation/deadline partial token | prepared state is immutable; K/V, RNG, and output commit at one actor boundary after the final control check | fault/cancel/expiry injection at every token phase and exact before/after-commit outcome matrix | M5 |
| Output or command backpressure deadlock | bounded channels; full output blocks only its owner; independently reserved terminal slot; disconnect cancellation does not depend on command capacity | deterministic saturation, stalled receiver, sibling progress, disconnect, and shutdown ownership tests | M5/M6 |
| Scheduler starvation | equal-weight token-quantum DRR, stable order, reactivation credit cap, blocked requests consume no service | every-prefix maximum service lag, runnable-gap bound, independent reference trace, and continuous-arrival stress | M5 |
| RNG stream corruption across batching | request-owned versioned stream; preview advances only with token commit; stable probability/tie order | independent Python vectors plus batch/chunk/cancel/retry permutation tests | M5 |
| Cache-trace allocation or oracle explosion | canonical closed JSONL schema; file/line/page/event/prediction caps; complete validation before replay; uniform-geometry MIN; explicit state cap on variable-byte DP | adversarial parser corpus, deterministic arbitrary-byte smoke, oversized configuration tests, and exhaustive tiny differentials | M3 |
| Slow client or abandoned SSE | deadlines, bounded output channel, disconnect cancellation | stalled/disconnect black-box tests | M6 |
| High-cardinality telemetry | bounded metric labels; IDs only in sampled traces | metrics cardinality test | M6 |
| Secret/prompt disclosure | allowlisted output; tokenless typed errors; metadata-only `Debug` for models, tensors, adapter work, sampling previews, and reusable scratch | sentinel debug/error tests now; server log capture next | M1/M5/M6 |
| Illegal SIMD or memory unsafety | scalar fallback, runtime CPUID, narrow C ABI | sanitizer/random differential tests | M4 |
| Dependency or CI compromise | lockfiles, exact dependencies, pinned CI action revisions | CI and release audit | ongoing |
| Disk exhaustion | no committed checkpoints; allocation-aware CAS ceiling, staging/orphan accounting, and mandatory filesystem reserve | exact injected budget arithmetic and stable reserve endpoints | M1/M2 |

## Parser rejection matrix

The corpus must cover wrong magic/version/endianness/critical flags; every
truncation boundary; invalid UTF-8/control/path characters; excessive and
duplicate names/IDs; noncanonical hashes/order; zero or excessive rank and
dimensions; overflow in offsets, products, alignment, page rounding, and
record counts; unsupported dtype/encoding; logical/stored size disagreement;
overlap/misalignment/out-of-file ranges; missing/trailing data; digest
mismatch; valid page reordering; and unknown required adapter fields.

M2 I/O tests cover short reads, retained-descriptor replacement resistance,
truncation/extension, cancellation before and after physical reads, cancellation
during hashing, and cancellation immediately before both publication paths.
No failure may expose verified status or release an owned payload charge before
worker completion. Same-UID concurrent content mutation remains a documented
residual risk, not a filesystem-isolation claim.

M3 traces additionally reject non-ASCII/CR framing, a missing terminal LF,
noncanonical JSON, unknown or duplicate fields, sequence gaps, unknown pages,
invalid geometry, noncausal or dangling router signals, duplicate experts,
and score overflow. The entire catalog and event stream validates before a
policy can emit a plausible partial result. The CLI rejects symbolic-link and
non-regular trace leaves. It performs one no-follow, nonblocking open, checks
the retained descriptor, reads at most the 5 MiB ceiling plus one byte, and
rejects initial/read/final length disagreement. Thus a leaf swap cannot redirect
the read and a FIFO cannot block it. Ordinary symlinks in ancestor components,
hard links, and equal-length writes through the same inode are documented
same-UID residual risks, not rejected path classes. Router metadata is an
explicit normalized scalar payload charge rather than allocator/RSS usage;
that payload and the tiny exact-oracle state count have separate pre-replay
ceilings.

M3 evidence verification treats its own directory as untrusted. It opens the
root, `figures` directory, and exact closed file set with retained no-follow
descriptors; classifies every entry; sums all declared lengths before reading;
and performs bounded reads with final metadata checks. Capture builds in a
private tmpfs child and revalidates clean HEAD, the historical harness blob,
and the executable digest before the staged directory is renamed into place.

M5 request validation additionally rejects empty/oversized prompts, invalid
token IDs, context and prompt-plus-generation overflow, zero or excessive page,
chunk, queue, batch, output, worker, and trace caps, arithmetic overflow in
every semantic charge, invalid temperature/top-k/top-p values, deadline
overflow, duplicate external identity when exposed, and unsupported sampling
versions. Rejection precedes prompt copying and cannot mutate queue order,
deficits, RNG, high-water counters, or another request's admission plan.

M5's logical ledger is exact only for declared semantic payload capacities and
fixed metadata charges. Allocator control blocks, executor internals, code
pages, and filesystem page cache remain outside it; fresh-child `VmHWM` and a
configured reserve are separate evidence. The generated width-8,
1,024-position adapter-v3 fixture now demonstrates paged state, streaming
attention, fixed scratch, and transactional adapter mechanics; it does not
demonstrate hostile tenant isolation or production long-context behavior.

## Abuse cases for later serving

M6 tests malformed/deep JSON, oversized bodies, excessive messages/tokens,
invalid sampling floats, NaN/Infinity representations, request floods, queue
starvation, deadline overflow, stalled readers, disconnect storms, invalid
UTF-8, unknown models, and metric scraping during load. Authentication and TLS
are explicitly not supplied by the local server; operators needing remote
access must place a reviewed gateway in front, outside this project's default.

## Residual risks

- SHA-256 cannot identify a malicious but internally consistent model.
- Bounded error-message construction and the earlier format-parser layer still
  use standard-library allocation paths that may abort under catastrophic host
  exhaustion. Runtime model success-path capacities are fallible and identity
  is unpublished on error, but end-to-end allocator control remains incomplete.
- Resource accounting cannot exactly predict allocator, runtime, or kernel page
  overhead; RSS evidence and a reserve are required.
- Memory-mapped or cached filesystem data may affect host page cache outside
  process RSS. RunnelMoE uses explicit reads and reports this limitation.
- Floating-point agreement is tolerance-based across compilers/ISAs.
- Local processes with the same user privileges can observe files and may be
  able to inspect process memory or modify an already-open regular file.
- Descriptor-relative traversal prevents pathname redirection, but it is not
  hostile same-UID isolation. Every demanded page is still authenticated
  before use.
- The M1 eager convenience reader remains for a trusted local directory that is
  not concurrently replaced. M2 descriptor-relative reads and transactional
  CAS publication are the hardened runtime path.
- The M3 simulator is not a production isolation boundary. Its instantaneous
  prefetch model omits I/O latency, cancellation, in-flight ownership, and
  storage contention; conclusions are limited to modeled byte traffic and
  cache state.

Security issues should follow [SECURITY.md](../SECURITY.md), not a public issue.
