# ADR 0006: compact BF16 expert GEMV with isolated AVX2 dispatch

- Status: accepted; measurement outcome pending
- Date: 2026-08-03
- Milestone: M4

## Context

The scalar M1 runtime is intentionally easy to audit, while the M2 data plane
shows that expert bytes can move through a bounded verified cache. M4 must add
one real optimized compute path without weakening either boundary. The first
slice needs a compact representation, an independently callable reference, a
small cross-language ABI, runtime ISA checks, sanitizer coverage, full-model
parity, and evidence whose acceptance does not depend on a favorable timing.

The current tiny fixture is useful for end-to-end correctness but its 8- and
12-element dot products are not a credible throughput benchmark. A separate
bounded synthetic microbenchmark therefore measures production-shaped GEMVs,
while only the tiny generated adapter participates in CI and runtime parity.

This is an independent implementation from public numeric-format, compiler,
ISA, and FFI specifications. No implementation, fixture, prose, or benchmark
result from the Kimi K3 C prior-art repository is used.

## Decision

### Artifact and adapter contract

RMOA remains at format version 1, which already has a closed `bf16-le` dtype.
The existing `runnel.tiny-causal-moe` adapter version 1 is unchanged and still
requires every tensor to be `f32-le`. Adapter version 2 keeps the same topology
and uses `bf16-le` only for the twelve routed-expert `gate`, `up`, and `down`
matrices. Embedding, attention, normalization, router, and LM-head tensors
remain `f32-le`. A dtype in the wrong role or adapter version fails closed.

BF16 storage is one little-endian unsigned 16-bit word per element. Widening is
exact:

```text
f32_bits = u32(bf16_bits) << 16
```

Fixture generation converts finite f32 recipe values with round-to-nearest,
ties-to-even. Runtime loading does not requantize; it validates each word,
rejects either infinity and every NaN encoding, preserves both signed zeros
and finite subnormals, and retains compact host-endian `u16` storage. It must
not materialize a complete f32 expert matrix during model construction.

The deterministic formula happens to make every tiny expert coefficient
exactly BF16-representable. The expected v2 object length is therefore 5,600
bytes rather than v1's 7,904 bytes: the expert payload is 2,304 rather than
4,608 bytes. These values remain design calculations until generated artifact
tests verify them; they are not a performance result.

### Authoritative operation

M4 optimizes exactly one production operation:

```text
y[row] = sum(column=0..columns-1,
             widen_bf16(weight[row, column]) * input[column])
```

Weights are dense row-major BF16. Input and output are f32. Dimensions are
positive. The safe Rust reference accumulates columns in ascending order and
is independently callable for every supported host. Tiny adapter v2 invokes
the operation three times per selected expert: gate `[12, 8]`, up `[12, 8]`,
and down `[8, 12]`. SiLU, the gated product, expert weighting, and accumulation
across selected experts remain safe Rust. M4 does not fuse the complete MLP,
thread one GEMV, or change routing semantics.

### Native boundary

A new `runnel-kernels` crate owns compact matrices, safe dispatch, and the only
reviewed Rust unsafe island. Existing crates continue inheriting the workspace
`unsafe_code = "forbid"` lint. The kernel crate instead denies unsafe
operations inside unsafe functions, documents every proof obligation, and
keeps native declarations, calls, and x86 control-register reads in one
reviewed module with minimal documented unsafe blocks. Its public API is safe.

The versioned C ABI uses no Rust layout, enum, boolean, allocation, callback,
or ownership type:

```c
int32_t runnel_bf16_gemv_avx2_v1(
    const void *weights, size_t weight_bytes,
    const void *input, size_t input_bytes,
    void *output, size_t output_bytes,
    size_t rows, size_t columns);
```

The shared C header freezes status values: `0` success, `1` null pointer, `2`
invalid/zero dimension, `3` size overflow, `4` byte-length mismatch, `5`
natural-alignment failure, `6` address-range overflow, `7` range overlap, and
`8` unavailable ISA. Rust maps every known value to a typed error; an unknown
value is an internal native-contract violation and never triggers fallback or
publication of workspace output.

Rust and C both validate non-null addresses and nonzero dimensions. Before any
length equality, range, or cast, they separately checked-multiply all three
expected byte counts: `rows * columns * sizeof(uint16_t)`,
`columns * sizeof(float)`, and `rows * sizeof(float)`. They then require those
exact byte lengths, natural `uint16_t`/`float` alignment, checked
address-plus-length arithmetic, and pairwise nonoverlap of all three byte
ranges before casting the raw addresses. A C
caller must additionally guarantee that each range names live readable or
writable storage for its declared length; C cannot prove an allocation extent.
For the complete call duration, direct C callers must keep weights and input
immutable and give the call exclusive write access to output. Concurrent calls
may share immutable weights/input but their output ranges must be disjoint.
The safe Rust wrapper discharges that obligation from borrowed slices and a
caller-sized reusable `GemvWorkspace`. Both backends compute into its temporary
output and publish only after status, length, and finiteness checks pass, so a
failure cannot expose a partial result. A finite-input wrapper is constructed
before dispatch and may be reused while that immutable input slice remains
borrowed. Workspace allocation and input validation are outside benchmark
timing; the identical final output copy is included for both variants.

The exported C entry point is baseline-safe and returns a fixed signed status
code. It performs a second AVX2 capability check before calling an internal
function compiled with a function-local `target("avx2")` attribute. The native
loop uses unaligned vector loads, widens eight BF16 words in registers, uses
separate f32 multiply and add instructions, and finishes every tail scalarly.
It uses neither FMA nor global `-mavx2`, `-march=native`, fast-math, FP
contraction, heap allocation, mutable global state, or process floating-point
mode changes.

The Cargo feature `native-avx2` is enabled by default but produces a native
object only when host and target triples are identical Linux x86_64 GNU targets.
That supported native build invokes `cc` and `ar` directly with fixed arguments
and fails if either tool or required attribute is unavailable; it never invokes
a shell or silently drops the candidate. Cross builds, non-GNU environments,
and other architectures deterministically compile the safe scalar path and a
typed unavailable result without invoking a C toolchain. `--no-default-features`
is the required scalar-only build gate on every host. The build script records
rerun inputs and emits its archive only below Cargo's target directory. No
generated object or binary is committed.

### Dispatch

The public selector distinguishes a request from the backend actually used:

```text
BackendRequest = Auto | Scalar | Avx2
BackendKind    = Scalar | Avx2
```

`Auto` selects AVX2 only after Rust runtime detection succeeds; otherwise it
uses scalar. Forced `Avx2` returns `backend_unavailable` and never silently
falls back. A pure selector accepts an injected capability set for tests.
Tests may remove a real capability but may never enable one absent from the
host. C rechecks AVX2 defensively. AVX-512, AMX, ARM NEON, and ARM SVE are
future extension points, not compiled stubs or support claims.

Backend selection is an adapter-v2 operation only. Adapter v1 always retains
its existing safe-Rust f32 `linear` path, never invokes or reports a BF16/AVX2
backend, and remains byte-for-byte and golden-output compatible. Adapter v2
model construction accepts an explicit backend request and reports the selected
expert backend; its ordinary constructor uses `Auto`. A forced BF16 backend
request for adapter v1 is a typed `backend_incompatible_with_adapter` error,
not a silent representation or execution change. Kernel errors and nonfinite
outputs are detected before `SequenceState` is committed. Concurrent calls
share no mutable native state.

### Numerical gate

Correctness separates representation, kernel order, and model semantics:

1. Exhaustive BF16 decoding covers all 65,536 bit patterns. Finite widening is
   bit-exact; signed zeros are preserved; nonfinite encodings are classified
   and rejected before compute. Conversion tests cover ties-to-even, sign
   symmetry, subnormals, exponent carry, maximum finite values, and finite
   inputs that round to infinity.
2. Scalar and AVX2 GEMV are each compared with an independent f64 accumulator
   over exactly widened BF16 inputs. With `u = 2^-24`,
   `gamma(k) = k*u/(1-k*u)`, and `sum_abs = sum(|w[i]*x[i]|)`, each component
   must satisfy
   `abs_error <= gamma(2*columns)*sum_abs + 1e-7`. The raw error and bound are
   retained.
3. AVX2-versus-scalar absolute/relative differences, worst index, and ULP
   distance are retained as diagnostics. Reordered accumulation can legitimately
   differ sharply from ascending scalar order on ill-conditioned cancellation,
   so there is no universal fixed cross-backend tolerance beyond both paths'
   f64 forward-error gates. Exact hand-computable cases and row-distinct f64
   references prevent that bound from hiding indexing or transpose errors.
4. Tiny v2 scalar and AVX2 execution require exact routes, expert IDs, greedy
   tokens, and deterministic repetition. Logits and route weights use the
   existing `atol = 1e-5`, `rtol = 1e-4` model gate against the independently
   extended Python/PyTorch oracle. Tiny v1 identity and goldens remain exact.

Random differential shapes cover every tail from columns 1 through 33, the
255/256/257 and 4095/4096/4097 boundaries, asymmetric orientations, and rows
1, 2, 3, 7, 8, and 17. Weight pointers cover every two-byte offset modulo 32;
f32 input/output pointers cover every four-byte offset modulo 32. Row-distinct
values, immutable-input snapshots, output canaries, finite extremes,
subnormals, cancellation, overflow, and typed validation errors detect
transpose, overread, overwrite, and partial-publication failures.

A standalone C harness calls the ABI directly under AddressSanitizer and
UndefinedBehaviorSanitizer. It compiles with warnings as errors, strict
prototype/conversion/shadow diagnostics, no sanitizer recovery, and a fresh
tmpfs target. It covers nulls, length mismatches, zero/overflowing dimensions,
natural misalignment, overlap, all tails, canaries, and repeated concurrent
calls. Unsupported hosts exercise typed dispatch, never an illegal
instruction.

### Preregistered performance evidence

The named baseline is `rust-scalar-bf16-gemv-v1`; the candidate is
`c-avx2-bf16-gemv-v1`. Both consume the identical compact matrix and f32 input,
perform the same number of calls, use one thread, and produce a validated
output. This comparison isolates the dispatched kernel, not BF16-versus-f32
representation error.

Five fixed cells each process approximately 134.2 million element-products per
observation so call overhead and working-set effects remain visible rather
than calibrated away:

| Cell | Rows | Columns | Calls/observation | BF16 matrix bytes | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| tail | 257 | 513 | 1,019 | 263,682 | descriptive |
| L2 | 512 | 512 | 512 | 524,288 | descriptive |
| LLC boundary | 2,048 | 2,048 | 32 | 8,388,608 | descriptive |
| streaming expand | 8,192 | 2,048 | 8 | 33,554,432 | primary |
| streaming contract | 2,048 | 8,192 | 8 | 33,554,432 | primary |

The five canonical fixture case IDs are `tail-257x513`, `l2-512x512`,
`llc-2048x2048`, `stream-expand-8192x2048`, and
`stream-contract-2048x8192`. The thirteen canonical cell IDs and their fixed
pre-shuffle order are:

| Cell ID | Fixture case | Comparison |
| --- | --- | --- |
| `avx-natural-tail` | `tail-257x513` | scalar/AVX2, natural |
| `avx-natural-l2` | `l2-512x512` | scalar/AVX2, natural |
| `avx-natural-llc` | `llc-2048x2048` | scalar/AVX2, natural |
| `avx-natural-stream-expand` | `stream-expand-8192x2048` | scalar/AVX2, natural |
| `avx-natural-stream-contract` | `stream-contract-2048x8192` | scalar/AVX2, natural |
| `avx-offset-tail` | `tail-257x513` | scalar/AVX2, vector-offset |
| `avx-offset-stream-expand` | `stream-expand-8192x2048` | scalar/AVX2, vector-offset |
| `avx-offset-stream-contract` | `stream-contract-2048x8192` | scalar/AVX2, vector-offset |
| `avx-two-worker-stream-expand` | `stream-expand-8192x2048` | scalar/AVX2, two worker |
| `avx-two-worker-stream-contract` | `stream-contract-2048x8192` | scalar/AVX2, two worker |
| `staged-natural-llc` | `llc-2048x2048` | scalar BF16/staged f32 |
| `staged-natural-stream-expand` | `stream-expand-8192x2048` | scalar BF16/staged f32 |
| `staged-natural-stream-contract` | `stream-contract-2048x8192` | scalar BF16/staged f32 |

The primary buffers use their naturally aligned owned allocation and record
each pointer modulo 64; both variants receive the same addresses. Tail and
both streaming cells also run with weights offset two bytes and input/output
offset four bytes as descriptive unaligned cases. Tiny 12-by-8 and 8-by-12
calls are correctness and overhead diagnostics, never a throughput basis.

The performance fixture is closed before measurement. SHA-256 counter streams
use the domains `runnel-m4-weight-v1\0` and `runnel-m4-input-v1\0`, followed by
a two-byte little-endian case-ID length, its ASCII bytes, and a little-endian
u64 block index. Each digest byte `d` maps to the signed integer
`q = i16(d) - 128`. Weight source values are `q / 128`, converted by the frozen
BF16 ties-to-even routine; input values are `q / 64` as f32, with element zero
set to exactly 1. Both distributions therefore contain signed normal values
and zero, but no NaN, infinity, or subnormal. Every value is an exact binary
fraction. Special-value behavior belongs to the untimed correctness corpus.
The canonical little-endian matrix/input bytes and their SHA-256 digests are
retained per case.

Before timing, the harness generates and hashes the case, prepares the compact
matrix, finite input, caller output, and reusable workspace, touches every page
after affinity is set, and passes the f64 correctness gate. One timed invocation
uses the same shipped `PreparedGemv::run` API as the runtime. The interval
includes the backend match, C status/structural checks for the candidate,
compute into workspace, output-finiteness scan, and the identical final copy
for both variants. Allocation, fixture generation, input validation, first
touch, hashing, logging, and output comparison remain outside. A monotonic raw
clock surrounds only the fixed call batch. The call-indexed timed sink retains
evidence from every repeated call; after the clock stops, both that sink and
the final output buffer are consumed and validated.

The exact experiment has thirteen paired comparison cells:

- five one-thread scalar/AVX2 cells at the table's natural allocations;
- three one-thread scalar/AVX2 vector-offset cells for tail, streaming expand,
  and streaming contract;
- two natural-allocation, two-worker scalar/AVX2 cells for the streaming
  orientations; and
- three natural-allocation safe-Rust scalar-BF16/staged-f32 cells for the LLC,
  streaming-expand, and streaming-contract shapes.

Each cell child completes five unmeasured warmup pairs in the fixed order
baseline/candidate, candidate/baseline, baseline/candidate,
candidate/baseline, baseline/candidate, followed by 30 measured pairs.
The raw grid is exactly `13 * 2 * 30 = 780` timing rows. Fifteen pairs use
baseline then candidate and fifteen candidate then baseline. One
SHA-256 counter stream hashes `"runnel-m4-cell-order-v1\0" || u64_le(counter)`
to shuffle the table's thirteen-cell order. A second hashes
`"runnel-m4-pair-order-v1\0" || u16_le(cell_id_length) || cell_id ||
u64_le(counter)` to shuffle an initial vector of fifteen baseline/candidate
pairs followed by fifteen candidate/baseline pairs. Every counter starts at
zero; each digest is consumed as four little-endian u64 words. Fisher-Yates
visits indices from `length - 1` down through 1, draws `j` in `0..=index`, and
uses rejection below `floor(2^64 / (index + 1)) * (index + 1)` before taking
the remainder. The two streams independently determine cell and measured-pair
orders; warmups are not shuffled. Child sequence, pair sequence, and order are
raw fields.
Failures, interruptions, timeouts, page faults, context switches, and CPU-time
deltas remain raw; no outlier is removed.

All 30 measured pairs (60 variant rows) in a cell must complete successfully
before that cell receives a paired interval or any interval-based statement.
The fixed 780-row grid may contain explicit failure-status rows, but a cell
with any unsuccessful row is marked incomplete: it reports status counts only,
emits no inferential interval, and cannot satisfy either a cell-specific or
general claim rule. Failed rows and their successful partners are never
silently discarded or replaced.

Repeated calls use identical mathematical inputs, so final-output consumption
alone is not an anti-elision proof for the inlinable Rust reference. Every call
passes its prepared input/output through `std::hint::black_box`, completes the
same status and output barrier in both variants, and folds one call-indexed
output word into a batch sink inside the timed interval. The sink and exact
executed-call counter are validated after timing. Barrier and sink work are
identical, retained in both elapsed times, and covered by a regression test
that detects a missing call.

The kernel has no internal worker. Single-thread children are restricted to the
lowest allowed CPU. For the two-worker diagnostic, the harness selects the two
lowest allowed CPUs with distinct `(physical_package_id, core_id)` topology
and restricts the child to exactly that pair. Before the start barrier, each
worker binds itself to one different selected CPU, verifies its affinity mask
is that singleton, and records `sched_getcpu()` before and after its timed
batch; any bind or residency mismatch fails the row. Each worker has a separate
output while sharing immutable weights. If two physical cores are unavailable,
the diagnostic is explicitly unsupported rather than oversubscribed. Worker
creation is excluded and synchronization is included. The result is aggregate
two-GEMV throughput, never single-GEMV latency. SMT sibling relationships are
recorded. The host has one NUMA node, so M4 makes no multi-NUMA result.

The primary statistic is the paired candidate/baseline elapsed-time ratio,
lower being better; each paired difference is `candidate_ns - baseline_ns`.
For baseline elapsed time, candidate elapsed time, the 30 paired ratios, and
the 30 paired differences separately, each complete cell reports count, mean,
sample standard deviation (denominator `n - 1`), minimum, p50/median, p95, and
maximum. The p50 is the mean of the two central sorted values for even `n`; p95
is nearest-rank `ceil(0.95 * n) - 1` in zero-based indexing with no
interpolation. Only the ratio median receives a deterministic 10,000-resample
95% percentile-bootstrap interval. Each
bootstrap replicate resamples the 30 complete paired ratios—not the 60 variant
rows—with replacement and takes the ordinary sample median (the mean of the
two central values for this even sample size). Its random u64 stream is
`SHA-256("runnel-m4-bootstrap-v1\0" || u16_le(cell_id_length) || cell_id ||
u64_le(counter))`, consumed as four little-endian u64 words per digest. Index
draws use rejection below `floor(2^64 / 30) * 30` and then remainder 30. After
sorting the 10,000 replicate medians, the nearest-rank endpoints are zero-based
indices 249 and 9,749. Logical element-products/s and BF16 bytes/s are
diagnostic and are explicitly not hardware bandwidth. Cells are not pooled,
intervals are unadjusted exploratory per-cell descriptions, and there is no
omnibus inference.

An exploratory cell-specific statement of lower elapsed time is allowed only
if all correctness gates pass and that cell's unadjusted ratio interval lies
wholly below one; it must retain the cell and multiplicity caveat. The only
preregistered general claim rule is the conjunction of both natural one-thread
streaming orientations. It requires both observed median ratios to be at most
0.95 and both interval upper bounds below one. The 0.95 rule describes the
observed medians; it is not a confidence bound for a minimum 5% effect.
Favorable timing is not an M4 acceptance condition. Ambiguous or negative
results are committed and no other cell is substituted post hoc.

A secondary safe-Rust strategy diagnostic compares repeated scalar BF16
on-the-fly calls with one timed BF16-to-f32 widening pass into preallocated
scratch followed by the same number of ascending-order scalar f32 GEMVs. It
keeps staging time inside the strategy interval. The f32 scratch is twice the
BF16 source size, so simultaneous source-plus-scratch peak is three times BF16;
a persistent preexpanded-only strategy would retain twice BF16 after discarding
the source. It introduces no second native ABI and cannot support the primary
AVX2-versus-scalar claim.

### Evidence custody and resource bounds

The schema-v1 evidence record freezes the implementation commit, harness and
binary hashes, compiler commands/flags, case identities, tolerance formulas,
warmups, repetitions, seeds, order, clocks, timeouts, primary/secondary roles,
and claim rules. It retains `environment.json`, `experiment.json`,
`cases.jsonl`, `correctness.jsonl`, every timing row in `observations.jsonl`,
`summary.json`, and summary-sourced SVGs. Verification uses closed schemas and
file sets, exact grids and joins, bounded duplicate-key JSON parsing, and
byte-regenerates summaries and figures.

The clean-commit harness performs a two-job, locked/offline, nonincremental
release build in a private mode-0700 tmpfs child and rechecks the historical
harness blob and executable before publication. Primary, vector-offset, and
staging children inherit one-CPU affinity; two-worker children inherit their
recorded two-physical-core set. Locale and timezone are controlled. On x86 the
harness records MXCSR before and after every kernel-calling thread, including
each individually pinned two-worker thread, requires round-to-nearest with FTZ
and DAZ clear, and rejects any change without modifying the register.
Allowlisted metadata records CPU model/features, cpuset and affinity,
cache/SMT/NUMA topology, governor/frequency availability, toolchains and flags,
memory/tmpfs reserve, load average, and pre/post process resource counters. It
never changes turbo, governor, firewall, host cache state, or floating-point
mode.

Live benchmark memory is capped at 256 MiB, evidence at 16 MiB, child stdout
and stderr at 1 MiB each, one cell child at 180 seconds, and full capture at
15 minutes. Capture requires at least 2 GiB free on its tmpfs build filesystem
after the reserve. No build or measurement uses the constrained root
filesystem, downloads a model, or creates a multi-gigabyte artifact.

## Consequences

M4 adds one auditable unsafe island and one compact expert path rather than a
general tensor library. Adapter v1 remains the exact reference. Adapter v2 can
prove compact storage, numerical parity, dispatch, and sanitized native code,
but it does not establish support for arbitrary models, MXFP4, BF16 arithmetic,
threaded GEMM, other ISAs, or multi-NUMA execution. Timings characterize only
the named synthetic cells on the recorded host.

## Primary sources

- [NVIDIA CUDA Math API: `__nv_bfloat16`](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/struct____nv__bfloat16.html)
  documents the 1-sign, 8-exponent, 7-significand-bit storage and
  round-to-nearest-even conversions. It is consulted as a numeric-format
  specification; no CUDA code is used.
- [Rust `is_x86_feature_detected!`](https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html)
  documents runtime x86 feature detection.
- [GCC x86 function attributes](https://gcc.gnu.org/onlinedocs/gcc-15.2.0/gcc/x86-Function-Attributes.html)
  document function-local target options and the caller's responsibility to
  dispatch only to a supported ISA.
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
  is the instruction/intrinsic semantics reference.
- Nicholas J. Higham, [*Accuracy and Stability of Numerical Algorithms*,
  Chapter 3](https://epubs.siam.org/doi/10.1137/1.9780898718027.ch3),
  is the numerical-analysis reference for `gamma(k)` and inner-product
  forward-error bounds.
- [Rust Reference: external blocks](https://doc.rust-lang.org/stable/reference/items/external-blocks.html)
  documents the C ABI and the unsafe declaration/call obligation.
