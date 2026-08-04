# RunnelMoE

RunnelMoE is an independent project building a model-agnostic inference runtime
for sparse Mixture-of-Experts models whose expert weights are larger than
available RAM. It is designed to make the hard tradeoffs visible: verified
storage, bounded memory, asynchronous I/O, cache policy, numerical parity,
request scheduling, and honest measurement.

> Project status: M0 design/provenance, the M1 tiny-reference runtime, the M2
> verified out-of-core data plane, M3 cache-policy research, and the M4
> compact-BF16/AVX2 kernel have passed their milestone gates. M3 closes a
> synthetic modeled-traffic gate, not a runtime speedup gate. M4 verifies one
> narrow kernel and tiny adapter, with timing limited to fixed synthetic GEMV
> batches on one recorded host; neither milestone is an end-to-end inference
> speedup claim.

M5 is in progress. Its published synchronous slice now includes paged
transactional decoder state, deterministic sampling, bounded FIFO admission,
equal-weight token-quantum DRR, expert-sorted continuous batches, exact logical
ownership accounting, output backpressure, deadlines, cancellation, and
failure-atomic cleanup. The concurrent actor, full stress/differential matrix,
and preregistered evidence are not complete, so this is not an M5 closure or
performance claim.

The runtime's central contract is simple: a configured
resident-memory ceiling
must remain enforceable while expert tensors move between an immutable
content-addressed object store, a bounded DRAM cache, and compute. Corrupt or
ambiguous artifacts fail closed. Optimized paths remain differential-testable
against an intentionally plain scalar implementation and an independent
Python/PyTorch oracle.

## Why this project exists

Sparse models turn parameter capacity into a data-movement problem. A token
uses only a few experts, but routing, cache admission, prefetch, batching, and
storage latency interact. RunnelMoE is intended to become an experiment
platform and local runtime for studying those interactions under explicit
resource budgets. It is not a claim that commodity CPUs make frontier-scale
models practical.

## Current executable slice

M1 supplies a strict, eager-memory-budgeted RMOA artifact reader; a
formula-generated 7,904-byte synthetic tensor object; scalar Rust tokenization,
two-head causal attention, stable top-2 MoE routing, and greedy generation; an
independently organized vectorized PyTorch oracle; and a black-box CLI. The
fixture contains no trained data or committed weight file.

M2 adds a descriptor-retaining content-addressed store, resumable verified
staging with manifest-last publication, a bounded positional reader, a fixed
asynchronous reader pool, a 64-byte-quantized byte-capacity LRU cache, explicit
leases, fault tests, bounded traces, cache counters, and separate RSS samples.
The tiny adapter integration reconstructs verified tensors before scalar
execution; it does not yet stream expert pages during each token's compute.

The M3 research harness adds strict policy-neutral JSONL traces, byte LRU,
SLRU, TinyLFU admission, causal router-aware admission/prefetch, uniform-page
Bélády/MIN, and a bounded exact variable-byte oracle. It measures modeled
physical traffic and cache pollution; it does not predict storage latency or
runtime throughput. In the accepted exploratory matrix, the four candidates'
72 unadjusted interval positions versus LRU split 29 below one, 21 overlapping
one, and 22 above one. The workload-sensitive
[raw summary](benchmarks/raw/m3-cache-policies-20260803/summary.json) and
[M3 review](docs/reviews/M3_REVIEW.md) support no general winner. The production
M2 cache remains independently testable and is not silently replaced by a
research policy.

M4 adds a 5,600-byte adapter-v2 synthetic object whose twelve routed-expert
matrices remain compact BF16, an independently callable safe-Rust scalar
reference, and one runtime-dispatched C AVX2 GEMV behind a small validated ABI.
Scalar and forced-AVX2 tiny-model paths preserve selected expert IDs and tokens
exactly and pass the declared router-score, route-weight, and logit tolerances
against independently generated PyTorch vectors. On the recorded AMD EPYC
7R13 VM, the two preregistered one-thread streaming cells had AVX2/scalar
paired median batch-time ratios of
0.1767 and 0.1832, with both unadjusted 95% bootstrap intervals below one.
These fixed synthetic results satisfy the preregistered M4 rule; they do not
measure token throughput, serving latency, storage I/O, or other hardware.
See the [raw summary](benchmarks/raw/m4-bf16-gemv-20260803/summary.json) and
[M4 review](docs/reviews/M4_REVIEW.md).

Run the complete artifact-to-token demo after fetching the locked Rust
dependencies once:

```console
cargo run --locked -p runnel -- demo --prompt moe --max-new-tokens 4 --json
```

The stable fixture result is generated IDs `[15, 11, 20, 9]`, decoded as
`"njsh"`. This is a systems-test vector, not language-model-quality evidence.
See [development](docs/DEVELOPMENT.md) for the complete verification suite.

Exercise sync/cached numerical parity and a forced-eviction three-page trace:

```console
cargo run --locked -p runnel -- data-plane-demo --json
```

This command reports exact byte and cache-transition accounting alongside
volatile I/O-wait and RSS observations. It is a correctness/observability demo,
not a throughput comparison. The committed [M2 review](docs/reviews/M2_REVIEW.md)
and [schema-v2 raw evidence](benchmarks/raw/m2-data-plane-forced-eviction-20260803/summary.json)
record the closed gate and its limitations.

Run a small offline M3 policy matrix:

```console
cargo run --locked -p runnel-sim --bin runnel-cache-sim -- \
  matrix --family markov_clusters --replicate 0 --measured-steps 64
```

The JSON contains all six frozen policies at three capacities. This command is
a deterministic functional smoke test, not the 30-seed accepted experiment or
a timing benchmark. Cache-policy semantics are frozen in
[ADR-0005](docs/adr/0005-cache-policy-research.md).

Emit the M4 tiny-model correctness ledger:

```console
cargo run --locked -p runnel --bin runnel-m4-model-check
```

This produces three JSONL correctness records for tiny-v1 preservation and
tiny-v2 scalar/AVX2 execution. It is not a timing benchmark. The accepted
AVX2 record is typed `unsupported` on a host without that ISA. The accepted
performance procedure is frozen in
[ADR-0006](docs/adr/0006-bf16-avx2-expert-kernel.md).

## Architecture contract

- Rust owns parsing, verified storage/cache, the scalar reference runtime,
  dispatch, and the bounded synchronous multi-request scheduler; later slices
  add the concurrent owner and serving.
- A narrow C ABI contains the measured AVX2 GEMV; scalar Rust remains the
  independently callable correctness baseline.
- Python/PyTorch is used only as an independently structured oracle, golden
  vector generator, and analysis environment.
- CI and demos will use tiny generated data. No proprietary or multi-gigabyte
  checkpoint will be required.
- Network listeners, when added, will bind to loopback by default.

See [design](docs/DESIGN.md), [artifact format](docs/FORMAT.md),
[threat model](docs/THREAT_MODEL.md),
[benchmark contract](docs/BENCHMARKING.md), [claim ledger](docs/CLAIMS.md),
and [prior-art record](docs/PRIOR_ART.md).

## Run the repository contract check

Requirements: Python 3.12+ and Git. The M0 check uses only the Python standard
library.

    python3 scripts/verify_repository.py

Executable milestone commands are recorded in
[development](docs/DEVELOPMENT.md). Raw benchmark outputs, including
environment metadata, live under
`benchmarks/raw/`; generated summaries will never be the sole evidence for a
claim.

## Independence

RunnelMoE is not a fork, port, rename, or Kimi implementation. Kimi K3 is one
source of model-system requirements and may receive an optional adapter only
after the model-agnostic runtime is established. No code, prose, fixtures,
artwork, or benchmark numbers from the referenced C project are used here. Its
distinctive directory layout was not used as a template; shared governance and
documentation filenames are conventional or mission-required. The boundary and studied sources are recorded in
[ADR-0001](docs/adr/0001-clean-room-and-system-boundaries.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
