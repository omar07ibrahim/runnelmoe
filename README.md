# RunnelMoE

RunnelMoE is an independent project building a model-agnostic inference runtime
for sparse Mixture-of-Experts models whose expert weights are larger than
available RAM. It is designed to make the hard tradeoffs visible: verified
storage, bounded memory, asynchronous I/O, cache policy, numerical parity,
request scheduling, and honest measurement.

> Project status: M0 design/provenance, the M1 tiny-reference runtime, and the
> M2 verified out-of-core data plane have passed their milestone gates. M3
> cache-policy research is next. No performance improvement is claimed.

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

## Architecture contract

- Rust owns parsing, verified storage/cache, and the scalar reference runtime;
  later milestones add scheduling, optimized kernels, and serving.
- A narrow C ABI will contain measured SIMD kernels; scalar Rust will stay the
  correctness baseline.
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
environment metadata, will live under
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
