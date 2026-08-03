# RunnelMoE

RunnelMoE is an independent project building a model-agnostic inference runtime
for sparse Mixture-of-Experts models whose expert weights are larger than
available RAM. It is designed to make the hard tradeoffs visible: verified
storage, bounded memory, asynchronous I/O, cache policy, numerical parity,
request scheduling, and honest measurement.

> Project status: design baseline (M0). There are no performance claims yet.
> The planned first supported model is a tiny deterministic synthetic adapter
> created by this project.

The planned runtime's central contract is simple: a configured
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

## Architecture contract

- Rust will own parsing, scheduling, resource accounting, and the safe runtime.
- A narrow C ABI will contain measured SIMD kernels; scalar Rust will stay the
  correctness baseline.
- Python/PyTorch will be used only as an independently structured oracle, fixture
  generator, and analysis environment.
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

Later milestone commands will be added only after they are executable. Raw
benchmark outputs, including environment metadata, will live under
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
