# M4 review record

- Milestone: compact-BF16 adapter and isolated AVX2 expert GEMV
- Review date: 2026-08-03 UTC
- Scope: BF16 representation, adapter-v2 loading and execution, safe scalar
  reference, C ABI and AVX2 dispatch, numerical/model differentials,
  sanitizers and portability, evidence capture, statistics, and claims
- Local verdict: pass
- Publication verdict: pass; the evidence commit passed every job in
  [public CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30817508436),
  and this closure commit is subject to the same checks before merge
- Implementation and measured commit:
  `035d217baf0901809fa02bf0a5c11c1a490198c2`
- Evidence publication commit:
  `ae69481def4b320ff619090ce7793ef3d65ace33`
- Accepted raw evidence:
  [`m4-bf16-gemv-20260803`](../../benchmarks/raw/m4-bf16-gemv-20260803/experiment.json)
- Correctness verdict: pass; 13 kernel and three complete-model checks
- Performance verdict: preregistered rule satisfied for the two exact primary
  synthetic cells on the recorded host

## Acceptance evidence

Adapter version 2 preserves the tiny-v1 topology and formula while storing
only its twelve routed-expert gate, up, and down matrices as little-endian
BF16. The generated tensor object is 5,600 bytes rather than v1's 7,904 bytes.
The formula happens to be exactly BF16-representable, so this demonstrates
representation custody and execution parity rather than quantization quality
on a lossy model.

`runnel-kernels` retains compact validated `u16` weights, exposes an
independently callable ascending-column safe-Rust GEMV, and places one AVX2
implementation behind a fixed C ABI. The safe wrapper validates sizes,
alignment, nonoverlap, finite input, runtime capability, native status, and
finite output before publication. Forced AVX2 fails with a typed unavailable
error when unsupported; `Auto` may select it only after runtime detection.
Adapter v1 remains on its existing f32 path, and scalar-only/cross builds do
not produce the native object.

The numerical suite exhausts all 65,536 BF16 words and covers rounding ties,
subnormals, finite extremes, all vector tails through 33 columns,
255/256/257 and 4095/4096/4097 boundaries, asymmetric shapes, pointer offsets,
canaries, overlap, overflow, partial publication, unsupported dispatch, and
concurrent direct-ABI calls. Scalar and AVX2 outputs are independently bounded
against an f64 decoded-BF16 accumulator. A direct C harness runs under ASan and
UBSan with warnings and sanitizer recovery disabled.

The complete-model ledger preserves tiny-v1 custody and checks both forced
scalar and forced AVX2 tiny-v2 runs against separately generated PyTorch
goldens. Selected experts and greedy tokens are exact across all positions and
two cache-backed generation repetitions. The largest observed tolerance ratios
were 0.014520 for logits, 0.008523 for router scores, and 0.002309 for route
weights, all below the acceptance limit of one.

## Local and public verification

The reviewed source and record passed these gates:

```console
python3 scripts/verify_repository.py
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo test -p runnel-kernels --all-targets --no-default-features --locked
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s oracle/tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts.tests.test_run_m4_experiment -v
python3 scripts/run_m4_experiment.py verify \
  --input benchmarks/raw/m4-bf16-gemv-20260803 --check
```

The evidence binary's 20 protocol/unit tests and all 31 M4 Python custody,
schema, scheduling, anti-elision, failure-retention, and statistics tests
passed. The implementation commit passed the full Rust, oracle, sanitizer,
scalar-only, AArch64 cross-target, rustdoc, and repository-contract suite in
[public CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30816592087).
The evidence commit repeated those jobs and added archival byte regeneration
of the accepted M4 record.

## Committed experiment

The standard-library capture harness required the named implementation commit
to equal clean `HEAD`. It checked the historical harness blob, parsed the
closed native compiler and archiver command vectors from that commit, created
a private mode-0700 target below a verified tmpfs root, and performed a
two-job, locked, offline, nonincremental release build. The caller could not
supply either executable. Absolute private build paths were replaced with
stable placeholders in the allowlisted record.

Correctness ran before timing. Each cell child then flushed five excluded
warmup events and sixty measured variant rows; the parent preserved the longest
valid prefix on any failure and could only synthesize explicit missing-grid
rows. The completed capture has five case rows, 16 correctness rows, and 780
timing rows: thirteen cells times 30 complete pairs times two variants. Every
timing row is successful, has its exact call count, retains stable output,
call-indexed sink, address, affinity, residency, and MXCSR joins, and records
per-row resource evidence.

The canonical commands are:

```console
python3 scripts/run_m4_experiment.py capture \
  --build-root /dev/shm \
  --output benchmarks/raw/m4-bf16-gemv-20260803 \
  --commit 035d217baf0901809fa02bf0a5c11c1a490198c2

python3 scripts/run_m4_experiment.py verify \
  --input benchmarks/raw/m4-bf16-gemv-20260803 --check
```

The verifier rejects the wrong file set, symlinks, non-regular or oversized
files, duplicate JSON keys, non-finite numbers, schema/grid/order/join drift,
failed proof custody, and summary or figure drift. It returned thirteen
complete cells, 780 observation rows, successful correctness, and a satisfied
general rule. The eight files total 770,816 bytes.

| Committed evidence file | SHA-256 |
| --- | --- |
| `environment.json` | `1023bfb306da42ecb05a9ba9ee092bcf1373bd923d78109854fd11c45e290520` |
| `experiment.json` | `f8abb19bf97331b56f8a0c10415b87859f2c9522fc104bcfcf85b242c7354a8e` |
| `cases.jsonl` | `a4a8df26f9c9578faec7ccaf6f85140003b417960cef1892d6274204838c6448` |
| `correctness.jsonl` | `f8dcb1a3b175e15de64fc6ae38a2428f636abf7ed9e7353e48940c53f8dcea87` |
| `observations.jsonl` | `ba1e0d04fb5007beaf68be6daa6161ac3f39cf557d6f1fc07ab1f5146d2e66c1` |
| `summary.json` | `c54d806ef0e9d9c9d7fa2490feaa6c58a0ab5191846831dc34c4dbd7d165d3f9` |
| `figures/elapsed-time.svg` | `73c64b6dd5245ccef66b7e013c50a49c32c93dcbc823eef824df31a3e678b7aa` |
| `figures/paired-ratios.svg` | `83e15f74c4215664d1a868c01971dae1b7f6af906b91d089bac391ccfb2f50a5` |

The recorded harness hash is
`93f2ef3155f604b30bbc9b5667f920ff961a72272a18e6d7f7743eb590c4a52e`.
The kernel and model executable hashes are respectively
`41e9d12a502f66bb2c48261f73a0077ab2553992f0d98ccdb6f424b824767976`
and `7d5d3ab23434a24d6400e474255ef32725be4462f4e709e352e71d483c232633`.

## Result interpretation

The primary comparison is the per-pair candidate/scalar elapsed-time ratio,
lower being better. Each cell has 30 complete pairs with fifteen of each
execution order. No observation was removed. The reported interval is a
deterministic 10,000-resample percentile bootstrap over the 30 paired ratios;
cells are not pooled and the per-cell intervals are unadjusted.

| Preregistered primary cell | Median ratio | Unadjusted 95% interval | Median elapsed reduction |
| --- | ---: | ---: | ---: |
| natural one-thread streaming expand, `8,192 x 2,048` | 0.176712 | `[0.175869, 0.178408]` | 82.33% |
| natural one-thread streaming contract, `2,048 x 8,192` | 0.183190 | `[0.176564, 0.188793]` | 81.68% |

Both medians are at most 0.95 and both upper bounds are below one. With every
correctness gate passing, this satisfies the preregistered conjunction for
these exact cells and host. The 0.95 threshold applies to observed medians; it
is not a confidence-bounded minimum 5% effect.

The expand ratios had standard deviation 0.00768 and range
`[0.16848, 0.20801]`; contract had standard deviation 0.01556 and range
`[0.16045, 0.21944]`. Baseline-first and candidate-first medians differed
slightly, especially for contract, but both order strata remained far below
one. The balanced schedule keeps both orders in the frozen all-row estimate;
no separate causal order-effect claim is made.

The secondary staged-f32 strategy includes one BF16-to-f32 widening pass plus
the scalar f32 calls. Its candidate/scalar-BF16 median ratios were 1.008619
`[1.006956, 1.010100]` at LLC, 1.043614
`[1.041938, 1.046765]` for streaming expand, and 1.058083
`[1.055485, 1.071087]` for streaming contract. Staging did not help in this
capture and requires three times the BF16 source footprint while both source
and scratch coexist. This is a secondary negative diagnostic, not a universal
storage recommendation.

Two-worker cells retained singleton worker affinity on two distinct physical
cores and completed successfully. They measure aggregate two-GEMV throughput,
not single-GEMV latency, and are noisier than the one-worker cells. Offset
effects changed direction across shapes, so no general alignment benefit or
penalty is claimed. Logical BF16 bytes/s is a workload rate, not measured
hardware bandwidth.

The recorded host had four logical AMD EPYC 7R13 CPUs, two physical cores, one
NUMA node, AVX2, and no exposed AVX-512. Its one-, five-, and fifteen-minute
load averages were 3.88, 3.76, and 11.28; swap was nearly exhausted and CPU
frequency/governor readings were unavailable. Timed rows had zero major page
faults but 8,067 involuntary context switches in aggregate, with a maximum of
310 in one row. The formal interval describes within-run resampling, not
host-to-host or run-to-run uncertainty.

The two generated SVGs derive only from `summary.json`; archival verification
recreates their bytes from raw observations. Source inspection found valid
finite SVG content, accessible titles/descriptions, no external payload, and
no private path. The JSON summary remains authoritative.

## Independent bounded reviews

Separate source, native-boundary, harness, statistics, custody, and
documentation reviews were performed. Findings fixed before capture included
moving prepared dispatch and two-worker setup outside timing, retaining
post-start partial evidence, prefix-preserving child output, retaining f64
proofs when cross-scalar diagnostics are unavailable, exact model-generation
joins, CPU/topology and MXCSR custody, stable sink/address/output joins, one
absolute capture deadline, and compiler/archiver argv verification before the
private build.

Final source re-audit reported no open P0/P1 issue. The statistics reviewer
independently rebuilt all 390 measured pairs, every descriptive statistic,
and all thirteen deterministic 10,000-resample intervals with exact agreement.
The custody reviewer independently reconciled the historical commit and
harness, compiler/archiver identities and argv, schemas and cardinalities,
fixture digests, balanced order, every correctness/timing row, CPU/MXCSR and
anti-elision proofs, and byte regeneration. The only pre-merge integration gap
was adding the eight files and M4 archival verification to the repository
contract and CI; evidence commit `ae69481…` closed it and passed publicly.

## Clean-room and claim audit

The BF16 representation, C kernel, Rust wrapper, adapter, fixtures, oracle
extension, tests, harness, figures, prose, and measurements were implemented
from this project's preregistered specification and cited numeric-format,
compiler, ISA, Rust FFI, and numerical-analysis sources. No implementation,
prose, layout, fixture, artwork, or result from the credited Kimi K3 C
prior-art repository was reused, and no Kimi checkpoint was acquired.

The measured claim is limited to two named synthetic GEMV cells on the
recorded host. It is not described as a 5–9x model or inference speedup. M4
measures no TTFT, decode throughput, storage latency, I/O overlap, production
traffic, or hardware bandwidth, and it supports no general CPU/compiler/model
ranking.

## Residual boundaries

- Tiny-v2's formula is exactly BF16-representable; this is not evidence for
  accuracy after lossy conversion of trained weights.
- The optimized operation is dense row-major BF16-by-f32 GEMV. It is not a
  fused expert MLP, GEMM, BF16 arithmetic engine, or general tensor library.
- Timing covers fixed synthetic repeated batches on one busy shared AWS VM.
  Its bootstrap interval does not capture rerun or hardware variation.
- AVX2 is the only native candidate. AVX-512, AMX, ARM NEON/SVE, other ABIs,
  and multi-NUMA execution are unsupported.
- Two-worker results are aggregate throughput diagnostics; offset cells are
  exploratory and no alignment effect is inferred.
- Adapter v2 reconstructs its tiny authenticated object before execution. M4
  does not establish per-token out-of-core expert streaming.
- The evidence branch must be integrated with a merge commit, not squashed or
  rebased, so historical commit `035d217…` remains reachable to archival CI.
