# Roadmap and acceptance gates

Milestones are vertical and close only after implementation, tests, relevant
measurements, an independent bounded review, documentation, and a green atomic
push. A checkbox is evidence-bearing only when its linked command or artifact
exists.

## M0 — contracts and provenance

- [x] All required governance and design documents pass
  `python3 scripts/verify_repository.py`.
- [x] The project name has a dated availability/conflict check.
- [x] A public repository exists under `omar07ibrahim`; default branch is
  protected by green CI practice.
- [x] A reviewer can identify every external source and distinguish targets
  from project measurements.

## M1 — exact tiny reference runtime

- [x] A deterministic generator emits a tiny hash-verified checkpoint.
- [x] Scalar Rust covers tokenization, causal state, top-k routing, expert
  dispatch, logits, greedy generation, and malformed-input rejection.
- [x] A separately organized PyTorch oracle agrees on routes, logits
  (declared tolerance), and generated tokens.
- [x] Committed source fixtures and offline Rust tests pass from a clean
  checkout. Evidence: [M1 review](reviews/M1_REVIEW.md) and
  [green main CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30790520786).

## M2 — out-of-core data plane

- [x] Immutable SHA-256 objects support resumable staging and atomic publish.
- [x] Sync and async backends preserve parity under bounded reads.
- [x] A byte-capacity DRAM cache exposes hit, miss, admission, eviction,
  prefetch-usefulness, read-byte, wait-time, and RSS observations.
- [x] Truncation, corruption, reordering, cancellation, and low-budget fault
  tests fail safely.

Evidence: [M2 review](reviews/M2_REVIEW.md),
[green protected CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30797378731),
and the committed schema-v2
[experiment contract](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/experiment.json),
[raw trials](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/observations.jsonl),
and [generated summary](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/summary.json).

## M3 — cache policy research

- [x] Trace replay implements byte-aware LRU, SLRU, TinyLFU, router-aware
  admission and prefetch, and offline Bélády/MIN.
- [x] Synthetic trace definitions, seeds, raw JSONL, analysis code, repeated
  trials, uncertainty, and online-to-optimal gaps are committed.
- [x] Prefetch usefulness is reported separately from demand hits.

Evidence: [M3 review](reviews/M3_REVIEW.md),
[green protected CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30804817207),
the schema-v2
[experiment contract](../benchmarks/raw/m3-cache-policies-20260803/experiment.json),
[trace ledger](../benchmarks/raw/m3-cache-policies-20260803/traces.jsonl),
[3,240 raw observations](../benchmarks/raw/m3-cache-policies-20260803/observations.jsonl),
and [generated summary](../benchmarks/raw/m3-cache-policies-20260803/summary.json).
Results are exploratory synthetic modeled-byte comparisons, not
runtime-performance evidence.

## M4 — kernels and portability

- [x] Tiny adapter v2 retains compact BF16 expert matrices; a narrow C ABI
  provides runtime-dispatched AVX2 while safe Rust scalar remains independently
  testable and tiny v1 remains unchanged.
- [x] Sanitizer and differential tests cover alignment, tails, special values,
  compact representation decoding, and unsupported ISA fallback.
- [x] Append-only evidence retains paired warmups/repetitions, uncertainty,
  machine metadata, raw data, and token/logit parity; any speedup claim meets
  the preregistered cell-specific threshold.

The accepted architecture, numerical bounds, sanitizer gate, fixed benchmark
cells, and no-speedup fallback are frozen in
[ADR-0006](adr/0006-bf16-avx2-expert-kernel.md). A favorable timing is not
required to close M4.

Evidence: [M4 review](reviews/M4_REVIEW.md),
[green evidence CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30817508436),
the accepted [experiment contract](../benchmarks/raw/m4-bf16-gemv-20260803/experiment.json),
[five case rows](../benchmarks/raw/m4-bf16-gemv-20260803/cases.jsonl),
[16 correctness rows](../benchmarks/raw/m4-bf16-gemv-20260803/correctness.jsonl),
[780 timing rows](../benchmarks/raw/m4-bf16-gemv-20260803/observations.jsonl),
and [generated summary](../benchmarks/raw/m4-bf16-gemv-20260803/summary.json).
All thirteen cells completed 30 pairs and the two preregistered primary cells
met the frozen general rule. Timing remains specific to fixed synthetic GEMV
batches on the recorded shared host; favorable timing was not required to
close the implementation, correctness, and custody gates.

## M5 — state and multi-request scheduling

- [x] The transaction boundary, paged-state layout, deterministic DRR policy,
  sampling stream, memory ledger, KPI hierarchy, fixed workload matrix, and
  conservative claim rules are preregistered in
  [ADR-0007](adr/0007-transactional-paged-scheduling.md).
- [ ] Chunked prefill and bounded paged state preserve single-request parity.
- [x] Continuous batching, expert coalescing, resident deterministic
  preemption, cancellation, backpressure, and seeded sampling pass stress
  tests.
- [x] The sealed harness-owned run observer is allocation-bounded before
  release, reconciles exact internal timing milestones with service and ledger
  evidence, and is behavior-neutral under both policies and feature boundaries.
- [ ] The ordered closed-schema 26-row correctness producer and hostile-input
  verifier pass before any timing capture begins.
- [ ] Evidence reports TTFT, prefill/decode throughput, p50/p95, fairness, and
  observed memory ceilings.

M5 timing is evidence, not an acceptance condition. The hard gates are exact
per-request semantics, the frozen numerical tolerance, transaction rollback,
complete logical-ledger identities, bounded queues, deterministic sampling and
trace replay, cancellation/deadline cleanup, and maximum service lag/runnable
gap. Tiny adapter v3 extends the generated fixture to 1,024 positions solely
to exercise multi-page state and streaming attention; it is not a large-model
or natural-language performance claim.

## M6 — production surface

- [ ] A documented OpenAI-compatible subset supports JSON and SSE, greedy and
  seeded top-k/top-p/temperature sampling, deadlines, cancellation, logs, and
  Prometheus metrics.
- [ ] The default listener is loopback-only and black-box failure/load tests
  pass.

## M7 — audited release

- [ ] The release path integrates scheduler expert work with authenticated M2
  cache leases under one fixed budget; eager tiny weights are not presented as
  end-to-end out-of-core execution.
- [ ] Original generated-from-source diagrams, identity, docs site, benchmark
  dashboard, runnable demo, report, and interview walkthrough are complete.
- [ ] Clean-clone verification and all CI checks pass; links and claims are
  audited; no copied material is found by an independent reviewer.
- [ ] Repository metadata, issues/milestones, tag, and release are public.
