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

- [ ] A narrow C ABI provides runtime-dispatched AVX2 on this host while scalar
  remains independently testable.
- [ ] Sanitizer and differential tests cover alignment, tails, special values,
  compact representation decoding, and unsupported ISA fallback.
- [ ] Any speedup claim has warmups, repetitions, dispersion, machine metadata,
  raw data, and token/logit parity.

## M5 — state and multi-request scheduling

- [ ] Chunked prefill and bounded paged state preserve single-request parity.
- [ ] Continuous batching, expert coalescing, deterministic preemption,
  cancellation, backpressure, and seeded sampling pass stress tests.
- [ ] Evidence reports TTFT, prefill/decode throughput, p50/p95, fairness, and
  observed memory ceilings.

## M6 — production surface

- [ ] A documented OpenAI-compatible subset supports JSON and SSE, greedy and
  seeded top-k/top-p/temperature sampling, deadlines, cancellation, logs, and
  Prometheus metrics.
- [ ] The default listener is loopback-only and black-box failure/load tests
  pass.

## M7 — audited release

- [ ] Original generated-from-source diagrams, identity, docs site, benchmark
  dashboard, runnable demo, report, and interview walkthrough are complete.
- [ ] Clean-clone verification and all CI checks pass; links and claims are
  audited; no copied material is found by an independent reviewer.
- [ ] Repository metadata, issues/milestones, tag, and release are public.
