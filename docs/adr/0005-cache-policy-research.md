# ADR 0005: independent byte-accounted cache-policy research

- Status: accepted
- Date: 2026-08-03
- Milestone: M3

## Context

The production data plane introduced in M2 deliberately implements one small,
auditable byte-aware LRU. Its events describe decisions already made by that
cache, so those events are not an independent workload trace and cannot fairly
compare replacement policies. M3 needs a causal trace model, policy-independent
replay, a supported offline optimum, and evidence that cannot hide speculative
I/O behind a demand-hit counter.

This work is an independent implementation from published algorithm
descriptions. No implementation from the Kimi K3 C prior-art repository, or
from another cache simulator, is used.

## Decision

### Boundary and trace model

Add a safe Rust crate and CLI named `runnel-sim` and `runnel-cache-sim`. The
simulator does not depend on `runnel-store`; an integration test instead checks
that both independently produce the same LRU counters on a shared M2 trace.

Input is bounded canonical JSON Lines. A closed header is followed by a sorted
page catalog and a dense event stream. Page descriptors separate:

- `logical_bytes`, the physical bytes charged to a successful fill; and
- `charge_bytes`, the bytes occupying the simulated page pool.

Events are demand accesses or router signals. A router signal contains only
integer fixed-point scores and targets a later request step. The generator
emits a prediction before the target route is revealed. Online policy state
receives the current event plus an immutable catalog-only view; its type cannot
expose the event suffix. A separate accounting prepass derives one metadata
ceiling scalar from the validated event stream before constructing policy
state. That scalar is used only to guard and report normalized metadata, never
to rank, admit, prefetch, or evict a page. The fill model is
`instant-between-events-v1`: a successful prefetch is resident before the next
event. It models traffic and cache pollution, not storage latency, overlap,
TTFT, or throughput.

All input limits are explicit. Validation rejects unknown or duplicate fields,
noncanonical lines, sequence gaps, invalid page geometry, unknown references,
noncausal signals, excessive prediction lists, and arithmetic overflow before
replay begins. File input is opened once with no-follow and nonblocking flags,
classified through the retained descriptor, and read through a `limit + 1`
ceiling. A size change is rejected. This closes leaf replacement and FIFO
blocking races; ancestor symlinks, hard links, and equal-length same-inode
mutation remain outside the filesystem-isolation claim.

### Policies

The named online baseline is byte-capacity LRU. M3 also implements:

- SLRU with a preregistered 75% protected byte target. New fills enter
  probation; a probation demand hit promotes the page; protected overflow
  demotes its LRU page before capacity eviction.
- TinyLFU admission over byte LRU. A deterministic four-row count-min sketch
  and doorkeeper observe demands only. If multiple byte-sized victims are
  required, the project-specific extension compares candidate frequency
  density with aggregate victim density using checked integer arithmetic.
  Equality preserves incumbents.
- `router-admit`, a project-original predictive SLRU composition that uses
  only active causal router scores to retain likely-near-future expert pages.
- `router-prefetch`, the same admission policy plus bounded expert-group
  prefetch. It sorts scores and identifiers deterministically, admits all
  absent pages of one expert or none, and may displace probation pages but not
  protected pages. Predictions never alter the demand route or model output.

The fixed parameters are a four-by-2,048 TinyLFU sketch, 4-bit-equivalent
saturating counters, deterministic hash seeds, an aging sample of ten estimated
cache entries, router support of at least eight observations, minimum router
score 100,000 ppm, at most two predicted experts, at most six pages, and at
most six page fills per signal. These parameters are frozen before the measured
seeds; favorable measurements are not required for milestone acceptance.

Router state is keyed by request, target step, and layer. A later signal cannot
overwrite another layer or another outstanding target. Signals older than the
current demand step retire, while same-step signals remain live for every page
of that step. The reported router metadata value is a deterministic normalized
payload charge, not allocator telemetry:

```text
active signal payload = 24 + 8 * selected_prediction_count bytes
```

The 24-byte record covers request, target step, layer, and count fields; each
prediction covers its expert and fixed-point score. An instrumentation-only
prepass outside online policy state derives an exact ceiling from all nonempty
post-threshold/top-K signals. Replay receives only that scalar, tracks checked
current and peak payload, and confirms that the complete signal stream consumes
the ceiling. Heap allocator overhead is deliberately excluded and must be
observed through RSS in an integration benchmark rather than guessed here.

Victim plans are complete before resident state changes. Oversized pages are
served and bypassed. Checked counters enforce:

```text
demand_accesses = ordinary_hits + useful_prefetch_hits + demand_misses
demand_logical_bytes = demand_hit_bytes + demand_miss_bytes
total_physical_load_bytes = demand_load_bytes + prefetch_load_bytes
prefetch_load_bytes = useful_prefetch_bytes + wasted_prefetch_bytes
resident_charge_bytes <= capacity_bytes
```

An admitted prefetch becomes useful exactly once, on its first demand. An
unused prefetched page becomes wasted when evicted or at trace finalization.
Redundant and dropped prefetches are separate and cause no physical read.

### Offline references

Bélády/MIN is called optimal only when every page has equal `charge_bytes` and
equal `logical_bytes`. On a miss, the fetched candidate participates in the
farthest-next-use choice, so the oracle can bypass it. Router signals are
ignored. Unequal geometry is a typed unsupported error, never a heuristic
fallback.

Variable-sized caching is not solved by farthest-next-use. A separately bounded
exact dynamic program is used only for tiny differential tests and small
examples; larger variable-byte results carry no optimality label.

### Preregistered evidence

The primary suite has 128 synthetic experts, three 65,536-byte pages per
expert, top-2 routing, 4,096 measured tokens after 512 generator burn-in tokens,
and 32-, 64-, and 128-page capacities. Thirty paired seeds are retained for
each of six openly generated families:

1. stationary harmonic popularity;
2. hot traffic interrupted by scans;
3. phase-shifting hot sets;
4. cyclic pressure, labeled pathological;
5. clustered Markov routing; and
6. IID uniform routing, the predictability negative control.

Expert identifiers are seed-permuted so identifier tie-breaking cannot stand
in for popularity. SHA-256 derives every seed and pins each generated trace.
The complete matrix is 180 traces by three capacities by six policies, or
3,240 unaggregated rows.

The seed domain, Xoshiro256** transition, unbiased bounded draw, permutation,
family parameters, predictor saturation and ceil-halving, and route-digest
encoding are part of the closed experiment contract. During capture, an
independent Python parser checks each generated canonical trace, reconstructs
all 4,096 measured top-2 routes, and recomputes their domain-separated digest.
The 512 burn-in routes are not present in the emitted JSONL, so the full-route
digest remains simulator-attested and is labeled as such.

Primary comparisons use paired trace seeds. Each cell reports count, mean,
median, sample deviation, p50, p95, minimum, maximum, and a deterministic
10,000-resample 95% percentile bootstrap interval for paired ratios and
differences. These intervals describe variability across generated workloads,
not simulator timing. They are unadjusted, exploratory per-cell descriptions.
Their position below, across, or above one is recorded only for the four online
candidate policies; LRU is the baseline and Bélády is the oracle, so both are
not applicable. Families are not pooled and no “at least one cell improves” or
other omnibus conclusion is permitted.

The primary online-to-optimal ratio is total physical bytes divided by uniform
Bélády demand-fill bytes. Non-prefetch admission policies additionally report
demand-fill gap. A prefetch policy's demand-fill ratio is only a diagnostic,
because prefetch can move the same I/O out of the demand path. Useful, wasted,
redundant, and dropped speculative bytes are all reported independently.

Expanded stochastic traces are regenerated from committed generator
definitions instead of stored. A trace ledger records seeds, parameters, and
digests; raw observations, environment, experiment contract, summary, and SVGs
are append-only evidence. The complete M3 evidence directory is capped at
16 MiB. The capture harness performs a fresh locked, offline release build in
a private tmpfs directory from a clean commit, records the toolchain and closed
build command, and rechecks HEAD, the commit's harness blob, and the binary hash
before publication. Verification opens the exact evidence file set through
retained no-follow descriptors, sums sizes before reads, and regenerates every
summary and figure from bounded raw ledgers. CI runs a bounded smoke matrix;
the full matrix is produced from one clean commit.

### Pre-capture audit amendment

An independent pre-capture review on 2026-08-03 found that the original wording
could be read as an uncorrected existential test across 72 eligible cells. No
primary evidence had been captured. This amendment removes that omnibus claim,
marks baseline/oracle comparisons not applicable, and makes the intervals
explicitly exploratory. It weakens the permitted inference and does not select
a favorable family, capacity, or policy. The same review required self-build
provenance, independent measured-route reconstruction, summary-sourced figures,
and bounded archival reads before the first accepted run.

## Consequences

The simulator can compare policy traffic under a clear model, but it cannot by
itself predict latency or production speedup. Policy metadata is bounded and
reported separately from page-pool bytes. M3 policy findings do not change the
M2 cache until a later integration experiment independently justifies that
complexity.

## Primary sources

- L. A. Bélády, “A Study of Replacement Algorithms for a Virtual-Storage
  Computer,” *IBM Systems Journal* 5(2), 1966,
  [DOI 10.1147/sj.52.0078](https://doi.org/10.1147/sj.52.0078).
- R. Karedla, J. S. Love, and B. G. Wherry, “Caching Strategies to Improve
  Disk System Performance,” *IEEE Computer* 27(3), 1994,
  [DOI 10.1109/2.268884](https://doi.org/10.1109/2.268884).
- G. Einziger, R. Friedman, and B. Manes, “TinyLFU: A Highly Efficient Cache
  Admission Policy,” *ACM Transactions on Storage* 13(4), 2017,
  [arXiv:1512.00727v2](https://arxiv.org/abs/1512.00727v2).
- D. Berger, N. Beckmann, and M. Harchol-Balter, “Practical Bounds on Optimal
  Caching with Variable Object Sizes,” *Proceedings of the ACM on Measurement
  and Analysis of Computing Systems* 2(2), 2018,
  [arXiv:1711.03709](https://arxiv.org/abs/1711.03709).
- D. Blackman and S. Vigna, “Scrambled Linear Pseudorandom Number
  Generators,” *ACM Transactions on Mathematical Software* 47(4), 2021,
  [DOI 10.1145/3460772](https://doi.org/10.1145/3460772). The independently
  written deterministic generator follows the published xoshiro256**
  transition. The authors' official
  [xoshiro256**](https://prng.di.unimi.it/xoshiro256starstar.c) and
  [SplitMix64](https://prng.di.unimi.it/splitmix64.c) reference files were
  consulted for algorithm identity and provenance only; each carries a public
  domain dedication with an unrestricted-use fallback permission. No source
  text was copied.
- Router-aware motivation comes from the high-level observations in
  [EdgeMoE](https://arxiv.org/abs/2308.14352v2) and
  [MoE-Infinity](https://arxiv.org/abs/2401.14361v3). No code or reported
  performance number is reused.
