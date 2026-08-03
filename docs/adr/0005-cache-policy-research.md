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
emits a prediction before the target route is revealed. Online policy APIs
never receive a trace suffix. The fill model is
`instant-between-events-v1`: a successful prefetch is resident before the next
event. It models traffic and cache pollution, not storage latency, overlap,
TTFT, or throughput.

All input limits are explicit. Validation rejects unknown or duplicate fields,
noncanonical lines, sequence gaps, invalid page geometry, unknown references,
noncausal signals, excessive prediction lists, and arithmetic overflow before
replay begins.

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

Primary comparisons use paired trace seeds. Each cell reports count, mean,
median, sample deviation, p50, p95, minimum, maximum, and a deterministic
10,000-resample 95% percentile bootstrap interval for paired ratios and
differences. These intervals describe variability across generated workloads,
not simulator timing. An effect is called improved or regressed only when its
complete interval is respectively below or above one; otherwise it is
inconclusive. Families are not pooled into a universal result.

The primary online-to-optimal ratio is total physical bytes divided by uniform
Bélády demand-fill bytes. Non-prefetch admission policies additionally report
demand-fill gap. A prefetch policy's demand-fill ratio is only a diagnostic,
because prefetch can move the same I/O out of the demand path. Useful, wasted,
redundant, and dropped speculative bytes are all reported independently.

Expanded stochastic traces are regenerated from committed generator
definitions instead of stored. A trace ledger records seeds, parameters, and
digests; raw observations, environment, experiment contract, summary, and SVGs
are append-only evidence. The complete M3 evidence directory is capped at
16 MiB. CI runs a bounded smoke matrix; the full matrix is produced from one
clean commit.

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
- Router-aware motivation comes from the high-level observations in
  [EdgeMoE](https://arxiv.org/abs/2308.14352v2) and
  [MoE-Infinity](https://arxiv.org/abs/2401.14361v3). No code or reported
  performance number is reused.
