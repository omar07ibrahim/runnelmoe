# Benchmark and experiment contract

Performance is a result, not a project adjective. No number becomes a
RunnelMoE claim unless its raw observations, environment, command, correctness
gate, and analysis are committed together.

## Required experiment record

Each experiment directory contains:

- `experiment.json`: schema version, UTC time, git commit and dirty flag,
  hypothesis, named baseline, candidate, exact command, fixture/trace digest,
  warmup count, measured repetitions, seed schedule, timeout, correctness
  tolerance, and primary/secondary metrics;
- `environment.json`: OS/kernel, architecture, CPU model/count/flags, memory,
  filesystem, storage device class when safely discoverable, compiler and
  runtime versions, build profile/features, relevant environment settings, and
  benchmark harness version. Capture only an explicit non-secret allowlist;
  never dump the process environment, credential files, usernames, or absolute
  private paths. Commands use repository-relative paths and redact unrelated
  host arguments;
- `observations.jsonl`: one unaggregated record per repetition;
- `summary.json`: generated statistics and correctness outcome; and
- an optional generated figure plus a script that regenerates it.

Raw observations are append-only evidence. A correction creates a new
experiment ID and explains why the earlier run is invalid; it does not silently
replace favorable or unfavorable samples.

## Procedure

1. State a falsifiable hypothesis and the expected mechanism.
2. Name one authoritative baseline. For kernels this is scalar; for I/O it is
   synchronous exact reads; for cache studies it is no-cache and byte-aware
   LRU; for scheduling it is deterministic single-request FIFO.
3. Pin the fixture or trace by SHA-256 and record every tunable parameter.
4. Run the correctness gate first. Logits use a declared absolute and relative
   tolerance; routing and greedy tokens require exact equality.
5. Warm until initialization and page-fault effects are intentionally included
   or excluded, and say which. Warmups are never samples.
6. Interleave or randomize baseline/candidate order when host drift could bias
   results. Record the order.
7. Collect repeated trials. The default microbenchmark minimum is 30 samples;
   longer end-to-end runs may use fewer only with a documented duration and
   uncertainty rationale.
8. Preserve failures and timeouts in raw data.
9. Generate the summary from raw data with committed code.
10. Interpret effect size, dispersion, confounders, and negative results—not
    only the best sample.

## Statistics

Report sample count, median, arithmetic mean when meaningful, standard
deviation, p50/p95, minimum, and maximum. The primary comparison reports a
paired ratio or difference when trials are paired. A deterministic seeded
bootstrap provides a 95% confidence interval; the method and resample count
are recorded. Do not claim an improvement when the interval or known host
noise makes direction ambiguous.

Throughput uses total completed work over elapsed wall time, not the mean of
per-item rates. Latency percentiles are calculated over individual operations
or requests and identify which population they represent.

## Metric definitions

### Runtime

- **time to first token (TTFT):** request admission to first emitted token;
- **inter-token latency:** consecutive emission timestamps for one sequence;
- **prefill throughput:** admitted prompt tokens divided by prefill wall time;
- **decode throughput:** emitted decode tokens divided by decode wall time;
- **bytes per token:** physical storage bytes successfully returned to the
  runtime divided by completed model tokens;
- **I/O wait:** wall duration work is ready but blocked on required reads;
- **peak accounted bytes:** maximum runtime budget ledger total;
- **peak RSS:** maximum sampled process resident-set size, reported separately
  because allocators and code pages are outside the logical ledger.

### Cache

- **demand hit:** a demand lookup finds an already verified resident page;
- **byte hit ratio:** demand bytes served resident / all demand bytes;
- **admission:** a verified page becomes resident by policy decision;
- **eviction:** resident eligibility is withdrawn to reclaim capacity;
- **useful prefetch:** a prefetched page receives a later demand hit before
  eviction;
- **wasted prefetch:** a prefetched page is evicted or the trace ends without
  demand;
- **prefetch coverage:** demand misses avoided by useful prefetch / all demand
  accesses;
- **online gap:** online-policy physical demand bytes divided by offline
  fixed-page Belady bytes, with the equal-size-page assumption explicit.

Object-count hit rates may be diagnostic but never substitute for byte rates.

### Scheduling

- **goodput:** requests completing within their declared deadline per second;
- **fairness:** Jain's index over per-request normalized service plus maximum
  service lag; both are reported because one aggregate can hide starvation;
- **queue delay:** admission enqueue to first scheduled model work;
- **cancellation cleanup:** cancellation to release of all request-owned
  accounted bytes.

## Host controls and limitations

Shared virtualized hosts are noisy. Record load average, CPU frequency/governor
when available, NUMA topology, affinity, and major page faults. Prefer a fixed
CPU set and bounded build concurrency. Do not flush system caches or change
host-wide settings without explicit authorization. Storage “cold” results must
use project-owned files and a documented method; otherwise call them warm or
uncontrolled.

This AWS bootstrap host has four vCPUs on an AMD EPYC 7R13 virtual machine,
one NUMA node, AVX2, and about 30 GiB total RAM. Free memory and disk are
volatile and must be captured per run. The host is suitable for tiny
correctness and controlled microbenchmarks, not frontier-model performance
claims.

## Review checklist

- Does the raw artifact identify a clean commit and exact command?
- Did numerical/token parity pass before timing?
- Are the baseline and hypothesis named?
- Are warmups, repetitions, trial order, failures, and uncertainty visible?
- Does every chart derive only from committed raw rows?
- Are units, aggregation population, and smaller/bigger direction clear?
- Are targets, estimates, and external results excluded from project results?
