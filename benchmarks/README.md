# Benchmarks

Every experiment follows [the benchmark contract](../docs/BENCHMARKING.md).
Unaggregated machine-readable observations belong in `raw/<experiment-id>/`.
Reports and figures must be reproducible from those observations. This
directory contains no claimed performance improvement.

## M2 correctness and observability evidence

`raw/m2-data-plane-forced-eviction-20260803/` is the accepted schema-v2
clean-commit evidence record for the verified data plane:

- [experiment.json](raw/m2-data-plane-forced-eviction-20260803/experiment.json)
  freezes the hypothesis, fixture and trace digests, release build, exact
  command, and correctness gate;
- [environment.json](raw/m2-data-plane-forced-eviction-20260803/environment.json)
  contains only allowlisted host and toolchain metadata;
- [observations.jsonl](raw/m2-data-plane-forced-eviction-20260803/observations.jsonl)
  preserves all 30 measured trials; and
- [summary.json](raw/m2-data-plane-forced-eviction-20260803/summary.json) is the
  generated statistical and correctness summary.

Every measured trial reproduced both frozen traces and exact full-generation
parity. In the combined numerical gate, the 7,904-byte tiny tensor page is
evicted by a 65,536-byte interference page during tensor assembly and is then
authenticated and reloaded before generation continues. The resulting
31-event trace records 20 hits, three misses, three admissions, two evictions,
239,424 logical demand bytes, and 81,344 physical read bytes. A separate
16-event `[0, 1, 0, 2, 2]` trace verifies full-page churn and a 17-byte tail.

The recorded 1,976 bytes per generated token is a command-scoped,
application-cache-cold measurement: one asynchronous cache-backed 7,904-byte
tensor-object payload load divided by the four subsequently generated tokens.
It counts payload bytes returned by the descriptor backend, not block-device
traffic; operating-system file-cache state was uncontrolled. It excludes
metadata, the synchronous comparator, and the separate forced-eviction path.
The runtime adapter reconstructs complete tensors before compute, so this is
not steady-state decode traffic or per-token expert paging.

The exact recorded invocation was:

```console
python3 scripts/run_m2_experiment.py m2-data-plane-forced-eviction-20260803 --warmups 3 --repetitions 30 --timeout-seconds 30 --bootstrap-seed 20260803 --bootstrap-resamples 10000
```

The earlier schema-v1
[`raw/m2-data-plane-20260803/`](raw/m2-data-plane-20260803/experiment.json)
record was generated from commit
`d7a39055d28049cf6721c4e2b8f599a39f77d7fa`. It remains append-only as a
preliminary record, but it separates ordinary generation parity from the
forced-eviction trace and therefore does not close the stronger M2 gate.

Shared-host wall times characterize these validation commands only and support
no speedup claim. Experiment directories are append-only; use a new ID for
every repetition.

## M3 cache-policy evidence contract

The first accepted M3 directory will contain exactly `environment.json`,
`experiment.json`, `traces.jsonl`, `observations.jsonl`, `summary.json`, and
three SVGs below `figures/`. The schema-v2 harness fresh-builds the simulator
from a named clean commit in private tmpfs, independently validates each
generated measured route, retains all 3,240 unaggregated policy rows, and
derives both JSON summaries and chart series from those rows. Verification
uses bounded retained descriptors and checks the harness blob from the recorded
commit rather than assuming the current working copy is identical.

The 10,000-resample intervals are unadjusted, exploratory descriptions within
each family/capacity/policy cell. LRU is the baseline, Bélády/MIN is the
uniform-page oracle, and neither receives an online-candidate outcome. No M3
traffic result or policy recommendation belongs in this README until the raw
directory is committed, independently audited, and linked from the claim
ledger.
