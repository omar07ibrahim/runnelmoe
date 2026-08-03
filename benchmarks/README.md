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

## M3 cache-policy evidence

`raw/m3-cache-policies-20260803/` is the accepted schema-v2 record for the
synthetic cache-policy study. It contains the frozen
[experiment contract](raw/m3-cache-policies-20260803/experiment.json),
[allowlisted environment](raw/m3-cache-policies-20260803/environment.json),
[180 trace ledgers](raw/m3-cache-policies-20260803/traces.jsonl),
[3,240 unaggregated observations](raw/m3-cache-policies-20260803/observations.jsonl),
and [108-cell generated summary](raw/m3-cache-policies-20260803/summary.json).
The three generated figures show the
[paired change versus LRU](raw/m3-cache-policies-20260803/figures/paired-change.svg),
[four-MiB gap to Bélády/MIN](raw/m3-cache-policies-20260803/figures/optimal-gap.svg),
and [four-MiB prefetch accounting](raw/m3-cache-policies-20260803/figures/prefetch-accounting.svg).
The summary, rather than either four-MiB slice, is authoritative for the full
six-family, three-capacity matrix.

Across the 72 online-candidate family/capacity cells, the unadjusted 95%
percentile-bootstrap interval for the median candidate/LRU total modeled
physical-byte ratio lay below one in 29 cells, overlapped one in 21, and lay
above one in 22. These are exploratory per-cell interval positions over 30
paired seeds, without family pooling, multiplicity adjustment, or an omnibus
conclusion. The 90 LRU-plus-candidate cell medians for total physical bytes
divided by the uniform-page oracle ranged from 1.0247395833 to 2.9245923913;
those post-hoc extrema are descriptions, not a general policy ranking.

All 540 router-prefetch observations keep ordinary demand hits,
useful-prefetch hits, useful and wasted physical-prefetch bytes, and redundant
or dropped no-read offers separate. Every row satisfies
`prefetch_load_bytes = prefetch_useful_bytes + prefetch_wasted_bytes`. The
scan-pollution cells illustrate why this matters: at two and four MiB,
prefetch reduced demand-fill bytes while increasing total modeled physical
bytes. Under cyclic pressure it moved the same traffic earlier rather than
reducing it.

The clean implementation commit is
`fcbaaebb7e211a228ee75d9cab713acc2b9890ac`. The canonical capture and archival
verification commands are:

```console
python3 scripts/run_m3_experiment.py capture \
  --build-root /dev/shm \
  --output benchmarks/raw/m3-cache-policies-20260803 \
  --commit fcbaaebb7e211a228ee75d9cab713acc2b9890ac \
  --measured-steps 4096

python3 scripts/run_m3_experiment.py verify \
  --input benchmarks/raw/m3-cache-policies-20260803 --check
```

The fresh-build harness used a private tmpfs target and a two-job,
locked/offline release build. It independently reconstructed every measured
route, bounded every input and subprocess output, and regenerated the summary
and figures from raw rows. Independent review recomputed all 3,672 statistical
objects, including 432 deterministic 10,000-resample intervals, with no
mismatch. These are deterministic synthetic modeled-traffic results. They
measure no storage latency, overlap, throughput, TTFT, or production-policy
effect.

## M4 compact-BF16 GEMV evidence

`raw/m4-bf16-gemv-20260803/` is the accepted schema-v1 record for the
compact-BF16 adapter and kernel study. It contains the frozen
[experiment contract](raw/m4-bf16-gemv-20260803/experiment.json),
[allowlisted environment](raw/m4-bf16-gemv-20260803/environment.json),
[five fixture cases](raw/m4-bf16-gemv-20260803/cases.jsonl),
[16 correctness checks](raw/m4-bf16-gemv-20260803/correctness.jsonl),
[780 unaggregated timing rows](raw/m4-bf16-gemv-20260803/observations.jsonl),
and [generated summary](raw/m4-bf16-gemv-20260803/summary.json). The generated
figures show [median batch elapsed times](raw/m4-bf16-gemv-20260803/figures/elapsed-time.svg)
and [paired candidate/scalar ratios](raw/m4-bf16-gemv-20260803/figures/paired-ratios.svg).
The summary, not the figures, is authoritative.

All thirteen cells completed 30 paired repetitions without an unsuccessful,
unsupported, censored, or missing timing row. Thirteen kernel checks and three
complete-model checks passed before timing. Tiny-v1 preservation and tiny-v2
scalar/forced-AVX2 paths retained exact selected experts and tokens and passed
the declared router-score, route-weight, and logit tolerances against the
independent goldens.

The preregistered primary result is deliberately narrow:

| Natural one-thread cell | Median paired AVX2/scalar batch-time ratio | Unadjusted 95% paired-bootstrap interval |
| --- | ---: | ---: |
| streaming expand, `8,192 x 2,048` | 0.176712 | `[0.175869, 0.178408]` |
| streaming contract, `2,048 x 8,192` | 0.183190 | `[0.176564, 0.188793]` |

Both observed medians are below the frozen 0.95 threshold and both interval
upper bounds are below one, so the conjunction satisfies the preregistered
general rule. The rule does not provide a confidence-bounded minimum 5%
effect. Cells are not pooled, intervals are unadjusted, and no omnibus result
is permitted.

The secondary safe-Rust staging diagnostic was unfavorable in this capture:
candidate/scalar-BF16 medians were 1.008619 for LLC, 1.043614 for streaming
expand, and 1.058083 for streaming contract, with all three intervals above
one. This includes the widening pass and a simultaneous source-plus-scratch
footprint three times the BF16 source; it is not a universal storage-strategy
result. Offset and two-worker cells remain exploratory diagnostics only.

The clean measured implementation commit is
`035d217baf0901809fa02bf0a5c11c1a490198c2`; the evidence publication commit
is `ae69481def4b320ff619090ce7793ef3d65ace33`. The canonical commands are:

```console
python3 scripts/run_m4_experiment.py capture \
  --build-root /dev/shm \
  --output benchmarks/raw/m4-bf16-gemv-20260803 \
  --commit 035d217baf0901809fa02bf0a5c11c1a490198c2

python3 scripts/run_m4_experiment.py verify \
  --input benchmarks/raw/m4-bf16-gemv-20260803 --check
```

The verifier recreates `summary.json` and both SVGs byte-for-byte from the raw
rows. The recorded four-vCPU shared VM was busy: its one-, five-, and
fifteen-minute load averages were 3.88, 3.76, and 11.28, frequency/governor
data were unavailable, and swap was nearly exhausted. Balanced paired order,
fixed CPU affinity, raw context-switch counts, zero timed major faults, stable
MXCSR, and full-row retention make the run auditable, but the interval
describes within-run paired resampling rather than run-to-run or host-to-host
uncertainty. These measurements are fixed synthetic kernel batches, not
end-to-end inference, token throughput, TTFT, storage I/O, hardware bandwidth,
arbitrary-model, other-ISA, or multi-NUMA evidence.
