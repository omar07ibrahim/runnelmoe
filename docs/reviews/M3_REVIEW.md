# M3 review record

- Milestone: audited cache-policy research
- Review date: 2026-08-03 UTC
- Scope: trace format and generator, online policy semantics, router-aware
  admission and prefetch, fixed-page and variable-byte oracles, byte-ledger
  metrics, Python differentials, evidence capture, statistics, and figures
- Local verdict: pass
- Publication verdict: pass; the evidence commit passed all required checks in
  [protected CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30804817207),
  and this closure commit is subject to the same checks before merge
- Implementation and measured commit:
  `fcbaaebb7e211a228ee75d9cab713acc2b9890ac`
- Evidence publication commit:
  `700de98110db58e05f5db24d29bb14f8a482cffc`
- Accepted raw evidence:
  [`m3-cache-policies-20260803`](../../benchmarks/raw/m3-cache-policies-20260803/experiment.json)
- Traffic verdict: mixed exploratory cell outcomes; no general winner
- Performance verdict: no runtime improvement claimed

## Acceptance evidence

The simulator consumes a strict, bounded, policy-neutral JSONL trace. It keeps
page metadata in an immutable catalog and exposes online policies only the
current event and catalog, so a policy cannot inspect the future event suffix.
The parser rejects duplicate keys, non-finite or out-of-range numbers,
malformed UTF-8, unexpected fields, missing identities, sequence errors,
oversized records, non-regular inputs, symlinks, growing files, and trailing
data. Retained descriptors and explicit byte limits close replacement and FIFO
input races.

The frozen online set is byte-capacity LRU, SLRU with a bounded protected
segment, TinyLFU admission with a bounded count-min sketch and doorkeeper,
causal router-aware admission, and causal router-aware grouped prefetch.
Bélády/MIN is the exact oracle for the uniform 65,536-byte page suite. A
separate dynamic program provides an exact variable-byte oracle on bounded
small traces. Router signals are keyed by request, target step, and layer;
scores, page lists, policy metadata, and full decisions are part of stable
digests.

The six generator families are stationary harmonic popularity, scan
pollution, phase shifts, cyclic pressure, clustered Markov routing, and IID
uniform routing. Each replicate uses a domain-separated SHA-256 seed and an
integer-only xoshiro256** generator. After 512 unrecorded burn-in routes, each
trace contains 4,096 measured top-2 routes. Every route expands to the three
ordered pages of both selected experts. The evidence grid is:

```text
6 families * 30 replicates = 180 trace ledgers
180 traces * 3 capacities * 6 policies = 3,240 raw observations
6 families * 3 capacities * 6 policies = 108 summary cells
```

The independent Python implementation covers the online policies without
calling Rust. Rust tests compare it event by event, compare both optimal
oracles with brute-force enumeration, and cross-check LRU against the real M2
`PageCache`. Prefetch accounting keeps ordinary hits, useful-prefetch hits,
physical useful/wasted loads, and redundant/dropped no-read offers distinct.

## Local verification

The reviewed implementation and record passed:

```console
python3 scripts/verify_repository.py
cargo fmt --all -- --check
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --locked --offline -- -D warnings
CARGO_BUILD_JOBS=2 cargo test --workspace --all-targets --locked --offline
RUSTDOCFLAGS="-D warnings" CARGO_BUILD_JOBS=2 cargo doc --workspace --no-deps --locked --offline
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest oracle.tests.test_cache_policy -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts.tests.test_run_m3_experiment -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -v
cargo run --locked --offline -p runnel-sim --bin runnel-cache-sim -- matrix --family markov_clusters --replicate 0 --measured-steps 64
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_m3_experiment.py verify --input benchmarks/raw/m3-cache-policies-20260803 --check
```

Results: all 214 workspace Rust tests passed, including 64 `runnel-sim` unit
tests and its integration targets; 13 independent Python cache-policy tests
and all 55 script tests, including 33 M3 evidence-harness tests, passed.
Formatting, Clippy, rustdoc, the repository contract, the deterministic matrix
smoke test, and archival byte regeneration passed.
The implementation commit also passed the complete PyTorch oracle job in
[protected CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30803118703).

## Committed experiment

The standard-library harness required the named implementation commit to be a
clean full `HEAD`. It verified the historical harness blob, created a private
mode-0700 directory on tmpfs, and performed a two-job, locked, offline,
nonincremental release build. A caller could not provide the executable. The
harness independently validated and discarded each expanded generated trace,
reconstructed its 4,096 measured routes, ran the complete policy matrix, and
checked the harness and binary hashes again before atomic publication.

The canonical reproduction command is explicit about the effective step
count. It is canonical rather than a claim about an unrecorded top-level shell
history; the artifact records the expanded build and simulator commands:

```console
python3 scripts/run_m3_experiment.py capture \
  --build-root /dev/shm \
  --output benchmarks/raw/m3-cache-policies-20260803 \
  --commit fcbaaebb7e211a228ee75d9cab713acc2b9890ac \
  --measured-steps 4096

python3 scripts/run_m3_experiment.py verify \
  --input benchmarks/raw/m3-cache-policies-20260803 --check
```

The read-only archival verifier rejects the wrong file set, symlinks,
non-regular files, size-limit violations, duplicate JSON keys, non-finite
numbers, broken joins, accounting errors, and summary or figure drift. It
returned 180 trace rows, 3,240 observation rows, and 5,893,502 total directory
bytes.

| Committed evidence file | SHA-256 |
| --- | --- |
| `environment.json` | `b0fbecf2c9cc9645b99a131e47e835ba948b86adce15985a12f2358a53989576` |
| `experiment.json` | `46c379a2991faaa9f7ed9311a7e5c54fb88755beeb8451bb846f68eff840bbff` |
| `traces.jsonl` | `8947fb0484f6ad1efe1568008a41abdb3252fc04e26609dce43da2662114ae82` |
| `observations.jsonl` | `e5b562c0dc046a7ece270aac1abbe8a762ae9097a374bf1b780f7f337e805a72` |
| `summary.json` | `2d4e9186f76605696799e062ebada248c8b956e44b13651916beee437ab5f0a4` |
| `figures/optimal-gap.svg` | `e6d4fc430d60ae5047ce877540219fac53ef464e82ef505e78d355fcb7b83055` |
| `figures/paired-change.svg` | `f42dc2891c31d3060b744d3921554fcde7b6e241305c4c73f1aaa95802712813` |
| `figures/prefetch-accounting.svg` | `1aafc3880dc39400e7a1671794984e494f09f1cc5b8a3eba58063719943337d1` |

The recorded harness and release binary hashes are respectively
`8be498257907a0565b774d0f2916449f709e484dbea53d2a569a7f6e29691beb`
and
`6dcce852d9407a0f2277a6866f4503fe3a01010508787046558175d01e7b6fe2`.

## Result interpretation

For each online-candidate family/capacity cell, the primary comparison is the
median of 30 replicate-paired total-physical-byte ratios to LRU. The reported
95% percentile-bootstrap intervals use 10,000 deterministic resamples. They
are exploratory, unadjusted per-cell descriptions: families are not pooled,
there is no multiplicity correction, and the preregistered contract forbids an
omnibus conclusion.

| Policy | Interval below 1 | Overlaps 1 | Above 1 |
| --- | ---: | ---: | ---: |
| SLRU | 7 | 6 | 5 |
| TinyLFU | 11 | 2 | 5 |
| Router admit | 7 | 6 | 5 |
| Router prefetch | 4 | 7 | 7 |
| **Total** | **29** | **21** | **22** |

Direction changes materially with the workload and capacity. At four MiB,
SLRU used 0.68218 times LRU's traffic under scan pollution
`[0.68186, 0.68279]`, but 1.64648 times LRU under phase shift
`[1.60742, 1.66211]`. TinyLFU's phase-shift ratio at the same capacity was
2.80273 `[2.75911, 2.84570]`. Conversely, at eight MiB under cyclic pressure,
TinyLFU's traffic was 31.150% below LRU and 1.024740 times the oracle. These
selected post-hoc contrasts expose sensitivity; they do not select a general
winner.

Across the 90 LRU-plus-candidate cells, median total physical traffic divided
by exact uniform-page Bélády/MIN ranged from 1.0247395833 to 2.9245923913.
The oracle scope is fixed-capacity, uniform-page admission. The gap is not a
latency or production optimality claim.

Router prefetch shows why demand misses cannot stand in for physical traffic.
At two and four MiB under scan pollution, it reduced demand-fill bytes while
total traffic rose 19.754% and 13.087% versus LRU. Under cyclic pressure it
eliminated demand fills but shifted the same 1,536 MiB into prefetch loads. All
540 prefetch observations satisfy
`prefetch_load_bytes = prefetch_useful_bytes + prefetch_wasted_bytes`; every
admitted prefetch is classified by trace end, and dropped offers perform no
read. Independently summarized component medians need not add exactly.

The figures are generated directly from `summary.json`; verification
recreates their bytes from raw observations. Source inspection confirmed valid
XML, finite coordinates, unique identifiers, accessible titles/descriptions,
legends, units, and explicit exploratory/no-timing annotations. No external
payload or private path appears in an SVG.

## Independent bounded reviews

Separate implementation, runtime-boundary, security, statistics, evidence,
and documentation reviews were performed. Findings fixed before capture
included ambiguous router-signal identity, missing signal metadata in trace
digests, descriptor replacement and FIFO handling, TinyLFU counter overflow,
future-event availability at the online-policy API, unbounded Python record
handling, duplicate ordinal semantics, lazy sketch validation, insufficient
subprocess bounds, weak generated-route independence, and summary acceptance
without chart byte regeneration.

Final code re-audits reported no open P0/P1 finding. The statistics reviewer
reconciled every raw identity and independently recomputed 3,672 summary
objects, including all 432 bootstrap intervals, with zero mismatches. The
integrity reviewer reproduced the release binary byte-for-byte, checked all
3,240 rows and decision digests, and independently regenerated 90 traces
covering stationary, scan, and phase workloads: 368,640 measured routes and
1,620 observations matched exactly. The slower remaining families were not
independently replayed end to end; all of their raw rows, joins, digests, and
accounting identities were still exhaustively reconciled.

## Clean-room and claim audit

M3 source, policy semantics, generator, fixtures, prose, figures, tests, and
measurements were created from this project's preregistered specification and
cited primary algorithm/PRNG sources. No implementation, prose, layout,
fixture, artwork, or benchmark result from the credited Kimi K3 C prior-art
repository was reused, and no Kimi checkpoint was acquired.

The measured claim is limited to deterministic synthetic modeled bytes.
Neither simulator execution time nor host I/O latency was measured. No policy
is described as generally superior, and the production M2 LRU cache is not
replaced by a research policy.

## Residual boundaries

- The six families are synthetic and openly regenerated; they are not traces
  from a deployed model or workload.
- Bélády/MIN is exact for the uniform-page suite. The separately tested
  variable-byte oracle is not used to label the primary matrix optimum.
- `instant-between-events-v1` models I/O volume and pollution, not latency,
  bandwidth, queueing, overlap, or prefetch lead time.
- The 512 burn-in routes are not emitted in JSONL. Their full-route digest is
  simulator-attested; every measured route was independently reconstructed.
- Policy metadata uses a normalized scalar ledger. It excludes allocator
  overhead, executable memory, and RSS.
- Capture-time free-memory, free-space, and cleanliness fields are historical
  snapshots enforced by the hashed harness rather than facts observable after
  publication.
- The evidence branch must be integrated with a merge commit, not squashed or
  rebased, so the implementation commit remains reachable to future archival
  CI checks.
- Runtime policy integration and measurement belong to later milestones; the
  independently tested M2 production cache remains unchanged.
