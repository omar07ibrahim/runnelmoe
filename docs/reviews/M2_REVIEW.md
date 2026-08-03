# M2 review record

- Milestone: verified out-of-core data plane
- Review date: 2026-08-03 UTC
- Scope: content-addressed storage, safe positional reads, resumable durable
  publication, sync/async I/O, byte-capacity cache, leases, metrics, traces,
  fault handling, tiny-runtime integration, and evidence harness
- Local verdict: pass
- Publication verdict: pass; the evidence commit passed all required checks in
  [protected CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30797378731),
  and this closure commit is subject to the same checks before merge
- Implementation and measured commit:
  `99772585e23d8f1ce3459ba6397d93fd1be0fc8a`
- Evidence publication commit:
  `0fe01325f65e2991b3a683017809a0d98bc2c8a7`
- Accepted raw evidence:
  [`m2-data-plane-forced-eviction-20260803`](../../benchmarks/raw/m2-data-plane-forced-eviction-20260803/experiment.json)
- Performance verdict: no improvement claimed

## Acceptance evidence

The implementation adds a descriptor-retaining content-addressed store.
SHA-256 object, page-table, and manifest bytes are independently verified;
stages are resumable; publication is no-replace and manifest-last; directory
durability is ordered and checked; and collection is lock-scoped and fails
closed on ambiguous mutable state.

The same verified positional-read contract backs synchronous reads and a fixed
asynchronous worker pool. A byte-capacity LRU cache accounts resident,
in-flight, retiring, and page-pool bytes; preserves request-order recency under
out-of-order completions; coalesces loads; maintains explicit leases; and emits
bounded counters and normalized traces. The terminal 17-byte tail page occupies
one 64-byte capacity quantum. The quantum is accounting granularity, not a
pointer-alignment guarantee.

The numerical gate uses one global one-page cache across the tiny tensor object
and the multi-page interference fixture. After the first tiny-model tensor is
copied, a full interference-page demand evicts the tiny page. The next tensor
demand evicts the interference page and authenticates a tiny-page reload; the
remaining tensor demands hit that reload. The complete cache-backed
`Generation`—all step logits, route scores, expert IDs and weights, and
generated tokens—must exactly equal the synchronous result.

## Local verification

The reviewed implementation passed:

```console
python3 scripts/verify_repository.py
cargo fmt --all -- --check
CARGO_BUILD_JOBS=2 cargo clippy --workspace --all-targets --locked --offline -- -D warnings
CARGO_BUILD_JOBS=2 cargo test --workspace --all-targets --locked --offline
RUSTDOCFLAGS="-D warnings" CARGO_BUILD_JOBS=2 cargo doc --workspace --no-deps --locked --offline
python3 -m py_compile scripts/run_m2_experiment.py scripts/tests/test_run_m2_experiment.py scripts/verify_repository.py
python3 -m unittest discover -s scripts/tests -v
cargo run --locked --offline -p runnel -- data-plane-demo --json
```

Results: 154 Rust tests passed, including 108 `runnel-store` tests; 22 evidence
harness tests passed; formatting, Clippy, rustdoc, Python compilation, the
repository contract, and CLI JSON contract passed. The store tests cover
descriptor traversal, substitution, corruption, truncation, page reordering,
checked limits, cancellation and deadlines, disk and page-pool boundaries,
resumable stages, publication ambiguity, collection, queue saturation,
single-flight loads, eviction, retiring leases, and sync/async byte and
full-generation parity.

## Committed experiment

From the clean implementation commit, the standard-library harness built the
release CLI with two jobs, locked dependencies, offline mode, and incremental
compilation disabled. It ran one correctness gate, three excluded warmups, and
30 measured repetitions with this recorded invocation:

```console
python3 scripts/run_m2_experiment.py m2-data-plane-forced-eviction-20260803 --warmups 3 --repetitions 30 --timeout-seconds 30 --bootstrap-seed 20260803 --bootstrap-resamples 10000
```

The append-only record contains the
[experiment contract](../../benchmarks/raw/m2-data-plane-forced-eviction-20260803/experiment.json),
[allowlisted environment](../../benchmarks/raw/m2-data-plane-forced-eviction-20260803/environment.json),
[30 raw observations](../../benchmarks/raw/m2-data-plane-forced-eviction-20260803/observations.jsonl),
and [generated summary](../../benchmarks/raw/m2-data-plane-forced-eviction-20260803/summary.json).

| Committed evidence file | SHA-256 |
| --- | --- |
| `environment.json` | `3e06accbf73cbde65d9b35aaafafd2db19ac09a5afe1fe9110878b6c0d4dd890` |
| `experiment.json` | `31d7f04d66eaf16eba3497dc8934f019eb2d9589256f5c151d36cdd978f0e2d1` |
| `observations.jsonl` | `153769863c49560513ce950f8f1e92ba57da9b912a2951fff5ac4a5df872175e` |
| `summary.json` | `132cacc1c0d379bea0aca837378b1f50306a92dc068080253062b30c25b6ec0d` |

The harness, release binary, gate projection, fixed trace, and forced-eviction
projection hashes are respectively:

- `2e7efa268989dd9c19cc079a355c348fbbcb8b48d1338615899c134398a8def6`;
- `81005d8d772f94d9c3ccc77a3f5332ce62a9d7c03dbb42563a2f28523ea0ba4a`;
- `6613b1a34d4e4ce98bcb6dc877c12377bf9483739a7f84d97bbd9b58c11a35bb`;
- `825fbfa7cdbafec4fabb55faa3b307108c7bb5b082af50e44bcfc9ef5bf3e4f0`;
  and
- `ae6f4ec113520d5ca5e2301cc5f06bee379e841597e05b66e53fc37bae619078`.

All 30 trials reproduced exact ordinary generation parity and exact
full-generation parity through forced eviction. The combined trace has 22
tensor-page demands and one interference-page demand. Its accounting is:

```text
logical demand bytes = 22 * 7,904 + 65,536 = 239,424
physical read bytes  =  2 * 7,904 + 65,536 =  81,344
23 demands           = 20 hits + 3 misses
3 misses             = 3 admissions; 2 replacements = 2 evictions
```

Its 31 normalized events prove the tiny-page eviction, interference-page
admission, interference-page eviction, authenticated tiny-page reload, and 20
subsequent hits. At completion, active loads, in-flight bytes, retiring bytes,
and leases are zero; the reloaded 7,904-byte page occupies 7,936 bytes under
64-byte capacity quantization. All prefetch counters are zero because this is a
demand-only trace.

The separate 16-event three-page sequence `[0, 1, 0, 2, 2]` produced four
misses, four admissions, three evictions, one hit, 196,642 logical demand
bytes, and 196,625 physical read bytes, including the 17-byte tail.

The recorded 1,976 bytes per generated token is defined narrowly: one
asynchronous cache-backed, application-cache-cold 7,904-byte tensor-object
payload load divided by the four subsequently generated tokens. The counter
measures payload bytes returned by the descriptor backend, not block-device
traffic; operating-system file-cache state was uncontrolled. It excludes
metadata, the synchronous comparator, and the separate forced-eviction path.
The adapter reconstructs complete verified tensors before scalar compute and
performs no per-token page I/O, so the ratio is not steady-state decode traffic
or expert streaming.

The earlier schema-v1
[`m2-data-plane-20260803`](../../benchmarks/raw/m2-data-plane-20260803/experiment.json)
record from commit `d7a39055d28049cf6721c4e2b8f599a39f77d7fa` is retained
append-only. It is preliminary evidence because it separates generation parity
from the eviction trace; it is not used to close the combined M2 gate.

## Independent bounded reviews

Separate cache/concurrency, storage/security, code/parity, and
evidence-methodology reviews were performed. Findings fixed before the final
gate included request-order LRU corruption by completion order, queue-full
state and trace divergence, incomplete prefetch-waste accounting, a payload
allocation copy, missing cancellation checkpoints, incomplete hard-link
publication durability, unsafe cleanup after ambiguous publication,
manifest publication before prerequisite directory syncs, unbounded subprocess
capture, incomplete wait/I/O/RSS observations, stale-binary acceptance, summary
acceptance without raw trace identity, and an evidence gate that did not yet
combine full-generation parity with eviction.

The final code and harness re-audits reported no open P0/P1 finding. The
schema-v2 evidence was then independently parsed with duplicate-key,
non-finite-value, trailing-data, schema, provenance, and privacy checks. Both
traces and every accounting equation were revalidated for every row; all
summary statistics and seeded 10,000-resample bootstrap intervals were
recomputed; and all 30 compact CLI stdout byte streams were reconstructed and
matched their recorded sizes and SHA-256 hashes. No mismatch was found.

## Clean-room and claim audit

M2 source, tests, prose, fixtures, traces, metrics, and measurements were
created from this project's specifications and primary platform/library
documentation. No implementation, prose, layout, fixture, artwork, or
benchmark result from the credited Kimi K3 C prior-art repository was reused,
and no Kimi checkpoint was acquired.

Claims are limited to the deterministic synthetic fixtures, exact correctness,
managed-byte accounting, and recorded observability. Volatile timings describe
one shared virtualized host and do not establish throughput, latency, or cache
policy improvement.

## Residual boundaries

- The M2 tiny-runtime adapter reconstructs complete verified tensors before
  scalar compute. Per-token expert-page streaming belongs to later runtime
  integration.
- The logical ledger covers managed page/cache bytes, not allocator metadata,
  executable pages, or all process memory. RSS is sampled separately.
- Same-UID hostile mutation outside retained descriptors is not a supported
  trust boundary; exact filesystem assumptions remain in the threat model.
- The pinned toolchain does not have Miri installed on this host. The M2 crate
  forbids unsafe Rust, and malformed-input, concurrency, and fault tests are
  the current dynamic evidence.
- Shared-host cache state, load, and CPU frequency were not controlled. The
  committed wall-time distribution is diagnostic only.

M3 owns comparative cache-policy replay, demand/prefetch separation, offline
Belady gaps, seeded traces, uncertainty, and any future cache-improvement
claim.
