# Claim ledger

Every externally visible claim has one of four states:

- **target** — intended behavior that has not passed its milestone gate;
- **verified** — reproduced from a clean worktree by a committed command;
- **measured** — supported by committed raw results and environment metadata;
- **unsupported** — explicitly not claimed.

| Claim | State | Evidence |
| --- | --- | --- |
| Required provenance and the clean-room boundary are documented | verified | baseline `fe2fd0ce…`, [green CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30788257437), [M0 review](reviews/M0_REVIEW.md) |
| The project can run tiny synthetic inference | verified | implementation `9d3454aa…`, [green main CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30790520786), [M1 review](reviews/M1_REVIEW.md) |
| Tiny runtime and independent PyTorch oracle agree | verified | all-position differential test at `9d3454aa…`, [green main CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30790520786), [M1 review](reviews/M1_REVIEW.md) |
| Cache-managed verified page payloads obey the configured M2 capacity ledger | verified | implementation `99772585…`, [green CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30797378731), [M2 review](reviews/M2_REVIEW.md), [30 schema-v2 trials](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/observations.jsonl); allocator metadata and total process memory are outside this ledger and RSS is reported separately |
| Synchronous and asynchronous readers return identical authenticated bytes on the deterministic multi-page fixture | verified | differential tests at `99772585…`, fixed 16-event trace in the [accepted experiment](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/experiment.json), [green CI](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30797378731) |
| Cache-backed tiny generation exactly equals synchronous generation through controlled eviction and authenticated reload | verified | complete `Generation` equality at `99772585…`, 30/30 forced-eviction parity passes in the [schema-v2 summary](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/summary.json), [M2 review](reviews/M2_REVIEW.md) |
| The tiny M2 validation records one 7,904-byte application-cache-cold payload load for four subsequently generated tokens (1,976 bytes/token) | measured | exact fixed-command ratio in the [experiment](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/experiment.json) and [raw trials](../benchmarks/raw/m2-data-plane-forced-eviction-20260803/observations.jsonl); counts descriptor-backend payload bytes, excludes metadata/synchronous comparator/forced-eviction path, and is not block-device traffic, steady-state decode, expert streaming, or a performance comparison |
| Any cache policy improves a named baseline | target | M3; must have raw evidence |
| AVX2 is faster than scalar on the measured host | target | M4; no speedup assumed |
| Multi-request scheduling is fair or faster | target | M5; metrics to be defined |
| OpenAI-compatible HTTP/SSE subset is available | target | M6 |
| Kimi K3 checkpoint execution is supported | unsupported | Optional future adapter; no checkpoint acquired |
| AVX-512, AMX, ARM NEON, or ARM SVE is supported | unsupported | No implementation or measurement |
| Frontier-scale inference is practical on a laptop | unsupported | Outside the project evidence |

Changes from target to verified or measured require the exact reproduction
command, raw artifact path, commit, and reviewer sign-off in the same change.
