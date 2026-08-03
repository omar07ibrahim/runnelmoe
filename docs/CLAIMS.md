# Claim ledger

Every externally visible claim has one of four states:

- **target** — intended behavior that has not passed its milestone gate;
- **verified** — reproduced from a clean worktree by a committed command;
- **measured** — supported by committed raw results and environment metadata;
- **unsupported** — explicitly not claimed.

| Claim | State | Evidence |
| --- | --- | --- |
| Required provenance and the clean-room boundary are documented | target | M0 commit, green CI, and durable independent-review record |
| The project can run inference | target | M1 |
| Tiny runtime and PyTorch oracle agree | target | M1 |
| Reads are bounded by a configured RAM budget | target | M2 |
| Sync and async paths are numerically equivalent | target | M2 |
| Any cache policy improves a named baseline | target | M3; must have raw evidence |
| AVX2 is faster than scalar on the measured host | target | M4; no speedup assumed |
| Multi-request scheduling is fair or faster | target | M5; metrics to be defined |
| OpenAI-compatible HTTP/SSE subset is available | target | M6 |
| Kimi K3 checkpoint execution is supported | unsupported | Optional future adapter; no checkpoint acquired |
| AVX-512, AMX, ARM NEON, or ARM SVE is supported | unsupported | No implementation or measurement |
| Frontier-scale inference is practical on a laptop | unsupported | Outside the project evidence |

Changes from target to verified or measured require the exact reproduction
command, raw artifact path, commit, and reviewer sign-off in the same change.
