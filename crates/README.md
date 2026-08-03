# Runtime crates

The M1 Rust workspace contains four executable boundaries:

- `runnel-format`: strict canonical RMOA parsing and byte verification;
- `runnel-fixture`: formula-derived tiny artifact generation;
- `runnel-runtime`: scalar tiny-adapter inference and generation; and
- `runnel`: black-box fixture, generate, and self-contained demo commands.

Storage/cache, scheduling, kernels, and serving arrive only with later vertical
milestones, never as empty interface scaffolding.
