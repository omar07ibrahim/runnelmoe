# Runtime crates

The M2 Rust workspace contains five executable boundaries:

- `runnel-format`: strict canonical RMOA parsing and byte verification;
- `runnel-fixture`: formula-derived tiny artifact generation;
- `runnel-runtime`: scalar tiny-adapter inference and generation; and
- `runnel-store`: descriptor-safe CAS ingestion, authenticated bounded page
  reads, fixed-worker asynchronous dispatch, and a byte-accounted page cache;
  and
- `runnel`: black-box fixture, generation, numerical-demo, and data-plane-demo
  commands.

Scheduling, optimized kernels, and serving arrive only with later vertical
milestones, never as empty interface scaffolding. The storage API is documented
in [`runnel-store`](runnel-store/README.md).
