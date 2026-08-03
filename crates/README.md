# Runtime crates

The Rust workspace currently contains seven implementation boundaries:

- `runnel-format`: strict canonical RMOA parsing and byte verification;
- `runnel-fixture`: formula-derived tiny artifact generation;
- `runnel-kernels`: compact BF16 storage, the independently callable safe
  scalar GEMV, and the isolated runtime-dispatched native ABI;
- `runnel-runtime`: scalar tiny-adapter inference and generation; and
- `runnel-store`: descriptor-safe CAS ingestion, authenticated bounded page
  reads, fixed-worker asynchronous dispatch, and a byte-accounted page cache;
- `runnel-sim`: independent deterministic cache-policy replay, causal workload
  generation, and offline reference algorithms; and
- `runnel`: black-box fixture, generation, numerical-demo, and data-plane-demo
  commands.

Scheduling and serving arrive only with later vertical milestones, never as
empty interface scaffolding. The storage API is documented in
[`runnel-store`](runnel-store/README.md).
The simulator boundary is documented in [`runnel-sim`](runnel-sim/README.md).
The compact compute boundary is documented in
[`runnel-kernels`](runnel-kernels/README.md).
