# `runnel-store`

`runnel-store` is RunnelMoE's verified out-of-core data plane. It keeps RMOA
object payloads out of the eager artifact representation and exposes only
SHA-256-authenticated logical pages.

The crate contains four deliberately separate boundaries:

- `ArtifactSource` and `StoredArtifact` retain descriptor-relative filesystem
  handles and authenticate bounded manifests and page tables;
- `Cas` imports trusted artifact IDs through private resumable stages, verifies
  bytes before no-replace publication, commits manifests last, and performs
  fail-closed mark/sweep collection;
- `SyncReader` is the authoritative bounded positional-read implementation,
  while `AsyncReader` dispatches that same implementation through a fixed
  worker set and bounded queue; and
- `PageCache` single-flights loads and manages byte-accounted `loading`,
  `resident`, and `retiring` states with explicit consumer leases.

The page-pool ledger rounds requested payload capacity to an explicit 64-byte
quantum. This is a capacity-accounting rule, not a pointer-alignment claim. Entry,
waiter, lease, queue, trace, manifest, and page-table metadata have independent
hard count/byte ceilings; process RSS is sampled separately because allocator
and runtime overhead are not inferred from logical payload bytes.

This is a local single-user storage boundary, not a hostile same-UID sandbox.
Artifact authenticity still depends on obtaining the expected root digest
through a trusted channel. See
[ADR-0004](../../docs/adr/0004-verified-data-plane.md) and the
[threat model](../../docs/THREAT_MODEL.md).
