# ADR-0002: Immutable tensor objects with a bounded manifest

- Status: accepted
- Date: 2026-08-03
- Deciders: Omar Ibrahim

## Context

Out-of-core inference must identify many independently readable tensors,
detect corruption before compute, avoid unbounded metadata allocation, and
publish interrupted downloads without exposing partial state.

## Decision

Use a directory artifact called RMOA (RunnelMoE Object Artifact), version 1:

- `manifest.json` is canonical UTF-8 JSON with a schema version, adapter
  identifier, model dimensions, tokenizer declaration, and ordered tensor
  descriptors. The artifact ID is the SHA-256 of these exact bytes and is
  supplied or recorded outside the file, avoiding a self-digest cycle.
- `objects/sha256/<hex>` stores immutable tensor byte strings. The path is the
  lowercase SHA-256 of the exact bytes.
- `page-tables/sha256/<hex>` stores required digest-bound binary page tables,
  keeping large page-hash sets outside the bounded JSON manifest.
- Each tensor descriptor declares semantic role, little-endian dtype, shape,
  byte-length, object digest, and offset.
- Staging uses a sibling temporary file; hash and expected length are verified;
  publication is an atomic no-replace rename followed by manifest publication.
- Readers use positional bounded reads. No object path is taken directly from
  an artifact, and symbolic links are rejected.

Limits are checked before allocation: manifest bytes, tensor count, rank,
dimension product, individual and aggregate bytes, chunk count, string length,
and supported dtype. Integer arithmetic is checked. Duplicate semantic tensor
roles, unknown required fields, overlapping ranges, digest mismatch, trailing
bytes where forbidden, NaN configuration values, and non-canonical hashes are
errors.

## Consequences

The directory form costs extra inodes but makes experts independently
addressable and resumable. Full SHA-256 costs CPU during ingestion; required
page hashes authenticate bounded runtime reads. The manifest digest binds
every manifest byte, and each manifest record binds an external page table.
A packed transport container may be added later but must unpack into the same
verified object contract.
