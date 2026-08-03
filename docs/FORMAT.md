# RMOA version 1 format

RMOA (RunnelMoE Object Artifact) is an immutable, content-addressed checkpoint
format. This document is the normative version-1 contract.

Implementation status matters: M1 implements the canonical manifest schema,
limits, eager memory budget, whole-object/page-table/page verification, and a
small local-filesystem convenience reader. Sections explicitly marked **M2
target** define the descriptor-safe storage/publication contract and are not an
M1 claim.

## Layout and identity

    artifact/
      manifest.json
      objects/sha256/<64 lowercase hexadecimal characters>
      page-tables/sha256/<64 lowercase hexadecimal characters>

The manifest is UTF-8 JSON serialized with the
[JSON Canonicalization Scheme (RFC 8785)](https://www.rfc-editor.org/rfc/rfc8785)
followed by exactly one ASCII LF byte. The artifact ID is
`sha256:<digest-of-those-exact-bytes>`. A trusted caller may supply an
expected artifact ID; otherwise the loader reports the computed ID without
claiming publisher authenticity.

Object and page-table filenames are derived only from their digest. Manifest
strings never supply paths.

## JSON rules

The generic parser rejects duplicate keys before materializing an object and
rejects null and floating-point numbers. Every integer is in the inclusive
JSON safe-integer range 0 through 9,007,199,254,740,991. After schema
validation, the loader
re-serializes with RFC 8785 and requires byte equality, including the final LF.
Schema strings must be NFC Unicode; identifier fields have the narrower ASCII
rules below.

The top-level object has exactly these keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `adapter` | object | exact keys `id` and `version` |
| `format` | string | literal `rmoa` |
| `model` | object | adapter-owned dimension map |
| `objects` | array | unique records ordered by digest |
| `tensors` | array | unique records ordered by increasing ID |
| `tokenizer` | object | exact keys `id`, `version`, and `vocab_size` |
| `version` | integer | literal `1` |

Adapter/tokenizer versions and vocabulary size are positive. Their IDs match
`[a-z][a-z0-9.-]{0,63}`. The generic `model` value is an object with at
most 128 unique ASCII identifier keys and unsigned-integer values. An adapter
must reject unknown, missing, zero, inconsistent, or over-limit dimensions.

The M1 adapter ID is `runnel.tiny-causal-moe`, version 1. Its model map has
exactly `context_length`, `expert_hidden_size`, `hidden_size`,
`num_experts`, `num_heads`, `num_layers`, `top_k`, and `vocab_size`.
All are positive; `top_k <= num_experts`,
`hidden_size % num_heads == 0`, and both vocabulary values are equal.

## Object records and page tables

Each object record has exactly:

| Key | Type | Constraint |
| --- | --- | --- |
| `digest` | string | `sha256:` plus 64 lowercase hex characters |
| `length` | integer | positive exact object bytes |
| `page_size` | integer | power of two, 65,536 through 2,097,152 |
| `page_table` | string | `sha256:` plus 64 lowercase hex characters |
| `page_table_length` | integer | exact byte length |

Object digests and page-table digests are each unique. Records are ordered
lexicographically by object digest.

A page table is binary:

| Byte range | Encoding |
| --- | --- |
| 0..8 | ASCII `RMOAPG1\n` |
| 8..12 | little-endian u32 version, exactly 1 |
| 12..16 | little-endian u32 page size |
| 16..24 | little-endian u64 object length |
| 24..32 | little-endian u64 page count |
| 32..64 | raw 32-byte whole-object SHA-256 |
| 64..end | ordered raw 32-byte SHA-256 for each page |

`page_count = ceil(object_length / page_size)`,
`page_table_length = 64 + 32 * page_count`, and both equations use checked
arithmetic. The final page may be short and its hash covers only present bytes.
The page-table file's own digest and length are verified completely before any
object page is exposed. Its header must exactly match the manifest record. The
entire verified table is retained in an immutable, fully memory-accounted
buffer for the lifetime of the artifact handle; page-hash lookups never reread
the mutable file descriptor.

## Tensor records

Each tensor record has exactly:

| Key | Type | Constraint |
| --- | --- | --- |
| `dtype` | string | `f32-le`, `bf16-le`, `u8`, or `i8` |
| `id` | integer | contiguous from zero |
| `length` | integer | exact logical bytes |
| `object` | string | digest of a declared object |
| `offset` | integer | byte offset aligned to dtype width |
| `role` | string | adapter semantic role |
| `shape` | array | row-major positive dimensions |

Rank is 1 through 8. Every dimension is 1 through 2,147,483,647. Length equals
the checked product of dimensions and dtype width. Roles match
`[a-z][a-z0-9_.-]{0,127}` and are unique.

Every tensor range fits its object. Ranges in one object do not overlap and,
when sorted by offset, cover bytes 0 through object length exactly with no
gaps, padding, aliases, or trailing data. Version 1 has no compression,
implicit strides, executable metadata, remote URLs, or quantization metadata.

## Format ceilings and operational defaults

Format ceilings are checked before allocation:

| Quantity | Version-1 ceiling |
| --- | ---: |
| Manifest bytes | 4 MiB |
| Objects / tensors | 65,536 each |
| JSON nesting depth | 12 |
| Any schema string | 256 UTF-8 bytes |
| Object bytes | 1 TiB |
| Aggregate object bytes | 8 TiB |
| One page table | 64 MiB |
| Aggregate page-table bytes | 1 GiB |

These are interoperability ceilings, not safe defaults. Unless an operator
explicitly lowers or raises them, the M1 reader accepts at most a 1 MiB manifest,
4,096 objects, 16,384 tensors, 2 GiB per object, 2 GiB aggregate object bytes,
8 MiB per page table, and 64 MiB aggregate page-table bytes. Raising a default
cannot exceed the format ceiling. Its independent eager budget defaults to 256
MiB and covers retained manifest, object, and page-table bytes.

M2 ingestion will additionally perform a worst-case preflight that includes
current CAS usage, all staging bytes, newly published bytes, and verified
orphan entries while preserving the configured disk reserve. Its hash loops
will check cancellation and deadline at every bounded buffer or page.

The full M2 runtime memory budget is independent: metadata, page tables,
open-handle state, resident pages, in-flight capacity, scratch, queues, and
sequence state must be reserved before use.

## Filesystem and publication rules — M2 target

On Linux, the M2 loader will retain file descriptors for the artifact root and
both digest directories and resolve children with `openat2` using
`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS`. A portable
fallback walks fixed path components with directory-relative `openat`,
`O_NOFOLLOW`, and `fstat`. It requires directories/regular files owned by
the expected handles and never reopens a checked pathname. Exact length is
checked on the retained regular-file descriptor.

M2 ingestion will create a randomly named sibling staging file with
`O_CREAT | O_EXCL` and mode 0600. It writes with bounded buffers, verifies
length and hashes, flushes and `fsync`s the file, then publishes without
replacement using `renameat2(RENAME_NOREPLACE)`. Where unavailable, a
same-directory `linkat`-then-unlink fallback provides no-replace publication.
The parent directory is `fsync`ed. Page tables publish before objects and the
manifest publishes last. Cancellation may leave complete verified but
unreferenced content-addressed files as well as unaddressed staging files.
Orphans are safe to ignore and may be garbage-collected only by comparing
digests against retained manifests; partial files are never addressable.

The M2 project CAS will have a configured disk budget; the host default is the
smaller of 2 GiB and available bytes above the mandatory 2 GiB filesystem
reserve.
Staging and orphan bytes count against it. Ingestion and garbage collection
hold a project-owned transaction lock. On cancellation, newly published
digests are removed only if no retained manifest references them. Garbage
collection uses retained directory descriptors, rejects unexpected entries,
and deletes only unreferenced regular digest files; it never follows links.

## Verified reads

The M1 convenience reader assumes a trusted, non-concurrently-mutated artifact
directory. On Unix it opens leaf files nonblocking with `O_NOFOLLOW`, rejects
non-regular inputs, bounds each read, and eagerly verifies all declared bytes
before exposure. It does not yet retain directory descriptors or defend
ancestor-directory replacement.

The M2 storage API will read whole logical pages. It will verify the retained
page-table entry before transitioning a buffer from `loading` to `resident`.
Consumers may receive tensor slices only after every intersecting page
verifies. Short reads, extra bytes, hash mismatch, mutation, cancellation, or
deadline reject the operation; no best-effort tensor is returned.

M1 commits accepted and rejected canonical vectors, every manifest and
page-table truncation boundary, object truncation cases, checked overflow,
reordered valid pages, leaf symlink/FIFO rejection, and digest corruption.
Concurrent filesystem substitution, cancellation, deadline, short positional
read, and publication fault injection close with M2.
