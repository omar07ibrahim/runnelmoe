# runnel-format

This crate implements the byte-level reader for the normative RMOA v1 contract
in [`docs/FORMAT.md`](../../docs/FORMAT.md), including the closed tiny-adapter
version set. It rejects ambiguous JSON and validates the manifest schema,
canonical representation, tensor coverage, content digests, page-table
headers, and every object page before returning a tensor slice.

Tiny fixtures are deliberately small, so `Artifact` uses an explicit eager-memory
budget and retains verified bytes. Its filesystem convenience reader does not
yet claim adversarial pathname safety. M2 must replace that traversal with
retained directory descriptors, Linux `openat2` resolution (or a component-wise
`openat`/`O_NOFOLLOW` fallback), exact descriptor metadata checks, bounded async
reads, transactional no-replace CAS publication, cancellation/deadlines, disk
reserve accounting, and reference-safe orphan collection. This TODO does not
relax M1 byte validation: neither object data nor a page hash is exposed before
all relevant lengths and SHA-256 values verify.
