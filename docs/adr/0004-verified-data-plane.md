# ADR-0004: Verified out-of-core data plane

- Status: accepted
- Date: 2026-08-03
- Gate: M2

## Context

RMOA version 1 defines a portable single-artifact directory, but a runtime
needs a multi-artifact store, crash-safe ingestion, bounded positional reads,
and a cache whose cancellation and eviction behavior cannot expose unverified
or prematurely reclaimed memory. The M1 convenience reader intentionally
assumes a trusted, stable directory and retains complete object payloads, so it
cannot be relabeled as an out-of-core implementation.

The storage boundary must also make failure semantics testable. In particular,
"cancelled" cannot mean both that publication did not happen and that a final
digest name may already be visible; an asynchronous caller cannot regain a
buffer while a worker still owns it; and an evicted page cannot release its
capacity while a compute lease still exists.

## Decision

### Portable artifacts and the internal CAS

Standalone RMOA remains the import and export representation documented by the
format contract. The runtime uses this separate internal layout:

```text
cas/
  transaction.lock
  manifests/sha256/<artifact-id-hex>
  page-tables/sha256/<page-table-digest-hex>
  objects/sha256/<object-digest-hex>
  staging/<validated-random-stage-name>
```

The manifest leaf contains the exact canonical manifest bytes whose SHA-256 is
its name. Every manifest present in `manifests/sha256` is a garbage-collection
root in M2. Artifact deletion and cross-process live-handle reference counting
are deliberately deferred. This makes import idempotence, open-by-artifact-ID,
manifest-last publication, and mark/sweep reachability unambiguous.

The CAS retains descriptors for its root, lock, staging directory, and three
digest directories. A standalone source likewise retains descriptors for its
root and two digest directories. Path components are never supplied by a
manifest. On Linux the resolver probes `openat2` once against a known-safe
component, then permanently selects either
`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS` or a
component-wise `openat(O_NOFOLLOW)` implementation. A target lookup failure
never causes a security downgrade. All leaves are opened nonblocking and then
checked through the retained descriptor for type, exact size, ownership, and
expected filesystem identity.

### Artifact handles and verified pages

Opening an artifact authenticates the expected manifest ID, completely reads
and verifies its bounded page tables, and retains exact object descriptors.
Object payloads are not eagerly retained. Each logical page specification
binds all of:

- object digest, page size, and page index;
- checked byte offset and exact logical length;
- expected page hash from the immutable retained page table; and
- the retained object handle and exact object length.

The same object bytes may legally be described with different page geometry,
so the cache key includes page size as well as digest and index. A lookup also
requires the stored page length and expected hash to match defensively.

The synchronous backend is authoritative. It performs bounded positional
reads, checks cancellation and deadline between chunks and interrupted calls,
rejects early EOF and post-open length changes, hashes the exact logical page,
and returns an opaque `VerifiedPage` only after the hash matches. A failed
operation retains and destroys or recycles its internal buffer; no public API
ever exposes a caller-owned buffer containing unverified bytes.

The asynchronous backend owns a fixed number of reader threads behind a
bounded queue. Each worker calls the same synchronous read-and-verify routine.
Queue admission, execution, completion, and shutdown preserve buffer ownership
and resource permits. Terminal cancellation is reported only after ownership
returns from a queued or running worker.

### Cache and memory ledger

The verified-page cache uses a mutex-linearized state machine with a
monotonically increasing generation per load:

```text
absent -> loading -> resident -> absent
                    |    |
                    |    +-> retiring -> absent
                    +-------- failure/cancellation -> absent
```

One physical load serves duplicate waiters. Each waiter has an independent
deadline and cancellation interest; one withdrawal does not cancel remaining
interests. The final withdrawal forbids publication and requests backend
cancellation. Completion and withdrawal linearize under the same state lock,
and stale-generation completions cannot publish.

An explicit lease count is maintained under that lock; `Arc` strong counts are
not treated as policy state. Normal eviction skips leased pages. Invalidation
or shutdown changes a leased resident to `retiring`, removes its eligibility,
and retains both storage and byte charge until the final lease drops.

One page pool accounts for loading, resident, and retiring buffers:

```text
page_pool_used = loading_bytes + resident_bytes + retiring_bytes
page_pool_used <= page_pool_capacity
loading_bytes <= max_inflight_bytes <= page_pool_capacity
```

The in-flight limit is a subset cap, not a second allocatable pool. Capacity is
reserved before allocation and moves with the buffer without being released or
double charged. Separate bounded limits cover entries, waiters, load slots,
backend queue items, retained metadata, leases, and trace events. Configuration
is rejected when one supported page cannot fit.

Demand, coalesced demand, prefetch, late prefetch, useful prefetch, wasted
prefetch, redundant prefetch, and budget-dropped prefetch are distinct events.
Tracing uses a bounded nonblocking sink and cannot influence policy decisions.
Metrics use bounded labels; page, object, expert, and request identities appear
only in sampled traces.

### Staging, publication, and durability

Ingestion runs under the project transaction lock and performs a disk preflight
using `f_bavail`. Current final content, staging files, verified orphans,
allocation-unit rounding, missing unique bytes, the configured CAS budget, and
the mandatory filesystem reserve are counted with checked arithmetic.

Each stage is created with `O_CREAT | O_EXCL`, mode 0600, and a random validated
name. A resume token binds the blob kind, expected digest, expected length, and
stage name. Resume reopens the leaf descriptor-relatively, validates metadata,
rehashes its prefix, truncates an incomplete object page to the preceding page
boundary, and continues with bounded positional writes. Serialized hash state
is never trusted. After copying, the retained staging descriptor is reread to
verify on-disk length, whole digest, and page hashes before it is synced.

Publication uses `renameat2(RENAME_NOREPLACE)` or, only when that operation is
known unsupported, same-directory `linkat` followed by `unlinkat`. It never
uses a check-then-overwrite rename. A preexisting final leaf is accepted only
after complete verification. Page tables publish first, objects second, and
the digest-named manifest last as the artifact commit point; each modified
directory is synced.

Cancellation is checked immediately before the no-replace operation. Once a
final digest name becomes addressable, cancellation no longer wins and the
implementation never rolls that publication back automatically. Failure to
sync the parent after addressability is reported as
`published_but_durability_unconfirmed`, allowing an idempotent retry to verify
the leaf and repeat the sync without pretending publication failed.

Garbage collection holds the exclusive transaction lock, authenticates every
manifest root, validates every candidate and the complete deletion plan, and
only then unlinks unreferenced regular digest leaves. Any malformed root,
unexpected entry, symlink, special file, or metadata inconsistency causes zero
deletions. Cancellation before sweep is mutation-free; cancellation during
sweep may leave a safe subset removed, followed by directory sync.

## Verification consequences

M2 closes only with fault injection across descriptor traversal, every
staging/publication transition, resume offsets, exact disk and memory budget
boundaries, cache completion/cancellation races, short reads, truncation,
extension, corruption, page reordering, and shutdown. A deterministic
multi-page fixture with a short final page supplements the one-page numerical
fixture. Sync and async runs must agree on verified bytes, routes, logits, and
generated tokens under a controlled trace with forced eviction; raw trace and
RSS evidence must accompany any measured claim.

Descriptor-relative access prevents pathname redirection but does not isolate
the CAS from a malicious process with the same UID that can modify an already
open regular file. Same-UID hostile mutation remains outside the threat model;
every demanded page is still authenticated before use, and no claim of
multi-tenant filesystem isolation is made.

## Alternatives rejected

- Treating each standalone artifact as an independent store leaves retained
  manifest roots and shared-object garbage collection undefined.
- Returning `read_page_into(&mut [u8])` exposes failed verification bytes to
  callers and makes buffer ownership under asynchronous cancellation unclear.
- Spawning an unbounded blocking task per miss weakens queue and worker limits.
- Charging in-flight and resident buffers as independent pools permits real
  memory to exceed the declared total during state transitions.
- Releasing a retiring page's charge before its final lease drops violates the
  memory ceiling.
- Falling back from `openat2` after a target-specific error creates
  behavior-dependent path resolution.
- Deleting incrementally while discovering GC roots permits a late malformed
  entry to turn a fail-closed scan into partial unsafe mutation.
