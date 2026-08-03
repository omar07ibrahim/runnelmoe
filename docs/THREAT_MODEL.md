# Threat model

## Assets and trust boundaries

The completed RunnelMoE system is intended to protect host availability,
bounded memory/disk use, numerical integrity, artifact identity, request
isolation, and confidentiality of local prompts/tokens. Controls become claims
only at their roadmap gate. The runtime binary and compiled adapters are
trusted. Model artifacts and manifests are untrusted; later HTTP requests,
client disconnect timing, and storage failures join that boundary. The
operating system, local administrator, compiler, and cryptographic
implementation are outside this model.

Version 1 is a single-user localhost service, not a hostile multi-tenant
sandbox. SHA-256 proves byte identity against a trusted manifest; it does not
prove publisher authenticity. Users must obtain expected root digests through
a trusted channel.

## Security invariants

M1 enforces bounded canonical parsing, digest-derived leaf names, exact
length/SHA/page verification, finite tensors and numerical outputs, a fixed
adapter schema, an eager retained-byte budget, and content-redacted external
errors. Its local convenience reader rejects Unix leaf symlinks and special
files, but assumes no concurrent mutation or ancestor-directory substitution.

The cross-milestone system invariants are:

- Parse limits are enforced before allocation or multiplication.
- Manifest data never supplies an absolute/relative object path or URL.
- An object is used only after exact length and digest verification.
- Unverified bytes cannot enter the resident cache or compute backend.
- Runtime-accounted memory and queue counts cannot exceed configured caps.
- Cancellation releases ownership without publishing partial state.
- Logs/metrics omit request content and secrets by default.
- The eventual server binds only to loopback unless an explicit operator
  override is provided; no public deployment is part of this project.
- SIMD entry is impossible until dtype, dimensions, bounds, alignment,
  aliasing, pointer lifetime, and runtime ISA are validated.

## Threats and mitigations

| Threat | Controls | Verification | Gate |
| --- | --- | --- | --- |
| Path traversal or symlink substitution | digest-derived names now; retained directory FDs plus `openat2` beneath/no-symlink or component-wise `openat(O_NOFOLLOW)` next | M1 leaf FIFO/symlink tests; M2 concurrent substitution tests | M1/M2 |
| Manifest allocation bomb | byte/count/rank/string caps and checked arithmetic before reserve | boundary tests now; fuzzing next | M1/M2 |
| Truncation, mutation, or reordered pages | eager exact length, whole/page SHA now; bounded positional reads and retained handles next | mutation corpus now; I/O fault injection next | M1/M2 |
| Cross-artifact object replay | root manifest identity plus object role/shape binding | expected-ID rejection now; valid-object swap fault injection next | M1/M2 |
| Partial-object publication | exclusive staging, hash and `fsync`, no-replace publication, manifest last | cancellation/crash at each transition | M2 |
| Duplicate concurrent load | one in-flight owner, waiters share completion/error | race tests and Loom where practical | M2 |
| Use-after-evict | reference-counted page leases; retire before reclaim | state-machine/property tests and Miri | M2 |
| Integer overflow/shape confusion | checked arithmetic and exact dtype/byte/shape validation | generated mutation corpus | M1 |
| NaN/Inf or unstable routing | finite checks and specified score/tie policy | special-value and tie tests | M1 |
| Compression bomb | compression absent from and rejected by RMOA v1 | exact-schema tests | M1 |
| Memory/queue denial of service | eager retained-byte cap now; full ledger/admission and bounded queues next | low-budget parser tests now; overload tests next | M1/M2/M5 |
| Slow client or abandoned SSE | deadlines, bounded output channel, disconnect cancellation | stalled/disconnect black-box tests | M6 |
| High-cardinality telemetry | bounded metric labels; IDs only in sampled traces | metrics cardinality test | M6 |
| Secret/prompt disclosure | allowlisted output and content-redacted errors | CLI stderr capture now; server log capture next | M1/M6 |
| Illegal SIMD or memory unsafety | scalar fallback, runtime CPUID, narrow C ABI | sanitizer/random differential tests | M4 |
| Dependency or CI compromise | lockfiles, exact dependencies, pinned CI action revisions | CI and release audit | ongoing |
| Disk exhaustion | no committed checkpoints now; reserve-aware CAS and bounded results next | disk preflight and low-space tests | M1/M2 |

## Parser rejection matrix

The corpus must cover wrong magic/version/endianness/critical flags; every
truncation boundary; invalid UTF-8/control/path characters; excessive and
duplicate names/IDs; noncanonical hashes/order; zero or excessive rank and
dimensions; overflow in offsets, products, alignment, page rounding, and
record counts; unsupported dtype/encoding; logical/stored size disagreement;
overlap/misalignment/out-of-file ranges; missing/trailing data; digest
mismatch; valid page reordering; and unknown required adapter fields.

M2 I/O tests inject short reads, interruption/retry, concurrent truncation,
replacement after open, and cancellation before read, during read, during
hashing, and immediately before publication. No failure may expose verified
status or leak accounted bytes.

## Abuse cases for later serving

M6 tests malformed/deep JSON, oversized bodies, excessive messages/tokens,
invalid sampling floats, NaN/Infinity representations, request floods, queue
starvation, deadline overflow, stalled readers, disconnect storms, invalid
UTF-8, unknown models, and metric scraping during load. Authentication and TLS
are explicitly not supplied by the local server; operators needing remote
access must place a reviewed gateway in front, outside this project's default.

## Residual risks

- SHA-256 cannot identify a malicious but internally consistent model.
- Resource accounting cannot exactly predict allocator, runtime, or kernel page
  overhead; RSS evidence and a reserve are required.
- Memory-mapped or cached filesystem data may affect host page cache outside
  process RSS. RunnelMoE uses explicit reads and reports this limitation.
- Floating-point agreement is tolerance-based across compilers/ISAs.
- Local processes with the same user privileges can observe files and may be
  able to inspect process memory.
- The M1 eager convenience reader is for a trusted local directory that is not
  concurrently replaced. Descriptor-relative race resistance and transactional
  publication are M2 gates.

Security issues should follow [SECURITY.md](../SECURITY.md), not a public issue.
