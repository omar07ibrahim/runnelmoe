# Threat model

## Assets and trust boundaries

RunnelMoE protects host availability, bounded memory/disk use, numerical
integrity, artifact identity, request isolation, and confidentiality of local
prompts/tokens. The runtime binary and compiled adapters are trusted. Model
artifacts, manifests, HTTP requests, client disconnect timing, and storage
failures are untrusted. The operating system, local administrator, compiler,
and cryptographic implementation are outside this model.

Version 1 is a single-user localhost service, not a hostile multi-tenant
sandbox. SHA-256 proves byte identity against a trusted manifest; it does not
prove publisher authenticity. Users must obtain expected root digests through
a trusted channel.

## Security invariants

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

| Threat | Controls | Verification |
| --- | --- | --- |
| Path traversal or symlink substitution | retained directory FDs, digest-derived names, Linux `openat2` beneath/no-symlink resolution or component-wise `openat(O_NOFOLLOW)`, `fstat`, never reopen | malicious manifest/filesystem tests |
| Manifest allocation bomb | byte/count/rank/string caps and checked arithmetic before reserve | boundary tests and parser fuzzing |
| Truncation, mutation, or reordered pages | exact positional reads, expected lengths, per-object/page SHA-256, opened handles | fault injection at every page |
| Cross-artifact object replay | root manifest identity plus object role/shape binding | swap valid objects between fixtures |
| Partial-object publication | same-directory exclusive staging, hash and `fsync`, no-replace rename/link, parent `fsync`, manifest last | cancellation/crash at each state transition |
| Duplicate concurrent load | one in-flight owner, waiters share verified completion/error | deterministic race tests and Loom where practical |
| Use-after-evict | reference-counted page leases; retire before reclaim | state-machine/property tests and Miri |
| Integer overflow/shape confusion | checked add/multiply/rounding and exact dtype-byte validation | generated boundary corpus |
| NaN/Inf metadata or unstable routing | finite/range checks and specified score/tie policy | special-value and tie tests |
| Compression bomb | compression unsupported in RMOA v1 | reject compression flags |
| Memory/queue denial of service | reserve capacity before allocation/submission; charge buffers, alignment, metadata, waiters, state, scratch, and channels; finite operational artifact caps | low-budget and overload tests |
| Slow client or abandoned SSE | deadlines, bounded output channel, disconnect cancellation | black-box stalled/disconnect tests |
| High-cardinality telemetry | bounded metric labels; IDs only in sampled trace stream | metrics cardinality test |
| Secret/prompt disclosure | structured allowlisted fields, no bodies/tokens, sanitized errors | log capture assertions |
| Illegal SIMD or memory unsafety | scalar fallback, runtime CPUID, narrow C ABI, sanitizer/random differential tests | M4 gate |
| Dependency or CI compromise | lockfiles, minimal dependencies, pinned CI action revisions, dependency review | CI and release audit |
| Disk exhaustion | artifact/result caps, free-space reserve, bounded caches, no large checkpoints | preflight and low-space tests |

## Parser rejection matrix

The corpus must cover wrong magic/version/endianness/critical flags; every
truncation boundary; invalid UTF-8/control/path characters; excessive and
duplicate names/IDs; noncanonical hashes/order; zero or excessive rank and
dimensions; overflow in offsets, products, alignment, page rounding, and
record counts; unsupported dtype/encoding; logical/stored size disagreement;
overlap/misalignment/out-of-file ranges; missing/trailing data; digest
mismatch; valid page reordering; and unknown required adapter fields.

I/O tests inject short reads, interruption/retry, concurrent truncation,
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

Security issues should follow [SECURITY.md](../SECURITY.md), not a public issue.
