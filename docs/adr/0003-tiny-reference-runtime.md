# ADR-0003: Tiny reference runtime and independent oracle

- Status: accepted
- Date: 2026-08-03
- Gate: M1

## Context

The out-of-core data plane needs an executable numerical and artifact contract
before cache, asynchronous I/O, scheduling, or kernels can be optimized. A real
checkpoint would be too large for CI, hard to audit, and unnecessary for
testing parser and decoder semantics. Using one implementation as both runtime
and oracle would hide shared mistakes.

## Decision

M1 uses four Rust boundaries: strict RMOA verification, deterministic fixture
generation, a scalar tiny-adapter runtime, and a CLI. Tiny-v1 freezes vocabulary
32, hidden width 8, two attention heads, one layer, four experts, top two,
expert width 12, and context 16. Its 22 `f32-le` tensors come from a closed
integer/power-of-two formula. The resulting 7,904-byte object is generated for
tests and demos but is not committed as a weight file.

The M1 artifact reader eagerly retains verified bytes under an independent 256
MiB default budget. It verifies canonical manifest bytes, tensor coverage,
whole objects, complete page tables, and every page before exposing tensors.
Its convenience filesystem opener is limited to a trusted directory without
concurrent ancestor replacement; descriptor-relative traversal and
transactional publication are M2 work.

The Rust runtime executes one token at a time with scalar KV state and only the
selected experts. A separately organized PyTorch oracle recomputes vectorized
full prefixes and all experts. Exact routes and generated IDs must match;
scores, weights, and logits use `atol=1e-5`, `rtol=1e-4`. Committed vectors bind
the exact artifact identity and every evaluated position.

Tiny-v1 has no positional encoding and supports greedy generation only. It is a
systems fixture, not a language-quality or performance claim.

## Consequences

- Parser, adapter, state, routing, and generation changes have a small offline
  differential test.
- Corrupt and incompatible artifacts fail before numerical use, and adapter
  descriptors are validated before tensor decoding.
- The fixture's permutation limitation and synthetic output are public and
  cannot be confused with model capability.
- M2 must preserve the numerical results while replacing eager path-based I/O
  with bounded descriptor-safe storage.

## Alternatives rejected

- A downloaded model checkpoint violates CI size and clean-room constraints.
- Committing generated object bytes obscures the formula and creates an
  unnecessary model-weight artifact.
- Reusing the Rust execution structure in Python weakens differential value.
- Building cache and async layers before scalar parity makes later failures
  harder to localize.
