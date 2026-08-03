# M1 review record

- Milestone: exact tiny reference runtime
- Review date: 2026-08-03 UTC
- Scope: RMOA parser/verifier, deterministic artifact generator, scalar Rust
  adapter, PyTorch oracle, committed vectors, CLI, CI, and public claims
- Local verdict: pass
- Remote CI: pending first feature-branch push

## Acceptance evidence

The deterministic artifact is generated from source rather than committed as a
weight file:

| Item | Frozen value |
| --- | --- |
| Artifact ID | `sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3` |
| Object | `sha256:6b2b8a1bbb2854084b1e1fe1e5787a9cfdb021b397e774fc7dbef79ac9d24bf6`, 7,904 bytes |
| Page table | `sha256:29383b56a150f9e5705f3666ca7707f21bc3fbd663ffd7249f4ecb938da6a62d`, 96 bytes |
| Prompt tokens | `moe` → `[1, 14, 16, 6]` |
| Generated tokens | `[15, 11, 20, 9]` |
| Decoded text | `njsh` |
| Floating comparison | `abs <= 1e-5 + 1e-4 * abs(oracle)` |

The primary differential test writes the artifact to a new temporary
directory, reopens the canonical manifest, page table, and object through
`runnel-format`, decodes verified `f32-le` tensors through the adapter, and
then compares every committed position. It does not construct the integration
model directly from fixture tensors.

Local commands reproduced from a clean index plus the working M1 change:

```console
python3 scripts/verify_repository.py
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --all-targets --locked --offline
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked --offline
python -m oracle.generate --check
python -m unittest discover -s oracle/tests -v
cargo run --locked --offline -p runnel -- demo --prompt moe --max-new-tokens 4 --json
```

Results: 43 Rust tests and 7 independent-oracle tests passed; Clippy and
rustdoc passed with warnings denied; four golden files regenerated in memory
without drift; the repository contract and CLI JSON contract passed. No
benchmark or performance result is claimed by M1.

## Numerical review

An independent bounded review checked tensor orientation, f32 loop order,
two-head split/concatenation, current-token causality, RMS normalization,
selected-only softmax, SwiGLU order, selected-rank mixture accumulation,
stable ties including signed zero, greedy ties, EOS/BOS, context admission,
state rollback/ownership, and full-prefix versus incremental parity.

Findings fixed during review included signed-zero ordering through
`total_cmp`, mutation of KV state before a fallible step completed,
cross-model state reuse, missing boundary cases, incomplete position-logit
evidence, and non-finite comparison acceptance. Final numerical verdict: pass
with no open P0/P1 finding.

## Security and robustness review

An independent bounded review exercised canonical/malformed JSON, every
manifest and page-table truncation boundary, object truncation/corruption,
page reordering, checked arithmetic, adapter metadata/dtype/shape rejection,
memory limits, CLI failure behavior, and external error content.

Findings fixed during review included a blocking FIFO open, leaf-symlink
following, allocation before context rejection, decoding before exact catalog
validation, prompt-character retention in public errors, overwrite wording in
the fixture helper, and documentation that presented M2 filesystem/publication
controls as M1 behavior. Unix leaf opens are now nonblocking/no-follow and
regular-file checked; the eager reader's trusted-directory limitation is
explicit. Final security verdict: pass with descriptor-relative race safety,
transactional publication, cancellation, and disk-reserve enforcement tracked
at M2.

## Clean-room and claim audit

M1 source, prose, fixture equations, vectors, names, and CLI output were
created from this repository's frozen specifications and primary language/tool
documentation. No source, layout, fixture, prose, artwork, or measurement from
the credited Kimi K3 C prior-art repository was reused. The upstream pin and
license context remain in `docs/PRIOR_ART.md`; no Kimi checkpoint was acquired.

Claims remain limited to tiny deterministic correctness. Tiny-v1 has no
positional encoding and is explicitly not a language-quality model. The M1
eager reader is not described as an out-of-core cache or adversarial filesystem
loader, and there are no latency, throughput, memory-scaling, or speedup claims.

## Residual work

M1 closes only after the feature commit and this review pass all remote CI jobs.
M2 owns descriptor-relative filesystem traversal, bounded sync/async positional
I/O, transactional CAS publication, cancellation/deadline behavior, full
memory/disk ledgers, cache leases, and fault injection under concurrent
substitution.
