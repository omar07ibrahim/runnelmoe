# Tiny causal MoE fixture

`spec.json` is the complete deterministic recipe for a synthetic one-layer
causal MoE. It contains dimensions, equations, tokenizer IDs, tensor roles,
and an integer/power-of-two tensor formula. There are deliberately no model
weight files.

The four `golden_*.json` files are generated only by the independent PyTorch
oracle in `oracle/`. They bind the exact RMOA/object/page-table identities,
dependency version and tolerance; every prompt/generated-token route and
position logit; every greedy decision; and the resulting token sequence. The
Rust differential test regenerates the RMOA bytes, verifies them through the
filesystem loader, and checks those committed oracle results.

Tiny-v1 intentionally has no positional encoding, so it is a parser/runtime
fixture rather than a language-quality model. See `oracle/README.md` for write
and verification commands.
