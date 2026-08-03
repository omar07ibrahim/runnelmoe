# `runnel-kernels`

This crate owns RunnelMoE's compact finite-BF16 matrix representation, the
ascending-column safe Rust GEMV reference, and one runtime-dispatched C AVX2
candidate. The public API is safe and publishes output only after structural,
native-status, and finiteness checks pass.

The default `native-avx2` feature builds the C object only for a native
`x86_64-unknown-linux-gnu` host/target pair. Other and cross targets remain
scalar-only. The portable gate is:

```console
cargo test -p runnel-kernels --all-targets --no-default-features
```

ABI, numerical, sanitizer, dispatch, and evidence requirements are frozen in
[ADR-0006](../../docs/adr/0006-bf16-avx2-expert-kernel.md). This is a narrow
GEMV boundary, not a general tensor or model library.
