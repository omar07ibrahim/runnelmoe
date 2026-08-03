# Tiny causal MoE fixture, adapter version 2

`spec.json` keeps the tiny-v1 topology and deterministic tensor recipe while
storing only the twelve routed-expert `gate`, `up`, and `down` matrices as
little-endian BF16. Every other tensor remains little-endian float32. The
independent PyTorch oracle explicitly converts each expert recipe tensor from
float32 to `torch.bfloat16` and back to float32 before model arithmetic, so all
matmuls still use float32 compute.

The four `golden_*.json` files are generated independently for this directory;
they are not aliases or copies of the tiny-v1 vectors. The recipe happens to be
exactly BF16-representable, so the round trip changes zero coefficients, but
the generated metadata and tests retain the storage path, tensor/byte counts,
and distinct adapter-v2 artifact identity.

From the repository root, verify these vectors with:

```console
python -m oracle.generate --check --spec fixtures/tiny-v2/spec.json
```
