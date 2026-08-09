# Tiny adapter-v3 fixture

This directory freezes the compact-BF16 fixture used by the M5 long-context
measurement contract. It is an original deterministic synthetic model: no
checkpoint or third-party model data is stored here.

Adapter v3 deliberately keeps adapter v2's tensor recipe, topology, storage
dtypes, generated 5,600-byte object, and page table byte-for-byte identical.
Only two manifest fields change:

- adapter version is 3;
- context length is 1,024 tokens.

Those changes give v3 a distinct manifest and artifact identity while keeping
short numerical comparisons isolated from any weight change. The committed
`moe` token, route, and logit vectors are therefore numerically identical to
v2 and are checked through the independent PyTorch oracle. Metadata records the
v3 fixture and artifact identities explicitly.

Long mechanics tests construct valid token IDs directly: `P(n)[0] = 1`, then
`P(n)[i] = [14, 16, 6][(i - 1) mod 3]` for `1 <= i < n`. This repeated
synthetic input has no language-quality interpretation and does not add a
second set of long golden logits.

From the repository root, verify the committed vectors with:

```console
python -m oracle.generate --check --spec fixtures/tiny-v3/spec.json
```

Regeneration is deliberate and scoped to this directory:

```console
python -m oracle.generate --write --spec fixtures/tiny-v3/spec.json
```

The object bytes remain generated on demand by `runnel-fixture`; they are not
committed as model data.
