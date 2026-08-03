# Independent PyTorch oracle

This directory contains the independently structured reference implementation
for the deterministic tiny causal MoE fixture. Production code does not import
it, and the oracle does not consume production routes or intermediate values.

The model has no checkpoint file. Every row-major `float32` tensor is rebuilt
from the tensor ID and flat index using the exact integer/power-of-two recipe in
`fixtures/tiny/spec.json`. This keeps the fixture auditable and avoids shipping
opaque or third-party weights.

From the repository root, install the exact CPU dependency closure and verify the
committed vectors:

```console
python -m pip install -r oracle/requirements.txt
python -m oracle.generate --check
python -m unittest discover -s oracle/tests -v
```

To deliberately regenerate the four golden JSON files after reviewing a spec
change:

```console
python -m oracle.generate --write
```

`--check` compares token and route IDs exactly, rejects non-finite values, and
compares floating-point values with the tolerances declared by the fixture. It
never rewrites files. PyTorch recomputes complete prefixes and all experts;
Rust independently executes incremental scalar KV state and only selected
experts.
