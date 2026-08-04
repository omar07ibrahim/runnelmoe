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
python -m oracle.generate --check --spec fixtures/tiny-v2/spec.json
python -m oracle.generate --check --spec fixtures/tiny-v3/spec.json
python -m oracle.sampling --check
python -m oracle.scheduler --check
python -m unittest discover -s oracle/tests -v
```

To deliberately regenerate the four golden JSON files after reviewing a spec
change:

```console
python -m oracle.generate --write
python -m oracle.generate --write --spec fixtures/tiny-v2/spec.json
python -m oracle.generate --write --spec fixtures/tiny-v3/spec.json
python -m oracle.sampling --write
python -m oracle.scheduler --write
```

The default command remains the frozen all-float32 adapter-v1 fixture. The
explicit spec commands independently generate or check adapters v2 and v3 in
their own directories. In both compact fixtures, only the twelve routed-expert
matrices make an explicit PyTorch `float32 -> bfloat16 -> float32` round trip.
This models compact BF16 storage with round-to-nearest, ties-to-even while
keeping all oracle arithmetic in float32. Adapter v3 retains v2's tensor bytes
and equations but raises the declared context cap from 16 to 1,024 tokens. No
earlier adapter source or golden file is rewritten.

`--check` compares token and route IDs exactly, rejects non-finite values, and
compares floating-point values with the tolerances declared by the fixture. It
never rewrites files. PyTorch recomputes complete prefixes and all experts;
Rust independently executes incremental scalar KV state and only selected
experts.

## Independent sampling oracle

`sampling.py` is a Python-standard-library reference for the M5 SplitMix64,
top-k/top-p, stable-softmax, and categorical-selection contract. It accepts
binary32 values as raw hexadecimal bit patterns and visits candidates in the
order frozen by ADR 0007. Its committed vectors and custody details are
documented in [`fixtures/scheduler/README.md`](../fixtures/scheduler/README.md).

The vector checker validates a closed canonical JSON schema, recomputes every
valid and invalid case, verifies the semantic vector identity, and then
requires byte equality with independent regeneration. Production code does not
import this oracle, and the oracle does not consume production outputs.

## Independent scheduler stress inputs

`scheduler.py` is a standard-library-only implementation of the domain-separated
SHA-256 word stream, rejection sampling, synthetic request descriptors, action
selection, and two-producer assignment frozen by ADR 0007. Its committed
`actor-stress-v1.json` fixture contains inputs and configuration only, including
the exact tiny-v3 scalar spec-file hash and authenticated artifact identity. It
does not import the Rust scheduler, model an actor result, or claim a golden
terminal digest. The checker performs bounded no-follow reads, rejects
noncanonical or open-schema JSON, recomputes every identity, and requires byte
equality with a fresh independent generation.

`actor_transcript.py` is a separate, standard-library-first verifier for the
logical results emitted by the feature-gated Rust actor golden. It regenerates
the 1,024 actions, reconstructs request identity and FIFO relations, validates
terminal, cleanup-authority, shutdown, recorder, and bounded-pump invariants,
then independently emits the fixed-width ADR 0007 transcript. The optional
`--validate-model` gate lazily loads PyTorch and recomputes every accepted
request from the authenticated tiny-v3 specification; Rust output must be an
exact generated prefix, and completed requests must match the complete oracle
sequence.

The accepted capture and semantic digest can be checked without writing a
binary transcript:

```console
python -m oracle.actor_transcript \
  fixtures/scheduler/actor-golden-v1.json \
  --expected-digest fixtures/scheduler/actor-golden-v1.sha256
```

The steady-state cross-language handshake uses only new exchange files in a
private tmpfs directory. Python never receives the Rust transcript, and the
final Rust invocation compares the independently produced bytes:

```console
exchange_dir="$(mktemp -d /dev/shm/runnel-actor-golden.XXXXXX)"
python -m oracle.actor_transcript \
  fixtures/scheduler/actor-golden-v1.json \
  --expected-digest fixtures/scheduler/actor-golden-v1.sha256 \
  --transcript "$exchange_dir/committed-transcript.bin"
RUNNEL_ACTOR_GOLDEN_CAPTURE="$exchange_dir/capture.json" \
  cargo test -p runnel-scheduler --test actor_script --no-default-features \
    --features actor-stress-instrumentation \
    deterministic_actor_semantic_golden_is_bounded_and_structurally_sound \
    --locked --offline
python -m oracle.actor_transcript "$exchange_dir/capture.json" \
  --expected-digest fixtures/scheduler/actor-golden-v1.sha256 \
  --transcript "$exchange_dir/python-transcript.bin" --validate-model
cmp "$exchange_dir/committed-transcript.bin" \
  "$exchange_dir/python-transcript.bin"
RUNNEL_ACTOR_PYTHON_TRANSCRIPT="$exchange_dir/python-transcript.bin" \
  cargo test -p runnel-scheduler --test actor_script --no-default-features \
    --features actor-stress-instrumentation \
    deterministic_actor_semantic_golden_is_bounded_and_structurally_sound \
    --locked --offline
```

The capture, expected-digest, and transcript readers are bounded
regular-file/no-follow paths; the transcript destination is create-new. The
`.sha256` file is the semantic transcript digest, while the committed capture's
physical diagnostics remain historical observations subject to bounds rather
than exact live equality. The binary transcript is an exchange artifact, not a
repository artifact.

## Independent cache-policy oracle

`cache_policy.py` is a separate standard-library reference for the M3 online
policy kernel. It parses the closed canonical trace independently, retains
ordered demand and router events, and implements byte-capacity LRU, SLRU,
TinyLFU, and router-admit with a different state layout from Rust. In
particular, router scores use a flat `(request, target_step, layer)` map, while
the Rust simulator uses nested ordered maps. Differential tests include
variable-size multi-victim eviction, SLRU demotion, TinyLFU aging and strict
admission, plus simultaneous multi-layer and multiple-outstanding router
signals. The Python oracle does not implement speculative prefetch or consume
Rust decisions as inputs.

Run it directly on a small canonical trace with:

```console
python3 oracle/cache_policy.py --trace fixtures/cache/m2-forced-eviction.jsonl --policy lru --capacity-bytes 65536
```
