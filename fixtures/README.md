# Synthetic fixtures

Fixtures are deterministic, tiny, and generated entirely by this project.
M1 commits the seedless integer formula, format limits, mutation-based malformed
corpus, artifact identity, and golden oracle outputs. The 7,904-byte checkpoint
object is generated only into a caller-selected directory; it is not committed
and contains no trained or third-party model weights.

M4 adds a separate adapter-v2 rendering of the same formula and topology. Only
the twelve expert gate/up/down matrices use BF16, producing a generated
5,600-byte object. The v1 fixture, its identity, and its goldens remain
immutable. Neither generated object is committed as model data.

M3 also commits three tiny canonical
[cache-policy traces](cache/README.md). They are hand-inspectable correctness
inputs; the larger stochastic experiment traces are regenerated from the
versioned Rust generator and pinned by digests rather than stored repeatedly.
