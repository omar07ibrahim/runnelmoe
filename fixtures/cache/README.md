# Cache-policy golden traces

These canonical JSONL files are small deterministic correctness inputs, not
performance evidence.

- `m2-forced-eviction.jsonl` is the policy-neutral form of the M2 demand
  schedule `[0, 1, 0, 2, 2]`. At a 65,536-byte capacity, byte LRU must produce
  four misses, one hit, four admissions, and three evictions.
- `router-perfect.jsonl` shows a causal router signal whose one prefetched page
  is demanded at its target step. Total physical bytes remain equal to the LRU
  demand fill even though the demand-load counter becomes zero.
- `variable-byte.jsonl` is a tiny nonuniform-charge input for the bounded exact
  dynamic program. Bélády/MIN must reject this geometry rather than call a
  farthest-next-use heuristic optimal.

Every record is compact ASCII JSON terminated by LF. The strict simulator
parser rejects edits that are not canonical.
