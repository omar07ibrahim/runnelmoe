# `runnel-sim`

`runnel-sim` is RunnelMoE's deterministic cache-policy research harness. It
replays policy-neutral demand events and causal router signals under an exact
byte capacity. The crate is independent from `runnel-store`; tests compare the
two implementations on a shared LRU schedule.

The simulator supports:

- no-cache and byte-LRU references;
- byte-target SLRU;
- TinyLFU admission over byte LRU, with a deterministic bounded sketch;
- project-original router-aware SLRU admission, with and without bounded
  expert-group prefetch; and
- exact Bélády/MIN replay for uniform page charge and miss cost only.

Variable-size farthest-next-use is not labeled optimal. A bounded exponential
dynamic program exists for tiny correctness cases instead.

## Cost model

Every catalog page declares `logical_bytes`, charged to physical traffic on a
successful fill, and `charge_bytes`, charged to the resident pool. The
`instant-between-events-v1` prefetch model treats an admitted prefetch as
resident before the next event. It measures traffic and cache pollution, not
latency, overlap, throughput, or TTFT.

Demand hits on unread prefetched pages are identified separately from ordinary
hits. Every admitted prefetch becomes either useful on its first demand or
wasted on eviction/finalization. Results enforce:

```text
total physical load bytes = demand load bytes + prefetch load bytes
prefetch load bytes = useful prefetch bytes + wasted prefetch bytes
resident charge bytes <= configured capacity bytes
```

Router scores are stored by exact `(request, target_step, layer)` identity, so
multiple layers and multiple outstanding targets coexist. The result's
`policy_metadata_bytes` is peak normalized payload—24 bytes per nonempty active
signal plus eight bytes per selected prediction—not Rust allocator usage. Its
precomputed limit is enforced with checked arithmetic; allocator overhead is
outside this simulator metric.

## CLI

All commands are offline and write machine-readable output to standard output.
Trace files are opened once with no-follow and nonblocking flags and read
through a retained regular-file descriptor with a hard `limit + 1` bound.
Concurrent length changes fail closed.

```bash
cargo run -p runnel-sim --bin runnel-cache-sim -- \
  generate --family markov_clusters --replicate 0 --measured-steps 64 \
  > /tmp/runnel-trace.jsonl

cargo run -p runnel-sim --bin runnel-cache-sim -- \
  simulate /tmp/runnel-trace.jsonl --policy tiny-lfu \
  --capacity-bytes 4194304

cargo run -p runnel-sim --bin runnel-cache-sim -- \
  matrix --family markov_clusters --replicate 0 --measured-steps 64
```

The full M3 experiment uses six generated families, 30 paired seeds, three
capacities, and six policies. The command's own execution time is not an M3
performance metric. See
[`ADR 0005`](../../docs/adr/0005-cache-policy-research.md) for frozen semantics
and [`docs/BENCHMARKING.md`](../../docs/BENCHMARKING.md) for the evidence
contract.
