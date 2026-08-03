use runnel_sim::{SimLimits, parse_trace};

#[test]
fn deterministic_arbitrary_bytes_never_escape_parser_limits_or_panic() {
    let limits = SimLimits {
        max_trace_bytes: 1_024,
        max_line_bytes: 256,
        max_pages: 16,
        max_events: 64,
        max_predictions_per_signal: 4,
        max_exact_oracle_states: 1_000,
    };
    let mut state = 0x7d55_9f2d_a13b_4c89_u64;
    for case in 0..4_096_usize {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let length = (usize::try_from(state >> 32).unwrap() + case) % 1_025;
        let mut bytes = Vec::with_capacity(length);
        for index in 0..length {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bytes.push((state as u8).wrapping_add(index as u8));
        }
        let _ = parse_trace(&bytes, limits);
    }
}

#[test]
fn bounded_valid_prefix_mutations_fail_closed() {
    let seed = b"{\"kind\":\"header\",\"schema\":\"runnel.cache-trace/1\",\"trace_id\":\"fuzz-seed\",\"page_count\":0,\"event_count\":0,\"charge_quantum\":1,\"prefetch_model\":\"instant-between-events-v1\"}\n";
    assert!(parse_trace(seed, SimLimits::default()).is_ok());
    for index in 0..seed.len() {
        for replacement in [0_u8, b' ', b'0', b'{', b'}', b'\n', 0xff] {
            let mut mutated = seed.to_vec();
            mutated[index] = replacement;
            let _ = parse_trace(&mutated, SimLimits::default());
        }
    }
}
