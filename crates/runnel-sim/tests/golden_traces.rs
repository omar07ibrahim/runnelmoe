use std::path::PathBuf;

use runnel_sim::{
    PolicySpec, RouterPolicyConfig, SimError, SimLimits, SimulationConfig,
    exact_variable_byte_cost, parse_trace, simulate,
};

fn fixture(name: &str) -> runnel_sim::ValidatedTrace {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/cache")
        .join(name);
    let bytes = std::fs::read(path).unwrap();
    parse_trace(&bytes, SimLimits::default()).unwrap()
}

#[test]
fn m2_golden_preserves_forced_eviction_counters() {
    let trace = fixture("m2-forced-eviction.jsonl");
    let result = simulate(
        &trace,
        &SimulationConfig {
            capacity_bytes: 65_536,
            policy: PolicySpec::Lru,
        },
    )
    .unwrap();
    assert_eq!(result.metrics.ordinary_demand_hits, 1);
    assert_eq!(result.metrics.demand_misses, 4);
    assert_eq!(result.metrics.admissions, 4);
    assert_eq!(result.metrics.evictions, 3);
    assert_eq!(result.metrics.demand_logical_bytes, 196_642);
    assert_eq!(result.metrics.demand_load_bytes, 196_625);
}

#[test]
fn perfect_router_golden_accounts_prefetch_as_physical_io() {
    let trace = fixture("router-perfect.jsonl");
    let result = simulate(
        &trace,
        &SimulationConfig {
            capacity_bytes: 4_096,
            policy: PolicySpec::RouterPrefetch {
                config: RouterPolicyConfig {
                    protected_fraction_ppm: 750_000,
                    minimum_score_ppm: 100_000,
                    max_experts_per_signal: 1,
                    max_pages_per_signal: 1,
                    max_prefetch_bytes_per_signal: 4_096,
                },
            },
        },
    )
    .unwrap();
    assert_eq!(result.metrics.demand_load_bytes, 0);
    assert_eq!(result.metrics.prefetch_load_bytes, 4_096);
    assert_eq!(result.metrics.total_physical_load_bytes, 4_096);
    assert_eq!(result.metrics.prefetch_useful_bytes, 4_096);
    assert_eq!(result.metrics.prefetch_wasted_bytes, 0);
}

#[test]
fn variable_golden_uses_exact_dp_and_rejects_uniform_min() {
    let trace = fixture("variable-byte.jsonl");
    assert_eq!(
        exact_variable_byte_cost(&trace, 4, SimLimits::default()).unwrap(),
        5
    );
    assert!(matches!(
        simulate(
            &trace,
            &SimulationConfig {
                capacity_bytes: 4,
                policy: PolicySpec::Belady,
            }
        ),
        Err(SimError::UnsupportedOracleGeometry(_))
    ));
}
