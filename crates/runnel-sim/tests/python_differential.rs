use std::io::Write as _;
use std::path::PathBuf;
use std::process::Command;

use runnel_sim::{
    ExpertPrediction, PageClass, PageDescriptor, PageId, PolicySpec, RouterPolicyConfig, SimLimits,
    SimulationConfig, TinyLfuConfig, TraceEvent, TraceHeader, serialize_trace, simulate,
};
use serde::Deserialize;

#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct OracleMetrics {
    demand_accesses: u64,
    ordinary_demand_hits: u64,
    demand_misses: u64,
    admissions: u64,
    bypasses: u64,
    evictions: u64,
    demand_load_bytes: u64,
    final_resident_charge_bytes: u64,
    peak_resident_charge_bytes: u64,
}

struct DifferentialCase<'a> {
    label: &'a str,
    geometry: &'a [(u64, u64)],
    demands: &'a [u32],
    capacity_bytes: u64,
    policy: PolicySpec,
    python_policy: &'a str,
    protected_fraction_ppm: u32,
    tiny: TinyLfuConfig,
}

fn canonical_trace(geometry: &[(u64, u64)], demands: &[u32]) -> Vec<u8> {
    let pages = geometry
        .iter()
        .enumerate()
        .map(|(index, (logical_bytes, charge_bytes))| PageDescriptor {
            id: PageId(u32::try_from(index).expect("small differential page catalog")),
            logical_bytes: *logical_bytes,
            charge_bytes: *charge_bytes,
            class: PageClass::Shared,
        })
        .collect::<Vec<_>>();
    let events = demands
        .iter()
        .enumerate()
        .map(|(index, page)| TraceEvent::Demand {
            sequence: u64::try_from(index).expect("small differential event stream"),
            request: 0,
            step: u64::try_from(index).expect("small differential step"),
            page: PageId(*page),
        })
        .collect::<Vec<_>>();
    let header = TraceHeader {
        kind: "header".to_owned(),
        schema: "runnel.cache-trace/1".to_owned(),
        trace_id: "python-differential".to_owned(),
        page_count: pages.len(),
        event_count: events.len(),
        charge_quantum: 1,
        prefetch_model: "instant-between-events-v1".to_owned(),
    };
    serialize_trace(&header, &pages, &events, SimLimits::default())
        .expect("serialize canonical differential trace")
}

fn run_python(
    trace_bytes: &[u8],
    policy: &str,
    capacity_bytes: u64,
    protected_fraction_ppm: u32,
    tiny: &TinyLfuConfig,
) -> OracleMetrics {
    let mut trace_file = tempfile::Builder::new()
        .prefix("runnel-python-differential-")
        .suffix(".jsonl")
        .tempfile()
        .expect("create differential trace file");
    trace_file
        .write_all(trace_bytes)
        .expect("write differential trace");
    trace_file.flush().expect("flush differential trace");

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracle/cache_policy.py");
    assert!(
        script.is_file(),
        "independent Python policy oracle is missing at {}",
        script.display()
    );
    let output = Command::new("python3")
        .arg(&script)
        .arg("--trace")
        .arg(trace_file.path())
        .arg("--policy")
        .arg(policy)
        .arg("--capacity-bytes")
        .arg(capacity_bytes.to_string())
        .arg("--protected-fraction-ppm")
        .arg(protected_fraction_ppm.to_string())
        .arg("--minimum-score-ppm")
        .arg("100000")
        .arg("--max-experts-per-signal")
        .arg("2")
        .arg("--sketch-depth")
        .arg(tiny.sketch_depth.to_string())
        .arg("--sketch-width")
        .arg(tiny.sketch_width.to_string())
        .arg("--sample-accesses")
        .arg(tiny.sample_accesses.to_string())
        .output()
        .unwrap_or_else(|error| {
            panic!("python3 is required for the differential test but could not start: {error}")
        });
    assert!(
        output.status.success(),
        "Python oracle failed for {policy}: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "Python oracle emitted invalid metrics for {policy}: {error}; stdout={}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn assert_trace_equal_to_python(
    label: &str,
    bytes: &[u8],
    capacity_bytes: u64,
    policy: PolicySpec,
    python_policy: &str,
    protected_fraction_ppm: u32,
    tiny: &TinyLfuConfig,
) -> OracleMetrics {
    let trace = runnel_sim::parse_trace(bytes, SimLimits::default())
        .expect("parse generated differential trace");
    let rust = simulate(
        &trace,
        &SimulationConfig {
            capacity_bytes,
            policy,
        },
    )
    .unwrap_or_else(|error| panic!("Rust simulation failed for {label}: {error}"));
    let python = run_python(
        bytes,
        python_policy,
        capacity_bytes,
        protected_fraction_ppm,
        tiny,
    );
    let expected = OracleMetrics {
        demand_accesses: rust.metrics.demand_accesses,
        ordinary_demand_hits: rust.metrics.ordinary_demand_hits,
        demand_misses: rust.metrics.demand_misses,
        admissions: rust.metrics.admissions,
        bypasses: rust.metrics.bypasses,
        evictions: rust.metrics.evictions,
        demand_load_bytes: rust.metrics.demand_load_bytes,
        final_resident_charge_bytes: rust.metrics.final_resident_charge_bytes,
        peak_resident_charge_bytes: rust.metrics.peak_resident_charge_bytes,
    };
    assert_eq!(python, expected, "independent oracle diverged for {label}");
    expected
}

fn assert_equal_to_python(case: DifferentialCase<'_>) {
    let bytes = canonical_trace(case.geometry, case.demands);
    let _ = assert_trace_equal_to_python(
        case.label,
        &bytes,
        case.capacity_bytes,
        case.policy,
        case.python_policy,
        case.protected_fraction_ppm,
        &case.tiny,
    );
}

#[test]
fn online_policies_match_the_independent_python_oracle() {
    let default_tiny = TinyLfuConfig {
        sketch_depth: 4,
        sketch_width: 64,
        sample_accesses: 100,
    };
    assert_equal_to_python(DifferentialCase {
        label: "variable-byte LRU multi-victim replay",
        geometry: &[(1, 2), (2, 3), (4, 5)],
        demands: &[0, 1, 0, 2, 2],
        capacity_bytes: 5,
        policy: PolicySpec::Lru,
        python_policy: "lru",
        protected_fraction_ppm: 750_000,
        tiny: default_tiny.clone(),
    });

    assert_equal_to_python(DifferentialCase {
        label: "variable-byte SLRU protected demotion replay",
        geometry: &[(3, 4), (3, 4), (3, 4), (3, 4), (2, 3), (4, 4)],
        demands: &[0, 1, 2, 3, 4, 0, 1, 2, 3, 5, 0],
        capacity_bytes: 20,
        policy: PolicySpec::Slru {
            protected_fraction_ppm: 750_000,
        },
        python_policy: "slru",
        protected_fraction_ppm: 750_000,
        tiny: default_tiny.clone(),
    });

    assert_equal_to_python(DifferentialCase {
        label: "TinyLFU strict density rejection",
        geometry: &[(7, 8), (7, 8), (7, 8)],
        demands: &[0, 1, 0, 1, 2, 0],
        capacity_bytes: 16,
        policy: PolicySpec::TinyLfu {
            config: default_tiny.clone(),
        },
        python_policy: "tiny-lfu",
        protected_fraction_ppm: 750_000,
        tiny: default_tiny,
    });

    let aging_tiny = TinyLfuConfig {
        sketch_depth: 4,
        sketch_width: 64,
        sample_accesses: 4,
    };
    assert_equal_to_python(DifferentialCase {
        label: "variable-byte TinyLFU aging and aggregate multi-victim replay",
        geometry: &[(2, 2), (1, 2), (4, 5), (3, 3)],
        demands: &[0, 1, 0, 1, 3, 0, 2],
        capacity_bytes: 8,
        policy: PolicySpec::TinyLfu {
            config: aging_tiny.clone(),
        },
        python_policy: "tiny-lfu",
        protected_fraction_ppm: 750_000,
        tiny: aging_tiny,
    });
}

#[test]
fn router_admit_matches_python_with_multiple_layers_and_outstanding_targets() {
    let pages = vec![
        PageDescriptor {
            id: PageId(0),
            logical_bytes: 1,
            charge_bytes: 1,
            class: PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal: 0,
            },
        },
        PageDescriptor {
            id: PageId(1),
            logical_bytes: 1,
            charge_bytes: 1,
            class: PageClass::Expert {
                layer: 0,
                expert: 1,
                ordinal: 0,
            },
        },
        PageDescriptor {
            id: PageId(2),
            logical_bytes: 1,
            charge_bytes: 1,
            class: PageClass::Expert {
                layer: 1,
                expert: 0,
                ordinal: 0,
            },
        },
        PageDescriptor {
            id: PageId(3),
            logical_bytes: 1,
            charge_bytes: 1,
            class: PageClass::Shared,
        },
    ];
    let events = vec![
        TraceEvent::Demand {
            sequence: 0,
            request: 7,
            step: 0,
            page: PageId(0),
        },
        TraceEvent::Demand {
            sequence: 1,
            request: 7,
            step: 0,
            page: PageId(2),
        },
        TraceEvent::RouterSignal {
            sequence: 2,
            request: 7,
            target_step: 2,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 900_000,
            }],
        },
        TraceEvent::RouterSignal {
            sequence: 3,
            request: 7,
            target_step: 1,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 600_000,
            }],
        },
        TraceEvent::RouterSignal {
            sequence: 4,
            request: 7,
            target_step: 1,
            layer: 1,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 300_000,
            }],
        },
        TraceEvent::Demand {
            sequence: 5,
            request: 7,
            step: 1,
            page: PageId(3),
        },
        TraceEvent::Demand {
            sequence: 6,
            request: 7,
            step: 2,
            page: PageId(1),
        },
        TraceEvent::Demand {
            sequence: 7,
            request: 7,
            step: 2,
            page: PageId(0),
        },
    ];
    let header = TraceHeader {
        kind: "header".to_owned(),
        schema: "runnel.cache-trace/1".to_owned(),
        trace_id: "python-router-differential".to_owned(),
        page_count: pages.len(),
        event_count: events.len(),
        charge_quantum: 1,
        prefetch_model: "instant-between-events-v1".to_owned(),
    };
    let bytes = serialize_trace(&header, &pages, &events, SimLimits::default())
        .expect("serialize router differential trace");
    let tiny = TinyLfuConfig {
        sketch_depth: 4,
        sketch_width: 64,
        sample_accesses: 100,
    };
    let metrics = assert_trace_equal_to_python(
        "router multi-layer and multi-target replay",
        &bytes,
        2,
        PolicySpec::RouterAdmit {
            config: RouterPolicyConfig {
                protected_fraction_ppm: 750_000,
                minimum_score_ppm: 100_000,
                max_experts_per_signal: 2,
                max_pages_per_signal: 1,
                max_prefetch_bytes_per_signal: 1,
            },
        },
        "router-admit",
        750_000,
        &tiny,
    );

    assert_eq!(metrics.ordinary_demand_hits, 1);
    assert_eq!(metrics.demand_misses, 4);
    assert_eq!(metrics.bypasses, 1);
    assert_eq!(metrics.evictions, 1);
}
