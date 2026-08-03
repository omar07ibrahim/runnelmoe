use std::collections::BTreeMap;

use runnel_sim::{
    PageClass, PageDescriptor, PageId, PolicySpec, SimError, SimLimits, SimulationConfig,
    TraceEvent, TraceHeader, exact_variable_byte_cost, parse_trace, serialize_trace, simulate,
};

const SCHEMA: &str = "runnel.cache-trace/1";
const PREFETCH_MODEL: &str = "instant-between-events-v1";

fn make_trace(geometry: &[(u64, u64)], demands: &[u32]) -> runnel_sim::ValidatedTrace {
    let pages: Vec<PageDescriptor> = geometry
        .iter()
        .enumerate()
        .map(|(id, (charge_bytes, logical_bytes))| PageDescriptor {
            id: PageId(u32::try_from(id).expect("small test page id")),
            logical_bytes: *logical_bytes,
            charge_bytes: *charge_bytes,
            class: PageClass::Shared,
        })
        .collect();
    let events: Vec<TraceEvent> = demands
        .iter()
        .enumerate()
        .map(|(sequence, page)| TraceEvent::Demand {
            sequence: u64::try_from(sequence).expect("small test sequence"),
            request: 0,
            step: u64::try_from(sequence).expect("small test step"),
            page: PageId(*page),
        })
        .collect();
    make_trace_from_parts(pages, events)
}

fn make_trace_from_parts(
    pages: Vec<PageDescriptor>,
    events: Vec<TraceEvent>,
) -> runnel_sim::ValidatedTrace {
    let header = TraceHeader {
        kind: "header".to_owned(),
        schema: SCHEMA.to_owned(),
        trace_id: "oracle-test".to_owned(),
        page_count: pages.len(),
        event_count: events.len(),
        charge_quantum: 1,
        prefetch_model: PREFETCH_MODEL.to_owned(),
    };
    let limits = SimLimits::default();
    let bytes = serialize_trace(&header, &pages, &events, limits).expect("serialize test trace");
    parse_trace(&bytes, limits).expect("parse test trace")
}

fn belady(trace: &runnel_sim::ValidatedTrace, capacity_bytes: u64) -> runnel_sim::SimulationResult {
    simulate(
        trace,
        &SimulationConfig {
            capacity_bytes,
            policy: PolicySpec::Belady,
        },
    )
    .expect("uniform Bélády replay")
}

fn brute_uniform_misses(sequence: &[u32], slots: usize) -> u64 {
    fn visit(
        sequence: &[u32],
        slots: usize,
        position: usize,
        residents: u32,
        memo: &mut BTreeMap<(usize, u32), u64>,
    ) -> u64 {
        if position == sequence.len() {
            return 0;
        }
        if let Some(cost) = memo.get(&(position, residents)) {
            return *cost;
        }

        let page = sequence[position];
        let page_mask = 1_u32 << page;
        let cost = if residents & page_mask != 0 {
            visit(sequence, slots, position + 1, residents, memo)
        } else {
            let mut continuation = visit(sequence, slots, position + 1, residents, memo);
            if slots != 0 {
                if residents.count_ones() < u32::try_from(slots).expect("small slot count") {
                    continuation = continuation.min(visit(
                        sequence,
                        slots,
                        position + 1,
                        residents | page_mask,
                        memo,
                    ));
                } else {
                    let mut victims = residents;
                    while victims != 0 {
                        let victim = 1_u32 << victims.trailing_zeros();
                        continuation = continuation.min(visit(
                            sequence,
                            slots,
                            position + 1,
                            (residents ^ victim) | page_mask,
                            memo,
                        ));
                        victims &= victims - 1;
                    }
                }
            }
            1 + continuation
        };
        memo.insert((position, residents), cost);
        cost
    }

    visit(sequence, slots, 0, 0, &mut BTreeMap::new())
}

#[test]
fn belady_is_exhaustively_equal_to_independent_brute_force() {
    const PAGE_COUNT: u32 = 3;
    const CHARGE_BYTES: u64 = 8;
    const LOGICAL_BYTES: u64 = 7;
    let geometry = vec![(CHARGE_BYTES, LOGICAL_BYTES); PAGE_COUNT as usize];

    for length in 1_u32..=6 {
        let sequence_count = PAGE_COUNT.pow(length);
        for mut encoded in 0..sequence_count {
            let mut sequence = Vec::with_capacity(length as usize);
            for _ in 0..length {
                sequence.push(encoded % PAGE_COUNT);
                encoded /= PAGE_COUNT;
            }
            let trace = make_trace(&geometry, &sequence);
            for slots in 0_usize..=3 {
                let expected_misses = brute_uniform_misses(&sequence, slots);
                let capacity_bytes = u64::try_from(slots).expect("small slot count") * CHARGE_BYTES;
                let result = belady(&trace, capacity_bytes);
                assert!(result.oracle_optimal);
                assert_eq!(
                    result.metrics.demand_load_bytes,
                    expected_misses * LOGICAL_BYTES,
                    "sequence={sequence:?}, slots={slots}"
                );
                assert_eq!(
                    exact_variable_byte_cost(&trace, capacity_bytes, SimLimits::default())
                        .expect("equal-size exact DP"),
                    expected_misses * LOGICAL_BYTES,
                    "DP sequence={sequence:?}, slots={slots}"
                );
            }
        }
    }
}

#[test]
fn belady_bypasses_a_tied_candidate_without_churning_residents() {
    let trace = make_trace(&[(1, 1); 3], &[0, 1, 2]);
    let result = belady(&trace, 1);

    assert_eq!(result.metrics.demand_misses, 3);
    assert_eq!(result.metrics.admissions, 1);
    assert_eq!(result.metrics.bypasses, 2);
    assert_eq!(result.metrics.evictions, 0);
    assert_eq!(result.metrics.final_resident_charge_bytes, 1);
    assert_eq!(result.metrics.total_physical_load_bytes, 3);
}

#[test]
fn belady_rejects_nonuniform_charge_and_logical_cost() {
    let charge_trace = make_trace(&[(8, 8), (16, 8)], &[0, 1]);
    let logical_trace = make_trace(&[(8, 7), (8, 8)], &[0, 1]);
    let config = SimulationConfig {
        capacity_bytes: 16,
        policy: PolicySpec::Belady,
    };

    assert!(matches!(
        simulate(&charge_trace, &config),
        Err(SimError::UnsupportedOracleGeometry(_))
    ));
    assert!(matches!(
        simulate(&logical_trace, &config),
        Err(SimError::UnsupportedOracleGeometry(_))
    ));
}

#[test]
fn belady_ignores_causal_router_signals() {
    let pages = vec![
        PageDescriptor {
            id: PageId(0),
            logical_bytes: 4,
            charge_bytes: 4,
            class: PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal: 0,
            },
        },
        PageDescriptor {
            id: PageId(1),
            logical_bytes: 4,
            charge_bytes: 4,
            class: PageClass::Expert {
                layer: 0,
                expert: 1,
                ordinal: 0,
            },
        },
    ];
    let without_signal = make_trace_from_parts(
        pages.clone(),
        vec![
            TraceEvent::Demand {
                sequence: 0,
                request: 5,
                step: 0,
                page: PageId(0),
            },
            TraceEvent::Demand {
                sequence: 1,
                request: 5,
                step: 1,
                page: PageId(1),
            },
        ],
    );
    let with_signal = make_trace_from_parts(
        pages,
        vec![
            TraceEvent::Demand {
                sequence: 0,
                request: 5,
                step: 0,
                page: PageId(0),
            },
            TraceEvent::RouterSignal {
                sequence: 1,
                request: 5,
                target_step: 1,
                layer: 0,
                predictions: vec![runnel_sim::ExpertPrediction {
                    expert: 1,
                    score_ppm: 900_000,
                }],
            },
            TraceEvent::Demand {
                sequence: 2,
                request: 5,
                step: 1,
                page: PageId(1),
            },
        ],
    );

    let baseline = belady(&without_signal, 4);
    let signaled = belady(&with_signal, 4);
    assert_eq!(signaled.metrics, baseline.metrics);
    assert_eq!(signaled.decision_sha256, baseline.decision_sha256);
}

#[test]
fn exact_dp_handles_a_variable_size_and_cost_counterexample() {
    // A next-use-only extension of MIN bypasses three-byte C because C is used
    // after A and B.  The byte-cost optimum instead evicts one one-byte page,
    // admits C, and pays five logical bytes rather than six.
    let trace = make_trace(&[(1, 1), (1, 1), (3, 2)], &[0, 1, 2, 0, 1, 2]);
    assert_eq!(
        exact_variable_byte_cost(&trace, 4, SimLimits::default())
            .expect("tiny variable-byte optimum"),
        5
    );
}

#[test]
fn exact_dp_enforces_page_demand_and_state_limits() {
    let too_many_pages = make_trace(&vec![(1, 1); 19], &(0_u32..19).collect::<Vec<_>>());
    assert!(matches!(
        exact_variable_byte_cost(&too_many_pages, 4, SimLimits::default()),
        Err(SimError::ExactOracleLimit(_))
    ));

    let too_many_demands = make_trace(&[(1, 1)], &vec![0; 201]);
    assert!(matches!(
        exact_variable_byte_cost(&too_many_demands, 1, SimLimits::default()),
        Err(SimError::ExactOracleLimit(_))
    ));

    let trace = make_trace(&[(1, 1)], &[0]);
    let tiny_limit = SimLimits {
        max_exact_oracle_states: 1,
        ..SimLimits::default()
    };
    assert!(matches!(
        exact_variable_byte_cost(&trace, 1, tiny_limit),
        Err(SimError::ExactOracleLimit(_))
    ));

    let overflowing_cost = make_trace(&[(u64::MAX, u64::MAX)], &[0, 0]);
    assert!(matches!(
        exact_variable_byte_cost(&overflowing_cost, 0, SimLimits::default()),
        Err(SimError::CounterOverflow("exact variable-byte cost"))
    ));
}
