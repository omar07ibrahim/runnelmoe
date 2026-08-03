use std::collections::VecDeque;

use runnel_sim::{
    ExpertPrediction, PageClass, PageDescriptor, PageId, PolicySpec, RouterPolicyConfig, SimLimits,
    SimulationConfig, SimulationResult, TinyLfuConfig, TraceEvent, TraceHeader, parse_trace,
    serialize_trace, simulate,
};

fn trace(
    page_geometry: &[(u64, u64, PageClass)],
    events: Vec<TraceEvent>,
) -> runnel_sim::ValidatedTrace {
    let pages = page_geometry
        .iter()
        .enumerate()
        .map(
            |(id, (logical_bytes, charge_bytes, class))| PageDescriptor {
                id: PageId(u32::try_from(id).unwrap()),
                logical_bytes: *logical_bytes,
                charge_bytes: *charge_bytes,
                class: class.clone(),
            },
        )
        .collect::<Vec<_>>();
    let header = TraceHeader {
        kind: "header".to_owned(),
        schema: "runnel.cache-trace/1".to_owned(),
        trace_id: "policy-test".to_owned(),
        page_count: pages.len(),
        event_count: events.len(),
        charge_quantum: 1,
        prefetch_model: "instant-between-events-v1".to_owned(),
    };
    let limits = SimLimits::default();
    let bytes = serialize_trace(&header, &pages, &events, limits).unwrap();
    parse_trace(&bytes, limits).unwrap()
}

fn demand_events(pages: &[u32]) -> Vec<TraceEvent> {
    pages
        .iter()
        .enumerate()
        .map(|(sequence, page)| TraceEvent::Demand {
            sequence: u64::try_from(sequence).unwrap(),
            request: 0,
            step: u64::try_from(sequence).unwrap(),
            page: PageId(*page),
        })
        .collect()
}

fn shared_pages(count: usize, logical: u64, charge: u64) -> Vec<(u64, u64, PageClass)> {
    vec![(logical, charge, PageClass::Shared); count]
}

fn run(
    trace: &runnel_sim::ValidatedTrace,
    capacity_bytes: u64,
    policy: PolicySpec,
) -> SimulationResult {
    simulate(
        trace,
        &SimulationConfig {
            capacity_bytes,
            policy,
        },
    )
    .unwrap()
}

fn router(prefetch: bool, max_pages: usize, max_bytes: u64) -> PolicySpec {
    let config = RouterPolicyConfig {
        protected_fraction_ppm: 750_000,
        minimum_score_ppm: 100_000,
        max_experts_per_signal: 2,
        max_pages_per_signal: max_pages,
        max_prefetch_bytes_per_signal: max_bytes,
    };
    if prefetch {
        PolicySpec::RouterPrefetch { config }
    } else {
        PolicySpec::RouterAdmit { config }
    }
}

#[test]
fn lru_matches_the_m2_forced_eviction_schedule() {
    let trace = trace(&shared_pages(3, 17, 64), demand_events(&[0, 1, 0, 2, 2]));
    let result = run(&trace, 64, PolicySpec::Lru);

    assert_eq!(result.metrics.demand_accesses, 5);
    assert_eq!(result.metrics.ordinary_demand_hits, 1);
    assert_eq!(result.metrics.demand_misses, 4);
    assert_eq!(result.metrics.admissions, 4);
    assert_eq!(result.metrics.evictions, 3);
    assert_eq!(result.metrics.demand_load_bytes, 4 * 17);
    assert_eq!(result.metrics.final_resident_charge_bytes, 64);
}

fn reference_lru(sequence: &[u32], slots: usize) -> (u64, u64, u64, u64, u64) {
    let mut resident = VecDeque::new();
    let mut hits = 0;
    let mut misses = 0;
    let mut admissions = 0;
    let mut evictions = 0;
    let mut bypasses = 0;
    for page in sequence {
        if let Some(index) = resident.iter().position(|candidate| candidate == page) {
            resident.remove(index);
            resident.push_back(*page);
            hits += 1;
        } else {
            misses += 1;
            if slots == 0 {
                bypasses += 1;
            } else {
                if resident.len() == slots {
                    resident.pop_front();
                    evictions += 1;
                }
                resident.push_back(*page);
                admissions += 1;
            }
        }
    }
    (hits, misses, admissions, evictions, bypasses)
}

#[test]
fn lru_is_exhaustively_equal_to_an_independent_vector_reference() {
    const PAGES: u32 = 3;
    for length in 1_u32..=6 {
        for mut encoded in 0..PAGES.pow(length) {
            let mut sequence = Vec::new();
            for _ in 0..length {
                sequence.push(encoded % PAGES);
                encoded /= PAGES;
            }
            let trace = trace(
                &shared_pages(PAGES as usize, 1, 1),
                demand_events(&sequence),
            );
            for slots in 0..=3 {
                let result = run(&trace, slots, PolicySpec::Lru);
                let expected = reference_lru(&sequence, usize::try_from(slots).unwrap());
                assert_eq!(
                    (
                        result.metrics.ordinary_demand_hits,
                        result.metrics.demand_misses,
                        result.metrics.admissions,
                        result.metrics.evictions,
                        result.metrics.bypasses,
                    ),
                    expected,
                    "sequence={sequence:?}, slots={slots}"
                );
            }
        }
    }
}

#[test]
fn slru_protects_reused_pages_from_scan_pollution() {
    let trace = trace(
        &shared_pages(5, 1, 1),
        demand_events(&[0, 1, 0, 2, 3, 4, 0]),
    );
    let lru = run(&trace, 3, PolicySpec::Lru);
    let slru = run(
        &trace,
        3,
        PolicySpec::Slru {
            protected_fraction_ppm: 750_000,
        },
    );

    assert_eq!(lru.metrics.ordinary_demand_hits, 1);
    assert_eq!(slru.metrics.ordinary_demand_hits, 2);
    assert!(slru.metrics.demand_load_bytes < lru.metrics.demand_load_bytes);
}

#[test]
fn tinylfu_rejects_a_one_hit_scan_candidate_atomically() {
    let trace = trace(&shared_pages(3, 7, 8), demand_events(&[0, 1, 0, 1, 2, 0]));
    let lru = run(&trace, 16, PolicySpec::Lru);
    let tiny = run(
        &trace,
        16,
        PolicySpec::TinyLfu {
            config: TinyLfuConfig {
                sketch_depth: 4,
                sketch_width: 64,
                sample_accesses: 100,
            },
        },
    );

    assert_eq!(lru.metrics.demand_misses, 4);
    assert_eq!(tiny.metrics.demand_misses, 3);
    assert_eq!(tiny.metrics.bypasses, 1);
    assert_eq!(tiny.metrics.evictions, 0);
}

#[test]
fn router_admission_preserves_a_predicted_resident() {
    let pages = vec![
        (
            1,
            1,
            PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal: 0,
            },
        ),
        (
            1,
            1,
            PageClass::Expert {
                layer: 0,
                expert: 1,
                ordinal: 0,
            },
        ),
    ];
    let events = vec![
        TraceEvent::Demand {
            sequence: 0,
            request: 0,
            step: 0,
            page: PageId(0),
        },
        TraceEvent::RouterSignal {
            sequence: 1,
            request: 0,
            target_step: 1,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 900_000,
            }],
        },
        TraceEvent::Demand {
            sequence: 2,
            request: 0,
            step: 1,
            page: PageId(1),
        },
        TraceEvent::Demand {
            sequence: 3,
            request: 0,
            step: 1,
            page: PageId(0),
        },
    ];
    let trace = trace(&pages, events);
    let slru = run(
        &trace,
        1,
        PolicySpec::Slru {
            protected_fraction_ppm: 750_000,
        },
    );
    let predictive = run(&trace, 1, router(false, 1, 1));

    assert_eq!(slru.metrics.demand_misses, 3);
    assert_eq!(predictive.metrics.demand_misses, 2);
    assert_eq!(predictive.metrics.ordinary_demand_hits, 1);
    assert_eq!(predictive.metrics.bypasses, 1);
}

#[test]
fn router_keeps_multilayer_and_outstanding_targets_with_exact_metadata_charge() {
    let pages = vec![
        (
            1,
            1,
            PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal: 0,
            },
        ),
        (
            1,
            1,
            PageClass::Expert {
                layer: 1,
                expert: 0,
                ordinal: 0,
            },
        ),
        (1, 1, PageClass::Shared),
        (1, 1, PageClass::Shared),
        (1, 1, PageClass::Shared),
    ];
    let events = vec![
        TraceEvent::Demand {
            sequence: 0,
            request: 0,
            step: 0,
            page: PageId(0),
        },
        TraceEvent::Demand {
            sequence: 1,
            request: 0,
            step: 0,
            page: PageId(1),
        },
        TraceEvent::RouterSignal {
            sequence: 2,
            request: 0,
            target_step: 1,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 900_000,
            }],
        },
        TraceEvent::RouterSignal {
            sequence: 3,
            request: 0,
            target_step: 1,
            layer: 1,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 900_000,
            }],
        },
        TraceEvent::RouterSignal {
            sequence: 4,
            request: 0,
            target_step: 2,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 900_000,
            }],
        },
        TraceEvent::Demand {
            sequence: 5,
            request: 0,
            step: 1,
            page: PageId(2),
        },
        TraceEvent::Demand {
            sequence: 6,
            request: 0,
            step: 1,
            page: PageId(3),
        },
        TraceEvent::Demand {
            sequence: 7,
            request: 0,
            step: 1,
            page: PageId(4),
        },
        TraceEvent::Demand {
            sequence: 8,
            request: 0,
            step: 1,
            page: PageId(1),
        },
        TraceEvent::Demand {
            sequence: 9,
            request: 0,
            step: 2,
            page: PageId(0),
        },
    ];
    let result = run(&trace(&pages, events), 2, router(false, 2, 2));

    assert_eq!(result.metrics.demand_accesses, 7);
    assert_eq!(result.metrics.ordinary_demand_hits, 2);
    assert_eq!(result.metrics.demand_misses, 5);
    assert_eq!(result.metrics.admissions, 2);
    assert_eq!(result.metrics.bypasses, 3);
    assert_eq!(result.metrics.evictions, 0);
    assert_eq!(result.metrics.demand_load_bytes, 5);
    assert_eq!(result.metrics.policy_metadata_bytes, 96);
    assert_eq!(result.metrics.policy_metadata_limit_bytes, 96);
}

#[test]
fn router_does_not_retain_or_charge_an_empty_post_threshold_signal() {
    let pages = vec![
        (
            1,
            1,
            PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal: 0,
            },
        ),
        (1, 1, PageClass::Shared),
    ];
    let events = vec![
        TraceEvent::RouterSignal {
            sequence: 0,
            request: 0,
            target_step: 0,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 99_999,
            }],
        },
        TraceEvent::Demand {
            sequence: 1,
            request: 0,
            step: 0,
            page: PageId(1),
        },
    ];
    let result = run(&trace(&pages, events), 1, router(false, 1, 1));

    assert_eq!(result.metrics.policy_metadata_bytes, 0);
    assert_eq!(result.metrics.policy_metadata_limit_bytes, 0);
}

#[test]
fn prefetch_decision_digest_commits_to_offered_page_identities() {
    fn identity_trace(expert_zero_page: u32) -> runnel_sim::ValidatedTrace {
        let other_page = 1 - expert_zero_page;
        let mut pages = vec![(1, 1, PageClass::Shared); 3];
        pages[usize::try_from(expert_zero_page).unwrap()].2 = PageClass::Expert {
            layer: 0,
            expert: 0,
            ordinal: 0,
        };
        pages[usize::try_from(other_page).unwrap()].2 = PageClass::Expert {
            layer: 0,
            expert: 1,
            ordinal: 0,
        };
        trace(
            &pages,
            vec![
                TraceEvent::RouterSignal {
                    sequence: 0,
                    request: 0,
                    target_step: 0,
                    layer: 0,
                    predictions: vec![ExpertPrediction {
                        expert: 0,
                        score_ppm: 900_000,
                    }],
                },
                TraceEvent::Demand {
                    sequence: 1,
                    request: 0,
                    step: 0,
                    page: PageId(2),
                },
            ],
        )
    }

    let first = run(&identity_trace(0), 3, router(true, 1, 1));
    let second = run(&identity_trace(1), 3, router(true, 1, 1));

    assert_eq!(first.metrics, second.metrics);
    assert_ne!(first.trace_sha256, second.trace_sha256);
    assert_ne!(first.decision_sha256, second.decision_sha256);
}

#[test]
fn router_without_signals_has_identical_slru_decisions() {
    let trace = trace(
        &shared_pages(4, 1, 1),
        demand_events(&[0, 1, 0, 2, 3, 0, 1]),
    );
    let slru = run(
        &trace,
        3,
        PolicySpec::Slru {
            protected_fraction_ppm: 750_000,
        },
    );
    let router = run(&trace, 3, router(false, 2, 2));

    let mut router_metrics = router.metrics.clone();
    router_metrics.policy_metadata_bytes = slru.metrics.policy_metadata_bytes;
    router_metrics.policy_metadata_limit_bytes = slru.metrics.policy_metadata_limit_bytes;
    assert_eq!(router_metrics, slru.metrics);
    assert_eq!(router.decision_sha256, slru.decision_sha256);
}

#[test]
fn perfect_prefetch_moves_io_without_hiding_it() {
    let pages = vec![(
        7,
        8,
        PageClass::Expert {
            layer: 0,
            expert: 3,
            ordinal: 0,
        },
    )];
    let events = vec![
        TraceEvent::RouterSignal {
            sequence: 0,
            request: 0,
            target_step: 0,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 3,
                score_ppm: 900_000,
            }],
        },
        TraceEvent::Demand {
            sequence: 1,
            request: 0,
            step: 0,
            page: PageId(0),
        },
    ];
    let trace = trace(&pages, events);
    let lru = run(&trace, 8, PolicySpec::Lru);
    let prefetched = run(&trace, 8, router(true, 1, 7));

    assert_eq!(lru.metrics.demand_load_bytes, 7);
    assert_eq!(prefetched.metrics.demand_load_bytes, 0);
    assert_eq!(prefetched.metrics.prefetch_load_bytes, 7);
    assert_eq!(prefetched.metrics.total_physical_load_bytes, 7);
    assert_eq!(
        prefetched.metrics.total_physical_load_bytes,
        lru.metrics.total_physical_load_bytes
    );
    assert_eq!(prefetched.metrics.useful_prefetch_hits, 1);
    assert_eq!(prefetched.metrics.prefetch_useful, 1);
    assert_eq!(prefetched.metrics.prefetch_wasted, 0);
}

#[test]
fn unused_prefetch_is_wasted_at_finalization() {
    let pages = vec![
        (
            7,
            8,
            PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal: 0,
            },
        ),
        (
            7,
            8,
            PageClass::Expert {
                layer: 0,
                expert: 1,
                ordinal: 0,
            },
        ),
    ];
    let events = vec![
        TraceEvent::RouterSignal {
            sequence: 0,
            request: 0,
            target_step: 0,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 900_000,
            }],
        },
        TraceEvent::Demand {
            sequence: 1,
            request: 0,
            step: 0,
            page: PageId(1),
        },
    ];
    let trace = trace(&pages, events);
    let result = run(&trace, 16, router(true, 1, 7));

    assert_eq!(result.metrics.prefetch_admitted, 1);
    assert_eq!(result.metrics.prefetch_useful, 0);
    assert_eq!(result.metrics.prefetch_wasted, 1);
    assert_eq!(result.metrics.prefetch_load_bytes, 7);
    assert_eq!(result.metrics.prefetch_wasted_bytes, 7);
    assert_eq!(result.metrics.total_physical_load_bytes, 14);
}

#[test]
fn expert_group_prefetch_is_all_or_none() {
    let mut pages = Vec::new();
    for ordinal in 0..3 {
        pages.push((
            1,
            1,
            PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal,
            },
        ));
    }
    pages.push((
        1,
        1,
        PageClass::Expert {
            layer: 0,
            expert: 1,
            ordinal: 0,
        },
    ));
    let trace = trace(
        &pages,
        vec![
            TraceEvent::RouterSignal {
                sequence: 0,
                request: 0,
                target_step: 0,
                layer: 0,
                predictions: vec![ExpertPrediction {
                    expert: 0,
                    score_ppm: 900_000,
                }],
            },
            TraceEvent::Demand {
                sequence: 1,
                request: 0,
                step: 0,
                page: PageId(3),
            },
        ],
    );
    let result = run(&trace, 2, router(true, 3, 3));

    assert_eq!(result.metrics.prefetch_admitted, 0);
    assert_eq!(result.metrics.prefetch_dropped, 3);
    assert_eq!(result.metrics.prefetch_load_bytes, 0);

    let page_limited = run(&trace, 3, router(true, 2, 3));
    assert_eq!(page_limited.metrics.prefetch_admitted, 0);
    assert_eq!(page_limited.metrics.prefetch_dropped, 3);
    assert_eq!(page_limited.metrics.prefetch_load_bytes, 0);
}

#[test]
fn scaling_all_byte_units_preserves_policy_decisions() {
    let sequence = demand_events(&[0, 1, 0, 2, 0, 1]);
    let small = trace(&shared_pages(3, 5, 8), sequence.clone());
    let large = trace(&shared_pages(3, 20, 32), sequence);
    let small_result = run(&small, 16, PolicySpec::Lru);
    let large_result = run(&large, 64, PolicySpec::Lru);

    assert_eq!(small_result.decision_sha256, large_result.decision_sha256);
    assert_eq!(
        small_result.metrics.demand_accesses,
        large_result.metrics.demand_accesses
    );
    assert_eq!(
        small_result.metrics.demand_misses,
        large_result.metrics.demand_misses
    );
    assert_eq!(
        small_result.metrics.evictions,
        large_result.metrics.evictions
    );
    assert_eq!(
        large_result.metrics.demand_logical_bytes,
        small_result.metrics.demand_logical_bytes * 4
    );
    assert_eq!(
        large_result.metrics.demand_load_bytes,
        small_result.metrics.demand_load_bytes * 4
    );
    assert_eq!(
        large_result.metrics.peak_resident_charge_bytes,
        small_result.metrics.peak_resident_charge_bytes * 4
    );
}

#[test]
fn invalid_policy_allocations_fail_before_replay() {
    let trace = trace(&shared_pages(1, 1, 1), demand_events(&[0]));
    assert!(
        simulate(
            &trace,
            &SimulationConfig {
                capacity_bytes: 1,
                policy: PolicySpec::Slru {
                    protected_fraction_ppm: 1_000_001,
                },
            }
        )
        .is_err()
    );
    assert!(
        simulate(
            &trace,
            &SimulationConfig {
                capacity_bytes: 1,
                policy: PolicySpec::TinyLfu {
                    config: TinyLfuConfig {
                        sketch_depth: 9,
                        sketch_width: usize::MAX,
                        sample_accesses: 0,
                    },
                },
            }
        )
        .is_err()
    );
}
