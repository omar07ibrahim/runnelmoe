use std::collections::{BTreeMap, BTreeSet};

use runnel_sim::{PageClass, TraceEvent, TraceFamily, generate_trace};
use sha2::{Digest, Sha256};

const PAGE_BYTES: u64 = 65_536;

fn measured_routes(generated: &runnel_sim::GeneratedTrace) -> Vec<[u32; 2]> {
    let experts_by_page = generated
        .trace
        .pages()
        .iter()
        .map(|page| match &page.class {
            PageClass::Expert { expert, .. } => (page.id, *expert),
            PageClass::Shared => panic!("synthetic catalog contains no shared pages"),
        })
        .collect::<BTreeMap<_, _>>();
    let demand_experts = generated
        .trace
        .events()
        .iter()
        .filter_map(|event| match event {
            TraceEvent::Demand { page, .. } => Some(experts_by_page[page]),
            TraceEvent::RouterSignal { .. } => None,
        })
        .collect::<Vec<_>>();
    demand_experts
        .chunks_exact(6)
        .map(|demands| {
            assert_eq!(&demands[0..3], &[demands[0]; 3]);
            assert_eq!(&demands[3..6], &[demands[3]; 3]);
            [demands[0], demands[3]]
        })
        .collect()
}

#[test]
fn every_family_is_deterministic_and_has_distinct_top_two_routes() {
    for family in TraceFamily::ALL {
        let first = generate_trace(family, 7, 48).unwrap();
        let second = generate_trace(family, 7, 48).unwrap();

        assert_eq!(first.canonical_bytes, second.canonical_bytes, "{family}");
        assert_eq!(first.seed_sha256, second.seed_sha256, "{family}");
        assert_eq!(
            first.full_route_sha256, second.full_route_sha256,
            "{family}"
        );
        assert_eq!(
            first.measured_route_sha256, second.measured_route_sha256,
            "{family}"
        );
        assert_eq!(measured_routes(&first).len(), 48);
        assert!(
            measured_routes(&first)
                .iter()
                .all(|route| route[0] != route[1]),
            "{family}"
        );
    }
}

#[test]
fn seed_contract_uses_the_frozen_domain_family_and_decimal_replicate() {
    let generated = generate_trace(TraceFamily::ScanPollution, 12, 1).unwrap();
    let mut expected = Sha256::new();
    expected.update(b"runnel.m3-trace/v1\0scan_pollution\0");
    expected.update(b"12");
    assert_eq!(generated.seed_sha256, hex::encode(expected.finalize()));
}

#[test]
fn cyclic_pressure_visits_every_permuted_expert_once_per_cycle() {
    let generated = generate_trace(TraceFamily::CyclicPressure, 2, 64).unwrap();
    let experts = measured_routes(&generated)
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>();
    assert_eq!(experts.len(), 128);
}

#[test]
fn each_scan_pollution_epoch_has_sixteen_routes_with_distinct_cold_experts() {
    let generated = generate_trace(TraceFamily::ScanPollution, 2, 64).unwrap();
    let routes = measured_routes(&generated);
    let cold_experts = routes[48..64]
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(cold_experts.len(), 32);
}

#[test]
fn catalog_has_three_uniform_pages_for_each_expert() {
    let generated = generate_trace(TraceFamily::StationaryZipf, 0, 8).unwrap();
    assert_eq!(generated.trace.pages().len(), 128 * 3);
    for (index, page) in generated.trace.pages().iter().enumerate() {
        assert_eq!(usize::try_from(page.id.0).unwrap(), index);
        assert_eq!(page.logical_bytes, PAGE_BYTES);
        assert_eq!(page.charge_bytes, PAGE_BYTES);
        assert_eq!(
            page.class,
            PageClass::Expert {
                layer: 0,
                expert: u32::try_from(index / 3).unwrap(),
                ordinal: u32::try_from(index % 3).unwrap(),
            }
        );
    }
}

#[test]
fn longer_generation_cannot_change_routes_or_signals_in_the_prefix() {
    for family in TraceFamily::ALL {
        let short = generate_trace(family, 11, 24).unwrap();
        let long = generate_trace(family, 11, 57).unwrap();
        assert_eq!(short.seed_sha256, long.seed_sha256, "{family}");
        assert_eq!(
            measured_routes(&short),
            measured_routes(&long)[..24],
            "{family}"
        );
        assert_eq!(
            short.trace.events(),
            &long.trace.events()[..short.trace.events().len()],
            "{family}"
        );

        let short_signals = short
            .trace
            .events()
            .iter()
            .filter_map(|event| match event {
                TraceEvent::RouterSignal {
                    target_step,
                    predictions,
                    ..
                } => Some((*target_step, predictions.clone())),
                TraceEvent::Demand { .. } => None,
            })
            .collect::<Vec<_>>();
        let long_prefix_signals = long
            .trace
            .events()
            .iter()
            .filter_map(|event| match event {
                TraceEvent::RouterSignal {
                    target_step,
                    predictions,
                    ..
                } if *target_step < 24 => Some((*target_step, predictions.clone())),
                TraceEvent::RouterSignal { .. } | TraceEvent::Demand { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(short_signals, long_prefix_signals, "{family}");
    }
}

#[test]
fn signals_are_causal_bounded_and_follow_the_prior_route() {
    let generated = generate_trace(TraceFamily::MarkovClusters, 3, 96).unwrap();
    let first_signal_target = generated
        .trace
        .events()
        .iter()
        .find_map(|event| match event {
            TraceEvent::RouterSignal { target_step, .. } => Some(*target_step),
            TraceEvent::Demand { .. } => None,
        });
    assert_eq!(first_signal_target, Some(0));

    let mut first_demand_sequence = BTreeMap::new();
    let mut last_demand_sequence = BTreeMap::new();
    for event in generated.trace.events() {
        if let TraceEvent::Demand { sequence, step, .. } = event {
            first_demand_sequence.entry(*step).or_insert(*sequence);
            last_demand_sequence.insert(*step, *sequence);
        }
    }

    for event in generated.trace.events() {
        if let TraceEvent::RouterSignal {
            sequence,
            request,
            target_step,
            layer,
            predictions,
        } = event
        {
            assert_eq!(*request, 0);
            assert_eq!(*layer, 0);
            assert!(*sequence < first_demand_sequence[target_step]);
            if *target_step != 0 {
                assert!(*sequence > last_demand_sequence[&(target_step - 1)]);
            }
            assert!((1..=2).contains(&predictions.len()));
            assert!(
                predictions
                    .iter()
                    .all(|prediction| prediction.score_ppm >= 100_000)
            );
            assert!(
                predictions
                    .iter()
                    .map(|prediction| u64::from(prediction.score_ppm))
                    .sum::<u64>()
                    <= 1_000_000
            );
            if predictions.len() == 2 {
                assert_ne!(predictions[0].expert, predictions[1].expert);
            }
        }
    }
}

#[test]
fn stable_small_trace_digests_are_pinned() {
    let expected = [
        (
            TraceFamily::StationaryZipf,
            "e4e0a5994b0434009170e60b3cd05108703963939c0ec7f73eb98b621dd37833",
            "70d2d41829d85ad7baa6b0065b3a485343d00206e411e2ea0716780935543bf2",
            "8f68c1bbc194d58f1ba13c7e80407d0d54e16d9be1f928b71962057bff7cd0ce",
            "86e4c9c6b325853ecd862ec8b073f2e3518414f5a084443bfaa218b4b2cb5caf",
        ),
        (
            TraceFamily::ScanPollution,
            "c56cf530d8d2b76a68950726e37cd991f73836b0e67fa2eee89c33efc185fae6",
            "1ffc1bc6a8081e42db78225cf66b1ea22cb2631a6641f0b30c164ff3d3c18adf",
            "869e23f7352701ffd321d4efb442f785160736657a0d154b11e95d054259354a",
            "612ef848864b5a1c0d2188ea7c77b413aa40793da3160d1b8164e1f003e1817e",
        ),
        (
            TraceFamily::PhaseShift,
            "85934bb0fb59179d8e2a57c8d87923ef166b82c158620d4bbbfc9202ddfc58ba",
            "d3f113c50f09baeb369353a8151297968b363ba8cd6ee0012af5936e9cee9bfa",
            "863cd7b2efefcfb62088d086473ecc887072d6f32b506f4d50427341d5e570c6",
            "ff1e2d102ae4860230ab49fe84d52c5c93675fe6fee2e828259255c56391d916",
        ),
        (
            TraceFamily::CyclicPressure,
            "7a608d1619c4c39a63eadd2d87a41ff96fabaa1d286b1fd88fdce19e7eebfea0",
            "88a6abd7a7b7631c66347ca7944d54cc63dff2ef1f27d12c8ba040535c9bbac2",
            "122b5c3c80ad783a2f17699645d7e4f8a956d2b2e132a14a32de4c1edcca56bb",
            "df5f9d17b82776451367f945b6b4793eb8e1c31da1de2bdf8fb1b813c511103d",
        ),
        (
            TraceFamily::MarkovClusters,
            "b73d32050cd2c2184c48370df1a8c1bb434573fff7426e11aa3fb456e00130a2",
            "307f52bf61586bc81393af7139ca1f7368a6fff9c97fc9b8443565e83fd4b97c",
            "1b3c92eeafa5a44b7d397f1526f1e1f888c4617f795127e27600ee97c36c453f",
            "b88007226552edf6cfae3a783e66a678f7bdb33840efa3fbf1e6479f207ed0ff",
        ),
        (
            TraceFamily::IidUniform,
            "c5e1a643b8abc8a24b97707092286a806e9df17bc3c5661f1c483b4c04cc7ad8",
            "b63ff9d33eca8729aa067336da3ed13cda423e071afa4d61fd9b42323284dea6",
            "706493e14a766bd2924422759e930cb60257da17438f9ce6565cf47645af9e7e",
            "e0a35b63fc3a524f361080790290142e692df6d9b05f52534c1651692c44b371",
        ),
    ];
    for (family, seed, full, measured, trace) in expected {
        let generated = generate_trace(family, 5, 16).unwrap();
        assert_eq!(generated.seed_sha256, seed);
        assert_eq!(generated.full_route_sha256, full);
        assert_eq!(generated.measured_route_sha256, measured);
        assert_eq!(generated.trace.sha256(), trace);
    }
}

#[test]
fn full_length_trace_stays_below_the_five_mib_parser_limit() {
    let generated = generate_trace(TraceFamily::StationaryZipf, 0, 4_096).unwrap();
    assert!(generated.canonical_bytes.len() < 5 * 1024 * 1024);
}

#[test]
fn invalid_measured_lengths_are_rejected_before_allocation() {
    assert!(generate_trace(TraceFamily::IidUniform, 0, 0).is_err());
    assert!(generate_trace(TraceFamily::IidUniform, 0, 4_097).is_err());
}
