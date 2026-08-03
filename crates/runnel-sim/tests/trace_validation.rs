use runnel_sim::{
    ExpertPrediction, PageClass, PageDescriptor, PageId, SimError, SimLimits, TraceEvent,
    TraceHeader, parse_trace, serialize_trace,
};
use sha2::{Digest as _, Sha256};

fn valid_parts() -> (TraceHeader, Vec<PageDescriptor>, Vec<TraceEvent>) {
    let pages = vec![
        PageDescriptor {
            id: PageId(0),
            logical_bytes: 3_000,
            charge_bytes: 4_096,
            class: PageClass::Shared,
        },
        PageDescriptor {
            id: PageId(1),
            logical_bytes: 4_096,
            charge_bytes: 4_096,
            class: PageClass::Expert {
                layer: 0,
                expert: 0,
                ordinal: 1,
            },
        },
        PageDescriptor {
            id: PageId(2),
            logical_bytes: 2_048,
            charge_bytes: 4_096,
            class: PageClass::Expert {
                layer: 0,
                expert: 1,
                ordinal: 0,
            },
        },
    ];
    let events = vec![
        TraceEvent::RouterSignal {
            sequence: 0,
            request: 7,
            target_step: 0,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 0,
                score_ppm: 750_000,
            }],
        },
        TraceEvent::Demand {
            sequence: 1,
            request: 7,
            step: 0,
            page: PageId(1),
        },
        TraceEvent::Demand {
            sequence: 2,
            request: 7,
            step: 0,
            page: PageId(0),
        },
        TraceEvent::RouterSignal {
            sequence: 3,
            request: 7,
            target_step: 1,
            layer: 0,
            predictions: vec![ExpertPrediction {
                expert: 1,
                score_ppm: 600_000,
            }],
        },
        TraceEvent::Demand {
            sequence: 4,
            request: 7,
            step: 1,
            page: PageId(2),
        },
    ];
    let header = TraceHeader {
        kind: "header".into(),
        schema: "runnel.cache-trace/1".into(),
        trace_id: "trace-validation.01".into(),
        page_count: pages.len(),
        event_count: events.len(),
        charge_quantum: 4_096,
        prefetch_model: "instant-between-events-v1".into(),
    };
    (header, pages, events)
}

fn raw_trace(header: &TraceHeader, pages: &[PageDescriptor], events: &[TraceEvent]) -> Vec<u8> {
    let mut bytes = Vec::new();
    serde_json::to_writer(&mut bytes, header).unwrap();
    bytes.push(b'\n');
    for page in pages {
        serde_json::to_writer(&mut bytes, page).unwrap();
        bytes.push(b'\n');
    }
    for event in events {
        serde_json::to_writer(&mut bytes, event).unwrap();
        bytes.push(b'\n');
    }
    bytes
}

fn assert_invalid(result: Result<Vec<u8>, SimError>) {
    assert!(matches!(result, Err(SimError::InvalidTrace(_))));
}

#[test]
fn canonical_serializer_round_trips_and_hashes_exact_bytes() {
    let (header, pages, events) = valid_parts();
    let bytes = serialize_trace(&header, &pages, &events, SimLimits::default()).unwrap();

    assert!(bytes.is_ascii());
    assert!(bytes.ends_with(b"\n"));
    assert!(!bytes.contains(&b'\r'));
    let parsed = parse_trace(&bytes, SimLimits::default()).unwrap();
    assert_eq!(parsed.header(), &header);
    assert_eq!(parsed.pages(), pages);
    assert_eq!(parsed.events(), events);
    assert_eq!(parsed.sha256(), hex::encode(Sha256::digest(&bytes)));
    assert_eq!(
        serialize_trace(
            parsed.header(),
            parsed.pages(),
            parsed.events(),
            SimLimits::default()
        )
        .unwrap(),
        bytes
    );
}

#[test]
fn framing_rejects_missing_lf_cr_non_ascii_and_extra_records() {
    let (header, pages, events) = valid_parts();
    let bytes = serialize_trace(&header, &pages, &events, SimLimits::default()).unwrap();

    let mut missing_lf = bytes.clone();
    missing_lf.pop();
    assert!(parse_trace(&missing_lf, SimLimits::default()).is_err());

    let mut with_cr = bytes.clone();
    with_cr.insert(with_cr.len() - 1, b'\r');
    assert!(parse_trace(&with_cr, SimLimits::default()).is_err());

    let mut non_ascii = bytes.clone();
    non_ascii.insert(non_ascii.len() - 1, 0x80);
    assert!(parse_trace(&non_ascii, SimLimits::default()).is_err());

    let mut extra = bytes;
    extra.extend_from_slice(b"{}\n");
    assert!(parse_trace(&extra, SimLimits::default()).is_err());
}

#[test]
fn all_parser_limits_are_enforced_before_replay() {
    let (header, pages, events) = valid_parts();
    let bytes = serialize_trace(&header, &pages, &events, SimLimits::default()).unwrap();

    let limits = SimLimits {
        max_trace_bytes: bytes.len() - 1,
        ..SimLimits::default()
    };
    assert!(parse_trace(&bytes, limits).is_err());

    let limits = SimLimits {
        max_line_bytes: bytes.iter().position(|byte| *byte == b'\n').unwrap() - 1,
        ..SimLimits::default()
    };
    assert!(parse_trace(&bytes, limits).is_err());

    let limits = SimLimits {
        max_pages: pages.len() - 1,
        ..SimLimits::default()
    };
    assert!(parse_trace(&bytes, limits).is_err());

    let limits = SimLimits {
        max_events: events.len() - 1,
        ..SimLimits::default()
    };
    assert!(parse_trace(&bytes, limits).is_err());

    let limits = SimLimits {
        max_predictions_per_signal: 0,
        ..SimLimits::default()
    };
    assert!(parse_trace(&bytes, limits).is_err());
}

#[test]
fn noncanonical_unknown_and_duplicate_fields_are_rejected() {
    let (header, pages, events) = valid_parts();
    let bytes = serialize_trace(&header, &pages, &events, SimLimits::default()).unwrap();

    let mut spaced = bytes.clone();
    spaced.insert(1, b' ');
    assert!(matches!(
        parse_trace(&spaced, SimLimits::default()),
        Err(SimError::InvalidTrace(_))
    ));

    let text = String::from_utf8(bytes).unwrap();
    let unknown = text.replacen(
        "{\"kind\":\"header\",",
        "{\"kind\":\"header\",\"surprise\":0,",
        1,
    );
    assert!(matches!(
        parse_trace(unknown.as_bytes(), SimLimits::default()),
        Err(SimError::Json(_))
    ));

    let duplicate = text.replacen(
        "{\"kind\":\"header\",",
        "{\"kind\":\"header\",\"kind\":\"header\",",
        1,
    );
    assert!(matches!(
        parse_trace(duplicate.as_bytes(), SimLimits::default()),
        Err(SimError::Json(_))
    ));
}

#[test]
fn header_literals_identifier_counts_and_quantum_are_closed() {
    let (header, pages, events) = valid_parts();

    for mutate in [
        |header: &mut TraceHeader| header.kind = "trace".into(),
        |header: &mut TraceHeader| header.schema = "runnel.cache-trace/2".into(),
        |header: &mut TraceHeader| header.prefetch_model = "eventual".into(),
        |header: &mut TraceHeader| header.trace_id = "Uppercase".into(),
        |header: &mut TraceHeader| header.charge_quantum = 0,
    ] {
        let mut invalid = header.clone();
        mutate(&mut invalid);
        assert_invalid(serialize_trace(
            &invalid,
            &pages,
            &events,
            SimLimits::default(),
        ));
    }

    for invalid_id in ["", "-leading", "has/slash", "contains space"] {
        let mut invalid = header.clone();
        invalid.trace_id = invalid_id.into();
        assert_invalid(serialize_trace(
            &invalid,
            &pages,
            &events,
            SimLimits::default(),
        ));
    }
    let mut too_long = header.clone();
    too_long.trace_id = "a".repeat(65);
    assert_invalid(serialize_trace(
        &too_long,
        &pages,
        &events,
        SimLimits::default(),
    ));

    let mut wrong_count = header.clone();
    wrong_count.page_count += 1;
    assert_invalid(serialize_trace(
        &wrong_count,
        &pages,
        &events,
        SimLimits::default(),
    ));
    let mut wrong_count = header;
    wrong_count.event_count -= 1;
    assert_invalid(serialize_trace(
        &wrong_count,
        &pages,
        &events,
        SimLimits::default(),
    ));
}

#[test]
fn catalog_requires_dense_ids_valid_byte_geometry_and_unique_expert_tuples() {
    let (header, pages, events) = valid_parts();
    let check = |mutate: fn(&mut Vec<PageDescriptor>)| {
        let mut invalid = pages.clone();
        mutate(&mut invalid);
        assert_invalid(serialize_trace(
            &header,
            &invalid,
            &events,
            SimLimits::default(),
        ));
    };

    check(|pages| pages[1].id = PageId(2));
    check(|pages| pages[0].logical_bytes = 0);
    check(|pages| pages[0].charge_bytes = 0);
    check(|pages| pages[0].logical_bytes = pages[0].charge_bytes + 1);
    check(|pages| pages[0].charge_bytes = 4_097);
    check(|pages| {
        let duplicate = pages[1].class.clone();
        pages[2].class = duplicate;
    });
}

#[test]
fn demands_require_dense_sequences_known_pages_and_monotone_request_steps() {
    let (header, pages, events) = valid_parts();

    let mut gap = events.clone();
    gap[2] = TraceEvent::Demand {
        sequence: 9,
        request: 7,
        step: 0,
        page: PageId(0),
    };
    assert_invalid(serialize_trace(&header, &pages, &gap, SimLimits::default()));

    let mut unknown = events.clone();
    unknown[4] = TraceEvent::Demand {
        sequence: 4,
        request: 7,
        step: 1,
        page: PageId(99),
    };
    assert_invalid(serialize_trace(
        &header,
        &pages,
        &unknown,
        SimLimits::default(),
    ));

    let regressing = vec![
        TraceEvent::Demand {
            sequence: 0,
            request: 1,
            step: 1,
            page: PageId(0),
        },
        TraceEvent::Demand {
            sequence: 1,
            request: 1,
            step: 0,
            page: PageId(0),
        },
    ];
    let mut regressing_header = header;
    regressing_header.event_count = regressing.len();
    assert_invalid(serialize_trace(
        &regressing_header,
        &pages,
        &regressing,
        SimLimits::default(),
    ));
}

#[test]
fn router_signals_are_causal_bounded_unique_and_reference_catalog_experts() {
    let (header, pages, events) = valid_parts();
    let signal = |predictions: Vec<ExpertPrediction>| {
        vec![
            TraceEvent::RouterSignal {
                sequence: 0,
                request: 1,
                target_step: 0,
                layer: 0,
                predictions,
            },
            TraceEvent::Demand {
                sequence: 1,
                request: 1,
                step: 0,
                page: PageId(1),
            },
        ]
    };
    let check = |invalid_events: Vec<TraceEvent>| {
        let mut invalid_header = header.clone();
        invalid_header.event_count = invalid_events.len();
        assert_invalid(serialize_trace(
            &invalid_header,
            &pages,
            &invalid_events,
            SimLimits::default(),
        ));
    };

    check(signal(vec![ExpertPrediction {
        expert: 0,
        score_ppm: 1_000_001,
    }]));
    check(signal(vec![
        ExpertPrediction {
            expert: 0,
            score_ppm: 600_000,
        },
        ExpertPrediction {
            expert: 1,
            score_ppm: 500_000,
        },
    ]));
    check(signal(vec![
        ExpertPrediction {
            expert: 0,
            score_ppm: 400_000,
        },
        ExpertPrediction {
            expert: 0,
            score_ppm: 300_000,
        },
    ]));
    check(signal(vec![ExpertPrediction {
        expert: 99,
        score_ppm: 500_000,
    }]));

    check(vec![
        TraceEvent::Demand {
            sequence: 0,
            request: 1,
            step: 0,
            page: PageId(1),
        },
        TraceEvent::RouterSignal {
            sequence: 1,
            request: 1,
            target_step: 0,
            layer: 0,
            predictions: vec![],
        },
        TraceEvent::Demand {
            sequence: 2,
            request: 1,
            step: 0,
            page: PageId(1),
        },
    ]);

    check(vec![
        TraceEvent::RouterSignal {
            sequence: 0,
            request: 1,
            target_step: 0,
            layer: 0,
            predictions: vec![],
        },
        TraceEvent::RouterSignal {
            sequence: 1,
            request: 1,
            target_step: 0,
            layer: 0,
            predictions: vec![],
        },
        TraceEvent::Demand {
            sequence: 2,
            request: 1,
            step: 0,
            page: PageId(1),
        },
    ]);

    check(vec![TraceEvent::RouterSignal {
        sequence: 0,
        request: 1,
        target_step: 0,
        layer: 0,
        predictions: vec![],
    }]);

    let limits = SimLimits {
        max_predictions_per_signal: 0,
        ..SimLimits::default()
    };
    assert_invalid(serialize_trace(&header, &pages, &events, limits));
}

#[test]
fn declared_record_counts_cannot_hide_missing_or_trailing_data() {
    let (header, pages, events) = valid_parts();
    let bytes = raw_trace(&header, &pages, &events);

    let mut lines: Vec<&[u8]> = bytes.split_inclusive(|byte| *byte == b'\n').collect();
    lines.pop();
    let missing: Vec<u8> = lines.into_iter().flatten().copied().collect();
    assert!(parse_trace(&missing, SimLimits::default()).is_err());

    let extra_event = TraceEvent::Demand {
        sequence: events.len() as u64,
        request: 7,
        step: 2,
        page: PageId(0),
    };
    let mut trailing = bytes;
    trailing.extend_from_slice(&serde_json::to_vec(&extra_event).unwrap());
    trailing.push(b'\n');
    assert!(parse_trace(&trailing, SimLimits::default()).is_err());
}
