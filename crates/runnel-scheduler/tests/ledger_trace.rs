use std::collections::BTreeMap;

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    BatchRequestSpec, CancelDisposition, ErrorCategory, LedgerCategory, LedgerMutationKind,
    LedgerOwnership, LedgerSnapshot, LedgerTraceCursor, LedgerTraceRead, RequestId, RequestPhase,
    RequestSpec, SchedulerConfig, SchedulerEngine, SchedulerError, SchedulerLimits,
    ServiceTraceCursor, TerminalOutcome,
};

const CATEGORY_COUNT: usize = LedgerCategory::ALL.len();

fn scalar_engine(mut limits: SchedulerLimits) -> SchedulerEngine<TinyModel> {
    // Keep every test step at one position so service and ledger boundaries
    // can be compared without relying on a partially consumed StepReport.
    limits.worker_count = 1;
    limits.max_active_requests = 1;
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    let model = {
        let fixture = FixtureArtifact::build_v3();
        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
            .expect("authenticated tiny-v3 artifact");
        TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
            .expect("scalar tiny-v3 model")
    };
    let config = SchedulerConfig::new(&model, limits).expect("scheduler configuration");
    SchedulerEngine::new(model, config).expect("scheduler engine")
}

fn sum_by_ownership(categories: &[u64; CATEGORY_COUNT], ownership: LedgerOwnership) -> u64 {
    LedgerCategory::ALL
        .into_iter()
        .filter(|category| category.ownership() == ownership)
        .map(|category| categories[category.index()])
        .try_fold(0_u64, u64::checked_add)
        .expect("replay ownership total")
}

/// Independently replays the sequence-zero snapshot and complete mutation
/// groups, including historical peaks, then reconciles every public counter.
fn assert_replays_to(read: &LedgerTraceRead<'_>, observed: LedgerSnapshot) {
    let initial = read.initial_snapshot().ledger_snapshot();
    assert_eq!(read.initial_snapshot().sequence(), 0);
    assert_eq!(initial.request_used(), 0);

    let mut used: [u64; CATEGORY_COUNT] =
        std::array::from_fn(|index| initial.category(LedgerCategory::ALL[index]).used());
    let mut peak: [u64; CATEGORY_COUNT] =
        std::array::from_fn(|index| initial.category(LedgerCategory::ALL[index]).peak());
    let mut request_used = initial.request_used();
    let mut request_peak = initial.request_peak();
    let mut shared_used = initial.shared_used();
    let mut shared_peak = initial.shared_peak();
    let mut total_used = initial.total_used();
    let mut total_peak = initial.total_peak();
    let mut request_owner_used = BTreeMap::<u64, [u64; CATEGORY_COUNT]>::new();

    let events = read.events();
    let mut start = 0_usize;
    let mut expected_sequence = 1_u64;
    while start < events.len() {
        let sequence = events[start].sequence();
        assert_eq!(sequence, expected_sequence);
        let kind = events[start].kind();
        let mut end = start;
        let mut previous_key = None;
        while end < events.len() && events[end].sequence() == sequence {
            let event = events[end];
            assert_eq!(event.kind(), kind);
            assert_ne!(event.bytes(), 0);
            assert_eq!(event.bytes() % 64, 0);
            assert_eq!(event.owner().ownership(), event.category().ownership());
            assert_eq!(
                event.owner().request_id().map(RequestId::get),
                Some(event.owner().evidence_id())
            );
            let owner = request_owner_used
                .entry(event.owner().evidence_id())
                .or_insert([0_u64; CATEGORY_COUNT]);
            let key = (event.owner().evidence_id(), event.category().evidence_id());
            assert!(previous_key.is_none_or(|previous| key > previous));
            previous_key = Some(key);

            let category = event.category().index();
            match event.kind() {
                LedgerMutationKind::Acquire => {
                    assert_eq!(
                        event.signed_delta_bytes(),
                        i64::try_from(event.bytes()).unwrap()
                    );
                    used[category] = used[category]
                        .checked_add(event.bytes())
                        .expect("replay category acquire");
                    owner[category] = owner[category]
                        .checked_add(event.bytes())
                        .expect("replay owner acquire");
                }
                LedgerMutationKind::Release => {
                    assert_eq!(
                        event.signed_delta_bytes(),
                        -i64::try_from(event.bytes()).unwrap()
                    );
                    used[category] = used[category]
                        .checked_sub(event.bytes())
                        .expect("replay category release");
                    owner[category] = owner[category]
                        .checked_sub(event.bytes())
                        .expect("release must not underflow its attributed owner");
                }
            }
            end += 1;
        }

        for category in LedgerCategory::ALL {
            let index = category.index();
            peak[index] = peak[index].max(used[index]);
        }
        request_used = sum_by_ownership(&used, LedgerOwnership::Request);
        request_peak = request_peak.max(request_used);
        shared_used = sum_by_ownership(&used, LedgerOwnership::Shared);
        shared_peak = shared_peak.max(shared_used);
        total_used = request_used
            .checked_add(shared_used)
            .expect("replay aggregate total");
        total_peak = total_peak.max(total_used);
        expected_sequence += 1;
        start = end;
    }

    assert_eq!(read.status().retained_mutations(), expected_sequence - 1);
    for category in LedgerCategory::ALL
        .into_iter()
        .filter(|category| category.ownership() == LedgerOwnership::Request)
    {
        let owner_total = request_owner_used
            .values()
            .map(|categories| categories[category.index()])
            .try_fold(0_u64, u64::checked_add)
            .expect("owner-attributed category total");
        assert_eq!(owner_total, used[category.index()]);
    }
    if observed.request_used() == 0 {
        assert!(
            request_owner_used
                .values()
                .all(|categories| categories.iter().all(|bytes| *bytes == 0))
        );
    }
    for category in LedgerCategory::ALL {
        let replayed = category.index();
        assert_eq!(observed.category(category).used(), used[replayed]);
        assert_eq!(observed.category(category).peak(), peak[replayed]);
        assert_eq!(
            observed.category(category).limit(),
            initial.category(category).limit()
        );
    }
    assert_eq!(observed.request_used(), request_used);
    assert_eq!(observed.request_peak(), request_peak);
    assert_eq!(observed.shared_used(), shared_used);
    assert_eq!(observed.shared_peak(), shared_peak);
    assert_eq!(observed.total_used(), total_used);
    assert_eq!(observed.total_peak(), total_peak);
    assert_eq!(observed.total_limit(), initial.total_limit());
}

fn event_shape(read: &LedgerTraceRead<'_>) -> Vec<(u64, LedgerMutationKind, u64, LedgerCategory)> {
    read.events()
        .iter()
        .map(|event| {
            (
                event.sequence(),
                event.kind(),
                event.owner().evidence_id(),
                event.category(),
            )
        })
        .collect()
}

#[test]
fn replay_covers_admit_promote_terminal_reap_without_per_token_deltas() {
    let mut limits = SchedulerLimits::tiny();
    limits.trace_capacity = 64;
    let mut engine = scalar_engine(limits);
    let initial = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("initial ledger trace");
    assert!(initial.events().is_empty());
    assert_eq!(initial.status().event_limit(), 64);
    assert!(initial.status().healthy());
    assert_replays_to(&initial, engine.ledger_snapshot());
    drop(initial);

    let request_id = engine
        .try_submit(RequestSpec::new(&[0, 1], 3, SamplingPolicy::Greedy, None))
        .expect("request admission");
    let admitted = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("admission trace");
    assert_eq!(
        event_shape(&admitted),
        [
            LedgerCategory::PromptStorage,
            LedgerCategory::RequestRecord,
            LedgerCategory::RequestSlot,
            LedgerCategory::Output,
            LedgerCategory::Terminal,
        ]
        .map(|category| (1, LedgerMutationKind::Acquire, request_id.get(), category))
    );
    assert_replays_to(&admitted, engine.ledger_snapshot());

    let request_record = admitted
        .events()
        .iter()
        .copied()
        .find(|event| event.category() == LedgerCategory::RequestRecord)
        .expect("request-record event");
    let event_debug = format!("{request_record:?}");
    assert!(event_debug.contains("<redacted>"));
    assert!(!event_debug.contains("512"));
    assert_eq!(
        format!("{:?}", request_record.owner()),
        "Request(\"<redacted>\")"
    );
    assert!(format!("{:?}", admitted.initial_snapshot()).contains("<redacted>"));
    assert!(format!("{admitted:?}").contains("events: \"<redacted>\""));
    drop(admitted);

    let first = engine.step().expect("first service position");
    assert_eq!(first.committed_positions, 1);
    let after_first = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("post-promotion trace");
    assert_eq!(
        &event_shape(&after_first)[5..],
        &[
            (
                2,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::ActiveState,
            ),
            (
                2,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::PendingTransaction,
            ),
        ]
    );
    assert_replays_to(&after_first, engine.ledger_snapshot());
    let ledger_cursor = after_first.next_cursor();
    let ledger_after_first = engine.ledger_snapshot();
    drop(after_first);
    let service_cursor = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("first service event")
        .next_cursor();

    let second = engine.step().expect("second service position");
    assert_eq!(second.committed_positions, 1);
    assert_eq!(engine.ledger_snapshot(), ledger_after_first);
    let no_token_delta = engine
        .ledger_trace_since(ledger_cursor)
        .expect("per-token ledger frontier");
    assert!(no_token_delta.events().is_empty());
    assert!(no_token_delta.status().healthy());
    drop(no_token_delta);
    let second_service = engine
        .service_trace_since(service_cursor)
        .expect("second service suffix");
    assert_eq!(second_service.events().len(), 1);
    assert_eq!(second_service.events()[0].request_id(), request_id);
    drop(second_service);

    for _ in 0..4 {
        if engine.request_phase(request_id).expect("retained request") == RequestPhase::Terminal {
            break;
        }
        engine.step().expect("finish request");
    }
    assert_eq!(
        engine.request_phase(request_id).expect("terminal request"),
        RequestPhase::Terminal
    );
    let terminal = engine
        .take_terminal(request_id)
        .expect("terminal read")
        .expect("terminal retained");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(
        engine
            .drain_events(request_id, usize::MAX)
            .expect("drain retained output")
            .len(),
        terminal.emitted_tokens()
    );

    let final_read = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("complete lifecycle trace");
    assert_eq!(
        event_shape(&final_read),
        vec![
            (
                1,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::PromptStorage
            ),
            (
                1,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::RequestRecord
            ),
            (
                1,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::RequestSlot
            ),
            (
                1,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::Output
            ),
            (
                1,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::Terminal
            ),
            (
                2,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::ActiveState
            ),
            (
                2,
                LedgerMutationKind::Acquire,
                request_id.get(),
                LedgerCategory::PendingTransaction
            ),
            (
                3,
                LedgerMutationKind::Release,
                request_id.get(),
                LedgerCategory::PromptStorage
            ),
            (
                3,
                LedgerMutationKind::Release,
                request_id.get(),
                LedgerCategory::ActiveState
            ),
            (
                3,
                LedgerMutationKind::Release,
                request_id.get(),
                LedgerCategory::PendingTransaction
            ),
            (
                4,
                LedgerMutationKind::Release,
                request_id.get(),
                LedgerCategory::RequestRecord
            ),
            (
                4,
                LedgerMutationKind::Release,
                request_id.get(),
                LedgerCategory::RequestSlot
            ),
            (
                4,
                LedgerMutationKind::Release,
                request_id.get(),
                LedgerCategory::Output
            ),
            (
                4,
                LedgerMutationKind::Release,
                request_id.get(),
                LedgerCategory::Terminal
            ),
        ]
    );
    assert!(final_read.status().healthy());
    assert_eq!(final_read.status().retained_mutations(), 4);
    assert_replays_to(&final_read, engine.ledger_snapshot());
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    let future_cursor = final_read.next_cursor();
    assert!(
        engine
            .ledger_trace_since(future_cursor)
            .expect("frontier read")
            .events()
            .is_empty()
    );
    drop(final_read);

    let mut fresh = scalar_engine(limits);
    assert_eq!(
        fresh
            .ledger_trace_since(future_cursor)
            .expect_err("foreign future ordinal must fail")
            .category(),
        ErrorCategory::InvalidRequest
    );
    assert_eq!(
        fresh
            .shutdown()
            .expect("foreign-cursor engine shutdown")
            .remaining_shared_bytes,
        0
    );

    assert_eq!(
        engine.shutdown().expect("shutdown").remaining_shared_bytes,
        0
    );
    assert!(matches!(
        engine
            .ledger_trace_since(LedgerTraceCursor::origin())
            .expect_err("successful shutdown destroys ledger trace"),
        SchedulerError::SchedulerClosed
    ));
    assert!(engine.ledger_snapshot().current_is_zero());
}

#[test]
fn atomic_batch_is_one_canonical_sequence_and_public_rollbacks_are_absent() {
    let mut limits = SchedulerLimits::tiny();
    limits.max_active_requests = 2;
    limits.trace_capacity = 64;
    let mut cursor_source = scalar_engine(limits);
    cursor_source
        .try_submit(RequestSpec::new(&[0], 1, SamplingPolicy::Greedy, None))
        .expect("foreign-cursor source admission");
    let foreign_mid_sequence = cursor_source
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("foreign-cursor source trace")
        .next_cursor();
    assert_eq!(foreign_mid_sequence.event_index(), 5);
    cursor_source
        .shutdown()
        .expect("foreign-cursor source shutdown");

    let mut engine = scalar_engine(limits);
    let before = engine.ledger_snapshot();

    let first_prompt = [0_u32];
    let second_prompt = [1_u32, 2];
    let offers = [
        BatchRequestSpec::absolute(RequestSpec::new(
            &first_prompt,
            1,
            SamplingPolicy::Greedy,
            None,
        )),
        BatchRequestSpec::absolute(RequestSpec::new(
            &second_prompt,
            1,
            SamplingPolicy::Greedy,
            None,
        )),
    ];
    drop(
        engine
            .prepare_submit_batch(&offers)
            .expect("prepare dropped batch"),
    );
    assert_eq!(engine.ledger_snapshot(), before);
    let after_drop = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("trace after dropped prepare");
    assert!(after_drop.events().is_empty());
    assert_eq!(after_drop.status().retained_mutations(), 0);
    drop(after_drop);

    let expiring = [BatchRequestSpec::absolute(RequestSpec::new(
        &first_prompt,
        1,
        SamplingPolicy::Greedy,
        Some(10),
    ))];
    let error = engine
        .prepare_submit_batch(&expiring)
        .expect("prepare expiring batch")
        .commit_prepared_batch(10)
        .expect_err("inclusive deadline rolls preparation back");
    assert_eq!(error.category(), ErrorCategory::DeadlineExceeded);
    assert_eq!(engine.ledger_snapshot(), before);
    assert!(
        engine
            .ledger_trace_since(LedgerTraceCursor::origin())
            .expect("trace after release failure")
            .events()
            .is_empty()
    );

    let admission = engine
        .prepare_submit_batch(&offers)
        .expect("prepare atomic batch")
        .commit_prepared_batch(11)
        .expect("commit atomic batch");
    let ids = admission
        .accepted()
        .map(|accepted| accepted.request_id())
        .collect::<Vec<_>>();
    assert_eq!(ids.iter().map(|id| id.get()).collect::<Vec<_>>(), [1, 2]);

    let read = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("atomic batch trace");
    assert_eq!(read.events().len(), 10);
    assert!(read.events().iter().all(|event| event.sequence() == 1));
    assert!(
        read.events()
            .iter()
            .all(|event| event.kind() == LedgerMutationKind::Acquire)
    );
    let expected = ids
        .iter()
        .flat_map(|id| {
            [
                LedgerCategory::PromptStorage,
                LedgerCategory::RequestRecord,
                LedgerCategory::RequestSlot,
                LedgerCategory::Output,
                LedgerCategory::Terminal,
            ]
            .map(|category| (id.get(), category.evidence_id()))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        read.events()
            .iter()
            .map(|event| (event.owner().evidence_id(), event.category().evidence_id()))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        engine
            .ledger_trace_since(foreign_mid_sequence)
            .expect_err("foreign cursor must not bisect an atomic mutation")
            .category(),
        ErrorCategory::InvalidRequest
    );
    assert_replays_to(&read, engine.ledger_snapshot());
    drop(read);

    for id in &ids {
        assert_eq!(
            engine.cancel(*id).expect("cancel batch request"),
            CancelDisposition::Requested
        );
    }
    let cleanup = engine.step().expect("resolve batch cancellation");
    assert_eq!(cleanup.committed_positions, 0);
    assert_eq!(cleanup.terminal_decisions, ids.len());
    for id in ids {
        assert_eq!(
            engine
                .take_terminal(id)
                .expect("take batch terminal")
                .expect("batch terminal retained")
                .outcome(),
            TerminalOutcome::Cancelled
        );
    }
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    let final_read = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("batch cleanup trace");
    assert!(final_read.status().healthy());
    assert_replays_to(&final_read, engine.ledger_snapshot());
    drop(final_read);
    assert_eq!(
        engine
            .shutdown()
            .expect("batch shutdown")
            .remaining_shared_bytes,
        0
    );
}

#[test]
fn whole_ledger_mutation_overflow_is_atomic_and_service_trace_stays_healthy() {
    let mut limits = SchedulerLimits::tiny();
    limits.trace_capacity = 5;
    let mut engine = scalar_engine(limits);
    assert_eq!(
        engine.config().shared_static_charges().trace_bytes(),
        5 * 128
    );
    let request_id = engine
        .try_submit(RequestSpec::new(&[0], 2, SamplingPolicy::Greedy, None))
        .expect("exact-full admission");
    let exact_full = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("exact-full ledger trace");
    assert_eq!(exact_full.events().len(), 5);
    assert_eq!(exact_full.status().retained_mutations(), 1);
    assert!(exact_full.status().healthy());
    let full_cursor = exact_full.next_cursor();
    drop(exact_full);
    assert!(
        engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("empty service trace")
            .status()
            .healthy()
    );

    assert_eq!(
        engine
            .step()
            .expect("overflowing promotion")
            .committed_positions,
        1
    );
    let overflowed = engine
        .ledger_trace_since(full_cursor)
        .expect("overflowed ledger frontier");
    assert!(overflowed.events().is_empty());
    assert_eq!(overflowed.status().retained_events(), 5);
    assert_eq!(overflowed.status().retained_mutations(), 1);
    assert!(overflowed.status().overflowed());
    drop(overflowed);
    let service = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("independent service trace");
    assert_eq!(service.events().len(), 1);
    assert_eq!(service.events()[0].request_id(), request_id);
    assert!(service.status().healthy());
    drop(service);

    assert_eq!(
        engine
            .step()
            .expect("complete after ledger overflow")
            .committed_positions,
        1
    );
    assert!(
        engine
            .ledger_trace_since(LedgerTraceCursor::origin())
            .expect("sticky ledger overflow")
            .status()
            .overflowed()
    );
    let output = engine
        .drain_events(request_id, usize::MAX)
        .expect("drain overflow-case output");
    let terminal = engine
        .take_terminal(request_id)
        .expect("terminal query")
        .expect("terminal retained");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(output.len(), terminal.emitted_tokens());
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    assert_eq!(
        engine
            .shutdown()
            .expect("overflow shutdown")
            .remaining_shared_bytes,
        0
    );
}

#[test]
fn service_overflow_does_not_poison_a_still_complete_ledger_trace() {
    let mut limits = SchedulerLimits::tiny();
    limits.trace_capacity = 8;
    let mut engine = scalar_engine(limits);
    let prompt = [1_u32; 8];
    let request_id = engine
        .try_submit(RequestSpec::new(&prompt, 4, SamplingPolicy::Greedy, None))
        .expect("long request");

    for _ in 0..9 {
        assert_eq!(
            engine
                .step()
                .expect("one long-request position")
                .committed_positions,
            1
        );
    }
    let service = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("overflowed service trace");
    assert_eq!(service.events().len(), 8);
    assert!(service.status().overflowed());
    drop(service);
    let ledger = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("independent ledger trace");
    assert_eq!(ledger.events().len(), 7);
    assert_eq!(ledger.status().retained_mutations(), 2);
    assert!(ledger.status().healthy());
    assert_replays_to(&ledger, engine.ledger_snapshot());
    drop(ledger);

    for _ in 0..4 {
        if engine
            .request_phase(request_id)
            .expect("long request retained")
            == RequestPhase::Terminal
        {
            break;
        }
        engine.step().expect("finish long request");
    }
    assert_eq!(
        engine.request_phase(request_id).expect("long terminal"),
        RequestPhase::Terminal
    );
    let output = engine
        .drain_events(request_id, usize::MAX)
        .expect("drain long output");
    let terminal = engine
        .take_terminal(request_id)
        .expect("long terminal query")
        .expect("long terminal retained");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(output.len(), terminal.emitted_tokens());
    assert_eq!(
        engine
            .shutdown()
            .expect("long shutdown")
            .remaining_shared_bytes,
        0
    );
}
