use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    BatchRequestSpec, CancelDisposition, ErrorCategory, RequestId, RequestPhase, RequestSpec,
    SchedulerConfig, SchedulerEngine, SchedulerError, SchedulerLimits, SchedulingPolicy,
    ServicePhase, ServiceTraceCursor, StepReport, TerminalOutcome,
};

const REQUEST_COUNT: usize = 16;
const FAIRNESS_HORIZON: usize = 16;
const CONTINUOUS_TURNS: usize = 1_000;
const ANCHOR_LAST_TURN: usize = 352;
const ANCHOR_POSITIONS: usize = 23;
const P1: [u32; 1] = [1];
const P16: [u32; 16] = [1, 14, 16, 6, 14, 16, 6, 14, 16, 6, 14, 16, 6, 14, 16, 6];

type TraceTuple = (u64, usize, ServicePhase);

#[derive(Debug, PartialEq, Eq)]
struct FairnessReplay {
    service_counts: [usize; REQUEST_COUNT],
    max_lag_numerator: usize,
    max_runnable_gap: usize,
    jain_numerator: usize,
    jain_denominator: usize,
}

fn scalar_engine(limits: SchedulerLimits, policy: SchedulingPolicy) -> SchedulerEngine<TinyModel> {
    let model = {
        let fixture = FixtureArtifact::build_v3();
        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
            .expect("authenticated tiny-v3 artifact");
        TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
            .expect("scalar tiny-v3 model")
    };
    let config = SchedulerConfig::new(&model, limits)
        .expect("scheduler configuration")
        .with_scheduling_policy(policy);
    SchedulerEngine::new(model, config).expect("scheduler engine")
}

fn batch_spec(prompt: &[u32], max_new_tokens: usize) -> BatchRequestSpec<'_> {
    BatchRequestSpec::absolute(RequestSpec::new(
        prompt,
        max_new_tokens,
        SamplingPolicy::Greedy,
        None,
    ))
}

fn accepted_ids<const N: usize>(admission: &runnel_scheduler::BatchAdmission) -> [RequestId; N] {
    assert_eq!(admission.accepted_count(), N);
    assert_eq!(admission.rejected_count(), 0);
    let mut accepted = admission.accepted();
    let ids = std::array::from_fn(|offered_index| {
        let item = accepted.next().expect("accepted FIFO prefix");
        assert_eq!(item.offered_index(), offered_index);
        assert_eq!(item.admitted_ns(), admission.release_ns());
        item.request_id()
    });
    assert!(accepted.next().is_none());
    ids
}

fn trace_tuple(event: runnel_scheduler::ServiceTraceEvent) -> TraceTuple {
    (event.request_id().get(), event.position(), event.phase())
}

fn replay_fairness(trace: &[TraceTuple; FAIRNESS_HORIZON]) -> FairnessReplay {
    let mut service_counts = [0_usize; REQUEST_COUNT];
    let mut max_lag_numerator = 0_usize;
    for (index, &(request_id, _, _)) in trace.iter().enumerate() {
        let request_index = usize::try_from(request_id - 1).expect("request index fits usize");
        assert!(request_index < REQUEST_COUNT);
        service_counts[request_index] += 1;
        let prefix = index + 1;
        for count in service_counts {
            max_lag_numerator = max_lag_numerator
                .max(prefix.saturating_sub(REQUEST_COUNT.checked_mul(count).expect("lag product")));
        }
    }

    let mut max_runnable_gap = 0_usize;
    for request_id in 1..=REQUEST_COUNT as u64 {
        let mut previous = None;
        let mut request_gap = 0_usize;
        for (index, &(observed_id, _, _)) in trace.iter().enumerate() {
            if observed_id != request_id {
                continue;
            }
            request_gap = request_gap.max(previous.map_or(index, |prior| index - prior - 1));
            previous = Some(index);
        }
        request_gap =
            request_gap.max(previous.map_or(FAIRNESS_HORIZON, |last| FAIRNESS_HORIZON - last - 1));
        max_runnable_gap = max_runnable_gap.max(request_gap);
    }

    let service_sum = service_counts.iter().sum::<usize>();
    let square_sum = service_counts
        .iter()
        .map(|count| count.checked_mul(*count).expect("count square"))
        .sum::<usize>();
    FairnessReplay {
        service_counts,
        max_lag_numerator,
        max_runnable_gap,
        jain_numerator: service_sum
            .checked_mul(service_sum)
            .expect("Jain numerator"),
        jain_denominator: REQUEST_COUNT
            .checked_mul(square_sum)
            .expect("Jain denominator"),
    }
}

fn first_sixteen_service(policy: SchedulingPolicy) -> [TraceTuple; FAIRNESS_HORIZON] {
    let mut engine = scalar_engine(SchedulerLimits::evidence(), policy);
    let offers = [batch_spec(&P16, 8); REQUEST_COUNT];
    let admission = engine
        .prepare_submit_batch(&offers)
        .expect("prepare fairness batch")
        .commit_prepared_batch(0)
        .expect("commit fairness batch");
    let ids = accepted_ids::<REQUEST_COUNT>(&admission);

    loop {
        let retained = engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect("fairness trace status")
            .status()
            .retained_events();
        if retained >= FAIRNESS_HORIZON {
            break;
        }
        let report = engine.step().expect("fairness step");
        assert!(report.committed_positions > 0);
        assert_eq!(report.selected_positions, report.committed_positions);
        assert_eq!(report.terminal_decisions, 0);
        assert_eq!(engine.snapshot().output_blocked_requests, 0);
    }

    let read = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("fairness trace prefix");
    assert!(read.status().healthy());
    assert!(read.events().len() >= FAIRNESS_HORIZON);
    let mut prefix = [(0_u64, 0_usize, ServicePhase::Prefill); FAIRNESS_HORIZON];
    for (destination, event) in prefix.iter_mut().zip(read.events()) {
        *destination = trace_tuple(*event);
    }
    drop(read);
    for id in ids {
        assert_ne!(
            engine
                .request_phase(id)
                .expect("request retained at horizon"),
            RequestPhase::Terminal
        );
        assert_eq!(
            engine.cancel(id).expect("cancel horizon request"),
            CancelDisposition::Requested
        );
    }
    let cleanup = engine.step().expect("resolve horizon cleanup");
    assert_eq!(cleanup.committed_positions, 0);
    assert_eq!(cleanup.selected_positions, 0);
    assert_eq!(cleanup.waves, 0);
    assert_eq!(cleanup.terminal_decisions, REQUEST_COUNT);
    for id in ids {
        let terminal = engine
            .take_terminal(id)
            .expect("take horizon terminal")
            .expect("horizon terminal retained");
        assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
        if terminal.emitted_tokens() != 0 {
            let drained = engine
                .drain_events(id, usize::MAX)
                .expect("drain horizon output");
            assert_eq!(drained.len(), terminal.emitted_tokens());
            assert!(drained.iter().all(|event| event.request_id() == id));
        }
        assert_eq!(
            engine
                .request_phase(id)
                .expect_err("horizon request reaped")
                .category(),
            ErrorCategory::InvalidRequest
        );
    }
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    let shutdown = engine.shutdown().expect("fairness shutdown");
    assert_eq!(shutdown.terminated_requests, 0);
    assert_eq!(shutdown.discarded_output_events, 0);
    assert_eq!(shutdown.released_request_bytes, 0);
    assert_eq!(shutdown.remaining_shared_bytes, 0);
    prefix
}

#[test]
fn direct_first_sixteen_trace_replays_frozen_policy_fairness() {
    let candidate = first_sixteen_service(SchedulingPolicy::DeficitContinuousExpertCoalesce);
    let baseline = first_sixteen_service(SchedulingPolicy::FifoRunToCompletion);

    let expected_candidate = std::array::from_fn(|index| {
        (
            u64::try_from(index + 1).expect("candidate request ID"),
            0,
            ServicePhase::Prefill,
        )
    });
    let expected_baseline = std::array::from_fn(|position| (1, position, ServicePhase::Prefill));
    assert_eq!(candidate, expected_candidate);
    assert_eq!(baseline, expected_baseline);

    assert_eq!(
        replay_fairness(&candidate),
        FairnessReplay {
            service_counts: [1; REQUEST_COUNT],
            max_lag_numerator: 15,
            max_runnable_gap: 15,
            jain_numerator: 256,
            jain_denominator: 256,
        }
    );
    let mut baseline_counts = [0_usize; REQUEST_COUNT];
    baseline_counts[0] = FAIRNESS_HORIZON;
    assert_eq!(
        replay_fairness(&baseline),
        FairnessReplay {
            service_counts: baseline_counts,
            max_lag_numerator: 16,
            max_runnable_gap: 16,
            jain_numerator: 256,
            jain_denominator: 4_096,
        }
    );
}

fn expected_continuous_event(turn: usize) -> TraceTuple {
    if turn <= ANCHOR_LAST_TURN && turn.is_multiple_of(REQUEST_COUNT) {
        let position = turn / REQUEST_COUNT;
        let phase = if position < P16.len() {
            ServicePhase::Prefill
        } else {
            ServicePhase::Decode
        };
        (1, position, phase)
    } else {
        let anchors_through_turn = if turn <= ANCHOR_LAST_TURN {
            turn / REQUEST_COUNT + 1
        } else {
            ANCHOR_POSITIONS
        };
        let completed_churners = turn + 1 - anchors_through_turn;
        (
            u64::try_from(completed_churners + 1).expect("churner request ID"),
            0,
            ServicePhase::Prefill,
        )
    }
}

fn continuous_limits() -> SchedulerLimits {
    let mut limits = SchedulerLimits::evidence();
    limits.worker_count = 1;
    limits.batch_width = 1;
    limits.waves_per_step = 1;
    limits.max_active_requests = 16;
    limits.max_outstanding_requests = 64;
    limits.max_queued_requests = 64;
    limits.max_retained_terminal_results = 64;
    limits.output_capacity_per_request = 2;
    limits.trace_capacity = 1_024;
    limits
}

#[test]
fn continuous_arrival_1000_has_no_starvation_or_cleanup_service() {
    let mut engine = scalar_engine(
        continuous_limits(),
        SchedulingPolicy::DeficitContinuousExpertCoalesce,
    );
    let pristine_ledger = engine.ledger_snapshot();
    let mut initial_offers = [batch_spec(&P1, 1); REQUEST_COUNT];
    initial_offers[0] = batch_spec(&P16, 8);
    let admission = engine
        .prepare_submit_batch(&initial_offers)
        .expect("prepare continuous initial batch")
        .commit_prepared_batch(0)
        .expect("commit continuous initial batch");
    let initial_ids = accepted_ids::<REQUEST_COUNT>(&admission);
    assert_eq!(initial_ids[0].get(), 1);
    assert_eq!(initial_ids[REQUEST_COUNT - 1].get(), 16);

    let mut live = Vec::with_capacity(64);
    live.extend(initial_ids);
    let live_capacity = live.capacity();
    assert!(live_capacity >= 64);
    let mut trace_cursor = ServiceTraceCursor::origin();
    let mut observed_trace = [(0_u64, 0_usize, ServicePhase::Prefill); CONTINUOUS_TURNS];
    let mut anchor_terminal_turn = None;
    let mut completed_churners = 0_usize;

    for (turn, observed_slot) in observed_trace.iter_mut().enumerate() {
        let admitted = engine
            .try_submit(RequestSpec::new(&P1, 1, SamplingPolicy::Greedy, None))
            .expect("continuous churner accepted");
        assert_eq!(
            admitted.get(),
            u64::try_from(17 + turn).expect("offered request ID")
        );
        live.push(admitted);
        assert_eq!(live.capacity(), live_capacity);

        let expected_promotions = if turn == 0 {
            REQUEST_COUNT
        } else {
            let previous = expected_continuous_event(turn - 1);
            usize::from(previous.0 != 1 || previous.1 + 1 == ANCHOR_POSITIONS)
        };
        let expected = expected_continuous_event(turn);
        let report = engine.step().expect("one continuous service turn");
        assert_eq!(report.promoted_requests, expected_promotions);
        assert_eq!(report.waves, 1);
        assert_eq!(report.selected_positions, 1);
        assert_eq!(report.committed_positions, 1);
        assert_eq!(
            report.terminal_decisions,
            usize::from(expected.0 != 1 || expected.1 + 1 == ANCHOR_POSITIONS)
        );
        assert_eq!(engine.snapshot().output_blocked_requests, 0);

        let read = engine
            .service_trace_since(trace_cursor)
            .expect("incremental continuous trace");
        assert_eq!(read.events().len(), 1);
        assert_eq!(read.status().retained_events(), turn + 1);
        assert_eq!(read.status().event_limit(), 1_024);
        assert!(read.status().healthy());
        let observed = trace_tuple(read.events()[0]);
        assert_eq!(observed, expected);
        *observed_slot = observed;
        trace_cursor = read.next_cursor();
        drop(read);

        let expected_output = if expected.0 == 1 {
            if expected.1 >= P16.len() - 1 {
                Some((1_u64, expected.1 + 1 - P16.len()))
            } else {
                None
            }
        } else {
            Some((expected.0, 0))
        };
        let mut observed_output = None;
        let mut terminal_count = 0_usize;
        let mut index = 0_usize;
        while index < live.len() {
            let id = live[index];
            let terminal =
                engine.request_phase(id).expect("live request phase") == RequestPhase::Terminal;
            let result = if terminal {
                terminal_count += 1;
                Some(
                    engine
                        .take_terminal(id)
                        .expect("take continuous terminal")
                        .expect("continuous terminal retained"),
                )
            } else {
                None
            };
            let should_drain = result
                .as_ref()
                .is_none_or(|terminal| terminal.emitted_tokens() != 0);
            if should_drain {
                for output in engine
                    .drain_events(id, usize::MAX)
                    .expect("drain continuous output")
                {
                    assert_eq!(output.request_id(), id);
                    assert!(
                        observed_output
                            .replace((id.get(), output.output_index()))
                            .is_none(),
                        "more than one output was pending in a service turn"
                    );
                }
            }
            if let Some(terminal) = result {
                assert_eq!(terminal.request_id(), id);
                assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
                if id == initial_ids[0] {
                    assert_eq!(terminal.committed_positions(), ANCHOR_POSITIONS);
                    assert_eq!(terminal.emitted_tokens(), 8);
                    assert_eq!(anchor_terminal_turn.replace(turn), None);
                } else {
                    assert_eq!(terminal.committed_positions(), 1);
                    assert_eq!(terminal.emitted_tokens(), 1);
                    completed_churners += 1;
                }
                assert_eq!(
                    engine
                        .request_phase(id)
                        .expect_err("completed request reaped")
                        .category(),
                    ErrorCategory::InvalidRequest
                );
                live.remove(index);
            } else {
                index += 1;
            }
        }
        assert_eq!(terminal_count, report.terminal_decisions);
        assert_eq!(observed_output, expected_output);
        assert_eq!(live.capacity(), live_capacity);
        assert!(live.len() <= 64);
    }

    assert_eq!(anchor_terminal_turn, Some(ANCHOR_LAST_TURN));
    assert_eq!(completed_churners, 977);
    assert_eq!(live.len(), 38);
    for (offset, id) in live.iter().copied().enumerate() {
        assert_eq!(
            id.get(),
            u64::try_from(979 + offset).expect("final live ID")
        );
    }

    let complete = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("complete continuous trace");
    assert_eq!(complete.events().len(), CONTINUOUS_TURNS);
    assert_eq!(complete.status().retained_events(), CONTINUOUS_TURNS);
    assert_eq!(complete.status().event_limit(), 1_024);
    assert!(complete.status().healthy());
    for (turn, event) in complete.events().iter().copied().enumerate() {
        assert_eq!(trace_tuple(event), expected_continuous_event(turn));
        assert_eq!(trace_tuple(event), observed_trace[turn]);
    }
    drop(complete);

    for id in live.iter().copied() {
        assert_eq!(
            engine.cancel(id).expect("cancel final live request"),
            CancelDisposition::Requested
        );
    }
    let cleanup = engine.step().expect("bounded final cleanup");
    assert_eq!(
        cleanup,
        StepReport {
            terminal_decisions: live.len(),
            ..StepReport::default()
        }
    );
    let post_cleanup_trace = engine
        .service_trace_since(trace_cursor)
        .expect("cleanup trace frontier");
    assert!(post_cleanup_trace.events().is_empty());
    assert_eq!(
        post_cleanup_trace.status().retained_events(),
        CONTINUOUS_TURNS
    );
    assert!(post_cleanup_trace.status().healthy());
    drop(post_cleanup_trace);

    for id in live.iter().copied() {
        let terminal = engine
            .take_terminal(id)
            .expect("take cancelled cleanup terminal")
            .expect("cleanup terminal retained");
        assert_eq!(terminal.request_id(), id);
        assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
        assert_eq!(terminal.committed_positions(), 0);
        assert_eq!(terminal.emitted_tokens(), 0);
        assert_eq!(
            engine
                .request_phase(id)
                .expect_err("cancelled cleanup request reaped")
                .category(),
            ErrorCategory::InvalidRequest
        );
    }
    let cleaned = engine.ledger_snapshot();
    assert_eq!(cleaned.request_used(), 0);
    assert_eq!(cleaned.shared_used(), pristine_ledger.shared_used());
    assert_eq!(cleaned.total_used(), pristine_ledger.total_used());

    let shutdown = engine.shutdown().expect("continuous shutdown");
    assert_eq!(shutdown.terminated_requests, 0);
    assert_eq!(shutdown.discarded_output_events, 0);
    assert_eq!(shutdown.released_request_bytes, 0);
    assert_eq!(shutdown.remaining_shared_bytes, 0);
    assert!(engine.ledger_snapshot().current_is_zero());
    assert!(matches!(
        engine
            .service_trace_since(ServiceTraceCursor::origin())
            .expect_err("successful shutdown releases trace storage"),
        SchedulerError::SchedulerClosed
    ));
}
