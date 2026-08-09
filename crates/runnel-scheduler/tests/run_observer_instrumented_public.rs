#![cfg(feature = "m5-run-observer-instrumentation")]

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SampleConfig, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    BatchRequestSpec, CancelDisposition, LedgerTraceCursor, RequestId, RequestPhase, RunObserver,
    SchedulerConfig, SchedulerEngine, SchedulerLimits, SchedulingPolicy, ServicePhase,
    ServiceTraceCursor, TerminalOutcome,
};

#[cfg(feature = "deterministic-checkpoint-instrumentation")]
use runnel_scheduler::{CheckpointAction, CheckpointDirective, CheckpointPoint};

fn evidence_engine() -> SchedulerEngine<TinyModel> {
    evidence_engine_with(SchedulingPolicy::DeficitContinuousExpertCoalesce, 4)
}

fn evidence_engine_with(
    policy: SchedulingPolicy,
    waves_per_step: u64,
) -> SchedulerEngine<TinyModel> {
    let fixture = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
        .expect("authenticated tiny-v3 artifact");
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .expect("scalar tiny-v3 model");
    let mut limits = SchedulerLimits::evidence();
    limits.waves_per_step = waves_per_step;
    let config = SchedulerConfig::new(&model, limits)
        .expect("frozen M5 scheduler configuration")
        .with_scheduling_policy(policy);
    SchedulerEngine::new(model, config).expect("frozen M5 scheduler engine")
}

fn output_blocked_engine() -> SchedulerEngine<TinyModel> {
    let fixture = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
        .expect("authenticated tiny-v3 artifact");
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .expect("scalar tiny-v3 model");
    let mut limits = SchedulerLimits::evidence();
    limits.output_capacity_per_request = 1;
    limits.waves_per_step = 4;
    let config = SchedulerConfig::new(&model, limits).expect("output-blocked configuration");
    SchedulerEngine::new(model, config).expect("output-blocked engine")
}

fn observed_admission(
    engine: &mut SchedulerEngine<TinyModel>,
    observer: &mut RunObserver,
    offers: &[BatchRequestSpec<'_>],
) -> Vec<RequestId> {
    let admission = engine
        .prepare_submit_batch(offers)
        .expect("prepared observed batch")
        .commit_prepared_batch_observed(observer)
        .expect("published observed batch");
    assert_eq!(admission.accepted_count(), offers.len());
    assert_eq!(admission.rejected_count(), 0);
    admission
        .accepted()
        .map(|accepted| accepted.request_id())
        .collect()
}

fn run_observed_to_terminal(
    engine: &mut SchedulerEngine<TinyModel>,
    observer: &mut RunObserver,
    requests: &[RequestId],
) {
    for _ in 0..64 {
        if requests.iter().all(|request| {
            engine.request_phase(*request).expect("observed phase") == RequestPhase::Terminal
        }) {
            return;
        }
        engine
            .step_observed(observer)
            .expect("observed scheduler step");
    }
    panic!("observed requests did not terminate within the deterministic bound");
}

#[derive(Debug, PartialEq, Eq)]
struct SemanticCapture {
    reports: Vec<[usize; 9]>,
    service: Vec<(u64, usize, u8)>,
    ledger: Vec<(u64, u64, u8, u8, u64)>,
    outputs: Vec<(u64, usize, u32)>,
    terminals: Vec<(u64, TerminalOutcome, usize, usize)>,
}

fn capture_semantics(
    policy: SchedulingPolicy,
    waves_per_step: u64,
    observed: bool,
) -> SemanticCapture {
    let prompt_a = [1_u32, 14, 16];
    let prompt_b = [6_u32];
    let sampling = [41_u64, 73_u64].map(|seed| {
        SamplingPolicy::Sample(SampleConfig {
            seed,
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
        })
    });
    let offers = [
        BatchRequestSpec::absolute(runnel_scheduler::RequestSpec::new(
            &prompt_a,
            4,
            sampling[0],
            None,
        )),
        BatchRequestSpec::absolute(runnel_scheduler::RequestSpec::new(
            &prompt_b,
            4,
            sampling[1],
            None,
        )),
    ];
    let mut engine = evidence_engine_with(policy, waves_per_step);
    let mut observer = observed.then(|| engine.prepare_run_observer().expect("parity observer"));
    let admission = if let Some(observer) = observer.as_mut() {
        engine
            .prepare_submit_batch(&offers)
            .expect("prepare observed parity batch")
            .commit_prepared_batch_observed(observer)
            .expect("publish observed parity batch")
    } else {
        engine
            .prepare_submit_batch(&offers)
            .expect("prepare baseline parity batch")
            .commit_prepared_batch(0)
            .expect("publish baseline parity batch")
    };
    let requests: Vec<_> = admission
        .accepted()
        .map(|accepted| accepted.request_id())
        .collect();
    let mut reports = Vec::new();
    for _ in 0..64 {
        if requests.iter().all(|request| {
            engine.request_phase(*request).expect("parity phase") == RequestPhase::Terminal
        }) {
            break;
        }
        let report = if let Some(observer) = observer.as_mut() {
            engine
                .step_observed(observer)
                .expect("observed parity step")
        } else {
            engine.step().expect("baseline parity step")
        };
        reports.push([
            report.promoted_requests,
            report.waves,
            report.selected_positions,
            report.expert_tasks,
            report.expert_groups,
            report.committed_positions,
            report.preempted_requests,
            report.resumed_requests,
            report.terminal_decisions,
        ]);
    }
    assert!(requests.iter().all(|request| {
        engine
            .request_phase(*request)
            .expect("terminal parity phase")
            == RequestPhase::Terminal
    }));

    let service = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("parity service trace");
    let service = service
        .events()
        .iter()
        .map(|event| {
            (
                event.request_id().get(),
                event.position(),
                event.phase().evidence_bit(),
            )
        })
        .collect();
    let mut outputs = Vec::new();
    let mut terminals = Vec::new();
    for request in requests {
        let events = if let Some(observer) = observer.as_mut() {
            engine
                .drain_events_observed(request, usize::MAX, observer)
                .expect("observed parity drain")
        } else {
            engine
                .drain_events(request, usize::MAX)
                .expect("baseline parity drain")
        };
        outputs.extend(events.into_iter().map(|event| {
            (
                event.request_id().get(),
                event.output_index(),
                event.token(),
            )
        }));
        let terminal = if let Some(observer) = observer.as_mut() {
            engine
                .take_terminal_observed(request, observer)
                .expect("observed parity terminal read")
        } else {
            engine
                .take_terminal(request)
                .expect("baseline parity terminal read")
        }
        .expect("retained parity terminal");
        terminals.push((
            terminal.request_id().get(),
            terminal.outcome(),
            terminal.committed_positions(),
            terminal.emitted_tokens(),
        ));
    }
    if let Some(observer) = observer.as_mut() {
        let totals = observer.read().totals();
        let summed = reports.iter().fold([0_usize; 9], |mut sums, report| {
            for (sum, value) in sums.iter_mut().zip(report) {
                *sum += value;
            }
            sums
        });
        assert_eq!(totals.step_count, reports.len() as u64);
        assert_eq!(totals.wave_count, summed[1] as u64);
        assert_eq!(totals.selected_positions, summed[2] as u64);
        assert_eq!(totals.expert_contributions, summed[3] as u64);
        assert_eq!(totals.expert_group_calls, summed[4] as u64);
        assert_eq!(totals.committed_positions, summed[5] as u64);
        assert_eq!(totals.preempted_survivors, summed[6] as u64);
        assert_eq!(totals.resumed_requests, summed[7] as u64);
        assert_eq!(totals.terminal_decisions, summed[8] as u64);
        let status = observer.finish();
        assert!(status.healthy(), "parity observer: {:?}", status.failure());
    }
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    let ledger = engine
        .ledger_trace_since(LedgerTraceCursor::origin())
        .expect("complete parity ledger trace");
    let ledger = ledger
        .events()
        .iter()
        .map(|event| {
            (
                event.sequence(),
                event.owner().evidence_id(),
                event.category().evidence_id(),
                event.kind().evidence_id(),
                event.bytes(),
            )
        })
        .collect();
    SemanticCapture {
        reports,
        service,
        ledger,
        outputs,
        terminals,
    }
}

#[test]
fn observed_path_preserves_semantics_and_captures_closed_lifecycle() {
    let prompt_a = [1_u32, 14];
    let prompt_b = [16_u32];
    let offers = [
        BatchRequestSpec::absolute(runnel_scheduler::RequestSpec::new(
            &prompt_a,
            3,
            SamplingPolicy::Greedy,
            None,
        )),
        BatchRequestSpec::absolute(runnel_scheduler::RequestSpec::new(
            &prompt_b,
            3,
            SamplingPolicy::Greedy,
            None,
        )),
    ];
    let mut engine = evidence_engine();
    let mut observer = engine.prepare_run_observer().expect("bounded run observer");
    assert_eq!(observer.clock_id(), "std-instant-origin-nanoseconds-v1");
    let allocation_fingerprint = observer.allocation_fingerprint();
    let initial = observer.read().status();
    assert_eq!(initial.request_capacity(), 32);
    assert_eq!(initial.output_timestamp_capacity(), 256);
    assert_eq!(
        initial.requested_bytes(),
        32 * std::mem::size_of::<runnel_scheduler::RunRequestObservation>()
            + 256 * std::mem::size_of::<Option<u64>>()
    );
    assert!(!initial.started());
    assert!(initial.healthy());

    let requests = observed_admission(&mut engine, &mut observer, &offers);
    run_observed_to_terminal(&mut engine, &mut observer, &requests);

    let service = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("complete observed service trace");
    assert!(service.status().healthy());
    assert_eq!(service.events().len(), 7);
    let semantic_trace: Vec<_> = service
        .events()
        .iter()
        .map(|event| (event.request_id().get(), event.position(), event.phase()))
        .collect();
    drop(service);
    assert_eq!(
        semantic_trace
            .iter()
            .filter(|(_, _, phase)| *phase == ServicePhase::Prefill)
            .count(),
        3
    );
    assert_eq!(
        semantic_trace
            .iter()
            .filter(|(_, _, phase)| *phase == ServicePhase::Decode)
            .count(),
        4
    );

    let mut tokens = Vec::new();
    for request in &requests {
        tokens.extend(
            engine
                .drain_events_observed(*request, usize::MAX, &mut observer)
                .expect("observed output drain")
                .into_iter()
                .map(|event| {
                    (
                        event.request_id().get(),
                        event.output_index(),
                        event.token(),
                    )
                }),
        );
        let terminal = engine
            .take_terminal_observed(*request, &mut observer)
            .expect("observed terminal read")
            .expect("retained observed terminal");
        assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
        assert_eq!(terminal.emitted_tokens(), 3);
    }
    assert_eq!(tokens.len(), 6);
    assert_eq!(engine.ledger_snapshot().request_used(), 0);

    let read = observer.read();
    assert_eq!(read.release_ns(), Some(read.requests()[0].admitted_ns()));
    assert_eq!(read.requests().len(), 2);
    assert_eq!(read.status().output_timestamp_count(), 6);
    assert_eq!(read.totals().committed_positions, 7);
    assert_eq!(read.totals().terminal_decisions, 2);
    assert_eq!(read.totals().state_sample_count, 7);
    assert_eq!(read.wave_occupancy_histogram()[0], 0);
    assert_eq!(
        read.wave_occupancy_histogram().iter().sum::<u64>(),
        read.totals().wave_count
    );
    assert_eq!(read.totals().state_live_token_sample_sum, 25);
    assert_eq!(read.totals().state_allocated_page_slot_sample_sum, 208);
    for (request_index, request) in read.requests().iter().enumerate() {
        assert_eq!(request.terminal_outcome(), Some(TerminalOutcome::Completed));
        assert_eq!(request.emitted_tokens(), 3);
        assert!(request.request_owned_zero_ns().is_some());
        assert_eq!(request.worker_quiescent_ns(), None);
        let stamps = read
            .output_commit_ns(request_index)
            .expect("local output timestamp row");
        assert_eq!(stamps.len(), 3);
        assert!(stamps.iter().all(Option::is_some));
        assert!(stamps.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(request.first_work_start_ns() >= Some(request.admitted_ns()));
        assert!(request.prefill_complete_ns() >= request.first_work_start_ns());
        assert!(request.first_decode_start_ns() >= request.prefill_complete_ns());
        assert!(request.terminal_decided_ns() >= stamps.last().copied().flatten());
        assert!(request.request_owned_zero_ns() >= request.terminal_decided_ns());
    }
    drop(read);
    let finished = observer.finish();
    assert!(finished.finished());
    assert!(
        finished.healthy(),
        "observer failure: {:?}",
        finished.failure()
    );
    let debug = format!("{observer:?}");
    assert!(debug.contains("domain: \"<redacted>\""));
    assert!(debug.contains("requests: \"<redacted>\""));
    assert!(debug.contains("timestamps: \"<redacted>\""));
    assert_eq!(observer.allocation_fingerprint(), allocation_fingerprint);
}

#[test]
fn observer_on_off_is_semantically_exact_across_policies_and_step_partitions() {
    for policy in [
        SchedulingPolicy::FifoRunToCompletion,
        SchedulingPolicy::DeficitContinuousExpertCoalesce,
    ] {
        for waves_per_step in [1_u64, 4] {
            let baseline = capture_semantics(policy, waves_per_step, false);
            let observed = capture_semantics(policy, waves_per_step, true);
            assert_eq!(observed, baseline, "{policy:?}, waves={waves_per_step}");
        }
    }
}

#[test]
fn observer_is_engine_bound_and_rejects_reuse_before_publication() {
    let prompt = [1_u32];
    let offers = [BatchRequestSpec::absolute(
        runnel_scheduler::RequestSpec::new(&prompt, 1, SamplingPolicy::Greedy, None),
    )];
    let mut first = evidence_engine();
    let mut second = evidence_engine();
    let mut observer = first.prepare_run_observer().expect("first observer");
    let error = second
        .prepare_submit_batch(&offers)
        .expect("second prepared batch")
        .commit_prepared_batch_observed(&mut observer)
        .expect_err("foreign observer must be rejected");
    assert_eq!(
        error.category(),
        runnel_scheduler::ErrorCategory::InvalidRequest
    );
    assert!(!observer.read().status().started());

    let requests = observed_admission(&mut first, &mut observer, &offers);
    let error = first
        .step_observed(&mut second.prepare_run_observer().expect("foreign observer"))
        .expect_err("foreign step observer must be rejected");
    assert_eq!(
        error.category(),
        runnel_scheduler::ErrorCategory::InvalidRequest
    );
    run_observed_to_terminal(&mut first, &mut observer, &requests);
    first
        .drain_events_observed(requests[0], usize::MAX, &mut observer)
        .expect("drain one output");
    first
        .take_terminal_observed(requests[0], &mut observer)
        .expect("take terminal")
        .expect("retained terminal");
    assert!(observer.finish().healthy());
    first
        .step_observed(&mut observer)
        .expect("empty step remains behaviorally valid after observer finish");
    assert_eq!(
        observer.read().status().failure(),
        Some(runnel_scheduler::RunObserverFailure::CallbackAfterFinish)
    );
}

#[test]
fn pre_release_runtime_callback_poison_prevents_later_publication() {
    let prompt = [1_u32];
    let offers = [BatchRequestSpec::absolute(
        runnel_scheduler::RequestSpec::new(&prompt, 1, SamplingPolicy::Greedy, None),
    )];
    let mut engine = evidence_engine();
    let mut observer = engine.prepare_run_observer().expect("pristine observer");

    let empty = engine
        .step_observed(&mut observer)
        .expect("empty scheduler step remains valid");
    assert_eq!(empty, runnel_scheduler::StepReport::default());
    assert_eq!(observer.read().totals(), Default::default());
    assert_eq!(
        observer.read().status().failure(),
        Some(runnel_scheduler::RunObserverFailure::InvalidPhaseBoundary)
    );

    let error = engine
        .prepare_submit_batch(&offers)
        .expect("prepare after observer poison")
        .commit_prepared_batch_observed(&mut observer)
        .expect_err("non-pristine observer must fail before publication");
    assert_eq!(
        error.category(),
        runnel_scheduler::ErrorCategory::InvalidRequest
    );
    assert_eq!(engine.snapshot().queued_requests, 0);
    assert_eq!(engine.snapshot().active_requests, 0);
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
    assert!(!observer.read().status().started());
}

#[test]
fn exact_observer_capacity_retains_32_requests_and_256_outputs() {
    let prompt = [1_u32];
    let offers = [BatchRequestSpec::absolute(runnel_scheduler::RequestSpec::new(
        &prompt,
        8,
        SamplingPolicy::Greedy,
        None,
    )); 32];
    let mut engine = evidence_engine();
    let mut observer = engine
        .prepare_run_observer()
        .expect("exact-capacity observer");
    let fingerprint = observer.allocation_fingerprint();
    let requests = observed_admission(&mut engine, &mut observer, &offers);
    run_observed_to_terminal(&mut engine, &mut observer, &requests);
    for request in requests {
        assert_eq!(
            engine
                .drain_events_observed(request, usize::MAX, &mut observer)
                .expect("capacity output drain")
                .len(),
            8
        );
        assert_eq!(
            engine
                .take_terminal_observed(request, &mut observer)
                .expect("capacity terminal read")
                .expect("capacity terminal")
                .outcome(),
            TerminalOutcome::Completed
        );
    }
    let status = observer.finish();
    assert!(status.healthy(), "capacity failure: {:?}", status.failure());
    assert_eq!(status.request_count(), 32);
    assert_eq!(status.output_timestamp_count(), 256);
    assert_eq!(observer.allocation_fingerprint(), fingerprint);
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}

#[test]
fn observer_records_zero_for_terminal_first_and_zero_output_reap_orders() {
    let prompt = [1_u32];
    let offers = [BatchRequestSpec::absolute(
        runnel_scheduler::RequestSpec::new(&prompt, 2, SamplingPolicy::Greedy, None),
    )];
    let mut engine = evidence_engine();
    let mut observer = engine
        .prepare_run_observer()
        .expect("terminal-first observer");
    let request = observed_admission(&mut engine, &mut observer, &offers)[0];
    run_observed_to_terminal(&mut engine, &mut observer, &[request]);

    let terminal = engine
        .take_terminal_observed(request, &mut observer)
        .expect("terminal-first read")
        .expect("terminal-first value");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(
        observer.read().requests()[0].request_owned_zero_ns(),
        None,
        "buffered outputs still own the request"
    );
    assert_eq!(
        engine
            .drain_events_observed(request, usize::MAX, &mut observer)
            .expect("terminal-first final drain")
            .len(),
        2
    );
    assert!(
        observer.read().requests()[0]
            .request_owned_zero_ns()
            .is_some()
    );
    assert!(observer.finish().healthy());
    assert_eq!(engine.ledger_snapshot().request_used(), 0);

    let zero_offers = [BatchRequestSpec::absolute(
        runnel_scheduler::RequestSpec::new(&prompt, 0, SamplingPolicy::Greedy, None),
    )];
    let mut engine = evidence_engine();
    let mut observer = engine.prepare_run_observer().expect("zero-output observer");
    let request = observed_admission(&mut engine, &mut observer, &zero_offers)[0];
    run_observed_to_terminal(&mut engine, &mut observer, &[request]);
    let terminal = engine
        .take_terminal_observed(request, &mut observer)
        .expect("zero-output terminal read")
        .expect("zero-output terminal");
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
    assert_eq!(terminal.emitted_tokens(), 0);
    let read = observer.read();
    assert!(
        read.output_commit_ns(0)
            .expect("zero-output row")
            .is_empty()
    );
    assert!(read.requests()[0].request_owned_zero_ns().is_some());
    drop(read);
    assert!(observer.finish().healthy());
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}

#[test]
fn queued_and_output_blocked_cancellation_have_closed_observer_lifecycles() {
    let prompt = [1_u32];
    let offers = [BatchRequestSpec::absolute(
        runnel_scheduler::RequestSpec::new(&prompt, 2, SamplingPolicy::Greedy, None),
    )];
    let mut engine = evidence_engine();
    let mut observer = engine.prepare_run_observer().expect("queued observer");
    let request = observed_admission(&mut engine, &mut observer, &offers)[0];
    assert_eq!(
        engine
            .cancel_observed(request, &mut observer)
            .expect("queued cancellation"),
        CancelDisposition::Requested
    );
    let report = engine
        .step_observed(&mut observer)
        .expect("queued cancel step");
    assert_eq!(report.committed_positions, 0);
    assert_eq!(report.terminal_decisions, 1);
    assert!(
        engine
            .drain_events_observed(request, usize::MAX, &mut observer)
            .expect("queued cancel drain")
            .is_empty()
    );
    assert_eq!(
        engine
            .take_terminal_observed(request, &mut observer)
            .expect("queued terminal read")
            .expect("queued terminal")
            .outcome(),
        TerminalOutcome::Cancelled
    );
    let read = observer.read();
    let queued = &read.requests()[0];
    assert_eq!(queued.first_work_start_ns(), None);
    assert_eq!(queued.terminal_decided_ns(), queued.worker_quiescent_ns());
    assert!(queued.worker_quiescent_ns() <= queued.request_owned_zero_ns());
    drop(read);
    assert!(observer.finish().healthy());
    assert_eq!(engine.ledger_snapshot().request_used(), 0);

    let blocked_offers = [BatchRequestSpec::absolute(
        runnel_scheduler::RequestSpec::new(&prompt, 3, SamplingPolicy::Greedy, None),
    )];
    let mut engine = output_blocked_engine();
    let mut observer = engine
        .prepare_run_observer()
        .expect("output-blocked observer");
    let request = observed_admission(&mut engine, &mut observer, &blocked_offers)[0];
    let report = engine
        .step_observed(&mut observer)
        .expect("fill bounded output queue");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(
        engine.request_phase(request).expect("blocked phase"),
        RequestPhase::OutputBlocked
    );
    assert_eq!(
        engine
            .cancel_observed(request, &mut observer)
            .expect("output-blocked cancellation"),
        CancelDisposition::Requested
    );
    let report = engine
        .step_observed(&mut observer)
        .expect("output-blocked cancel step");
    assert_eq!(report.committed_positions, 0);
    assert_eq!(report.terminal_decisions, 1);
    assert_eq!(
        engine
            .drain_events_observed(request, usize::MAX, &mut observer)
            .expect("output-blocked drain")
            .len(),
        1
    );
    assert_eq!(
        engine
            .take_terminal_observed(request, &mut observer)
            .expect("output-blocked terminal read")
            .expect("output-blocked terminal")
            .outcome(),
        TerminalOutcome::Cancelled
    );
    let read = observer.read();
    let blocked = &read.requests()[0];
    assert_eq!(blocked.committed_positions(), 1);
    assert_eq!(blocked.emitted_tokens(), 1);
    assert!(blocked.cancel_linearized_ns() <= blocked.terminal_decided_ns());
    assert_eq!(blocked.terminal_decided_ns(), blocked.worker_quiescent_ns());
    drop(read);
    assert!(observer.finish().healthy());
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}

#[cfg(feature = "deterministic-checkpoint-instrumentation")]
#[test]
fn cancellation_observer_closes_every_selected_work_checkpoint() {
    for (point, expected_commits, expected_outputs) in [
        (CheckpointPoint::PostRouterPreExpert, 0, 0),
        (CheckpointPoint::ReadyToCommitPrePlan, 0, 0),
        (CheckpointPoint::CompositePermitPreFinalSnapshot, 0, 0),
        (CheckpointPoint::PostFinalSnapshot, 1, 1),
    ] {
        let prompt = [1_u32];
        let offers = [BatchRequestSpec::absolute(
            runnel_scheduler::RequestSpec::new(&prompt, 2, SamplingPolicy::Greedy, None),
        )];
        let mut engine = evidence_engine();
        let mut observer = engine
            .prepare_run_observer()
            .expect("checkpoint-matrix observer");
        let request = observed_admission(&mut engine, &mut observer, &offers)[0];
        let mut plan = engine
            .prepare_checkpoint_plan(&[CheckpointDirective::new(
                request,
                0,
                point,
                CheckpointAction::Cancel,
            )])
            .expect("checkpoint-matrix plan");

        let report = engine
            .step_with_checkpoint_plan_observed(&mut plan, &mut observer)
            .expect("checkpoint-matrix step");
        assert_eq!(report.committed_positions, expected_commits, "{point:?}");
        assert_eq!(report.terminal_decisions, 1, "{point:?}");
        plan.ensure_complete().expect("checkpoint directive fired");
        assert_eq!(
            plan.records()
                .next()
                .expect("checkpoint record")
                .effect()
                .expect("checkpoint effect")
                .cancellation(),
            Some(CancelDisposition::Requested),
            "{point:?}"
        );

        assert_eq!(
            engine
                .drain_events_observed(request, usize::MAX, &mut observer)
                .expect("checkpoint-matrix drain")
                .len(),
            expected_outputs,
            "{point:?}"
        );
        assert_eq!(
            engine
                .take_terminal_observed(request, &mut observer)
                .expect("checkpoint-matrix terminal read")
                .expect("checkpoint-matrix terminal")
                .outcome(),
            TerminalOutcome::Cancelled,
            "{point:?}"
        );
        let read = observer.read();
        let request = &read.requests()[0];
        assert!(request.first_work_start_ns().is_some(), "{point:?}");
        assert_eq!(request.committed_positions(), expected_commits, "{point:?}");
        assert_eq!(request.emitted_tokens(), expected_outputs, "{point:?}");
        assert!(
            request.cancel_linearized_ns() <= request.terminal_decided_ns(),
            "{point:?}"
        );
        assert!(
            request.terminal_decided_ns() <= request.worker_quiescent_ns(),
            "{point:?}"
        );
        assert!(
            request.worker_quiescent_ns() <= request.request_owned_zero_ns(),
            "{point:?}"
        );
        drop(read);
        assert!(observer.finish().healthy(), "{point:?}");
        assert_eq!(engine.ledger_snapshot().request_used(), 0, "{point:?}");
    }
}

#[cfg(feature = "deterministic-checkpoint-instrumentation")]
#[test]
fn post_snapshot_cancellation_records_cas_terminal_quiescence_and_reap_order() {
    let prompt = [1_u32];
    let offers = [BatchRequestSpec::absolute(
        runnel_scheduler::RequestSpec::new(&prompt, 2, SamplingPolicy::Greedy, None),
    )];
    let mut engine = evidence_engine();
    let mut observer = engine
        .prepare_run_observer()
        .expect("cancellation observer");
    let requests = observed_admission(&mut engine, &mut observer, &offers);
    let request = requests[0];
    let mut plan = engine
        .prepare_checkpoint_plan(&[CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::PostFinalSnapshot,
            CheckpointAction::Cancel,
        )])
        .expect("post-snapshot cancellation plan");

    let report = engine
        .step_with_checkpoint_plan_observed(&mut plan, &mut observer)
        .expect("observed checkpoint step");
    assert_eq!(report.committed_positions, 1);
    assert_eq!(report.terminal_decisions, 1);
    plan.ensure_complete().expect("checkpoint plan completed");
    assert_eq!(
        plan.records()
            .next()
            .expect("checkpoint record")
            .effect()
            .expect("checkpoint effect")
            .cancellation(),
        Some(CancelDisposition::Requested)
    );
    assert_eq!(
        engine.request_phase(request).expect("cancelled phase"),
        RequestPhase::Terminal
    );
    let output = engine
        .drain_events_observed(request, usize::MAX, &mut observer)
        .expect("one post-snapshot output");
    assert_eq!(output.len(), 1);
    let terminal = engine
        .take_terminal_observed(request, &mut observer)
        .expect("cancelled terminal read")
        .expect("cancelled terminal");
    assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
    assert_eq!(terminal.committed_positions(), 1);
    assert_eq!(terminal.emitted_tokens(), 1);

    let read = observer.read();
    let request = &read.requests()[0];
    let cancel = request
        .cancel_linearized_ns()
        .expect("cancel CAS timestamp");
    let output = read
        .output_commit_ns(0)
        .expect("local cancellation timestamp row")[0]
        .expect("commit timestamp");
    let terminal = request.terminal_decided_ns().expect("terminal timestamp");
    let quiescent = request
        .worker_quiescent_ns()
        .expect("worker quiescence timestamp");
    let zero = request
        .request_owned_zero_ns()
        .expect("request ownership-zero timestamp");
    assert!(cancel <= output);
    assert!(output <= terminal);
    assert!(terminal <= quiescent);
    assert!(quiescent <= zero);
    drop(read);
    assert!(observer.finish().healthy());
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}
