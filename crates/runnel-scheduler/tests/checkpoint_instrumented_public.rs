#![cfg(feature = "deterministic-checkpoint-instrumentation")]

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    CancelDisposition, CheckpointAction, CheckpointDirective, CheckpointPoint, RequestPhase,
    RequestSpec, SchedulerConfig, SchedulerEngine, SchedulerLimits, ServiceTraceCursor,
    TerminalOutcome,
};

#[test]
fn feature_on_external_consumer_runs_a_concrete_bounded_checkpoint_plan() {
    let fixture = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
        .expect("authenticated tiny-v3 artifact");
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .expect("scalar tiny-v3 model");
    let config = SchedulerConfig::new(&model, SchedulerLimits::tiny())
        .expect("checkpoint scheduler configuration");
    let mut engine = SchedulerEngine::new(model, config).expect("checkpoint scheduler engine");
    let request = engine
        .try_submit(RequestSpec::new(&[1], 1, SamplingPolicy::Greedy, None))
        .expect("accepted checkpoint request");
    let mut plan = engine
        .prepare_checkpoint_plan(&[CheckpointDirective::new(
            request,
            0,
            CheckpointPoint::PostRouterPreExpert,
            CheckpointAction::Cancel,
        )])
        .expect("engine-bound checkpoint plan");

    let report = engine
        .step_with_checkpoint_plan(&mut plan)
        .expect("checkpoint-controlled step");
    assert_eq!(report.committed_positions, 0);
    assert_eq!(report.expert_tasks, 0);
    assert_eq!(report.terminal_decisions, 1);
    plan.ensure_complete().expect("complete checkpoint plan");
    let record = plan.records().next().expect("checkpoint record");
    assert_eq!(record.directive().request_id(), request);
    assert_eq!(record.directive().position(), 0);
    assert_eq!(
        record.directive().point(),
        CheckpointPoint::PostRouterPreExpert
    );
    assert_eq!(record.directive().action(), CheckpointAction::Cancel);
    assert_eq!(
        record.effect().expect("checkpoint effect").cancellation(),
        Some(CancelDisposition::Requested)
    );
    assert!(
        engine
            .drain_events(request, usize::MAX)
            .expect("cancelled checkpoint output")
            .is_empty()
    );
    assert_eq!(
        engine
            .take_terminal(request)
            .expect("checkpoint terminal query")
            .expect("checkpoint terminal")
            .outcome(),
        TerminalOutcome::Cancelled
    );
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}

#[test]
fn frozen_cancellation_pressure_12_hits_three_distinct_boundaries() {
    let fixture = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
        .expect("authenticated tiny-v3 artifact");
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .expect("scalar tiny-v3 model");
    let config = SchedulerConfig::new(&model, SchedulerLimits::evidence())
        .expect("frozen M5 scheduler configuration");
    let mut engine = SchedulerEngine::new(model, config).expect("frozen M5 scheduler engine");
    let mut prompt = Vec::with_capacity(896);
    prompt.push(1);
    prompt.extend((0..895).map(|index| [14, 16, 6][index % 3]));
    let mut requests = Vec::with_capacity(12);
    for expected in 1_u64..=12 {
        let request = engine
            .try_submit(RequestSpec::new(&prompt, 8, SamplingPolicy::Greedy, None))
            .expect("accepted frozen cancellation request");
        assert_eq!(request.get(), expected);
        requests.push(request);
    }
    assert_eq!(
        engine
            .cancel(requests[1])
            .expect("queued ID 2 cancellation"),
        CancelDisposition::Requested
    );
    let mut plan = engine
        .prepare_checkpoint_plan(&[
            CheckpointDirective::new(
                requests[4],
                0,
                CheckpointPoint::PostFinalSnapshot,
                CheckpointAction::Cancel,
            ),
            CheckpointDirective::new(
                requests[7],
                8,
                CheckpointPoint::PostRouterPreExpert,
                CheckpointAction::Cancel,
            ),
            CheckpointDirective::new(
                requests[10],
                895,
                CheckpointPoint::CompositePermitPreFinalSnapshot,
                CheckpointAction::Cancel,
            ),
        ])
        .expect("frozen cancellation checkpoint plan");

    let mut promoted = 0_usize;
    let mut selected = 0_usize;
    let mut expert_tasks = 0_usize;
    let mut committed = 0_usize;
    let mut terminal_decisions = 0_usize;
    for _ in 0..4_096 {
        if requests.iter().all(|request| {
            engine
                .request_phase(*request)
                .expect("retained request phase")
                == RequestPhase::Terminal
        }) {
            break;
        }
        let report = engine
            .step_with_checkpoint_plan(&mut plan)
            .expect("frozen cancellation scheduler step");
        promoted += report.promoted_requests;
        selected += report.selected_positions;
        expert_tasks += report.expert_tasks;
        committed += report.committed_positions;
        terminal_decisions += report.terminal_decisions;
    }
    assert!(requests.iter().all(|request| {
        engine
            .request_phase(*request)
            .expect("final retained phase")
            == RequestPhase::Terminal
    }));
    plan.ensure_complete()
        .expect("all frozen cancellation checkpoints fired");
    assert_eq!(promoted, 11);
    assert_eq!(selected, 8_130);
    // Tiny-v3 routes each surviving position to two experts. The post-router
    // cancellation suppresses both expert calls for exactly one position.
    assert_eq!(expert_tasks, 16_258);
    assert_eq!(committed, 8_128);
    assert_eq!(terminal_decisions, 12);
    assert!(plan.records().all(|record| {
        record.effect().is_some_and(|effect| {
            effect.cancellation() == Some(CancelDisposition::Requested)
                && effect.deadline().is_none()
        })
    }));

    let service = engine
        .service_trace_since(ServiceTraceCursor::origin())
        .expect("complete frozen service trace");
    assert!(service.status().healthy());
    assert_eq!(service.events().len(), 8_128);
    for (index, request) in requests.iter().copied().enumerate() {
        let observed = service
            .events()
            .iter()
            .filter(|event| event.request_id() == request)
            .count();
        let expected = match index + 1 {
            2 => 0,
            5 => 1,
            8 => 8,
            11 => 895,
            _ => 903,
        };
        assert_eq!(
            observed,
            expected,
            "service count for request ID {}",
            index + 1
        );
    }
    drop(service);
    assert!(
        engine
            .ledger_trace_since(runnel_scheduler::LedgerTraceCursor::origin())
            .expect("frozen cancellation ledger trace")
            .status()
            .healthy()
    );

    for (index, request) in requests.into_iter().enumerate() {
        let events = engine
            .drain_events(request, usize::MAX)
            .expect("frozen cancellation output");
        let terminal = engine
            .take_terminal(request)
            .expect("frozen cancellation terminal query")
            .expect("frozen cancellation terminal");
        match index + 1 {
            2 => {
                assert!(events.is_empty());
                assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
                assert_eq!(terminal.committed_positions(), 0);
            }
            5 => {
                assert!(events.is_empty());
                assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
                assert_eq!(terminal.committed_positions(), 1);
            }
            8 => {
                assert!(events.is_empty());
                assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
                assert_eq!(terminal.committed_positions(), 8);
            }
            11 => {
                assert!(events.is_empty());
                assert_eq!(terminal.outcome(), TerminalOutcome::Cancelled);
                assert_eq!(terminal.committed_positions(), 895);
            }
            _ => {
                assert_eq!(events.len(), 8);
                assert_eq!(terminal.outcome(), TerminalOutcome::Completed);
                assert_eq!(terminal.committed_positions(), 903);
            }
        }
    }
    assert_eq!(engine.ledger_snapshot().request_used(), 0);
}
