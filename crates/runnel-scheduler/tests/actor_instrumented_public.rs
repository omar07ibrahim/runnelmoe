#![cfg(feature = "actor-stress-instrumentation")]

use std::time::Duration;

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    ActorSemanticObservationKind, RequestSpec, SchedulerActor, SchedulerConfig, SchedulerLimits,
    TerminalOutcome, TryRecvOutput,
};
use tokio::time::timeout;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn feature_on_external_consumer_observes_a_bounded_actor_lifetime() {
    let fixture = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
        .expect("authenticated tiny-v3 artifact");
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .expect("scalar tiny-v3 model");
    let mut limits = SchedulerLimits::tiny();
    limits.max_new_tokens = 2;
    limits.output_capacity_per_request = 2;
    let config = SchedulerConfig::new(&model, limits).expect("scheduler configuration");
    let (actor, probe) = SchedulerActor::spawn_instrumented_with_capacity(model, config, 4)
        .expect("instrumented scheduler actor");
    let recorder = probe.recorder().expect("semantic recorder");
    let initial = recorder.status();
    assert_eq!(initial.observation_count(), 0);
    assert_eq!(initial.observation_limit(), 4);
    assert!(initial.allocated_capacity() >= initial.observation_limit());
    assert!(initial.healthy());

    timeout(Duration::from_secs(5), probe.wait_quiescent())
        .await
        .expect("initial quiescence timeout")
        .expect("initial quiescence");
    let client = actor.client();
    let submission = client
        .try_submit(RequestSpec::new(&[1], 2, SamplingPolicy::Greedy, None))
        .expect("bounded submission");
    let mut handle = timeout(Duration::from_secs(5), submission.wait())
        .await
        .expect("admission timeout")
        .expect("engine admission");
    let request_id = handle.request_id();
    let accepted = handle.accepted_request_stress_witness();
    assert_eq!(accepted.request_id(), request_id);
    assert_ne!(accepted.control_generation(), 0);
    assert_ne!(accepted.endpoint_generation(), 0);
    let terminal = timeout(Duration::from_secs(5), handle.terminal())
        .await
        .expect("terminal timeout")
        .expect("terminal result");
    assert_eq!(terminal.request_id(), request_id);
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);

    let mut output_indices = Vec::new();
    for expected_index in 0..2 {
        let (result, receive) = handle.try_recv_output_with_complete_stress_witness();
        let event = match result.expect("witnessed output receive") {
            TryRecvOutput::Output(event) => event,
            other => panic!("completed request omitted output {expected_index}: {other:?}"),
        };
        assert_eq!(event.request_id(), request_id);
        output_indices.push(event.output_index());
        let primary = receive.primary();
        assert!(primary.boundary_reached());
        assert_eq!(primary.slot_index(), accepted.endpoint_slot_index());
        assert_eq!(primary.slot_generation(), accepted.endpoint_generation());
        assert_eq!(primary.consumed_output(), Some(event));
        assert!(!receive.cached_eof());
        assert_eq!(
            receive.opportunistic_eof().boundary_reached(),
            expected_index == 1
        );
    }
    assert_eq!(output_indices, [0, 1]);
    let (cached, receive) = handle.try_recv_output_with_complete_stress_witness();
    assert_eq!(cached.expect("cached EOF"), TryRecvOutput::Eof);
    assert!(receive.cached_eof());
    assert!(!receive.primary().boundary_reached());
    assert!(!receive.opportunistic_eof().boundary_reached());
    drop(handle);

    let reaped = timeout(Duration::from_secs(5), probe.wait_quiescent())
        .await
        .expect("reap quiescence timeout")
        .expect("reap quiescence");
    assert_eq!(reaped.outstanding_requests, 0);
    assert_eq!(reaped.request_bytes, 0);

    let recording = recorder.recording().expect("semantic recording");
    let final_status = recording.status();
    assert_eq!(final_status.observation_count(), 4);
    assert_eq!(final_status.observation_limit(), 4);
    assert_eq!(
        final_status.allocated_capacity(),
        initial.allocated_capacity()
    );
    assert!(final_status.healthy());
    assert!(
        recording
            .observations()
            .iter()
            .all(|observation| observation.request_id() == request_id)
    );
    for (kind, expected) in [
        (ActorSemanticObservationKind::Output, 2),
        (ActorSemanticObservationKind::Terminal, 1),
        (ActorSemanticObservationKind::OutputEof, 1),
    ] {
        assert_eq!(
            recording
                .observations()
                .iter()
                .filter(|observation| observation.kind() == kind)
                .count(),
            expected
        );
    }

    let report = timeout(Duration::from_secs(5), actor.shutdown())
        .await
        .expect("shutdown timeout")
        .expect("cooperative shutdown");
    assert_eq!(report.accepted_submissions(), 1);
    assert_eq!(report.rejected_submissions(), 0);
    assert_eq!(report.engine().remaining_shared_bytes, 0);
}
