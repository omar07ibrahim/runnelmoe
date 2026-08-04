#![cfg(not(feature = "actor-stress-instrumentation"))]

use std::time::Duration;

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    RequestSpec, SchedulerActor, SchedulerConfig, SchedulerLimits, TerminalOutcome, TryRecvOutput,
};
use tokio::time::timeout;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn feature_off_public_actor_completes_and_releases_one_request() {
    let fixture = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default())
        .expect("authenticated tiny-v3 artifact");
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .expect("scalar tiny-v3 model");
    let mut limits = SchedulerLimits::tiny();
    limits.max_new_tokens = 2;
    limits.output_capacity_per_request = 2;
    let config = SchedulerConfig::new(&model, limits).expect("scheduler configuration");
    let actor = SchedulerActor::spawn(model, config).expect("scheduler actor");
    let client = actor.client();

    let submission = client
        .try_submit(RequestSpec::new(&[1], 2, SamplingPolicy::Greedy, None))
        .expect("bounded submission");
    let mut handle = timeout(Duration::from_secs(5), submission.wait())
        .await
        .expect("admission timeout")
        .expect("engine admission");
    let request_id = handle.request_id();
    let terminal = timeout(Duration::from_secs(5), handle.terminal())
        .await
        .expect("terminal timeout")
        .expect("terminal result");
    assert_eq!(terminal.request_id(), request_id);
    assert_eq!(terminal.outcome(), TerminalOutcome::Completed);

    let mut output_indices = Vec::new();
    loop {
        match timeout(Duration::from_secs(5), handle.recv_output())
            .await
            .expect("output timeout")
            .expect("output receive")
        {
            TryRecvOutput::Output(event) => {
                assert_eq!(event.request_id(), request_id);
                output_indices.push(event.output_index());
            }
            TryRecvOutput::Eof => break,
            TryRecvOutput::Empty => panic!("awaited receive returned empty"),
        }
    }
    assert_eq!(output_indices, [0, 1]);
    drop(handle);

    let report = timeout(Duration::from_secs(5), actor.shutdown())
        .await
        .expect("shutdown timeout")
        .expect("cooperative shutdown");
    assert_eq!(report.accepted_submissions(), 1);
    assert_eq!(report.rejected_submissions(), 0);
    assert_eq!(report.shutdown_cancellations(), 0);
    assert_eq!(report.engine().terminated_requests, 0);
    assert_eq!(report.engine().discarded_output_events, 0);
    assert_eq!(report.engine().released_request_bytes, 0);
    assert_eq!(report.engine().remaining_shared_bytes, 0);
}
