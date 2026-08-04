//! Race-invariant actor harness substrate.
//!
//! This module authenticates immutable inputs, constructs fresh actors, and
//! normalizes cleanup and semantic effects. It deliberately excludes accepted
//! request publication, producer orchestration, action records, and interval
//! clocks because the deterministic golden and genuine race require different
//! synchronization contracts for those responsibilities.

use std::collections::{BTreeMap, BTreeSet};

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    ActorControlCasWitness, ActorDisconnectDisposition, ActorProbe, ActorProbeSnapshot,
    ActorReceiveWitness, ActorRequestDropWitnessSink, ActorSemanticObservationKind,
    ActorShutdownReport, ActorStressRecorder, ActorStressRecorderStatus, ActorStressRecording,
    OutputEvent, RequestHandle, RequestSpec, SchedulerActor, SchedulerError, SchedulerLimits,
    SchedulerResult, TerminalOutcome, TerminalResult, TryRecvOutput,
};
use sha2::{Digest, Sha256};

use super::{
    Action, ActorConfig, Descriptor, FIXTURE_BYTES, FIXTURE_FILE_DIGEST, FIXTURE_ID, Fixture,
    TINY_V3_SPEC_BYTES, actions, canonical_json_ascii, descriptors, expected_actor_config,
};

pub(super) const REQUEST_COUNT: usize = 64;
pub(super) const ACTION_COUNT: usize = 1_024;
pub(super) const OBSERVATION_LIMIT: usize = 675;
pub(super) const PUMP_ENTRY_LIMIT: u64 = 4_096;
pub(super) const EOS_TOKEN_ID: u32 = 0;

pub(super) const EXPECTED_SPEC_DIGEST: &str =
    "sha256:ed57d7961e65c76223c169cabebaff9c02d8293da026abb0c0c0a22d38079845";
pub(super) const EXPECTED_ARTIFACT_ID: &str =
    "sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c";
pub(super) const EXPECTED_OBJECT_DIGEST: &str =
    "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab";
pub(super) const EXPECTED_PAGE_TABLE_DIGEST: &str =
    "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c";

pub(super) type HarnessResult<T> = Result<T, String>;

pub(super) struct AuthenticatedWorkload {
    pub(super) actor_config: ActorConfig,
    pub(super) descriptors: Vec<Descriptor>,
    pub(super) actions: Vec<Action>,
}

pub(super) struct FreshActor {
    pub(super) actor: SchedulerActor,
    pub(super) probe: ActorProbe,
    pub(super) recorder: ActorStressRecorder,
    pub(super) initial_probe: ActorProbeSnapshot,
    pub(super) initial_recorder: ActorStressRecorderStatus,
    pub(super) max_outstanding_requests: u64,
    pub(super) output_capacity_per_request: usize,
}

pub(super) struct CleanupResult {
    pub(super) terminal: TerminalResult,
    pub(super) outputs: Vec<OutputEvent>,
}

#[derive(Clone, Copy)]
pub(super) enum DropWitnessContext {
    Script,
    Cleanup,
}

impl DropWitnessContext {
    fn arm_error(self, error: impl std::fmt::Display) -> String {
        match self {
            Self::Script => format!("failed to arm destructor witness: {error}"),
            Self::Cleanup => format!("failed to arm cleanup destructor witness: {error}"),
        }
    }

    fn take_error(self, error: impl std::fmt::Display) -> String {
        match self {
            Self::Script => format!("failed to take destructor witness: {error}"),
            Self::Cleanup => format!("failed to take cleanup destructor witness: {error}"),
        }
    }

    const fn missing_error(self) -> &'static str {
        match self {
            Self::Script => "destructor did not publish its witness",
            Self::Cleanup => "cleanup destructor did not publish its witness",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct OutputRecord {
    pub(super) client_index: u32,
    pub(super) request_id: u64,
    pub(super) output_index: u32,
    pub(super) token_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct TerminalRecord {
    pub(super) client_index: u32,
    pub(super) request_id: u64,
    pub(super) outcome: SemanticTerminalOutcome,
    pub(super) error: u8,
    pub(super) committed_positions: u32,
    pub(super) emitted_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) enum SemanticTerminalOutcome {
    Completed,
    Cancelled,
}

impl SemanticTerminalOutcome {
    pub(super) const fn golden_code(self) -> u8 {
        match self {
            Self::Completed => 0,
            Self::Cancelled => 1,
        }
    }

    #[allow(dead_code, reason = "used by the preregistered race-history encoder")]
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(super) struct EofRecord {
    pub(super) client_index: u32,
    pub(super) request_id: u64,
}

pub(super) struct SemanticRecords {
    pub(super) outputs: Vec<OutputRecord>,
    pub(super) terminals: Vec<TerminalRecord>,
    pub(super) eofs: Vec<EofRecord>,
}

pub(super) fn authenticated_workload() -> HarnessResult<AuthenticatedWorkload> {
    require(
        digest_label(TINY_V3_SPEC_BYTES) == EXPECTED_SPEC_DIGEST,
        "tiny-v3 spec digest changed",
    )?;
    let artifact = FixtureArtifact::build_v3();
    let identity = artifact.identity();
    require(
        identity.artifact_id.to_string() == EXPECTED_ARTIFACT_ID,
        "artifact ID changed",
    )?;
    require(
        identity.object_digest.to_string() == EXPECTED_OBJECT_DIGEST,
        "object digest changed",
    )?;
    require(identity.object_length == 5_600, "object length changed")?;
    require(
        identity.page_table_digest.to_string() == EXPECTED_PAGE_TABLE_DIGEST,
        "page-table digest changed",
    )?;
    require(
        identity.page_table_length == 96,
        "page-table length changed",
    )?;

    let fixture: Fixture = serde_json::from_slice(FIXTURE_BYTES)
        .map_err(|error| format!("actor fixture did not parse: {error}"))?;
    require(
        digest_label(FIXTURE_BYTES) == FIXTURE_FILE_DIGEST,
        "actor fixture file digest changed",
    )?;
    require(fixture.fixture_id == FIXTURE_ID, "actor fixture ID changed")?;
    require(
        fixture.actor_config == expected_actor_config(),
        "actor fixture configuration changed",
    )?;
    require(
        fixture.parameters.action_count == ACTION_COUNT,
        "fixture action count changed",
    )?;
    require(
        fixture.parameters.request_count == REQUEST_COUNT,
        "fixture request count changed",
    )?;
    let descriptor_stream = descriptors(REQUEST_COUNT);
    require(
        fixture.descriptor_vectors.count == descriptor_stream.len()
            && fixture.descriptor_vectors.digest
                == digest_label(&canonical_json_ascii(&descriptor_stream)),
        "executed descriptor stream differs from the authenticated corpus",
    )?;
    let action_stream = actions(ACTION_COUNT);
    require(
        fixture.action_vectors.count == action_stream.len()
            && fixture.action_vectors.digest == digest_label(&canonical_json_ascii(&action_stream)),
        "executed action stream differs from the authenticated corpus",
    )?;
    Ok(AuthenticatedWorkload {
        actor_config: fixture.actor_config,
        descriptors: descriptor_stream,
        actions: action_stream,
    })
}

pub(super) async fn spawn_fresh_actor(config: &ActorConfig) -> HarnessResult<FreshActor> {
    let fixture_artifact = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture_artifact.to_parts(), Limits::default())
        .map_err(|error| format!("tiny-v3 artifact authentication failed: {error}"))?;
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .map_err(|error| format!("scalar tiny-v3 construction failed: {error}"))?;
    let limits = actor_limits_from_config(config)?;
    let max_outstanding_requests = limits.max_outstanding_requests;
    let output_capacity_per_request = usize::try_from(limits.output_capacity_per_request)
        .map_err(|_| "output capacity does not fit this host".to_owned())?;
    let scheduler_config = runnel_scheduler::SchedulerConfig::new(&model, limits)
        .map_err(|error| format!("actor configuration failed: {error}"))?;
    let (actor, probe) = SchedulerActor::spawn_instrumented_with_capacity(
        model,
        scheduler_config,
        OBSERVATION_LIMIT,
    )
    .map_err(|error| format!("instrumented actor construction failed: {error}"))?;
    let recorder = probe
        .recorder()
        .map_err(|error| format!("semantic recorder unavailable: {error}"))?;
    let initial_recorder = recorder.status();
    require(
        initial_recorder.observation_count() == 0,
        "recorder was not empty",
    )?;
    require(
        initial_recorder.observation_limit() == OBSERVATION_LIMIT,
        "recorder logical limit changed",
    )?;
    require(
        initial_recorder.allocated_capacity() >= OBSERVATION_LIMIT,
        "recorder did not preallocate its logical limit",
    )?;
    require(initial_recorder.healthy(), "recorder began unhealthy")?;
    let initial_probe = probe
        .wait_quiescent()
        .await
        .map_err(|error| format!("initial quiescence failed: {error}"))?;
    require(
        initial_probe.quiescent(),
        "initial actor state was not quiescent",
    )?;
    Ok(FreshActor {
        actor,
        probe,
        recorder,
        initial_probe,
        initial_recorder,
        max_outstanding_requests,
        output_capacity_per_request,
    })
}

pub(super) fn request_spec(descriptor: &Descriptor) -> HarnessResult<RequestSpec<'_>> {
    Ok(RequestSpec::new(
        &descriptor.prompt,
        to_usize(descriptor.max_new_tokens, "max new tokens")?,
        SamplingPolicy::Greedy,
        descriptor.deadline_ns.0.map(u64::from),
    ))
}

pub(super) fn receive_once_with_witness(
    handle: &mut RequestHandle,
) -> (SchedulerResult<TryRecvOutput>, ActorReceiveWitness) {
    handle.try_recv_output_with_complete_stress_witness()
}

pub(super) fn drop_receiver_with_witness(
    handle: RequestHandle,
    context: DropWitnessContext,
) -> HarnessResult<(
    SchedulerResult<ActorDisconnectDisposition>,
    Option<ActorControlCasWitness>,
)> {
    let sink = ActorRequestDropWitnessSink::new();
    drop_receiver_with_preallocated_witness(handle, sink, context)
}

pub(super) fn drop_receiver_with_preallocated_witness(
    mut handle: RequestHandle,
    sink: ActorRequestDropWitnessSink,
    context: DropWitnessContext,
) -> HarnessResult<(
    SchedulerResult<ActorDisconnectDisposition>,
    Option<ActorControlCasWitness>,
)> {
    handle
        .arm_drop_witness(sink.clone())
        .map_err(|error| context.arm_error(error))?;
    drop(handle);
    let witness = sink
        .take()
        .map_err(|error| context.take_error(error))?
        .ok_or_else(|| context.missing_error().to_owned())?;
    Ok(witness.into_parts())
}

pub(super) async fn finish_receiver(
    _client_index: usize,
    mut handle: RequestHandle,
) -> HarnessResult<CleanupResult> {
    let request_id = handle.request_id();
    let terminal = handle
        .terminal()
        .await
        .map_err(|error| format!("cleanup terminal receive failed: {error}"))?;
    require(
        terminal.request_id() == request_id,
        "cleanup terminal identity changed",
    )?;
    let mut outputs = Vec::new();
    loop {
        match handle
            .recv_output()
            .await
            .map_err(|error| format!("cleanup output receive failed: {error}"))?
        {
            TryRecvOutput::Output(event) => {
                require(
                    event.request_id() == request_id,
                    "cleanup output identity changed",
                )?;
                outputs.push(event);
            }
            TryRecvOutput::Eof => break,
            TryRecvOutput::Empty => {
                return Err("awaited cleanup receive returned empty".to_owned());
            }
        }
    }
    let (disconnect, control) = drop_receiver_with_witness(handle, DropWitnessContext::Cleanup)?;
    require(
        control.is_none(),
        "terminal-plus-EOF cleanup destructor performed a second control mutation",
    )?;
    let error = match disconnect {
        Err(error) => error,
        Ok(_) => {
            return Err("terminal-plus-EOF cleanup disconnect remained generation-live".to_owned());
        }
    };
    require(
        matches!(&error, SchedulerError::RequestNotFound),
        format!("cleanup destructor returned an unexpected error: {error}"),
    )?;
    Ok(CleanupResult { terminal, outputs })
}

pub(super) fn validate_shutdown(
    report: ActorShutdownReport,
    accepted: usize,
    rejected: usize,
    final_request_bytes: u64,
    final_shared_bytes: u64,
) -> HarnessResult<()> {
    require(
        report.accepted_submissions() == accepted,
        "actor acceptance count changed",
    )?;
    require(
        report.rejected_submissions() == rejected,
        "actor rejection count changed",
    )?;
    require(
        report.shutdown_cancellations() == 0,
        "shutdown cancelled a request",
    )?;
    let engine = report.engine();
    require(
        engine.terminated_requests == 0,
        "shutdown terminalized a request",
    )?;
    require(
        engine.discarded_output_events == 0,
        "shutdown discarded output",
    )?;
    require(
        engine.released_request_bytes == 0,
        "shutdown released request-owned bytes",
    )?;
    require(
        engine.remaining_shared_bytes == 0,
        "shutdown retained shared bytes",
    )?;
    require(final_request_bytes == 0, "final request ledger is nonzero")?;
    require(final_shared_bytes == 0, "final shared ledger is nonzero")
}

pub(super) fn collect_semantic_records(
    recording: &ActorStressRecording,
    accepted_ids: &[u64; REQUEST_COUNT],
    descriptors: &[Descriptor],
) -> HarnessResult<SemanticRecords> {
    let mut clients_by_id = BTreeMap::new();
    for (index, &request_id) in accepted_ids.iter().enumerate() {
        if request_id != 0 {
            require(
                clients_by_id.insert(request_id, index).is_none(),
                "accepted request ID was duplicated",
            )?;
        }
    }
    let mut outputs = Vec::new();
    let mut terminals = Vec::new();
    let mut eofs = Vec::new();
    for observation in recording.observations() {
        let request_id = observation.request_id().get();
        let client = *clients_by_id
            .get(&request_id)
            .ok_or_else(|| format!("observation refers to unknown request {request_id}"))?;
        match observation.kind() {
            ActorSemanticObservationKind::Output => {
                let event = observation
                    .output_event()
                    .ok_or_else(|| "output observation omitted its event".to_owned())?;
                outputs.push(OutputRecord {
                    client_index: to_u32(client, "client index")?,
                    request_id,
                    output_index: to_u32(event.output_index(), "output index")?,
                    token_id: event.token(),
                });
            }
            ActorSemanticObservationKind::Terminal => {
                let terminal = observation
                    .terminal_result()
                    .ok_or_else(|| "terminal observation omitted its result".to_owned())?;
                let (outcome, error) = terminal_outcome(terminal.outcome())?;
                terminals.push(TerminalRecord {
                    client_index: to_u32(client, "client index")?,
                    request_id,
                    outcome,
                    error,
                    committed_positions: to_u32(
                        terminal.committed_positions(),
                        "committed positions",
                    )?,
                    emitted_tokens: to_u32(terminal.emitted_tokens(), "emitted tokens")?,
                });
            }
            ActorSemanticObservationKind::OutputEof => eofs.push(EofRecord {
                client_index: to_u32(client, "client index")?,
                request_id,
            }),
        }
    }
    outputs.sort_unstable_by_key(|record| {
        (
            record.request_id,
            record.output_index,
            record.client_index,
            record.token_id,
        )
    });
    terminals.sort_unstable_by_key(|record| (record.request_id, record.client_index));
    eofs.sort_unstable_by_key(|record| (record.request_id, record.client_index));
    require_unique(&outputs, "output record")?;
    require_unique(&terminals, "terminal record")?;
    require_unique(&eofs, "EOF record")?;

    let mut output_counts = BTreeMap::<u32, usize>::new();
    for output in &outputs {
        let expected = output_counts.entry(output.client_index).or_default();
        require(
            to_usize(output.output_index, "output index")? == *expected,
            format!(
                "client {} output publications are not consecutive",
                output.client_index
            ),
        )?;
        *expected += 1;
    }
    let terminal_clients = terminals
        .iter()
        .map(|record| record.client_index)
        .collect::<BTreeSet<_>>();
    let eof_clients = eofs
        .iter()
        .map(|record| record.client_index)
        .collect::<BTreeSet<_>>();
    require(
        terminal_clients.len() == terminals.len(),
        "client has duplicate terminal records",
    )?;
    require(
        eof_clients.len() == eofs.len(),
        "client has duplicate EOF records",
    )?;
    for (&request_id, &client) in &clients_by_id {
        let client_u32 = to_u32(client, "client index")?;
        require(
            terminal_clients.contains(&client_u32),
            "accepted client lacks terminal record",
        )?;
        require(
            eof_clients.contains(&client_u32),
            "accepted client lacks EOF record",
        )?;
        let terminal = terminals
            .iter()
            .find(|terminal| terminal.client_index == client_u32)
            .ok_or_else(|| "terminal lookup failed".to_owned())?;
        require(
            terminal.request_id == request_id,
            "terminal request identity changed",
        )?;
        let output_count = output_counts.get(&client_u32).copied().unwrap_or(0);
        require(
            to_usize(terminal.emitted_tokens, "emitted tokens")? == output_count,
            "terminal emitted-token count differs from publications",
        )?;
        require(
            output_count <= to_usize(descriptors[client].max_new_tokens, "max new tokens")?,
            "output count exceeds descriptor limit",
        )?;
        let prompt_length = descriptors[client].prompt.len();
        let max_new_tokens = to_usize(descriptors[client].max_new_tokens, "max new tokens")?;
        require(
            prompt_length > 0 && max_new_tokens > 0,
            "accepted descriptor has an empty progress envelope",
        )?;
        let maximum_positions = prompt_length
            .checked_add(max_new_tokens)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| "descriptor progress envelope overflowed".to_owned())?;
        let committed_positions = to_usize(terminal.committed_positions, "committed positions")?;
        require(
            committed_positions <= maximum_positions,
            format!("client {client} committed positions exceed its descriptor envelope"),
        )?;
        let expected_emitted = committed_positions.saturating_sub(prompt_length - 1);
        require(
            output_count == expected_emitted,
            format!("client {client} committed and emitted progress is inconsistent"),
        )?;
        if terminal.outcome == SemanticTerminalOutcome::Completed {
            require(
                output_count > 0,
                format!("client {client} completed without an output"),
            )?;
            if output_count < max_new_tokens {
                let last = outputs
                    .iter()
                    .rev()
                    .find(|output| output.client_index == client_u32)
                    .ok_or_else(|| format!("client {client} completed without publication"))?;
                require(
                    last.token_id == EOS_TOKEN_ID,
                    format!("client {client} completed early without EOS"),
                )?;
            }
        }
    }
    Ok(SemanticRecords {
        outputs,
        terminals,
        eofs,
    })
}

fn actor_limits_from_config(config: &ActorConfig) -> HarnessResult<SchedulerLimits> {
    Ok(SchedulerLimits {
        worker_count: to_u64(config.worker_count, "worker count")?,
        command_capacity: to_u64(config.command_capacity, "command capacity")?,
        max_outstanding_requests: to_u64(
            config.max_outstanding_requests,
            "maximum outstanding requests",
        )?,
        max_active_requests: to_u64(config.max_active_requests, "maximum active requests")?,
        max_queued_requests: to_u64(config.max_queued_requests, "maximum queued requests")?,
        max_retained_terminal_results: to_u64(
            config.max_retained_terminal_results,
            "maximum retained terminal results",
        )?,
        max_prompt_tokens: to_u64(config.max_prompt_tokens, "maximum prompt tokens")?,
        max_new_tokens: to_u64(config.max_new_tokens, "maximum new tokens")?,
        max_context_tokens: to_u64(config.max_context_tokens, "maximum context tokens")?,
        state_page_tokens: to_u64(config.state_page_tokens, "state page tokens")?,
        output_capacity_per_request: to_u64(
            config.output_capacity_per_request,
            "output capacity per request",
        )?,
        batch_width: to_u64(config.batch_width, "batch width")?,
        waves_per_step: to_u64(config.waves_per_step, "waves per step")?,
        trace_capacity: to_u64(config.trace_capacity, "trace capacity")?,
        logical_memory_limit_bytes: config.logical_memory_limit_bytes,
        page_pool_partition_bytes: config.page_pool_partition_bytes,
        admission_reserve_bytes: config.admission_reserve_bytes,
    })
}

pub(super) fn terminal_outcome(
    outcome: TerminalOutcome,
) -> HarnessResult<(SemanticTerminalOutcome, u8)> {
    match outcome {
        TerminalOutcome::Completed => Ok((SemanticTerminalOutcome::Completed, 0)),
        TerminalOutcome::Cancelled => Ok((SemanticTerminalOutcome::Cancelled, 0)),
        TerminalOutcome::DeadlineExceeded => {
            Err("deadline-free golden request reached deadline-exceeded terminal state".to_owned())
        }
        TerminalOutcome::Failed { category } => Err(format!(
            "fault-free golden request failed with category {}",
            category.as_str()
        )),
    }
}

pub(super) fn action_ordinal(action: &Action) -> u32 {
    match action {
        Action::Submit { ordinal, .. }
        | Action::Cancel { ordinal, .. }
        | Action::Drop { ordinal, .. }
        | Action::Drain { ordinal, .. }
        | Action::Wake { ordinal, .. } => *ordinal,
    }
}

pub(super) fn action_producer(action: &Action) -> u32 {
    match action {
        Action::Submit { producer, .. }
        | Action::Cancel { producer, .. }
        | Action::Drop { producer, .. }
        | Action::Drain { producer, .. }
        | Action::Wake { producer, .. } => *producer,
    }
}

pub(super) fn validate_producer_assignment(action: &Action) -> HarnessResult<()> {
    let (producer, expected) = match action {
        Action::Submit {
            producer,
            submit_attempt,
            request_index,
            ..
        } => {
            if let Some(request_index) = request_index.0 {
                require(
                    to_usize(request_index, "submit client index")? < REQUEST_COUNT,
                    "submit client index is out of range",
                )?;
            }
            (*producer, *submit_attempt % 2)
        }
        Action::Cancel {
            producer,
            request_index,
            ..
        } => {
            require(
                to_usize(*request_index, "client index")? < REQUEST_COUNT,
                "cancel client index is out of range",
            )?;
            (*producer, (*request_index + 1) % 2)
        }
        Action::Drop {
            producer,
            request_index,
            ..
        }
        | Action::Drain {
            producer,
            request_index,
            ..
        } => {
            require(
                to_usize(*request_index, "client index")? < REQUEST_COUNT,
                "receiver client index is out of range",
            )?;
            (*producer, *request_index % 2)
        }
        Action::Wake {
            ordinal,
            producer,
            selector_index,
        } => {
            require(
                to_usize(*selector_index, "wake selector")? < REQUEST_COUNT,
                "wake selector is out of range",
            )?;
            (*producer, *ordinal % 2)
        }
    };
    require(
        producer == expected,
        format!(
            "action {} producer {producer} differs from frozen formula {expected}",
            action_ordinal(action)
        ),
    )
}

pub(super) fn action_target_index(action: &Action) -> Option<usize> {
    let index = match action {
        Action::Submit { request_index, .. } => request_index.0?,
        Action::Cancel { request_index, .. }
        | Action::Drop { request_index, .. }
        | Action::Drain { request_index, .. } => *request_index,
        Action::Wake { .. } => return None,
    };
    usize::try_from(index).ok()
}

pub(super) fn action_schema_client_index(action: &Action) -> Option<u32> {
    match action {
        Action::Submit { request_index, .. } => request_index.0,
        Action::Cancel { request_index, .. }
        | Action::Drop { request_index, .. }
        | Action::Drain { request_index, .. } => Some(*request_index),
        Action::Wake { selector_index, .. } => Some(*selector_index),
    }
}

pub(super) fn require(condition: bool, message: impl Into<String>) -> HarnessResult<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

pub(super) fn require_unique<T: Ord>(records: &[T], label: &str) -> HarnessResult<()> {
    require(
        records.windows(2).all(|pair| pair[0] != pair[1]),
        format!("duplicate {label}"),
    )
}

pub(super) fn to_u8(value: u32, field: &str) -> HarnessResult<u8> {
    u8::try_from(value).map_err(|_| format!("{field} does not fit u8"))
}

pub(super) fn to_u32(value: usize, field: &str) -> HarnessResult<u32> {
    u32::try_from(value).map_err(|_| format!("{field} does not fit u32"))
}

pub(super) fn to_u64(value: usize, field: &str) -> HarnessResult<u64> {
    u64::try_from(value).map_err(|_| format!("{field} does not fit u64"))
}

pub(super) fn to_usize(value: u32, field: &str) -> HarnessResult<usize> {
    usize::try_from(value).map_err(|_| format!("{field} does not fit usize"))
}

pub(super) fn digest_label(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("sha256:{digest:x}")
}
