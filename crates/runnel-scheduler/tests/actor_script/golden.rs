use std::{
    collections::{BTreeMap, BTreeSet},
    fs::OpenOptions,
    io::{Read, Write},
    mem::size_of,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use runnel_fixture::FixtureArtifact;
use runnel_format::{Artifact, Limits};
use runnel_runtime::{BackendRequest, SamplingPolicy, TinyModel};
use runnel_scheduler::{
    ActorDisconnectDisposition, ActorProbe, ActorProbeSnapshot, ActorRequestDropWitnessSink,
    ActorSemanticObservationKind, ActorShutdownReport, ActorStressRecorderStatus,
    ActorStressRecording, ActorTryPopKind, CancelDisposition, ErrorCategory, OutputEvent,
    RequestCancellation, RequestHandle, RequestSpec, SchedulerActor, SchedulerClient,
    SchedulerError, SchedulerLimits, TerminalOutcome, TerminalResult, TryRecvOutput,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

use super::{
    Action, Descriptor, FIXTURE_BYTES, FIXTURE_FILE_DIGEST, FIXTURE_ID, Fixture,
    TINY_V3_SPEC_BYTES, actions, descriptors, expected_actor_config,
};

const REQUEST_COUNT: usize = 64;
const ACTION_COUNT: usize = 1_024;
const OBSERVATION_LIMIT: usize = 675;
const PUMP_ENTRY_LIMIT: u64 = 4_096;
const TRANSCRIPT_DOMAIN: &[u8] = b"runnel-m5-actor-semantic-transcript-v2\0";
const CAPTURE_SCHEMA: &str = "runnel.actor-semantic-capture/1";
const MAX_CAPTURE_BYTES: usize = 1024 * 1024;
const MAX_TRANSCRIPT_BYTES: usize = 47_823;
const EOS_TOKEN_ID: u32 = 0;
const EXPECTED_TRANSCRIPT_DIGEST: &str =
    "sha256:2711f6b6b28849dd9cb9692f75d97f09d24520645d7b0ccaf7c4c2fd023ddd7a";
const EXPECTED_SPEC_DIGEST: &str =
    "sha256:ed57d7961e65c76223c169cabebaff9c02d8293da026abb0c0c0a22d38079845";
const EXPECTED_ARTIFACT_ID: &str =
    "sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c";
const EXPECTED_OBJECT_DIGEST: &str =
    "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab";
const EXPECTED_PAGE_TABLE_DIGEST: &str =
    "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c";

type HarnessResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum ResultCode {
    SubmitAccepted = 0,
    SubmitOfferExhausted = 1,
    CancelRequested = 2,
    CancelAlreadyRequested = 3,
    CancelAlreadyTerminal = 4,
    ReceiverDropped = 5,
    DrainOutput = 6,
    DrainEmpty = 7,
    DrainEof = 8,
    WakeSignaled = 9,
    TargetUnavailable = 10,
    Error = 11,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ActionRecord {
    ordinal: u32,
    kind: u8,
    producer: u8,
    result: ResultCode,
    error: u8,
    submit_attempt: u32,
    client_index: u32,
    request_id: u64,
    value0: u32,
    value1: u32,
}

impl ActionRecord {
    fn from_action(action: &Action, request_id: u64) -> HarnessResult<Self> {
        let record = match action {
            Action::Submit {
                exhausted,
                ordinal,
                producer,
                request_index,
                submit_attempt,
            } => Self {
                ordinal: *ordinal,
                kind: 0,
                producer: to_u8(*producer, "producer")?,
                result: if *exhausted {
                    ResultCode::SubmitOfferExhausted
                } else {
                    ResultCode::Error
                },
                error: 0,
                submit_attempt: *submit_attempt,
                client_index: request_index.0.unwrap_or(u32::MAX),
                request_id,
                value0: 0,
                value1: 0,
            },
            Action::Cancel {
                ordinal,
                producer,
                request_index,
            } => Self {
                ordinal: *ordinal,
                kind: 1,
                producer: to_u8(*producer, "producer")?,
                result: ResultCode::TargetUnavailable,
                error: 0,
                submit_attempt: u32::MAX,
                client_index: *request_index,
                request_id,
                value0: 0,
                value1: 0,
            },
            Action::Drop {
                ordinal,
                producer,
                request_index,
            } => Self {
                ordinal: *ordinal,
                kind: 2,
                producer: to_u8(*producer, "producer")?,
                result: ResultCode::TargetUnavailable,
                error: 0,
                submit_attempt: u32::MAX,
                client_index: *request_index,
                request_id,
                value0: 0,
                value1: 0,
            },
            Action::Drain {
                ordinal,
                producer,
                request_index,
            } => Self {
                ordinal: *ordinal,
                kind: 3,
                producer: to_u8(*producer, "producer")?,
                result: ResultCode::TargetUnavailable,
                error: 0,
                submit_attempt: u32::MAX,
                client_index: *request_index,
                request_id,
                value0: 0,
                value1: 0,
            },
            Action::Wake {
                ordinal,
                producer,
                selector_index,
            } => Self {
                ordinal: *ordinal,
                kind: 4,
                producer: to_u8(*producer, "producer")?,
                result: ResultCode::WakeSignaled,
                error: 0,
                submit_attempt: u32::MAX,
                client_index: *selector_index,
                request_id: 0,
                value0: 0,
                value1: 0,
            },
        };
        Ok(record)
    }
}

struct RequestState {
    authorities: Mutex<Vec<Option<RequestCancellation>>>,
    ids: [AtomicU64; REQUEST_COUNT],
    receiver_alive: [AtomicBool; REQUEST_COUNT],
    next_expected_id: AtomicU64,
}

impl RequestState {
    fn new() -> Self {
        Self {
            authorities: Mutex::new(
                std::iter::repeat_with(|| None)
                    .take(REQUEST_COUNT)
                    .collect(),
            ),
            ids: std::array::from_fn(|_| AtomicU64::new(0)),
            receiver_alive: std::array::from_fn(|_| AtomicBool::new(false)),
            next_expected_id: AtomicU64::new(1),
        }
    }

    fn accepted_id(&self, index: usize) -> u64 {
        self.ids[index].load(Ordering::Acquire)
    }

    fn register(&self, index: usize, handle: &RequestHandle) -> HarnessResult<()> {
        require(
            index < REQUEST_COUNT,
            "accepted client index is out of range",
        )?;
        let request_id = handle.request_id().get();
        let expected = self.next_expected_id.fetch_add(1, Ordering::AcqRel);
        require(
            request_id == expected,
            format!("accepted request ID {request_id} is not consecutive; expected {expected}"),
        )?;
        self.ids[index]
            .compare_exchange(0, request_id, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| format!("client {index} was accepted more than once"))?;
        let mut authorities = self
            .authorities
            .lock()
            .map_err(|_| "request-authority table is poisoned".to_owned())?;
        require(
            authorities[index].is_none(),
            format!("client {index} already has a cancellation authority"),
        )?;
        authorities[index] = Some(handle.cancellation());
        self.receiver_alive[index].store(true, Ordering::Release);
        Ok(())
    }

    fn authority(&self, index: usize) -> HarnessResult<Option<RequestCancellation>> {
        self.authorities
            .lock()
            .map_err(|_| "request-authority table is poisoned".to_owned())
            .map(|authorities| authorities[index].clone())
    }

    fn mark_receiver_dropped(&self, index: usize) -> HarnessResult<()> {
        require(
            self.receiver_alive[index].swap(false, Ordering::AcqRel),
            format!("client {index} receiver was already absent"),
        )
    }

    fn live_receiver_count(&self) -> usize {
        self.receiver_alive
            .iter()
            .filter(|alive| alive.load(Ordering::Acquire))
            .count()
    }

    fn drop_all_authorities(&self) -> HarnessResult<()> {
        let authorities = {
            let mut table = self
                .authorities
                .lock()
                .map_err(|_| "request-authority table is poisoned".to_owned())?;
            table
                .iter_mut()
                .filter_map(Option::take)
                .collect::<Vec<_>>()
        };
        drop(authorities);
        Ok(())
    }
}

enum ProducerCommand {
    Action {
        action: Action,
        response: oneshot::Sender<HarnessResult<ActionRecord>>,
    },
    Cleanup {
        client_index: usize,
        response: oneshot::Sender<HarnessResult<CleanupResult>>,
    },
}

struct CleanupResult {
    terminal: TerminalResult,
    outputs: Vec<OutputEvent>,
}

struct Producer {
    index: u32,
    client: SchedulerClient,
    probe: ActorProbe,
    state: Arc<RequestState>,
    descriptors: Arc<Vec<Descriptor>>,
    max_outstanding_requests: u64,
    handles: Vec<Option<RequestHandle>>,
}

impl Producer {
    fn new(
        index: u32,
        client: SchedulerClient,
        probe: ActorProbe,
        state: Arc<RequestState>,
        descriptors: Arc<Vec<Descriptor>>,
        max_outstanding_requests: u64,
    ) -> Self {
        Self {
            index,
            client,
            probe,
            state,
            descriptors,
            max_outstanding_requests,
            handles: std::iter::repeat_with(|| None)
                .take(REQUEST_COUNT)
                .collect(),
        }
    }

    async fn run(mut self, mut commands: mpsc::Receiver<ProducerCommand>) -> HarnessResult<()> {
        while let Some(command) = commands.recv().await {
            match command {
                ProducerCommand::Action { action, response } => {
                    let _ = response.send(self.execute(action).await);
                }
                ProducerCommand::Cleanup {
                    client_index,
                    response,
                } => {
                    let _ = response.send(self.cleanup_receiver(client_index).await);
                }
            }
        }
        require(
            self.handles.iter().all(Option::is_none),
            format!("producer {} exited with a live receiver", self.index),
        )
    }

    async fn execute(&mut self, action: Action) -> HarnessResult<ActionRecord> {
        validate_producer_assignment(&action)?;
        require(
            action_producer(&action) == self.index,
            "action was routed to the wrong persistent producer",
        )?;
        let request_id = action_client_index(&action)
            .map(|index| self.state.accepted_id(index))
            .unwrap_or(0);
        let mut record = ActionRecord::from_action(&action, request_id)?;
        match action {
            Action::Submit {
                exhausted,
                request_index,
                ..
            } => {
                if exhausted {
                    require(
                        request_index.0.is_none(),
                        "exhausted offer carried an index",
                    )?;
                } else {
                    let index = to_usize(
                        request_index
                            .0
                            .ok_or_else(|| "in-range offer omitted its index".to_owned())?,
                        "client index",
                    )?;
                    self.submit(index, &mut record).await?;
                }
            }
            Action::Cancel { request_index, .. } => {
                self.cancel(to_usize(request_index, "client index")?, &mut record)?;
            }
            Action::Drop { request_index, .. } => {
                self.drop_receiver(to_usize(request_index, "client index")?, &mut record)?;
            }
            Action::Drain { request_index, .. } => {
                self.drain(to_usize(request_index, "client index")?, &mut record)?;
            }
            Action::Wake { .. } => {
                let witness = self
                    .probe
                    .wake_and_wait()
                    .await
                    .map_err(|error| format!("global wake failed: {error}"))?;
                require(
                    witness.after_park_epoch > witness.before_park_epoch,
                    "wake did not acknowledge a later park epoch",
                )?;
            }
        }
        Ok(record)
    }

    async fn submit(&mut self, index: usize, record: &mut ActionRecord) -> HarnessResult<()> {
        require(
            index % 2 == to_usize(self.index, "producer")?,
            "submit is not at home",
        )?;
        require(
            self.handles[index].is_none(),
            "submit index already owns a receiver",
        )?;
        let descriptor = &self.descriptors[index];
        let request = RequestSpec::new(
            &descriptor.prompt,
            to_usize(descriptor.max_new_tokens, "max new tokens")?,
            SamplingPolicy::Greedy,
            descriptor.deadline_ns.0.map(u64::from),
        );
        let (submission, witness) = self.client.try_submit_with_witness(request);
        let submission = submission.map_err(|error| {
            format!("quiescent in-range submit failed before actor admission: {error}")
        })?;
        require(
            witness.command_slot().is_some(),
            "submit did not claim a command slot",
        )?;
        require(witness.ticket() != 0, "submit command ticket is zero")?;
        require(
            witness.ready_commit_sequence() != 0,
            "submit did not reach the ready FIFO",
        )?;
        match submission.wait().await {
            Ok(handle) => {
                let request_id = handle.request_id().get();
                self.state.register(index, &handle)?;
                self.handles[index] = Some(handle);
                record.result = ResultCode::SubmitAccepted;
                record.request_id = request_id;
            }
            Err(SchedulerError::ResourceExhausted {
                resource,
                required,
                limit,
            }) => {
                let expected_limit = self.max_outstanding_requests;
                require(
                    resource == "request slot count"
                        && required == expected_limit
                        && limit == expected_limit,
                    "in-range rejection was not exact bounded request-slot saturation",
                )?;
                require(
                    to_u64(self.state.live_receiver_count(), "live receiver count")?
                        == expected_limit,
                    "request-slot rejection occurred without every bounded slot being live",
                )?;
                record.result = ResultCode::Error;
                record.error = error_code(ErrorCategory::ResourceExhausted)?;
                record.request_id = 0;
            }
            Err(error) => {
                return Err(format!("unexpected engine admission failure: {error}"));
            }
        }
        Ok(())
    }

    fn cancel(&self, index: usize, record: &mut ActionRecord) -> HarnessResult<()> {
        let Some(authority) = self.state.authority(index)? else {
            return Ok(());
        };
        let (result, witness) = authority.cancel_with_stress_witness();
        require(
            witness.boundary_reached(),
            "cancel missed its control-word boundary",
        )?;
        require(
            witness.expected_generation() != 0,
            "cancel generation is zero",
        )?;
        match result {
            Ok(CancelDisposition::Requested) => record.result = ResultCode::CancelRequested,
            Ok(CancelDisposition::AlreadyRequested) => {
                return Err(
                    "script cancellation remained requested after acknowledged quiescence"
                        .to_owned(),
                );
            }
            Ok(CancelDisposition::AlreadyTerminal) => {
                record.result = ResultCode::CancelAlreadyTerminal;
            }
            Err(error) if error.category() == ErrorCategory::InvalidRequest => {
                record.result = ResultCode::Error;
                record.error = error_code(error.category())?;
            }
            Err(error) => return Err(format!("unexpected cancellation failure: {error}")),
        }
        Ok(())
    }

    fn drop_receiver(&mut self, index: usize, record: &mut ActionRecord) -> HarnessResult<()> {
        require(
            index % 2 == to_usize(self.index, "producer")?,
            "drop is not at home",
        )?;
        let Some(mut handle) = self.handles[index].take() else {
            return Ok(());
        };
        let sink = ActorRequestDropWitnessSink::new();
        handle
            .arm_drop_witness(sink.clone())
            .map_err(|error| format!("failed to arm destructor witness: {error}"))?;
        drop(handle);
        let witness = sink
            .take()
            .map_err(|error| format!("failed to take destructor witness: {error}"))?
            .ok_or_else(|| "destructor did not publish its witness".to_owned())?;
        let (result, control) = witness.into_parts();
        let control = control.ok_or_else(|| "destructor omitted its control witness".to_owned())?;
        require(
            control.boundary_reached(),
            "destructor missed its control boundary",
        )?;
        match result {
            Ok(
                ActorDisconnectDisposition::Requested
                | ActorDisconnectDisposition::AlreadyRequested
                | ActorDisconnectDisposition::AlreadyTerminal,
            ) => {}
            Err(error) => return Err(format!("receiver destructor failed: {error}")),
        }
        self.state.mark_receiver_dropped(index)?;
        record.result = ResultCode::ReceiverDropped;
        Ok(())
    }

    fn drain(&mut self, index: usize, record: &mut ActionRecord) -> HarnessResult<()> {
        require(
            index % 2 == to_usize(self.index, "producer")?,
            "drain is not at home",
        )?;
        let Some(handle) = self.handles[index].as_mut() else {
            return Ok(());
        };
        let (result, witness) = handle.try_recv_output_with_stress_witness();
        require(
            witness.kind() == ActorTryPopKind::Primary,
            "drain witness kind changed",
        )?;
        let result = result.map_err(|error| format!("nonblocking drain failed: {error}"))?;
        match result {
            TryRecvOutput::Output(event) => {
                require(
                    witness.boundary_reached(),
                    "output drain used a sentinel witness",
                )?;
                require(
                    witness.consumed_output() == Some(event),
                    "drain witness output identity differs from its result",
                )?;
                require(
                    witness.drained_after() == witness.drained_before() + 1,
                    "output drain did not advance exactly once",
                )?;
                require(
                    event.request_id().get() == record.request_id,
                    "drain output belongs to another request",
                )?;
                record.result = ResultCode::DrainOutput;
                record.value0 = to_u32(event.output_index(), "output index")?;
                record.value1 = event.token();
            }
            TryRecvOutput::Empty => {
                require(
                    witness.boundary_reached(),
                    "empty drain used a sentinel witness",
                )?;
                require(
                    witness.consumed_output().is_none(),
                    "empty drain consumed output",
                )?;
                require(
                    witness.drained_after() == witness.drained_before(),
                    "empty drain changed its drain counter",
                )?;
                record.result = ResultCode::DrainEmpty;
            }
            TryRecvOutput::Eof => {
                require(
                    witness.consumed_output().is_none(),
                    "EOF drain consumed output",
                )?;
                if witness.boundary_reached() {
                    require(
                        witness.drained_after() == witness.drained_before(),
                        "EOF drain changed its drain counter",
                    )?;
                }
                record.result = ResultCode::DrainEof;
            }
        }
        Ok(())
    }

    async fn cleanup_receiver(&mut self, index: usize) -> HarnessResult<CleanupResult> {
        require(
            index % 2 == to_usize(self.index, "producer")?,
            "cleanup is not at home",
        )?;
        let mut handle = self.handles[index]
            .take()
            .ok_or_else(|| format!("client {index} has no cleanup receiver"))?;
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
        let sink = ActorRequestDropWitnessSink::new();
        handle
            .arm_drop_witness(sink.clone())
            .map_err(|error| format!("failed to arm cleanup destructor witness: {error}"))?;
        drop(handle);
        let witness = sink
            .take()
            .map_err(|error| format!("failed to take cleanup destructor witness: {error}"))?
            .ok_or_else(|| "cleanup destructor did not publish its witness".to_owned())?;
        let (disconnect, control) = witness.into_parts();
        require(
            control.is_none(),
            "terminal-plus-EOF cleanup destructor performed a second control mutation",
        )?;
        let error = match disconnect {
            Err(error) => error,
            Ok(_) => {
                return Err(
                    "terminal-plus-EOF cleanup disconnect remained generation-live".to_owned(),
                );
            }
        };
        require(
            error.category() == ErrorCategory::InvalidRequest,
            format!("cleanup destructor returned an unexpected error: {error}"),
        )?;
        self.state.mark_receiver_dropped(index)?;
        Ok(CleanupResult { terminal, outputs })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct OutputRecord {
    client_index: u32,
    request_id: u64,
    output_index: u32,
    token_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct TerminalRecord {
    client_index: u32,
    request_id: u64,
    outcome: u8,
    error: u8,
    committed_positions: u32,
    emitted_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct EofRecord {
    client_index: u32,
    request_id: u64,
}

struct SemanticRecords {
    outputs: Vec<OutputRecord>,
    terminals: Vec<TerminalRecord>,
    eofs: Vec<EofRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CleanupCancellation {
    Requested,
    AlreadyRequested,
    AlreadyTerminal,
    InvalidRequest,
}

#[derive(Clone, Copy)]
struct ShutdownRecord {
    accepted_submissions: u32,
    rejected_submissions: u32,
    shutdown_cancellations: u32,
    terminated_requests: u32,
    discarded_output_events: u32,
    released_request_bytes: u64,
    remaining_shared_bytes: u64,
    final_request_bytes: u64,
    final_shared_bytes: u64,
}

#[derive(Serialize)]
struct LogicalCapture {
    schema: &'static str,
    workload: CaptureWorkload,
    action_results: Vec<CaptureAction>,
    cleanup_cancellations: Vec<CaptureCleanupCancellation>,
    observations: Vec<CaptureObservation>,
    shutdown: CaptureShutdown,
    diagnostics: CaptureDiagnostics,
}

#[derive(Serialize)]
struct CaptureWorkload {
    fixture_schema: &'static str,
    specification: &'static str,
    fixture_id: &'static str,
    fixture_file_sha256: &'static str,
}

#[derive(Serialize)]
struct CaptureAction {
    ordinal: u32,
    kind: &'static str,
    producer: u8,
    submit_attempt: Option<u32>,
    client_index: Option<u32>,
    result: &'static str,
    error: Option<&'static str>,
    request_id: Option<u64>,
    output: Option<CaptureOutput>,
}

#[derive(Serialize)]
struct CaptureOutput {
    output_index: u32,
    token_id: u32,
}

#[derive(Serialize)]
struct CaptureCleanupCancellation {
    client_index: u32,
    disposition: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum CaptureObservation {
    Output {
        request_id: u64,
        output_index: u32,
        token_id: u32,
    },
    Terminal {
        request_id: u64,
        outcome: &'static str,
        error: Option<&'static str>,
        committed_positions: u32,
        emitted_tokens: u32,
    },
    Eof {
        request_id: u64,
    },
}

#[derive(Serialize)]
struct CaptureShutdown {
    accepted_submissions: u32,
    rejected_submissions: u32,
    shutdown_cancellations: u32,
    terminated_requests: u32,
    discarded_output_events: u32,
    released_request_bytes: u64,
    remaining_shared_bytes: u64,
    final_request_bytes: u64,
    final_shared_bytes: u64,
}

#[derive(Serialize)]
struct CaptureDiagnostics {
    initial_pump_entries: u64,
    final_pump_entries: u64,
    pump_entries_delta: u64,
    engine_steps: u64,
    observer_count: usize,
    observer_limit: usize,
    observer_initial_capacity: usize,
    observer_final_capacity: usize,
    observer_overflowed: bool,
    observer_poisoned: bool,
}

struct GoldenRun {
    transcript: Vec<u8>,
    capture: Vec<u8>,
    digest: String,
    accepted: usize,
    rejected: usize,
    outputs: usize,
    terminals: usize,
    eofs: usize,
    engine_steps: u64,
    pump_entries: u64,
}

async fn run_golden() -> HarnessResult<GoldenRun> {
    authenticate_inputs()?;
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
                == digest_label(&super::canonical_json_ascii(&descriptor_stream)),
        "executed descriptor stream differs from the authenticated corpus",
    )?;
    let descriptors = Arc::new(descriptor_stream);
    let action_stream = actions(ACTION_COUNT);
    require(
        fixture.action_vectors.count == action_stream.len()
            && fixture.action_vectors.digest
                == digest_label(&super::canonical_json_ascii(&action_stream)),
        "executed action stream differs from the authenticated corpus",
    )?;

    let fixture_artifact = FixtureArtifact::build_v3();
    let artifact = Artifact::from_bytes(fixture_artifact.to_parts(), Limits::default())
        .map_err(|error| format!("tiny-v3 artifact authentication failed: {error}"))?;
    let model = TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar)
        .map_err(|error| format!("scalar tiny-v3 construction failed: {error}"))?;
    let limits = actor_limits_from_config(&fixture.actor_config)?;
    let max_outstanding_requests = limits.max_outstanding_requests;
    let output_capacity_per_request = usize::try_from(limits.output_capacity_per_request)
        .map_err(|_| "output capacity does not fit this host".to_owned())?;
    let config = runnel_scheduler::SchedulerConfig::new(&model, limits)
        .map_err(|error| format!("actor configuration failed: {error}"))?;
    let (actor, probe) =
        SchedulerActor::spawn_instrumented_with_capacity(model, config, OBSERVATION_LIMIT)
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

    let initial = probe
        .wait_quiescent()
        .await
        .map_err(|error| format!("initial quiescence failed: {error}"))?;
    require(initial.quiescent(), "initial actor state was not quiescent")?;
    let state = Arc::new(RequestState::new());
    let client = actor.client();
    let (producer_zero, receiver_zero) = mpsc::channel(1);
    let (producer_one, receiver_one) = mpsc::channel(1);
    let task_zero = tokio::spawn(
        Producer::new(
            0,
            client.clone(),
            probe.clone(),
            Arc::clone(&state),
            Arc::clone(&descriptors),
            max_outstanding_requests,
        )
        .run(receiver_zero),
    );
    let task_one = tokio::spawn(
        Producer::new(
            1,
            client,
            probe.clone(),
            Arc::clone(&state),
            Arc::clone(&descriptors),
            max_outstanding_requests,
        )
        .run(receiver_one),
    );
    let producers = [producer_zero, producer_one];

    let mut action_records = Vec::with_capacity(ACTION_COUNT);
    for (expected_ordinal, action) in action_stream.into_iter().enumerate() {
        require(
            to_usize(action_ordinal(&action), "action ordinal")? == expected_ordinal,
            "action order is not consecutive",
        )?;
        let producer = to_usize(action_producer(&action), "producer")?;
        let (response, received) = oneshot::channel();
        producers[producer]
            .send(ProducerCommand::Action { action, response })
            .await
            .map_err(|_| format!("producer {producer} command channel closed"))?;
        let record = received
            .await
            .map_err(|_| format!("producer {producer} dropped its action response"))??;
        action_records.push(record);
        let quiescent = probe
            .wait_quiescent()
            .await
            .map_err(|error| format!("post-action quiescence failed: {error}"))?;
        require(quiescent.quiescent(), "post-action state was not quiescent")?;
        require(
            quiescent.outstanding_requests
                == to_u64(state.live_receiver_count(), "live receiver count")?,
            format!("live request count diverged after action {expected_ordinal}"),
        )?;
    }

    let hold = probe
        .hold_pump()
        .await
        .map_err(|error| format!("cleanup pump hold failed: {error}"))?;
    let mut cleanup_cancellations = BTreeMap::new();
    for index in 0..REQUEST_COUNT {
        if state.accepted_id(index) != 0 {
            let authority = state
                .authority(index)?
                .ok_or_else(|| format!("accepted client {index} lacks cancellation authority"))?;
            let (result, witness) = authority.cancel_with_stress_witness();
            require(
                witness.boundary_reached() && witness.expected_generation() != 0,
                "cleanup cancel missed control boundary",
            )?;
            let disposition = match result {
                Ok(CancelDisposition::Requested) => CleanupCancellation::Requested,
                Ok(CancelDisposition::AlreadyRequested) => {
                    return Err(
                        "cleanup cancellation remained requested after acknowledged quiescence"
                            .to_owned(),
                    );
                }
                Ok(CancelDisposition::AlreadyTerminal) => CleanupCancellation::AlreadyTerminal,
                Err(error) if error.category() == ErrorCategory::InvalidRequest => {
                    CleanupCancellation::InvalidRequest
                }
                Err(error) => return Err(format!("cleanup cancellation failed: {error}")),
            };
            if state.receiver_alive[index].load(Ordering::Acquire) {
                require(
                    disposition != CleanupCancellation::InvalidRequest,
                    format!("live client {index} cancellation authority was stale"),
                )?;
            } else {
                require(
                    matches!(
                        disposition,
                        CleanupCancellation::AlreadyTerminal | CleanupCancellation::InvalidRequest
                    ),
                    format!("reaped client {index} accepted a nonterminal cleanup cancellation"),
                )?;
            }
            require(
                cleanup_cancellations.insert(index, disposition).is_none(),
                "cleanup client was cancelled twice",
            )?;
        }
    }
    hold.release()
        .map_err(|error| format!("cleanup pump release failed: {error}"))?;
    let after_cancel = probe
        .wait_quiescent()
        .await
        .map_err(|error| format!("post-cancel quiescence failed: {error}"))?;
    let remaining_receivers = state.live_receiver_count();
    require(
        after_cancel.outstanding_requests == to_u64(remaining_receivers, "receiver count")?,
        "post-cancel live count does not equal remaining receivers",
    )?;

    let mut cleanup_results = BTreeMap::new();
    for index in 0..REQUEST_COUNT {
        if !state.receiver_alive[index].load(Ordering::Acquire) {
            continue;
        }
        let before = state.live_receiver_count();
        let producer = index % 2;
        let (response, received) = oneshot::channel();
        producers[producer]
            .send(ProducerCommand::Cleanup {
                client_index: index,
                response,
            })
            .await
            .map_err(|_| format!("producer {producer} cleanup channel closed"))?;
        let result = received
            .await
            .map_err(|_| format!("producer {producer} dropped its cleanup response"))??;
        validate_cleanup_result(index, &result, &action_records)?;
        require(
            cleanup_results.insert(index, result).is_none(),
            "duplicate cleanup result",
        )?;
        let reaped = probe
            .wait_quiescent()
            .await
            .map_err(|error| format!("client {index} reap quiescence failed: {error}"))?;
        require(
            reaped.outstanding_requests == to_u64(before - 1, "post-reap count")?,
            format!("client {index} endpoint generation did not reap exactly once"),
        )?;
        let authority = state
            .authority(index)?
            .ok_or_else(|| format!("cleaned client {index} lacks its retained authority"))?;
        let (retained, witness) = authority.cancel_with_stress_witness();
        require(
            matches!(retained, Ok(CancelDisposition::AlreadyTerminal)),
            format!("cleaned client {index} authority lost its terminal tombstone"),
        )?;
        require(
            witness.boundary_reached() && witness.expected_generation() != 0,
            format!("cleaned client {index} retained authority lacked a control-word witness"),
        )?;
        let reprobed = probe.wait_quiescent().await.map_err(|error| {
            format!("client {index} tombstone-probe quiescence failed: {error}")
        })?;
        require(
            reprobed.outstanding_requests == to_u64(before - 1, "post-probe count")?,
            format!("client {index} terminal authority changed the live request count"),
        )?;
    }
    require(
        state.live_receiver_count() == 0,
        "cleanup left a receiver alive",
    )?;
    state.drop_all_authorities()?;
    let pre_shutdown = probe
        .wait_quiescent()
        .await
        .map_err(|error| format!("pre-shutdown quiescence failed: {error}"))?;
    require(
        pre_shutdown.outstanding_requests == 0,
        "pre-shutdown requests remain",
    )?;
    require(
        pre_shutdown.request_bytes == 0,
        "pre-shutdown request ledger is nonzero",
    )?;
    require(
        pre_shutdown.command_reserved == 0
            && pre_shutdown.command_ready == 0
            && pre_shutdown.command_in_flight == 0
            && pre_shutdown.command_responded == 0,
        "pre-shutdown command table is not empty",
    )?;

    drop(producers);
    task_zero
        .await
        .map_err(|error| format!("producer zero panicked: {error}"))??;
    task_one
        .await
        .map_err(|error| format!("producer one panicked: {error}"))??;

    let report = actor
        .shutdown()
        .await
        .map_err(|error| format!("cooperative shutdown failed: {error}"))?;
    let final_probe = probe
        .snapshot()
        .map_err(|error| format!("final actor snapshot failed: {error}"))?;
    require(
        final_probe.owner_done,
        "actor owner was not done after shutdown",
    )?;
    let pump_entries = final_probe
        .pump_entries
        .checked_sub(initial.pump_entries)
        .ok_or_else(|| "pump-entry counter moved backwards".to_owned())?;
    require(
        pump_entries <= PUMP_ENTRY_LIMIT,
        format!("golden run used {pump_entries} pump entries"),
    )?;

    let accepted = action_records
        .iter()
        .filter(|record| record.result == ResultCode::SubmitAccepted)
        .count();
    let rejected = action_records
        .iter()
        .filter(|record| record.kind == 0 && record.result == ResultCode::Error)
        .count();
    validate_shutdown(
        report,
        accepted,
        rejected,
        final_probe.request_bytes,
        final_probe.shared_bytes,
    )?;
    let recording = recorder
        .recording()
        .map_err(|error| format!("semantic recording failed: {error}"))?;
    let final_status = recording.status();
    require(
        final_status.healthy(),
        "semantic recorder overflowed or was poisoned",
    )?;
    require(
        final_status.observation_limit() == OBSERVATION_LIMIT,
        "recorder limit changed",
    )?;
    require(
        final_status.allocated_capacity() == initial_recorder.allocated_capacity(),
        "semantic recorder allocation grew during the run",
    )?;
    let semantic = collect_semantic_records(&recording, &state, &descriptors)?;
    validate_action_records(&action_records, &semantic, output_capacity_per_request)?;
    validate_cleanup_records(
        &cleanup_results,
        &cleanup_cancellations,
        &action_records,
        &semantic,
    )?;
    require(
        final_status.observation_count()
            == semantic.outputs.len() + semantic.terminals.len() + semantic.eofs.len(),
        "recorder status count differs from retained semantic records",
    )?;
    require(
        semantic.terminals.len() == accepted,
        "terminal count differs from acceptance count",
    )?;
    require(
        semantic.eofs.len() == accepted,
        "EOF count differs from acceptance count",
    )?;

    let shutdown = shutdown_record(report, final_probe.request_bytes, final_probe.shared_bytes)?;
    let capture = build_capture(
        &action_records,
        &cleanup_cancellations,
        &recording,
        shutdown,
        initial,
        final_probe,
        initial_recorder,
        final_status,
        report.engine_steps(),
        pump_entries,
    )?;
    let transcript = serialize_transcript(&action_records, &semantic, shutdown)?;
    let digest = digest_label(&transcript);
    Ok(GoldenRun {
        transcript,
        capture,
        digest,
        accepted,
        rejected,
        outputs: semantic.outputs.len(),
        terminals: semantic.terminals.len(),
        eofs: semantic.eofs.len(),
        engine_steps: report.engine_steps(),
        pump_entries,
    })
}

fn authenticate_inputs() -> HarnessResult<()> {
    require(
        digest_label(TINY_V3_SPEC_BYTES) == EXPECTED_SPEC_DIGEST,
        "tiny-v3 spec digest changed",
    )?;
    let fixture = FixtureArtifact::build_v3();
    let identity = fixture.identity();
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
    )
}

fn actor_limits_from_config(config: &super::ActorConfig) -> HarnessResult<SchedulerLimits> {
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

fn validate_cleanup_result(
    index: usize,
    result: &CleanupResult,
    actions: &[ActionRecord],
) -> HarnessResult<()> {
    let request_id = result.terminal.request_id();
    let client_index = to_u32(index, "client index")?;
    let first_output_index = actions
        .iter()
        .filter(|record| {
            record.client_index == client_index && record.result == ResultCode::DrainOutput
        })
        .count();
    for (offset, event) in result.outputs.iter().enumerate() {
        require(
            event.request_id() == request_id,
            "cleanup output identity changed",
        )?;
        require(
            event.output_index() == first_output_index + offset,
            format!("client {index} cleanup FIFO has a gap"),
        )?;
    }
    Ok(())
}

fn validate_shutdown(
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

fn validate_action_records(
    actions: &[ActionRecord],
    semantic: &SemanticRecords,
    output_capacity_per_request: usize,
) -> HarnessResult<()> {
    let mut drained = [0_u32; REQUEST_COUNT];
    let mut eof_seen = [false; REQUEST_COUNT];
    for record in actions {
        require(
            (record.result == ResultCode::Error) == (record.error != 0),
            format!("action {} has an inconsistent error code", record.ordinal),
        )?;
        require(
            record.result != ResultCode::CancelAlreadyRequested,
            format!(
                "action {} observed an already-requested cancellation after quiescence",
                record.ordinal
            ),
        )?;
        if record.result == ResultCode::CancelRequested {
            let terminal = semantic
                .terminals
                .iter()
                .find(|terminal| terminal.request_id == record.request_id)
                .ok_or_else(|| {
                    format!(
                        "action {} cancellation has no terminal record",
                        record.ordinal
                    )
                })?;
            require(
                terminal.outcome == 1,
                format!(
                    "action {} requested cancellation but did not terminate cancelled",
                    record.ordinal
                ),
            )?;
        }
        if record.result == ResultCode::DrainOutput {
            let client = to_usize(record.client_index, "drain client index")?;
            require(
                client < REQUEST_COUNT,
                format!("action {} drain client is out of range", record.ordinal),
            )?;
            require(
                !eof_seen[client],
                format!("action {} drains output after EOF", record.ordinal),
            )?;
            require(
                record.value0 == drained[client],
                format!(
                    "action {} violates client {} FIFO prefix: got {}, expected {}",
                    record.ordinal, client, record.value0, drained[client]
                ),
            )?;
            let publication = semantic
                .outputs
                .iter()
                .find(|output| {
                    output.client_index == record.client_index
                        && output.output_index == record.value0
                })
                .ok_or_else(|| {
                    format!(
                        "action {} drain has no matching client/index publication",
                        record.ordinal
                    )
                })?;
            require(
                publication.request_id == record.request_id
                    && publication.token_id == record.value1,
                format!(
                    "action {} drain differs from its exact semantic publication",
                    record.ordinal
                ),
            )?;
            drained[client] = drained[client]
                .checked_add(1)
                .ok_or_else(|| "drain prefix count overflowed".to_owned())?;
        } else if record.result == ResultCode::DrainEof {
            let client = to_usize(record.client_index, "EOF client index")?;
            require(
                client < REQUEST_COUNT,
                format!("action {} EOF client is out of range", record.ordinal),
            )?;
            let published = semantic
                .outputs
                .iter()
                .filter(|output| output.client_index == record.client_index)
                .count();
            require(
                to_usize(drained[client], "drain prefix count")? == published,
                format!(
                    "action {} acknowledges EOF before conserving client {} publications",
                    record.ordinal, client
                ),
            )?;
            eof_seen[client] = true;
        } else {
            require(
                record.value0 == 0 && record.value1 == 0,
                format!("action {} has nonzero reserved values", record.ordinal),
            )?;
        }
    }
    for (client, &drained_count) in drained.iter().enumerate() {
        let client_index = to_u32(client, "client index")?;
        let published = semantic
            .outputs
            .iter()
            .filter(|output| output.client_index == client_index)
            .count();
        let consumed = to_usize(drained_count, "drain prefix count")?;
        let retained = published
            .checked_sub(consumed)
            .ok_or_else(|| format!("client {client} consumed more output than was published"))?;
        require(
            retained <= output_capacity_per_request,
            format!("client {client} retained output exceeds the frozen endpoint capacity"),
        )?;
    }
    Ok(())
}

fn validate_cleanup_records(
    cleanup: &BTreeMap<usize, CleanupResult>,
    cancellations: &BTreeMap<usize, CleanupCancellation>,
    actions: &[ActionRecord],
    semantic: &SemanticRecords,
) -> HarnessResult<()> {
    let accepted_clients = semantic
        .terminals
        .iter()
        .map(|terminal| terminal.client_index)
        .collect::<BTreeSet<_>>();
    let cancellation_clients = cancellations
        .keys()
        .map(|&client| to_u32(client, "cleanup cancellation client"))
        .collect::<HarnessResult<BTreeSet<_>>>()?;
    require(
        cancellation_clients == accepted_clients,
        "cleanup did not probe every accepted cancellation authority exactly once",
    )?;
    for (&client, &disposition) in cancellations {
        let client_index = to_u32(client, "cleanup cancellation client")?;
        let terminal = semantic
            .terminals
            .iter()
            .find(|terminal| terminal.client_index == client_index)
            .ok_or_else(|| format!("cleanup cancellation client {client} lacks a terminal"))?;
        require(
            disposition != CleanupCancellation::AlreadyRequested,
            format!("cleanup cancellation for client {client} remained requested after quiescence"),
        )?;
        if disposition == CleanupCancellation::Requested {
            require(
                terminal.outcome == 1,
                format!("cleanup cancellation for client {client} did not win terminal state"),
            )?;
        }
    }
    for (&client, result) in cleanup {
        let client_index = to_u32(client, "client index")?;
        let terminal = semantic
            .terminals
            .iter()
            .find(|record| record.client_index == client_index)
            .ok_or_else(|| format!("cleanup client {client} has no semantic terminal"))?;
        let (outcome, error) = terminal_outcome(result.terminal.outcome())?;
        require(
            terminal.request_id == result.terminal.request_id().get()
                && terminal.outcome == outcome
                && terminal.error == error
                && terminal.committed_positions
                    == to_u32(result.terminal.committed_positions(), "committed positions")?
                && terminal.emitted_tokens
                    == to_u32(result.terminal.emitted_tokens(), "emitted tokens")?,
            format!("cleanup client {client} terminal differs from its semantic record"),
        )?;

        let published = semantic
            .outputs
            .iter()
            .filter(|record| record.client_index == client_index)
            .collect::<Vec<_>>();
        let action_consumed = actions
            .iter()
            .filter(|record| {
                record.client_index == client_index && record.result == ResultCode::DrainOutput
            })
            .collect::<Vec<_>>();
        for (offset, record) in action_consumed.iter().enumerate() {
            let expected = published.get(offset).ok_or_else(|| {
                format!("cleanup client {client} action prefix exceeds publications")
            })?;
            require(
                record.request_id == expected.request_id
                    && record.value0 == expected.output_index
                    && record.value1 == expected.token_id,
                format!("cleanup client {client} action prefix differs at offset {offset}"),
            )?;
        }
        for (offset, event) in result.outputs.iter().enumerate() {
            let publication_offset = action_consumed
                .len()
                .checked_add(offset)
                .ok_or_else(|| "cleanup publication offset overflowed".to_owned())?;
            let expected = published
                .get(publication_offset)
                .ok_or_else(|| format!("cleanup client {client} suffix exceeds publications"))?;
            require(
                event.request_id().get() == expected.request_id
                    && to_u32(event.output_index(), "output index")? == expected.output_index
                    && event.token() == expected.token_id,
                format!("cleanup client {client} suffix differs at output {publication_offset}"),
            )?;
        }
        require(
            action_consumed.len() + result.outputs.len() == published.len(),
            format!("cleanup client {client} did not consume the exact FIFO suffix"),
        )?;
    }
    Ok(())
}

fn collect_semantic_records(
    recording: &runnel_scheduler::ActorStressRecording,
    state: &RequestState,
    descriptors: &[Descriptor],
) -> HarnessResult<SemanticRecords> {
    let mut clients_by_id = BTreeMap::new();
    for index in 0..REQUEST_COUNT {
        let request_id = state.accepted_id(index);
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
    outputs.sort_unstable();
    terminals.sort_unstable();
    eofs.sort_unstable();
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
        if terminal.outcome == 0 {
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

fn shutdown_record(
    report: ActorShutdownReport,
    final_request_bytes: u64,
    final_shared_bytes: u64,
) -> HarnessResult<ShutdownRecord> {
    let engine = report.engine();
    Ok(ShutdownRecord {
        accepted_submissions: to_u32(report.accepted_submissions(), "accepted submissions")?,
        rejected_submissions: to_u32(report.rejected_submissions(), "rejected submissions")?,
        shutdown_cancellations: to_u32(report.shutdown_cancellations(), "shutdown cancellations")?,
        terminated_requests: to_u32(engine.terminated_requests, "terminated requests")?,
        discarded_output_events: to_u32(engine.discarded_output_events, "discarded output events")?,
        released_request_bytes: to_u64(engine.released_request_bytes, "released request bytes")?,
        remaining_shared_bytes: to_u64(engine.remaining_shared_bytes, "remaining shared bytes")?,
        final_request_bytes,
        final_shared_bytes,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "every diagnostic boundary is an independently checked capture field"
)]
fn build_capture(
    actions: &[ActionRecord],
    cleanup_cancellations: &BTreeMap<usize, CleanupCancellation>,
    recording: &ActorStressRecording,
    shutdown: ShutdownRecord,
    initial_probe: ActorProbeSnapshot,
    final_probe: ActorProbeSnapshot,
    initial_recorder: ActorStressRecorderStatus,
    final_recorder: ActorStressRecorderStatus,
    engine_steps: u64,
    pump_entries_delta: u64,
) -> HarnessResult<Vec<u8>> {
    let action_results = actions
        .iter()
        .map(capture_action)
        .collect::<HarnessResult<Vec<_>>>()?;
    let cleanup_cancellations = cleanup_cancellations
        .iter()
        .map(|(&client_index, &disposition)| {
            Ok(CaptureCleanupCancellation {
                client_index: to_u32(client_index, "cleanup client index")?,
                disposition: cancel_disposition_name(disposition),
            })
        })
        .collect::<HarnessResult<Vec<_>>>()?;
    let observations = capture_observations(recording)?;
    let capture = LogicalCapture {
        schema: CAPTURE_SCHEMA,
        workload: CaptureWorkload {
            fixture_schema: "runnel.actor-stress-vectors/2",
            specification: "runnel-m5-actor-stress-v1",
            fixture_id: FIXTURE_ID,
            fixture_file_sha256: FIXTURE_FILE_DIGEST,
        },
        action_results,
        cleanup_cancellations,
        observations,
        shutdown: CaptureShutdown {
            accepted_submissions: shutdown.accepted_submissions,
            rejected_submissions: shutdown.rejected_submissions,
            shutdown_cancellations: shutdown.shutdown_cancellations,
            terminated_requests: shutdown.terminated_requests,
            discarded_output_events: shutdown.discarded_output_events,
            released_request_bytes: shutdown.released_request_bytes,
            remaining_shared_bytes: shutdown.remaining_shared_bytes,
            final_request_bytes: shutdown.final_request_bytes,
            final_shared_bytes: shutdown.final_shared_bytes,
        },
        diagnostics: CaptureDiagnostics {
            initial_pump_entries: initial_probe.pump_entries,
            final_pump_entries: final_probe.pump_entries,
            pump_entries_delta,
            engine_steps,
            observer_count: final_recorder.observation_count(),
            observer_limit: final_recorder.observation_limit(),
            observer_initial_capacity: initial_recorder.allocated_capacity(),
            observer_final_capacity: final_recorder.allocated_capacity(),
            observer_overflowed: final_recorder.overflowed(),
            observer_poisoned: final_recorder.poisoned(),
        },
    };
    let bytes = super::canonical_json_ascii(&capture);
    require(
        bytes.len() <= MAX_CAPTURE_BYTES,
        "logical actor capture exceeds its one-MiB bound",
    )?;
    Ok(bytes)
}

fn capture_action(record: &ActionRecord) -> HarnessResult<CaptureAction> {
    Ok(CaptureAction {
        ordinal: record.ordinal,
        kind: match record.kind {
            0 => "submit",
            1 => "cancel",
            2 => "drop",
            3 => "drain",
            4 => "wake",
            _ => return Err("unregistered actor action kind".to_owned()),
        },
        producer: record.producer,
        submit_attempt: (record.submit_attempt != u32::MAX).then_some(record.submit_attempt),
        client_index: (record.client_index != u32::MAX).then_some(record.client_index),
        result: result_name(record.result),
        error: error_name(record.error)?,
        request_id: (record.request_id != 0).then_some(record.request_id),
        output: (record.result == ResultCode::DrainOutput).then_some(CaptureOutput {
            output_index: record.value0,
            token_id: record.value1,
        }),
    })
}

fn capture_observations(
    recording: &ActorStressRecording,
) -> HarnessResult<Vec<CaptureObservation>> {
    let mut observations = Vec::new();
    observations
        .try_reserve_exact(recording.observations().len())
        .map_err(|_| "logical actor observation capture allocation failed".to_owned())?;
    for observation in recording.observations() {
        let request_id = observation.request_id().get();
        observations.push(match observation.kind() {
            ActorSemanticObservationKind::Output => {
                let event = observation
                    .output_event()
                    .ok_or_else(|| "output observation omitted its event".to_owned())?;
                CaptureObservation::Output {
                    request_id,
                    output_index: to_u32(event.output_index(), "output index")?,
                    token_id: event.token(),
                }
            }
            ActorSemanticObservationKind::Terminal => {
                let terminal = observation
                    .terminal_result()
                    .ok_or_else(|| "terminal observation omitted its result".to_owned())?;
                let (outcome, error) = capture_terminal_outcome(terminal.outcome())?;
                CaptureObservation::Terminal {
                    request_id,
                    outcome,
                    error,
                    committed_positions: to_u32(
                        terminal.committed_positions(),
                        "committed positions",
                    )?,
                    emitted_tokens: to_u32(terminal.emitted_tokens(), "emitted tokens")?,
                }
            }
            ActorSemanticObservationKind::OutputEof => CaptureObservation::Eof { request_id },
        });
    }
    Ok(observations)
}

const fn result_name(result: ResultCode) -> &'static str {
    match result {
        ResultCode::SubmitAccepted => "submit_accepted",
        ResultCode::SubmitOfferExhausted => "submit_offer_exhausted",
        ResultCode::CancelRequested => "cancel_requested",
        ResultCode::CancelAlreadyRequested => "cancel_already_requested",
        ResultCode::CancelAlreadyTerminal => "cancel_already_terminal",
        ResultCode::ReceiverDropped => "receiver_dropped",
        ResultCode::DrainOutput => "drain_output",
        ResultCode::DrainEmpty => "drain_empty",
        ResultCode::DrainEof => "drain_eof",
        ResultCode::WakeSignaled => "wake_signaled",
        ResultCode::TargetUnavailable => "target_unavailable",
        ResultCode::Error => "error",
    }
}

fn capture_terminal_outcome(
    outcome: TerminalOutcome,
) -> HarnessResult<(&'static str, Option<&'static str>)> {
    match outcome {
        TerminalOutcome::Completed => Ok(("completed", None)),
        TerminalOutcome::Cancelled => Ok(("cancelled", None)),
        TerminalOutcome::DeadlineExceeded => Ok(("deadline_exceeded", None)),
        TerminalOutcome::Failed { category } => {
            let error = error_name(error_code(category)?)?
                .ok_or_else(|| "failed terminal omitted its error category".to_owned())?;
            Ok(("failed", Some(error)))
        }
    }
}

fn error_name(error: u8) -> HarnessResult<Option<&'static str>> {
    match error {
        0 => Ok(None),
        1 => Ok(Some("invalid_request")),
        2 => Ok(Some("unsupported")),
        3 => Ok(Some("resource_exhausted")),
        4 => Ok(Some("cancelled")),
        5 => Ok(Some("deadline_exceeded")),
        6 => Ok(Some("internal")),
        _ => Err("unregistered actor semantic error code".to_owned()),
    }
}

fn serialize_transcript(
    actions: &[ActionRecord],
    semantic: &SemanticRecords,
    shutdown: ShutdownRecord,
) -> HarnessResult<Vec<u8>> {
    require(
        actions.len() == ACTION_COUNT,
        "transcript action count changed",
    )?;
    let expected_len = TRANSCRIPT_DOMAIN
        .len()
        .checked_add(4 * size_of::<u32>())
        .and_then(|length| length.checked_add(actions.len().checked_mul(33)?))
        .and_then(|length| length.checked_add(semantic.outputs.len().checked_mul(21)?))
        .and_then(|length| length.checked_add(semantic.terminals.len().checked_mul(25)?))
        .and_then(|length| length.checked_add(semantic.eofs.len().checked_mul(13)?))
        .and_then(|length| length.checked_add(57))
        .ok_or_else(|| "semantic transcript length overflows".to_owned())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected_len)
        .map_err(|_| "semantic transcript allocation failed".to_owned())?;
    let allocated_capacity = bytes.capacity();
    bytes.extend_from_slice(TRANSCRIPT_DOMAIN);
    put_u32(&mut bytes, to_u32(actions.len(), "action count")?);
    put_u32(&mut bytes, to_u32(semantic.outputs.len(), "output count")?);
    put_u32(
        &mut bytes,
        to_u32(semantic.terminals.len(), "terminal count")?,
    );
    put_u32(&mut bytes, to_u32(semantic.eofs.len(), "EOF count")?);
    for action in actions {
        bytes.push(0x01);
        put_u32(&mut bytes, action.ordinal);
        bytes.push(action.kind);
        bytes.push(action.producer);
        bytes.push(action.result as u8);
        bytes.push(action.error);
        put_u32(&mut bytes, action.submit_attempt);
        put_u32(&mut bytes, action.client_index);
        put_u64(&mut bytes, action.request_id);
        put_u32(&mut bytes, action.value0);
        put_u32(&mut bytes, action.value1);
    }
    for output in &semantic.outputs {
        bytes.push(0x02);
        put_u32(&mut bytes, output.client_index);
        put_u64(&mut bytes, output.request_id);
        put_u32(&mut bytes, output.output_index);
        put_u32(&mut bytes, output.token_id);
    }
    for terminal in &semantic.terminals {
        bytes.push(0x03);
        put_u32(&mut bytes, terminal.client_index);
        put_u64(&mut bytes, terminal.request_id);
        bytes.push(terminal.outcome);
        bytes.push(terminal.error);
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        put_u32(&mut bytes, terminal.committed_positions);
        put_u32(&mut bytes, terminal.emitted_tokens);
    }
    for eof in &semantic.eofs {
        bytes.push(0x04);
        put_u32(&mut bytes, eof.client_index);
        put_u64(&mut bytes, eof.request_id);
    }
    bytes.push(0x05);
    put_u32(&mut bytes, shutdown.accepted_submissions);
    put_u32(&mut bytes, shutdown.rejected_submissions);
    put_u32(&mut bytes, shutdown.shutdown_cancellations);
    put_u32(&mut bytes, shutdown.terminated_requests);
    put_u32(&mut bytes, shutdown.discarded_output_events);
    put_u32(&mut bytes, 0);
    put_u64(&mut bytes, shutdown.released_request_bytes);
    put_u64(&mut bytes, shutdown.remaining_shared_bytes);
    put_u64(&mut bytes, shutdown.final_request_bytes);
    put_u64(&mut bytes, shutdown.final_shared_bytes);
    require(
        bytes.len() == expected_len,
        "semantic transcript length differs from its fixed-width plan",
    )?;
    require(
        bytes.capacity() == allocated_capacity,
        "semantic transcript allocation grew during serialization",
    )?;
    Ok(bytes)
}

fn terminal_outcome(outcome: TerminalOutcome) -> HarnessResult<(u8, u8)> {
    match outcome {
        TerminalOutcome::Completed => Ok((0, 0)),
        TerminalOutcome::Cancelled => Ok((1, 0)),
        TerminalOutcome::DeadlineExceeded => {
            Err("deadline-free golden request reached deadline-exceeded terminal state".to_owned())
        }
        TerminalOutcome::Failed { category } => Err(format!(
            "fault-free golden request failed with category {}",
            category.as_str()
        )),
    }
}

const fn cancel_disposition_name(disposition: CleanupCancellation) -> &'static str {
    match disposition {
        CleanupCancellation::Requested => "requested",
        CleanupCancellation::AlreadyRequested => "already_requested",
        CleanupCancellation::AlreadyTerminal => "already_terminal",
        CleanupCancellation::InvalidRequest => "invalid_request",
    }
}

fn error_code(category: ErrorCategory) -> HarnessResult<u8> {
    match category {
        ErrorCategory::InvalidRequest => Ok(1),
        ErrorCategory::Unsupported => Ok(2),
        ErrorCategory::ResourceExhausted => Ok(3),
        ErrorCategory::Cancelled => Ok(4),
        ErrorCategory::DeadlineExceeded => Ok(5),
        ErrorCategory::Internal => Ok(6),
        _ => Err("unregistered scheduler error category".to_owned()),
    }
}

fn action_ordinal(action: &Action) -> u32 {
    match action {
        Action::Submit { ordinal, .. }
        | Action::Cancel { ordinal, .. }
        | Action::Drop { ordinal, .. }
        | Action::Drain { ordinal, .. }
        | Action::Wake { ordinal, .. } => *ordinal,
    }
}

fn action_producer(action: &Action) -> u32 {
    match action {
        Action::Submit { producer, .. }
        | Action::Cancel { producer, .. }
        | Action::Drop { producer, .. }
        | Action::Drain { producer, .. }
        | Action::Wake { producer, .. } => *producer,
    }
}

fn validate_producer_assignment(action: &Action) -> HarnessResult<()> {
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

fn action_client_index(action: &Action) -> Option<usize> {
    let index = match action {
        Action::Submit { request_index, .. } => request_index.0?,
        Action::Cancel { request_index, .. }
        | Action::Drop { request_index, .. }
        | Action::Drain { request_index, .. } => *request_index,
        Action::Wake { .. } => return None,
    };
    usize::try_from(index).ok()
}

fn require(condition: bool, message: impl Into<String>) -> HarnessResult<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn require_unique<T: Ord>(records: &[T], label: &str) -> HarnessResult<()> {
    require(
        records.windows(2).all(|pair| pair[0] != pair[1]),
        format!("duplicate {label}"),
    )
}

fn to_u8(value: u32, field: &str) -> HarnessResult<u8> {
    u8::try_from(value).map_err(|_| format!("{field} does not fit u8"))
}

fn to_u32(value: usize, field: &str) -> HarnessResult<u32> {
    u32::try_from(value).map_err(|_| format!("{field} does not fit u32"))
}

fn to_u64(value: usize, field: &str) -> HarnessResult<u64> {
    u64::try_from(value).map_err(|_| format!("{field} does not fit u64"))
}

fn to_usize(value: u32, field: &str) -> HarnessResult<usize> {
    usize::try_from(value).map_err(|_| format!("{field} does not fit usize"))
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn digest_label(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("sha256:{digest:x}")
}

fn read_bounded_regular_file(
    path: &std::ffi::OsStr,
    maximum_bytes: usize,
    label: &str,
) -> HarnessResult<Vec<u8>> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let mut file = options
        .open(path)
        .map_err(|error| format!("cannot open {label} as a no-follow file: {error}"))?;
    let before = file
        .metadata()
        .map_err(|error| format!("cannot inspect {label}: {error}"))?;
    require(
        before.file_type().is_file(),
        format!("{label} is not a regular file"),
    )?;
    let declared_len = usize::try_from(before.len())
        .map_err(|_| format!("{label} length does not fit this host"))?;
    require(
        declared_len <= maximum_bytes,
        format!("{label} exceeds its frozen bound"),
    )?;
    let capacity = declared_len
        .checked_add(1)
        .ok_or_else(|| format!("{label} read capacity overflowed"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| format!("cannot allocate the bounded {label} read"))?;
    let read_limit = u64::try_from(maximum_bytes)
        .map_err(|_| format!("{label} bound does not fit u64"))?
        .checked_add(1)
        .ok_or_else(|| format!("{label} read bound overflowed"))?;
    (&mut file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {label}: {error}"))?;
    require(
        bytes.len() <= maximum_bytes,
        format!("{label} exceeds its frozen bound"),
    )?;
    let after = file
        .metadata()
        .map_err(|error| format!("cannot re-inspect {label}: {error}"))?;
    require(
        before.len() == after.len()
            && before.modified().ok() == after.modified().ok()
            && bytes.len() == declared_len,
        format!("{label} changed while it was read"),
    )?;
    Ok(bytes)
}

fn exchange_independent_capture(run: &GoldenRun) -> HarnessResult<()> {
    if let Some(path) = std::env::var_os("RUNNEL_ACTOR_GOLDEN_CAPTURE") {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(path)
            .map_err(|_| "cannot create the requested logical actor capture".to_owned())?;
        file.write_all(&run.capture)
            .map_err(|_| "cannot write the requested logical actor capture".to_owned())?;
        file.sync_all()
            .map_err(|_| "cannot sync the requested logical actor capture".to_owned())?;
    }
    if let Some(path) = std::env::var_os("RUNNEL_ACTOR_PYTHON_TRANSCRIPT") {
        let transcript = read_bounded_regular_file(
            &path,
            MAX_TRANSCRIPT_BYTES,
            "independent Python transcript",
        )?;
        require(
            transcript == run.transcript,
            "independent Python transcript differs byte-for-byte from Rust",
        )?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deterministic_actor_semantic_golden_is_bounded_and_structurally_sound() {
    let run = timeout(Duration::from_secs(30), run_golden())
        .await
        .expect("deterministic actor golden exceeded 30 seconds")
        .unwrap_or_else(|error| panic!("deterministic actor golden failed: {error}"));
    assert_eq!(
        run.digest, EXPECTED_TRANSCRIPT_DIGEST,
        "actor semantic transcript changed from the accepted v2 golden"
    );
    exchange_independent_capture(&run)
        .unwrap_or_else(|error| panic!("actor golden capture exchange failed: {error}"));
    assert!(!run.transcript.is_empty());
    assert!(run.accepted > 0);
    assert_eq!(run.terminals, run.accepted);
    assert_eq!(run.eofs, run.accepted);
    assert!(run.outputs <= 547);
    assert!(run.pump_entries <= PUMP_ENTRY_LIMIT);
    if std::env::var_os("RUNNEL_PRINT_ACTOR_GOLDEN").is_some() {
        eprintln!(
            "actor semantic golden: digest={}, bytes={}, accepted={}, rejected={}, outputs={}, terminals={}, eofs={}, engine_steps={}, pump_entries={}",
            run.digest,
            run.transcript.len(),
            run.accepted,
            run.rejected,
            run.outputs,
            run.terminals,
            run.eofs,
            run.engine_steps,
            run.pump_entries,
        );
    }
}
