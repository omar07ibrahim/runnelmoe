//! Closed-schema evidence for one genuine actor-race repetition.
//!
//! This module owns the seam between concurrent structural history, ordered
//! cleanup, endpoint semantic observations, and cooperative shutdown.  It
//! constructs only the exact `Repetition` object frozen by ADR 0007; compact
//! top-level framing and durable publication belong to the 32-run layer.

use std::sync::Arc;

use runnel_scheduler::{
    ActorProbeSnapshot, ActorShutdownReport, ActorStressRecorderStatus, ActorStressRecording,
};
use serde::Serialize;

use super::cleanup::{
    CleanupAuthorityCapture, CleanupReceiverCapture, ProbeSnapshotCapture, RaceCleanupCapture,
    RaceCleanupParts, TerminalOutcomeCapture,
};
use super::*;
use crate::Descriptor;
use crate::common::{
    EXPECTED_ARTIFACT_ID, EXPECTED_OBJECT_DIGEST, EXPECTED_PAGE_TABLE_DIGEST, EXPECTED_SPEC_DIGEST,
    OBSERVATION_LIMIT, PUMP_ENTRY_LIMIT, SemanticRecords, SemanticTerminalOutcome,
    collect_semantic_records, require, to_u64, validate_shutdown,
};

pub(super) const REPETITION_COUNT: usize = 32;
const REPETITION_COUNT_U64: u64 = 32;
const SCHEMA: &str = "runnel.actor-race-history/1";
const SPECIFICATION: &str = "runnel-m5-actor-stress-v1";
const VECTOR_SCHEMA: &str = "runnel.actor-stress-vectors/2";

/// Exact lexicographically keyed top-level capture from ADR 0007.
#[derive(Serialize)]
pub(super) struct ActorRaceCapture {
    repetition_count: u64,
    repetitions: Vec<RepetitionCapture>,
    schema: &'static str,
    workload: WorkloadCapture,
}

impl ActorRaceCapture {
    pub(super) fn new(repetitions: Vec<RepetitionCapture>) -> HarnessResult<Self> {
        require(
            repetitions.len() == REPETITION_COUNT,
            "actor race capture has the wrong repetition count",
        )?;
        for (expected, repetition) in repetitions.iter().enumerate() {
            require(
                repetition.repetition == to_u64(expected, "actor race repetition index")?,
                "actor race repetitions are not indexed consecutively",
            )?;
        }
        Ok(Self {
            repetition_count: REPETITION_COUNT_U64,
            repetitions,
            schema: SCHEMA,
            workload: WorkloadCapture::exact(),
        })
    }
}

/// Exact lexicographically keyed authenticated workload identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct WorkloadCapture {
    artifact_id: &'static str,
    artifact_object_sha256: &'static str,
    artifact_page_table_sha256: &'static str,
    model_spec_sha256: &'static str,
    specification: &'static str,
    vector_file_sha256: &'static str,
    vector_id: &'static str,
    vector_schema: &'static str,
}

impl WorkloadCapture {
    const fn exact() -> Self {
        Self {
            artifact_id: EXPECTED_ARTIFACT_ID,
            artifact_object_sha256: EXPECTED_OBJECT_DIGEST,
            artifact_page_table_sha256: EXPECTED_PAGE_TABLE_DIGEST,
            model_spec_sha256: EXPECTED_SPEC_DIGEST,
            specification: SPECIFICATION,
            vector_file_sha256: crate::FIXTURE_FILE_DIGEST,
            vector_id: crate::FIXTURE_ID,
            vector_schema: VECTOR_SCHEMA,
        }
    }
}

/// Exact seven-position `Observation` tuple from ADR 0007.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct ObservationCapture(
    ObservationKind,
    u64,
    Option<u64>,
    Option<u64>,
    Option<TerminalOutcomeCapture>,
    Option<u64>,
    Option<u64>,
);

impl ObservationCapture {
    const fn output(request_id: u64, output_index: u64, token_id: u64) -> Self {
        Self(
            ObservationKind::Output,
            request_id,
            Some(output_index),
            Some(token_id),
            None,
            None,
            None,
        )
    }

    const fn terminal(
        request_id: u64,
        outcome: TerminalOutcomeCapture,
        committed_positions: u64,
        emitted_tokens: u64,
    ) -> Self {
        Self(
            ObservationKind::Terminal,
            request_id,
            None,
            None,
            Some(outcome),
            Some(committed_positions),
            Some(emitted_tokens),
        )
    }

    const fn output_eof(request_id: u64) -> Self {
        Self(
            ObservationKind::OutputEof,
            request_id,
            None,
            None,
            None,
            None,
            None,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ObservationKind {
    Output,
    Terminal,
    OutputEof,
}

/// Exact lexicographically keyed `RecorderStatus` object from ADR 0007.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct RecorderStatusCapture {
    allocated_capacity: u64,
    observation_count: u64,
    observation_limit: u64,
    overflowed: bool,
    poisoned: bool,
}

impl RecorderStatusCapture {
    fn normalize(status: ActorStressRecorderStatus) -> HarnessResult<Self> {
        Ok(Self {
            allocated_capacity: to_u64(
                status.allocated_capacity(),
                "race recorder allocated capacity",
            )?,
            observation_count: to_u64(
                status.observation_count(),
                "race recorder observation count",
            )?,
            observation_limit: to_u64(
                status.observation_limit(),
                "race recorder observation limit",
            )?,
            overflowed: status.overflowed(),
            poisoned: status.poisoned(),
        })
    }

    const fn healthy(self) -> bool {
        !self.overflowed && !self.poisoned
    }
}

/// Exact lexicographically keyed `Diagnostics` object from ADR 0007.
#[derive(Debug, Serialize)]
struct DiagnosticsCapture {
    action_counter_final: u64,
    barrier_released: bool,
    cleanup_counter_final: u64,
    engine_steps_delta: u64,
    final_engine_steps: u64,
    final_pump_entries: u64,
    initial_engine_steps: u64,
    initial_pump_entries: u64,
    overlap_pair: RaceOverlapPair,
    pump_entries_delta: u64,
    recorder_final: RecorderStatusCapture,
    recorder_initial: RecorderStatusCapture,
}

/// Exact lexicographically keyed `Shutdown` object from ADR 0007.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct ShutdownCapture {
    accepted_submissions: u64,
    discarded_output_events: u64,
    engine_steps: u64,
    rejected_submissions: u64,
    released_request_bytes: u64,
    remaining_shared_bytes: u64,
    shutdown_cancellations: u64,
    terminated_requests: u64,
}

impl ShutdownCapture {
    fn normalize(report: ActorShutdownReport) -> HarnessResult<Self> {
        let engine = report.engine();
        Ok(Self {
            accepted_submissions: to_u64(
                report.accepted_submissions(),
                "race shutdown accepted submissions",
            )?,
            discarded_output_events: to_u64(
                engine.discarded_output_events,
                "race shutdown discarded output events",
            )?,
            engine_steps: report.engine_steps(),
            rejected_submissions: to_u64(
                report.rejected_submissions(),
                "race shutdown rejected submissions",
            )?,
            released_request_bytes: to_u64(
                engine.released_request_bytes,
                "race shutdown released request bytes",
            )?,
            remaining_shared_bytes: to_u64(
                engine.remaining_shared_bytes,
                "race shutdown remaining shared bytes",
            )?,
            shutdown_cancellations: to_u64(
                report.shutdown_cancellations(),
                "race shutdown cancellations",
            )?,
            terminated_requests: to_u64(
                engine.terminated_requests,
                "race shutdown terminated requests",
            )?,
        })
    }
}

/// Fully owned exact `Repetition` object from ADR 0007.
#[derive(Serialize)]
pub(super) struct RepetitionCapture {
    actions: CompleteRaceHistory,
    cleanup_authorities: Vec<CleanupAuthorityCapture>,
    cleanup_receivers: Vec<CleanupReceiverCapture>,
    diagnostics: DiagnosticsCapture,
    observations: Vec<ObservationCapture>,
    post_shutdown: ProbeSnapshotCapture,
    pre_cleanup: ProbeSnapshotCapture,
    pre_shutdown: ProbeSnapshotCapture,
    repetition: u64,
    shutdown: ShutdownCapture,
}

/// State that can exist only after both producers and ordered cleanup finish.
pub(super) struct RepetitionDraft {
    repetition: u64,
    descriptors: Arc<[Descriptor]>,
    output_capacity_per_request: usize,
    history: CompleteRaceHistory,
    cleanup: RaceCleanupCapture,
    accepted_ids: [u64; REQUEST_COUNT],
    initial_probe: ProbeSnapshotCapture,
    initial_recorder: ActorStressRecorderStatus,
}

impl RepetitionDraft {
    #[allow(
        clippy::too_many_arguments,
        reason = "each argument is a separately authenticated race-capture boundary"
    )]
    pub(super) fn new(
        repetition: u64,
        descriptors: Arc<[Descriptor]>,
        output_capacity_per_request: usize,
        history: CompleteRaceHistory,
        cleanup: RaceCleanupCapture,
        accepted_ids: [u64; REQUEST_COUNT],
        initial_probe: ProbeSnapshotCapture,
        initial_recorder: ActorStressRecorderStatus,
    ) -> HarnessResult<Self> {
        require(
            repetition < REPETITION_COUNT_U64,
            "race repetition index is out of range",
        )?;
        require(
            descriptors.len() == REQUEST_COUNT,
            "race repetition descriptor table has the wrong length",
        )?;
        require(
            output_capacity_per_request != 0,
            "race repetition output capacity is zero",
        )?;
        initial_probe.validate_quiescent("initial race")?;
        require(
            initial_probe.command_in_flight == 0
                && initial_probe.command_ready == 0
                && initial_probe.command_reserved == 0
                && initial_probe.command_responded == 0
                && initial_probe.outstanding_requests == 0
                && initial_probe.request_bytes == 0,
            "initial race snapshot retained operation-owned state",
        )?;
        cleanup
            .pre_cleanup()
            .validate_not_before(initial_probe, "pre-cleanup")?;
        require(
            cleanup.pre_cleanup().shared_bytes == initial_probe.shared_bytes
                && cleanup.pre_shutdown().shared_bytes == initial_probe.shared_bytes,
            "race shared ledger changed before shutdown",
        )?;
        Ok(Self {
            repetition,
            descriptors,
            output_capacity_per_request,
            history,
            cleanup,
            accepted_ids,
            initial_probe,
            initial_recorder,
        })
    }

    pub(super) fn finalize(
        self,
        recording: ActorStressRecording,
        report: ActorShutdownReport,
        post_shutdown_raw: ActorProbeSnapshot,
    ) -> HarnessResult<RepetitionCapture> {
        let accepted = self
            .history
            .actions()
            .filter(|action| action.7 == RaceResult::SubmitAccepted)
            .count();
        let rejected = self
            .history
            .actions()
            .filter(|action| action.2 == RaceKind::Submit && action.7 == RaceResult::Error)
            .count();
        require(
            accepted
                .checked_add(rejected)
                .is_some_and(|count| count == REQUEST_COUNT),
            "race submissions do not conserve the request corpus",
        )?;

        let post_shutdown = ProbeSnapshotCapture::normalize_post_shutdown(post_shutdown_raw)?;
        post_shutdown.validate_not_before(self.cleanup.pre_shutdown(), "post-shutdown")?;
        validate_shutdown(
            report,
            accepted,
            rejected,
            post_shutdown.request_bytes,
            post_shutdown.shared_bytes,
        )?;
        let shutdown = ShutdownCapture::normalize(report)?;
        require(
            shutdown.engine_steps == post_shutdown.engine_steps,
            "shutdown engine steps differ from the post-shutdown probe",
        )?;

        let records =
            collect_semantic_records(&recording, &self.accepted_ids, self.descriptors.as_ref())?;
        let recorder_initial = RecorderStatusCapture::normalize(self.initial_recorder)?;
        let recorder_final = RecorderStatusCapture::normalize(recording.status())?;
        validate_recorder_statuses(recorder_initial, recorder_final, &records, accepted)?;

        let parts = self.cleanup.into_parts();
        validate_capture_conservation(
            &self.history,
            &parts,
            &records,
            &self.accepted_ids,
            self.output_capacity_per_request,
        )?;
        let observations = normalize_observations(records)?;
        require(
            recorder_final.observation_count
                == to_u64(observations.len(), "normalized race observation count")?,
            "race recorder count differs from normalized observations",
        )?;

        let engine_steps_delta = post_shutdown
            .engine_steps
            .checked_sub(self.initial_probe.engine_steps)
            .ok_or_else(|| "race engine-step counter regressed".to_owned())?;
        let pump_entries_delta = post_shutdown
            .pump_entries
            .checked_sub(self.initial_probe.pump_entries)
            .ok_or_else(|| "race pump-entry counter regressed".to_owned())?;
        require(
            pump_entries_delta <= PUMP_ENTRY_LIMIT,
            "race actor exceeded its pump-entry limit",
        )?;
        let cleanup_counter_final = parts.cleanup_counter_final;
        let diagnostics = DiagnosticsCapture {
            action_counter_final: ACTION_COUNTER_FINAL,
            barrier_released: true,
            cleanup_counter_final,
            engine_steps_delta,
            final_engine_steps: post_shutdown.engine_steps,
            final_pump_entries: post_shutdown.pump_entries,
            initial_engine_steps: self.initial_probe.engine_steps,
            initial_pump_entries: self.initial_probe.pump_entries,
            overlap_pair: self.history.overlap_pair(),
            pump_entries_delta,
            recorder_final,
            recorder_initial,
        };

        Ok(RepetitionCapture {
            actions: self.history,
            cleanup_authorities: parts.cleanup_authorities,
            cleanup_receivers: parts.cleanup_receivers,
            diagnostics,
            observations,
            post_shutdown,
            pre_cleanup: parts.pre_cleanup,
            pre_shutdown: parts.pre_shutdown,
            repetition: self.repetition,
            shutdown,
        })
    }
}

fn validate_recorder_statuses(
    initial: RecorderStatusCapture,
    final_status: RecorderStatusCapture,
    records: &SemanticRecords,
    accepted: usize,
) -> HarnessResult<()> {
    let observation_limit = to_u64(OBSERVATION_LIMIT, "race observation limit")?;
    require(
        initial.healthy() && final_status.healthy(),
        "race semantic recorder became unhealthy",
    )?;
    require(
        initial.observation_limit == observation_limit
            && final_status.observation_limit == observation_limit,
        "race recorder logical limit changed",
    )?;
    require(
        initial.allocated_capacity == final_status.allocated_capacity
            && initial.allocated_capacity >= observation_limit,
        "race recorder allocation changed or is undersized",
    )?;
    require(
        initial.observation_count == 0,
        "fresh race recorder was not empty",
    )?;
    let expected = records
        .outputs
        .len()
        .checked_add(
            accepted
                .checked_mul(2)
                .ok_or_else(|| "race terminal/EOF observation count overflowed".to_owned())?,
        )
        .ok_or_else(|| "race observation count overflowed".to_owned())?;
    require(
        records.terminals.len() == accepted && records.eofs.len() == accepted,
        "race terminal/EOF observation count differs from acceptance",
    )?;
    require(
        final_status.observation_count == to_u64(expected, "race expected observation count")?,
        "race final recorder count differs from semantic conservation",
    )
}

fn normalize_observations(records: SemanticRecords) -> HarnessResult<Vec<ObservationCapture>> {
    let total = records
        .outputs
        .len()
        .checked_add(records.terminals.len())
        .and_then(|count| count.checked_add(records.eofs.len()))
        .ok_or_else(|| "normalized race observation count overflowed".to_owned())?;
    let mut observations = Vec::new();
    observations
        .try_reserve_exact(total)
        .map_err(|_| "race observation capture allocation failed".to_owned())?;
    for output in records.outputs {
        observations.push(ObservationCapture::output(
            output.request_id,
            u64::from(output.output_index),
            u64::from(output.token_id),
        ));
    }
    for terminal in records.terminals {
        require(
            terminal.error == 0,
            "race terminal retained an unsupported error category",
        )?;
        let outcome = match terminal.outcome {
            SemanticTerminalOutcome::Completed => TerminalOutcomeCapture::Completed,
            SemanticTerminalOutcome::Cancelled => TerminalOutcomeCapture::Cancelled,
        };
        observations.push(ObservationCapture::terminal(
            terminal.request_id,
            outcome,
            u64::from(terminal.committed_positions),
            u64::from(terminal.emitted_tokens),
        ));
    }
    for eof in records.eofs {
        observations.push(ObservationCapture::output_eof(eof.request_id));
    }
    require(
        observations.len() == total,
        "normalized race observation count changed",
    )?;
    Ok(observations)
}

fn validate_capture_conservation(
    history: &CompleteRaceHistory,
    parts: &RaceCleanupParts,
    records: &SemanticRecords,
    accepted_ids: &[u64; REQUEST_COUNT],
    output_capacity_per_request: usize,
) -> HarnessResult<()> {
    let mut accepted_from_actions = [0_u64; REQUEST_COUNT];
    let mut accepted_witnesses = [None; REQUEST_COUNT];
    let mut dropped = [false; REQUEST_COUNT];
    let mut cancellation_publisher_count = [0_u8; REQUEST_COUNT];
    let mut cleanup_cancellation_forced = [false; REQUEST_COUNT];
    let mut drained: [Vec<RaceOutput>; REQUEST_COUNT] = std::array::from_fn(|_| Vec::new());

    for action in history.actions() {
        if action.7 == RaceResult::SubmitAccepted {
            let client = action
                .4
                .ok_or_else(|| "accepted race action omitted its client".to_owned())?;
            let client = usize::try_from(client)
                .map_err(|_| "accepted race client does not fit usize".to_owned())?;
            require(
                client < REQUEST_COUNT && accepted_from_actions[client] == 0,
                "accepted race client is out of range or duplicated",
            )?;
            accepted_from_actions[client] = action
                .9
                .ok_or_else(|| "accepted race action omitted its request ID".to_owned())?;
            accepted_witnesses[client] = Some(action.12);
        }
    }
    require(
        accepted_from_actions == *accepted_ids,
        "accepted registry differs from complete race actions",
    )?;

    for action in history.actions() {
        let client = action
            .4
            .map(|index| {
                usize::try_from(index)
                    .map_err(|_| "race action client index does not fit usize".to_owned())
            })
            .transpose()?;
        if let (Some(client), Some(request_id)) = (client, action.9) {
            require(
                client < REQUEST_COUNT && accepted_ids[client] == request_id,
                "targeted race action request identity differs from acceptance",
            )?;
        }
        if action.13.boundary_reached() {
            let client = client.ok_or_else(|| {
                "reached race control witness omitted its client index".to_owned()
            })?;
            let accepted = accepted_witnesses
                .get(client)
                .copied()
                .flatten()
                .ok_or_else(|| "reached race control has no accepted witness".to_owned())?;
            require(
                action.13.2 == accepted.2 && action.13.3 == accepted.3,
                "race control witness identity differs from acceptance",
            )?;
            record_preterminal_c_publisher(&mut cancellation_publisher_count[client], action.13)?;
        }
        for endpoint in [action.14, action.15]
            .into_iter()
            .filter(|endpoint| endpoint.boundary_reached())
        {
            let client = client.ok_or_else(|| {
                "reached race endpoint witness omitted its client index".to_owned()
            })?;
            let accepted = accepted_witnesses
                .get(client)
                .copied()
                .flatten()
                .ok_or_else(|| "reached race endpoint has no accepted witness".to_owned())?;
            require(
                endpoint.2 == accepted.4 && endpoint.3 == accepted.5,
                "race endpoint witness identity differs from acceptance",
            )?;
        }
        if let Some(client) = client.filter(|_| action.7 == RaceResult::ReceiverDropped) {
            require(!dropped[client], "race receiver was dropped more than once")?;
            dropped[client] = true;
        }
        if let Some(client) = client.filter(|_| action.7 == RaceResult::DrainOutput) {
            let output = action
                .10
                .ok_or_else(|| "race output drain omitted its output".to_owned())?;
            let expected_index = to_u64(drained[client].len(), "race drained output count")?;
            require(
                output.1 == expected_index,
                "script-drained race outputs are not a FIFO prefix",
            )?;
            drained[client].push(output);
        }
    }

    let accepted_count = accepted_ids
        .iter()
        .filter(|request_id| **request_id != 0)
        .count();
    require(
        parts.cleanup_authorities.len() == accepted_count,
        "cleanup authority set differs from accepted actions",
    )?;
    let mut authority_cursor = parts.cleanup_authorities.iter();
    for (client, &request_id) in accepted_ids.iter().enumerate() {
        if request_id == 0 {
            continue;
        }
        let authority = authority_cursor
            .next()
            .ok_or_else(|| "cleanup authority set ended early".to_owned())?;
        require(
            authority.2 == to_u64(client, "cleanup authority client index")?
                && authority.3 == request_id,
            "cleanup authorities are not the exact ascending accepted set",
        )?;
        let published_c =
            record_preterminal_c_publisher(&mut cancellation_publisher_count[client], authority.5)?;
        if authority.5.6 == Some(ControlDispositionCapture::Requested) {
            // Unlike a producer action, this mutation occurs while the pump
            // is held and therefore must govern the later terminal decision.
            require(
                published_c,
                "pump-held cleanup request did not publish the cancellation bit",
            )?;
            cleanup_cancellation_forced[client] = true;
        }
    }
    require(
        authority_cursor.next().is_none(),
        "cleanup authority set retained an extra record",
    )?;

    let expected_live = accepted_ids
        .iter()
        .enumerate()
        .filter(|(client, request_id)| **request_id != 0 && !dropped[*client])
        .count();
    require(
        parts.cleanup_receivers.len() == expected_live,
        "cleanup receiver set differs from accepted-minus-dropped actions",
    )?;
    let cleanup_records = parts
        .cleanup_authorities
        .len()
        .checked_add(parts.cleanup_receivers.len())
        .ok_or_else(|| "cleanup capture record count overflowed".to_owned())?;
    require(
        to_u64(cleanup_records, "cleanup capture record count")?
            .checked_mul(2)
            .is_some_and(|expected| expected == parts.cleanup_counter_final),
        "cleanup counter differs from its exact record arrays",
    )?;

    let mut semantic_outputs: [Vec<RaceOutput>; REQUEST_COUNT] =
        std::array::from_fn(|_| Vec::new());
    for output in &records.outputs {
        let client = usize::try_from(output.client_index)
            .map_err(|_| "semantic output client does not fit usize".to_owned())?;
        require(
            client < REQUEST_COUNT,
            "semantic output client is out of range",
        )?;
        semantic_outputs[client].push(RaceOutput(
            output.request_id,
            u64::from(output.output_index),
            u64::from(output.token_id),
        ));
    }
    for (client, &request_id) in accepted_ids.iter().enumerate() {
        if request_id != 0 {
            validate_undrained_output_capacity(
                &semantic_outputs[client],
                &drained[client],
                output_capacity_per_request,
                client,
            )?;
        }
    }

    let mut terminals = [None; REQUEST_COUNT];
    for terminal in &records.terminals {
        let client = usize::try_from(terminal.client_index)
            .map_err(|_| "semantic terminal client does not fit usize".to_owned())?;
        require(
            client < REQUEST_COUNT && terminals[client].is_none(),
            "semantic terminal client is out of range or duplicated",
        )?;
        terminals[client] = Some(terminal);
        validate_terminal_cancellation(
            terminal.outcome,
            cancellation_publisher_count[client],
            cleanup_cancellation_forced[client],
        )?;
    }

    let mut receiver_cursor = parts.cleanup_receivers.iter();
    for (client, &request_id) in accepted_ids.iter().enumerate() {
        if request_id == 0 || dropped[client] {
            continue;
        }
        let receiver = receiver_cursor
            .next()
            .ok_or_else(|| "cleanup receiver set ended early".to_owned())?;
        require(
            receiver.2 == to_u64(client, "cleanup receiver client index")?
                && receiver.3 == request_id,
            "cleanup receivers are not the exact ascending live set",
        )?;
        let terminal = terminals[client]
            .ok_or_else(|| "cleanup receiver lacks its semantic terminal".to_owned())?;
        let expected_outcome = match terminal.outcome {
            SemanticTerminalOutcome::Completed => TerminalOutcomeCapture::Completed,
            SemanticTerminalOutcome::Cancelled => TerminalOutcomeCapture::Cancelled,
        };
        require(
            receiver.4.0 == terminal.request_id
                && receiver.4.1 == expected_outcome
                && receiver.4.2 == u64::from(terminal.committed_positions)
                && receiver.4.3 == u64::from(terminal.emitted_tokens),
            "cleanup terminal differs from its semantic observation",
        )?;
        let prefix_len = drained[client].len();
        require(
            semantic_outputs[client]
                .get(prefix_len..)
                .is_some_and(|suffix| suffix == receiver.5.as_slice()),
            "cleanup output suffix does not complete semantic publications",
        )?;
    }
    require(
        receiver_cursor.next().is_none(),
        "cleanup receiver set retained an extra record",
    )
}

fn validate_undrained_output_capacity(
    published: &[RaceOutput],
    drained: &[RaceOutput],
    output_capacity_per_request: usize,
    client: usize,
) -> HarnessResult<()> {
    require(
        published.starts_with(drained),
        "script-drained outputs differ from semantic publications",
    )?;
    let undrained = published
        .len()
        .checked_sub(drained.len())
        .ok_or_else(|| format!("client {client} drained more output than was published"))?;
    require(
        undrained <= output_capacity_per_request,
        format!("client {client} undrained output exceeds the actor endpoint capacity"),
    )
}

fn validate_terminal_cancellation(
    outcome: SemanticTerminalOutcome,
    cancellation_publisher_count: u8,
    cleanup_cancellation_forced: bool,
) -> HarnessResult<()> {
    require(
        cancellation_publisher_count <= 1,
        "request retained duplicate preterminal cancellation publishers",
    )?;
    require(
        outcome != SemanticTerminalOutcome::Cancelled || cancellation_publisher_count == 1,
        "cancelled terminal does not have exactly one preterminal cancellation publisher",
    )?;
    require(
        !cleanup_cancellation_forced || outcome == SemanticTerminalOutcome::Cancelled,
        "pump-held cleanup cancellation did not reach a cancelled terminal",
    )
}

fn record_preterminal_c_publisher(
    publisher_count: &mut u8,
    control: ControlWitnessCapture,
) -> HarnessResult<bool> {
    control.validate()?;
    let loaded = control.4.0;
    let resulting = control.5.0;
    let publishes_c = control.boundary_reached()
        && control.6 == Some(ControlDispositionCapture::Requested)
        && loaded & CANCELLED_FLAG == 0
        && loaded & TERMINAL_FLAG == 0
        && resulting & CANCELLED_FLAG != 0;
    if !publishes_c {
        return Ok(false);
    }
    require(
        *publisher_count == 0,
        "request retained duplicate preterminal cancellation publishers",
    )?;
    *publisher_count = (*publisher_count)
        .checked_add(1)
        .ok_or_else(|| "cancellation publisher count overflowed".to_owned())?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_tuples_have_exact_width_and_sentinels() {
        assert_eq!(
            serde_json::to_string(&ObservationCapture::output(7, 3, 11)).expect("output JSON"),
            r#"["output",7,3,11,null,null,null]"#,
        );
        assert_eq!(
            serde_json::to_string(&ObservationCapture::terminal(
                7,
                TerminalOutcomeCapture::Cancelled,
                9,
                4,
            ))
            .expect("terminal JSON"),
            r#"["terminal",7,null,null,"cancelled",9,4]"#,
        );
        assert_eq!(
            serde_json::to_string(&ObservationCapture::output_eof(7)).expect("EOF JSON"),
            r#"["output_eof",7,null,null,null,null,null]"#,
        );
    }

    #[test]
    fn workload_capture_has_exact_sorted_identity_keys() {
        assert_eq!(
            serde_json::to_string(&WorkloadCapture::exact()).expect("workload JSON"),
            concat!(
                r#"{"artifact_id":"sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c","#,
                r#""artifact_object_sha256":"sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab","#,
                r#""artifact_page_table_sha256":"sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c","#,
                r#""model_spec_sha256":"sha256:ed57d7961e65c76223c169cabebaff9c02d8293da026abb0c0c0a22d38079845","#,
                r#""specification":"runnel-m5-actor-stress-v1","#,
                r#""vector_file_sha256":"sha256:eca1faeee91a41d19d98be7ffdad6fc5cebb9027f3e7a634c01ea1cc394fb574","#,
                r#""vector_id":"sha256:5010492fb74eda207511b26811992ed4779814185b9f184663b37a37747bd051","#,
                r#""vector_schema":"runnel.actor-stress-vectors/2"}"#,
            ),
        );
    }

    #[test]
    fn late_script_cancellation_can_complete_but_cleanup_cancellation_is_forcing() {
        validate_terminal_cancellation(SemanticTerminalOutcome::Completed, 1, false)
            .expect("late scripted cancellation may lose to a frozen completion");
        validate_terminal_cancellation(SemanticTerminalOutcome::Cancelled, 1, false)
            .expect("witnessed scripted cancellation may cancel");
        assert!(
            validate_terminal_cancellation(SemanticTerminalOutcome::Cancelled, 0, false).is_err()
        );
        assert!(
            validate_terminal_cancellation(SemanticTerminalOutcome::Completed, 1, true).is_err()
        );
        validate_terminal_cancellation(SemanticTerminalOutcome::Cancelled, 1, true)
            .expect("pump-held cleanup cancellation must cancel");
    }

    #[test]
    fn only_one_exact_preterminal_c_transition_counts_as_the_publisher() {
        let word = |flags| (1_u64 << 3) | flags;
        let requested = |operation, loaded, resulting| {
            ControlWitnessCapture::normalize_parts(
                operation,
                true,
                0,
                1,
                word(loaded),
                word(resulting),
                ControlOutcome::Disposition(ControlDispositionCapture::Requested),
            )
            .expect("valid requested transition")
        };

        let disconnect_only = requested(
            ControlOperationCapture::Disconnect,
            CANCELLED_FLAG,
            CANCELLED_FLAG | DISCONNECTED_FLAG,
        );
        let mut publisher_count = 0;
        assert!(
            !record_preterminal_c_publisher(&mut publisher_count, disconnect_only)
                .expect("D-only transition")
        );
        assert_eq!(publisher_count, 0);
        assert!(
            validate_terminal_cancellation(
                SemanticTerminalOutcome::Cancelled,
                publisher_count,
                false,
            )
            .is_err()
        );

        let cancel_publisher = requested(ControlOperationCapture::Cancel, 0, CANCELLED_FLAG);
        assert!(
            record_preterminal_c_publisher(&mut publisher_count, cancel_publisher)
                .expect("first C publisher")
        );
        assert_eq!(publisher_count, 1);

        let disconnect_publisher = requested(
            ControlOperationCapture::Disconnect,
            0,
            CANCELLED_FLAG | DISCONNECTED_FLAG,
        );
        assert!(
            record_preterminal_c_publisher(&mut publisher_count, disconnect_publisher).is_err()
        );
        assert!(
            validate_terminal_cancellation(SemanticTerminalOutcome::Cancelled, 2, false).is_err()
        );
    }

    #[test]
    fn undrained_output_suffix_is_bounded_by_runtime_capacity() {
        let published = [
            RaceOutput(7, 0, 11),
            RaceOutput(7, 1, 12),
            RaceOutput(7, 2, 13),
        ];
        let drained = [published[0]];
        validate_undrained_output_capacity(&published, &drained, 2, 0)
            .expect("two retained outputs fit the endpoint");
        assert!(validate_undrained_output_capacity(&published, &drained, 1, 0).is_err());
        assert!(
            validate_undrained_output_capacity(&published, &[RaceOutput(7, 0, 99)], 2, 0,).is_err()
        );
    }
}
