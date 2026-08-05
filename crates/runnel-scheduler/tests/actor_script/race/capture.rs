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
    OBSERVATION_LIMIT, PUMP_ENTRY_LIMIT, SemanticRecords, SemanticTerminalOutcome,
    collect_semantic_records, require, to_u64, validate_shutdown,
};

const REPETITION_COUNT: u64 = 32;

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
    history: CompleteRaceHistory,
    cleanup: RaceCleanupCapture,
    accepted_ids: [u64; REQUEST_COUNT],
    initial_probe: ProbeSnapshotCapture,
    initial_recorder: ActorStressRecorderStatus,
}

impl RepetitionDraft {
    pub(super) fn new(
        repetition: u64,
        descriptors: Arc<[Descriptor]>,
        history: CompleteRaceHistory,
        cleanup: RaceCleanupCapture,
        accepted_ids: [u64; REQUEST_COUNT],
        initial_probe: ProbeSnapshotCapture,
        initial_recorder: ActorStressRecorderStatus,
    ) -> HarnessResult<Self> {
        require(
            repetition < REPETITION_COUNT,
            "race repetition index is out of range",
        )?;
        require(
            descriptors.len() == REQUEST_COUNT,
            "race repetition descriptor table has the wrong length",
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
        validate_capture_conservation(&self.history, &parts, &records, &self.accepted_ids)?;
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
) -> HarnessResult<()> {
    let mut accepted_from_actions = [0_u64; REQUEST_COUNT];
    let mut accepted_witnesses = [None; REQUEST_COUNT];
    let mut dropped = [false; REQUEST_COUNT];
    let mut cancellation_forced = [false; REQUEST_COUNT];
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
            if matches!(
                action.13.6,
                Some(
                    ControlDispositionCapture::Requested
                        | ControlDispositionCapture::AlreadyRequested
                )
            ) {
                cancellation_forced[client] = true;
            }
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
        if matches!(
            authority.5.6,
            Some(
                ControlDispositionCapture::Requested | ControlDispositionCapture::AlreadyRequested
            )
        ) {
            cancellation_forced[client] = true;
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
    for client in 0..REQUEST_COUNT {
        require(
            semantic_outputs[client].starts_with(&drained[client]),
            "script-drained outputs differ from semantic publications",
        )?;
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
        if cancellation_forced[client] {
            require(
                terminal.outcome == SemanticTerminalOutcome::Cancelled,
                "witnessed cancellation did not reach a cancelled terminal",
            )?;
        }
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
}
