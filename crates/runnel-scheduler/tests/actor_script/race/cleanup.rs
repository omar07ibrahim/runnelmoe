//! Ordered post-producer cleanup for one actor-race repetition.
//!
//! Cleanup deliberately owns a counter domain separate from producer action
//! intervals. The coordinator is single-threaded, but it still records both
//! boundaries so the serialized history proves that every retained authority
//! and receiver was visited exactly once in ascending client order.

use std::sync::atomic::{AtomicU64, Ordering};

use runnel_scheduler::{
    ActorProbe, ActorProbeSnapshot, ActorRequestDropWitnessSink, OutputEvent, RequestHandle,
    TerminalOutcome, TerminalResult,
};
use serde::Serialize;

use super::execution::RaceProducerExit;
use super::*;
use crate::Descriptor;
use crate::common::{HarnessResult, finish_receiver_with_preallocated_witness, require, to_u64};

/// Exact `CleanupAuthority` tuple from ADR 0007.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct CleanupAuthorityCapture(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) Option<RaceError>,
    pub(super) ControlWitnessCapture,
);

/// Exact `CleanupReceiver` tuple from ADR 0007.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(super) struct CleanupReceiverCapture(
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) u64,
    pub(super) TerminalCapture,
    pub(super) Vec<RaceOutput>,
    pub(super) bool,
    pub(super) ProbeSnapshotCapture,
);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TerminalOutcomeCapture {
    Completed,
    Cancelled,
}

/// Exact `Terminal` tuple from ADR 0007.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct TerminalCapture(
    pub(super) u64,
    pub(super) TerminalOutcomeCapture,
    pub(super) u64,
    pub(super) u64,
);

impl TerminalCapture {
    fn normalize(terminal: TerminalResult, expected_request_id: u64) -> HarnessResult<Self> {
        let request_id = terminal.request_id().get();
        require(request_id != 0, "cleanup terminal request ID is zero")?;
        require(
            request_id == expected_request_id,
            "cleanup terminal identity differs from its accepted identity",
        )?;
        let outcome = match terminal.outcome() {
            TerminalOutcome::Completed => TerminalOutcomeCapture::Completed,
            TerminalOutcome::Cancelled => TerminalOutcomeCapture::Cancelled,
            TerminalOutcome::DeadlineExceeded => {
                return Err("cleanup reached an uncapturable deadline terminal".to_owned());
            }
            TerminalOutcome::Failed { category } => {
                return Err(format!(
                    "cleanup reached an uncapturable failed terminal: {category:?}"
                ));
            }
        };
        Ok(Self(
            request_id,
            outcome,
            to_u64(
                terminal.committed_positions(),
                "cleanup committed-position count",
            )?,
            to_u64(terminal.emitted_tokens(), "cleanup emitted-token count")?,
        ))
    }
}

/// Exact lexicographically keyed `ProbeSnapshot` object from ADR 0007.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct ProbeSnapshotCapture {
    pub(super) command_in_flight: u64,
    pub(super) command_ready: u64,
    pub(super) command_reserved: u64,
    pub(super) command_responded: u64,
    pub(super) dirty: bool,
    pub(super) engine_steps: u64,
    pub(super) outstanding_requests: u64,
    pub(super) owner_done: bool,
    pub(super) park_epoch: u64,
    pub(super) parked: bool,
    pub(super) pump_entries: u64,
    pub(super) pump_hold_observed: u64,
    pub(super) pump_hold_released: u64,
    pub(super) pump_hold_requested: u64,
    pub(super) pump_in_flight: bool,
    pub(super) request_bytes: u64,
    pub(super) shared_bytes: u64,
}

impl ProbeSnapshotCapture {
    pub(super) fn normalize_quiescent(
        snapshot: ActorProbeSnapshot,
        context: &'static str,
    ) -> HarnessResult<Self> {
        if !snapshot.quiescent() {
            return Err(format!("{context} snapshot is not quiescent"));
        }
        Self::normalize(snapshot)
    }

    pub(super) fn normalize_post_shutdown(snapshot: ActorProbeSnapshot) -> HarnessResult<Self> {
        let capture = Self::normalize(snapshot)?;
        require(
            capture.owner_done && capture.dirty && !capture.parked && !capture.pump_in_flight,
            "post-shutdown snapshot has an invalid owner state",
        )?;
        require(
            capture.command_occupancy_is_zero(),
            "post-shutdown snapshot retained command occupancy",
        )?;
        require(
            capture.outstanding_requests == 0
                && capture.request_bytes == 0
                && capture.shared_bytes == 0,
            "post-shutdown snapshot retained actor-owned state",
        )?;
        require(
            capture.pump_hold_requested == capture.pump_hold_observed
                && capture.pump_hold_observed == capture.pump_hold_released,
            "post-shutdown snapshot retained an incomplete pump hold",
        )?;
        Ok(capture)
    }

    fn normalize(snapshot: ActorProbeSnapshot) -> HarnessResult<Self> {
        Ok(Self {
            command_in_flight: to_u64(
                snapshot.command_in_flight,
                "snapshot in-flight command count",
            )?,
            command_ready: to_u64(snapshot.command_ready, "snapshot ready command count")?,
            command_reserved: to_u64(snapshot.command_reserved, "snapshot reserved command count")?,
            command_responded: to_u64(
                snapshot.command_responded,
                "snapshot responded command count",
            )?,
            dirty: snapshot.dirty,
            engine_steps: snapshot.engine_steps,
            outstanding_requests: snapshot.outstanding_requests,
            owner_done: snapshot.owner_done,
            park_epoch: snapshot.park_epoch,
            parked: snapshot.parked,
            pump_entries: snapshot.pump_entries,
            pump_hold_observed: snapshot.pump_hold_observed,
            pump_hold_released: snapshot.pump_hold_released,
            pump_hold_requested: snapshot.pump_hold_requested,
            pump_in_flight: snapshot.pump_in_flight,
            request_bytes: snapshot.request_bytes,
            shared_bytes: snapshot.shared_bytes,
        })
    }

    const fn command_occupancy_is_zero(self) -> bool {
        self.command_in_flight == 0
            && self.command_ready == 0
            && self.command_reserved == 0
            && self.command_responded == 0
    }

    pub(super) fn validate_quiescent(self, context: &'static str) -> HarnessResult<()> {
        require(
            self.parked && !self.dirty && !self.pump_in_flight && !self.owner_done,
            format!("{context} snapshot is not quiescent"),
        )?;
        require(
            self.command_occupancy_is_zero(),
            format!("{context} snapshot retained command occupancy"),
        )?;
        require(
            self.pump_hold_requested == self.pump_hold_observed
                && self.pump_hold_observed == self.pump_hold_released,
            format!("{context} snapshot retained an incomplete pump hold"),
        )
    }

    pub(super) fn validate_not_before(
        self,
        previous: Self,
        context: &'static str,
    ) -> HarnessResult<()> {
        require(
            self.park_epoch >= previous.park_epoch
                && self.pump_entries >= previous.pump_entries
                && self.engine_steps >= previous.engine_steps
                && self.pump_hold_requested >= previous.pump_hold_requested
                && self.pump_hold_observed >= previous.pump_hold_observed
                && self.pump_hold_released >= previous.pump_hold_released,
            format!("{context} snapshot regressed a monotone actor counter"),
        )
    }

    fn validate_request_ledger_presence(self, context: &'static str) -> HarnessResult<()> {
        require(
            (self.outstanding_requests == 0) == (self.request_bytes == 0),
            format!("{context} snapshot request count and ledger presence differ"),
        )
    }
}

pub(super) struct RaceCleanupParts {
    pub(super) cleanup_authorities: Vec<CleanupAuthorityCapture>,
    pub(super) cleanup_counter_final: u64,
    pub(super) cleanup_receivers: Vec<CleanupReceiverCapture>,
    pub(super) pre_cleanup: ProbeSnapshotCapture,
    pub(super) pre_shutdown: ProbeSnapshotCapture,
}

/// All cleanup-owned fields needed by the enclosing repetition capture.
#[derive(Debug)]
pub(super) struct RaceCleanupCapture {
    cleanup_authorities: Vec<CleanupAuthorityCapture>,
    cleanup_counter_final: u64,
    cleanup_receivers: Vec<CleanupReceiverCapture>,
    pre_cleanup: ProbeSnapshotCapture,
    pre_shutdown: ProbeSnapshotCapture,
}

impl RaceCleanupCapture {
    pub(super) fn authority_count(&self) -> usize {
        self.cleanup_authorities.len()
    }

    pub(super) fn receiver_count(&self) -> usize {
        self.cleanup_receivers.len()
    }

    pub(super) const fn counter_final(&self) -> u64 {
        self.cleanup_counter_final
    }

    pub(super) const fn pre_cleanup(&self) -> ProbeSnapshotCapture {
        self.pre_cleanup
    }

    pub(super) const fn pre_shutdown(&self) -> ProbeSnapshotCapture {
        self.pre_shutdown
    }

    pub(super) fn into_parts(self) -> RaceCleanupParts {
        RaceCleanupParts {
            cleanup_authorities: self.cleanup_authorities,
            cleanup_counter_final: self.cleanup_counter_final,
            cleanup_receivers: self.cleanup_receivers,
            pre_cleanup: self.pre_cleanup,
            pre_shutdown: self.pre_shutdown,
        }
    }

    fn validate(
        &self,
        registry: &RaceRegistrySnapshot,
        descriptors: &[Descriptor],
    ) -> HarnessResult<()> {
        registry.require_all_authorities_present()?;
        require(
            descriptors.len() == REQUEST_COUNT,
            "cleanup capture descriptor table has the wrong length",
        )?;
        require(
            self.cleanup_authorities.len() == registry.accepted_count(),
            "cleanup capture authority count differs from the accepted registry",
        )?;
        require(
            self.cleanup_receivers.len() == registry.live_receiver_count(),
            "cleanup capture receiver count differs from the live registry",
        )?;

        self.pre_cleanup
            .validate_quiescent("captured pre-cleanup")?;
        self.pre_cleanup
            .validate_request_ledger_presence("captured pre-cleanup")?;
        require(
            self.pre_cleanup.outstanding_requests
                == to_u64(
                    registry.live_receiver_count(),
                    "captured pre-cleanup live receiver count",
                )?,
            "captured pre-cleanup request count differs from live receiver custody",
        )?;

        for ((client_index, metadata), record) in registry
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.map(|metadata| (index, metadata)))
            .zip(&self.cleanup_authorities)
        {
            let client_index = to_u64(client_index, "captured cleanup authority client index")?;
            require(
                record.2 == client_index,
                "cleanup capture authorities are not the exact ascending accepted set",
            )?;
            require(
                record.3 == metadata.identity.request_id && record.3 != 0,
                "cleanup capture authority identity differs from acceptance",
            )?;
            record.5.validate()?;
            require(
                record.5.0 == ControlOperationCapture::Cancel && record.5.boundary_reached(),
                "cleanup capture authority omitted its cancel boundary",
            )?;
            record.5.require_identity(metadata.identity)?;
            let result = cleanup_authority_result(record.4, record.5)?;
            validate_cleanup_authority_outcome(metadata.receiver, result, record.4, record.5)?;
        }

        let expected_hold_epoch = self
            .pre_cleanup
            .pump_hold_requested
            .checked_add(1)
            .ok_or_else(|| "cleanup pump-hold epoch overflowed".to_owned())?;
        let mut preceding_snapshot = self.pre_cleanup;
        let live_receivers = registry
            .entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                entry
                    .filter(|metadata| matches!(metadata.receiver, ReceiverOwnership::Owned { .. }))
                    .map(|metadata| (index, metadata))
            });
        for (receiver_ordinal, ((client_index, metadata), record)) in
            live_receivers.zip(&self.cleanup_receivers).enumerate()
        {
            let descriptor = descriptors
                .get(client_index)
                .ok_or_else(|| "cleanup receiver descriptor is unavailable".to_owned())?;
            require(
                usize::try_from(descriptor.index)
                    .map_err(|_| "cleanup descriptor index does not fit usize".to_owned())?
                    == client_index,
                "cleanup capture descriptor order changed",
            )?;
            let maximum_emitted = u64::from(descriptor.max_new_tokens);
            let prompt_prefix = descriptor
                .prompt
                .len()
                .checked_sub(1)
                .ok_or_else(|| "cleanup descriptor prompt is empty".to_owned())?;
            let maximum_committed = to_u64(prompt_prefix, "cleanup descriptor prompt prefix")?
                .checked_add(maximum_emitted)
                .ok_or_else(|| "cleanup terminal position bound overflowed".to_owned())?;
            let prompt_prefix = to_u64(prompt_prefix, "cleanup descriptor prompt prefix")?;
            let client_index = to_u64(client_index, "captured cleanup receiver client index")?;
            require(
                record.2 == client_index,
                "cleanup capture receivers are not the exact ascending live set",
            )?;
            require(
                record.3 == metadata.identity.request_id && record.3 != 0,
                "cleanup capture receiver identity differs from acceptance",
            )?;
            require(
                record.4.0 == record.3 && record.4.0 != 0,
                "cleanup capture terminal identity differs from its receiver",
            )?;
            require(
                record.4.2 <= maximum_committed && record.4.3 <= maximum_emitted,
                "cleanup capture terminal exceeds its descriptor bounds",
            )?;
            require(
                record.4.3 == record.4.2.saturating_sub(prompt_prefix),
                "cleanup capture terminal progress is internally inconsistent",
            )?;
            if record.4.1 == TerminalOutcomeCapture::Completed {
                require(
                    record.4.3 != 0,
                    "completed cleanup terminal emitted no output",
                )?;
            }
            require(record.6, "cleanup capture omitted its EOF acknowledgement")?;

            let authority = self
                .cleanup_authorities
                .iter()
                .find(|authority| authority.2 == client_index)
                .ok_or_else(|| {
                    "cleanup receiver has no corresponding authority capture".to_owned()
                })?;
            if authority.5.6 == Some(ControlDispositionCapture::Requested) {
                require(
                    record.4.1 == TerminalOutcomeCapture::Cancelled,
                    "requested cleanup cancellation did not reach a cancelled terminal",
                )?;
            }

            let mut previous_output_index: Option<u64> = None;
            require(
                to_u64(record.5.len(), "cleanup capture output suffix length")? <= record.4.3,
                "cleanup capture output suffix exceeds terminal emissions",
            )?;
            for output in &record.5 {
                require(
                    output.0 == record.3,
                    "cleanup capture output identity differs from its receiver",
                )?;
                require(
                    output.2 <= u64::from(u32::MAX),
                    "cleanup capture output token is out of range",
                )?;
                require(
                    output.1 < record.4.3 && output.1 < maximum_emitted,
                    "cleanup capture output index exceeds terminal emissions",
                )?;
                if let Some(previous) = previous_output_index {
                    let expected = previous
                        .checked_add(1)
                        .ok_or_else(|| "cleanup capture output index overflowed".to_owned())?;
                    require(
                        output.1 == expected,
                        "cleanup capture output suffix is not consecutive FIFO order",
                    )?;
                }
                previous_output_index = Some(output.1);
            }
            if let Some(last_output_index) = previous_output_index {
                require(
                    last_output_index
                        .checked_add(1)
                        .is_some_and(|after_last| after_last == record.4.3),
                    "cleanup capture output suffix does not end at terminal emissions",
                )?;
            }

            record.7.validate_quiescent("captured cleanup post-drop")?;
            record
                .7
                .validate_request_ledger_presence("captured cleanup post-drop")?;
            record
                .7
                .validate_not_before(preceding_snapshot, "captured cleanup post-drop")?;
            require(
                record.7.request_bytes <= preceding_snapshot.request_bytes,
                "cleanup post-drop snapshot increased request ledger bytes",
            )?;
            require(
                record.7.pump_hold_requested == expected_hold_epoch
                    && record.7.pump_hold_observed == expected_hold_epoch
                    && record.7.pump_hold_released == expected_hold_epoch,
                "cleanup post-drop snapshot does not follow the one cleanup pump hold",
            )?;
            require(
                record.7.shared_bytes == self.pre_cleanup.shared_bytes,
                "cleanup post-drop snapshot changed static shared ledger bytes",
            )?;
            let consumed = receiver_ordinal
                .checked_add(1)
                .ok_or_else(|| "cleanup receiver ordinal overflowed".to_owned())?;
            let expected_outstanding = registry
                .live_receiver_count()
                .checked_sub(consumed)
                .ok_or_else(|| "cleanup receiver conservation underflowed".to_owned())?;
            require(
                record.7.outstanding_requests
                    == to_u64(expected_outstanding, "captured post-drop request count")?,
                "cleanup capture post-drop request count did not decrease by one",
            )?;
            preceding_snapshot = record.7;
        }

        self.pre_shutdown
            .validate_quiescent("captured pre-shutdown")?;
        self.pre_shutdown
            .validate_request_ledger_presence("captured pre-shutdown")?;
        self.pre_shutdown
            .validate_not_before(preceding_snapshot, "captured pre-shutdown")?;
        require(
            self.pre_shutdown.request_bytes <= preceding_snapshot.request_bytes,
            "pre-shutdown snapshot increased request ledger bytes",
        )?;
        require(
            self.pre_shutdown.pump_hold_requested == expected_hold_epoch
                && self.pre_shutdown.pump_hold_observed == expected_hold_epoch
                && self.pre_shutdown.pump_hold_released == expected_hold_epoch,
            "pre-shutdown snapshot does not conserve the one cleanup pump hold",
        )?;
        require(
            self.pre_shutdown.shared_bytes == self.pre_cleanup.shared_bytes,
            "pre-shutdown snapshot changed static shared ledger bytes",
        )?;
        require(
            self.pre_shutdown.outstanding_requests == 0,
            "cleanup capture retained pre-shutdown requests",
        )?;
        require(
            self.pre_shutdown.request_bytes == 0,
            "cleanup capture retained pre-shutdown request ledger bytes",
        )?;
        validate_cleanup_counter(
            &self.cleanup_authorities,
            &self.cleanup_receivers,
            self.cleanup_counter_final,
        )
    }
}

struct ReceiverScratch {
    events: Vec<OutputEvent>,
    outputs: Vec<RaceOutput>,
    sink: ActorRequestDropWitnessSink,
    logical_output_capacity: usize,
}

/// Reconciles producer custody, then performs ADR 0007's ordered cleanup.
///
/// All buffers and destructor-witness sinks are allocated before the first
/// cleanup boundary. Authority objects are removed atomically from the
/// registry and retained until every receiver has acknowledged terminal and
/// EOF, then dropped with no registry mutex held.
pub(super) async fn run_ordered_cleanup(
    exits: [RaceProducerExit; PRODUCER_COUNT],
    registry: &RaceRegistry,
    probe: &ActorProbe,
    descriptors: &[Descriptor],
) -> HarnessResult<RaceCleanupCapture> {
    let registry_before = registry.snapshot()?;
    registry_before.require_all_authorities_present()?;
    let mut handles = reconcile_producer_handles(exits, &registry_before)?;
    let live_receiver_count = registry_before.live_receiver_count();
    let accepted_count = registry_before.accepted_count();

    let pre_cleanup_raw = probe
        .wait_quiescent()
        .await
        .map_err(|error| format!("pre-cleanup quiescence failed: {error}"))?;
    let pre_cleanup = ProbeSnapshotCapture::normalize_quiescent(pre_cleanup_raw, "pre-cleanup")?;
    pre_cleanup.validate_request_ledger_presence("pre-cleanup")?;
    require(
        pre_cleanup.command_occupancy_is_zero(),
        "pre-cleanup command table is not empty",
    )?;
    require(
        pre_cleanup.outstanding_requests
            == to_u64(live_receiver_count, "pre-cleanup live receiver count")?,
        "pre-cleanup request count differs from retained receiver custody",
    )?;

    let mut cleanup_authorities = Vec::new();
    cleanup_authorities
        .try_reserve_exact(accepted_count)
        .map_err(|_| "cleanup-authority capture allocation failed".to_owned())?;
    let mut cleanup_receivers = Vec::new();
    cleanup_receivers
        .try_reserve_exact(live_receiver_count)
        .map_err(|_| "cleanup-receiver capture allocation failed".to_owned())?;
    let mut receiver_scratch = preallocate_receiver_scratch(&handles, descriptors)?;
    let counter = AtomicU64::new(0);

    let hold = probe
        .hold_pump()
        .await
        .map_err(|error| format!("cleanup pump hold failed: {error}"))?;
    let taken_authorities = registry.take_authorities()?;
    require(
        taken_authorities.len() == accepted_count,
        "cleanup authority count differs from accepted registry count",
    )?;
    registry.snapshot()?.require_all_authorities_absent()?;

    let mut previous_authority_index = None;
    for authority in &taken_authorities {
        if let Some(previous) = previous_authority_index {
            require(
                authority.client_index > previous,
                "cleanup authorities are not in ascending client order",
            )?;
        }
        previous_authority_index = Some(authority.client_index);
        let metadata = registry_before.get(authority.client_index).ok_or_else(|| {
            format!(
                "cleanup authority client {} was absent from the registry snapshot",
                authority.client_index
            )
        })?;
        require(
            metadata.identity == authority.identity,
            "cleanup authority identity differs from the registry snapshot",
        )?;
        let invocation = next_cleanup_boundary(&counter)?;
        let (result, witness) = authority.authority.cancel_with_stress_witness();
        let (race_result, error, control) =
            ControlWitnessCapture::normalize_cancel(result, witness, authority.identity)?;
        validate_cleanup_authority_outcome(metadata.receiver, race_result, error, control)?;
        let response = next_cleanup_boundary(&counter)?;
        let expected_response = invocation
            .checked_add(1)
            .ok_or_else(|| "cleanup authority interval overflowed".to_owned())?;
        require(
            response == expected_response,
            "cleanup authority interval is not adjacent",
        )?;
        cleanup_authorities.push(CleanupAuthorityCapture(
            invocation,
            response,
            to_u64(authority.client_index, "cleanup authority client index")?,
            authority.identity.request_id,
            error,
            control,
        ));
    }

    hold.release()
        .map_err(|error| format!("cleanup pump release failed: {error}"))?;
    let after_cancel_raw = probe
        .wait_quiescent()
        .await
        .map_err(|error| format!("post-cancel quiescence failed: {error}"))?;
    let mut preceding = ProbeSnapshotCapture::normalize_quiescent(after_cancel_raw, "post-cancel")?;
    preceding.validate_request_ledger_presence("post-cancel")?;
    preceding.validate_not_before(pre_cleanup, "post-cancel")?;
    require(
        preceding.command_occupancy_is_zero(),
        "post-cancel command table is not empty",
    )?;
    require(
        preceding.outstanding_requests
            == to_u64(live_receiver_count, "post-cancel live receiver count")?,
        "post-cancel request count differs from retained receiver custody",
    )?;
    require(
        preceding.request_bytes <= pre_cleanup.request_bytes,
        "post-cancel snapshot increased request ledger bytes",
    )?;
    require(
        preceding.shared_bytes == pre_cleanup.shared_bytes,
        "post-cancel snapshot changed static shared ledger bytes",
    )?;

    for client_index in 0..REQUEST_COUNT {
        let Some(metadata) = registry_before.get(client_index) else {
            continue;
        };
        let ReceiverOwnership::Owned { producer } = metadata.receiver else {
            continue;
        };
        let handle = handles[client_index]
            .take()
            .ok_or_else(|| format!("live cleanup receiver {client_index} lost producer custody"))?;
        let scratch = receiver_scratch[client_index]
            .take()
            .ok_or_else(|| format!("live cleanup receiver {client_index} lacks scratch storage"))?;

        let invocation = next_cleanup_boundary(&counter)?;
        let cleanup = finish_receiver_with_preallocated_witness(
            client_index,
            handle,
            scratch.events,
            scratch.sink,
        )
        .await?;
        registry.mark_receiver_consumed(client_index, producer, metadata.identity)?;

        let terminal = TerminalCapture::normalize(cleanup.terminal, metadata.identity.request_id)?;
        let mut outputs = scratch.outputs;
        require(
            cleanup.outputs.len() <= scratch.logical_output_capacity,
            "cleanup output suffix exceeds its descriptor bound",
        )?;
        let mut previous_output_index: Option<u64> = None;
        for event in cleanup.outputs {
            let output = RaceOutput::from_event(event)?;
            require(
                output.0 == metadata.identity.request_id,
                "cleanup output identity differs from its accepted identity",
            )?;
            if let Some(previous) = previous_output_index {
                let expected = previous
                    .checked_add(1)
                    .ok_or_else(|| "cleanup output index overflowed".to_owned())?;
                require(
                    output.1 == expected,
                    "cleanup output suffix is not consecutive FIFO order",
                )?;
            }
            previous_output_index = Some(output.1);
            require(
                outputs.len() < scratch.logical_output_capacity,
                "cleanup output capture exhausted its descriptor bound",
            )?;
            outputs.push(output);
        }

        let post_drop_raw = probe.wait_quiescent().await.map_err(|error| {
            format!("client {client_index} post-drop quiescence failed: {error}")
        })?;
        let post_drop_quiescent =
            ProbeSnapshotCapture::normalize_quiescent(post_drop_raw, "post-drop")?;
        post_drop_quiescent.validate_request_ledger_presence("post-drop")?;
        post_drop_quiescent.validate_not_before(preceding, "post-drop")?;
        require(
            post_drop_quiescent.command_occupancy_is_zero(),
            "post-drop command table is not empty",
        )?;
        let expected_outstanding = preceding
            .outstanding_requests
            .checked_sub(1)
            .ok_or_else(|| "post-drop request count underflowed".to_owned())?;
        require(
            post_drop_quiescent.outstanding_requests == expected_outstanding,
            "cleanup receiver did not reap exactly one request record",
        )?;
        require(
            post_drop_quiescent.request_bytes <= preceding.request_bytes,
            "post-drop snapshot increased request ledger bytes",
        )?;
        require(
            post_drop_quiescent.shared_bytes == pre_cleanup.shared_bytes,
            "post-drop snapshot changed static shared ledger bytes",
        )?;
        preceding = post_drop_quiescent;

        let response = next_cleanup_boundary(&counter)?;
        let expected_response = invocation
            .checked_add(1)
            .ok_or_else(|| "cleanup receiver interval overflowed".to_owned())?;
        require(
            response == expected_response,
            "cleanup receiver interval is not adjacent",
        )?;
        cleanup_receivers.push(CleanupReceiverCapture(
            invocation,
            response,
            to_u64(client_index, "cleanup receiver client index")?,
            metadata.identity.request_id,
            terminal,
            outputs,
            true,
            post_drop_quiescent,
        ));
    }

    require(
        handles.iter().all(Option::is_none),
        "cleanup retained a producer-local receiver handle",
    )?;
    require(
        receiver_scratch.iter().all(Option::is_none),
        "cleanup retained receiver scratch storage",
    )?;
    let registry_after_receivers = registry.snapshot()?;
    registry_after_receivers.require_all_authorities_absent()?;
    require(
        registry_after_receivers.live_receiver_count() == 0,
        "cleanup registry retained a live receiver",
    )?;

    // `take_authorities` released the registry mutex before returning. Keep
    // this explicit drop here so authority destruction cannot occur under it.
    drop(taken_authorities);
    let pre_shutdown_raw = probe
        .wait_quiescent()
        .await
        .map_err(|error| format!("pre-shutdown quiescence failed: {error}"))?;
    let pre_shutdown = ProbeSnapshotCapture::normalize_quiescent(pre_shutdown_raw, "pre-shutdown")?;
    pre_shutdown.validate_request_ledger_presence("pre-shutdown")?;
    pre_shutdown.validate_not_before(preceding, "pre-shutdown")?;
    require(
        pre_shutdown.command_occupancy_is_zero(),
        "pre-shutdown command table is not empty",
    )?;
    require(
        pre_shutdown.outstanding_requests == 0,
        "pre-shutdown requests remain",
    )?;
    require(
        pre_shutdown.request_bytes == 0,
        "pre-shutdown request ledger is nonzero",
    )?;
    require(
        pre_shutdown.request_bytes <= preceding.request_bytes,
        "pre-shutdown snapshot increased request ledger bytes",
    )?;
    require(
        pre_shutdown.shared_bytes == pre_cleanup.shared_bytes,
        "pre-shutdown snapshot changed static shared ledger bytes",
    )?;

    let capture = RaceCleanupCapture {
        cleanup_authorities,
        cleanup_counter_final: counter.load(Ordering::SeqCst),
        cleanup_receivers,
        pre_cleanup,
        pre_shutdown,
    };
    capture.validate(&registry_before, descriptors)?;
    Ok(capture)
}

fn reconcile_producer_handles(
    exits: [RaceProducerExit; PRODUCER_COUNT],
    registry: &RaceRegistrySnapshot,
) -> HarnessResult<[Option<RequestHandle>; REQUEST_COUNT]> {
    let mut seen = [false; PRODUCER_COUNT];
    let mut merged = std::array::from_fn(|_| None);
    for exit in exits {
        let producer = usize::from(exit.index);
        require(
            producer < PRODUCER_COUNT,
            "producer exit index is out of range",
        )?;
        require(!seen[producer], "producer exit index is duplicated")?;
        seen[producer] = true;
        require(
            exit.handles.len() == REQUEST_COUNT,
            "producer exit handle table has the wrong length",
        )?;
        for (client_index, handle) in exit.handles.into_iter().enumerate() {
            let Some(handle) = handle else {
                continue;
            };
            require(
                client_index % PRODUCER_COUNT == producer,
                "producer exit retained a non-home receiver",
            )?;
            require(
                merged[client_index].is_none(),
                "producer exits duplicated receiver custody",
            )?;
            let metadata = registry
                .get(client_index)
                .ok_or_else(|| format!("producer retained unregistered receiver {client_index}"))?;
            require(
                metadata.receiver
                    == ReceiverOwnership::Owned {
                        producer: exit.index,
                    },
                "producer receiver custody differs from the registry",
            )?;
            require(
                AcceptedIdentity::from_stored_handle(&handle)? == metadata.identity,
                "producer receiver identity differs from the registry",
            )?;
            merged[client_index] = Some(handle);
        }
    }
    require(
        seen.iter().all(|value| *value),
        "producer exit set is incomplete",
    )?;

    for (client_index, merged_handle) in merged.iter().enumerate() {
        match registry.get(client_index) {
            None => require(
                merged_handle.is_none(),
                "unaccepted client retained a receiver",
            )?,
            Some(metadata) => match metadata.receiver {
                ReceiverOwnership::Consumed => require(
                    merged_handle.is_none(),
                    "consumed receiver remained producer-owned",
                )?,
                ReceiverOwnership::Owned { producer } => {
                    require(
                        usize::from(producer) == client_index % PRODUCER_COUNT,
                        "registry receiver has the wrong home producer",
                    )?;
                    let handle = merged_handle.as_ref().ok_or_else(|| {
                        format!("registry live receiver {client_index} lacks producer custody")
                    })?;
                    require(
                        AcceptedIdentity::from_stored_handle(handle)? == metadata.identity,
                        "merged receiver identity differs from the registry",
                    )?;
                }
            },
        }
    }
    Ok(merged)
}

fn preallocate_receiver_scratch(
    handles: &[Option<RequestHandle>; REQUEST_COUNT],
    descriptors: &[Descriptor],
) -> HarnessResult<[Option<ReceiverScratch>; REQUEST_COUNT]> {
    require(
        descriptors.len() == REQUEST_COUNT,
        "cleanup descriptor table has the wrong length",
    )?;
    let mut scratch = std::array::from_fn(|_| None);
    for (client_index, ((handle, descriptor), scratch_slot)) in handles
        .iter()
        .zip(descriptors)
        .zip(scratch.iter_mut())
        .enumerate()
    {
        require(
            usize::try_from(descriptor.index)
                .map_err(|_| "descriptor index does not fit usize".to_owned())?
                == client_index,
            "cleanup descriptor order changed",
        )?;
        if handle.is_none() {
            continue;
        }
        let capacity = usize::try_from(descriptor.max_new_tokens)
            .map_err(|_| "descriptor output bound does not fit usize".to_owned())?;
        require(capacity != 0, "cleanup descriptor output bound is zero")?;
        let mut events = Vec::new();
        events
            .try_reserve_exact(capacity)
            .map_err(|_| format!("client {client_index} cleanup output allocation failed"))?;
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(capacity)
            .map_err(|_| format!("client {client_index} output capture allocation failed"))?;
        *scratch_slot = Some(ReceiverScratch {
            events,
            outputs,
            sink: ActorRequestDropWitnessSink::new(),
            logical_output_capacity: capacity,
        });
    }
    Ok(scratch)
}

fn validate_cleanup_authority_outcome(
    ownership: ReceiverOwnership,
    result: RaceResult,
    error: Option<RaceError>,
    control: ControlWitnessCapture,
) -> HarnessResult<()> {
    control.validate()?;
    require(
        control.0 == ControlOperationCapture::Cancel && control.boundary_reached(),
        "cleanup authority omitted its cancel control boundary",
    )?;
    match ownership {
        ReceiverOwnership::Owned { .. } => {
            require(error.is_none(), "live cleanup authority was stale")?;
            require(
                matches!(
                    result,
                    RaceResult::CancelRequested | RaceResult::CancelAlreadyTerminal
                ),
                "live cleanup authority reached an invalid disposition",
            )?;
            require(
                matches!(
                    control.6,
                    Some(
                        ControlDispositionCapture::Requested
                            | ControlDispositionCapture::AlreadyTerminal
                    )
                ),
                "live cleanup authority control disposition changed",
            )
        }
        ReceiverOwnership::Consumed => match result {
            RaceResult::CancelAlreadyTerminal => {
                require(
                    error.is_none(),
                    "terminal cleanup tombstone retained an error",
                )?;
                require(
                    control.6 == Some(ControlDispositionCapture::AlreadyTerminal),
                    "terminal cleanup tombstone changed disposition",
                )
            }
            RaceResult::Error => {
                require(
                    error == Some(RaceError::request_not_found()),
                    "stale cleanup authority did not retain the exact typed error",
                )?;
                require(
                    control.6.is_none(),
                    "stale cleanup authority retained a disposition",
                )
            }
            _ => Err("consumed cleanup authority reached a nonterminal disposition".to_owned()),
        },
    }
}

fn cleanup_authority_result(
    error: Option<RaceError>,
    control: ControlWitnessCapture,
) -> HarnessResult<RaceResult> {
    match (error, control.6) {
        (None, Some(ControlDispositionCapture::Requested)) => Ok(RaceResult::CancelRequested),
        (None, Some(ControlDispositionCapture::AlreadyRequested)) => {
            Ok(RaceResult::CancelAlreadyRequested)
        }
        (None, Some(ControlDispositionCapture::AlreadyTerminal)) => {
            Ok(RaceResult::CancelAlreadyTerminal)
        }
        (Some(error), None) if error == RaceError::request_not_found() => Ok(RaceResult::Error),
        (None, None) => Err("cleanup authority omitted both error and disposition".to_owned()),
        (Some(_), Some(_)) => {
            Err("cleanup authority retained both error and disposition".to_owned())
        }
        (Some(_), None) => Err("cleanup authority retained an uncapturable error".to_owned()),
    }
}

fn next_cleanup_boundary(counter: &AtomicU64) -> HarnessResult<u64> {
    counter
        .fetch_add(1, Ordering::SeqCst)
        .checked_add(1)
        .ok_or_else(|| "cleanup interval counter overflowed".to_owned())
}

fn validate_cleanup_counter(
    authorities: &[CleanupAuthorityCapture],
    receivers: &[CleanupReceiverCapture],
    final_value: u64,
) -> HarnessResult<()> {
    let mut expected_invocation = 1_u64;
    for (invocation, response) in authorities
        .iter()
        .map(|record| (record.0, record.1))
        .chain(receivers.iter().map(|record| (record.0, record.1)))
    {
        require(
            invocation == expected_invocation,
            "cleanup invocation sequence is not consecutive",
        )?;
        let expected_response = invocation
            .checked_add(1)
            .ok_or_else(|| "cleanup response validation overflowed".to_owned())?;
        require(
            response == expected_response,
            "cleanup response is not adjacent to its invocation",
        )?;
        expected_invocation = response
            .checked_add(1)
            .ok_or_else(|| "cleanup counter validation overflowed".to_owned())?;
    }
    let record_count = authorities
        .len()
        .checked_add(receivers.len())
        .ok_or_else(|| "cleanup record count overflowed usize".to_owned())?;
    let expected_final = to_u64(record_count, "cleanup record count")?
        .checked_mul(2)
        .ok_or_else(|| "cleanup record count overflowed".to_owned())?;
    require(
        final_value == expected_final,
        "cleanup counter final differs from its record count",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiescent_snapshot() -> ActorProbeSnapshot {
        ActorProbeSnapshot {
            dirty: false,
            parked: true,
            pump_in_flight: false,
            owner_done: false,
            park_epoch: 7,
            pump_entries: 11,
            engine_steps: 13,
            command_reserved: 0,
            command_ready: 0,
            command_in_flight: 0,
            command_responded: 0,
            outstanding_requests: 3,
            request_bytes: 17,
            shared_bytes: 19,
            pump_hold_requested: 5,
            pump_hold_observed: 5,
            pump_hold_released: 5,
        }
    }

    fn one_live_registry() -> RaceRegistrySnapshot {
        let identity = AcceptedIdentity {
            request_id: 7,
            control_slot: 2,
            control_generation: 3,
            endpoint_slot: 4,
            endpoint_generation: 5,
        };
        let mut entries = [None; REQUEST_COUNT];
        entries[0] = Some(AcceptedMetadata {
            identity,
            receiver: ReceiverOwnership::Owned { producer: 0 },
            authority_present: true,
        });
        RaceRegistrySnapshot { entries }
    }

    fn one_live_cleanup_capture(outcome: TerminalOutcomeCapture) -> RaceCleanupCapture {
        let registry = one_live_registry();
        let identity = registry.get(0).expect("accepted metadata").identity;
        let loaded = identity.control_generation << 3;
        let control = ControlWitnessCapture::normalize_parts(
            ControlOperationCapture::Cancel,
            true,
            identity.control_slot,
            identity.control_generation,
            loaded,
            loaded | CANCELLED_FLAG,
            ControlOutcome::Disposition(ControlDispositionCapture::Requested),
        )
        .expect("requested cancellation witness");
        let mut pre_cleanup_raw = quiescent_snapshot();
        pre_cleanup_raw.outstanding_requests = 1;
        let pre_cleanup =
            ProbeSnapshotCapture::normalize_quiescent(pre_cleanup_raw, "synthetic pre-cleanup")
                .expect("pre-cleanup capture");
        let mut zero_raw = quiescent_snapshot();
        zero_raw.outstanding_requests = 0;
        zero_raw.request_bytes = 0;
        zero_raw.pump_hold_requested = 6;
        zero_raw.pump_hold_observed = 6;
        zero_raw.pump_hold_released = 6;
        let zero = ProbeSnapshotCapture::normalize_quiescent(zero_raw, "synthetic zero")
            .expect("zero capture");
        RaceCleanupCapture {
            cleanup_authorities: vec![CleanupAuthorityCapture(
                1,
                2,
                0,
                identity.request_id,
                None,
                control,
            )],
            cleanup_counter_final: 4,
            cleanup_receivers: vec![CleanupReceiverCapture(
                3,
                4,
                0,
                identity.request_id,
                TerminalCapture(identity.request_id, outcome, 0, 0),
                Vec::new(),
                true,
                zero,
            )],
            pre_cleanup,
            pre_shutdown: zero,
        }
    }

    #[test]
    fn probe_snapshot_capture_has_exact_quiescent_accounting() {
        let capture = ProbeSnapshotCapture::normalize_quiescent(quiescent_snapshot(), "unit-test")
            .expect("quiescent snapshot");
        assert!(capture.command_occupancy_is_zero());
        assert_eq!(capture.park_epoch, 7);
        assert_eq!(capture.outstanding_requests, 3);
        assert_eq!(capture.request_bytes, 17);
        assert_eq!(capture.shared_bytes, 19);

        let mut invalid = quiescent_snapshot();
        invalid.dirty = true;
        assert!(ProbeSnapshotCapture::normalize_quiescent(invalid, "unit-test").is_err());

        let mut stopped = quiescent_snapshot();
        stopped.dirty = true;
        stopped.parked = false;
        stopped.owner_done = true;
        stopped.outstanding_requests = 0;
        stopped.request_bytes = 0;
        stopped.shared_bytes = 0;
        ProbeSnapshotCapture::normalize_post_shutdown(stopped).expect("post-shutdown snapshot");
        stopped.shared_bytes = 1;
        assert!(ProbeSnapshotCapture::normalize_post_shutdown(stopped).is_err());
    }

    #[test]
    fn cleanup_counter_issues_one_based_seqcst_boundaries() {
        let counter = AtomicU64::new(0);
        assert_eq!(next_cleanup_boundary(&counter).expect("first"), 1);
        assert_eq!(next_cleanup_boundary(&counter).expect("second"), 2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn whole_cleanup_capture_rejects_cross_record_corruption() {
        let registry = one_live_registry();
        let descriptors = crate::descriptors(REQUEST_COUNT);
        one_live_cleanup_capture(TerminalOutcomeCapture::Cancelled)
            .validate(&registry, &descriptors)
            .expect("valid cleanup capture");

        let completed = one_live_cleanup_capture(TerminalOutcomeCapture::Completed);
        assert!(completed.validate(&registry, &descriptors).is_err());

        let mut wrong_receiver = one_live_cleanup_capture(TerminalOutcomeCapture::Cancelled);
        wrong_receiver.cleanup_receivers[0].2 = 1;
        assert!(wrong_receiver.validate(&registry, &descriptors).is_err());

        let mut wrong_count = one_live_cleanup_capture(TerminalOutcomeCapture::Cancelled);
        wrong_count.cleanup_receivers[0].7.outstanding_requests = 1;
        assert!(wrong_count.validate(&registry, &descriptors).is_err());

        let prompt_prefix = u64::try_from(descriptors[0].prompt.len() - 1).expect("prompt prefix");
        let mut inconsistent_progress = one_live_cleanup_capture(TerminalOutcomeCapture::Cancelled);
        inconsistent_progress.cleanup_receivers[0].4.2 = prompt_prefix + 1;
        assert!(
            inconsistent_progress
                .validate(&registry, &descriptors)
                .is_err()
        );

        let mut truncated_suffix = one_live_cleanup_capture(TerminalOutcomeCapture::Cancelled);
        truncated_suffix.cleanup_receivers[0].4.2 = prompt_prefix + 5;
        truncated_suffix.cleanup_receivers[0].4.3 = 5;
        truncated_suffix.cleanup_receivers[0]
            .5
            .push(RaceOutput(7, 0, 1));
        assert!(truncated_suffix.validate(&registry, &descriptors).is_err());

        let mut residual_ledger = one_live_cleanup_capture(TerminalOutcomeCapture::Cancelled);
        residual_ledger.cleanup_receivers[0].7.request_bytes = 1;
        assert!(residual_ledger.validate(&registry, &descriptors).is_err());
    }

    #[test]
    fn cleanup_captures_use_frozen_tuple_and_key_shapes() {
        let terminal = TerminalCapture(1, TerminalOutcomeCapture::Cancelled, 2, 3);
        assert_eq!(
            serde_json::to_string(&terminal).expect("terminal JSON"),
            r#"[1,"cancelled",2,3]"#
        );
        let authority = CleanupAuthorityCapture(
            1,
            2,
            3,
            4,
            Some(RaceError::request_not_found()),
            ControlWitnessCapture(
                ControlOperationCapture::Cancel,
                true,
                5,
                6,
                Hex64(56),
                Hex64(56),
                None,
            ),
        );
        assert_eq!(
            serde_json::to_value(authority)
                .expect("authority JSON")
                .as_array()
                .expect("authority tuple")
                .len(),
            6
        );
        let mut raw = quiescent_snapshot();
        raw.outstanding_requests = 0;
        let snapshot =
            ProbeSnapshotCapture::normalize_quiescent(raw, "unit-test").expect("probe capture");
        let receiver = CleanupReceiverCapture(
            1,
            2,
            3,
            4,
            TerminalCapture(4, TerminalOutcomeCapture::Cancelled, 5, 6),
            Vec::new(),
            true,
            snapshot,
        );
        assert_eq!(
            serde_json::to_value(receiver)
                .expect("receiver JSON")
                .as_array()
                .expect("receiver tuple")
                .len(),
            8
        );
        let value = serde_json::to_value(
            ProbeSnapshotCapture::normalize_quiescent(quiescent_snapshot(), "unit-test")
                .expect("probe capture"),
        )
        .expect("probe JSON");
        let keys = value
            .as_object()
            .expect("probe object")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            keys,
            [
                "command_in_flight",
                "command_ready",
                "command_reserved",
                "command_responded",
                "dirty",
                "engine_steps",
                "outstanding_requests",
                "owner_done",
                "park_epoch",
                "parked",
                "pump_entries",
                "pump_hold_observed",
                "pump_hold_released",
                "pump_hold_requested",
                "pump_in_flight",
                "request_bytes",
                "shared_bytes",
            ]
        );
    }
}
