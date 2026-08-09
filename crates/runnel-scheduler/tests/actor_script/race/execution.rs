use std::sync::Arc;

use runnel_scheduler::{ActorProbe, ActorRequestDropWitnessSink, RequestHandle, SchedulerClient};
use tokio::sync::Barrier;

use super::super::Descriptor;
use super::super::common::{
    DropWitnessContext, drop_receiver_with_preallocated_witness, receive_once_with_witness,
    request_spec,
};
use super::*;

/// One authenticated action plus every allocation needed by its destructor
/// witness path. A sink is present for every `Drop`, including a drop whose
/// target later proves unavailable in the genuine race.
struct PreparedRaceAction {
    key: RaceActionKey,
    drop_sink: Option<ActorRequestDropWitnessSink>,
}

impl PreparedRaceAction {
    fn new(action: &Action) -> HarnessResult<Self> {
        let key = RaceActionKey::authenticate(action)?;
        let drop_sink = (key.kind == RaceKind::ReceiverDrop).then(ActorRequestDropWitnessSink::new);
        let prepared = Self { key, drop_sink };
        prepared.validate()?;
        Ok(prepared)
    }

    fn validate(&self) -> HarnessResult<()> {
        self.key.validate()?;
        require(
            self.drop_sink.is_some() == (self.key.kind == RaceKind::ReceiverDrop),
            "race drop-sink preallocation differs from the action kind",
        )
    }
}

/// The producer-owned receivers returned after the concurrent phase. Cleanup
/// consumes these tables only after both producer tasks have joined.
pub(super) struct RaceProducerExit {
    pub(super) index: u8,
    pub(super) handles: Vec<Option<RequestHandle>>,
}

impl RaceProducerExit {
    pub(super) fn live_receiver_count(&self) -> usize {
        self.handles.iter().flatten().count()
    }

    pub(super) fn take_handle(
        &mut self,
        client_index: usize,
    ) -> HarnessResult<Option<RequestHandle>> {
        require(
            client_index < REQUEST_COUNT,
            "producer-exit client index is out of range",
        )?;
        require(
            client_index % PRODUCER_COUNT == usize::from(self.index),
            "producer exit was asked for a non-home receiver",
        )?;
        Ok(self.handles[client_index].take())
    }
}

/// One persistent producer. All heap-backed storage is constructed before
/// `run` reaches the shared start barrier.
pub(super) struct RaceProducer {
    index: u8,
    client: SchedulerClient,
    probe: ActorProbe,
    descriptors: Arc<[Descriptor]>,
    registry: Arc<RaceRegistry>,
    history: Arc<RaceHistory>,
    actions: Option<Vec<PreparedRaceAction>>,
    handles: Vec<Option<RequestHandle>>,
}

impl RaceProducer {
    fn new(
        index: u8,
        client: SchedulerClient,
        probe: ActorProbe,
        descriptors: Arc<[Descriptor]>,
        registry: Arc<RaceRegistry>,
        history: Arc<RaceHistory>,
        actions: Vec<PreparedRaceAction>,
    ) -> HarnessResult<Self> {
        let producer_index = usize::from(index);
        require(
            producer_index < PRODUCER_COUNT,
            "race producer index is out of range",
        )?;
        require(
            actions.len() == PRODUCER_ACTION_COUNTS[producer_index],
            "race producer action count changed",
        )?;
        require(
            actions.capacity() >= PRODUCER_ACTION_COUNTS[producer_index],
            "race producer action storage was not preallocated",
        )?;
        for action in &actions {
            action.validate()?;
            require(
                usize::from(action.key.producer) == producer_index,
                "prepared action belongs to another race producer",
            )?;
        }

        let handles = preallocate_handle_table()?;
        Ok(Self {
            index,
            client,
            probe,
            descriptors,
            registry,
            history,
            actions: Some(actions),
            handles,
        })
    }

    /// Waits exactly once on the three-party start barrier, then executes the
    /// producer's authenticated subsequence without an orchestration channel,
    /// explicit yield, sleep, retry, or cross-producer gate.
    pub(super) async fn run(mut self, start: Arc<Barrier>) -> HarnessResult<RaceProducerExit> {
        let actions = self
            .actions
            .take()
            .ok_or_else(|| "race producer action storage was already taken".to_owned())?;
        start.wait().await;

        for action in actions {
            self.execute(action).await?;
        }

        for (client_index, handle) in self.handles.iter().enumerate() {
            require(
                client_index % PRODUCER_COUNT == usize::from(self.index) || handle.is_none(),
                "race producer retained a non-home receiver",
            )?;
        }
        Ok(RaceProducerExit {
            index: self.index,
            handles: self.handles,
        })
    }

    async fn execute(&mut self, action: PreparedRaceAction) -> HarnessResult<()> {
        action.validate()?;
        require(
            action.key.producer == self.index,
            "race action reached the wrong producer",
        )?;
        let PreparedRaceAction { key, drop_sink } = action;
        match key.kind {
            RaceKind::Submit if key.client_index.is_none() => {
                require(drop_sink.is_none(), "exhausted submit retained a drop sink")?;
                self.execute_exhausted_submit(key)
            }
            RaceKind::Submit => {
                require(drop_sink.is_none(), "in-range submit retained a drop sink")?;
                self.execute_submit(key).await
            }
            RaceKind::Cancel => {
                require(drop_sink.is_none(), "cancel retained a drop sink")?;
                self.execute_cancel(key)
            }
            RaceKind::ReceiverDrop => {
                let sink = drop_sink.ok_or_else(|| {
                    "receiver-drop action omitted its preallocated sink".to_owned()
                })?;
                self.execute_drop(key, sink)
            }
            RaceKind::Drain => {
                require(drop_sink.is_none(), "drain retained a drop sink")?;
                self.execute_drain(key)
            }
            RaceKind::Wake => {
                require(drop_sink.is_none(), "wake retained a drop sink")?;
                self.execute_wake(key).await
            }
        }
    }

    fn execute_exhausted_submit(&self, key: RaceActionKey) -> HarnessResult<()> {
        require(
            key.submit_attempt
                .is_some_and(|attempt| attempt >= REQUEST_COUNT as u64),
            "exhausted submit has an in-range attempt",
        )?;
        // The authenticated no-op has no target or actor boundary. This is
        // therefore the exact invocation boundary required by ADR 0007.
        let invocation = self
            .history
            .begin_action(key.ordinal as usize, self.index)?;
        let evidence = RaceActionEvidence::with_result(RaceResult::SubmitOfferExhausted);
        self.history.record_action(invocation, evidence)
    }

    async fn execute_submit(&mut self, key: RaceActionKey) -> HarnessResult<()> {
        let client_index = key_client_index(key);
        require(
            client_index % PRODUCER_COUNT == usize::from(self.index),
            "in-range race submit is not on its home producer",
        )?;
        require(
            self.handles[client_index].is_none(),
            "race submit client already owns a receiver",
        )?;
        let descriptor = self
            .descriptors
            .get(client_index)
            .ok_or_else(|| "race submit descriptor is unavailable".to_owned())?;
        let request = request_spec(descriptor)?;

        // All validation and request-view construction precedes the interval.
        // The counter reservation is immediately before actor API entry.
        let invocation = self
            .history
            .begin_action(key.ordinal as usize, self.index)?;
        let (submission, command) = self.client.try_submit_with_witness(request);
        let command = CommandWitnessCapture::normalize(command, true)?;
        let submission = submission
            .map_err(|error| format!("race submit failed before engine admission: {error}"))?;

        let mut evidence = match submission.wait().await {
            Ok(handle) => {
                // Store the sole receiver first, then publish every shared
                // accepted field atomically before action evidence/response.
                self.handles[client_index] = Some(handle);
                let stored = self.handles[client_index]
                    .as_ref()
                    .ok_or_else(|| "stored race receiver disappeared".to_owned())?;
                let identity = self
                    .registry
                    .publish_stored(client_index, self.index, stored)?;
                let mut evidence = RaceActionEvidence::with_result(RaceResult::SubmitAccepted);
                evidence.request_id = Some(identity.request_id);
                evidence.accepted = AcceptedWitnessCapture::from_identity(identity);
                evidence
            }
            Err(error) => {
                require(
                    self.handles[client_index].is_none(),
                    "rejected race submit retained a receiver",
                )?;
                let normalized = RaceError::normalize(&error)?;
                require(
                    normalized == RaceError::request_slots_exhausted(),
                    "race admission rejection was not exact request-slot saturation",
                )?;
                let mut evidence = RaceActionEvidence::with_result(RaceResult::Error);
                evidence.error = Some(normalized);
                evidence
            }
        };
        evidence.command = command;
        self.history.record_action(invocation, evidence)
    }

    fn execute_cancel(&self, key: RaceActionKey) -> HarnessResult<()> {
        let client_index = key_client_index(key);
        require(
            client_index % PRODUCER_COUNT != usize::from(self.index),
            "race cancellation did not run on the opposite producer",
        )?;

        // Target lookup is the first operation after interval invocation.
        let invocation = self
            .history
            .begin_action(key.ordinal as usize, self.index)?;
        let target = self.registry.lookup_cancel(client_index)?;
        let evidence = if let Some(target) = target {
            let (result, witness) = target.authority.cancel_with_stress_witness();
            let (result, error, control) =
                ControlWitnessCapture::normalize_cancel(result, witness, target.identity)?;
            let mut evidence = RaceActionEvidence::with_result(result);
            evidence.error = error;
            evidence.request_id = Some(target.identity.request_id);
            evidence.control = control;
            evidence
        } else {
            unavailable_evidence(None)
        };
        self.history.record_action(invocation, evidence)
    }

    fn execute_drop(
        &mut self,
        key: RaceActionKey,
        sink: ActorRequestDropWitnessSink,
    ) -> HarnessResult<()> {
        let client_index = key_client_index(key);
        require(
            client_index % PRODUCER_COUNT == usize::from(self.index),
            "race receiver drop is not on its home producer",
        )?;

        // Receiver metadata lookup is the first operation after invocation.
        let invocation = self
            .history
            .begin_action(key.ordinal as usize, self.index)?;
        let target = self.registry.lookup_receiver(client_index, self.index)?;
        let evidence = match target {
            None => {
                require(
                    self.handles[client_index].is_none(),
                    "unpublished race receiver exists in the local table",
                )?;
                unavailable_evidence(None)
            }
            Some(target) if target.ownership == ReceiverOwnership::Consumed => {
                require(
                    self.handles[client_index].is_none(),
                    "consumed race receiver remains in the local table",
                )?;
                unavailable_evidence(Some(target.identity.request_id))
            }
            Some(target) => {
                require(
                    target.ownership
                        == ReceiverOwnership::Owned {
                            producer: self.index,
                        },
                    "race registry reports receiver ownership by another producer",
                )?;
                let stored = self.handles[client_index].as_ref().ok_or_else(|| {
                    "owned race receiver is absent from the local table".to_owned()
                })?;
                require_handle_identity(stored, target.identity)?;
                let handle = self.handles[client_index]
                    .take()
                    .ok_or_else(|| "validated race receiver disappeared".to_owned())?;
                let (result, witness) = drop_receiver_with_preallocated_witness(
                    handle,
                    sink,
                    DropWitnessContext::Script,
                )?;
                let witness = witness.ok_or_else(|| {
                    "race receiver destructor omitted its control witness".to_owned()
                })?;
                let (result, error, control) =
                    ControlWitnessCapture::normalize_disconnect(result, witness, target.identity)?;

                // Mark the sole receiver consumed exactly once, and only after
                // its destructor has actually relinquished the endpoint.
                self.registry
                    .mark_receiver_consumed(client_index, self.index, target.identity)?;
                let mut evidence = RaceActionEvidence::with_result(result);
                evidence.error = error;
                evidence.request_id = Some(target.identity.request_id);
                evidence.control = control;
                evidence
            }
        };
        self.history.record_action(invocation, evidence)
    }

    fn execute_drain(&mut self, key: RaceActionKey) -> HarnessResult<()> {
        let client_index = key_client_index(key);
        require(
            client_index % PRODUCER_COUNT == usize::from(self.index),
            "race drain is not on its home producer",
        )?;

        // Receiver metadata lookup is the first operation after invocation.
        let invocation = self
            .history
            .begin_action(key.ordinal as usize, self.index)?;
        let target = self.registry.lookup_receiver(client_index, self.index)?;
        let evidence = match target {
            None => {
                require(
                    self.handles[client_index].is_none(),
                    "unpublished race receiver exists in the local table",
                )?;
                unavailable_evidence(None)
            }
            Some(target) if target.ownership == ReceiverOwnership::Consumed => {
                require(
                    self.handles[client_index].is_none(),
                    "consumed race receiver remains in the local table",
                )?;
                unavailable_evidence(Some(target.identity.request_id))
            }
            Some(target) => {
                require(
                    target.ownership
                        == ReceiverOwnership::Owned {
                            producer: self.index,
                        },
                    "race registry reports receiver ownership by another producer",
                )?;
                let handle = self.handles[client_index].as_mut().ok_or_else(|| {
                    "owned race receiver is absent from the local table".to_owned()
                })?;
                require_handle_identity(handle, target.identity)?;
                let (result, witness) = receive_once_with_witness(handle);
                let receive = NormalizedReceive::normalize(result, witness, target.identity)?;
                let mut evidence = RaceActionEvidence::with_result(receive.result);
                evidence.request_id = Some(target.identity.request_id);
                evidence.output = receive.output;
                evidence.primary = receive.primary;
                evidence.opportunistic = receive.opportunistic_eof;
                evidence.cached_eof = receive.cached_eof;
                evidence
            }
        };
        self.history.record_action(invocation, evidence)
    }

    async fn execute_wake(&self, key: RaceActionKey) -> HarnessResult<()> {
        // Wake has no target. The counter reservation is immediately before
        // entering the global wake-and-acknowledgement API.
        let invocation = self
            .history
            .begin_action(key.ordinal as usize, self.index)?;
        let wake = self
            .probe
            .wake_and_wait()
            .await
            .map_err(|error| format!("race global wake failed: {error}"))?;
        let mut evidence = RaceActionEvidence::with_result(RaceResult::WakeSignaled);
        evidence.wake = WakeWitnessCapture::normalize(wake)?;
        self.history.record_action(invocation, evidence)
    }
}

/// Authenticates, partitions, and preallocates the complete two-producer
/// execution state. Callers construct the three-party barrier only after this
/// function succeeds.
pub(super) fn prepare_race_producers(
    client: SchedulerClient,
    probe: ActorProbe,
    descriptors: Arc<[Descriptor]>,
    actions: &[Action],
    registry: Arc<RaceRegistry>,
    history: Arc<RaceHistory>,
) -> HarnessResult<[RaceProducer; PRODUCER_COUNT]> {
    require(
        descriptors.len() == REQUEST_COUNT,
        "race descriptor count changed",
    )?;
    for (expected_index, descriptor) in descriptors.iter().enumerate() {
        require(
            usize::try_from(descriptor.index).ok() == Some(expected_index),
            "race descriptor order changed",
        )?;
    }
    require(
        history.slots.len() == ACTION_COUNT && history.counter_value() == 0,
        "race history is not fresh",
    )?;
    require(
        history
            .slots
            .iter()
            .all(|slot| !slot.started.load(Ordering::SeqCst)),
        "race history has a pre-started action",
    )?;
    let registry_snapshot = registry.snapshot()?;
    require(
        registry_snapshot.accepted_count() == 0,
        "race registry is not fresh",
    )?;

    let partitions = prepare_action_partitions(actions)?;
    for partition in &partitions {
        for action in partition {
            let ordinal = action.key.ordinal as usize;
            require(
                history.slots[ordinal].key == action.key,
                "race producer action differs from its history slot",
            )?;
        }
    }

    let [producer_zero, producer_one] = partitions;
    Ok([
        RaceProducer::new(
            0,
            client.clone(),
            probe.clone(),
            Arc::clone(&descriptors),
            Arc::clone(&registry),
            Arc::clone(&history),
            producer_zero,
        )?,
        RaceProducer::new(
            1,
            client,
            probe,
            descriptors,
            registry,
            history,
            producer_one,
        )?,
    ])
}

fn prepare_action_partitions(
    actions: &[Action],
) -> HarnessResult<[Vec<PreparedRaceAction>; PRODUCER_COUNT]> {
    require(
        actions.len() == ACTION_COUNT,
        "race producer action corpus length changed",
    )?;
    require(
        actions == generated_actions(ACTION_COUNT).as_slice(),
        "race producer actions differ from the authenticated corpus",
    )?;

    let mut partitions = [Vec::new(), Vec::new()];
    for (producer, partition) in partitions.iter_mut().enumerate() {
        partition
            .try_reserve_exact(PRODUCER_ACTION_COUNTS[producer])
            .map_err(|_| "race producer action allocation failed".to_owned())?;
    }

    for (expected_ordinal, action) in actions.iter().enumerate() {
        let prepared = PreparedRaceAction::new(action)?;
        require(
            prepared.key.ordinal as usize == expected_ordinal,
            "race producer action order is not consecutive",
        )?;
        partitions[usize::from(prepared.key.producer)].push(prepared);
    }
    validate_action_partitions(&partitions, actions)?;
    Ok(partitions)
}

fn validate_action_partitions(
    partitions: &[Vec<PreparedRaceAction>; PRODUCER_COUNT],
    actions: &[Action],
) -> HarnessResult<()> {
    let mut seen = [false; ACTION_COUNT];
    for (producer, partition) in partitions.iter().enumerate() {
        require(
            partition.len() == PRODUCER_ACTION_COUNTS[producer],
            "race producer partition count changed",
        )?;
        require(
            partition.capacity() >= PRODUCER_ACTION_COUNTS[producer],
            "race producer partition was not preallocated",
        )?;
        let mut previous_ordinal = None;
        for action in partition {
            action.validate()?;
            require(
                usize::from(action.key.producer) == producer,
                "race partition contains another producer's action",
            )?;
            let ordinal = action.key.ordinal as usize;
            require(
                ordinal < ACTION_COUNT,
                "race partition ordinal is out of range",
            )?;
            require(!seen[ordinal], "race partition duplicated an action")?;
            require(
                previous_ordinal.is_none_or(|previous| previous < ordinal),
                "race producer subsequence order changed",
            )?;
            require(
                RaceActionKey::authenticate(&actions[ordinal])? == action.key,
                "race partition action differs from its authenticated source",
            )?;
            previous_ordinal = Some(ordinal);
            seen[ordinal] = true;
        }
    }
    require(
        seen.iter().all(|present| *present),
        "race partition omitted an action",
    )
}

fn preallocate_handle_table() -> HarnessResult<Vec<Option<RequestHandle>>> {
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(REQUEST_COUNT)
        .map_err(|_| "race producer handle-table allocation failed".to_owned())?;
    handles.resize_with(REQUEST_COUNT, || None);
    require(
        handles.len() == REQUEST_COUNT && handles.capacity() >= REQUEST_COUNT,
        "race producer handle table was not fully preallocated",
    )?;
    Ok(handles)
}

fn key_client_index(key: RaceActionKey) -> usize {
    debug_assert!(
        key.client_index
            .is_some_and(|index| index < REQUEST_COUNT as u64)
    );
    key.client_index.unwrap_or(0) as usize
}

fn unavailable_evidence(request_id: Option<u64>) -> RaceActionEvidence {
    let mut evidence = RaceActionEvidence::with_result(RaceResult::TargetUnavailable);
    evidence.request_id = request_id;
    evidence
}

fn require_handle_identity(
    handle: &RequestHandle,
    expected: AcceptedIdentity,
) -> HarnessResult<()> {
    let observed = AcceptedIdentity::from_stored_handle(handle)?;
    require(
        observed == expected,
        "producer-local receiver identity differs from the registry",
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::common::authenticated_workload;
    use super::*;

    #[test]
    fn producer_surface_is_send_and_statically_reachable() {
        fn require_send<T: Send>() {}

        require_send::<RaceProducer>();
        require_send::<RaceProducerExit>();
        let _ = prepare_race_producers;
        let _ = RaceProducer::run;
        let _ = RaceProducerExit::live_receiver_count;
        let _ = RaceProducerExit::take_handle;
    }

    #[test]
    fn static_partition_is_complete_ordered_and_fully_preallocated() {
        let workload = authenticated_workload().expect("authenticated workload");
        let partitions =
            prepare_action_partitions(&workload.actions).expect("prepared race partitions");

        assert_eq!(partitions[0].len(), PRODUCER_ACTION_COUNTS[0]);
        assert_eq!(partitions[1].len(), PRODUCER_ACTION_COUNTS[1]);
        assert!(partitions[0].capacity() >= PRODUCER_ACTION_COUNTS[0]);
        assert!(partitions[1].capacity() >= PRODUCER_ACTION_COUNTS[1]);
        assert_eq!(
            partitions
                .iter()
                .flatten()
                .filter(|action| action.key.kind == RaceKind::ReceiverDrop)
                .count(),
            partitions
                .iter()
                .flatten()
                .filter(|action| action.drop_sink.is_some())
                .count()
        );
        validate_action_partitions(&partitions, &workload.actions).expect("valid static partition");

        for (producer, partition) in partitions.iter().enumerate() {
            assert!(
                partition
                    .windows(2)
                    .all(|pair| pair[0].key.ordinal < pair[1].key.ordinal)
            );
            assert!(
                partition
                    .iter()
                    .all(|action| usize::from(action.key.producer) == producer)
            );
        }
    }

    #[test]
    fn static_partition_rejects_any_authenticated_corpus_change() {
        let workload = authenticated_workload().expect("authenticated workload");
        let mut altered = workload.actions;
        altered.swap(0, 1);
        assert!(prepare_action_partitions(&altered).is_err());
    }
}
