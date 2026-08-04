use std::sync::{Mutex, MutexGuard};

use runnel_scheduler::{RequestCancellation, RequestHandle};

use super::common::{HarnessResult, REQUEST_COUNT, require, to_u64};

const PRODUCER_COUNT: usize = 2;
const REQUEST_SLOT_COUNT: u64 = 16;
const MAX_CONTROL_GENERATION: u64 = u64::MAX >> 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AcceptedIdentity {
    request_id: u64,
    control_slot: u64,
    control_generation: u64,
    endpoint_slot: u64,
    endpoint_generation: u64,
}

impl AcceptedIdentity {
    fn from_stored_handle(handle: &RequestHandle) -> HarnessResult<Self> {
        let witness = handle.accepted_request_stress_witness();
        Self::normalize(
            handle.request_id().get(),
            Self {
                request_id: witness.request_id().get(),
                control_slot: to_u64(witness.control_slot_index(), "accepted control slot")?,
                control_generation: witness.control_generation(),
                endpoint_slot: to_u64(witness.endpoint_slot_index(), "accepted endpoint slot")?,
                endpoint_generation: witness.endpoint_generation(),
            },
            REQUEST_SLOT_COUNT,
        )
    }

    fn normalize(
        expected_request_id: u64,
        candidate: Self,
        slot_limit: u64,
    ) -> HarnessResult<Self> {
        require(expected_request_id != 0, "accepted request ID is zero")?;
        require(
            candidate.request_id == expected_request_id,
            "accepted witness request ID differs from its handle",
        )?;
        require(slot_limit != 0, "accepted identity slot limit is zero")?;
        require(
            candidate.control_slot < slot_limit,
            "accepted control slot is out of range",
        )?;
        require(
            candidate.control_generation != 0
                && candidate.control_generation <= MAX_CONTROL_GENERATION,
            "accepted control generation is invalid",
        )?;
        require(
            candidate.endpoint_slot < slot_limit,
            "accepted endpoint slot is out of range",
        )?;
        require(
            candidate.endpoint_generation != 0,
            "accepted endpoint generation is zero",
        )?;
        Ok(candidate)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReceiverOwnership {
    Owned { producer: u8 },
    Consumed,
}

struct AcceptedEntry {
    identity: AcceptedIdentity,
    authority: Option<RequestCancellation>,
    receiver: ReceiverOwnership,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AcceptedMetadata {
    identity: AcceptedIdentity,
    receiver: ReceiverOwnership,
    authority_present: bool,
}

impl AcceptedEntry {
    fn metadata(&self) -> AcceptedMetadata {
        AcceptedMetadata {
            identity: self.identity,
            receiver: self.receiver,
            authority_present: self.authority.is_some(),
        }
    }
}

#[derive(Clone)]
struct CancelTarget {
    identity: AcceptedIdentity,
    authority: RequestCancellation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReceiverTarget {
    identity: AcceptedIdentity,
    ownership: ReceiverOwnership,
}

struct TakenAuthority {
    client_index: usize,
    identity: AcceptedIdentity,
    authority: RequestCancellation,
}

type AcceptedTable = [Option<AcceptedEntry>; REQUEST_COUNT];

struct RaceRegistry {
    entries: Mutex<AcceptedTable>,
}

impl RaceRegistry {
    fn new() -> Self {
        Self {
            entries: Mutex::new(std::array::from_fn(|_| None)),
        }
    }

    fn lock(&self) -> HarnessResult<MutexGuard<'_, AcceptedTable>> {
        self.entries
            .lock()
            .map_err(|_| "accepted registry is poisoned".to_owned())
    }

    /// The caller stores the handle in its producer-local table before this
    /// complete shared publication and before reserving the action response.
    fn publish_stored(
        &self,
        client_index: usize,
        producer: u8,
        handle: &RequestHandle,
    ) -> HarnessResult<AcceptedIdentity> {
        require(
            client_index < REQUEST_COUNT,
            "accepted client index is out of range",
        )?;
        require(
            usize::from(producer) < PRODUCER_COUNT,
            "accepted producer is out of range",
        )?;
        require(
            client_index % PRODUCER_COUNT == usize::from(producer),
            "accepted receiver is not owned by its home producer",
        )?;

        let identity = AcceptedIdentity::from_stored_handle(handle)?;
        let entry = AcceptedEntry {
            identity,
            authority: Some(handle.cancellation()),
            receiver: ReceiverOwnership::Owned { producer },
        };

        let mut entries = self.lock()?;
        require(
            entries[client_index].is_none(),
            format!("client {client_index} was accepted more than once"),
        )?;
        for accepted in entries.iter().flatten() {
            require(
                accepted.identity.request_id != identity.request_id,
                "accepted request ID was duplicated",
            )?;
            require(
                (
                    accepted.identity.control_slot,
                    accepted.identity.control_generation,
                ) != (identity.control_slot, identity.control_generation),
                "accepted control identity was duplicated",
            )?;
            require(
                (
                    accepted.identity.endpoint_slot,
                    accepted.identity.endpoint_generation,
                ) != (identity.endpoint_slot, identity.endpoint_generation),
                "accepted endpoint identity was duplicated",
            )?;
        }
        entries[client_index] = Some(entry);
        Ok(identity)
    }

    /// Clones under the publication mutex and returns before any control API
    /// is entered.
    fn lookup_cancel(&self, client_index: usize) -> HarnessResult<Option<CancelTarget>> {
        require(
            client_index < REQUEST_COUNT,
            "cancellation client index is out of range",
        )?;
        let target = {
            let entries = self.lock()?;
            let Some(entry) = entries[client_index].as_ref() else {
                return Ok(None);
            };
            let authority = entry.authority.as_ref().cloned().ok_or_else(|| {
                format!("accepted client {client_index} cancellation authority was already taken")
            })?;
            CancelTarget {
                identity: entry.identity,
                authority,
            }
        };
        Ok(Some(target))
    }

    /// Returns only metadata; the sole receiver remains producer-local.
    fn lookup_receiver(
        &self,
        client_index: usize,
        producer: u8,
    ) -> HarnessResult<Option<ReceiverTarget>> {
        require(
            client_index < REQUEST_COUNT,
            "receiver client index is out of range",
        )?;
        require(
            usize::from(producer) < PRODUCER_COUNT,
            "receiver producer is out of range",
        )?;
        require(
            client_index % PRODUCER_COUNT == usize::from(producer),
            "receiver lookup is not on its home producer",
        )?;
        Ok(self.lock()?[client_index]
            .as_ref()
            .map(|entry| ReceiverTarget {
                identity: entry.identity,
                ownership: entry.receiver,
            }))
    }

    /// Called only after the producer-local handle was actually consumed.
    fn mark_receiver_consumed(
        &self,
        client_index: usize,
        producer: u8,
        expected: AcceptedIdentity,
    ) -> HarnessResult<()> {
        require(
            client_index < REQUEST_COUNT,
            "consumed receiver client index is out of range",
        )?;
        require(
            usize::from(producer) < PRODUCER_COUNT,
            "consumed receiver producer is out of range",
        )?;
        require(
            client_index % PRODUCER_COUNT == usize::from(producer),
            "receiver was consumed by a non-home producer",
        )?;
        let mut entries = self.lock()?;
        let entry = entries[client_index]
            .as_mut()
            .ok_or_else(|| format!("client {client_index} receiver was never accepted"))?;
        require(
            entry.identity == expected,
            format!("client {client_index} receiver identity changed"),
        )?;
        require(
            entry.receiver == ReceiverOwnership::Owned { producer },
            format!("client {client_index} receiver was already consumed"),
        )?;
        entry.receiver = ReceiverOwnership::Consumed;
        Ok(())
    }

    fn snapshot(&self) -> HarnessResult<RaceRegistrySnapshot> {
        let entries = self.lock()?;
        Ok(RaceRegistrySnapshot {
            entries: std::array::from_fn(|index| {
                entries[index].as_ref().map(AcceptedEntry::metadata)
            }),
        })
    }

    /// Validates every entry before mutation. The returned authorities retain
    /// ascending client order and are always dropped outside the mutex.
    fn take_authorities(&self) -> HarnessResult<Vec<TakenAuthority>> {
        let mut taken = Vec::new();
        taken
            .try_reserve_exact(REQUEST_COUNT)
            .map_err(|_| "cancellation-authority collection allocation failed".to_owned())?;

        let mut entries = self.lock()?;
        for (client_index, entry) in entries.iter().enumerate() {
            if let Some(entry) = entry {
                require(
                    entry.authority.is_some(),
                    format!("accepted client {client_index} authority was already taken"),
                )?;
            }
        }
        for (client_index, entry) in entries.iter_mut().enumerate() {
            let Some(entry) = entry else {
                continue;
            };
            let authority = entry
                .authority
                .take()
                .ok_or_else(|| format!("validated client {client_index} authority disappeared"))?;
            taken.push(TakenAuthority {
                client_index,
                identity: entry.identity,
                authority,
            });
        }
        Ok(taken)
    }
}

#[derive(Clone, Debug)]
struct RaceRegistrySnapshot {
    entries: [Option<AcceptedMetadata>; REQUEST_COUNT],
}

impl RaceRegistrySnapshot {
    fn get(&self, client_index: usize) -> Option<AcceptedMetadata> {
        self.entries.get(client_index).copied().flatten()
    }

    fn accepted_count(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    fn live_receiver_count(&self) -> usize {
        self.entries
            .iter()
            .flatten()
            .filter(|entry| matches!(entry.receiver, ReceiverOwnership::Owned { .. }))
            .count()
    }

    fn accepted_ids(&self) -> [u64; REQUEST_COUNT] {
        std::array::from_fn(|index| {
            self.entries[index].map_or(0, |entry| entry.identity.request_id)
        })
    }

    fn require_all_authorities_present(&self) -> HarnessResult<()> {
        require(
            self.entries
                .iter()
                .flatten()
                .all(|entry| entry.authority_present),
            "accepted registry snapshot has a missing authority",
        )
    }

    fn require_all_authorities_absent(&self) -> HarnessResult<()> {
        require(
            self.entries
                .iter()
                .flatten()
                .all(|entry| !entry.authority_present),
            "accepted registry retained an authority after take",
        )
    }
}

#[cfg(test)]
mod tests {
    use runnel_scheduler::{CancelDisposition, RequestHandle};

    use super::*;
    use crate::common::{
        authenticated_workload, finish_receiver, request_spec, spawn_fresh_actor, validate_shutdown,
    };

    fn valid_identity() -> AcceptedIdentity {
        AcceptedIdentity {
            request_id: 1,
            control_slot: 0,
            control_generation: 1,
            endpoint_slot: 0,
            endpoint_generation: 1,
        }
    }

    #[test]
    fn accepted_identity_validation_is_closed_and_bounded() {
        let valid = valid_identity();
        assert_eq!(
            AcceptedIdentity::normalize(1, valid, REQUEST_SLOT_COUNT).expect("valid identity"),
            valid
        );
        assert!(AcceptedIdentity::normalize(0, valid, REQUEST_SLOT_COUNT).is_err());
        assert!(AcceptedIdentity::normalize(2, valid, REQUEST_SLOT_COUNT).is_err());
        assert!(AcceptedIdentity::normalize(1, valid, 0).is_err());

        let mut invalid = valid;
        invalid.control_slot = REQUEST_SLOT_COUNT;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid = valid;
        invalid.control_generation = 0;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid.control_generation = MAX_CONTROL_GENERATION + 1;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid = valid;
        invalid.endpoint_slot = REQUEST_SLOT_COUNT;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid = valid;
        invalid.endpoint_generation = 0;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn registry_publishes_one_complete_entry_and_consumes_receiver_once() {
        let workload = authenticated_workload().expect("authenticated workload");
        let fresh = spawn_fresh_actor(&workload.actor_config)
            .await
            .expect("fresh actor");
        let actor = fresh.actor;
        let probe = fresh.probe;
        let client = actor.client();
        let request = request_spec(&workload.descriptors[0]).expect("request spec");
        let (submission, command) = client.try_submit_with_witness(request);
        assert!(command.command_slot().is_some());
        let handle = submission
            .expect("submit command")
            .wait()
            .await
            .expect("engine admission");

        let registry = RaceRegistry::new();
        assert!(registry.lookup_cancel(0).expect("lookup").is_none());
        assert!(registry.lookup_receiver(0, 0).expect("lookup").is_none());
        assert!(
            registry
                .mark_receiver_consumed(0, 0, valid_identity())
                .is_err()
        );
        assert!(registry.publish_stored(0, 1, &handle).is_err());

        let mut handles = std::iter::repeat_with(|| None)
            .take(REQUEST_COUNT)
            .collect::<Vec<Option<RequestHandle>>>();
        handles[0] = Some(handle);
        let stored = handles[0].as_ref().expect("stored receiver");
        let identity = registry
            .publish_stored(0, 0, stored)
            .expect("complete publication");
        assert_eq!(identity.request_id, 1);
        assert!(identity.control_generation <= MAX_CONTROL_GENERATION);
        assert!(registry.publish_stored(0, 0, stored).is_err());
        assert!(registry.publish_stored(2, 0, stored).is_err());

        let before = registry.snapshot().expect("snapshot");
        before
            .require_all_authorities_present()
            .expect("complete authorities");
        assert_eq!(before.accepted_count(), 1);
        assert_eq!(before.live_receiver_count(), 1);
        assert_eq!(before.accepted_ids()[0], 1);
        assert_eq!(
            before.get(0).expect("metadata").receiver,
            ReceiverOwnership::Owned { producer: 0 }
        );

        let receiver = registry
            .lookup_receiver(0, 0)
            .expect("receiver lookup")
            .expect("receiver target");
        assert_eq!(receiver.identity, identity);
        assert_eq!(receiver.ownership, ReceiverOwnership::Owned { producer: 0 });
        assert!(registry.lookup_receiver(0, 1).is_err());
        let cancel = registry
            .lookup_cancel(0)
            .expect("cancel lookup")
            .expect("cancel target");
        assert_eq!(cancel.identity, identity);
        assert!(matches!(
            cancel.authority.cancel(),
            Ok(CancelDisposition::Requested | CancelDisposition::AlreadyTerminal)
        ));

        let handle = handles[0].take().expect("owned receiver");
        let cleanup = finish_receiver(0, handle).await.expect("receiver cleanup");
        assert_eq!(cleanup.terminal.request_id().get(), identity.request_id);
        let mut wrong = identity;
        wrong.endpoint_generation += 1;
        assert!(registry.mark_receiver_consumed(0, 0, wrong).is_err());
        registry
            .mark_receiver_consumed(0, 0, identity)
            .expect("receiver consumption");
        assert!(registry.mark_receiver_consumed(0, 0, identity).is_err());

        let consumed = registry.snapshot().expect("consumed snapshot");
        assert_eq!(consumed.accepted_count(), 1);
        assert_eq!(consumed.live_receiver_count(), 0);
        assert_eq!(consumed.accepted_ids(), before.accepted_ids());
        assert_eq!(
            consumed.get(0).expect("metadata").receiver,
            ReceiverOwnership::Consumed
        );

        let authorities = registry.take_authorities().expect("authority take");
        assert_eq!(authorities.len(), 1);
        assert_eq!(authorities[0].client_index, 0);
        assert_eq!(authorities[0].identity, identity);
        registry
            .snapshot()
            .expect("post-take snapshot")
            .require_all_authorities_absent()
            .expect("authorities absent");
        assert!(registry.take_authorities().is_err());
        drop(cancel);
        assert!(authorities[0].authority.cancel().is_ok());
        drop(authorities);

        drop(handles);
        drop(client);
        let pre_shutdown = probe
            .wait_quiescent()
            .await
            .expect("pre-shutdown quiescence");
        assert_eq!(pre_shutdown.outstanding_requests, 0);
        assert_eq!(pre_shutdown.request_bytes, 0);
        let report = actor.shutdown().await.expect("shutdown");
        let post_shutdown = probe.snapshot().expect("post-shutdown snapshot");
        validate_shutdown(
            report,
            1,
            0,
            post_shutdown.request_bytes,
            post_shutdown.shared_bytes,
        )
        .expect("zero-effect shutdown");
    }

    #[test]
    fn registry_bounds_fail_before_mutation() {
        let registry = RaceRegistry::new();
        assert!(registry.lookup_cancel(REQUEST_COUNT).is_err());
        assert!(registry.lookup_receiver(REQUEST_COUNT, 0).is_err());
        assert!(registry.lookup_receiver(0, 1).is_err());
        assert!(
            registry
                .mark_receiver_consumed(REQUEST_COUNT, 0, valid_identity())
                .is_err()
        );
        assert!(
            registry
                .mark_receiver_consumed(0, 2, valid_identity())
                .is_err()
        );
        let snapshot = registry.snapshot().expect("snapshot");
        assert_eq!(snapshot.accepted_count(), 0);
        assert_eq!(snapshot.live_receiver_count(), 0);
        assert!(registry.take_authorities().expect("empty take").is_empty());
    }
}
