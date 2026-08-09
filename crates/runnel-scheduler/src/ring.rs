//! Allocation-stable FIFO run-to-completion and unit-cost deficit round robin.
//!
//! The ring deliberately separates visiting a member from resolving that
//! visit.  A visit advances the retained cursor and marks the member visited;
//! exactly one of `mark_selected` or `mark_blocked` then resolves it.  A
//! selected member retains its one credit until commit, while a recoverable
//! pre-commit failure releases the reservation without consuming that credit.

use std::fmt;

use crate::{
    config::SchedulingPolicy,
    id::{IdentityKind, RequestId, RoundEpoch, RoundEpochIssuer, SlotKey},
};

const MAX_DEFICIT: u8 = 1;

/// Deterministic policy or fixed-capacity failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RingError {
    InvalidCapacity,
    AllocationFailure,
    CapacityExceeded,
    DuplicateMember,
    RequestOrderViolation,
    EpochExhausted,
    RoundAlreadyOpen,
    RoundNotOpen,
    VisitPending,
    NoVisitPending,
    StaleVisit,
    OutstandingReservation,
    StaleReservation,
    InternalInvariant,
}

/// Whether resolving a visit left the current epoch open or closed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoundProgress {
    Open { epoch: RoundEpoch, remaining: usize },
    Closed { epoch: RoundEpoch },
}

impl RoundProgress {
    pub(crate) const fn is_closed(self) -> bool {
        matches!(self, Self::Closed { .. })
    }
}

/// One epoch-scoped member visit. It is consumed by its disposition method.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RingVisit {
    slot_key: SlotKey,
    request_id: RequestId,
    epoch: RoundEpoch,
}

impl RingVisit {
    pub(crate) const fn slot_key(&self) -> SlotKey {
        self.slot_key
    }

    pub(crate) const fn request_id(&self) -> RequestId {
        self.request_id
    }
}

/// One selected service quantum awaiting commit or recoverable release.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ServiceReservation {
    slot_key: SlotKey,
    request_id: RequestId,
    selected_epoch: RoundEpoch,
}

/// A validated, non-escaping right to debit one service quantum.
///
/// Dropping an unapplied permit recovers the credit for a later round. This
/// lets the scheduler nest DRR publication inside the adapter's fallible
/// state-commit callback without a half-committed scheduling decision.
pub(crate) struct ServiceCommitPermit<'a> {
    ring: &'a mut DrrRing,
    member_index: usize,
    applied: bool,
    removed: bool,
}

impl ServiceCommitPermit<'_> {
    pub(crate) fn apply(&mut self) {
        if self.applied || self.removed {
            return;
        }
        let member = &mut self.ring.members[self.member_index];
        member.reservation_epoch = None;
        member.deficit = 0;
        self.applied = true;
    }

    /// Removes the already validated selected member without another
    /// fallible lookup. A selected member has already resolved its visit, so
    /// removal cannot change the current round's remaining-visit count.
    pub(crate) fn remove_member(&mut self) {
        if self.removed {
            return;
        }
        self.ring.members.remove(self.member_index);
        self.ring.retain_cursor_after_removal(self.member_index);
        self.ring.normalize_closed_snapshot_cursor();
        self.removed = true;
    }
}

impl Drop for ServiceCommitPermit<'_> {
    fn drop(&mut self) {
        if !self.applied && !self.removed {
            let member = &mut self.ring.members[self.member_index];
            member.reservation_epoch = None;
            member.deficit = MAX_DEFICIT;
        }
    }
}

impl ServiceReservation {
    pub(crate) const fn slot_key(&self) -> SlotKey {
        self.slot_key
    }

    pub(crate) const fn request_id(&self) -> RequestId {
        self.request_id
    }
}

/// Result of removing a generation-tagged member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RemovedMember {
    #[cfg(test)]
    request_id: RequestId,
    closed_round: Option<RoundEpoch>,
}

impl RemovedMember {
    #[cfg(test)]
    pub(crate) const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[cfg(test)]
    pub(crate) const fn closed_round(self) -> Option<RoundEpoch> {
        self.closed_round
    }
}

#[derive(Debug)]
struct Member {
    slot_key: SlotKey,
    request_id: RequestId,
    join_epoch: RoundEpoch,
    last_visited: Option<RoundEpoch>,
    deficit: u8,
    reservation_epoch: Option<RoundEpoch>,
}

#[derive(Clone, Copy, Debug)]
struct OpenRound {
    epoch: RoundEpoch,
    remaining: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingVisit {
    slot_key: SlotKey,
    request_id: RequestId,
    epoch: RoundEpoch,
}

/// Fixed-capacity deterministic scheduling membership ring.
///
/// Construction is its only allocation point. Insertion, visiting, credit
/// resolution, and removal never grow or replace the backing allocation.
pub(crate) struct DrrRing {
    members: Vec<Member>,
    membership_capacity: usize,
    policy: SchedulingPolicy,
    cursor: usize,
    epochs: RoundEpochIssuer,
    open_round: Option<OpenRound>,
    closed_snapshot_epoch: Option<RoundEpoch>,
    pending_visit: Option<PendingVisit>,
    last_inserted_request: Option<RequestId>,
}

impl fmt::Debug for DrrRing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DrrRing")
            .field("membership_capacity", &self.membership_capacity)
            .field("policy", &self.policy)
            .field("member_count", &self.members.len())
            .field("cursor", &self.cursor)
            .field(
                "open_epoch",
                &self.open_round.map(|round| round.epoch.get()),
            )
            .field(
                "round_remaining",
                &self.open_round.map(|round| round.remaining),
            )
            .field(
                "closed_snapshot_epoch",
                &self.closed_snapshot_epoch.map(RoundEpoch::get),
            )
            .field("visit_pending", &self.pending_visit.is_some())
            .field(
                "reserved_count",
                &self
                    .members
                    .iter()
                    .filter(|member| member.reservation_epoch.is_some())
                    .count(),
            )
            .finish()
    }
}

impl DrrRing {
    /// Fallibly reserves the complete membership storage.
    #[cfg(test)]
    pub(crate) fn try_with_capacity(membership_capacity: usize) -> Result<Self, RingError> {
        Self::try_with_capacity_and_policy(
            membership_capacity,
            SchedulingPolicy::DeficitContinuousExpertCoalesce,
        )
    }

    /// Fallibly reserves membership storage for one immutable policy.
    pub(crate) fn try_with_capacity_and_policy(
        membership_capacity: usize,
        policy: SchedulingPolicy,
    ) -> Result<Self, RingError> {
        if membership_capacity == 0 {
            return Err(RingError::InvalidCapacity);
        }
        let mut members = Vec::new();
        members
            .try_reserve_exact(membership_capacity)
            .map_err(|_| RingError::AllocationFailure)?;
        Ok(Self {
            members,
            membership_capacity,
            policy,
            cursor: 0,
            epochs: RoundEpochIssuer::new(),
            open_round: None,
            closed_snapshot_epoch: None,
            pending_visit: None,
            last_inserted_request: None,
        })
    }

    #[cfg(test)]
    fn try_with_capacity_and_next_epoch(
        membership_capacity: usize,
        next_epoch: u64,
    ) -> Result<Self, RingError> {
        let mut ring = Self::try_with_capacity(membership_capacity)?;
        ring.epochs = RoundEpochIssuer::from_next(next_epoch);
        Ok(ring)
    }

    #[cfg(test)]
    pub(crate) const fn capacity(&self) -> usize {
        self.membership_capacity
    }

    pub(crate) fn len(&self) -> usize {
        self.members.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub(crate) fn current_epoch(&self) -> Option<RoundEpoch> {
        self.open_round.map(|round| round.epoch)
    }

    pub(crate) fn insert(
        &mut self,
        slot_key: SlotKey,
        request_id: RequestId,
    ) -> Result<(), RingError> {
        if self.members.len() == self.membership_capacity {
            return Err(RingError::CapacityExceeded);
        }
        if self
            .members
            .iter()
            .any(|member| member.slot_key == slot_key || member.request_id == request_id)
        {
            return Err(RingError::DuplicateMember);
        }
        if self
            .last_inserted_request
            .is_some_and(|previous| request_id <= previous)
        {
            return Err(RingError::RequestOrderViolation);
        }

        // The issuer always points at the next epoch. During an open round,
        // that is precisely current + 1; while closed, it is the next round
        // that will be opened. Exhaustion therefore rejects a member that
        // could never become eligible.
        let join_epoch = self.epochs.peek().map_err(|error| match error.kind() {
            IdentityKind::RoundEpoch => RingError::EpochExhausted,
            _ => RingError::InternalInvariant,
        })?;

        // Capacity was reserved by the constructor and len is below the
        // logical cap, so this push cannot allocate.
        self.members.push(Member {
            slot_key,
            request_id,
            join_epoch,
            last_visited: None,
            deficit: 0,
            reservation_epoch: None,
        });
        self.last_inserted_request = Some(request_id);
        Ok(())
    }

    /// Explicitly opens one membership snapshot.
    ///
    /// `Ok(None)` means the ring is empty and does not consume an epoch.
    pub(crate) fn open_round(&mut self) -> Result<Option<RoundEpoch>, RingError> {
        if self.open_round.is_some() {
            return Err(RingError::RoundAlreadyOpen);
        }
        if self.pending_visit.is_some() {
            return Err(RingError::InternalInvariant);
        }
        if self.members.is_empty() {
            return Ok(None);
        }
        if self
            .members
            .iter()
            .any(|member| member.reservation_epoch.is_some())
        {
            return Err(RingError::OutstandingReservation);
        }

        let epoch = self.epochs.issue().map_err(|error| match error.kind() {
            IdentityKind::RoundEpoch => RingError::EpochExhausted,
            _ => RingError::InternalInvariant,
        })?;
        let remaining = match self.policy {
            SchedulingPolicy::FifoRunToCompletion => usize::from(
                self.members
                    .first()
                    .is_some_and(|member| member.join_epoch <= epoch),
            ),
            SchedulingPolicy::DeficitContinuousExpertCoalesce => self
                .members
                .iter()
                .filter(|member| member.join_epoch <= epoch)
                .count(),
        };
        if remaining == 0 {
            return Err(RingError::InternalInvariant);
        }
        self.open_round = Some(OpenRound { epoch, remaining });
        self.closed_snapshot_epoch = None;
        Ok(Some(epoch))
    }

    /// Visits exactly one member in the current snapshot and advances the
    /// retained cursor. The returned token must be resolved exactly once.
    pub(crate) fn next_visit(&mut self) -> Result<RingVisit, RingError> {
        if self.pending_visit.is_some() {
            return Err(RingError::VisitPending);
        }
        let round = self.open_round.ok_or(RingError::RoundNotOpen)?;
        if round.remaining == 0 || self.members.is_empty() {
            return Err(RingError::InternalInvariant);
        }

        let member_index = match self.policy {
            SchedulingPolicy::FifoRunToCompletion => {
                let member = self.members.first().ok_or(RingError::InternalInvariant)?;
                if member.join_epoch > round.epoch || member.last_visited == Some(round.epoch) {
                    return Err(RingError::InternalInvariant);
                }
                0
            }
            SchedulingPolicy::DeficitContinuousExpertCoalesce => (0..self.members.len())
                .map(|offset| wrapped_index(self.cursor, offset, self.members.len()))
                .find(|&index| {
                    let member = &self.members[index];
                    member.join_epoch <= round.epoch && member.last_visited != Some(round.epoch)
                })
                .ok_or(RingError::InternalInvariant)?,
        };

        let member = &mut self.members[member_index];
        member.last_visited = Some(round.epoch);
        let pending = PendingVisit {
            slot_key: member.slot_key,
            request_id: member.request_id,
            epoch: round.epoch,
        };
        self.cursor = successor_index(member_index, self.members.len());
        let open_round = self
            .open_round
            .as_mut()
            .ok_or(RingError::InternalInvariant)?;
        open_round.remaining = open_round
            .remaining
            .checked_sub(1)
            .ok_or(RingError::InternalInvariant)?;
        self.pending_visit = Some(pending);

        Ok(RingVisit {
            slot_key: pending.slot_key,
            request_id: pending.request_id,
            epoch: pending.epoch,
        })
    }

    /// Resolves a runnable visit and reserves its one service credit.
    pub(crate) fn mark_selected(
        &mut self,
        visit: RingVisit,
    ) -> Result<(ServiceReservation, RoundProgress), RingError> {
        let member_index = self.validate_pending_visit(&visit)?;
        let member = &mut self.members[member_index];
        if member.reservation_epoch.is_some() {
            return Err(RingError::OutstandingReservation);
        }

        // Equal weight and unit cost specialize the DRR saturation operation
        // to this assignment. A recovered credit remains one; zero becomes
        // one. No member can ever hold a deficit above `MAX_DEFICIT`.
        member.deficit = MAX_DEFICIT;
        member.reservation_epoch = Some(visit.epoch);
        let reservation = ServiceReservation {
            slot_key: member.slot_key,
            request_id: member.request_id,
            selected_epoch: visit.epoch,
        };
        let progress = self.finish_visit()?;
        Ok((reservation, progress))
    }

    /// Resolves a non-runnable visit and clears any unreserved credit.
    pub(crate) fn mark_blocked(&mut self, visit: RingVisit) -> Result<RoundProgress, RingError> {
        let member_index = self.validate_pending_visit(&visit)?;
        let member = &mut self.members[member_index];
        if member.reservation_epoch.is_some() {
            return Err(RingError::OutstandingReservation);
        }
        member.deficit = 0;
        self.finish_visit()
    }

    /// Debits one reserved credit at the successful commit linearization point.
    #[cfg(test)]
    pub(crate) fn commit_credit(
        &mut self,
        reservation: ServiceReservation,
    ) -> Result<(), RingError> {
        self.with_validated_credit_commit(reservation, |mut permit| permit.apply())
    }

    /// Validates the reservation before invoking a callback with an
    /// infallible, single-use publication permit.
    pub(crate) fn with_validated_credit_commit<R, F>(
        &mut self,
        reservation: ServiceReservation,
        apply: F,
    ) -> Result<R, RingError>
    where
        F: for<'permit> FnOnce(ServiceCommitPermit<'permit>) -> R,
    {
        let member_index = self.member_index_for_reservation(&reservation)?;
        let permit = ServiceCommitPermit {
            ring: self,
            member_index,
            applied: false,
            removed: false,
        };
        Ok(apply(permit))
    }

    /// Releases a pre-commit reservation while retaining its one credit.
    ///
    /// The member was already marked visited, so it cannot be selected again
    /// until a later explicitly opened epoch.
    pub(crate) fn recover_credit(
        &mut self,
        reservation: ServiceReservation,
    ) -> Result<(), RingError> {
        let member = self.member_for_reservation_mut(&reservation)?;
        member.reservation_epoch = None;
        member.deficit = MAX_DEFICIT;
        Ok(())
    }

    /// Removes a generation-tagged member without disturbing successor order.
    pub(crate) fn remove(&mut self, slot_key: SlotKey) -> Result<RemovedMember, RingError> {
        let index = self
            .members
            .iter()
            .position(|member| member.slot_key == slot_key)
            .ok_or(RingError::StaleReservation)?;
        let round = self.open_round;
        let member = &self.members[index];
        let removes_unvisited_eligible = round.is_some_and(|open| {
            let belongs_to_snapshot = match self.policy {
                SchedulingPolicy::FifoRunToCompletion => index == 0,
                SchedulingPolicy::DeficitContinuousExpertCoalesce => true,
            };
            belongs_to_snapshot
                && member.join_epoch <= open.epoch
                && member.last_visited != Some(open.epoch)
        });
        let removes_pending = self
            .pending_visit
            .is_some_and(|pending| pending.slot_key == slot_key);
        #[cfg(test)]
        let request_id = member.request_id;

        if removes_unvisited_eligible {
            let open = self
                .open_round
                .as_mut()
                .ok_or(RingError::InternalInvariant)?;
            open.remaining = open
                .remaining
                .checked_sub(1)
                .ok_or(RingError::InternalInvariant)?;
        }
        if removes_pending {
            self.pending_visit = None;
        }

        self.members.remove(index);
        self.retain_cursor_after_removal(index);

        let closed_round = if let Some(open) = self.open_round {
            if open.remaining == 0 && self.pending_visit.is_none() {
                self.record_round_closed(open.epoch);
                Some(open.epoch)
            } else {
                None
            }
        } else {
            self.normalize_closed_snapshot_cursor();
            None
        };

        Ok(RemovedMember {
            #[cfg(test)]
            request_id,
            closed_round,
        })
    }

    pub(crate) fn contains(&self, slot_key: SlotKey) -> bool {
        self.members
            .iter()
            .any(|member| member.slot_key == slot_key)
    }

    pub(crate) fn member_matches(&self, slot_key: SlotKey, request_id: RequestId) -> bool {
        self.members
            .iter()
            .any(|member| member.slot_key == slot_key && member.request_id == request_id)
    }

    /// Iterates authenticated resident membership without allocating.
    pub(crate) fn member_states(
        &self,
    ) -> impl ExactSizeIterator<Item = (SlotKey, RequestId, bool, u8)> + '_ {
        self.members.iter().map(|member| {
            (
                member.slot_key,
                member.request_id,
                member.reservation_epoch.is_some(),
                member.deficit,
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn deficit(&self, slot_key: SlotKey) -> Option<u8> {
        self.members
            .iter()
            .find(|member| member.slot_key == slot_key)
            .map(|member| member.deficit)
    }

    #[cfg(test)]
    pub(crate) fn has_reservation(&self, slot_key: SlotKey) -> bool {
        self.members
            .iter()
            .find(|member| member.slot_key == slot_key)
            .is_some_and(|member| member.reservation_epoch.is_some())
    }

    fn validate_pending_visit(&self, visit: &RingVisit) -> Result<usize, RingError> {
        let pending = self.pending_visit.ok_or(RingError::NoVisitPending)?;
        if pending.slot_key != visit.slot_key
            || pending.request_id != visit.request_id
            || pending.epoch != visit.epoch
        {
            return Err(RingError::StaleVisit);
        }
        self.members
            .iter()
            .position(|member| {
                member.slot_key == visit.slot_key && member.request_id == visit.request_id
            })
            .ok_or(RingError::StaleVisit)
    }

    fn finish_visit(&mut self) -> Result<RoundProgress, RingError> {
        self.pending_visit = None;
        let round = self.open_round.ok_or(RingError::InternalInvariant)?;
        if round.remaining == 0 {
            self.record_round_closed(round.epoch);
            Ok(RoundProgress::Closed { epoch: round.epoch })
        } else {
            Ok(RoundProgress::Open {
                epoch: round.epoch,
                remaining: round.remaining,
            })
        }
    }

    fn member_for_reservation_mut(
        &mut self,
        reservation: &ServiceReservation,
    ) -> Result<&mut Member, RingError> {
        let index = self.member_index_for_reservation(reservation)?;
        Ok(&mut self.members[index])
    }

    fn member_index_for_reservation(
        &self,
        reservation: &ServiceReservation,
    ) -> Result<usize, RingError> {
        let index = self
            .members
            .iter()
            .position(|member| {
                member.slot_key == reservation.slot_key
                    && member.request_id == reservation.request_id
            })
            .ok_or(RingError::StaleReservation)?;
        let member = &self.members[index];
        if member.reservation_epoch != Some(reservation.selected_epoch)
            || member.deficit != MAX_DEFICIT
        {
            return Err(RingError::StaleReservation);
        }
        Ok(index)
    }

    fn retain_cursor_after_removal(&mut self, removed_index: usize) {
        if self.members.is_empty() {
            self.cursor = 0;
        } else if removed_index < self.cursor {
            self.cursor -= 1;
        } else if self.cursor >= self.members.len() {
            self.cursor = 0;
        }
    }

    /// Retains the just-closed membership snapshot until the next continuous
    /// round opens. Later terminal/cancellation removals re-run normalization,
    /// so a newcomer cannot occupy the rollover cursor while any member of
    /// that snapshot still survives.
    fn record_round_closed(&mut self, epoch: RoundEpoch) {
        self.open_round = None;
        if self.policy == SchedulingPolicy::DeficitContinuousExpertCoalesce {
            self.closed_snapshot_epoch = Some(epoch);
            self.normalize_closed_snapshot_cursor();
        }
    }

    /// Points at the first cyclic survivor from the retained closed snapshot.
    /// The bounded scan performs no dynamic allocation. Once no such member
    /// remains, future insertions cannot belong to the old epoch, so the normal
    /// physical successor is final and the marker can be cleared.
    fn normalize_closed_snapshot_cursor(&mut self) {
        let Some(closed_epoch) = self.closed_snapshot_epoch else {
            return;
        };
        if self.policy != SchedulingPolicy::DeficitContinuousExpertCoalesce {
            self.closed_snapshot_epoch = None;
            return;
        }
        if self.members.is_empty() {
            self.cursor = 0;
            self.closed_snapshot_epoch = None;
            return;
        }
        debug_assert!(self.cursor < self.members.len());
        if let Some(index) = (0..self.members.len())
            .map(|offset| wrapped_index(self.cursor, offset, self.members.len()))
            .find(|&index| self.members[index].join_epoch <= closed_epoch)
        {
            self.cursor = index;
        } else {
            self.closed_snapshot_epoch = None;
        }
    }
}

fn successor_index(index: usize, length: usize) -> usize {
    if index + 1 == length { 0 } else { index + 1 }
}

fn wrapped_index(start: usize, offset: usize, length: usize) -> usize {
    let until_end = length - start;
    if offset < until_end {
        start + offset
    } else {
        offset - until_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{request_id_for_test, slot_generation_for_test};

    fn key(index: usize) -> SlotKey {
        SlotKey::new(index, slot_generation_for_test(index as u64 + 1))
    }

    fn add(ring: &mut DrrRing, raw_id: u64) {
        ring.insert(key(raw_id as usize), request_id_for_test(raw_id))
            .expect("member insertion");
    }

    fn fifo_ring(capacity: usize) -> DrrRing {
        DrrRing::try_with_capacity_and_policy(capacity, SchedulingPolicy::FifoRunToCompletion)
            .expect("FIFO ring")
    }

    fn select_and_commit(ring: &mut DrrRing) -> (RequestId, RoundProgress) {
        let visit = ring.next_visit().expect("next visit");
        let request_id = visit.request_id();
        let (reservation, progress) = ring.mark_selected(visit).expect("selection");
        ring.commit_credit(reservation).expect("commit credit");
        (request_id, progress)
    }

    fn select_commit_and_maybe_remove(
        ring: &mut DrrRing,
        remove: bool,
    ) -> (RequestId, RoundProgress) {
        let visit = ring.next_visit().expect("next visit");
        let request_id = visit.request_id();
        let (reservation, progress) = ring.mark_selected(visit).expect("selection");
        ring.with_validated_credit_commit(reservation, |mut permit| {
            permit.apply();
            if remove {
                permit.remove_member();
            }
        })
        .expect("commit credit");
        (request_id, progress)
    }

    #[test]
    fn fifo_round_reselects_the_oldest_member_until_removal() {
        let mut ring = fifo_ring(3);
        add(&mut ring, 1);
        add(&mut ring, 2);
        add(&mut ring, 3);

        for _ in 0..3 {
            ring.open_round().expect("FIFO round").expect("epoch");
            let (selected, progress) = select_and_commit(&mut ring);
            assert_eq!(selected.get(), 1);
            assert!(progress.is_closed());
            assert_eq!(ring.closed_snapshot_epoch, None);
        }

        ring.remove(key(1)).expect("remove completed FIFO head");
        ring.open_round().expect("next FIFO round").expect("epoch");
        let (selected, progress) = select_and_commit(&mut ring);
        assert_eq!(selected.get(), 2);
        assert!(progress.is_closed());
    }

    #[test]
    fn fifo_blocked_head_never_exposes_a_runnable_sibling() {
        let mut ring = fifo_ring(2);
        add(&mut ring, 1);
        add(&mut ring, 2);

        ring.open_round().expect("FIFO round").expect("epoch");
        let visit = ring.next_visit().expect("FIFO head visit");
        assert_eq!(visit.request_id().get(), 1);
        assert!(ring.mark_blocked(visit).expect("blocked head").is_closed());

        ring.open_round().expect("next FIFO round").expect("epoch");
        let visit = ring.next_visit().expect("same FIFO head visit");
        assert_eq!(visit.request_id().get(), 1);
        assert!(ring.mark_blocked(visit).expect("blocked head").is_closed());
    }

    #[test]
    fn removing_a_fifo_non_head_does_not_close_the_head_snapshot() {
        let mut ring = fifo_ring(3);
        add(&mut ring, 1);
        add(&mut ring, 2);
        add(&mut ring, 3);
        ring.open_round().expect("FIFO round").expect("epoch");

        ring.remove(key(2)).expect("remove non-head");
        assert!(ring.current_epoch().is_some());
        let (selected, progress) = select_and_commit(&mut ring);
        assert_eq!(selected.get(), 1);
        assert!(progress.is_closed());

        ring.open_round().expect("next FIFO round").expect("epoch");
        ring.remove(key(1)).expect("remove unvisited head");
        assert!(ring.current_epoch().is_none());
        ring.open_round().expect("successor round").expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 3);
    }

    #[test]
    fn construction_is_fallible_and_logical_capacity_never_grows() {
        assert_eq!(
            DrrRing::try_with_capacity(0).expect_err("zero capacity"),
            RingError::InvalidCapacity
        );
        assert_eq!(
            DrrRing::try_with_capacity(usize::MAX).expect_err("impossible reservation"),
            RingError::AllocationFailure
        );

        let mut ring = DrrRing::try_with_capacity(2).expect("ring");
        assert_eq!(ring.capacity(), 2);
        add(&mut ring, 1);
        add(&mut ring, 2);
        assert_eq!(
            ring.insert(key(3), request_id_for_test(3))
                .expect_err("logical cap"),
            RingError::CapacityExceeded
        );
        let removed = ring.remove(key(1)).expect("remove");
        assert_eq!(removed.request_id(), request_id_for_test(1));
        ring.insert(key(3), request_id_for_test(3))
            .expect("reuse reserved allocation");
        assert_eq!(ring.capacity(), 2);
        assert_eq!(ring.len(), 2);
    }

    #[test]
    fn max_epoch_opens_once_then_exhaustion_is_permanent() {
        let mut ring =
            DrrRing::try_with_capacity_and_next_epoch(2, u64::MAX).expect("ring at final epoch");
        add(&mut ring, 1);

        assert_eq!(
            ring.open_round()
                .expect("final round")
                .expect("nonempty")
                .get(),
            u64::MAX
        );
        let (_, progress) = select_and_commit(&mut ring);
        assert!(progress.is_closed());
        for _ in 0..3 {
            assert_eq!(
                ring.open_round().expect_err("epoch must remain exhausted"),
                RingError::EpochExhausted
            );
        }
        assert_eq!(
            ring.insert(key(2), request_id_for_test(2))
                .expect_err("new member has no join epoch"),
            RingError::EpochExhausted
        );
    }

    #[test]
    fn member_joining_open_round_waits_until_next_epoch() {
        let mut ring = DrrRing::try_with_capacity(3).expect("ring");
        let allocation = ring.members.as_ptr();
        let allocation_capacity = ring.members.capacity();
        add(&mut ring, 1);
        add(&mut ring, 2);
        let first_epoch = ring.open_round().expect("round").expect("epoch");

        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
        add(&mut ring, 3);
        let (second, progress) = select_and_commit(&mut ring);
        assert_eq!(second.get(), 2);
        assert_eq!(progress, RoundProgress::Closed { epoch: first_epoch });

        let second_epoch = ring.open_round().expect("round").expect("epoch");
        assert_eq!(second_epoch.get(), first_epoch.get() + 1);
        let mut trace = Vec::new();
        loop {
            let (request, progress) = select_and_commit(&mut ring);
            trace.push(request.get());
            if progress.is_closed() {
                break;
            }
        }
        // Snapshot survivors retain their cyclic order ahead of a member that
        // joined while the prior epoch was open.
        assert_eq!(trace, [1, 2, 3]);
        assert_eq!(ring.members.as_ptr(), allocation);
        assert_eq!(ring.members.capacity(), allocation_capacity);
    }

    #[test]
    fn selected_terminal_removal_renormalizes_across_a_newcomer_wrap_point() {
        let mut ring = DrrRing::try_with_capacity(4).expect("ring");
        for request_id in 1..=3 {
            add(&mut ring, request_id);
        }
        ring.cursor = 2;
        ring.open_round().expect("rotated round").expect("epoch");

        let visit = ring.next_visit().expect("rotated member visit");
        assert_eq!(visit.request_id().get(), 3);
        let (terminal_reservation, progress) = ring.mark_selected(visit).expect("selection");
        assert!(!progress.is_closed());
        add(&mut ring, 4);
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
        let (selected, progress) = select_and_commit(&mut ring);
        assert_eq!(selected.get(), 2);
        assert!(progress.is_closed());
        assert_eq!(ring.cursor, 2);
        assert_eq!(ring.members[ring.cursor].request_id.get(), 3);

        ring.with_validated_credit_commit(terminal_reservation, |mut permit| {
            permit.apply();
            permit.remove_member();
        })
        .expect("terminal removal");
        assert_eq!(ring.members[ring.cursor].request_id.get(), 1);
        ring.open_round()
            .expect("post-terminal round")
            .expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
    }

    #[test]
    fn closed_round_cancellation_renormalizes_across_a_newcomer_wrap_point() {
        let mut ring = DrrRing::try_with_capacity(4).expect("ring");
        for request_id in 1..=3 {
            add(&mut ring, request_id);
        }
        ring.cursor = 2;
        ring.open_round().expect("rotated round").expect("epoch");

        let visit = ring.next_visit().expect("member three visit");
        assert_eq!(visit.request_id().get(), 3);
        assert!(!ring.mark_blocked(visit).expect("blocked three").is_closed());
        add(&mut ring, 4);
        for request_id in [1, 2] {
            let visit = ring.next_visit().expect("snapshot visit");
            assert_eq!(visit.request_id().get(), request_id);
            let progress = ring.mark_blocked(visit).expect("blocked snapshot member");
            assert_eq!(progress.is_closed(), request_id == 2);
        }
        assert_eq!(ring.members[ring.cursor].request_id.get(), 3);

        ring.remove(key(3)).expect("cancel normalized survivor");
        assert_eq!(ring.members[ring.cursor].request_id.get(), 1);
        ring.open_round()
            .expect("post-cancellation round")
            .expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
    }

    #[test]
    fn recovered_credit_survivor_stays_ahead_of_mid_round_arrival() {
        let mut ring = DrrRing::try_with_capacity(3).expect("ring");
        add(&mut ring, 1);
        add(&mut ring, 2);
        ring.open_round().expect("initial round").expect("epoch");
        let visit = ring.next_visit().expect("recoverable visit");
        let (reservation, progress) = ring.mark_selected(visit).expect("selection");
        assert!(!progress.is_closed());
        ring.recover_credit(reservation).expect("recover credit");
        add(&mut ring, 3);
        let visit = ring.next_visit().expect("closing blocked visit");
        assert_eq!(visit.request_id().get(), 2);
        assert!(ring.mark_blocked(visit).expect("blocked close").is_closed());

        ring.open_round().expect("recovery round").expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
    }

    #[test]
    fn final_epoch_rollover_preserves_survivor_order_and_failed_open_keeps_marker() {
        let mut ring =
            DrrRing::try_with_capacity_and_next_epoch(2, u64::MAX - 1).expect("near-final ring");
        add(&mut ring, 1);
        let penultimate = ring
            .open_round()
            .expect("penultimate round")
            .expect("epoch");
        assert_eq!(penultimate.get(), u64::MAX - 1);
        add(&mut ring, 2);
        let visit = ring.next_visit().expect("penultimate survivor visit");
        assert_eq!(visit.request_id().get(), 1);
        assert!(
            ring.mark_blocked(visit)
                .expect("close penultimate")
                .is_closed()
        );

        let final_epoch = ring.open_round().expect("final round").expect("epoch");
        assert_eq!(final_epoch.get(), u64::MAX);
        assert_eq!(ring.closed_snapshot_epoch, None);
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
        let (newcomer, progress) = select_and_commit(&mut ring);
        assert_eq!(newcomer.get(), 2);
        assert!(progress.is_closed());
        assert_eq!(ring.closed_snapshot_epoch, Some(final_epoch));

        for _ in 0..2 {
            assert_eq!(
                ring.open_round().expect_err("epoch space exhausted"),
                RingError::EpochExhausted
            );
            assert_eq!(ring.closed_snapshot_epoch, Some(final_epoch));
        }
    }

    #[test]
    fn terminal_churn_cannot_put_mid_round_arrivals_ahead_of_a_survivor() {
        let mut ring = DrrRing::try_with_capacity(32).expect("ring");
        for request_id in 1..=16 {
            add(&mut ring, request_id);
        }
        ring.open_round().expect("initial round").expect("epoch");

        for request_id in 1..=16 {
            let (selected, progress) = select_commit_and_maybe_remove(&mut ring, request_id != 1);
            assert_eq!(selected.get(), request_id);
            assert_eq!(progress.is_closed(), request_id == 16);
            if request_id < 16 {
                add(&mut ring, 16 + request_id);
            }
        }

        assert_eq!(ring.len(), 16);
        assert_eq!(
            ring.members
                .iter()
                .map(|member| member.request_id.get())
                .collect::<Vec<_>>(),
            (std::iter::once(1).chain(17..=31)).collect::<Vec<_>>()
        );
        ring.open_round().expect("post-churn round").expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
    }

    #[test]
    fn removal_driven_round_close_retains_a_snapshot_survivor_before_newcomers() {
        let mut ring = DrrRing::try_with_capacity(3).expect("ring");
        add(&mut ring, 1);
        add(&mut ring, 2);
        ring.open_round().expect("initial round").expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
        add(&mut ring, 3);

        let removed = ring.remove(key(2)).expect("remove final unvisited member");
        assert!(removed.closed_round().is_some());
        ring.open_round()
            .expect("post-removal round")
            .expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
    }

    #[test]
    fn removal_driven_close_keeps_physical_successor_when_no_snapshot_member_survives() {
        let mut ring = DrrRing::try_with_capacity(2).expect("ring");
        add(&mut ring, 1);
        ring.open_round().expect("initial round").expect("epoch");
        add(&mut ring, 2);

        let removed = ring.remove(key(1)).expect("remove sole snapshot member");
        assert!(removed.closed_round().is_some());
        ring.open_round().expect("newcomer round").expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 2);
    }

    #[test]
    fn removing_cursor_member_continues_at_its_successor() {
        let mut ring = DrrRing::try_with_capacity(3).expect("ring");
        add(&mut ring, 1);
        add(&mut ring, 2);
        add(&mut ring, 3);
        ring.open_round().expect("round").expect("epoch");

        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
        let removed = ring.remove(key(2)).expect("remove cursor member");
        assert_eq!(removed.request_id().get(), 2);
        assert_eq!(removed.closed_round(), None);
        let (next, progress) = select_and_commit(&mut ring);
        assert_eq!(next.get(), 3);
        assert!(progress.is_closed());

        ring.open_round().expect("next round").expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
    }

    #[test]
    fn removing_pending_last_visit_closes_round_and_invalidates_token() {
        let mut ring = DrrRing::try_with_capacity(1).expect("ring");
        add(&mut ring, 1);
        let epoch = ring.open_round().expect("round").expect("epoch");
        let visit = ring.next_visit().expect("pending visit");

        let removed = ring.remove(key(1)).expect("remove pending member");
        assert_eq!(removed.closed_round(), Some(epoch));
        assert!(ring.is_empty());
        assert_eq!(
            ring.mark_blocked(visit).expect_err("stale visit"),
            RingError::NoVisitPending
        );
        assert_eq!(ring.open_round().expect("empty ring"), None);
    }

    #[test]
    fn blocked_commit_and_recoverable_credit_follow_distinct_rules() {
        let mut ring = DrrRing::try_with_capacity(2).expect("ring");
        add(&mut ring, 1);
        add(&mut ring, 2);
        ring.open_round().expect("round one").expect("epoch");

        let retry_visit = ring.next_visit().expect("request one visit");
        let (retry_reservation, open) = ring.mark_selected(retry_visit).expect("reserve one");
        assert!(!open.is_closed());
        assert_eq!(ring.deficit(key(1)), Some(1));
        assert!(ring.has_reservation(key(1)));
        ring.recover_credit(retry_reservation)
            .expect("recoverable failure");
        assert_eq!(ring.deficit(key(1)), Some(1));
        assert!(!ring.has_reservation(key(1)));

        let blocked_visit = ring.next_visit().expect("request two visit");
        assert_eq!(blocked_visit.request_id().get(), 2);
        assert!(
            ring.mark_blocked(blocked_visit)
                .expect("blocked")
                .is_closed()
        );
        assert_eq!(ring.deficit(key(2)), Some(0));

        // Request one cannot be revisited in its failed epoch. Its preserved
        // credit is capped at one when selected in the next epoch.
        assert_eq!(
            ring.next_visit().expect_err("round is closed"),
            RingError::RoundNotOpen
        );
        ring.open_round().expect("round two").expect("epoch");
        let retry_visit = ring.next_visit().expect("request one retry");
        assert_eq!(retry_visit.request_id().get(), 1);
        let (retry_reservation, _) = ring.mark_selected(retry_visit).expect("retry select");
        assert_eq!(ring.deficit(key(1)), Some(MAX_DEFICIT));
        ring.commit_credit(retry_reservation).expect("retry commit");
        assert_eq!(ring.deficit(key(1)), Some(0));
    }

    #[test]
    fn outstanding_selection_blocks_next_epoch_until_resolved() {
        let mut ring = DrrRing::try_with_capacity(1).expect("ring");
        add(&mut ring, 1);
        ring.open_round().expect("round").expect("epoch");
        let visit = ring.next_visit().expect("visit");
        let (reservation, progress) = ring.mark_selected(visit).expect("selection");
        assert!(progress.is_closed());
        assert_eq!(
            ring.open_round().expect_err("reservation unresolved"),
            RingError::OutstandingReservation
        );
        ring.recover_credit(reservation).expect("recover");
        ring.open_round().expect("later epoch").expect("epoch");
    }

    #[test]
    fn terminal_commit_permit_removes_member_and_retains_round_cursor() {
        let mut ring = DrrRing::try_with_capacity(3).expect("ring");
        add(&mut ring, 1);
        add(&mut ring, 2);
        add(&mut ring, 3);
        ring.open_round().expect("round").expect("epoch");

        let visit = ring.next_visit().expect("first visit");
        let (reservation, progress) = ring.mark_selected(visit).expect("first selection");
        assert!(!progress.is_closed());
        ring.with_validated_credit_commit(reservation, |mut permit| {
            permit.apply();
            permit.remove_member();
        })
        .expect("terminal member commit");

        assert_eq!(ring.len(), 2);
        assert!(!ring.contains(key(1)));
        let (second, progress) = select_and_commit(&mut ring);
        assert_eq!(second.get(), 2);
        assert!(!progress.is_closed());
        let (third, progress) = select_and_commit(&mut ring);
        assert_eq!(third.get(), 3);
        assert!(progress.is_closed());
    }

    #[test]
    fn dropping_unapplied_commit_permit_recovers_selected_credit() {
        let mut ring = DrrRing::try_with_capacity(1).expect("ring");
        add(&mut ring, 1);
        ring.open_round().expect("round").expect("epoch");
        let visit = ring.next_visit().expect("visit");
        let (reservation, progress) = ring.mark_selected(visit).expect("selection");
        assert!(progress.is_closed());

        ring.with_validated_credit_commit(reservation, |_| {})
            .expect("validated permit");
        assert_eq!(ring.deficit(key(1)), Some(MAX_DEFICIT));
        assert!(!ring.has_reservation(key(1)));
        ring.open_round().expect("retry round").expect("epoch");
        assert_eq!(select_and_commit(&mut ring).0.get(), 1);
    }

    #[test]
    fn duplicate_and_out_of_order_members_do_not_mutate_ring() {
        let mut ring = DrrRing::try_with_capacity(3).expect("ring");
        add(&mut ring, 2);
        assert_eq!(
            ring.insert(key(2), request_id_for_test(3))
                .expect_err("duplicate slot key"),
            RingError::DuplicateMember
        );
        assert_eq!(
            ring.insert(key(3), request_id_for_test(2))
                .expect_err("duplicate request ID"),
            RingError::DuplicateMember
        );
        assert_eq!(
            ring.insert(key(1), request_id_for_test(1))
                .expect_err("request IDs must increase"),
            RingError::RequestOrderViolation
        );
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn independent_first_sixteen_service_trace_visits_every_member_once() {
        let mut ring = DrrRing::try_with_capacity(16).expect("ring");
        for request_id in 1..=16 {
            add(&mut ring, request_id);
        }
        ring.open_round().expect("fairness round").expect("epoch");

        let mut observed = [0_u64; 16];
        for (index, value) in observed.iter_mut().enumerate() {
            let (request_id, progress) = select_and_commit(&mut ring);
            *value = request_id.get();
            assert_eq!(progress.is_closed(), index == 15);
        }

        // This literal is the independently specified accepted-ingress trace,
        // not a second call into ring ordering logic.
        let expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert_eq!(observed, expected);
        assert!(
            ring.members.iter().all(|member| {
                member.deficit <= MAX_DEFICIT && member.reservation_epoch.is_none()
            })
        );
    }

    #[test]
    fn fifo_first_sixteen_service_trace_stays_on_the_oldest_member() {
        let mut ring = fifo_ring(16);
        for request_id in 1..=16 {
            add(&mut ring, request_id);
        }

        let mut observed = [0_u64; 16];
        for value in &mut observed {
            ring.open_round()
                .expect("FIFO fairness round")
                .expect("epoch");
            let (request_id, progress) = select_and_commit(&mut ring);
            *value = request_id.get();
            assert!(progress.is_closed());
        }

        assert_eq!(observed, [1; 16]);
        assert!(
            ring.members.iter().all(|member| {
                member.deficit <= MAX_DEFICIT && member.reservation_epoch.is_none()
            })
        );
    }

    #[test]
    fn debug_snapshot_omits_member_identities() {
        let mut ring = DrrRing::try_with_capacity(2).expect("ring");
        add(&mut ring, 9);
        let debug = format!("{ring:?}");
        assert!(debug.contains("member_count: 1"));
        assert!(!debug.contains("RequestId"));
        assert!(!debug.contains("SlotKey"));
    }
}
