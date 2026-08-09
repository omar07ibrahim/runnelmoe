//! Bounded, append-only semantic ledger evidence.
//!
//! A trace starts from one immutable snapshot taken after the scheduler's
//! static shared acquisition. Later durable request mutations are retained as
//! complete groups of category deltas. The recorder never allocates after
//! construction: when an entire mutation cannot fit, it retains none of that
//! mutation and marks overflow sticky.

use std::{collections::TryReserveError, fmt, mem::size_of};

use crate::{
    RequestId,
    ledger::{LedgerCategory, LedgerOwnership, LedgerSnapshot},
};

/// Maximum in-memory representation reserved for one ledger evidence event.
pub(crate) const LEDGER_TRACE_EVENT_CAPACITY_BYTES: u64 = 64;

/// Typed logical owner of one ledger category delta.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum LedgerTraceOwner {
    /// Scheduler-wide ownership. Its stable evidence identifier is zero.
    Shared,
    /// Ownership retained by one accepted request.
    Request(RequestId),
}

impl LedgerTraceOwner {
    /// Returns the stable numeric evidence identifier.
    ///
    /// Zero is reserved for shared ownership. Request identities are nonzero,
    /// so owner identifiers have one unambiguous canonical order.
    #[must_use]
    pub const fn evidence_id(self) -> u64 {
        match self {
            Self::Shared => 0,
            Self::Request(request_id) => request_id.get(),
        }
    }

    /// Returns the ledger ownership class required by this owner.
    #[must_use]
    pub const fn ownership(self) -> LedgerOwnership {
        match self {
            Self::Shared => LedgerOwnership::Shared,
            Self::Request(_) => LedgerOwnership::Request,
        }
    }

    /// Returns the request identity for request ownership.
    #[must_use]
    pub const fn request_id(self) -> Option<RequestId> {
        match self {
            Self::Shared => None,
            Self::Request(request_id) => Some(request_id),
        }
    }
}

impl fmt::Debug for LedgerTraceOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shared => formatter.write_str("Shared"),
            Self::Request(_) => formatter
                .debug_tuple("Request")
                .field(&"<redacted>")
                .finish(),
        }
    }
}

/// Direction of one durable logical-capacity mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LedgerMutationKind {
    Acquire = 0,
    Release = 1,
}

impl LedgerMutationKind {
    /// Returns the stable compact evidence identifier (`0=acquire`,
    /// `1=release`).
    #[must_use]
    pub const fn evidence_id(self) -> u8 {
        self as u8
    }
}

/// Immutable sequence-zero baseline for the mutation suffix.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LedgerTraceInitialSnapshot {
    snapshot: LedgerSnapshot,
}

impl LedgerTraceInitialSnapshot {
    pub(crate) const fn new(snapshot: LedgerSnapshot) -> Self {
        Self { snapshot }
    }

    /// Returns the baseline sequence, which is always zero.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        0
    }

    /// Returns the exact logical ledger state captured before mutation one.
    #[must_use]
    pub const fn ledger_snapshot(self) -> LedgerSnapshot {
        self.snapshot
    }
}

impl fmt::Debug for LedgerTraceInitialSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LedgerTraceInitialSnapshot")
            .field("sequence", &0_u64)
            .field("ledger_snapshot", &"<redacted>")
            .finish()
    }
}

/// One category delta in a complete durable ledger mutation.
///
/// Events are ordered canonically by `(owner.evidence_id(),
/// category.evidence_id())`. Every event belonging to the same mutation
/// repeats its sequence number.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LedgerTraceEvent {
    sequence: u64,
    owner: LedgerTraceOwner,
    category: LedgerCategory,
    kind: LedgerMutationKind,
    bytes: u64,
}

impl LedgerTraceEvent {
    const fn new(
        sequence: u64,
        owner: LedgerTraceOwner,
        category: LedgerCategory,
        kind: LedgerMutationKind,
        bytes: u64,
    ) -> Self {
        Self {
            sequence,
            owner,
            category,
            kind,
            bytes,
        }
    }

    /// Returns the contiguous mutation sequence. Sequence zero belongs to the
    /// initial snapshot; retained events begin at one.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    /// Returns the typed logical owner of this category delta.
    #[must_use]
    pub const fn owner(self) -> LedgerTraceOwner {
        self.owner
    }

    /// Returns the closed ledger category changed by this event.
    #[must_use]
    pub const fn category(self) -> LedgerCategory {
        self.category
    }

    /// Returns whether the bytes were acquired or released.
    #[must_use]
    pub const fn kind(self) -> LedgerMutationKind {
        self.kind
    }

    /// Returns the aligned logical byte delta.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Returns the signed replay delta: acquisitions are positive and
    /// releases are negative.
    #[must_use]
    pub const fn signed_delta_bytes(self) -> i64 {
        let bytes = self.bytes as i64;
        match self.kind {
            LedgerMutationKind::Acquire => bytes,
            LedgerMutationKind::Release => -bytes,
        }
    }
}

impl fmt::Debug for LedgerTraceEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LedgerTraceEvent")
            .field("sequence", &self.sequence)
            .field("owner", &self.owner)
            .field("category", &self.category)
            .field("kind", &self.kind)
            .field("bytes", &"<redacted>")
            .finish()
    }
}

const _: () = assert!(
    size_of::<LedgerTraceEvent>() <= LEDGER_TRACE_EVENT_CAPACITY_BYTES as usize,
    "ledger trace event exceeds its independent slot capacity"
);

/// An opaque ordinal into the retained ledger-event prefix.
///
/// A cursor carries no engine cookie. Pair it only with the engine that
/// produced it: a foreign ordinal which happens to equal a mutation boundary
/// selects an unrelated suffix, while an ordinal inside a complete mutation is
/// rejected. Evidence consumers should read from [`Self::origin`] or retain
/// and apply every contiguous suffix from one engine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LedgerTraceCursor(usize);

impl LedgerTraceCursor {
    /// Returns the beginning of a ledger trace.
    #[must_use]
    pub const fn origin() -> Self {
        Self(0)
    }

    /// Returns the index of the next retained event this cursor would read.
    #[must_use]
    pub const fn event_index(self) -> usize {
        self.0
    }

    const fn from_retained_len(retained_len: usize) -> Self {
        Self(retained_len)
    }
}

/// Completeness and fixed-capacity metadata for one ledger-trace read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedgerTraceStatus {
    retained_events: usize,
    event_limit: usize,
    retained_mutations: u64,
    overflowed: bool,
}

impl LedgerTraceStatus {
    /// Returns the sequence assigned to the immutable initial snapshot.
    #[must_use]
    pub const fn initial_sequence(self) -> u64 {
        0
    }

    /// Returns the number of retained category events.
    #[must_use]
    pub const fn retained_events(self) -> usize {
        self.retained_events
    }

    /// Returns the configured independent ledger-event limit.
    #[must_use]
    pub const fn event_limit(self) -> usize {
        self.event_limit
    }

    /// Returns the number and final sequence of fully retained mutations.
    #[must_use]
    pub const fn retained_mutations(self) -> u64 {
        self.retained_mutations
    }

    /// Returns whether at least one whole mutation could not be retained.
    #[must_use]
    pub const fn overflowed(self) -> bool {
        self.overflowed
    }

    /// Returns whether the recorder has retained every mutation since its
    /// initial snapshot.
    ///
    /// This is global recorder health. A non-origin incremental read still
    /// requires the state produced by all earlier contiguous suffixes.
    #[must_use]
    pub const fn healthy(self) -> bool {
        !self.overflowed
    }
}

/// One allocation-free borrowed read of a ledger trace.
#[must_use = "a trace read carries both events and completeness status"]
pub struct LedgerTraceRead<'trace> {
    initial_snapshot: LedgerTraceInitialSnapshot,
    events: &'trace [LedgerTraceEvent],
    start_cursor: LedgerTraceCursor,
    next_cursor: LedgerTraceCursor,
    status: LedgerTraceStatus,
}

impl<'trace> LedgerTraceRead<'trace> {
    /// Returns the immutable sequence-zero baseline.
    ///
    /// It is sufficient to seed a read from [`LedgerTraceCursor::origin`]. An
    /// incremental consumer must also retain the state produced by every
    /// earlier contiguous suffix; the baseline does not replace omitted
    /// events.
    #[must_use]
    pub const fn initial_snapshot(&self) -> LedgerTraceInitialSnapshot {
        self.initial_snapshot
    }

    /// Returns only newly retained events beginning at `start_cursor`.
    #[must_use]
    pub const fn events(&self) -> &'trace [LedgerTraceEvent] {
        self.events
    }

    /// Returns the validated cursor supplied for this read.
    #[must_use]
    pub const fn start_cursor(&self) -> LedgerTraceCursor {
        self.start_cursor
    }

    /// Returns the frontier for the next incremental read.
    #[must_use]
    pub const fn next_cursor(&self) -> LedgerTraceCursor {
        self.next_cursor
    }

    /// Returns trace completeness and fixed-capacity metadata.
    #[must_use]
    pub const fn status(&self) -> LedgerTraceStatus {
        self.status
    }
}

impl fmt::Debug for LedgerTraceRead<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LedgerTraceRead")
            .field("initial_snapshot", &self.initial_snapshot)
            .field("start_cursor", &self.start_cursor)
            .field("next_cursor", &self.next_cursor)
            .field("event_count", &self.events.len())
            .field("status", &self.status)
            .field("events", &"<redacted>")
            .finish()
    }
}

/// One internal change offered to the canonical whole-mutation recorder.
#[derive(Clone, Copy)]
pub(crate) struct LedgerTraceChange {
    owner: LedgerTraceOwner,
    category: LedgerCategory,
    bytes: u64,
}

impl LedgerTraceChange {
    pub(crate) const fn new(owner: LedgerTraceOwner, category: LedgerCategory, bytes: u64) -> Self {
        Self {
            owner,
            category,
            bytes,
        }
    }

    const fn key(self) -> (u64, u8) {
        (self.owner.evidence_id(), self.category.evidence_id())
    }
}

/// Preallocated recorder owned by the capacity ledger.
pub(crate) struct LedgerTraceRecorder {
    initial_snapshot: LedgerTraceInitialSnapshot,
    events: Vec<LedgerTraceEvent>,
    event_limit: usize,
    retained_mutations: u64,
    overflowed: bool,
}

impl LedgerTraceRecorder {
    pub(crate) fn try_new(
        event_limit: usize,
        snapshot: LedgerSnapshot,
    ) -> Result<Self, TryReserveError> {
        let mut events = Vec::new();
        events.try_reserve_exact(event_limit)?;
        Ok(Self {
            initial_snapshot: LedgerTraceInitialSnapshot::new(snapshot),
            events,
            event_limit,
            retained_mutations: 0,
            overflowed: false,
        })
    }

    /// Records a mutation in canonical owner/category order without allocating.
    /// Callers must supply strictly increasing, unique owner/category keys.
    pub(crate) fn record_mutation<I>(&mut self, kind: LedgerMutationKind, changes: I)
    where
        I: Clone + Iterator<Item = LedgerTraceChange>,
    {
        if self.overflowed {
            return;
        }

        let Some(sequence) = self.retained_mutations.checked_add(1) else {
            self.overflowed = true;
            return;
        };

        let mut event_count = 0_usize;
        let mut previous_key = None;
        for change in changes.clone().filter(|change| change.bytes != 0) {
            let key = change.key();
            if change.owner.ownership() != change.category.ownership()
                || change.bytes > i64::MAX as u64
                || previous_key.is_some_and(|previous| key <= previous)
            {
                self.overflowed = true;
                return;
            }
            let Some(next_count) = event_count.checked_add(1) else {
                self.overflowed = true;
                return;
            };
            event_count = next_count;
            previous_key = Some(key);
        }

        // Empty reservations carry no semantic delta and therefore do not
        // consume a trace sequence.
        if event_count == 0 {
            return;
        }
        let Some(required) = self.events.len().checked_add(event_count) else {
            self.overflowed = true;
            return;
        };
        if required > self.event_limit || required > self.events.capacity() {
            self.overflowed = true;
            return;
        }

        for change in changes.filter(|change| change.bytes != 0) {
            self.events.push(LedgerTraceEvent::new(
                sequence,
                change.owner,
                change.category,
                kind,
                change.bytes,
            ));
        }
        self.retained_mutations = sequence;
    }

    pub(crate) fn read_since(&self, cursor: LedgerTraceCursor) -> Option<LedgerTraceRead<'_>> {
        let start = cursor.event_index();
        if start > self.events.len()
            || (start > 0
                && start < self.events.len()
                && self.events[start - 1].sequence() == self.events[start].sequence())
        {
            return None;
        }
        let next_cursor = LedgerTraceCursor::from_retained_len(self.events.len());
        Some(LedgerTraceRead {
            initial_snapshot: self.initial_snapshot,
            events: &self.events[start..],
            start_cursor: cursor,
            next_cursor,
            status: LedgerTraceStatus {
                retained_events: self.events.len(),
                event_limit: self.event_limit,
                retained_mutations: self.retained_mutations,
                overflowed: self.overflowed,
            },
        })
    }
}

impl fmt::Debug for LedgerTraceRecorder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LedgerTraceRecorder")
            .field("initial_snapshot", &self.initial_snapshot)
            .field("retained_events", &self.events.len())
            .field("event_limit", &self.event_limit)
            .field("retained_mutations", &self.retained_mutations)
            .field("overflowed", &self.overflowed)
            .field("events", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        id::request_id_for_test,
        ledger::{CapacityLedger, LEDGER_CATEGORY_COUNT, LedgerCharge},
    };

    fn unlimited_categories() -> [u64; LEDGER_CATEGORY_COUNT] {
        [u64::MAX; LEDGER_CATEGORY_COUNT]
    }

    fn charge(category: LedgerCategory, bytes: u64) -> LedgerCharge {
        LedgerCharge::from_aligned(category, bytes).expect("aligned test charge")
    }

    #[test]
    fn event_representation_fits_its_independent_half_slot() {
        assert!(size_of::<LedgerTraceEvent>() <= 64);
        assert_eq!(crate::trace::TRACE_SLOT_CHARGE_BYTES, 128);
        assert_eq!(crate::trace::SERVICE_TRACE_EVENT_CAPACITY_BYTES, 64);
        assert_eq!(LEDGER_TRACE_EVENT_CAPACITY_BYTES, 64);
    }

    #[test]
    fn recorder_retains_canonical_owner_category_order_and_repeats_one_sequence() {
        let ledger = CapacityLedger::new(u64::MAX, unlimited_categories());
        let mut recorder = LedgerTraceRecorder::try_new(8, ledger.snapshot()).unwrap();
        let first = request_id_for_test(9);
        let second = request_id_for_test(3);
        let changes = [
            LedgerTraceChange::new(
                LedgerTraceOwner::Request(second),
                LedgerCategory::PromptStorage,
                64,
            ),
            LedgerTraceChange::new(
                LedgerTraceOwner::Request(second),
                LedgerCategory::RequestRecord,
                64,
            ),
            LedgerTraceChange::new(LedgerTraceOwner::Request(first), LedgerCategory::Output, 64),
        ];
        recorder.record_mutation(LedgerMutationKind::Acquire, changes.into_iter());
        let read = recorder.read_since(LedgerTraceCursor::origin()).unwrap();
        assert_eq!(read.initial_snapshot().sequence(), 0);
        assert_eq!(
            read.events()
                .iter()
                .map(|event| (
                    event.sequence(),
                    event.owner().evidence_id(),
                    event.category().evidence_id(),
                ))
                .collect::<Vec<_>>(),
            vec![(1, 3, 0), (1, 3, 1), (1, 9, 5)]
        );
        assert_eq!(read.status().retained_mutations(), 1);
        assert!(read.status().healthy());
    }

    #[test]
    fn noncanonical_or_duplicate_input_marks_the_trace_unhealthy_atomically() {
        let ledger = CapacityLedger::new(u64::MAX, unlimited_categories());
        let mut recorder = LedgerTraceRecorder::try_new(8, ledger.snapshot()).unwrap();
        let owner = LedgerTraceOwner::Request(request_id_for_test(3));
        recorder.record_mutation(
            LedgerMutationKind::Acquire,
            [
                LedgerTraceChange::new(owner, LedgerCategory::Output, 64),
                LedgerTraceChange::new(owner, LedgerCategory::PromptStorage, 64),
            ]
            .into_iter(),
        );
        let read = recorder.read_since(LedgerTraceCursor::origin()).unwrap();
        assert!(read.events().is_empty());
        assert_eq!(read.status().retained_mutations(), 0);
        assert!(read.status().overflowed());
    }

    #[test]
    fn whole_mutation_overflow_retains_no_partial_group_and_is_sticky() {
        let ledger = CapacityLedger::new(u64::MAX, unlimited_categories());
        let mut recorder = LedgerTraceRecorder::try_new(2, ledger.snapshot()).unwrap();
        let owner = LedgerTraceOwner::Request(request_id_for_test(1));
        recorder.record_mutation(
            LedgerMutationKind::Acquire,
            [LedgerTraceChange::new(
                owner,
                LedgerCategory::PromptStorage,
                64,
            )]
            .into_iter(),
        );
        recorder.record_mutation(
            LedgerMutationKind::Acquire,
            [
                LedgerTraceChange::new(owner, LedgerCategory::RequestRecord, 64),
                LedgerTraceChange::new(owner, LedgerCategory::Output, 64),
            ]
            .into_iter(),
        );
        recorder.record_mutation(
            LedgerMutationKind::Release,
            [LedgerTraceChange::new(
                owner,
                LedgerCategory::PromptStorage,
                64,
            )]
            .into_iter(),
        );

        let read = recorder.read_since(LedgerTraceCursor::origin()).unwrap();
        assert_eq!(read.events().len(), 1);
        assert_eq!(read.events()[0].sequence(), 1);
        assert_eq!(read.status().retained_mutations(), 1);
        assert!(read.status().overflowed());
    }

    #[test]
    fn exact_full_is_healthy_until_the_next_whole_mutation() {
        let ledger = CapacityLedger::new(u64::MAX, unlimited_categories());
        let mut recorder = LedgerTraceRecorder::try_new(1, ledger.snapshot()).unwrap();
        let owner = LedgerTraceOwner::Request(request_id_for_test(1));
        let change = LedgerTraceChange::new(owner, LedgerCategory::Output, 64);
        recorder.record_mutation(LedgerMutationKind::Acquire, [change].into_iter());
        assert!(
            recorder
                .read_since(LedgerTraceCursor::origin())
                .unwrap()
                .status()
                .healthy()
        );
        recorder.record_mutation(LedgerMutationKind::Release, [change].into_iter());
        assert!(
            recorder
                .read_since(LedgerTraceCursor::origin())
                .unwrap()
                .status()
                .overflowed()
        );
    }

    #[test]
    fn debug_redacts_request_identity_bytes_snapshot_and_events() {
        let owner = LedgerTraceOwner::Request(request_id_for_test(77));
        let event = LedgerTraceEvent::new(
            1,
            owner,
            LedgerCategory::Output,
            LedgerMutationKind::Acquire,
            4_096,
        );
        let event_debug = format!("{event:?}");
        assert!(event_debug.contains("<redacted>"));
        assert!(!event_debug.contains("77"));
        assert!(!event_debug.contains("4096"));
        assert_eq!(event.signed_delta_bytes(), 4_096);
        assert_eq!(
            LedgerTraceEvent::new(
                2,
                owner,
                LedgerCategory::Output,
                LedgerMutationKind::Release,
                4_096,
            )
            .signed_delta_bytes(),
            -4_096
        );

        let ledger = CapacityLedger::new(u64::MAX, unlimited_categories());
        let mut recorder = LedgerTraceRecorder::try_new(2, ledger.snapshot()).unwrap();
        recorder.record_mutation(
            LedgerMutationKind::Acquire,
            [LedgerTraceChange::new(owner, LedgerCategory::Output, 4_096)].into_iter(),
        );
        let read = recorder.read_since(LedgerTraceCursor::origin()).unwrap();
        let read_debug = format!("{read:?}");
        assert!(read_debug.contains("events: \"<redacted>\""));
        assert!(!read_debug.contains("4096"));
    }

    #[test]
    fn cursor_reads_complete_suffix_and_rejects_interior_or_future_ordinal() {
        let mut ledger = CapacityLedger::new(128, unlimited_categories());
        let initial = ledger
            .acquire(&[charge(LedgerCategory::WorkerScratch, 64)])
            .unwrap();
        let mut recorder = LedgerTraceRecorder::try_new(2, ledger.snapshot()).unwrap();
        let owner = LedgerTraceOwner::Request(request_id_for_test(1));
        recorder.record_mutation(
            LedgerMutationKind::Acquire,
            [
                LedgerTraceChange::new(owner, LedgerCategory::PromptStorage, 64),
                LedgerTraceChange::new(owner, LedgerCategory::Output, 64),
            ]
            .into_iter(),
        );
        let first = recorder.read_since(LedgerTraceCursor::origin()).unwrap();
        assert_eq!(
            first
                .initial_snapshot()
                .ledger_snapshot()
                .category(LedgerCategory::WorkerScratch)
                .used(),
            64
        );
        let cursor = first.next_cursor();
        assert!(recorder.read_since(cursor).unwrap().events().is_empty());
        assert!(recorder.read_since(LedgerTraceCursor(1)).is_none());
        assert!(recorder.read_since(LedgerTraceCursor(3)).is_none());
        ledger.release(initial).unwrap();
    }
}
