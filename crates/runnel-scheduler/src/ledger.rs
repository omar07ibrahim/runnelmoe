//! Exact logical-capacity accounting for scheduler-owned memory.
//!
//! This ledger accounts for declared semantic capacities. It does not attempt
//! to estimate allocator metadata or resident set size. Every charge is rounded
//! independently to [`LEDGER_QUANTUM_BYTES`] before it enters an acquire plan.

use std::fmt;

use thiserror::Error;

/// Logical accounting quantum used by every scheduler charge.
pub const LEDGER_QUANTUM_BYTES: u64 = 64;

/// Whether a category is retained by one request or shared by the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerOwnership {
    Request,
    Shared,
}

/// Closed set of scheduler logical-memory categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LedgerCategory {
    PromptStorage,
    RequestRecord,
    RequestSlot,
    ActiveState,
    PendingTransaction,
    Output,
    Terminal,
    WorkerScratch,
    SamplingScratch,
    CoalescedBatch,
    ModelResident,
    PagePool,
    Trace,
    AdmissionReserve,
    ActorCommand,
    ActorControl,
}

impl LedgerCategory {
    /// Every category in stable snapshot order.
    pub const ALL: [Self; 16] = [
        Self::PromptStorage,
        Self::RequestRecord,
        Self::RequestSlot,
        Self::ActiveState,
        Self::PendingTransaction,
        Self::Output,
        Self::Terminal,
        Self::WorkerScratch,
        Self::SamplingScratch,
        Self::CoalescedBatch,
        Self::ModelResident,
        Self::PagePool,
        Self::Trace,
        Self::AdmissionReserve,
        Self::ActorCommand,
        Self::ActorControl,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn ownership(self) -> LedgerOwnership {
        match self {
            Self::PromptStorage
            | Self::RequestRecord
            | Self::RequestSlot
            | Self::ActiveState
            | Self::PendingTransaction
            | Self::Output
            | Self::Terminal => LedgerOwnership::Request,
            Self::WorkerScratch
            | Self::SamplingScratch
            | Self::CoalescedBatch
            | Self::ModelResident
            | Self::PagePool
            | Self::Trace
            | Self::AdmissionReserve
            | Self::ActorCommand
            | Self::ActorControl => LedgerOwnership::Shared,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PromptStorage => "prompt_storage",
            Self::RequestRecord => "request_record",
            Self::RequestSlot => "request_slot",
            Self::ActiveState => "active_state",
            Self::PendingTransaction => "pending_transaction",
            Self::Output => "output",
            Self::Terminal => "terminal",
            Self::WorkerScratch => "worker_scratch",
            Self::SamplingScratch => "sampling_scratch",
            Self::CoalescedBatch => "coalesced_batch",
            Self::ModelResident => "model_resident",
            Self::PagePool => "page_pool",
            Self::Trace => "trace",
            Self::AdmissionReserve => "admission_reserve",
            Self::ActorCommand => "actor_command",
            Self::ActorControl => "actor_control",
        }
    }
}

impl fmt::Display for LedgerCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Number of entries in every fixed ledger array.
pub const LEDGER_CATEGORY_COUNT: usize = LedgerCategory::ALL.len();

/// Scope reported by a sanitized ledger failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerScope {
    Category(LedgerCategory),
    RequestOwned,
    Shared,
    Aggregate,
}

impl fmt::Display for LedgerScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Category(category) => category.fmt(formatter),
            Self::RequestOwned => formatter.write_str("request_owned"),
            Self::Shared => formatter.write_str("shared"),
            Self::Aggregate => formatter.write_str("aggregate"),
        }
    }
}

/// Payload-free errors from logical capacity accounting.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LedgerError {
    #[error("ledger charge rounding overflow for {category}")]
    ChargeRoundingOverflow { category: LedgerCategory },

    #[error("ledger charge for {category} is not aligned to 64 bytes")]
    UnalignedCharge { category: LedgerCategory },

    #[error("ledger arithmetic overflow in {scope}")]
    ArithmeticOverflow { scope: LedgerScope },

    #[error("ledger capacity exhausted in {scope}: required {required}, limit {limit}")]
    CapacityExceeded {
        scope: LedgerScope,
        required: u64,
        limit: u64,
    },

    #[error("ledger release underflow in {scope}: releasing {releasing}, available {available}")]
    ReleaseUnderflow {
        scope: LedgerScope,
        releasing: u64,
        available: u64,
    },

    #[error("a provisional ledger acquisition is no longer current")]
    StaleProvisionalAcquisition,

    #[error("ledger mutation identity space is exhausted")]
    MutationIdentityExhausted,

    #[error("ledger reservation partitions do not exactly cover the aggregate")]
    ReservationPartitionMismatch,
}

impl LedgerError {
    /// Capacity failures are the ledger's `resource_exhausted` class.
    #[must_use]
    pub const fn is_resource_exhausted(self) -> bool {
        matches!(self, Self::CapacityExceeded { .. })
    }
}

/// One independently rounded entry in an atomic acquire plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedgerCharge {
    category: LedgerCategory,
    bytes: u64,
}

impl LedgerCharge {
    /// Rounds one semantic payload capacity upward to the ledger quantum.
    pub fn from_payload(category: LedgerCategory, payload_bytes: u64) -> Result<Self, LedgerError> {
        let bytes =
            round_up(payload_bytes).ok_or(LedgerError::ChargeRoundingOverflow { category })?;
        Ok(Self { category, bytes })
    }

    /// Accepts a charge already rounded to the ledger quantum.
    pub fn from_aligned(category: LedgerCategory, bytes: u64) -> Result<Self, LedgerError> {
        if !bytes.is_multiple_of(LEDGER_QUANTUM_BYTES) {
            return Err(LedgerError::UnalignedCharge { category });
        }
        Ok(Self { category, bytes })
    }

    #[must_use]
    pub const fn category(self) -> LedgerCategory {
        self.category
    }

    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Per-category values in one immutable ledger snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CategorySnapshot {
    used: u64,
    peak: u64,
    limit: u64,
}

impl CategorySnapshot {
    #[must_use]
    pub const fn used(self) -> u64 {
        self.used
    }

    #[must_use]
    pub const fn peak(self) -> u64 {
        self.peak
    }

    #[must_use]
    pub const fn limit(self) -> u64 {
        self.limit
    }
}

/// Exact current and high-water accounting with no dynamic storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedgerSnapshot {
    categories: [CategorySnapshot; LEDGER_CATEGORY_COUNT],
    request_used: u64,
    request_peak: u64,
    shared_used: u64,
    shared_peak: u64,
    total_used: u64,
    total_peak: u64,
    total_limit: u64,
}

impl LedgerSnapshot {
    #[must_use]
    pub const fn category(self, category: LedgerCategory) -> CategorySnapshot {
        self.categories[category.index()]
    }

    #[must_use]
    pub const fn request_used(self) -> u64 {
        self.request_used
    }

    #[must_use]
    pub const fn request_peak(self) -> u64 {
        self.request_peak
    }

    #[must_use]
    pub const fn shared_used(self) -> u64 {
        self.shared_used
    }

    #[must_use]
    pub const fn shared_peak(self) -> u64 {
        self.shared_peak
    }

    #[must_use]
    pub const fn total_used(self) -> u64 {
        self.total_used
    }

    #[must_use]
    pub const fn total_peak(self) -> u64 {
        self.total_peak
    }

    #[must_use]
    pub const fn total_limit(self) -> u64 {
        self.total_limit
    }

    #[must_use]
    pub const fn current_is_zero(self) -> bool {
        self.total_used == 0 && self.request_used == 0 && self.shared_used == 0
    }
}

/// A committed charge which must return to the same logical ledger once.
#[must_use = "committed ledger reservations must be released exactly once"]
#[derive(Debug, PartialEq, Eq)]
pub struct LedgerReservation {
    bytes: [u64; LEDGER_CATEGORY_COUNT],
    request_bytes: u64,
    shared_bytes: u64,
    total_bytes: u64,
}

impl LedgerReservation {
    pub(crate) const fn empty() -> Self {
        Self {
            bytes: [0; LEDGER_CATEGORY_COUNT],
            request_bytes: 0,
            shared_bytes: 0,
            total_bytes: 0,
        }
    }

    #[must_use]
    #[cfg(test)]
    pub const fn bytes(&self, category: LedgerCategory) -> u64 {
        self.bytes[category.index()]
    }

    #[must_use]
    #[cfg(test)]
    pub const fn request_bytes(&self) -> u64 {
        self.request_bytes
    }

    #[must_use]
    #[cfg(test)]
    pub const fn shared_bytes(&self) -> u64 {
        self.shared_bytes
    }

    #[must_use]
    #[cfg(test)]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Splits one category from this reservation with no ledger mutation.
    /// The first result owns the selected category and the second owns all
    /// remaining categories.
    #[must_use = "both reservation partitions must be retained or released"]
    pub fn split_category(
        self,
        selected: LedgerCategory,
    ) -> (LedgerReservation, LedgerReservation) {
        self.split_with(|category| category == selected)
    }

    /// Splits a closed set of categories with no ledger mutation. Duplicate
    /// category selectors are idempotent; every charged byte appears in
    /// exactly one result.
    #[must_use = "both reservation partitions must be retained or released"]
    #[cfg(test)]
    pub fn split_categories(
        self,
        selected: &[LedgerCategory],
    ) -> (LedgerReservation, LedgerReservation) {
        let mut mask = [false; LEDGER_CATEGORY_COUNT];
        for category in selected {
            mask[category.index()] = true;
        }
        self.split_with(|category| mask[category.index()])
    }

    /// Partitions all request-owned and shared categories with no ledger
    /// mutation. The request-owned reservation is returned first.
    #[must_use = "both reservation partitions must be retained or released"]
    #[cfg(test)]
    pub fn split_by_ownership(self) -> (LedgerReservation, LedgerReservation) {
        self.split_with(|category| category.ownership() == LedgerOwnership::Request)
    }

    fn split_with(
        self,
        selected: impl Fn(LedgerCategory) -> bool,
    ) -> (LedgerReservation, LedgerReservation) {
        let mut selected_bytes = [0_u64; LEDGER_CATEGORY_COUNT];
        let mut remainder_bytes = [0_u64; LEDGER_CATEGORY_COUNT];
        let mut selected_request = 0_u64;
        let mut selected_shared = 0_u64;
        let mut selected_total = 0_u64;

        for category in LedgerCategory::ALL {
            let index = category.index();
            let bytes = self.bytes[index];
            if selected(category) {
                selected_bytes[index] = bytes;
                selected_total += bytes;
                match category.ownership() {
                    LedgerOwnership::Request => selected_request += bytes,
                    LedgerOwnership::Shared => selected_shared += bytes,
                }
            } else {
                remainder_bytes[index] = bytes;
            }
        }

        let selected_reservation = LedgerReservation {
            bytes: selected_bytes,
            request_bytes: selected_request,
            shared_bytes: selected_shared,
            total_bytes: selected_total,
        };
        let remainder_reservation = LedgerReservation {
            bytes: remainder_bytes,
            request_bytes: self.request_bytes - selected_request,
            shared_bytes: self.shared_bytes - selected_shared,
            total_bytes: self.total_bytes - selected_total,
        };
        (selected_reservation, remainder_reservation)
    }

    /// Splits an already validated exact partition without allocation or a
    /// second ledger mutation. Batch admission constructs every partition
    /// before its publication boundary and proves that their sum is this
    /// reservation.
    fn split_prevalidated(
        self,
        selected: ReservationPartition,
    ) -> (LedgerReservation, LedgerReservation) {
        debug_assert!(
            LedgerCategory::ALL
                .iter()
                .all(|category| selected.bytes[category.index()] <= self.bytes[category.index()])
        );
        debug_assert!(selected.request_bytes <= self.request_bytes);
        debug_assert!(selected.shared_bytes <= self.shared_bytes);
        debug_assert!(selected.total_bytes <= self.total_bytes);

        let remainder_bytes =
            std::array::from_fn(|index| self.bytes[index] - selected.bytes[index]);
        let selected_reservation = LedgerReservation {
            bytes: selected.bytes,
            request_bytes: selected.request_bytes,
            shared_bytes: selected.shared_bytes,
            total_bytes: selected.total_bytes,
        };
        let remainder_reservation = LedgerReservation {
            bytes: remainder_bytes,
            request_bytes: self.request_bytes - selected.request_bytes,
            shared_bytes: self.shared_bytes - selected.shared_bytes,
            total_bytes: self.total_bytes - selected.total_bytes,
        };
        (selected_reservation, remainder_reservation)
    }

    #[must_use]
    pub(crate) const fn is_empty(&self) -> bool {
        self.total_bytes == 0
    }
}

/// Fixed-size descriptor for one prevalidated partition of a committed
/// reservation. It carries no ownership until `split_prevalidated` consumes
/// the aggregate reservation at publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReservationPartition {
    bytes: [u64; LEDGER_CATEGORY_COUNT],
    request_bytes: u64,
    shared_bytes: u64,
    total_bytes: u64,
}

/// Proof-carrying ordered partitions whose checked sum exactly equals one
/// aggregate provisional reservation.
pub(crate) struct ExactReservationPartitions {
    partitions: Vec<ReservationPartition>,
    next: usize,
}

impl ExactReservationPartitions {
    pub(crate) const fn empty() -> Self {
        Self {
            partitions: Vec::new(),
            next: 0,
        }
    }

    pub(crate) fn split_next(
        &mut self,
        aggregate: LedgerReservation,
    ) -> (LedgerReservation, LedgerReservation) {
        debug_assert!(self.next < self.partitions.len());
        let partition = self.partitions[self.next];
        self.next += 1;
        aggregate.split_prevalidated(partition)
    }

    #[must_use]
    pub(crate) fn is_complete(&self) -> bool {
        self.next == self.partitions.len()
    }
}

impl ReservationPartition {
    pub(crate) fn from_charges(charges: &[LedgerCharge]) -> Result<Self, LedgerError> {
        let plan = aggregate_plan(charges)?;
        Ok(Self {
            bytes: plan.bytes,
            request_bytes: plan.request,
            shared_bytes: plan.shared,
            total_bytes: plan.total,
        })
    }
}

/// A just-acquired charge which may still be rolled back after allocation
/// failure without changing any high-water counter.
#[must_use = "provisional acquisitions must be committed or rolled back"]
#[derive(Debug, PartialEq, Eq)]
pub struct ProvisionalAcquisition {
    reservation: LedgerReservation,
    mutation: u64,
    previous_category_peaks: [u64; LEDGER_CATEGORY_COUNT],
    previous_request_peak: u64,
    previous_shared_peak: u64,
    previous_total_peak: u64,
}

/// An exclusively borrowed provisional acquisition with exact RAII rollback.
///
/// Both the acquire and inverse transition are validated before the acquire
/// is applied. The exclusive ledger borrow prevents an intervening mutation,
/// so dropping an armed permit restores current usage and every prior peak
/// without an allocation or error path.
#[must_use = "provisional ledger permits must be committed or dropped"]
pub(crate) struct ProvisionalLedgerPermit<'ledger> {
    ledger: &'ledger mut CapacityLedger,
    reservation: LedgerReservation,
    armed: bool,
    rollback: Projection,
    rollback_mutation: u64,
    previous_category_peaks: [u64; LEDGER_CATEGORY_COUNT],
    previous_request_peak: u64,
    previous_shared_peak: u64,
    previous_total_peak: u64,
}

impl fmt::Debug for ProvisionalLedgerPermit<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProvisionalLedgerPermit")
            .field("armed", &self.armed)
            .finish()
    }
}

impl ProvisionalLedgerPermit<'_> {
    /// Makes the provisional bytes durable. This changes neither usage nor
    /// high-water accounting and cannot fail while the exclusive borrow is
    /// retained.
    pub(crate) fn commit(mut self) -> LedgerReservation {
        self.armed = false;
        std::mem::replace(&mut self.reservation, LedgerReservation::empty())
    }

    /// Consumes checked partition descriptors only after proving their sum is
    /// exactly the held aggregate reservation.
    pub(crate) fn prove_exact_partitions(
        &self,
        partitions: Vec<ReservationPartition>,
    ) -> Result<ExactReservationPartitions, LedgerError> {
        let mut bytes = [0_u64; LEDGER_CATEGORY_COUNT];
        for partition in &partitions {
            for category in LedgerCategory::ALL {
                let index = category.index();
                bytes[index] = bytes[index].checked_add(partition.bytes[index]).ok_or(
                    LedgerError::ArithmeticOverflow {
                        scope: LedgerScope::Category(category),
                    },
                )?;
            }
        }
        let summary = summarize_bytes(bytes)?;
        if summary.bytes != self.reservation.bytes
            || summary.request != self.reservation.request_bytes
            || summary.shared != self.reservation.shared_bytes
            || summary.total != self.reservation.total_bytes
        {
            return Err(LedgerError::ReservationPartitionMismatch);
        }
        Ok(ExactReservationPartitions {
            partitions,
            next: 0,
        })
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> LedgerSnapshot {
        self.ledger.snapshot()
    }
}

impl Drop for ProvisionalLedgerPermit<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        self.ledger
            .apply_release(&self.rollback, self.rollback_mutation);
        self.ledger.peaks = self.previous_category_peaks;
        self.ledger.request_peak = self.previous_request_peak;
        self.ledger.shared_peak = self.previous_shared_peak;
        self.ledger.total_peak = self.previous_total_peak;
    }
}

/// An exclusively borrowed, prevalidated ledger release.
///
/// Preparing a permit neither consumes the referenced reservations nor
/// changes the ledger. The exclusive ledger borrow prevents another mutation
/// from invalidating the prepared projection. Applying the permit is therefore
/// infallible; dropping it without applying it leaves the ledger unchanged.
#[must_use = "prepared ledger releases must be applied or deliberately abandoned"]
#[derive(Debug)]
pub struct ReleasePermit<'ledger> {
    ledger: &'ledger mut CapacityLedger,
    projection: Projection,
    mutation: u64,
}

impl ReleasePermit<'_> {
    /// Commits the already validated release as one ledger mutation.
    pub fn apply(self) {
        self.ledger.apply_release(&self.projection, self.mutation);
    }
}

/// Fixed-array semantic capacity ledger.
#[derive(Debug)]
pub struct CapacityLedger {
    limits: [u64; LEDGER_CATEGORY_COUNT],
    used: [u64; LEDGER_CATEGORY_COUNT],
    peaks: [u64; LEDGER_CATEGORY_COUNT],
    request_used: u64,
    request_peak: u64,
    shared_used: u64,
    shared_peak: u64,
    total_used: u64,
    total_peak: u64,
    total_limit: u64,
    mutation: u64,
}

impl CapacityLedger {
    /// Constructs an empty ledger without allocating a map or other dynamic
    /// bookkeeping. Category limits and the aggregate limit are independent.
    #[must_use]
    pub const fn new(total_limit: u64, category_limits: [u64; LEDGER_CATEGORY_COUNT]) -> Self {
        Self {
            limits: category_limits,
            used: [0; LEDGER_CATEGORY_COUNT],
            peaks: [0; LEDGER_CATEGORY_COUNT],
            request_used: 0,
            request_peak: 0,
            shared_used: 0,
            shared_peak: 0,
            total_used: 0,
            total_peak: 0,
            total_limit,
            mutation: 0,
        }
    }

    /// Atomically validates and commits a complete plan.
    #[cfg(test)]
    pub fn acquire(&mut self, charges: &[LedgerCharge]) -> Result<LedgerReservation, LedgerError> {
        let plan = aggregate_plan(charges)?;
        let projection = self.project_state(&plan)?;
        let mutation = self.next_mutation()?;
        self.apply_acquire(&projection, mutation);
        Ok(reservation_from_plan(&plan))
    }

    /// Checks the exact category and aggregate fit without changing usage,
    /// peaks, or the mutation identity.
    pub fn can_acquire(&self, charges: &[LedgerCharge]) -> Result<(), LedgerError> {
        let plan = aggregate_plan(charges)?;
        self.project_state(&plan)?;
        self.next_mutation()?;
        Ok(())
    }

    /// Returns the snapshot that a committed acquire would produce without
    /// changing this ledger. FIFO admission uses this to stop at an unfit head
    /// without transient usage or high-water changes.
    #[cfg(test)]
    pub fn projected_snapshot(
        &self,
        charges: &[LedgerCharge],
    ) -> Result<LedgerSnapshot, LedgerError> {
        let plan = aggregate_plan(charges)?;
        let projection = self.project_state(&plan)?;
        self.next_mutation()?;
        Ok(self.snapshot_for_projection(&projection))
    }

    /// Atomically acquires a plan while retaining the exact pre-acquire peaks.
    ///
    /// Callers use this form before a fallible physical allocation, then call
    /// [`Self::commit_provisional`] or [`Self::rollback_provisional`] before
    /// any other ledger mutation.
    pub fn acquire_provisional(
        &mut self,
        charges: &[LedgerCharge],
    ) -> Result<ProvisionalAcquisition, LedgerError> {
        let plan = aggregate_plan(charges)?;
        let projection = self.project_state(&plan)?;
        // Reserve another identity for an exact rollback transition.
        let mutation = self
            .mutation
            .checked_add(1)
            .filter(|mutation| mutation.checked_add(1).is_some())
            .ok_or(LedgerError::MutationIdentityExhausted)?;
        let provisional = ProvisionalAcquisition {
            reservation: reservation_from_plan(&plan),
            mutation,
            previous_category_peaks: self.peaks,
            previous_request_peak: self.request_peak,
            previous_shared_peak: self.shared_peak,
            previous_total_peak: self.total_peak,
        };
        self.apply_acquire(&projection, mutation);
        Ok(provisional)
    }

    /// Acquires one aggregate plan and holds an exact, infallible rollback
    /// transition behind an exclusive borrow.
    pub(crate) fn acquire_provisional_permit(
        &mut self,
        charges: &[LedgerCharge],
    ) -> Result<ProvisionalLedgerPermit<'_>, LedgerError> {
        let plan = aggregate_plan(charges)?;
        let projection = self.project_state(&plan)?;
        let mutation = self
            .mutation
            .checked_add(1)
            .filter(|mutation| mutation.checked_add(1).is_some())
            .ok_or(LedgerError::MutationIdentityExhausted)?;
        let rollback_mutation = mutation + 1;
        let rollback = Projection {
            categories: self.used,
            request: self.request_used,
            shared: self.shared_used,
            total: self.total_used,
        };
        let permit = ProvisionalLedgerPermit {
            reservation: reservation_from_plan(&plan),
            armed: true,
            rollback,
            rollback_mutation,
            previous_category_peaks: self.peaks,
            previous_request_peak: self.request_peak,
            previous_shared_peak: self.shared_peak,
            previous_total_peak: self.total_peak,
            ledger: self,
        };
        permit.ledger.apply_acquire(&projection, mutation);
        Ok(permit)
    }

    /// Makes a provisional charge durable after its physical allocation is
    /// known to have succeeded. This has no byte or high-water delta.
    pub fn commit_provisional(
        &self,
        provisional: ProvisionalAcquisition,
    ) -> Result<LedgerReservation, LedgerError> {
        if self.mutation != provisional.mutation {
            return Err(LedgerError::StaleProvisionalAcquisition);
        }
        Ok(provisional.reservation)
    }

    /// Reverses the most recent provisional acquisition and restores every
    /// high-water counter to its exact pre-acquire value.
    pub fn rollback_provisional(
        &mut self,
        provisional: ProvisionalAcquisition,
    ) -> Result<(), LedgerError> {
        if self.mutation != provisional.mutation {
            return Err(LedgerError::StaleProvisionalAcquisition);
        }
        let summary = summarize_bytes(provisional.reservation.bytes)?;
        let projection = self.project_release(&provisional.reservation.bytes, &summary)?;
        let mutation = self.next_mutation()?;

        self.apply_release(&projection, mutation);
        self.peaks = provisional.previous_category_peaks;
        self.request_peak = provisional.previous_request_peak;
        self.shared_peak = provisional.previous_shared_peak;
        self.total_peak = provisional.previous_total_peak;
        Ok(())
    }

    /// Atomically releases one committed reservation. Peaks are historical and
    /// therefore do not fall on a normal release.
    #[cfg(test)]
    pub fn release(&mut self, reservation: LedgerReservation) -> Result<(), LedgerError> {
        self.prepare_release([&reservation])?.apply();
        Ok(())
    }

    /// Prevalidates one atomic release across borrowed reservations.
    ///
    /// Category bytes are aggregated with checked arithmetic before the
    /// release projection and next mutation identity are validated. Neither
    /// the ledger nor any reservation is changed during preparation. Applying
    /// the permit commits only the ledger transition; the caller remains
    /// responsible for disposing the corresponding reservation owners once.
    pub fn prepare_release<'reservation>(
        &mut self,
        reservations: impl IntoIterator<Item = &'reservation LedgerReservation>,
    ) -> Result<ReleasePermit<'_>, LedgerError> {
        let mut bytes = [0_u64; LEDGER_CATEGORY_COUNT];
        for reservation in reservations {
            for category in LedgerCategory::ALL {
                let index = category.index();
                bytes[index] = bytes[index].checked_add(reservation.bytes[index]).ok_or(
                    LedgerError::ArithmeticOverflow {
                        scope: LedgerScope::Category(category),
                    },
                )?;
            }
        }
        let summary = summarize_bytes(bytes)?;
        let projection = self.project_release(&bytes, &summary)?;
        let mutation = self.next_mutation()?;
        Ok(ReleasePermit {
            ledger: self,
            projection,
            mutation,
        })
    }

    #[must_use]
    pub fn snapshot(&self) -> LedgerSnapshot {
        let categories = std::array::from_fn(|index| CategorySnapshot {
            used: self.used[index],
            peak: self.peaks[index],
            limit: self.limits[index],
        });
        LedgerSnapshot {
            categories,
            request_used: self.request_used,
            request_peak: self.request_peak,
            shared_used: self.shared_used,
            shared_peak: self.shared_peak,
            total_used: self.total_used,
            total_peak: self.total_peak,
            total_limit: self.total_limit,
        }
    }

    fn next_mutation(&self) -> Result<u64, LedgerError> {
        self.mutation
            .checked_add(1)
            .ok_or(LedgerError::MutationIdentityExhausted)
    }

    fn project_state(&self, plan: &PlanSummary) -> Result<Projection, LedgerError> {
        let mut next = self.used;
        for category in LedgerCategory::ALL {
            let index = category.index();
            next[index] = next[index].checked_add(plan.bytes[index]).ok_or(
                LedgerError::ArithmeticOverflow {
                    scope: LedgerScope::Category(category),
                },
            )?;
            if next[index] > self.limits[index] {
                return Err(LedgerError::CapacityExceeded {
                    scope: LedgerScope::Category(category),
                    required: next[index],
                    limit: self.limits[index],
                });
            }
        }

        let request =
            self.request_used
                .checked_add(plan.request)
                .ok_or(LedgerError::ArithmeticOverflow {
                    scope: LedgerScope::RequestOwned,
                })?;
        let shared =
            self.shared_used
                .checked_add(plan.shared)
                .ok_or(LedgerError::ArithmeticOverflow {
                    scope: LedgerScope::Shared,
                })?;
        let total =
            self.total_used
                .checked_add(plan.total)
                .ok_or(LedgerError::ArithmeticOverflow {
                    scope: LedgerScope::Aggregate,
                })?;
        if total > self.total_limit {
            return Err(LedgerError::CapacityExceeded {
                scope: LedgerScope::Aggregate,
                required: total,
                limit: self.total_limit,
            });
        }

        Ok(Projection {
            categories: next,
            request,
            shared,
            total,
        })
    }

    #[cfg(test)]
    fn snapshot_for_projection(&self, projection: &Projection) -> LedgerSnapshot {
        let categories = std::array::from_fn(|index| CategorySnapshot {
            used: projection.categories[index],
            peak: self.peaks[index].max(projection.categories[index]),
            limit: self.limits[index],
        });
        LedgerSnapshot {
            categories,
            request_used: projection.request,
            request_peak: self.request_peak.max(projection.request),
            shared_used: projection.shared,
            shared_peak: self.shared_peak.max(projection.shared),
            total_used: projection.total,
            total_peak: self.total_peak.max(projection.total),
            total_limit: self.total_limit,
        }
    }

    fn project_release(
        &self,
        bytes: &[u64; LEDGER_CATEGORY_COUNT],
        summary: &PlanSummary,
    ) -> Result<Projection, LedgerError> {
        let mut next = self.used;
        for category in LedgerCategory::ALL {
            let index = category.index();
            next[index] =
                next[index]
                    .checked_sub(bytes[index])
                    .ok_or(LedgerError::ReleaseUnderflow {
                        scope: LedgerScope::Category(category),
                        releasing: bytes[index],
                        available: next[index],
                    })?;
        }
        let request = self.request_used.checked_sub(summary.request).ok_or(
            LedgerError::ReleaseUnderflow {
                scope: LedgerScope::RequestOwned,
                releasing: summary.request,
                available: self.request_used,
            },
        )?;
        let shared =
            self.shared_used
                .checked_sub(summary.shared)
                .ok_or(LedgerError::ReleaseUnderflow {
                    scope: LedgerScope::Shared,
                    releasing: summary.shared,
                    available: self.shared_used,
                })?;
        let total =
            self.total_used
                .checked_sub(summary.total)
                .ok_or(LedgerError::ReleaseUnderflow {
                    scope: LedgerScope::Aggregate,
                    releasing: summary.total,
                    available: self.total_used,
                })?;
        Ok(Projection {
            categories: next,
            request,
            shared,
            total,
        })
    }

    fn apply_acquire(&mut self, projection: &Projection, mutation: u64) {
        self.used = projection.categories;
        for category in LedgerCategory::ALL {
            let index = category.index();
            self.peaks[index] = self.peaks[index].max(self.used[index]);
        }
        self.request_used = projection.request;
        self.request_peak = self.request_peak.max(self.request_used);
        self.shared_used = projection.shared;
        self.shared_peak = self.shared_peak.max(self.shared_used);
        self.total_used = projection.total;
        self.total_peak = self.total_peak.max(self.total_used);
        self.mutation = mutation;
    }

    fn apply_release(&mut self, projection: &Projection, mutation: u64) {
        self.used = projection.categories;
        self.request_used = projection.request;
        self.shared_used = projection.shared;
        self.total_used = projection.total;
        self.mutation = mutation;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlanSummary {
    bytes: [u64; LEDGER_CATEGORY_COUNT],
    request: u64,
    shared: u64,
    total: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Projection {
    categories: [u64; LEDGER_CATEGORY_COUNT],
    request: u64,
    shared: u64,
    total: u64,
}

fn aggregate_plan(charges: &[LedgerCharge]) -> Result<PlanSummary, LedgerError> {
    let mut bytes = [0_u64; LEDGER_CATEGORY_COUNT];
    for charge in charges {
        let index = charge.category.index();
        bytes[index] =
            bytes[index]
                .checked_add(charge.bytes)
                .ok_or(LedgerError::ArithmeticOverflow {
                    scope: LedgerScope::Category(charge.category),
                })?;
    }
    summarize_bytes(bytes)
}

fn reservation_from_plan(plan: &PlanSummary) -> LedgerReservation {
    LedgerReservation {
        bytes: plan.bytes,
        request_bytes: plan.request,
        shared_bytes: plan.shared,
        total_bytes: plan.total,
    }
}

fn summarize_bytes(bytes: [u64; LEDGER_CATEGORY_COUNT]) -> Result<PlanSummary, LedgerError> {
    let mut request = 0_u64;
    let mut shared = 0_u64;
    for category in LedgerCategory::ALL {
        let value = bytes[category.index()];
        let (sum, scope) = match category.ownership() {
            LedgerOwnership::Request => (&mut request, LedgerScope::RequestOwned),
            LedgerOwnership::Shared => (&mut shared, LedgerScope::Shared),
        };
        *sum = sum
            .checked_add(value)
            .ok_or(LedgerError::ArithmeticOverflow { scope })?;
    }
    let total = request
        .checked_add(shared)
        .ok_or(LedgerError::ArithmeticOverflow {
            scope: LedgerScope::Aggregate,
        })?;
    Ok(PlanSummary {
        bytes,
        request,
        shared,
        total,
    })
}

fn round_up(bytes: u64) -> Option<u64> {
    let remainder = bytes % LEDGER_QUANTUM_BYTES;
    if remainder == 0 {
        Some(bytes)
    } else {
        bytes.checked_add(LEDGER_QUANTUM_BYTES - remainder)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CapacityLedger, LEDGER_CATEGORY_COUNT, LEDGER_QUANTUM_BYTES, LedgerCategory, LedgerCharge,
        LedgerError, LedgerOwnership, LedgerReservation, LedgerScope, ReservationPartition,
    };

    const MAX_ALIGNED: u64 = u64::MAX - (LEDGER_QUANTUM_BYTES - 1);

    fn unlimited_categories() -> [u64; LEDGER_CATEGORY_COUNT] {
        [u64::MAX; LEDGER_CATEGORY_COUNT]
    }

    fn charge(category: LedgerCategory, bytes: u64) -> LedgerCharge {
        LedgerCharge::from_payload(category, bytes).expect("valid test charge")
    }

    #[test]
    fn payload_rounding_is_exact_and_checked() {
        let category = LedgerCategory::PromptStorage;
        for (payload, expected) in [
            (0, 0),
            (1, 64),
            (63, 64),
            (64, 64),
            (65, 128),
            (MAX_ALIGNED, MAX_ALIGNED),
        ] {
            assert_eq!(charge(category, payload).bytes(), expected);
        }
        assert_eq!(
            LedgerCharge::from_payload(category, MAX_ALIGNED + 1),
            Err(LedgerError::ChargeRoundingOverflow { category })
        );
        assert_eq!(
            LedgerCharge::from_aligned(category, 65),
            Err(LedgerError::UnalignedCharge { category })
        );
    }

    #[test]
    fn exact_fit_succeeds_and_one_byte_short_limits_do_not_mutate() {
        let category = LedgerCategory::PromptStorage;
        let mut exact_limits = unlimited_categories();
        exact_limits[category.index()] = 64;
        let mut exact = CapacityLedger::new(64, exact_limits);
        let reservation = exact.acquire(&[charge(category, 1)]).unwrap();
        assert_eq!(exact.snapshot().category(category).used(), 64);
        exact.release(reservation).unwrap();

        let mut category_short_limits = unlimited_categories();
        category_short_limits[category.index()] = 63;
        let mut category_short = CapacityLedger::new(64, category_short_limits);
        let before = category_short.snapshot();
        assert_eq!(
            category_short.acquire(&[charge(category, 1)]),
            Err(LedgerError::CapacityExceeded {
                scope: LedgerScope::Category(category),
                required: 64,
                limit: 63,
            })
        );
        assert_eq!(category_short.snapshot(), before);

        let mut aggregate_short = CapacityLedger::new(63, unlimited_categories());
        let before = aggregate_short.snapshot();
        assert_eq!(
            aggregate_short.acquire(&[charge(category, 1)]),
            Err(LedgerError::CapacityExceeded {
                scope: LedgerScope::Aggregate,
                required: 64,
                limit: 63,
            })
        );
        assert_eq!(aggregate_short.snapshot(), before);
    }

    #[test]
    fn duplicate_entries_aggregate_before_any_mutation() {
        let mut ledger = CapacityLedger::new(256, unlimited_categories());
        let charges = [
            LedgerCharge::from_aligned(LedgerCategory::PromptStorage, 64).unwrap(),
            LedgerCharge::from_aligned(LedgerCategory::PromptStorage, 128).unwrap(),
            LedgerCharge::from_aligned(LedgerCategory::WorkerScratch, 64).unwrap(),
        ];
        let reservation = ledger.acquire(&charges).unwrap();
        assert_eq!(reservation.bytes(LedgerCategory::PromptStorage), 192);
        assert_eq!(reservation.bytes(LedgerCategory::WorkerScratch), 64);
        assert_eq!(reservation.request_bytes(), 192);
        assert_eq!(reservation.shared_bytes(), 64);
        assert_eq!(reservation.total_bytes(), 256);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.request_used(), 192);
        assert_eq!(snapshot.shared_used(), 64);
        assert_eq!(snapshot.total_used(), 256);
        ledger.release(reservation).unwrap();
        assert!(ledger.snapshot().current_is_zero());
    }

    #[test]
    fn reservation_splits_conserve_categories_ownership_and_ledger_state() {
        let mut ledger = CapacityLedger::new(512, unlimited_categories());
        let reservation = ledger
            .acquire(&[
                charge(LedgerCategory::PromptStorage, 1),
                charge(LedgerCategory::RequestRecord, 64),
                charge(LedgerCategory::ActiveState, 65),
                charge(LedgerCategory::PendingTransaction, 64),
                charge(LedgerCategory::WorkerScratch, 1),
            ])
            .unwrap();
        let before_split = ledger.snapshot();
        let original_total = reservation.total_bytes();
        let (prompt, retained) = reservation
            .split_categories(&[LedgerCategory::PromptStorage, LedgerCategory::PromptStorage]);
        assert_eq!(ledger.snapshot(), before_split);
        assert_eq!(prompt.bytes(LedgerCategory::PromptStorage), 64);
        assert_eq!(prompt.request_bytes(), 64);
        assert_eq!(prompt.shared_bytes(), 0);
        assert_eq!(
            prompt.total_bytes() + retained.total_bytes(),
            original_total
        );
        for category in LedgerCategory::ALL {
            assert_eq!(
                prompt.bytes(category) + retained.bytes(category),
                before_split.category(category).used()
            );
        }

        let (active, retained) = retained.split_categories(&[
            LedgerCategory::ActiveState,
            LedgerCategory::PendingTransaction,
        ]);
        assert_eq!(active.total_bytes(), 192);
        assert_eq!(active.request_bytes(), 192);
        assert_eq!(ledger.snapshot(), before_split);

        let (retained_request, shared) = retained.split_by_ownership();
        assert_eq!(retained_request.shared_bytes(), 0);
        assert_eq!(shared.request_bytes(), 0);
        assert_eq!(retained_request.total_bytes(), 64);
        assert_eq!(shared.total_bytes(), 64);
        assert_eq!(ledger.snapshot(), before_split);

        ledger.release(prompt).unwrap();
        ledger.release(active).unwrap();
        ledger.release(retained_request).unwrap();
        ledger.release(shared).unwrap();
        assert!(ledger.snapshot().current_is_zero());
        assert_eq!(ledger.snapshot().total_peak(), original_total);
    }

    #[test]
    fn category_and_aggregate_arithmetic_overflow_are_atomic() {
        let huge = LedgerCharge::from_aligned(LedgerCategory::PromptStorage, MAX_ALIGNED).unwrap();
        let one = LedgerCharge::from_aligned(LedgerCategory::PromptStorage, 64).unwrap();
        let mut ledger = CapacityLedger::new(u64::MAX, unlimited_categories());
        let before = ledger.snapshot();
        assert_eq!(
            ledger.acquire(&[huge, one]),
            Err(LedgerError::ArithmeticOverflow {
                scope: LedgerScope::Category(LedgerCategory::PromptStorage),
            })
        );
        assert_eq!(ledger.snapshot(), before);

        let shared = LedgerCharge::from_aligned(LedgerCategory::WorkerScratch, 64).unwrap();
        assert_eq!(
            ledger.acquire(&[huge, shared]),
            Err(LedgerError::ArithmeticOverflow {
                scope: LedgerScope::Aggregate,
            })
        );
        assert_eq!(ledger.snapshot(), before);
    }

    #[test]
    fn failed_multi_category_acquire_preserves_usage_and_peaks() {
        let mut limits = unlimited_categories();
        limits[LedgerCategory::Output.index()] = 64;
        let mut ledger = CapacityLedger::new(512, limits);
        let retained = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let before = ledger.snapshot();
        let error = ledger
            .acquire(&[
                charge(LedgerCategory::ActiveState, 64),
                charge(LedgerCategory::Output, 65),
            ])
            .unwrap_err();
        assert_eq!(
            error,
            LedgerError::CapacityExceeded {
                scope: LedgerScope::Category(LedgerCategory::Output),
                required: 128,
                limit: 64,
            }
        );
        assert_eq!(ledger.snapshot(), before);
        ledger.release(retained).unwrap();
    }

    #[test]
    fn fit_projection_is_exact_and_mutation_free() {
        let mut ledger = CapacityLedger::new(256, unlimited_categories());
        let retained = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let before = ledger.snapshot();
        let plan = [
            charge(LedgerCategory::ActiveState, 65),
            charge(LedgerCategory::WorkerScratch, 1),
        ];
        assert_eq!(ledger.can_acquire(&plan), Ok(()));
        let projected = ledger.projected_snapshot(&plan).unwrap();
        assert_eq!(projected.request_used(), 192);
        assert_eq!(projected.shared_used(), 64);
        assert_eq!(projected.total_used(), 256);
        assert_eq!(projected.total_peak(), 256);
        assert_eq!(ledger.snapshot(), before);

        let unfit = [charge(LedgerCategory::Output, 257)];
        assert!(matches!(
            ledger.can_acquire(&unfit),
            Err(LedgerError::CapacityExceeded {
                scope: LedgerScope::Aggregate,
                ..
            })
        ));
        assert_eq!(ledger.snapshot(), before);
        ledger.release(retained).unwrap();
    }

    #[test]
    fn provisional_rollback_restores_usage_and_high_water_exactly() {
        let mut ledger = CapacityLedger::new(512, unlimited_categories());
        let retained = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let before = ledger.snapshot();
        let provisional = ledger
            .acquire_provisional(&[
                charge(LedgerCategory::ActiveState, 128),
                charge(LedgerCategory::WorkerScratch, 64),
            ])
            .unwrap();
        assert_eq!(ledger.snapshot().total_used(), 256);
        assert_eq!(ledger.snapshot().total_peak(), 256);
        ledger.rollback_provisional(provisional).unwrap();
        assert_eq!(ledger.snapshot(), before);
        ledger.release(retained).unwrap();
    }

    #[test]
    fn provisional_commit_preserves_peak_and_releases_normally() {
        let mut ledger = CapacityLedger::new(128, unlimited_categories());
        let provisional = ledger
            .acquire_provisional(&[charge(LedgerCategory::SamplingScratch, 65)])
            .unwrap();
        let reservation = ledger.commit_provisional(provisional).unwrap();
        ledger.release(reservation).unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.total_used(), 0);
        assert_eq!(snapshot.total_peak(), 128);
        assert_eq!(
            snapshot.category(LedgerCategory::SamplingScratch).peak(),
            128
        );
    }

    #[test]
    fn exclusive_provisional_permit_drop_restores_every_current_and_peak_value() {
        let mut ledger = CapacityLedger::new(512, unlimited_categories());
        let retained = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let before = ledger.snapshot();
        {
            let permit = ledger
                .acquire_provisional_permit(&[
                    charge(LedgerCategory::RequestRecord, 128),
                    charge(LedgerCategory::Output, 64),
                ])
                .unwrap();
            assert_eq!(permit.snapshot().total_used(), before.total_used() + 192);
            assert_eq!(permit.snapshot().total_peak(), before.total_used() + 192);
        }
        assert_eq!(ledger.snapshot(), before);
        ledger.release(retained).unwrap();
    }

    #[test]
    fn exact_partition_proof_conserves_one_aggregate_reservation() {
        let mut ledger = CapacityLedger::new(1024, unlimited_categories());
        let first = [
            charge(LedgerCategory::PromptStorage, 64),
            charge(LedgerCategory::Output, 128),
        ];
        let second = [
            charge(LedgerCategory::PromptStorage, 128),
            charge(LedgerCategory::Terminal, 64),
        ];
        let aggregate = [first[0], first[1], second[0], second[1]];
        let permit = ledger.acquire_provisional_permit(&aggregate).unwrap();
        let mut proof = permit
            .prove_exact_partitions(vec![
                ReservationPartition::from_charges(&first).unwrap(),
                ReservationPartition::from_charges(&second).unwrap(),
            ])
            .unwrap();
        let aggregate = permit.commit();
        let (first_reservation, remainder) = proof.split_next(aggregate);
        let (second_reservation, remainder) = proof.split_next(remainder);
        assert!(proof.is_complete());
        assert!(remainder.is_empty());
        assert_eq!(first_reservation.total_bytes(), 192);
        assert_eq!(second_reservation.total_bytes(), 192);
        ledger.release(first_reservation).unwrap();
        ledger.release(second_reservation).unwrap();
        assert_eq!(ledger.snapshot().total_used(), 0);
    }

    #[test]
    fn later_smaller_use_does_not_lower_any_peak() {
        let mut ledger = CapacityLedger::new(512, unlimited_categories());
        let first = ledger
            .acquire(&[
                charge(LedgerCategory::PromptStorage, 128),
                charge(LedgerCategory::WorkerScratch, 128),
            ])
            .unwrap();
        ledger.release(first).unwrap();
        let second = ledger
            .acquire(&[
                charge(LedgerCategory::PromptStorage, 64),
                charge(LedgerCategory::WorkerScratch, 64),
            ])
            .unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.request_peak(), 128);
        assert_eq!(snapshot.shared_peak(), 128);
        assert_eq!(snapshot.total_peak(), 256);
        assert_eq!(snapshot.category(LedgerCategory::PromptStorage).peak(), 128);
        ledger.release(second).unwrap();
        assert_eq!(ledger.snapshot().total_peak(), 256);
    }

    #[test]
    fn release_underflow_is_prevalidated_without_partial_mutation() {
        let mut ledger = CapacityLedger::new(256, unlimited_categories());
        let retained = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let before = ledger.snapshot();
        let invalid = LedgerReservation {
            bytes: {
                let mut bytes = [0; LEDGER_CATEGORY_COUNT];
                bytes[LedgerCategory::PromptStorage.index()] = 128;
                bytes[LedgerCategory::WorkerScratch.index()] = 64;
                bytes
            },
            request_bytes: 128,
            shared_bytes: 64,
            total_bytes: 192,
        };
        assert_eq!(
            ledger.release(invalid),
            Err(LedgerError::ReleaseUnderflow {
                scope: LedgerScope::Category(LedgerCategory::PromptStorage),
                releasing: 128,
                available: 64,
            })
        );
        assert_eq!(ledger.snapshot(), before);
        ledger.release(retained).unwrap();
    }

    #[test]
    fn multi_reservation_release_permit_applies_as_one_mutation() {
        let mut ledger = CapacityLedger::new(256, unlimited_categories());
        let request = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let shared = ledger
            .acquire(&[charge(LedgerCategory::WorkerScratch, 128)])
            .unwrap();
        let before = ledger.snapshot();
        let mutation_before = ledger.mutation;

        ledger.prepare_release([&request, &shared]).unwrap().apply();
        drop((request, shared));

        let after = ledger.snapshot();
        assert!(after.current_is_zero());
        assert_eq!(after.request_peak(), before.request_peak());
        assert_eq!(after.shared_peak(), before.shared_peak());
        assert_eq!(after.total_peak(), before.total_peak());
        assert_eq!(ledger.mutation, mutation_before + 1);
    }

    #[test]
    fn failed_release_prepare_preserves_owner_and_snapshot() {
        let mut ledger = CapacityLedger::new(64, unlimited_categories());
        let reservation = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let before = ledger.snapshot();
        let mutation_before = ledger.mutation;

        assert_eq!(
            ledger
                .prepare_release([&reservation, &reservation])
                .unwrap_err(),
            LedgerError::ReleaseUnderflow {
                scope: LedgerScope::Category(LedgerCategory::PromptStorage),
                releasing: 128,
                available: 64,
            }
        );
        assert_eq!(ledger.snapshot(), before);
        assert_eq!(ledger.mutation, mutation_before);
        assert_eq!(reservation.total_bytes(), 64);

        ledger.mutation = u64::MAX;
        assert_eq!(
            ledger.prepare_release([&reservation]).unwrap_err(),
            LedgerError::MutationIdentityExhausted
        );
        assert_eq!(ledger.snapshot(), before);
        assert_eq!(reservation.total_bytes(), 64);

        ledger.mutation = mutation_before;
        ledger.release(reservation).unwrap();
        assert!(ledger.snapshot().current_is_zero());
    }

    #[test]
    fn release_prepare_batch_aggregation_overflow_is_atomic() {
        let mut ledger = CapacityLedger::new(u64::MAX, unlimited_categories());
        let reservation = ledger
            .acquire(&[
                LedgerCharge::from_aligned(LedgerCategory::PromptStorage, MAX_ALIGNED).unwrap(),
            ])
            .unwrap();
        let extra = LedgerReservation {
            bytes: {
                let mut bytes = [0; LEDGER_CATEGORY_COUNT];
                bytes[LedgerCategory::PromptStorage.index()] = 64;
                bytes
            },
            request_bytes: 64,
            shared_bytes: 0,
            total_bytes: 64,
        };
        let before = ledger.snapshot();
        let mutation_before = ledger.mutation;

        assert_eq!(
            ledger.prepare_release([&reservation, &extra]).unwrap_err(),
            LedgerError::ArithmeticOverflow {
                scope: LedgerScope::Category(LedgerCategory::PromptStorage),
            }
        );
        assert_eq!(ledger.snapshot(), before);
        assert_eq!(ledger.mutation, mutation_before);
        assert_eq!(reservation.total_bytes(), MAX_ALIGNED);
        assert_eq!(extra.total_bytes(), 64);

        drop(extra);
        ledger.release(reservation).unwrap();
        assert!(ledger.snapshot().current_is_zero());
    }

    #[test]
    fn dropped_release_permit_is_a_no_op() {
        let mut ledger = CapacityLedger::new(64, unlimited_categories());
        let reservation = ledger
            .acquire(&[charge(LedgerCategory::PromptStorage, 64)])
            .unwrap();
        let before = ledger.snapshot();
        let mutation_before = ledger.mutation;

        let permit = ledger.prepare_release([&reservation]).unwrap();
        drop(permit);

        assert_eq!(ledger.snapshot(), before);
        assert_eq!(ledger.mutation, mutation_before);
        ledger.release(reservation).unwrap();
        assert!(ledger.snapshot().current_is_zero());
    }

    #[test]
    fn every_closed_category_is_charged_and_final_usage_is_zero() {
        let mut limits = [0; LEDGER_CATEGORY_COUNT];
        let mut charges = [LedgerCharge {
            category: LedgerCategory::PromptStorage,
            bytes: 0,
        }; LEDGER_CATEGORY_COUNT];
        for category in LedgerCategory::ALL {
            limits[category.index()] = 64;
            charges[category.index()] = charge(category, 1);
        }
        let total = u64::try_from(LEDGER_CATEGORY_COUNT).unwrap() * 64;
        let mut ledger = CapacityLedger::new(total, limits);
        let reservation = ledger.acquire(&charges).unwrap();
        let snapshot = ledger.snapshot();

        let mut request_categories = 0_u64;
        let mut shared_categories = 0_u64;
        for category in LedgerCategory::ALL {
            assert_eq!(snapshot.category(category).used(), 64);
            assert_eq!(snapshot.category(category).peak(), 64);
            assert_eq!(snapshot.category(category).limit(), 64);
            match category.ownership() {
                LedgerOwnership::Request => request_categories += 1,
                LedgerOwnership::Shared => shared_categories += 1,
            }
        }
        assert_eq!(snapshot.request_used(), request_categories * 64);
        assert_eq!(snapshot.shared_used(), shared_categories * 64);
        assert_eq!(snapshot.total_used(), total);
        assert_eq!(snapshot.total_peak(), total);
        assert_eq!(snapshot.total_limit(), total);

        ledger.release(reservation).unwrap();
        let final_snapshot = ledger.snapshot();
        assert!(final_snapshot.current_is_zero());
        for category in LedgerCategory::ALL {
            assert_eq!(final_snapshot.category(category).used(), 0);
            assert_eq!(final_snapshot.category(category).peak(), 64);
        }
    }

    #[test]
    fn empty_plan_is_exactly_zero() {
        let mut ledger = CapacityLedger::new(0, [0; LEDGER_CATEGORY_COUNT]);
        let reservation = ledger.acquire(&[]).unwrap();
        assert_eq!(reservation.total_bytes(), 0);
        ledger.release(reservation).unwrap();
        assert_eq!(ledger.snapshot().total_peak(), 0);
        assert!(ledger.snapshot().current_is_zero());
    }
}
