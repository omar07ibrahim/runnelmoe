//! Allocation-stable, generation-tagged request control signals.
//!
//! Request handles publish cancellation and receiver disconnection without
//! entering the actor command lane. A binding's generation shares one atomic
//! word with its monotone state bits, so a stale handle cannot modify a reused
//! slot between a generation check and a flag update.

use std::{
    fmt,
    mem::size_of,
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use std::sync::Weak;

use crate::{
    error::{SchedulerError, SchedulerResult},
    request::CancelDisposition,
};

const FLAG_BITS: u32 = 3;
const CANCELLED: u64 = 1 << 0;
const DISCONNECTED: u64 = 1 << 1;
const TERMINAL: u64 = 1 << 2;
const FLAG_MASK: u64 = (1 << FLAG_BITS) - 1;

/// Largest generation that can share one `u64` with all control flags.
pub(crate) const MAX_CONTROL_GENERATION: u64 = u64::MAX >> FLAG_BITS;

/// Result of atomically publishing receiver disconnection.
#[allow(dead_code, reason = "used by the staged Tokio request handle")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DisconnectDisposition {
    Requested,
    AlreadyRequested,
    AlreadyTerminal,
}

/// Exact final-iteration observation for one instrumented control mutation.
///
/// This API exists only for the bounded actor stress harness. Its raw fields
/// are deliberately omitted from `Debug`; production code must use the
/// ordinary cancellation and disconnection methods instead.
#[cfg(any(test, feature = "actor-stress-instrumentation"))]
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ActorControlCasWitness {
    boundary_reached: bool,
    slot_index: usize,
    expected_generation: u64,
    loaded_word: u64,
    resulting_word: u64,
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl ActorControlCasWitness {
    #[allow(dead_code, reason = "used for no-call actor stress actions")]
    pub(crate) const fn sentinel() -> Self {
        Self {
            boundary_reached: false,
            slot_index: 0,
            expected_generation: 0,
            loaded_word: 0,
            resulting_word: 0,
        }
    }

    const fn observed(
        slot_index: usize,
        expected_generation: u64,
        loaded_word: u64,
        resulting_word: u64,
    ) -> Self {
        Self {
            boundary_reached: true,
            slot_index,
            expected_generation,
            loaded_word,
            resulting_word,
        }
    }

    #[must_use]
    pub const fn boundary_reached(self) -> bool {
        self.boundary_reached
    }

    #[must_use]
    pub const fn slot_index(self) -> usize {
        self.slot_index
    }

    #[must_use]
    pub const fn expected_generation(self) -> u64 {
        self.expected_generation
    }

    #[must_use]
    pub const fn loaded_word(self) -> u64 {
        self.loaded_word
    }

    #[must_use]
    pub const fn resulting_word(self) -> u64 {
        self.resulting_word
    }
}

#[cfg(any(test, feature = "actor-stress-instrumentation"))]
impl fmt::Debug for ActorControlCasWitness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActorControlCasWitness")
            .field("boundary_reached", &self.boundary_reached)
            .field("identity", &"<redacted>")
            .field("packed_words", &"<redacted>")
            .finish()
    }
}

/// Result of making terminal state visible to request handles.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MarkTerminalDisposition {
    Marked,
    AlreadyTerminal,
}

/// One freshly acquired view of a request's control state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ControlSnapshot {
    cancelled: bool,
    disconnected: bool,
    terminal: bool,
}

impl ControlSnapshot {
    pub(crate) const fn cancelled(self) -> bool {
        self.cancelled
    }

    pub(crate) const fn disconnected(self) -> bool {
        self.disconnected
    }

    pub(crate) const fn terminal(self) -> bool {
        self.terminal
    }
}

struct ControlTable {
    words: Box<[AtomicU64]>,
}

#[derive(Clone)]
#[cfg_attr(
    not(any(
        test,
        feature = "deterministic-checkpoint-instrumentation",
        feature = "m5-run-observer-instrumentation"
    )),
    allow(
        dead_code,
        reason = "weak engine-domain binding is used only by sealed instrumentation"
    )
)]
pub(crate) struct ControlDomain {
    table: Weak<ControlTable>,
}

impl ControlDomain {
    #[cfg_attr(
        not(any(
            test,
            feature = "deterministic-checkpoint-instrumentation",
            feature = "m5-run-observer-instrumentation"
        )),
        allow(
            dead_code,
            reason = "weak engine-domain binding is used only by sealed instrumentation"
        )
    )]
    pub(crate) fn same_table(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.table, &other.table)
    }
}

impl fmt::Debug for ControlDomain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlDomain")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl fmt::Debug for ControlTable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlTable")
            .field("slot_count", &self.words.len())
            .finish_non_exhaustive()
    }
}

/// Nonmutating preflight for a later allocation-free binding publication.
///
/// Dropping a prepared value has no effect. Its fields deliberately stay
/// private so a caller cannot manufacture a slot or generation.
pub(crate) struct PreparedControl {
    table: Arc<ControlTable>,
    index: usize,
    free_position: usize,
    previous_generation: u64,
    generation: NonZeroU64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ControlPressure {
    resource: &'static str,
    required: u64,
    limit: u64,
}

impl ControlPressure {
    pub(crate) const fn error(self) -> SchedulerError {
        SchedulerError::resource_exhausted(self.resource, self.required, self.limit)
    }
}

pub(crate) struct AvailableControls {
    prepared: Vec<PreparedControl>,
    pressure: Option<ControlPressure>,
}

impl AvailableControls {
    pub(crate) fn len(&self) -> usize {
        self.prepared.len()
    }

    pub(crate) const fn pressure(&self) -> Option<ControlPressure> {
        self.pressure
    }

    pub(crate) fn into_prefix(mut self, count: usize) -> Vec<PreparedControl> {
        debug_assert!(count <= self.prepared.len());
        self.prepared.truncate(count);
        self.prepared
    }
}

struct PreparedBatchControl {
    prepared: PreparedControl,
    published: u64,
}

pub(crate) const fn available_batch_control_bytes(maximum_offers: usize) -> Option<usize> {
    maximum_offers.checked_mul(size_of::<PreparedControl>())
}

pub(crate) fn batch_control_permit_bytes(maximum_accepted: usize) -> Option<usize> {
    maximum_accepted
        .checked_mul(size_of::<PreparedBatchControl>())?
        .checked_add(maximum_accepted.checked_mul(size_of::<ControlBinding>())?)
}

/// Exclusive, fully validated control bindings for an atomic batch publish.
/// Dropping the permit leaves the registry unchanged.
pub(crate) struct ControlBatchBindPermit<'registry> {
    registry: &'registry mut ControlRegistry,
    prepared: Vec<PreparedBatchControl>,
    bindings: Vec<ControlBinding>,
}

impl fmt::Debug for ControlBatchBindPermit<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlBatchBindPermit")
            .field("count", &self.prepared.len())
            .finish()
    }
}

impl ControlBatchBindPermit<'_> {
    /// Publishes every prevalidated control in prepared order without an
    /// allocation or error path.
    pub(crate) fn commit(mut self) -> Vec<ControlBinding> {
        for entry in self.prepared.drain(..) {
            let prepared = entry.prepared;
            let word = &self.registry.table.words[prepared.index];
            loop {
                let observed = word.load(Ordering::Acquire);
                debug_assert_eq!(generation_from_word(observed), prepared.previous_generation);
                debug_assert!(prepared.previous_generation == 0 || has_flag(observed, TERMINAL));
                if word
                    .compare_exchange(
                        observed,
                        entry.published,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    break;
                }
            }
            self.bindings.push(ControlBinding {
                table: prepared.table,
                index: prepared.index,
                generation: prepared.generation,
            });
            debug_assert_eq!(
                self.registry.free_slots.get(prepared.free_position),
                Some(&prepared.index)
            );
            self.registry.free_slots.swap_remove(prepared.free_position);
        }
        self.bindings
    }
}

impl fmt::Debug for PreparedControl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedControl")
            .field("identity", &"<redacted>")
            .finish()
    }
}

/// Cloneable, allocation-free handle to one generation of one control slot.
///
/// Cloning an `Arc` increments its existing control block and does not allocate
/// a per-request object. Every operation validates the packed generation in
/// the same atomic operation loop that changes its flags.
#[derive(Clone)]
pub(crate) struct ControlBinding {
    table: Arc<ControlTable>,
    index: usize,
    generation: NonZeroU64,
}

impl fmt::Debug for ControlBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlBinding")
            .field("identity", &"<redacted>")
            .finish()
    }
}

impl ControlBinding {
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) const fn stress_identity(&self) -> (usize, u64) {
        (self.index, self.generation.get())
    }

    /// Loads a new control snapshot with acquire ordering.
    ///
    /// A generation mismatch reports the scheduler's generic stale-request
    /// error and never reveals either generation or slot index.
    pub(crate) fn fresh_snapshot(&self) -> SchedulerResult<ControlSnapshot> {
        let word = self.word()?.load(Ordering::Acquire);
        self.validate_generation(word)?;
        Ok(snapshot_from_word(word))
    }

    /// Loads flags after the owning engine has validated this generation.
    ///
    /// The live request record excludes recycle/rebind until the surrounding
    /// composite operation finishes. This leaves only monotone flag races, so
    /// the final boundary can remain allocation-free and infallible.
    pub(crate) fn fresh_snapshot_prevalidated(&self) -> ControlSnapshot {
        let word = &self.table.words[self.index];
        let observed = word.load(Ordering::Acquire);
        debug_assert_eq!(
            generation_from_word(observed),
            self.generation.get(),
            "prevalidated request control generation changed"
        );
        snapshot_from_word(observed)
    }

    /// Publishes cancellation with one generation-checked atomic CAS.
    ///
    /// Within one generation only three bits can be added. A failed strong CAS
    /// therefore either makes this request satisfied/terminal/stale or can be
    /// retried after another monotone flag transition. The operation never
    /// consults a bounded command queue.
    pub(crate) fn cancel(&self) -> SchedulerResult<CancelDisposition> {
        let word = self.word()?;
        loop {
            let observed = word.load(Ordering::Acquire);
            self.validate_generation(observed)?;
            if has_flag(observed, TERMINAL) {
                return Ok(CancelDisposition::AlreadyTerminal);
            }
            if has_flag(observed, CANCELLED) {
                return Ok(CancelDisposition::AlreadyRequested);
            }
            if word
                .compare_exchange(
                    observed,
                    observed | CANCELLED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(CancelDisposition::Requested);
            }
        }
    }

    /// Performs cancellation and returns the exact final load/CAS iteration.
    ///
    /// Unlike a success-only tuple, this preserves the word loaded by a stale
    /// generation immediately before the typed invalid-request result.
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    #[allow(dead_code, reason = "called by the feature-gated actor stress wrapper")]
    pub(crate) fn cancel_with_stress_witness(
        &self,
    ) -> (SchedulerResult<CancelDisposition>, ActorControlCasWitness) {
        let expected_generation = self.generation.get();
        let word = match self.word() {
            Ok(word) => word,
            Err(error) => {
                return (Err(error), ActorControlCasWitness::sentinel());
            }
        };
        loop {
            let observed = word.load(Ordering::Acquire);
            let witness = ActorControlCasWitness::observed(
                self.index,
                expected_generation,
                observed,
                observed,
            );
            if let Err(error) = self.validate_generation(observed) {
                return (Err(error), witness);
            }
            if has_flag(observed, TERMINAL) {
                return (Ok(CancelDisposition::AlreadyTerminal), witness);
            }
            if has_flag(observed, CANCELLED) {
                return (Ok(CancelDisposition::AlreadyRequested), witness);
            }
            let resulting = observed | CANCELLED;
            if word
                .compare_exchange(observed, resulting, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return (
                    Ok(CancelDisposition::Requested),
                    ActorControlCasWitness::observed(
                        self.index,
                        expected_generation,
                        observed,
                        resulting,
                    ),
                );
            }
        }
    }

    /// Atomically makes disconnection and its implied cancellation visible.
    ///
    /// Disconnection is recorded even after terminal publication so an owner
    /// can still discard already committed, undrained output. The disposition
    /// remains `AlreadyTerminal` in that case.
    #[allow(dead_code, reason = "used by the staged Tokio request handle")]
    pub(crate) fn disconnect(&self) -> SchedulerResult<DisconnectDisposition> {
        let word = self.word()?;
        loop {
            let observed = word.load(Ordering::Acquire);
            self.validate_generation(observed)?;
            let terminal = has_flag(observed, TERMINAL);
            if has_flag(observed, DISCONNECTED) {
                return Ok(if terminal {
                    DisconnectDisposition::AlreadyTerminal
                } else {
                    DisconnectDisposition::AlreadyRequested
                });
            }
            if word
                .compare_exchange(
                    observed,
                    observed | CANCELLED | DISCONNECTED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(if terminal {
                    DisconnectDisposition::AlreadyTerminal
                } else {
                    DisconnectDisposition::Requested
                });
            }
        }
    }

    /// Performs disconnection and returns the exact final load/CAS iteration.
    #[cfg(any(test, feature = "actor-stress-instrumentation"))]
    pub(crate) fn disconnect_with_stress_witness(
        &self,
    ) -> (
        SchedulerResult<DisconnectDisposition>,
        ActorControlCasWitness,
    ) {
        let expected_generation = self.generation.get();
        let word = match self.word() {
            Ok(word) => word,
            Err(error) => {
                return (Err(error), ActorControlCasWitness::sentinel());
            }
        };
        loop {
            let observed = word.load(Ordering::Acquire);
            let witness = ActorControlCasWitness::observed(
                self.index,
                expected_generation,
                observed,
                observed,
            );
            if let Err(error) = self.validate_generation(observed) {
                return (Err(error), witness);
            }
            let terminal = has_flag(observed, TERMINAL);
            if has_flag(observed, DISCONNECTED) {
                return (
                    Ok(if terminal {
                        DisconnectDisposition::AlreadyTerminal
                    } else {
                        DisconnectDisposition::AlreadyRequested
                    }),
                    witness,
                );
            }
            let resulting = observed | CANCELLED | DISCONNECTED;
            if word
                .compare_exchange(observed, resulting, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return (
                    Ok(if terminal {
                        DisconnectDisposition::AlreadyTerminal
                    } else {
                        DisconnectDisposition::Requested
                    }),
                    ActorControlCasWitness::observed(
                        self.index,
                        expected_generation,
                        observed,
                        resulting,
                    ),
                );
            }
        }
    }

    /// Publishes terminal visibility while preserving concurrent control bits.
    #[cfg(test)]
    pub(crate) fn mark_terminal(&self) -> SchedulerResult<MarkTerminalDisposition> {
        let word = self.word()?;
        loop {
            let observed = word.load(Ordering::Acquire);
            self.validate_generation(observed)?;
            if has_flag(observed, TERMINAL) {
                return Ok(MarkTerminalDisposition::AlreadyTerminal);
            }
            if word
                .compare_exchange(
                    observed,
                    observed | TERMINAL,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(MarkTerminalDisposition::Marked);
            }
        }
    }

    /// Infallibly publishes terminal state after engine-owned prevalidation.
    ///
    /// Before calling this method, the single engine owner must have obtained
    /// a fresh generation-valid snapshot, completed every fallible terminal
    /// transition check, and retained the live request record that excludes
    /// control-slot recycling. Cancellation and disconnection may still race;
    /// `fetch_or` preserves either bit. The table and index are immutable for
    /// the binding's lifetime, so this publication allocates nothing and has
    /// no error path suitable for splitting a composite model-state commit.
    pub(crate) fn mark_terminal_prevalidated(&self) {
        let word = &self.table.words[self.index];
        debug_assert_eq!(
            generation_from_word(word.load(Ordering::Acquire)),
            self.generation.get(),
            "prevalidated request control generation changed"
        );
        word.fetch_or(TERMINAL, Ordering::AcqRel);
    }

    fn word(&self) -> SchedulerResult<&AtomicU64> {
        self.table
            .words
            .get(self.index)
            .ok_or_else(|| SchedulerError::internal("request control slot is out of range"))
    }

    fn validate_generation(&self, word: u64) -> SchedulerResult<()> {
        if generation_from_word(word) == self.generation.get() {
            Ok(())
        } else {
            Err(SchedulerError::request_not_found())
        }
    }
}

/// Single-owner lifecycle registry for the shared atomic control table.
///
/// Construction reserves every slot and free-list entry. Preparing a binding
/// is nonmutating; binding, terminal signaling, recycling, and handle cloning
/// cannot grow either allocation.
pub(crate) struct ControlRegistry {
    table: Arc<ControlTable>,
    free_slots: Vec<usize>,
}

impl fmt::Debug for ControlRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlRegistry")
            .field("capacity", &self.table.words.len())
            .field("available", &self.free_slots.len())
            .finish()
    }
}

impl ControlRegistry {
    pub(crate) fn matches_domain(&self, domain: &ControlDomain) -> bool {
        std::ptr::eq(Arc::as_ptr(&self.table), domain.table.as_ptr())
    }

    #[cfg_attr(
        not(any(
            test,
            feature = "deterministic-checkpoint-instrumentation",
            feature = "m5-run-observer-instrumentation"
        )),
        allow(
            dead_code,
            reason = "weak engine-domain binding is used only by sealed instrumentation"
        )
    )]
    pub(crate) fn checkpoint_domain(&self) -> ControlDomain {
        ControlDomain {
            table: Arc::downgrade(&self.table),
        }
    }

    /// Fallibly preallocates the complete control table and lifecycle free list.
    pub(crate) fn try_with_capacity(capacity: usize) -> SchedulerResult<Self> {
        if capacity == 0 {
            return Err(SchedulerError::invalid_request(
                "request control capacity",
                "must be nonzero",
            ));
        }

        let mut words = Vec::new();
        try_reserve_exact(&mut words, capacity, "request control slots")?;
        words.resize_with(capacity, || AtomicU64::new(0));

        let mut free_slots = Vec::new();
        try_reserve_exact(&mut free_slots, capacity, "request control lifecycle slots")?;
        for index in (0..capacity).rev() {
            free_slots.push(index);
        }

        Ok(Self {
            // `Arc::try_new` is not a stable API. This sole control block is
            // created before the registry can accept any request; all expected
            // capacity failures above are handled fallibly.
            table: Arc::new(ControlTable {
                words: words.into_boxed_slice(),
            }),
            free_slots,
        })
    }

    /// Selects a reusable slot and checks generation availability without
    /// consuming either one. Dropping the result is an exact rollback.
    pub(crate) fn prepare(&self) -> SchedulerResult<PreparedControl> {
        let mut saw_exhausted_generation = false;
        for (free_position, index) in self.free_slots.iter().copied().enumerate().rev() {
            let word = self
                .table
                .words
                .get(index)
                .ok_or_else(|| SchedulerError::internal("free control slot is out of range"))?
                .load(Ordering::Acquire);
            let previous_generation = generation_from_word(word);
            if previous_generation != 0 && !has_flag(word, TERMINAL) {
                return Err(SchedulerError::internal(
                    "free request control slot is not terminal",
                ));
            }
            let Some(next) = previous_generation.checked_add(1) else {
                saw_exhausted_generation = true;
                continue;
            };
            let Some(generation) = NonZeroU64::new(next) else {
                return Err(SchedulerError::internal(
                    "request control generation became zero",
                ));
            };
            if generation.get() > MAX_CONTROL_GENERATION {
                saw_exhausted_generation = true;
                continue;
            }
            return Ok(PreparedControl {
                table: Arc::clone(&self.table),
                index,
                free_position,
                previous_generation,
                generation,
            });
        }

        if saw_exhausted_generation {
            Err(SchedulerError::resource_exhausted(
                "request control generation",
                MAX_CONTROL_GENERATION + 1,
                MAX_CONTROL_GENERATION,
            ))
        } else {
            Err(SchedulerError::resource_exhausted(
                "request control slots",
                usize_to_u64(self.table.words.len()).saturating_add(1),
                usize_to_u64(self.table.words.len()),
            ))
        }
    }

    /// Prepares distinct viable controls up to `maximum` without mutating the
    /// registry. A short result carries the exact ordinary pressure class for
    /// the first unpreparable FIFO offer.
    pub(crate) fn prepare_available(&self, maximum: usize) -> SchedulerResult<AvailableControls> {
        let mut prepared = Vec::new();
        try_reserve_exact(&mut prepared, maximum, "prepared batch request controls")?;
        let mut saw_exhausted_generation = false;
        for (free_position, index) in self.free_slots.iter().copied().enumerate().rev() {
            if prepared.len() == maximum {
                break;
            }
            let word = self
                .table
                .words
                .get(index)
                .ok_or_else(|| SchedulerError::internal("free control slot is out of range"))?
                .load(Ordering::Acquire);
            let previous_generation = generation_from_word(word);
            if previous_generation != 0 && !has_flag(word, TERMINAL) {
                return Err(SchedulerError::internal(
                    "free request control slot is not terminal",
                ));
            }
            let Some(next) = previous_generation.checked_add(1) else {
                saw_exhausted_generation = true;
                continue;
            };
            let Some(generation) = NonZeroU64::new(next) else {
                return Err(SchedulerError::internal(
                    "request control generation became zero",
                ));
            };
            if generation.get() > MAX_CONTROL_GENERATION {
                saw_exhausted_generation = true;
                continue;
            }
            prepared.push(PreparedControl {
                table: Arc::clone(&self.table),
                index,
                free_position,
                previous_generation,
                generation,
            });
        }
        let pressure = (prepared.len() < maximum).then(|| {
            if saw_exhausted_generation {
                ControlPressure {
                    resource: "request control generation",
                    required: MAX_CONTROL_GENERATION + 1,
                    limit: MAX_CONTROL_GENERATION,
                }
            } else {
                ControlPressure {
                    resource: "request control slots",
                    required: usize_to_u64(self.table.words.len()).saturating_add(1),
                    limit: usize_to_u64(self.table.words.len()),
                }
            }
        });
        Ok(AvailableControls { prepared, pressure })
    }

    /// Validates a distinct prepared prefix and retains exclusive registry
    /// ownership until it is either dropped or published.
    pub(crate) fn begin_bind_batch(
        &mut self,
        prepared: Vec<PreparedControl>,
    ) -> SchedulerResult<ControlBatchBindPermit<'_>> {
        let count = prepared.len();
        let mut entries = Vec::new();
        try_reserve_exact(&mut entries, count, "prepared batch control bindings")?;
        let mut bindings = Vec::new();
        try_reserve_exact(&mut bindings, count, "committed batch control bindings")?;
        let mut previous_free_position = self.free_slots.len();
        for control in prepared {
            if !Arc::ptr_eq(&self.table, &control.table) {
                return Err(SchedulerError::internal(
                    "prepared request control belongs to another registry",
                ));
            }
            if control.free_position >= previous_free_position {
                return Err(SchedulerError::internal(
                    "prepared batch request control order is inconsistent",
                ));
            }
            if self.free_slots.get(control.free_position) != Some(&control.index) {
                return Err(SchedulerError::internal(
                    "prepared request control is no longer free",
                ));
            }
            previous_free_position = control.free_position;
            let word = self
                .table
                .words
                .get(control.index)
                .ok_or_else(|| SchedulerError::internal("prepared control slot is out of range"))?
                .load(Ordering::Acquire);
            if generation_from_word(word) != control.previous_generation {
                return Err(SchedulerError::internal(
                    "prepared request control generation changed",
                ));
            }
            if control.previous_generation != 0 && !has_flag(word, TERMINAL) {
                return Err(SchedulerError::internal(
                    "prepared request control slot is not terminal",
                ));
            }
            entries.push(PreparedBatchControl {
                published: pack_generation(control.generation)?,
                prepared: control,
            });
        }
        Ok(ControlBatchBindPermit {
            registry: self,
            prepared: entries,
            bindings,
        })
    }

    /// Atomically publishes a prepared generation and consumes its free slot.
    pub(crate) fn bind(&mut self, prepared: PreparedControl) -> SchedulerResult<ControlBinding> {
        if !Arc::ptr_eq(&self.table, &prepared.table) {
            return Err(SchedulerError::internal(
                "prepared request control belongs to another registry",
            ));
        }
        let free_position = prepared.free_position;
        if self.free_slots.get(free_position) != Some(&prepared.index) {
            return Err(SchedulerError::internal(
                "prepared request control is no longer free",
            ));
        }
        let word = self
            .table
            .words
            .get(prepared.index)
            .ok_or_else(|| SchedulerError::internal("prepared control slot is out of range"))?;
        let published = pack_generation(prepared.generation)?;

        loop {
            let observed = word.load(Ordering::Acquire);
            if generation_from_word(observed) != prepared.previous_generation {
                return Err(SchedulerError::internal(
                    "prepared request control generation changed",
                ));
            }
            if prepared.previous_generation != 0 && !has_flag(observed, TERMINAL) {
                return Err(SchedulerError::internal(
                    "prepared request control slot is not terminal",
                ));
            }
            if word
                .compare_exchange(observed, published, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }

        self.free_slots.swap_remove(free_position);
        Ok(ControlBinding {
            table: prepared.table,
            index: prepared.index,
            generation: prepared.generation,
        })
    }

    /// Returns a terminal binding's slot to the allocation-free free list.
    ///
    /// Existing clones remain safe: the next bind changes the generation in
    /// the same atomic word before any new handle becomes visible.
    pub(crate) fn recycle(&mut self, binding: &ControlBinding) -> SchedulerResult<()> {
        if !Arc::ptr_eq(&self.table, &binding.table) {
            return Err(SchedulerError::internal(
                "request control binding belongs to another registry",
            ));
        }
        let snapshot = binding.fresh_snapshot()?;
        if !snapshot.terminal() {
            return Err(SchedulerError::internal(
                "active request control cannot be recycled",
            ));
        }
        if self.free_slots.contains(&binding.index) {
            return Err(SchedulerError::internal(
                "request control slot was recycled twice",
            ));
        }
        if self.free_slots.len() >= self.table.words.len() {
            return Err(SchedulerError::internal(
                "request control free list exceeds capacity",
            ));
        }
        self.free_slots.push(binding.index);
        Ok(())
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.table.words.len()
    }

    #[cfg(test)]
    fn available(&self) -> usize {
        self.free_slots.len()
    }
}

fn snapshot_from_word(word: u64) -> ControlSnapshot {
    ControlSnapshot {
        cancelled: has_flag(word, CANCELLED),
        disconnected: has_flag(word, DISCONNECTED),
        terminal: has_flag(word, TERMINAL),
    }
}

const fn has_flag(word: u64, flag: u64) -> bool {
    word & flag != 0
}

const fn generation_from_word(word: u64) -> u64 {
    word >> FLAG_BITS
}

fn pack_generation(generation: NonZeroU64) -> SchedulerResult<u64> {
    if generation.get() > MAX_CONTROL_GENERATION {
        return Err(SchedulerError::resource_exhausted(
            "request control generation",
            generation.get(),
            MAX_CONTROL_GENERATION,
        ));
    }
    generation
        .get()
        .checked_shl(FLAG_BITS)
        .filter(|word| word & FLAG_MASK == 0)
        .ok_or_else(|| SchedulerError::internal("request control generation packing failed"))
}

fn try_reserve_exact<T>(
    values: &mut Vec<T>,
    count: usize,
    resource: &'static str,
) -> SchedulerResult<()> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| SchedulerError::allocation_failure(resource, u64::MAX))?;
    if bytes > isize::MAX as usize {
        return Err(SchedulerError::allocation_failure(
            resource,
            usize_to_u64(bytes),
        ));
    }
    values
        .try_reserve_exact(count)
        .map_err(|_| SchedulerError::allocation_failure(resource, usize_to_u64(bytes)))
}

const fn usize_to_u64(value: usize) -> u64 {
    if size_of::<usize>() > size_of::<u64>() && value > u64::MAX as usize {
        u64::MAX
    } else {
        value as u64
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use crate::error::ErrorCategory;

    use super::*;

    fn bind_one(registry: &mut ControlRegistry) -> ControlBinding {
        let prepared = registry.prepare().expect("control preparation");
        registry.bind(prepared).expect("control binding")
    }

    #[test]
    fn packed_generation_uses_all_non_flag_bits_without_wrapping() {
        let first = NonZeroU64::new(1).expect("nonzero");
        let last = NonZeroU64::new(MAX_CONTROL_GENERATION).expect("nonzero");
        let first_word = pack_generation(first).expect("first generation");
        let last_word = pack_generation(last).expect("last generation");

        assert_eq!(generation_from_word(first_word), 1);
        assert_eq!(generation_from_word(last_word), MAX_CONTROL_GENERATION);
        assert_eq!(first_word & FLAG_MASK, 0);
        assert_eq!(last_word & FLAG_MASK, 0);
        assert_eq!(last_word | FLAG_MASK, u64::MAX);

        let too_large = NonZeroU64::new(MAX_CONTROL_GENERATION + 1).expect("nonzero");
        let error = pack_generation(too_large).expect_err("generation must not wrap");
        assert_eq!(error.category(), ErrorCategory::ResourceExhausted);
    }

    #[test]
    fn construction_rejects_zero_and_impossible_capacity_without_panicking() {
        let zero = ControlRegistry::try_with_capacity(0).expect_err("zero must fail");
        assert_eq!(zero.category(), ErrorCategory::InvalidRequest);

        let impossible = ControlRegistry::try_with_capacity(usize::MAX)
            .expect_err("impossible allocation must fail");
        assert_eq!(impossible.category(), ErrorCategory::ResourceExhausted);
        assert!(!format!("{impossible:?}").contains("request_id"));
    }

    #[test]
    fn prepare_is_nonmutating_and_drop_is_an_exact_rollback() {
        let registry = ControlRegistry::try_with_capacity(2).expect("registry");
        let first = registry.prepare().expect("first preparation");
        let second = registry.prepare().expect("second preparation");

        assert_eq!(first.index, second.index);
        assert_eq!(first.generation, second.generation);
        assert_eq!(registry.available(), 2);
        drop((first, second));
        assert_eq!(registry.available(), 2);
    }

    #[test]
    fn bind_consumes_capacity_and_recycle_requires_terminal_state() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        let binding = bind_one(&mut registry);
        assert_eq!(registry.capacity(), 1);
        assert_eq!(registry.available(), 0);
        assert_eq!(
            registry
                .prepare()
                .expect_err("bound registry must be full")
                .category(),
            ErrorCategory::ResourceExhausted
        );
        assert_eq!(
            registry
                .recycle(&binding)
                .expect_err("active binding must not recycle")
                .category(),
            ErrorCategory::Internal
        );

        assert_eq!(
            binding.mark_terminal().expect("terminal publication"),
            MarkTerminalDisposition::Marked
        );
        registry.recycle(&binding).expect("terminal recycle");
        assert_eq!(registry.available(), 1);
        assert_eq!(
            registry
                .recycle(&binding)
                .expect_err("double recycle must fail")
                .category(),
            ErrorCategory::Internal
        );
    }

    #[test]
    fn multi_slot_allocation_and_recycling_are_exact_lifo() {
        let mut registry = ControlRegistry::try_with_capacity(3).expect("registry");
        let first = bind_one(&mut registry);
        let second = bind_one(&mut registry);
        let third = bind_one(&mut registry);
        assert_eq!(first.stress_identity(), (0, 1));
        assert_eq!(second.stress_identity(), (1, 1));
        assert_eq!(third.stress_identity(), (2, 1));

        first.mark_terminal().expect("first terminal");
        second.mark_terminal().expect("second terminal");
        registry.recycle(&first).expect("recycle first");
        registry.recycle(&second).expect("recycle second");

        let rebound_second = bind_one(&mut registry);
        let rebound_first = bind_one(&mut registry);
        assert_eq!(rebound_second.stress_identity(), (1, 2));
        assert_eq!(rebound_first.stress_identity(), (0, 2));
        assert_eq!(third.stress_identity(), (2, 1));
        assert_eq!(registry.available(), 0);
    }

    #[test]
    fn cancellation_is_idempotent_and_terminal_is_sticky() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        let binding = bind_one(&mut registry);
        assert_eq!(
            binding.cancel().expect("first cancel"),
            CancelDisposition::Requested
        );
        assert_eq!(
            binding.cancel().expect("second cancel"),
            CancelDisposition::AlreadyRequested
        );
        let cancelled = binding.fresh_snapshot().expect("snapshot");
        assert!(cancelled.cancelled());
        assert!(!cancelled.disconnected());
        assert!(!cancelled.terminal());

        assert_eq!(
            binding.mark_terminal().expect("mark terminal"),
            MarkTerminalDisposition::Marked
        );
        assert_eq!(
            binding.mark_terminal().expect("mark terminal twice"),
            MarkTerminalDisposition::AlreadyTerminal
        );
        assert_eq!(
            binding.cancel().expect("cancel after terminal"),
            CancelDisposition::AlreadyTerminal
        );
        let terminal = binding.fresh_snapshot().expect("terminal snapshot");
        assert!(terminal.cancelled());
        assert!(terminal.terminal());
    }

    #[test]
    fn disconnect_implies_cancellation_and_records_after_terminal() {
        let mut registry = ControlRegistry::try_with_capacity(2).expect("registry");
        let active = bind_one(&mut registry);
        assert_eq!(
            active.disconnect().expect("disconnect"),
            DisconnectDisposition::Requested
        );
        assert_eq!(
            active.disconnect().expect("repeat disconnect"),
            DisconnectDisposition::AlreadyRequested
        );
        let snapshot = active.fresh_snapshot().expect("active snapshot");
        assert!(snapshot.cancelled());
        assert!(snapshot.disconnected());
        assert!(!snapshot.terminal());

        let terminal = bind_one(&mut registry);
        terminal.mark_terminal().expect("mark terminal");
        assert_eq!(
            terminal.disconnect().expect("terminal disconnect"),
            DisconnectDisposition::AlreadyTerminal
        );
        let snapshot = terminal.fresh_snapshot().expect("terminal snapshot");
        assert!(snapshot.cancelled());
        assert!(snapshot.disconnected());
        assert!(snapshot.terminal());
    }

    #[test]
    fn stale_binding_cannot_cancel_or_disconnect_reused_slot() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        let stale = bind_one(&mut registry);
        stale.mark_terminal().expect("terminal publication");
        registry.recycle(&stale).expect("recycle");
        let current = bind_one(&mut registry);

        assert_eq!(
            stale
                .fresh_snapshot()
                .expect_err("stale snapshot must fail")
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(
            stale
                .cancel()
                .expect_err("stale cancel must fail")
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(
            stale
                .disconnect()
                .expect_err("stale disconnect must fail")
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(
            current.fresh_snapshot().expect("current snapshot"),
            ControlSnapshot::default()
        );
    }

    #[test]
    fn stale_compare_exchange_witness_cannot_cross_rebind_aba_boundary() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        let stale = bind_one(&mut registry);
        let stale_word = stale.word().expect("word").load(Ordering::Acquire);
        stale.mark_terminal().expect("terminal publication");
        registry.recycle(&stale).expect("recycle");
        let current = bind_one(&mut registry);

        let result = stale.word().expect("word").compare_exchange(
            stale_word,
            stale_word | CANCELLED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        assert!(result.is_err());
        assert_eq!(
            current.fresh_snapshot().expect("current snapshot"),
            ControlSnapshot::default()
        );
    }

    #[test]
    fn instrumented_cancel_witnesses_success_already_and_stale_final_loads() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        let stale = bind_one(&mut registry);

        let (requested, requested_witness) = stale.cancel_with_stress_witness();
        assert_eq!(
            requested.expect("cancel request"),
            CancelDisposition::Requested
        );
        assert!(requested_witness.boundary_reached());
        assert_eq!(requested_witness.expected_generation(), 1);
        assert_eq!(requested_witness.loaded_word(), 1 << FLAG_BITS);
        assert_eq!(
            requested_witness.resulting_word(),
            (1 << FLAG_BITS) | CANCELLED
        );

        let (already, already_witness) = stale.cancel_with_stress_witness();
        assert_eq!(
            already.expect("repeat cancel"),
            CancelDisposition::AlreadyRequested
        );
        assert_eq!(
            already_witness.loaded_word(),
            requested_witness.resulting_word()
        );
        assert_eq!(
            already_witness.resulting_word(),
            already_witness.loaded_word()
        );

        stale.mark_terminal().expect("terminal publication");
        let (after_terminal, terminal_witness) = stale.cancel_with_stress_witness();
        assert_eq!(
            after_terminal.expect("cancel after terminal"),
            CancelDisposition::AlreadyTerminal
        );
        assert_eq!(
            terminal_witness.loaded_word(),
            requested_witness.resulting_word() | TERMINAL
        );
        assert_eq!(
            terminal_witness.resulting_word(),
            terminal_witness.loaded_word()
        );
        registry.recycle(&stale).expect("recycle");
        let current = bind_one(&mut registry);
        let current_word = current.word().expect("word").load(Ordering::Acquire);
        let (stale_result, stale_witness) = stale.cancel_with_stress_witness();
        assert_eq!(
            stale_result.expect_err("stale cancel must fail").category(),
            ErrorCategory::InvalidRequest
        );
        assert!(stale_witness.boundary_reached());
        assert_eq!(stale_witness.expected_generation(), 1);
        assert_eq!(stale_witness.loaded_word(), current_word);
        assert_eq!(stale_witness.resulting_word(), current_word);
        assert_eq!(stale_witness.slot_index(), requested_witness.slot_index());
        assert_eq!(
            current.fresh_snapshot().expect("current snapshot"),
            ControlSnapshot::default()
        );
    }

    #[test]
    fn unobserved_control_witness_uses_only_zero_sentinels() {
        let witness = ActorControlCasWitness::sentinel();
        assert!(!witness.boundary_reached());
        assert_eq!(witness.slot_index(), 0);
        assert_eq!(witness.expected_generation(), 0);
        assert_eq!(witness.loaded_word(), 0);
        assert_eq!(witness.resulting_word(), 0);
    }

    #[test]
    fn instrumented_disconnect_witnesses_success_already_and_stale_final_loads() {
        let mut registry = ControlRegistry::try_with_capacity(2).expect("registry");
        let stale = bind_one(&mut registry);
        let terminal = bind_one(&mut registry);

        let (requested, requested_witness) = stale.disconnect_with_stress_witness();
        assert_eq!(
            requested.expect("disconnect request"),
            DisconnectDisposition::Requested
        );
        assert_eq!(requested_witness.loaded_word(), 1 << FLAG_BITS);
        assert_eq!(
            requested_witness.resulting_word(),
            (1 << FLAG_BITS) | CANCELLED | DISCONNECTED
        );

        let (already, already_witness) = stale.disconnect_with_stress_witness();
        assert_eq!(
            already.expect("repeat disconnect"),
            DisconnectDisposition::AlreadyRequested
        );
        assert_eq!(
            already_witness.loaded_word(),
            requested_witness.resulting_word()
        );
        assert_eq!(
            already_witness.resulting_word(),
            already_witness.loaded_word()
        );

        terminal.mark_terminal().expect("terminal publication");
        let (after_terminal, terminal_witness) = terminal.disconnect_with_stress_witness();
        assert_eq!(
            after_terminal.expect("disconnect after terminal"),
            DisconnectDisposition::AlreadyTerminal
        );
        assert_eq!(terminal_witness.loaded_word(), (1 << FLAG_BITS) | TERMINAL);
        assert_eq!(
            terminal_witness.resulting_word(),
            terminal_witness.loaded_word() | CANCELLED | DISCONNECTED
        );
        let (terminal_again, terminal_again_witness) = terminal.disconnect_with_stress_witness();
        assert_eq!(
            terminal_again.expect("repeat terminal disconnect"),
            DisconnectDisposition::AlreadyTerminal
        );
        assert_eq!(
            terminal_again_witness.resulting_word(),
            terminal_again_witness.loaded_word()
        );

        stale.mark_terminal().expect("terminal publication");
        registry.recycle(&stale).expect("recycle");
        let current = bind_one(&mut registry);
        let current_word = current.word().expect("word").load(Ordering::Acquire);
        let (stale_result, stale_witness) = stale.disconnect_with_stress_witness();
        assert_eq!(
            stale_result
                .expect_err("stale disconnect must fail")
                .category(),
            ErrorCategory::InvalidRequest
        );
        assert_eq!(stale_witness.loaded_word(), current_word);
        assert_eq!(stale_witness.resulting_word(), current_word);
        assert_eq!(stale_witness.expected_generation(), 1);
        assert_eq!(
            current.fresh_snapshot().expect("current snapshot"),
            ControlSnapshot::default()
        );
    }

    #[test]
    fn prepared_and_bound_values_cannot_cross_registries() {
        let mut first = ControlRegistry::try_with_capacity(1).expect("first registry");
        let mut second = ControlRegistry::try_with_capacity(1).expect("second registry");
        let prepared = first.prepare().expect("preparation");
        assert_eq!(
            second
                .bind(prepared)
                .expect_err("foreign preparation must fail")
                .category(),
            ErrorCategory::Internal
        );
        assert_eq!(first.available(), 1);
        assert_eq!(second.available(), 1);

        let binding = bind_one(&mut first);
        binding.mark_terminal().expect("terminal publication");
        assert_eq!(
            second
                .recycle(&binding)
                .expect_err("foreign binding must fail")
                .category(),
            ErrorCategory::Internal
        );
    }

    #[test]
    fn generation_exhaustion_is_permanent_and_sanitized() {
        let registry = ControlRegistry::try_with_capacity(1).expect("registry");
        registry.table.words[0].store(
            (MAX_CONTROL_GENERATION << FLAG_BITS) | TERMINAL,
            Ordering::Release,
        );
        let error = registry
            .prepare()
            .expect_err("generation must be exhausted");
        assert_eq!(error.category(), ErrorCategory::ResourceExhausted);
        let debug = format!("{error:?}");
        assert!(!debug.contains("slot_index"));
    }

    #[test]
    fn cancel_and_terminal_race_has_only_linearizable_outcomes() {
        for _ in 0..256 {
            let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
            let binding = bind_one(&mut registry);
            let cancel_binding = binding.clone();
            let terminal_binding = binding.clone();
            let barrier = Arc::new(Barrier::new(3));
            let cancel_barrier = Arc::clone(&barrier);
            let terminal_barrier = Arc::clone(&barrier);
            let cancel = thread::spawn(move || {
                cancel_barrier.wait();
                cancel_binding.cancel().expect("cancel race")
            });
            let terminal = thread::spawn(move || {
                terminal_barrier.wait();
                terminal_binding.mark_terminal().expect("terminal race")
            });
            barrier.wait();
            let cancel = cancel.join().expect("cancel thread");
            let terminal = terminal.join().expect("terminal thread");
            let snapshot = binding.fresh_snapshot().expect("race snapshot");

            assert_eq!(terminal, MarkTerminalDisposition::Marked);
            assert!(snapshot.terminal());
            match cancel {
                CancelDisposition::Requested => assert!(snapshot.cancelled()),
                CancelDisposition::AlreadyTerminal => assert!(!snapshot.cancelled()),
                CancelDisposition::AlreadyRequested => {
                    panic!("only one cancellation participant exists")
                }
            }
        }
    }

    #[test]
    fn concurrent_cancel_and_disconnect_preserve_both_monotone_bits() {
        for _ in 0..256 {
            let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
            let binding = bind_one(&mut registry);
            let cancel_binding = binding.clone();
            let disconnect_binding = binding.clone();
            let barrier = Arc::new(Barrier::new(3));
            let cancel_barrier = Arc::clone(&barrier);
            let disconnect_barrier = Arc::clone(&barrier);
            let cancel = thread::spawn(move || {
                cancel_barrier.wait();
                cancel_binding.cancel().expect("cancel race")
            });
            let disconnect = thread::spawn(move || {
                disconnect_barrier.wait();
                disconnect_binding.disconnect().expect("disconnect race")
            });
            barrier.wait();
            let cancel = cancel.join().expect("cancel thread");
            let disconnect = disconnect.join().expect("disconnect thread");
            let snapshot = binding.fresh_snapshot().expect("race snapshot");

            assert!(matches!(
                cancel,
                CancelDisposition::Requested | CancelDisposition::AlreadyRequested
            ));
            assert!(matches!(
                disconnect,
                DisconnectDisposition::Requested | DisconnectDisposition::AlreadyRequested
            ));
            assert!(snapshot.cancelled());
            assert!(snapshot.disconnected());
            assert!(!snapshot.terminal());
        }
    }

    #[test]
    fn snapshot_is_a_boundary_value_and_each_load_is_fresh() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        let binding = bind_one(&mut registry);
        let before = binding.fresh_snapshot().expect("before snapshot");
        binding.cancel().expect("cancel");
        let after_cancel = binding.fresh_snapshot().expect("after cancel snapshot");
        binding.mark_terminal().expect("terminal publication");
        let after_terminal = binding.fresh_snapshot().expect("terminal snapshot");

        assert_eq!(before, ControlSnapshot::default());
        assert!(after_cancel.cancelled());
        assert!(!after_cancel.terminal());
        assert!(after_terminal.cancelled());
        assert!(after_terminal.terminal());
    }

    #[test]
    fn prevalidated_terminal_publication_is_infallible_and_preserves_controls() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        let binding = bind_one(&mut registry);
        binding.cancel().expect("cancel");
        let validated = binding.fresh_snapshot().expect("prevalidation snapshot");
        assert!(validated.cancelled());
        assert!(!validated.terminal());

        binding.mark_terminal_prevalidated();

        let published = binding.fresh_snapshot().expect("published snapshot");
        assert!(published.cancelled());
        assert!(published.terminal());
        binding.mark_terminal_prevalidated();
        assert!(
            binding
                .fresh_snapshot()
                .expect("idempotent snapshot")
                .terminal()
        );
    }

    #[test]
    fn batch_bind_permit_drop_is_inert_and_commit_binds_distinct_lifo_slots() {
        let mut registry = ControlRegistry::try_with_capacity(3).expect("registry");
        let initial_free_slots = registry.free_slots.clone();
        let initial_capacity = registry.free_slots.capacity();
        let initial_words = registry
            .table
            .words
            .iter()
            .map(|word| word.load(Ordering::Acquire))
            .collect::<Vec<_>>();

        let available = registry.prepare_available(2).expect("available controls");
        let permit = registry
            .begin_bind_batch(available.into_prefix(2))
            .expect("batch control permit");
        drop(permit);
        assert_eq!(registry.free_slots, initial_free_slots);
        assert_eq!(registry.free_slots.capacity(), initial_capacity);
        assert_eq!(
            registry
                .table
                .words
                .iter()
                .map(|word| word.load(Ordering::Acquire))
                .collect::<Vec<_>>(),
            initial_words
        );

        let available = registry.prepare_available(2).expect("available controls");
        let bindings = registry
            .begin_bind_batch(available.into_prefix(2))
            .expect("batch control permit")
            .commit();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].stress_identity(), (0, 1));
        assert_eq!(bindings[1].stress_identity(), (1, 1));
        assert_eq!(registry.free_slots, [2]);
        assert_eq!(registry.free_slots.capacity(), initial_capacity);
    }

    #[test]
    fn stale_terminal_controls_racing_batch_rebind_never_mark_new_generation() {
        let mut registry = ControlRegistry::try_with_capacity(1).expect("registry");
        for _ in 0..256 {
            let stale = bind_one(&mut registry);
            stale.mark_terminal().expect("terminal publication");
            registry.recycle(&stale).expect("recycle terminal control");
            let available = registry.prepare_available(1).expect("available control");
            let permit = registry
                .begin_bind_batch(available.into_prefix(1))
                .expect("batch control permit");
            let raced = stale.clone();
            let barrier = Arc::new(Barrier::new(2));
            let worker_barrier = Arc::clone(&barrier);
            let (current, race_results) = thread::scope(|scope| {
                let worker = scope.spawn(move || {
                    worker_barrier.wait();
                    (raced.cancel(), raced.disconnect())
                });
                barrier.wait();
                let current = permit.commit().pop().expect("current binding");
                (current, worker.join().expect("stale control worker"))
            });
            assert!(
                matches!(race_results.0, Ok(CancelDisposition::AlreadyTerminal))
                    || matches!(
                        race_results.0,
                        Err(ref error) if error.category() == ErrorCategory::InvalidRequest
                    )
            );
            assert!(
                matches!(race_results.1, Ok(DisconnectDisposition::AlreadyTerminal))
                    || matches!(
                        race_results.1,
                        Err(ref error) if error.category() == ErrorCategory::InvalidRequest
                    )
            );
            assert_eq!(
                current.fresh_snapshot().expect("new generation snapshot"),
                ControlSnapshot::default()
            );
            current.mark_terminal().expect("current terminal");
            registry.recycle(&current).expect("current recycle");
        }
    }

    #[test]
    fn debug_output_never_contains_slot_or_generation_identity() {
        let mut registry = ControlRegistry::try_with_capacity(11).expect("registry");
        for _ in 0..7 {
            let binding = bind_one(&mut registry);
            binding.mark_terminal().expect("terminal publication");
            registry.recycle(&binding).expect("recycle");
        }
        let prepared = registry.prepare().expect("preparation");
        let generation = prepared.generation.get().to_string();
        let index = prepared.index.to_string();
        let prepared_debug = format!("{prepared:?}");
        assert!(!prepared_debug.contains(&generation));
        assert!(!prepared_debug.contains(&index));
        let binding = registry.bind(prepared).expect("binding");
        let binding_debug = format!("{binding:?}");
        assert!(!binding_debug.contains(&generation));
        assert!(!binding_debug.contains(&index));
        let registry_debug = format!("{registry:?}");
        assert!(registry_debug.contains("capacity"));
        assert!(registry_debug.contains("available"));
    }
}
