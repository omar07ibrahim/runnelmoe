//! Bounded, append-only semantic service evidence.
//!
//! The trace records successful model-position commits, not elapsed time. It
//! deliberately excludes prompt and output tokens, sampling state, router
//! decisions, deadlines, adapter identities, and clock observations. Evidence
//! consumers map the opaque engine-local request IDs to their own synthetic
//! workload indices.

use std::{fmt, mem::size_of};

use crate::RequestId;

/// Total logical ledger charge retained for every configured trace slot.
///
/// One slot independently reserves up to 64 bytes for a service event and up
/// to 64 bytes for a ledger event. This is a semantic capacity charge rather
/// than `size_of` or allocator RSS.
pub const TRACE_SLOT_CHARGE_BYTES: u64 = 128;

/// Maximum in-memory representation reserved for one service evidence event.
pub(crate) const SERVICE_TRACE_EVENT_CAPACITY_BYTES: u64 = 64;

/// The model-position phase frozen by the M5 evidence contract.
///
/// A position is prefill while its zero-based index is less than the prompt
/// length. Consequently the final prompt position, which can publish the
/// first output token, remains a prefill event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ServicePhase {
    Prefill = 0,
    Decode = 1,
}

impl ServicePhase {
    /// Returns the stable compact-evidence bit (`0=prefill`, `1=decode`).
    #[must_use]
    pub const fn evidence_bit(self) -> u8 {
        self as u8
    }

    /// Returns the stable human-readable phase name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

/// One successfully committed equal-cost model-position service quantum.
///
/// The fields are intentionally private. Accessors are an explicit opt-in to
/// semantic evidence, while `Debug` remains safe for ordinary diagnostic logs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServiceTraceEvent {
    request_id: RequestId,
    position: usize,
    phase: ServicePhase,
}

impl ServiceTraceEvent {
    pub(crate) const fn new(request_id: RequestId, position: usize, phase: ServicePhase) -> Self {
        Self {
            request_id,
            position,
            phase,
        }
    }

    /// Returns the opaque engine-local request identity.
    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    /// Returns the zero-based position committed by this event.
    #[must_use]
    pub const fn position(self) -> usize {
        self.position
    }

    /// Returns whether this committed position belongs to prefill or decode.
    #[must_use]
    pub const fn phase(self) -> ServicePhase {
        self.phase
    }
}

impl fmt::Debug for ServiceTraceEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServiceTraceEvent")
            .field("request_id", &"<redacted>")
            .field("position", &"<redacted>")
            .field("phase", &self.phase)
            .finish()
    }
}

const _: () = assert!(
    size_of::<ServiceTraceEvent>() <= SERVICE_TRACE_EVENT_CAPACITY_BYTES as usize,
    "service trace event exceeds its independent slot capacity"
);

/// An opaque ordinal into a retained service-trace prefix.
///
/// A cursor deliberately carries no engine cookie. Callers must pair a returned
/// cursor with the same engine; crossing engines can select an unrelated suffix
/// when the ordinal happens to be in range. Evidence consumers should either
/// read from [`Self::origin`] or persist and validate every contiguous suffix.
/// A cursor cannot manufacture a request ID.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ServiceTraceCursor(usize);

impl ServiceTraceCursor {
    /// Returns the beginning of an engine's retained trace.
    #[must_use]
    pub const fn origin() -> Self {
        Self(0)
    }

    /// Returns the zero-based index of the next event this cursor would read.
    #[must_use]
    pub const fn event_index(self) -> usize {
        self.0
    }

    pub(crate) const fn from_retained_len(retained_len: usize) -> Self {
        Self(retained_len)
    }
}

/// Health and fixed-capacity metadata coupled to every trace read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServiceTraceStatus {
    retained_events: usize,
    event_limit: usize,
    overflowed: bool,
}

impl ServiceTraceStatus {
    pub(crate) const fn new(retained_events: usize, event_limit: usize, overflowed: bool) -> Self {
        Self {
            retained_events,
            event_limit,
            overflowed,
        }
    }

    /// Returns the length of the immutable retained prefix.
    #[must_use]
    pub const fn retained_events(self) -> usize {
        self.retained_events
    }

    /// Returns the configured logical event limit.
    #[must_use]
    pub const fn event_limit(self) -> usize {
        self.event_limit
    }

    /// Returns whether at least one successful commit could not be retained.
    ///
    /// Reaching exactly `event_limit` is still complete. Only the next
    /// unrecordable successful commit makes this flag sticky.
    #[must_use]
    pub const fn overflowed(self) -> bool {
        self.overflowed
    }

    /// Returns whether the retained prefix is a complete service history.
    #[must_use]
    pub const fn healthy(self) -> bool {
        !self.overflowed
    }
}

/// One allocation-free borrowed read of an engine's append-only trace.
#[must_use = "a trace read carries both events and completeness status"]
pub struct ServiceTraceRead<'trace> {
    events: &'trace [ServiceTraceEvent],
    start_cursor: ServiceTraceCursor,
    next_cursor: ServiceTraceCursor,
    status: ServiceTraceStatus,
}

impl<'trace> ServiceTraceRead<'trace> {
    pub(crate) const fn new(
        events: &'trace [ServiceTraceEvent],
        start_cursor: ServiceTraceCursor,
        next_cursor: ServiceTraceCursor,
        status: ServiceTraceStatus,
    ) -> Self {
        Self {
            events,
            start_cursor,
            next_cursor,
            status,
        }
    }

    /// Returns only the newly retained suffix beginning at `start_cursor`.
    #[must_use]
    pub const fn events(&self) -> &'trace [ServiceTraceEvent] {
        self.events
    }

    /// Returns the validated cursor supplied for this read.
    #[must_use]
    pub const fn start_cursor(&self) -> ServiceTraceCursor {
        self.start_cursor
    }

    /// Returns the frontier to retain for the next incremental read.
    #[must_use]
    pub const fn next_cursor(&self) -> ServiceTraceCursor {
        self.next_cursor
    }

    /// Returns trace completeness and fixed-capacity metadata.
    #[must_use]
    pub const fn status(&self) -> ServiceTraceStatus {
        self.status
    }
}

impl fmt::Debug for ServiceTraceRead<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServiceTraceRead")
            .field("start_cursor", &self.start_cursor)
            .field("next_cursor", &self.next_cursor)
            .field("event_count", &self.events.len())
            .field("status", &self.status)
            .field("events", &"<redacted>")
            .finish()
    }
}
