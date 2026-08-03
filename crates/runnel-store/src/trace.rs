//! A bounded, nonblocking trace sink for sampled page identities.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Instant;

use crate::{AccessReason, PageKey};

/// The result of one observable cache or backend transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TraceOutcome {
    Hit,
    Miss,
    LoadStarted,
    LoadCoalesced,
    LatePrefetch,
    PrefetchCoalesced,
    Admitted,
    Evicted,
    Retired,
    LoadFailed,
    Cancelled,
    PrefetchUseful,
    PrefetchWasted,
    PrefetchRedundant,
    PrefetchDropped,
}

/// A sampled event. Unlike metrics, traces may carry a page identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceEvent {
    pub sequence: u64,
    pub monotonic_nanoseconds: u64,
    pub outcome: TraceOutcome,
    pub reason: AccessReason,
    pub key: PageKey,
    pub bytes: u64,
}

#[derive(Debug)]
struct TraceInner {
    capacity: usize,
    started: Instant,
    next_sequence: AtomicU64,
    dropped: AtomicU64,
    events: Mutex<VecDeque<TraceEvent>>,
}

/// A cloneable bounded trace queue.
///
/// Recording uses `try_lock`; contention and a full queue drop the new event.
/// This deliberately prevents tracing from feeding back into cache behavior.
#[derive(Clone, Debug)]
pub struct TraceSink {
    inner: Arc<TraceInner>,
}

impl TraceSink {
    /// Hard ceiling for retained trace events. Configuration validation uses
    /// the same bound, while direct construction clamps to it defensively.
    pub const MAX_CAPACITY: usize = 65_536;

    /// Creates a sink retaining at most `capacity` events. A zero-capacity sink
    /// is a cheap disabled sink that still counts dropped observations.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.min(Self::MAX_CAPACITY);
        Self {
            inner: Arc::new(TraceInner {
                capacity,
                started: Instant::now(),
                next_sequence: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
                // Do not eagerly allocate the configured maximum. Besides
                // keeping a disabled/idle sink cheap, this makes construction
                // safe even when called with an untrusted `usize` directly.
                events: Mutex::new(VecDeque::new()),
            }),
        }
    }

    /// Records an event without waiting for the consumer or the queue lock.
    pub(crate) fn record(
        &self,
        outcome: TraceOutcome,
        reason: AccessReason,
        key: PageKey,
        bytes: u64,
    ) {
        let sequence = self.inner.next_sequence.fetch_add(1, Ordering::Relaxed);
        let monotonic_nanoseconds = saturating_nanos(self.inner.started.elapsed().as_nanos());
        let event = TraceEvent {
            sequence,
            monotonic_nanoseconds,
            outcome,
            reason,
            key,
            bytes,
        };

        match self.inner.events.try_lock() {
            Ok(mut events) if events.len() < self.inner.capacity => events.push_back(event),
            Ok(_) | Err(TryLockError::WouldBlock) | Err(TryLockError::Poisoned(_)) => {
                increment_saturating(&self.inner.dropped);
            }
        }
    }

    /// Removes and returns all currently retained events in sequence order.
    #[must_use]
    pub fn drain(&self) -> Vec<TraceEvent> {
        let mut drained: Vec<_> = match self.inner.events.lock() {
            Ok(mut events) => events.drain(..).collect(),
            Err(poisoned) => poisoned.into_inner().drain(..).collect(),
        };
        // Producers reserve sequence numbers before acquiring the queue lock,
        // so concurrent insertions may arrive in a different order.
        drained.sort_unstable_by_key(|event| event.sequence);
        drained
    }

    /// Number of events dropped because tracing was disabled, full, or busy.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.inner.dropped.load(Ordering::Relaxed)
    }

    /// Maximum retained event count.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }
}

impl Default for TraceSink {
    fn default() -> Self {
        Self::new(0)
    }
}

fn increment_saturating(value: &AtomicU64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(1))
    });
}

fn saturating_nanos(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{TraceEvent, TraceOutcome, TraceSink};
    use crate::{AccessReason, PageKey};
    use runnel_format::Digest;

    fn key(index: u64) -> PageKey {
        PageKey::new(Digest::from_bytes([7; 32]), 65_536, index)
    }

    #[test]
    fn bounded_sink_drops_new_events_and_drains_in_order() {
        let sink = TraceSink::new(2);
        sink.record(TraceOutcome::Miss, AccessReason::Demand, key(0), 8);
        sink.record(TraceOutcome::Admitted, AccessReason::Demand, key(0), 8);
        sink.record(TraceOutcome::Hit, AccessReason::Demand, key(0), 8);

        let events = sink.drain();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, 0);
        assert_eq!(events[1].sequence, 1);
        assert_eq!(sink.dropped(), 1);
        assert!(sink.drain().is_empty());
    }

    #[test]
    fn zero_capacity_is_disabled_but_counted() {
        let sink = TraceSink::default();
        sink.record(TraceOutcome::Miss, AccessReason::Prefetch, key(1), 1);
        assert!(sink.drain().is_empty());
        assert_eq!(sink.dropped(), 1);
    }

    #[test]
    fn drain_orders_concurrently_inserted_sequences() {
        let sink = TraceSink::new(3);
        let event = |sequence, index| TraceEvent {
            sequence,
            monotonic_nanoseconds: sequence,
            outcome: TraceOutcome::Miss,
            reason: AccessReason::Demand,
            key: key(index),
            bytes: 1,
        };
        {
            let mut events = sink.inner.events.lock().unwrap();
            events.push_back(event(2, 2));
            events.push_back(event(0, 0));
            events.push_back(event(1, 1));
        }

        let sequences: Vec<_> = sink
            .drain()
            .into_iter()
            .map(|event| event.sequence)
            .collect();
        assert_eq!(sequences, [0, 1, 2]);
    }

    #[test]
    fn direct_construction_clamps_hostile_capacity_without_allocating_it() {
        let sink = TraceSink::new(usize::MAX);
        assert_eq!(sink.capacity(), TraceSink::MAX_CAPACITY);
        assert_eq!(sink.inner.events.lock().unwrap().capacity(), 0);
    }
}
