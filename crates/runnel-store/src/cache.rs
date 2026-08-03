//! Verified byte-capacity page cache with explicit leases and generations.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use runnel_format::Digest;
use tokio::sync::oneshot;

use crate::metrics::Metrics;
use crate::trace::TraceOutcome;
use crate::{
    AsyncReader, CancellationToken, Control, MetricsSnapshot, PageKey, PageSpec, ReadStats,
    StoreError, TraceSink, VerifiedPage,
};

/// Why a page was requested. This is bounded-cardinality metric context; page
/// identity is emitted only to the optional trace sink.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccessReason {
    Demand,
    Prefetch,
}

/// The cache's disposition of a prefetch hint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefetchOutcome {
    /// A new physical read was admitted.
    Started,
    /// A prefetch interest joined an existing demand read.
    CoalescedWithDemand,
    /// The same page was resident or already had prefetch interest.
    Redundant,
    /// Capacity, entry, or in-flight limits rejected the hint.
    DroppedBudget,
}

/// All cache limits are explicit and validated before resource allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheConfig {
    page_pool_capacity: u64,
    max_page_bytes: u64,
    max_inflight_bytes: u64,
    max_entries: usize,
    max_loads: usize,
    max_waiters: usize,
    max_waiters_per_page: usize,
    max_leases: usize,
    trace_capacity: usize,
}

impl CacheConfig {
    /// Hard ceiling for resident, loading, and retiring cache entries.
    pub const MAX_ENTRIES: usize = 1_048_576;
    /// Hard ceiling for simultaneous physical page loads.
    pub const MAX_LOADS: usize = 4_096;
    /// Hard ceiling for demand waiters across the cache.
    pub const MAX_WAITERS: usize = 1_048_576;
    /// Hard ceiling for demand waiters coalesced onto one page.
    pub const MAX_WAITERS_PER_PAGE: usize = 65_536;
    /// Hard ceiling for live leases and reserved lease deliveries.
    pub const MAX_LEASES: usize = 1_048_576;

    /// Starts a configuration with conservative bounded defaults for all
    /// limits except the two byte limits that define the supported workload.
    pub fn new(page_pool_capacity: u64, max_page_bytes: u64) -> Result<Self, StoreError> {
        let config = Self {
            page_pool_capacity,
            max_page_bytes,
            max_inflight_bytes: page_pool_capacity,
            max_entries: 4_096,
            max_loads: 4,
            max_waiters: 4_096,
            max_waiters_per_page: 128,
            max_leases: 4_096,
            trace_capacity: 1_024,
        };
        config.validate()?;
        Ok(config)
    }

    #[must_use]
    pub const fn with_max_inflight_bytes(mut self, value: u64) -> Self {
        self.max_inflight_bytes = value;
        self
    }

    #[must_use]
    pub const fn with_max_entries(mut self, value: usize) -> Self {
        self.max_entries = value;
        self
    }

    #[must_use]
    pub const fn with_max_loads(mut self, value: usize) -> Self {
        self.max_loads = value;
        self
    }

    #[must_use]
    pub const fn with_waiter_limits(mut self, total: usize, per_page: usize) -> Self {
        self.max_waiters = total;
        self.max_waiters_per_page = per_page;
        self
    }

    #[must_use]
    pub const fn with_max_leases(mut self, value: usize) -> Self {
        self.max_leases = value;
        self
    }

    #[must_use]
    pub const fn with_trace_capacity(mut self, value: usize) -> Self {
        self.trace_capacity = value;
        self
    }

    #[must_use]
    pub const fn page_pool_capacity(self) -> u64 {
        self.page_pool_capacity
    }

    #[must_use]
    pub const fn max_page_bytes(self) -> u64 {
        self.max_page_bytes
    }

    #[must_use]
    pub const fn max_inflight_bytes(self) -> u64 {
        self.max_inflight_bytes
    }

    fn validate(self) -> Result<(), StoreError> {
        if self.page_pool_capacity == 0 {
            return Err(StoreError::invalid_config(
                "page_pool_capacity",
                "must be greater than zero",
            ));
        }
        if self.max_page_bytes == 0 {
            return Err(StoreError::invalid_config(
                "max_page_bytes",
                "must be greater than zero",
            ));
        }
        let max_page_allocation = crate::fs::aligned_page_buffer_bytes(self.max_page_bytes)
            .map_err(|_| {
                StoreError::invalid_config(
                    "max_page_bytes",
                    "aligned supported-page allocation overflows",
                )
            })?;
        if max_page_allocation > self.page_pool_capacity {
            return Err(StoreError::invalid_config(
                "max_page_bytes",
                "one aligned supported-page allocation must fit in the page pool",
            ));
        }
        if self.max_inflight_bytes < max_page_allocation
            || self.max_inflight_bytes > self.page_pool_capacity
        {
            return Err(StoreError::invalid_config(
                "max_inflight_bytes",
                "must fit one aligned supported-page allocation and not exceed the page pool",
            ));
        }
        for (field, value) in [
            ("max_entries", self.max_entries),
            ("max_loads", self.max_loads),
            ("max_waiters", self.max_waiters),
            ("max_waiters_per_page", self.max_waiters_per_page),
            ("max_leases", self.max_leases),
        ] {
            if value == 0 {
                return Err(StoreError::invalid_config(
                    field,
                    "must be greater than zero",
                ));
            }
        }
        for (field, value, ceiling) in [
            ("max_entries", self.max_entries, Self::MAX_ENTRIES),
            ("max_loads", self.max_loads, Self::MAX_LOADS),
            ("max_waiters", self.max_waiters, Self::MAX_WAITERS),
            (
                "max_waiters_per_page",
                self.max_waiters_per_page,
                Self::MAX_WAITERS_PER_PAGE,
            ),
            ("max_leases", self.max_leases, Self::MAX_LEASES),
            (
                "trace_capacity",
                self.trace_capacity,
                TraceSink::MAX_CAPACITY,
            ),
        ] {
            if value > ceiling {
                return Err(StoreError::invalid_config(
                    field,
                    "exceeds the hard resource ceiling",
                ));
            }
        }
        if self.max_loads > self.max_entries {
            return Err(StoreError::invalid_config(
                "max_loads",
                "must not exceed the entry limit",
            ));
        }
        if self.max_waiters_per_page > self.max_waiters {
            return Err(StoreError::invalid_config(
                "max_waiters_per_page",
                "must not exceed the global waiter limit",
            ));
        }
        Ok(())
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            page_pool_capacity: 64 * 1024 * 1024,
            max_page_bytes: 2 * 1024 * 1024,
            max_inflight_bytes: 16 * 1024 * 1024,
            max_entries: 4_096,
            max_loads: 4,
            max_waiters: 4_096,
            max_waiters_per_page: 128,
            max_leases: 4_096,
            trace_capacity: 1_024,
        }
    }
}

/// An explicit policy lease over an authenticated immutable page.
///
/// Dropping the lease returns its slot and, for a retiring page, releases the
/// byte charge only when this is the final lease.
pub struct PageLease {
    page: VerifiedPage,
    cache: Weak<CacheInner>,
    key: PageKey,
    generation: u64,
}

impl PageLease {
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.page.bytes()
    }

    #[must_use]
    pub const fn key(&self) -> PageKey {
        self.key
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
}

impl fmt::Debug for PageLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PageLease")
            .field("key", &self.key)
            .field("bytes", &self.len())
            .finish_non_exhaustive()
    }
}

impl Drop for PageLease {
    fn drop(&mut self) {
        if let Some(cache) = self.cache.upgrade() {
            cache.release_lease(self.key, self.generation);
        }
    }
}

/// Cloneable verified page cache.
#[derive(Clone)]
pub struct PageCache {
    inner: Arc<CacheInner>,
}

impl PageCache {
    pub fn new(reader: AsyncReader, config: CacheConfig) -> Result<Self, StoreError> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(CacheInner {
                reader,
                config,
                state: Mutex::new(CacheState::default()),
                metrics: Metrics::default(),
                trace: TraceSink::new(config.trace_capacity),
            }),
        })
    }

    /// Returns a lease for a demanded page, coalescing a physical miss with all
    /// other waiters for the same immutable specification.
    pub async fn get(&self, spec: PageSpec, control: Control) -> Result<PageLease, StoreError> {
        control.check()?;
        let bytes = spec.length();
        self.inner.metrics.add_demand_bytes(bytes);
        let allocation_bytes = self.inner.validate_spec(&spec)?;
        let key = spec.key();
        let started_wait = Instant::now();
        let (completion, receiver) = oneshot::channel();
        let mut immediate = None;
        let mut waiter_registration = None;

        {
            let mut state = lock_unpoisoned(&self.inner.state);
            if state.shutdown {
                return Err(StoreError::shutdown());
            }
            state.tick = state.tick.saturating_add(1);
            let tick = state.tick;

            if let Some(mut entry) = state.entries.remove(&key) {
                if let Err(error) = self
                    .inner
                    .validate_identity(&entry, &spec, allocation_bytes)
                {
                    state.entries.insert(key, entry);
                    return Err(error);
                }
                match &mut entry.state {
                    EntryState::Resident(resident) => {
                        self.inner.metrics.hit();
                        if state.ledger.leases + state.ledger.lease_reservations
                            >= self.inner.config.max_leases
                        {
                            state.entries.insert(key, entry);
                            return Err(StoreError::resource_exhausted(
                                "cache leases",
                                1,
                                as_u64(self.inner.config.max_leases),
                            ));
                        }
                        resident.last_touch = tick;
                        resident.leases += 1;
                        state.ledger.leases += 1;
                        immediate = Some(PageLease {
                            page: resident.page.clone(),
                            cache: Arc::downgrade(&self.inner),
                            key,
                            generation: entry.generation,
                        });
                        state.entries.insert(key, entry);
                        self.inner.trace.record(
                            TraceOutcome::Hit,
                            AccessReason::Demand,
                            key,
                            bytes,
                        );
                    }
                    EntryState::Loading(loading) => {
                        self.inner.metrics.miss();
                        self.inner.trace.record(
                            TraceOutcome::Miss,
                            AccessReason::Demand,
                            key,
                            bytes,
                        );
                        if !loading.accepting_waiters {
                            state.entries.insert(key, entry);
                            return Err(StoreError::resource_exhausted(
                                "page reload pending cancellation",
                                bytes,
                                self.inner.config.page_pool_capacity,
                            ));
                        }
                        if let Err(error) = self.inner.reserve_waiter(&state, loading.waiters.len())
                        {
                            state.entries.insert(key, entry);
                            return Err(error);
                        }
                        let waiter_id = match state.next_waiter() {
                            Ok(waiter_id) => waiter_id,
                            Err(error) => {
                                state.entries.insert(key, entry);
                                return Err(error);
                            }
                        };
                        loading.waiters.push(Waiter {
                            id: waiter_id,
                            completion,
                        });
                        loading.last_touch = tick;
                        state.ledger.waiters += 1;
                        state.ledger.lease_reservations += 1;
                        waiter_registration = Some((entry.generation, waiter_id));
                        let outcome = if is_late_prefetch(loading.origin) {
                            self.inner.metrics.late_prefetch();
                            TraceOutcome::LatePrefetch
                        } else {
                            self.inner.metrics.coalesced_demand();
                            TraceOutcome::LoadCoalesced
                        };
                        state.entries.insert(key, entry);
                        self.inner
                            .trace
                            .record(outcome, AccessReason::Demand, key, bytes);
                    }
                    EntryState::Retiring(_) => {
                        state.entries.insert(key, entry);
                        return Err(StoreError::resource_exhausted(
                            "retiring page",
                            bytes,
                            self.inner.config.page_pool_capacity,
                        ));
                    }
                }
            } else {
                self.inner.metrics.miss();
                self.inner
                    .trace
                    .record(TraceOutcome::Miss, AccessReason::Demand, key, bytes);
                self.inner.reserve_waiter(&state, 0)?;
                let _handle = tokio::runtime::Handle::try_current()
                    .map_err(|_| StoreError::invariant("cache miss requires a Tokio runtime"))?;
                let generation = state.next_generation()?;
                let waiter_id = state.next_waiter()?;
                if !self
                    .inner
                    .reserve_load(&mut state, allocation_bytes, AccessReason::Demand)?
                {
                    return Err(StoreError::resource_exhausted(
                        "page pool",
                        bytes,
                        self.inner.config.page_pool_capacity,
                    ));
                }
                let cancellation = CancellationToken::new();
                state.ledger.waiters += 1;
                state.ledger.lease_reservations += 1;
                state.entries.insert(
                    key,
                    Entry {
                        fingerprint: spec.fingerprint(),
                        length: bytes,
                        allocation_bytes,
                        generation,
                        state: EntryState::Loading(LoadingEntry {
                            waiters: vec![Waiter {
                                id: waiter_id,
                                completion,
                            }],
                            cancellation: cancellation.clone(),
                            origin: AccessReason::Demand,
                            prefetch_interest: false,
                            accepting_waiters: true,
                            last_touch: tick,
                        }),
                    },
                );
                waiter_registration = Some((generation, waiter_id));
                match self.inner.submit_load(spec, generation, cancellation) {
                    Ok(()) => self.inner.trace.record(
                        TraceOutcome::LoadStarted,
                        AccessReason::Demand,
                        key,
                        bytes,
                    ),
                    Err(error) => {
                        self.inner
                            .unwind_unsubmitted_load(&mut state, key, generation)?;
                        self.inner.update_live_metrics(&state);
                        return Err(error);
                    }
                }
            }
            self.inner.update_live_metrics(&state);
        }

        if let Some(lease) = immediate {
            self.inner.mark_prefetch_use(key, lease.generation);
            return Ok(lease);
        }

        let (generation, waiter_id) = waiter_registration
            .ok_or_else(|| StoreError::invariant("cache waiter registration was lost"))?;
        let mut registration = WaiterRegistration {
            cache: Arc::downgrade(&self.inner),
            key,
            generation,
            waiter_id,
            armed: true,
        };
        let mut receiver = std::pin::pin!(receiver);

        loop {
            if let Err(error) = control.check() {
                registration.withdraw();
                self.inner
                    .metrics
                    .add_wait_nanoseconds(saturating_nanos(started_wait.elapsed().as_nanos()));
                self.inner
                    .trace
                    .record(TraceOutcome::Cancelled, AccessReason::Demand, key, bytes);
                return Err(error);
            }

            match tokio::time::timeout(Duration::from_millis(1), &mut receiver).await {
                Ok(Ok(result)) => {
                    registration.disarm();
                    self.inner
                        .metrics
                        .add_wait_nanoseconds(saturating_nanos(started_wait.elapsed().as_nanos()));
                    return result;
                }
                Ok(Err(_closed)) => {
                    registration.disarm();
                    return Err(StoreError::invariant(
                        "cache load dropped a waiter completion",
                    ));
                }
                Err(_elapsed) => {}
            }
        }
    }

    /// Submits a nonblocking-use hint. Admission is asynchronous; the outcome
    /// describes only whether the hint started or joined a bounded load.
    pub async fn prefetch(
        &self,
        spec: PageSpec,
        control: Control,
    ) -> Result<PrefetchOutcome, StoreError> {
        control.check()?;
        let bytes = spec.length();
        self.inner.metrics.add_prefetch_bytes(bytes);
        let allocation_bytes = self.inner.validate_spec(&spec)?;
        let key = spec.key();
        let outcome;

        {
            let mut state = lock_unpoisoned(&self.inner.state);
            if state.shutdown {
                return Err(StoreError::shutdown());
            }
            if let Some(mut entry) = state.entries.remove(&key) {
                if let Err(error) = self
                    .inner
                    .validate_identity(&entry, &spec, allocation_bytes)
                {
                    state.entries.insert(key, entry);
                    return Err(error);
                }
                outcome = match &mut entry.state {
                    EntryState::Loading(loading)
                        if loading.accepting_waiters
                            && loading.origin == AccessReason::Demand
                            && !loading.prefetch_interest =>
                    {
                        loading.prefetch_interest = true;
                        self.inner.metrics.coalesced_prefetch();
                        self.inner.trace.record(
                            TraceOutcome::PrefetchCoalesced,
                            AccessReason::Prefetch,
                            key,
                            bytes,
                        );
                        PrefetchOutcome::CoalescedWithDemand
                    }
                    EntryState::Loading(loading) if !loading.accepting_waiters => {
                        self.inner.metrics.dropped_prefetch();
                        self.inner.trace.record(
                            TraceOutcome::PrefetchDropped,
                            AccessReason::Prefetch,
                            key,
                            bytes,
                        );
                        PrefetchOutcome::DroppedBudget
                    }
                    EntryState::Loading(_) | EntryState::Resident(_) => {
                        self.inner.metrics.redundant_prefetch();
                        self.inner.trace.record(
                            TraceOutcome::PrefetchRedundant,
                            AccessReason::Prefetch,
                            key,
                            bytes,
                        );
                        PrefetchOutcome::Redundant
                    }
                    EntryState::Retiring(_) => {
                        self.inner.metrics.dropped_prefetch();
                        self.inner.trace.record(
                            TraceOutcome::PrefetchDropped,
                            AccessReason::Prefetch,
                            key,
                            bytes,
                        );
                        PrefetchOutcome::DroppedBudget
                    }
                };
                state.entries.insert(key, entry);
            } else {
                let _handle = tokio::runtime::Handle::try_current().map_err(|_| {
                    StoreError::invariant("cache prefetch requires a Tokio runtime")
                })?;
                state.tick = state.tick.saturating_add(1);
                let tick = state.tick;
                let generation = state.next_generation()?;
                if !self
                    .inner
                    .reserve_load(&mut state, allocation_bytes, AccessReason::Prefetch)?
                {
                    self.inner.metrics.dropped_prefetch();
                    self.inner.trace.record(
                        TraceOutcome::PrefetchDropped,
                        AccessReason::Prefetch,
                        key,
                        bytes,
                    );
                    return Ok(PrefetchOutcome::DroppedBudget);
                }
                let cancellation = CancellationToken::new();
                state.entries.insert(
                    key,
                    Entry {
                        fingerprint: spec.fingerprint(),
                        length: bytes,
                        allocation_bytes,
                        generation,
                        state: EntryState::Loading(LoadingEntry {
                            waiters: Vec::new(),
                            cancellation: cancellation.clone(),
                            origin: AccessReason::Prefetch,
                            prefetch_interest: true,
                            accepting_waiters: true,
                            last_touch: tick,
                        }),
                    },
                );
                outcome = match self.inner.submit_load(spec, generation, cancellation) {
                    Ok(()) => {
                        self.inner.trace.record(
                            TraceOutcome::LoadStarted,
                            AccessReason::Prefetch,
                            key,
                            bytes,
                        );
                        PrefetchOutcome::Started
                    }
                    Err(error) => {
                        self.inner
                            .unwind_unsubmitted_load(&mut state, key, generation)?;
                        self.inner.update_live_metrics(&state);
                        if error == StoreError::QueueFull {
                            self.inner.metrics.dropped_prefetch();
                            self.inner.trace.record(
                                TraceOutcome::PrefetchDropped,
                                AccessReason::Prefetch,
                                key,
                                bytes,
                            );
                            return Ok(PrefetchOutcome::DroppedBudget);
                        }
                        return Err(error);
                    }
                };
            }
            self.inner.update_live_metrics(&state);
        }
        Ok(outcome)
    }

    /// Invalidates a page. A leased page becomes retiring and remains charged.
    pub fn invalidate(&self, key: PageKey) -> bool {
        self.inner.invalidate(key)
    }

    /// Stops admission, cancels physical loads, fails current waiters, and
    /// retires leased pages. Worker completions release in-flight charges.
    pub fn shutdown(&self) {
        self.inner.shutdown();
    }

    #[must_use]
    pub fn metrics(&self) -> MetricsSnapshot {
        self.inner.metrics.snapshot(self.inner.trace.dropped())
    }

    #[must_use]
    pub fn trace_sink(&self) -> TraceSink {
        self.inner.trace.clone()
    }
}

impl fmt::Debug for PageCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PageCache")
            .field("config", &self.inner.config)
            .field("metrics", &self.metrics())
            .finish_non_exhaustive()
    }
}

struct CacheInner {
    reader: AsyncReader,
    config: CacheConfig,
    state: Mutex<CacheState>,
    metrics: Metrics,
    trace: TraceSink,
}

impl CacheInner {
    fn validate_spec(&self, spec: &PageSpec) -> Result<u64, StoreError> {
        let length = spec.length();
        if length == 0 || length > self.config.max_page_bytes {
            return Err(StoreError::resource_exhausted(
                "supported page bytes",
                length,
                self.config.max_page_bytes,
            ));
        }
        let allocation_bytes = spec.allocation_bytes()?;
        if allocation_bytes > self.config.page_pool_capacity
            || allocation_bytes > self.config.max_inflight_bytes
        {
            return Err(StoreError::resource_exhausted(
                "aligned page allocation",
                allocation_bytes,
                self.config
                    .page_pool_capacity
                    .min(self.config.max_inflight_bytes),
            ));
        }
        Ok(allocation_bytes)
    }

    fn validate_identity(
        &self,
        entry: &Entry,
        spec: &PageSpec,
        allocation_bytes: u64,
    ) -> Result<(), StoreError> {
        if entry.length != spec.length()
            || entry.allocation_bytes != allocation_bytes
            || entry.fingerprint != spec.fingerprint()
        {
            return Err(StoreError::Integrity {
                kind: "cache key fingerprint mismatch",
            });
        }
        Ok(())
    }

    fn reserve_waiter(
        &self,
        state: &CacheState,
        per_page_waiters: usize,
    ) -> Result<(), StoreError> {
        if state.ledger.waiters >= self.config.max_waiters {
            return Err(StoreError::resource_exhausted(
                "cache waiters",
                1,
                as_u64(self.config.max_waiters),
            ));
        }
        if per_page_waiters >= self.config.max_waiters_per_page {
            return Err(StoreError::resource_exhausted(
                "per-page cache waiters",
                1,
                as_u64(self.config.max_waiters_per_page),
            ));
        }
        if state.ledger.leases + state.ledger.lease_reservations >= self.config.max_leases {
            return Err(StoreError::resource_exhausted(
                "cache leases",
                1,
                as_u64(self.config.max_leases),
            ));
        }
        Ok(())
    }

    fn reserve_load(
        &self,
        state: &mut CacheState,
        allocation_bytes: u64,
        reason: AccessReason,
    ) -> Result<bool, StoreError> {
        if allocation_bytes > self.config.max_inflight_bytes
            || state.ledger.inflight_bytes > self.config.max_inflight_bytes - allocation_bytes
            || state.ledger.active_loads >= self.config.max_loads
        {
            return Ok(false);
        }

        let maximum_retained_before_load = self.config.page_pool_capacity - allocation_bytes;
        let mut projected_pool_bytes = state.ledger.page_pool_bytes;
        let mut projected_entries = state.entries.len();
        let mut candidates: Vec<_> = state
            .entries
            .iter()
            .filter_map(|(key, entry)| match &entry.state {
                EntryState::Resident(resident) if resident.leases == 0 => {
                    Some((*key, resident.last_touch, entry.allocation_bytes))
                }
                _ => None,
            })
            .collect();
        candidates.sort_unstable_by_key(|(key, last_touch, _bytes)| (*last_touch, *key));

        let mut victims = Vec::new();
        for (key, _last_touch, candidate_bytes) in candidates {
            if projected_pool_bytes <= maximum_retained_before_load
                && projected_entries < self.config.max_entries
            {
                break;
            }
            projected_pool_bytes =
                projected_pool_bytes
                    .checked_sub(candidate_bytes)
                    .ok_or(StoreError::Invariant {
                        problem: "cache victim charge exceeded the page-pool ledger",
                    })?;
            projected_entries = projected_entries
                .checked_sub(1)
                .ok_or(StoreError::Invariant {
                    problem: "cache victim count exceeded retained entries",
                })?;
            victims.push(key);
        }

        if projected_pool_bytes > maximum_retained_before_load
            || projected_entries >= self.config.max_entries
        {
            return Ok(false);
        }

        let reserved_pool_bytes = projected_pool_bytes.checked_add(allocation_bytes).ok_or(
            StoreError::ArithmeticOverflow {
                context: "reserving an aligned cache page allocation",
            },
        )?;
        let reserved_inflight_bytes = state
            .ledger
            .inflight_bytes
            .checked_add(allocation_bytes)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "reserving aligned in-flight page bytes",
            })?;
        let reserved_loads =
            state
                .ledger
                .active_loads
                .checked_add(1)
                .ok_or(StoreError::ArithmeticOverflow {
                    context: "reserving a cache load slot",
                })?;

        for victim in victims {
            self.evict_unleased(state, victim, reason);
        }
        debug_assert_eq!(state.ledger.page_pool_bytes, projected_pool_bytes);
        debug_assert_eq!(state.entries.len(), projected_entries);
        state.ledger.active_loads = reserved_loads;
        state.ledger.page_pool_bytes = reserved_pool_bytes;
        state.ledger.inflight_bytes = reserved_inflight_bytes;
        Ok(true)
    }

    fn evict_unleased(&self, state: &mut CacheState, key: PageKey, reason: AccessReason) {
        let Some(entry) = state.entries.remove(&key) else {
            return;
        };
        let EntryState::Resident(resident) = entry.state else {
            state.entries.insert(key, entry);
            return;
        };
        debug_assert_eq!(resident.leases, 0);
        state.ledger.page_pool_bytes -= entry.allocation_bytes;
        state.ledger.resident_bytes -= entry.allocation_bytes;
        self.metrics.eviction();
        self.trace
            .record(TraceOutcome::Evicted, reason, key, entry.length);
        if resident.prefetched_unused {
            self.metrics.wasted_prefetch();
            self.trace.record(
                TraceOutcome::PrefetchWasted,
                AccessReason::Prefetch,
                key,
                entry.length,
            );
        }
    }

    fn submit_load(
        self: &Arc<Self>,
        spec: PageSpec,
        generation: u64,
        cancellation: CancellationToken,
    ) -> Result<(), StoreError> {
        let key = spec.key();
        let cache = Arc::downgrade(self);
        self.reader.submit_observed(
            spec,
            Control::with_cancellation(cancellation),
            move |observed| {
                if let Some(cache) = cache.upgrade() {
                    cache.finish_observed_load(key, generation, observed);
                }
            },
        )
    }

    fn unwind_unsubmitted_load(
        &self,
        state: &mut CacheState,
        key: PageKey,
        generation: u64,
    ) -> Result<(), StoreError> {
        let Some(entry) = state.entries.remove(&key) else {
            return Err(StoreError::invariant(
                "unsubmitted cache load lost its reserved entry",
            ));
        };
        if entry.generation != generation {
            state.entries.insert(key, entry);
            return Err(StoreError::invariant(
                "unsubmitted cache load changed generation",
            ));
        }
        let EntryState::Loading(loading) = entry.state else {
            state.entries.insert(key, entry);
            return Err(StoreError::invariant(
                "unsubmitted cache load changed state",
            ));
        };

        state.ledger.active_loads -= 1;
        state.ledger.page_pool_bytes -= entry.allocation_bytes;
        state.ledger.inflight_bytes -= entry.allocation_bytes;
        state.ledger.waiters -= loading.waiters.len();
        state.ledger.lease_reservations -= loading.waiters.len();
        Ok(())
    }

    fn finish_observed_load(
        self: &Arc<Self>,
        key: PageKey,
        generation: u64,
        (result, stats): (Result<VerifiedPage, StoreError>, ReadStats),
    ) {
        self.metrics.add_physical_read_bytes(stats.physical_bytes());
        self.metrics.add_io_nanoseconds(stats.io_nanoseconds());
        self.complete_load(key, generation, result);
    }

    fn complete_load(
        self: &Arc<Self>,
        key: PageKey,
        generation: u64,
        result: Result<VerifiedPage, StoreError>,
    ) {
        let mut lease_deliveries = Vec::new();
        let mut error_deliveries = Vec::new();
        {
            let mut state = lock_unpoisoned(&self.state);
            let Some(entry) = state.entries.remove(&key) else {
                return;
            };
            if entry.generation != generation {
                state.entries.insert(key, entry);
                return;
            }
            let EntryState::Loading(loading) = entry.state else {
                state.entries.insert(key, entry);
                return;
            };

            state.ledger.active_loads -= 1;
            state.ledger.inflight_bytes -= entry.allocation_bytes;
            state.ledger.waiters -= loading.waiters.len();
            state.ledger.lease_reservations -= loading.waiters.len();
            let publish = loading.accepting_waiters && !state.shutdown;
            let origin = loading.origin;
            let prefetch_interest = loading.prefetch_interest;

            match result {
                Ok(page) if publish => {
                    if page.key() != key
                        || as_u64(page.bytes().len()) != entry.length
                        || page.allocation_bytes() > entry.allocation_bytes
                    {
                        state.ledger.page_pool_bytes -= entry.allocation_bytes;
                        let error = StoreError::invariant(
                            "verified reader exceeded its reserved page identity or allocation",
                        );
                        error_deliveries.extend(
                            loading
                                .waiters
                                .into_iter()
                                .map(|waiter| (waiter.completion, error.clone())),
                        );
                        self.trace
                            .record(TraceOutcome::LoadFailed, origin, key, entry.length);
                    } else {
                        let waiter_count = loading.waiters.len();
                        state.ledger.resident_bytes += entry.allocation_bytes;
                        state.ledger.leases += waiter_count;
                        self.metrics.admission();
                        let resident = ResidentEntry {
                            page: page.clone(),
                            leases: waiter_count,
                            prefetched_unused: prefetched_unused_on_admission(
                                prefetch_interest,
                                waiter_count,
                            ),
                            last_touch: loading.last_touch,
                        };
                        for waiter in loading.waiters {
                            lease_deliveries.push((
                                waiter.completion,
                                PageLease {
                                    page: page.clone(),
                                    cache: Arc::downgrade(self),
                                    key,
                                    generation,
                                },
                            ));
                        }
                        state.entries.insert(
                            key,
                            Entry {
                                fingerprint: entry.fingerprint,
                                length: entry.length,
                                allocation_bytes: entry.allocation_bytes,
                                generation,
                                state: EntryState::Resident(resident),
                            },
                        );
                        self.trace
                            .record(TraceOutcome::Admitted, origin, key, entry.length);
                    }
                }
                Ok(_page) => {
                    state.ledger.page_pool_bytes -= entry.allocation_bytes;
                    let error = if state.shutdown {
                        StoreError::shutdown()
                    } else {
                        StoreError::cancelled()
                    };
                    error_deliveries.extend(
                        loading
                            .waiters
                            .into_iter()
                            .map(|waiter| (waiter.completion, error.clone())),
                    );
                }
                Err(error) => {
                    state.ledger.page_pool_bytes -= entry.allocation_bytes;
                    error_deliveries.extend(
                        loading
                            .waiters
                            .into_iter()
                            .map(|waiter| (waiter.completion, error.clone())),
                    );
                    self.trace
                        .record(TraceOutcome::LoadFailed, origin, key, entry.length);
                }
            }
            self.update_live_metrics(&state);
        }

        for (completion, lease) in lease_deliveries {
            // On a withdrawn receiver `send` returns the lease; dropping that
            // value releases its explicit count outside the state mutex.
            let _ = completion.send(Ok(lease));
        }
        for (completion, error) in error_deliveries {
            let _ = completion.send(Err(error));
        }
    }

    fn mark_prefetch_use(&self, key: PageKey, generation: u64) {
        let mut useful = false;
        let mut bytes = 0;
        {
            let mut state = lock_unpoisoned(&self.state);
            if let Some(entry) = state.entries.get_mut(&key)
                && entry.generation == generation
            {
                bytes = entry.length;
                let unused = match &mut entry.state {
                    EntryState::Resident(resident) => &mut resident.prefetched_unused,
                    EntryState::Retiring(retiring) => &mut retiring.prefetched_unused,
                    EntryState::Loading(_) => return,
                };
                if *unused {
                    *unused = false;
                    useful = true;
                }
            }
        }
        if useful {
            self.metrics.useful_prefetch();
            self.trace.record(
                TraceOutcome::PrefetchUseful,
                AccessReason::Demand,
                key,
                bytes,
            );
        }
    }

    fn withdraw_waiter(&self, key: PageKey, generation: u64, waiter_id: u64) {
        let mut state = lock_unpoisoned(&self.state);
        let Some(mut entry) = state.entries.remove(&key) else {
            return;
        };
        if entry.generation != generation {
            state.entries.insert(key, entry);
            return;
        }
        let EntryState::Loading(loading) = &mut entry.state else {
            state.entries.insert(key, entry);
            return;
        };
        let Some(position) = loading
            .waiters
            .iter()
            .position(|waiter| waiter.id == waiter_id)
        else {
            state.entries.insert(key, entry);
            return;
        };
        let _waiter = loading.waiters.swap_remove(position);
        state.ledger.waiters -= 1;
        state.ledger.lease_reservations -= 1;
        if loading.waiters.is_empty() && !loading.prefetch_interest {
            loading.accepting_waiters = false;
            loading.cancellation.cancel();
        }
        state.entries.insert(key, entry);
        self.update_live_metrics(&state);
    }

    fn release_lease(&self, key: PageKey, generation: u64) {
        let mut state = lock_unpoisoned(&self.state);
        let Some(mut entry) = state.entries.remove(&key) else {
            return;
        };
        if entry.generation != generation {
            state.entries.insert(key, entry);
            return;
        }
        let mut retain = true;
        match &mut entry.state {
            EntryState::Resident(resident) if resident.leases > 0 => {
                resident.leases -= 1;
                state.ledger.leases -= 1;
            }
            EntryState::Retiring(retiring) if retiring.leases > 0 => {
                retiring.leases -= 1;
                state.ledger.leases -= 1;
                if retiring.leases == 0 {
                    state.ledger.page_pool_bytes -= entry.allocation_bytes;
                    state.ledger.retiring_bytes -= entry.allocation_bytes;
                    if retiring.prefetched_unused {
                        self.metrics.wasted_prefetch();
                        self.trace.record(
                            TraceOutcome::PrefetchWasted,
                            AccessReason::Prefetch,
                            key,
                            entry.length,
                        );
                    }
                    retain = false;
                }
            }
            _ => {}
        }
        if retain {
            state.entries.insert(key, entry);
        }
        self.update_live_metrics(&state);
    }

    fn invalidate(&self, key: PageKey) -> bool {
        let mut state = lock_unpoisoned(&self.state);
        let Some(mut entry) = state.entries.remove(&key) else {
            return false;
        };
        let retain = match entry.state {
            EntryState::Loading(ref mut loading) => {
                loading.accepting_waiters = false;
                loading.cancellation.cancel();
                true
            }
            EntryState::Resident(ref resident) if resident.leases == 0 => {
                state.ledger.page_pool_bytes -= entry.allocation_bytes;
                state.ledger.resident_bytes -= entry.allocation_bytes;
                if resident.prefetched_unused {
                    self.metrics.wasted_prefetch();
                    self.trace.record(
                        TraceOutcome::PrefetchWasted,
                        AccessReason::Prefetch,
                        key,
                        entry.length,
                    );
                }
                false
            }
            EntryState::Resident(resident) => {
                state.ledger.resident_bytes -= entry.allocation_bytes;
                state.ledger.retiring_bytes += entry.allocation_bytes;
                entry.state = EntryState::Retiring(RetiringEntry {
                    _page: resident.page,
                    leases: resident.leases,
                    prefetched_unused: resident.prefetched_unused,
                });
                self.trace.record(
                    TraceOutcome::Retired,
                    AccessReason::Demand,
                    key,
                    entry.length,
                );
                true
            }
            EntryState::Retiring(_) => true,
        };
        if retain {
            state.entries.insert(key, entry);
        }
        self.update_live_metrics(&state);
        true
    }

    fn shutdown(&self) {
        let mut failures = Vec::new();
        {
            let mut state = lock_unpoisoned(&self.state);
            if state.shutdown {
                return;
            }
            state.shutdown = true;
            let keys: Vec<_> = state.entries.keys().copied().collect();
            for key in keys {
                let Some(mut entry) = state.entries.remove(&key) else {
                    continue;
                };
                let retain = match entry.state {
                    EntryState::Loading(ref mut loading) => {
                        loading.accepting_waiters = false;
                        loading.cancellation.cancel();
                        state.ledger.waiters -= loading.waiters.len();
                        state.ledger.lease_reservations -= loading.waiters.len();
                        failures.extend(loading.waiters.drain(..).map(|waiter| waiter.completion));
                        true
                    }
                    EntryState::Resident(ref resident) if resident.leases == 0 => {
                        state.ledger.page_pool_bytes -= entry.allocation_bytes;
                        state.ledger.resident_bytes -= entry.allocation_bytes;
                        if resident.prefetched_unused {
                            self.metrics.wasted_prefetch();
                            self.trace.record(
                                TraceOutcome::PrefetchWasted,
                                AccessReason::Prefetch,
                                key,
                                entry.length,
                            );
                        }
                        false
                    }
                    EntryState::Resident(resident) => {
                        state.ledger.resident_bytes -= entry.allocation_bytes;
                        state.ledger.retiring_bytes += entry.allocation_bytes;
                        entry.state = EntryState::Retiring(RetiringEntry {
                            _page: resident.page,
                            leases: resident.leases,
                            prefetched_unused: resident.prefetched_unused,
                        });
                        true
                    }
                    EntryState::Retiring(_) => true,
                };
                if retain {
                    state.entries.insert(key, entry);
                }
            }
            self.update_live_metrics(&state);
        }
        for completion in failures {
            let _ = completion.send(Err(StoreError::shutdown()));
        }
        self.reader.shutdown();
    }

    fn update_live_metrics(&self, state: &CacheState) {
        debug_assert_eq!(
            state
                .ledger
                .inflight_bytes
                .checked_add(state.ledger.resident_bytes)
                .and_then(|value| value.checked_add(state.ledger.retiring_bytes)),
            Some(state.ledger.page_pool_bytes),
            "page-pool ledger must equal in-flight + resident + retiring charges"
        );
        self.metrics.set_live(
            as_u64(state.ledger.active_loads),
            state.ledger.page_pool_bytes,
            state.ledger.inflight_bytes,
            state.ledger.resident_bytes,
            state.ledger.retiring_bytes,
            as_u64(state.ledger.leases),
        );
    }
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<PageKey, Entry>,
    ledger: Ledger,
    generation: u64,
    waiter: u64,
    tick: u64,
    shutdown: bool,
}

impl CacheState {
    fn next_generation(&mut self) -> Result<u64, StoreError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| StoreError::invariant("cache generation exhausted"))?;
        Ok(self.generation)
    }

    fn next_waiter(&mut self) -> Result<u64, StoreError> {
        self.waiter = self
            .waiter
            .checked_add(1)
            .ok_or_else(|| StoreError::invariant("cache waiter identity exhausted"))?;
        Ok(self.waiter)
    }
}

#[derive(Default)]
struct Ledger {
    page_pool_bytes: u64,
    inflight_bytes: u64,
    resident_bytes: u64,
    retiring_bytes: u64,
    active_loads: usize,
    waiters: usize,
    lease_reservations: usize,
    leases: usize,
}

struct Entry {
    fingerprint: Digest,
    /// Logical page bytes used for identity, counters, and trace events.
    length: u64,
    /// Aligned payload allocation charged through every cache state.
    allocation_bytes: u64,
    generation: u64,
    state: EntryState,
}

enum EntryState {
    Loading(LoadingEntry),
    Resident(ResidentEntry),
    Retiring(RetiringEntry),
}

struct LoadingEntry {
    waiters: Vec<Waiter>,
    cancellation: CancellationToken,
    origin: AccessReason,
    prefetch_interest: bool,
    accepting_waiters: bool,
    /// Most recent admitted access while this generation was loading. Worker
    /// completion order must not rewrite request-order LRU policy.
    last_touch: u64,
}

struct ResidentEntry {
    page: VerifiedPage,
    leases: usize,
    prefetched_unused: bool,
    last_touch: u64,
}

struct RetiringEntry {
    _page: VerifiedPage,
    leases: usize,
    prefetched_unused: bool,
}

struct Waiter {
    id: u64,
    completion: oneshot::Sender<Result<PageLease, StoreError>>,
}

struct WaiterRegistration {
    cache: Weak<CacheInner>,
    key: PageKey,
    generation: u64,
    waiter_id: u64,
    armed: bool,
}

impl WaiterRegistration {
    fn disarm(&mut self) {
        self.armed = false;
    }

    fn withdraw(&mut self) {
        if self.armed {
            if let Some(cache) = self.cache.upgrade() {
                cache.withdraw_waiter(self.key, self.generation, self.waiter_id);
            }
            self.armed = false;
        }
    }
}

impl Drop for WaiterRegistration {
    fn drop(&mut self) {
        self.withdraw();
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn saturating_nanos(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn is_late_prefetch(origin: AccessReason) -> bool {
    origin == AccessReason::Prefetch
}

fn prefetched_unused_on_admission(prefetch_interest: bool, demand_waiters: usize) -> bool {
    prefetch_interest && demand_waiters == 0
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use runnel_fixture::{MultiPageFixture, PAGE_SIZE};
    use runnel_format::Limits;
    use tempfile::TempDir;

    use super::{
        AccessReason, CacheConfig, EntryState, PageCache, PrefetchOutcome, is_late_prefetch,
        lock_unpoisoned, prefetched_unused_on_admission,
    };
    use crate::async_io::WorkerGate;
    use crate::{
        ArtifactSource, AsyncReader, AsyncReaderConfig, CancellationToken, Control, ErrorCategory,
        PageSpec, StoreError, SyncReader, TraceOutcome,
    };

    struct OpenFixture {
        _temporary: TempDir,
        specs: Vec<PageSpec>,
    }

    fn open_fixture() -> OpenFixture {
        let temporary = tempfile::tempdir().expect("temporary root");
        let root = temporary.path().join("artifact");
        let fixture = MultiPageFixture::build();
        let identity = fixture.write_new(&root).expect("write fixture");
        let source = ArtifactSource::open(&root).expect("open artifact source");
        let artifact = source
            .open_with_expected_id(
                Limits::default(),
                identity.artifact_id,
                &Control::unbounded(),
            )
            .expect("open stored artifact");
        let specs = artifact
            .page_specs()
            .collect::<Result<Vec<_>, _>>()
            .expect("page specifications");
        OpenFixture {
            _temporary: temporary,
            specs,
        }
    }

    fn one_page_config() -> CacheConfig {
        CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE))
            .expect("cache config")
            .with_max_inflight_bytes(u64::from(PAGE_SIZE))
            .with_max_entries(1)
            .with_max_loads(1)
    }

    fn gated_cache(config: CacheConfig) -> (PageCache, WorkerGate) {
        gated_cache_with_queue(config, 8)
    }

    fn gated_cache_with_queue(
        config: CacheConfig,
        queue_capacity: usize,
    ) -> (PageCache, WorkerGate) {
        let reader = AsyncReader::new(
            SyncReader::new(),
            AsyncReaderConfig::new(1, queue_capacity).expect("async config"),
        )
        .expect("async reader");
        let gate = WorkerGate::new();
        reader.install_worker_gate(gate.clone());
        let cache = PageCache::new(reader, config).expect("page cache");
        (cache, gate)
    }

    async fn wait_until(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !predicate() {
            assert!(Instant::now() < deadline, "condition did not become true");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    fn wait_until_sync(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !predicate() {
            assert!(Instant::now() < deadline, "condition did not become true");
            std::thread::yield_now();
        }
    }

    #[test]
    fn exact_one_page_budget_is_valid() {
        let config = CacheConfig::new(65_536, 65_536).expect("exact page budget");
        assert_eq!(config.page_pool_capacity(), 65_536);
        assert_eq!(config.max_page_bytes(), 65_536);
        assert_eq!(config.max_inflight_bytes(), 65_536);
    }

    #[test]
    fn rejects_a_pool_that_cannot_hold_one_supported_page() {
        let error = CacheConfig::new(65_535, 65_536).expect_err("undersized pool");
        assert_eq!(error.category(), ErrorCategory::InvalidInput);
    }

    #[test]
    fn configuration_charges_aligned_page_allocations() {
        let error = CacheConfig::new(17, 17).expect_err("17 bytes allocate a 64-byte payload");
        assert_eq!(error.category(), ErrorCategory::InvalidInput);
        let overflow =
            CacheConfig::new(u64::MAX, u64::MAX).expect_err("aligned maximum must not overflow");
        assert_eq!(overflow.category(), ErrorCategory::InvalidInput);

        let config = CacheConfig::new(64, 17).expect("aligned page fits exactly");
        assert_eq!(config.page_pool_capacity(), 64);
        assert_eq!(config.max_inflight_bytes(), 64);
    }

    #[test]
    fn validates_all_derived_limits_at_cache_construction_boundary() {
        let config = CacheConfig::new(65_536, 65_536)
            .expect("base config")
            .with_max_loads(0);
        let error = config.validate().expect_err("zero load slots");
        assert_eq!(error.category(), ErrorCategory::InvalidInput);

        let inflight = CacheConfig::new(131_072, 65_536)
            .expect("base config")
            .with_max_inflight_bytes(65_535);
        assert!(inflight.validate().is_err());
    }

    #[test]
    fn hostile_count_limits_are_rejected_without_allocating() {
        let base = one_page_config();
        let hostile = [
            base.with_max_entries(usize::MAX),
            base.with_max_loads(usize::MAX),
            base.with_waiter_limits(usize::MAX, 1),
            base.with_waiter_limits(1, usize::MAX),
            base.with_max_leases(usize::MAX),
            base.with_trace_capacity(usize::MAX),
        ];
        for config in hostile {
            let error = config.validate().expect_err("hostile limit must fail");
            assert_eq!(error.category(), ErrorCategory::InvalidInput);
        }
    }

    #[test]
    fn load_origin_truth_table_keeps_late_and_useful_prefetch_distinct() {
        assert!(!is_late_prefetch(AccessReason::Demand));
        assert!(is_late_prefetch(AccessReason::Prefetch));

        assert!(!prefetched_unused_on_admission(false, 0));
        assert!(prefetched_unused_on_admission(true, 0));
        assert!(!prefetched_unused_on_admission(true, 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn waiter_limit_failure_preserves_the_live_loading_entry() {
        let fixture = open_fixture();
        let (cache, gate) = gated_cache(
            one_page_config()
                .with_waiter_limits(1, 1)
                .with_max_leases(2),
        );
        let first_cache = cache.clone();
        let specification = fixture.specs[0].clone();
        let first =
            tokio::spawn(async move { first_cache.get(specification, Control::unbounded()).await });
        gate.wait_until_entered();

        let rejected = cache
            .get(fixture.specs[0].clone(), Control::unbounded())
            .await;
        let loading = cache.metrics();
        gate.release();

        let rejected = rejected.expect_err("second waiter must exceed the configured limit");
        assert_eq!(rejected.category(), ErrorCategory::ResourceExhausted);
        assert_eq!(loading.active_loads, 1);
        assert_eq!(loading.inflight_bytes, u64::from(PAGE_SIZE));
        assert_eq!(loading.page_pool_bytes, u64::from(PAGE_SIZE));

        let lease = first
            .await
            .expect("first task")
            .expect("original waiter remains attached");
        assert_eq!(lease.len(), PAGE_SIZE as usize);
        let admitted = cache.metrics();
        assert_eq!(admitted.active_loads, 0);
        assert_eq!(admitted.inflight_bytes, 0);
        assert_eq!(admitted.resident_bytes, u64::from(PAGE_SIZE));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_full_prefetch_is_dropped_and_exactly_unwound() {
        let fixture = open_fixture();
        let page_bytes = u64::from(PAGE_SIZE);
        let config = CacheConfig::new(3 * page_bytes, page_bytes)
            .expect("three-page cache")
            .with_max_inflight_bytes(3 * page_bytes)
            .with_max_entries(3)
            .with_max_loads(3);
        let (cache, gate) = gated_cache_with_queue(config, 1);
        let trace = cache.trace_sink();

        assert_eq!(
            cache
                .prefetch(fixture.specs[0].clone(), Control::unbounded())
                .await
                .expect("first prefetch"),
            PrefetchOutcome::Started
        );
        gate.wait_until_entered();
        assert_eq!(
            cache
                .prefetch(fixture.specs[1].clone(), Control::unbounded())
                .await
                .expect("queued prefetch"),
            PrefetchOutcome::Started
        );
        assert_eq!(
            cache
                .prefetch(fixture.specs[2].clone(), Control::unbounded())
                .await
                .expect("saturated prefetch is a bounded drop"),
            PrefetchOutcome::DroppedBudget
        );

        let saturated = cache.metrics();
        assert_eq!(saturated.active_loads, 2);
        assert_eq!(saturated.inflight_bytes, 2 * page_bytes);
        assert_eq!(saturated.page_pool_bytes, 2 * page_bytes);
        assert_eq!(saturated.dropped_prefetches, 1);
        let rejected_key = fixture.specs[2].key();
        let rejected_prefetch_events: Vec<_> = trace
            .drain()
            .into_iter()
            .filter(|event| event.key == rejected_key)
            .collect();
        assert_eq!(rejected_prefetch_events.len(), 1);
        assert_eq!(
            rejected_prefetch_events[0].outcome,
            TraceOutcome::PrefetchDropped
        );
        assert_eq!(rejected_prefetch_events[0].reason, AccessReason::Prefetch);
        assert_eq!(rejected_prefetch_events[0].bytes, fixture.specs[2].length());
        assert_eq!(trace.dropped(), 0);

        let demand_error = cache
            .get(fixture.specs[2].clone(), Control::unbounded())
            .await
            .expect_err("queue-full demand receives its resource error");
        assert_eq!(demand_error, StoreError::QueueFull);
        let after_demand_rejection = cache.metrics();
        assert_eq!(after_demand_rejection.active_loads, 2);
        assert_eq!(after_demand_rejection.inflight_bytes, 2 * page_bytes);
        assert_eq!(after_demand_rejection.page_pool_bytes, 2 * page_bytes);
        assert_eq!(after_demand_rejection.dropped_prefetches, 1);
        let rejected_demand_events: Vec<_> = trace
            .drain()
            .into_iter()
            .filter(|event| event.key == rejected_key)
            .collect();
        assert_eq!(rejected_demand_events.len(), 1);
        assert_eq!(rejected_demand_events[0].outcome, TraceOutcome::Miss);
        assert_eq!(rejected_demand_events[0].reason, AccessReason::Demand);
        assert_eq!(rejected_demand_events[0].bytes, fixture.specs[2].length());
        assert_eq!(trace.dropped(), 0);

        gate.release();
        wait_until(|| cache.metrics().active_loads == 0).await;
        assert_eq!(cache.metrics().resident_bytes, 2 * page_bytes);
        assert_eq!(
            cache
                .prefetch(fixture.specs[2].clone(), Control::unbounded())
                .await
                .expect("failed submission left no stale entry"),
            PrefetchOutcome::Started
        );
        wait_until(|| cache.metrics().active_loads == 0).await;
        assert_eq!(
            cache.metrics().resident_bytes,
            2 * page_bytes + fixture.specs[2].allocation_bytes().unwrap()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn out_of_order_completion_preserves_coalesced_access_lru_order() {
        let fixture = open_fixture();
        let page_bytes = u64::from(PAGE_SIZE);
        let config = CacheConfig::new(2 * page_bytes, page_bytes)
            .expect("two-page cache")
            .with_max_inflight_bytes(2 * page_bytes)
            .with_max_entries(2)
            .with_max_loads(2);
        let reader = AsyncReader::new(
            SyncReader::new(),
            AsyncReaderConfig::new(2, 2).expect("async config"),
        )
        .expect("async reader");
        let recent_gate = WorkerGate::new();
        reader.install_worker_gate(recent_gate.clone());
        let cache = PageCache::new(reader, config).expect("page cache");
        let recent_spec = fixture.specs[0].clone();
        let older_spec = fixture.specs[1].clone();
        assert!(recent_spec.key() < older_spec.key());

        let first_recent_cache = cache.clone();
        let first_recent_spec = recent_spec.clone();
        let first_recent = tokio::spawn(async move {
            first_recent_cache
                .get(first_recent_spec, Control::unbounded())
                .await
        });
        recent_gate.wait_until_entered();

        let older_gate = WorkerGate::new();
        cache.inner.reader.install_worker_gate(older_gate.clone());
        let older_cache = cache.clone();
        let older_task_spec = older_spec.clone();
        let older =
            tokio::spawn(
                async move { older_cache.get(older_task_spec, Control::unbounded()).await },
            );
        older_gate.wait_until_entered();

        let coalesced_cache = cache.clone();
        let coalesced_spec = recent_spec.clone();
        let coalesced = tokio::spawn(async move {
            coalesced_cache
                .get(coalesced_spec, Control::unbounded())
                .await
        });
        wait_until(|| {
            let state = lock_unpoisoned(&cache.inner.state);
            matches!(
                state
                    .entries
                    .get(&recent_spec.key())
                    .map(|entry| &entry.state),
                Some(EntryState::Loading(loading)) if loading.waiters.len() == 2
            )
        })
        .await;

        // The most recently touched load completes first. The older load then
        // completes out of order and must retain its older request tick.
        recent_gate.release();
        let first_recent_lease = first_recent
            .await
            .expect("first recent task")
            .expect("first recent lease");
        let coalesced_lease = coalesced
            .await
            .expect("coalesced task")
            .expect("coalesced recent lease");
        older_gate.release();
        let older_lease = older.await.expect("older task").expect("older lease");
        drop(first_recent_lease);
        drop(coalesced_lease);
        drop(older_lease);

        let replacement = cache
            .get(fixture.specs[2].clone(), Control::unbounded())
            .await
            .expect("replacement page");
        let state = lock_unpoisoned(&cache.inner.state);
        assert!(state.entries.contains_key(&recent_spec.key()));
        assert!(!state.entries.contains_key(&older_spec.key()));
        assert!(state.entries.contains_key(&fixture.specs[2].key()));
        drop(state);
        drop(replacement);

        let evictions: Vec<_> = cache
            .trace_sink()
            .drain()
            .into_iter()
            .filter(|event| event.outcome == TraceOutcome::Evicted)
            .map(|event| event.key)
            .collect();
        assert_eq!(evictions, [older_spec.key()]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn one_waiter_withdrawal_preserves_the_other_interest() {
        let fixture = open_fixture();
        let key = fixture.specs[0].key();
        let (cache, gate) = gated_cache(
            one_page_config()
                .with_waiter_limits(2, 2)
                .with_max_leases(2),
        );
        let cancellation = CancellationToken::new();
        let first_cache = cache.clone();
        let first_spec = fixture.specs[0].clone();
        let first_control = Control::with_cancellation(cancellation.clone());
        let first = tokio::spawn(async move { first_cache.get(first_spec, first_control).await });
        gate.wait_until_entered();

        let second_cache = cache.clone();
        let second_spec = fixture.specs[0].clone();
        let second =
            tokio::spawn(async move { second_cache.get(second_spec, Control::unbounded()).await });
        wait_until(|| {
            let state = lock_unpoisoned(&cache.inner.state);
            matches!(
                state.entries.get(&key).map(|entry| &entry.state),
                Some(EntryState::Loading(loading)) if loading.waiters.len() == 2
            )
        })
        .await;

        cancellation.cancel();
        assert_eq!(
            first.await.expect("first task").unwrap_err(),
            StoreError::Cancelled
        );
        {
            let state = lock_unpoisoned(&cache.inner.state);
            let Some(EntryState::Loading(loading)) =
                state.entries.get(&key).map(|entry| &entry.state)
            else {
                panic!("shared physical load must remain active");
            };
            assert_eq!(loading.waiters.len(), 1);
            assert!(loading.accepting_waiters);
            assert!(!loading.cancellation.is_cancelled());
        }

        gate.release();
        let lease = second
            .await
            .expect("second task")
            .expect("remaining waiter completes");
        assert_eq!(lease.len(), PAGE_SIZE as usize);
        let completed = cache.metrics();
        assert_eq!(completed.active_loads, 0);
        assert_eq!(completed.inflight_bytes, 0);
        assert_eq!(completed.resident_bytes, u64::from(PAGE_SIZE));
        assert_eq!(completed.leases, 1);
        drop(lease);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn final_waiter_withdrawal_forbids_publication_until_worker_returns() {
        let fixture = open_fixture();
        let key = fixture.specs[0].key();
        let (cache, gate) = gated_cache(one_page_config());
        let cancellation = CancellationToken::new();
        let task_cache = cache.clone();
        let specification = fixture.specs[0].clone();
        let control = Control::with_cancellation(cancellation.clone());
        let demand = tokio::spawn(async move { task_cache.get(specification, control).await });
        gate.wait_until_entered();

        cancellation.cancel();
        assert_eq!(
            demand.await.expect("demand task").unwrap_err(),
            StoreError::Cancelled
        );
        let awaiting_worker = cache.metrics();
        assert_eq!(awaiting_worker.active_loads, 1);
        assert_eq!(awaiting_worker.inflight_bytes, u64::from(PAGE_SIZE));
        assert_eq!(awaiting_worker.page_pool_bytes, u64::from(PAGE_SIZE));
        assert_eq!(awaiting_worker.resident_bytes, 0);
        {
            let state = lock_unpoisoned(&cache.inner.state);
            let Some(EntryState::Loading(loading)) =
                state.entries.get(&key).map(|entry| &entry.state)
            else {
                panic!("cancelled physical load remains owned by the worker");
            };
            assert!(loading.waiters.is_empty());
            assert!(!loading.accepting_waiters);
            assert!(loading.cancellation.is_cancelled());
        }

        gate.release();
        wait_until(|| cache.metrics().active_loads == 0).await;
        let released = cache.metrics();
        assert_eq!(released.page_pool_bytes, 0);
        assert_eq!(released.inflight_bytes, 0);
        assert_eq!(released.resident_bytes, 0);
        assert_eq!(released.leases, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_generation_completion_cannot_publish_or_release_current_load() {
        let fixture = open_fixture();
        let key = fixture.specs[0].key();
        let (cache, gate) = gated_cache(one_page_config());
        let task_cache = cache.clone();
        let specification = fixture.specs[0].clone();
        let demand =
            tokio::spawn(async move { task_cache.get(specification, Control::unbounded()).await });
        gate.wait_until_entered();
        let generation = {
            let state = lock_unpoisoned(&cache.inner.state);
            state.entries.get(&key).unwrap().generation
        };

        cache.inner.complete_load(
            key,
            generation.checked_add(1).unwrap(),
            Err(StoreError::Cancelled),
        );
        let unchanged = cache.metrics();
        assert_eq!(unchanged.active_loads, 1);
        assert_eq!(unchanged.inflight_bytes, u64::from(PAGE_SIZE));
        assert_eq!(unchanged.page_pool_bytes, u64::from(PAGE_SIZE));
        assert_eq!(unchanged.resident_bytes, 0);

        gate.release();
        let lease = demand
            .await
            .expect("demand task")
            .expect("current generation completes");
        assert_eq!(lease.len(), PAGE_SIZE as usize);
        assert_eq!(cache.metrics().resident_bytes, u64::from(PAGE_SIZE));
        drop(lease);
    }

    #[test]
    fn admitted_load_completes_after_originating_runtime_is_dropped() {
        let fixture = open_fixture();
        let (cache, gate) = gated_cache(one_page_config());
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("first runtime");
        let outcome = runtime
            .block_on(cache.prefetch(fixture.specs[0].clone(), Control::unbounded()))
            .expect("prefetch admission");
        gate.wait_until_entered();

        drop(runtime);
        gate.release();
        wait_until_sync(|| cache.metrics().active_loads == 0);
        assert_eq!(outcome, PrefetchOutcome::Started);
        let completed = cache.metrics();
        assert_eq!(completed.inflight_bytes, 0);
        assert_eq!(completed.resident_bytes, u64::from(PAGE_SIZE));

        let second_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("second runtime");
        let lease = second_runtime
            .block_on(cache.get(fixture.specs[0].clone(), Control::unbounded()))
            .expect("resident page after runtime teardown");
        assert_eq!(lease.len(), PAGE_SIZE as usize);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn coalesced_prefetch_becomes_useful_after_all_demand_waiters_drop() {
        let fixture = open_fixture();
        let (cache, gate) = gated_cache(one_page_config());
        let demand_cache = cache.clone();
        let specification = fixture.specs[0].clone();
        let demand =
            tokio::spawn(
                async move { demand_cache.get(specification, Control::unbounded()).await },
            );
        gate.wait_until_entered();
        let outcome = cache
            .prefetch(fixture.specs[0].clone(), Control::unbounded())
            .await;
        demand.abort();
        let demand_result = demand.await;

        gate.release();
        wait_until(|| cache.metrics().active_loads == 0).await;
        assert_eq!(
            outcome.expect("coalesced prefetch"),
            PrefetchOutcome::CoalescedWithDemand
        );
        assert!(
            demand_result
                .expect_err("demand task was aborted")
                .is_cancelled()
        );
        assert_eq!(cache.metrics().resident_bytes, u64::from(PAGE_SIZE));
        let lease = cache
            .get(fixture.specs[0].clone(), Control::unbounded())
            .await
            .expect("consume retained prefetch");
        let snapshot = cache.metrics();
        assert_eq!(snapshot.coalesced_prefetches, 1);
        assert_eq!(snapshot.useful_prefetches, 1);
        assert_eq!(snapshot.wasted_prefetches, 0);
        drop(lease);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn coalesced_prefetch_becomes_wasted_after_all_demand_waiters_drop() {
        let fixture = open_fixture();
        let (cache, gate) = gated_cache(one_page_config());
        let demand_cache = cache.clone();
        let specification = fixture.specs[0].clone();
        let demand =
            tokio::spawn(
                async move { demand_cache.get(specification, Control::unbounded()).await },
            );
        gate.wait_until_entered();
        let outcome = cache
            .prefetch(fixture.specs[0].clone(), Control::unbounded())
            .await;
        demand.abort();
        let demand_result = demand.await;

        gate.release();
        wait_until(|| cache.metrics().active_loads == 0).await;
        assert_eq!(
            outcome.expect("coalesced prefetch"),
            PrefetchOutcome::CoalescedWithDemand
        );
        assert!(
            demand_result
                .expect_err("demand task was aborted")
                .is_cancelled()
        );
        assert!(cache.invalidate(fixture.specs[0].key()));
        let snapshot = cache.metrics();
        assert_eq!(snapshot.coalesced_prefetches, 1);
        assert_eq!(snapshot.useful_prefetches, 0);
        assert_eq!(snapshot.wasted_prefetches, 1);
        assert_eq!(snapshot.page_pool_bytes, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unused_prefetch_terminal_paths_keep_counter_and_trace_in_parity() {
        let fixture = open_fixture();
        let spec = fixture.specs[0].clone();
        let key = spec.key();

        let (invalidated_cache, invalidated_gate) = gated_cache(one_page_config());
        assert_eq!(
            invalidated_cache
                .prefetch(spec.clone(), Control::unbounded())
                .await
                .expect("prefetch for invalidation"),
            PrefetchOutcome::Started
        );
        invalidated_gate.wait_until_entered();
        invalidated_gate.release();
        wait_until(|| invalidated_cache.metrics().active_loads == 0).await;
        let invalidated_trace = invalidated_cache.trace_sink();
        let _admission_events = invalidated_trace.drain();

        assert!(invalidated_cache.invalidate(key));
        let invalidated = invalidated_cache.metrics();
        assert_eq!(invalidated.wasted_prefetches, 1);
        assert_eq!(invalidated.page_pool_bytes, 0);
        let invalidation_events = invalidated_trace.drain();
        assert_eq!(invalidation_events.len(), 1);
        assert_eq!(invalidation_events[0].outcome, TraceOutcome::PrefetchWasted);
        assert_eq!(invalidation_events[0].reason, AccessReason::Prefetch);
        assert_eq!(invalidation_events[0].key, key);
        assert_eq!(invalidation_events[0].bytes, spec.length());
        assert_eq!(invalidated_trace.dropped(), 0);

        let (shutdown_cache, shutdown_gate) = gated_cache(one_page_config());
        assert_eq!(
            shutdown_cache
                .prefetch(spec.clone(), Control::unbounded())
                .await
                .expect("prefetch for shutdown"),
            PrefetchOutcome::Started
        );
        shutdown_gate.wait_until_entered();
        shutdown_gate.release();
        wait_until(|| shutdown_cache.metrics().active_loads == 0).await;
        let shutdown_trace = shutdown_cache.trace_sink();
        let _admission_events = shutdown_trace.drain();

        shutdown_cache.shutdown();
        let shut_down = shutdown_cache.metrics();
        assert_eq!(shut_down.wasted_prefetches, 1);
        assert_eq!(shut_down.page_pool_bytes, 0);
        let shutdown_events = shutdown_trace.drain();
        assert_eq!(shutdown_events.len(), 1);
        assert_eq!(shutdown_events[0].outcome, TraceOutcome::PrefetchWasted);
        assert_eq!(shutdown_events[0].reason, AccessReason::Prefetch);
        assert_eq!(shutdown_events[0].key, key);
        assert_eq!(shutdown_events[0].bytes, spec.length());
        assert_eq!(shutdown_trace.dropped(), 0);
    }
}
