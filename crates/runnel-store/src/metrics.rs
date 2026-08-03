//! Bounded-cardinality cache and reader metrics.

use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

/// A point-in-time sample of the process resident set.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RssSample {
    /// Current resident bytes reported by the operating system.
    pub resident_bytes: u64,
    /// Peak resident bytes reported by the operating system.
    pub peak_bytes: u64,
}

impl RssSample {
    /// Samples Linux `/proc/self/status`.
    ///
    /// `None` means that the platform did not expose both bounded integer
    /// fields or that the sample could not be read. Cache behavior never
    /// depends on this best-effort observation.
    #[must_use]
    pub fn sample() -> Option<Self> {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        let mut resident_bytes = None;
        let mut peak_bytes = None;

        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmRSS:") {
                resident_bytes = parse_kib(value);
            } else if let Some(value) = line.strip_prefix("VmHWM:") {
                peak_bytes = parse_kib(value);
            }
        }

        Some(Self {
            resident_bytes: resident_bytes?,
            peak_bytes: peak_bytes?,
        })
    }
}

fn parse_kib(value: &str) -> Option<u64> {
    let mut fields = value.split_ascii_whitespace();
    let kib = fields.next()?.parse::<u64>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    kib.checked_mul(1024)
}

/// A consistent-enough lock-free metrics snapshot.
///
/// Counters are monotonic. Gauges are sampled independently and can describe
/// adjacent linearization points, which is appropriate for observability and
/// avoids placing metric collection on the cache policy's critical path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetricsSnapshot {
    pub demand_bytes: u64,
    pub prefetch_bytes: u64,
    pub physical_read_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub coalesced_demands: u64,
    pub late_prefetches: u64,
    pub coalesced_prefetches: u64,
    pub useful_prefetches: u64,
    pub wasted_prefetches: u64,
    pub redundant_prefetches: u64,
    pub dropped_prefetches: u64,
    pub wait_nanoseconds: u64,
    pub io_nanoseconds: u64,
    pub active_loads: u64,
    pub page_pool_bytes: u64,
    pub inflight_bytes: u64,
    pub resident_bytes: u64,
    pub retiring_bytes: u64,
    pub leases: u64,
    pub trace_events_dropped: u64,
    pub rss: Option<RssSample>,
}

#[derive(Debug, Default)]
pub(crate) struct Metrics {
    demand_bytes: AtomicU64,
    prefetch_bytes: AtomicU64,
    physical_read_bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    admissions: AtomicU64,
    evictions: AtomicU64,
    coalesced_demands: AtomicU64,
    late_prefetches: AtomicU64,
    coalesced_prefetches: AtomicU64,
    useful_prefetches: AtomicU64,
    wasted_prefetches: AtomicU64,
    redundant_prefetches: AtomicU64,
    dropped_prefetches: AtomicU64,
    wait_nanoseconds: AtomicU64,
    io_nanoseconds: AtomicU64,
    active_loads: AtomicU64,
    page_pool_bytes: AtomicU64,
    inflight_bytes: AtomicU64,
    resident_bytes: AtomicU64,
    retiring_bytes: AtomicU64,
    leases: AtomicU64,
}

impl Metrics {
    pub(crate) fn add_demand_bytes(&self, value: u64) {
        add(&self.demand_bytes, value);
    }

    pub(crate) fn add_prefetch_bytes(&self, value: u64) {
        add(&self.prefetch_bytes, value);
    }

    pub(crate) fn add_physical_read_bytes(&self, value: u64) {
        add(&self.physical_read_bytes, value);
    }

    pub(crate) fn hit(&self) {
        increment(&self.hits);
    }

    pub(crate) fn miss(&self) {
        increment(&self.misses);
    }

    pub(crate) fn admission(&self) {
        increment(&self.admissions);
    }

    pub(crate) fn eviction(&self) {
        increment(&self.evictions);
    }

    pub(crate) fn coalesced_demand(&self) {
        increment(&self.coalesced_demands);
    }

    pub(crate) fn late_prefetch(&self) {
        increment(&self.late_prefetches);
    }

    pub(crate) fn coalesced_prefetch(&self) {
        increment(&self.coalesced_prefetches);
    }

    pub(crate) fn useful_prefetch(&self) {
        increment(&self.useful_prefetches);
    }

    pub(crate) fn wasted_prefetch(&self) {
        increment(&self.wasted_prefetches);
    }

    pub(crate) fn redundant_prefetch(&self) {
        increment(&self.redundant_prefetches);
    }

    pub(crate) fn dropped_prefetch(&self) {
        increment(&self.dropped_prefetches);
    }

    pub(crate) fn add_wait_nanoseconds(&self, value: u64) {
        add(&self.wait_nanoseconds, value);
    }

    pub(crate) fn add_io_nanoseconds(&self, value: u64) {
        add(&self.io_nanoseconds, value);
    }

    pub(crate) fn set_live(
        &self,
        active_loads: u64,
        page_pool_bytes: u64,
        inflight_bytes: u64,
        resident_bytes: u64,
        retiring_bytes: u64,
        leases: u64,
    ) {
        self.active_loads.store(active_loads, Ordering::Relaxed);
        self.page_pool_bytes
            .store(page_pool_bytes, Ordering::Relaxed);
        self.inflight_bytes.store(inflight_bytes, Ordering::Relaxed);
        self.resident_bytes.store(resident_bytes, Ordering::Relaxed);
        self.retiring_bytes.store(retiring_bytes, Ordering::Relaxed);
        self.leases.store(leases, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self, trace_events_dropped: u64) -> MetricsSnapshot {
        MetricsSnapshot {
            demand_bytes: load(&self.demand_bytes),
            prefetch_bytes: load(&self.prefetch_bytes),
            physical_read_bytes: load(&self.physical_read_bytes),
            hits: load(&self.hits),
            misses: load(&self.misses),
            admissions: load(&self.admissions),
            evictions: load(&self.evictions),
            coalesced_demands: load(&self.coalesced_demands),
            late_prefetches: load(&self.late_prefetches),
            coalesced_prefetches: load(&self.coalesced_prefetches),
            useful_prefetches: load(&self.useful_prefetches),
            wasted_prefetches: load(&self.wasted_prefetches),
            redundant_prefetches: load(&self.redundant_prefetches),
            dropped_prefetches: load(&self.dropped_prefetches),
            wait_nanoseconds: load(&self.wait_nanoseconds),
            io_nanoseconds: load(&self.io_nanoseconds),
            active_loads: load(&self.active_loads),
            page_pool_bytes: load(&self.page_pool_bytes),
            inflight_bytes: load(&self.inflight_bytes),
            resident_bytes: load(&self.resident_bytes),
            retiring_bytes: load(&self.retiring_bytes),
            leases: load(&self.leases),
            trace_events_dropped,
            rss: RssSample::sample(),
        }
    }
}

fn increment(value: &AtomicU64) {
    add(value, 1);
}

fn add(value: &AtomicU64, amount: u64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(amount))
    });
}

fn load(value: &AtomicU64) -> u64 {
    value.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::{Metrics, RssSample, parse_kib};

    #[test]
    fn parses_proc_kib_without_accepting_extra_fields() {
        assert_eq!(parse_kib(" 123 kB"), Some(125_952));
        assert_eq!(parse_kib("123 MB"), None);
        assert_eq!(parse_kib("123 kB extra"), None);
        assert_eq!(parse_kib("18446744073709551615 kB"), None);
    }

    #[test]
    fn counters_saturate_and_live_gauges_replace() {
        let metrics = Metrics::default();
        metrics.add_demand_bytes(u64::MAX);
        metrics.add_demand_bytes(1);
        metrics.hit();
        metrics.set_live(1, 2, 3, 4, 5, 6);
        let snapshot = metrics.snapshot(7);
        assert_eq!(snapshot.demand_bytes, u64::MAX);
        assert_eq!(snapshot.hits, 1);
        assert_eq!(snapshot.active_loads, 1);
        assert_eq!(snapshot.page_pool_bytes, 2);
        assert_eq!(snapshot.inflight_bytes, 3);
        assert_eq!(snapshot.resident_bytes, 4);
        assert_eq!(snapshot.retiring_bytes, 5);
        assert_eq!(snapshot.leases, 6);
        assert_eq!(snapshot.trace_events_dropped, 7);
    }

    #[test]
    fn rss_sampling_is_best_effort() {
        if let Some(RssSample {
            resident_bytes,
            peak_bytes,
        }) = RssSample::sample()
        {
            assert!(resident_bytes > 0);
            assert!(peak_bytes >= resident_bytes);
        }
    }
}
