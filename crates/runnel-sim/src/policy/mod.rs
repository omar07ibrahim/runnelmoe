//! Deterministic online policy state used by the shared cache kernel.

use std::collections::BTreeMap;

use crate::model::PageCatalog;
use crate::{
    ExpertPrediction, PageClass, PageDescriptor, PageId, PolicySpec, RouterPolicyConfig, SimError,
    TinyLfuConfig,
};

const PPM: u64 = 1_000_000;
const SKETCH_COUNTER_MAX: u8 = 15;
const ROUTER_SIGNAL_BASE_BYTES: u64 = 24;
const ROUTER_PREDICTION_BYTES: u64 = 8;
const HASH_SEEDS: [u64; 8] = [
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
    0x4528_21e6_38d0_1377,
    0xbe54_66cf_34e9_0c6c,
    0xc0ac_29b7_c97c_50dd,
    0x3f84_d5b5_b547_0917,
];

pub(crate) fn router_signal_metadata_bytes(prediction_count: usize) -> Result<u64, SimError> {
    let predictions = u64::try_from(prediction_count)
        .map_err(|_| SimError::CounterOverflow("router prediction count"))?;
    ROUTER_SIGNAL_BASE_BYTES
        .checked_add(
            predictions
                .checked_mul(ROUTER_PREDICTION_BYTES)
                .ok_or(SimError::CounterOverflow("router prediction metadata"))?,
        )
        .ok_or(SimError::CounterOverflow("router signal metadata"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Segment {
    Lru,
    Probation,
    Protected,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentEntry {
    pub(crate) page: PageId,
    pub(crate) charge_bytes: u64,
    pub(crate) logical_bytes: u64,
    pub(crate) last_touch: u64,
    pub(crate) segment: Segment,
    pub(crate) prefetched_unused: bool,
}

pub(crate) struct DemandPlan<'a> {
    pub(crate) resident_bytes: u64,
    pub(crate) capacity_bytes: u64,
    pub(crate) candidate: &'a PageDescriptor,
    pub(crate) request: u64,
    pub(crate) step: u64,
}

pub(crate) struct PrefetchPlan<'a> {
    pub(crate) resident_bytes: u64,
    pub(crate) capacity_bytes: u64,
    pub(crate) absent_pages: &'a [PageId],
    pub(crate) request: u64,
    pub(crate) target_step: u64,
    pub(crate) candidate_score: u32,
}

#[derive(Clone, Debug)]
struct ActiveSignal {
    scores: BTreeMap<u32, u32>,
}

type ActiveSignals = BTreeMap<u64, BTreeMap<(u64, u32), ActiveSignal>>;

#[derive(Clone, Debug)]
struct TinyLfu {
    depth: usize,
    width: usize,
    counters: Vec<u8>,
    doorkeeper: Vec<u8>,
    sample_accesses: u64,
    observed: u64,
}

impl TinyLfu {
    fn new(config: &TinyLfuConfig) -> Result<Self, SimError> {
        if !(1..=HASH_SEEDS.len()).contains(&config.sketch_depth) {
            return Err(SimError::invalid_config(
                "TinyLFU sketch_depth must be between 1 and 8",
            ));
        }
        if !(64..=65_536).contains(&config.sketch_width) {
            return Err(SimError::invalid_config(
                "TinyLFU sketch_width must be between 64 and 65536",
            ));
        }
        if config.sample_accesses == 0 {
            return Err(SimError::invalid_config(
                "TinyLFU sample_accesses must be positive",
            ));
        }
        let counter_len = config
            .sketch_depth
            .checked_mul(config.sketch_width)
            .ok_or_else(|| SimError::invalid_config("TinyLFU sketch allocation overflows"))?;
        let door_len =
            config.sketch_width.checked_add(7).ok_or_else(|| {
                SimError::invalid_config("TinyLFU doorkeeper allocation overflows")
            })? / 8;
        Ok(Self {
            depth: config.sketch_depth,
            width: config.sketch_width,
            counters: vec![0; counter_len],
            doorkeeper: vec![0; door_len],
            sample_accesses: config.sample_accesses,
            observed: 0,
        })
    }

    fn splitmix(mut value: u64) -> u64 {
        // Independent SplitMix64 finalizer encoding; algorithm and reference
        // license provenance are recorded in docs/PRIOR_ART.md.
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn column(&self, page: PageId, row: usize) -> usize {
        let hash = Self::splitmix(u64::from(page.0) ^ HASH_SEEDS[row]);
        usize::try_from(hash % u64::try_from(self.width).expect("bounded width"))
            .expect("modulo width fits usize")
    }

    fn door_contains(&self, page: PageId) -> bool {
        (0..self.depth).all(|row| {
            let column = self.column(page, row);
            self.doorkeeper[column / 8] & (1 << (column % 8)) != 0
        })
    }

    fn set_door(&mut self, page: PageId) {
        for row in 0..self.depth {
            let column = self.column(page, row);
            self.doorkeeper[column / 8] |= 1 << (column % 8);
        }
    }

    fn estimate(&self, page: PageId) -> u64 {
        let minimum = (0..self.depth)
            .map(|row| self.counters[row * self.width + self.column(page, row)])
            .min()
            .unwrap_or(0);
        u64::from(minimum) + u64::from(self.door_contains(page))
    }

    fn observe(&mut self, page: PageId) {
        if self.door_contains(page) {
            let minimum = (0..self.depth)
                .map(|row| self.counters[row * self.width + self.column(page, row)])
                .min()
                .unwrap_or(0);
            for row in 0..self.depth {
                let column = self.column(page, row);
                let index = row * self.width + column;
                if self.counters[index] == minimum {
                    self.counters[index] = self.counters[index]
                        .saturating_add(1)
                        .min(SKETCH_COUNTER_MAX);
                }
            }
        } else {
            self.set_door(page);
        }
        self.observed += 1;
        if self.observed >= self.sample_accesses {
            for counter in &mut self.counters {
                *counter >>= 1;
            }
            self.doorkeeper.fill(0);
            self.observed >>= 1;
        }
    }

    fn metadata_bytes(&self) -> u64 {
        u64::try_from(self.counters.len() + self.doorkeeper.len()).expect("bounded allocation")
    }
}

#[derive(Clone, Debug)]
enum PolicyStateKind {
    NoCache,
    Lru,
    Slru {
        protected_target: u64,
    },
    TinyLfu(TinyLfu),
    Router {
        protected_target: u64,
        config: RouterPolicyConfig,
        prefetch: bool,
        active: ActiveSignals,
        current_metadata_bytes: u64,
        peak_metadata_bytes: u64,
        recorded_metadata_bytes: u64,
        metadata_limit_bytes: u64,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct PolicyState {
    kind: PolicyStateKind,
}

impl PolicyState {
    pub(crate) fn new(
        spec: &PolicySpec,
        capacity_bytes: u64,
        router_metadata_limit_bytes: u64,
    ) -> Result<Self, SimError> {
        let fraction_target = |fraction: u32| -> Result<u64, SimError> {
            if fraction > 1_000_000 {
                return Err(SimError::invalid_config(
                    "protected_fraction_ppm exceeds one million",
                ));
            }
            Ok((u128::from(capacity_bytes) * u128::from(fraction) / u128::from(PPM)) as u64)
        };
        let kind = match spec {
            PolicySpec::NoCache => PolicyStateKind::NoCache,
            PolicySpec::Lru => PolicyStateKind::Lru,
            PolicySpec::Slru {
                protected_fraction_ppm,
            } => PolicyStateKind::Slru {
                protected_target: fraction_target(*protected_fraction_ppm)?,
            },
            PolicySpec::TinyLfu { config } => PolicyStateKind::TinyLfu(TinyLfu::new(config)?),
            PolicySpec::RouterAdmit { config } | PolicySpec::RouterPrefetch { config } => {
                if config.minimum_score_ppm > 1_000_000 {
                    return Err(SimError::invalid_config(
                        "router minimum_score_ppm exceeds one million",
                    ));
                }
                if config.max_experts_per_signal == 0 || config.max_pages_per_signal == 0 {
                    return Err(SimError::invalid_config(
                        "router expert and page limits must be positive",
                    ));
                }
                PolicyStateKind::Router {
                    protected_target: fraction_target(config.protected_fraction_ppm)?,
                    config: config.clone(),
                    prefetch: matches!(spec, PolicySpec::RouterPrefetch { .. }),
                    active: BTreeMap::new(),
                    current_metadata_bytes: 0,
                    peak_metadata_bytes: 0,
                    recorded_metadata_bytes: 0,
                    metadata_limit_bytes: router_metadata_limit_bytes,
                }
            }
            PolicySpec::Belady => {
                return Err(SimError::invalid_config(
                    "Belady does not use the online policy kernel",
                ));
            }
        };
        Ok(Self { kind })
    }

    pub(crate) fn is_no_cache(&self) -> bool {
        matches!(self.kind, PolicyStateKind::NoCache)
    }

    pub(crate) fn prefetch_enabled(&self) -> bool {
        matches!(self.kind, PolicyStateKind::Router { prefetch: true, .. })
    }

    pub(crate) fn initial_segment(&self) -> Segment {
        match self.kind {
            PolicyStateKind::NoCache | PolicyStateKind::Lru | PolicyStateKind::TinyLfu(_) => {
                Segment::Lru
            }
            PolicyStateKind::Slru { .. } | PolicyStateKind::Router { .. } => Segment::Probation,
        }
    }

    pub(crate) fn observe_demand(&mut self, page: PageId) {
        if let PolicyStateKind::TinyLfu(tiny) = &mut self.kind {
            tiny.observe(page);
        }
    }

    pub(crate) fn enter_demand_step(&mut self, request: u64, step: u64) -> Result<(), SimError> {
        let PolicyStateKind::Router {
            active,
            current_metadata_bytes,
            ..
        } = &mut self.kind
        else {
            return Ok(());
        };
        let Some(signals) = active.get(&request) else {
            return Ok(());
        };
        let retired: Vec<(u64, u32)> = signals
            .keys()
            .take_while(|(target_step, _)| *target_step < step)
            .copied()
            .collect();
        let mut retired_bytes = 0_u64;
        for key in &retired {
            let signal = &signals[key];
            retired_bytes = retired_bytes
                .checked_add(router_signal_metadata_bytes(signal.scores.len())?)
                .ok_or(SimError::CounterOverflow("retired router metadata"))?;
        }
        if let Some(signals) = active.get_mut(&request) {
            for key in retired {
                signals.remove(&key);
            }
            if signals.is_empty() {
                active.remove(&request);
            }
        }
        *current_metadata_bytes = current_metadata_bytes
            .checked_sub(retired_bytes)
            .ok_or_else(|| SimError::invalid_config("router metadata ledger underflow"))?;
        Ok(())
    }

    pub(crate) fn record_signal(
        &mut self,
        request: u64,
        target_step: u64,
        layer: u32,
        predictions: &[ExpertPrediction],
    ) -> Result<(), SimError> {
        if predictions.is_empty() {
            return Ok(());
        }
        if let PolicyStateKind::Router {
            active,
            current_metadata_bytes,
            peak_metadata_bytes,
            recorded_metadata_bytes,
            metadata_limit_bytes,
            ..
        } = &mut self.kind
        {
            let scores: BTreeMap<u32, u32> = predictions
                .iter()
                .map(|prediction| (prediction.expert, prediction.score_ppm))
                .collect();
            if scores.len() != predictions.len() {
                return Err(SimError::invalid_config(
                    "router signal contains duplicate selected experts",
                ));
            }
            if active
                .get(&request)
                .is_some_and(|signals| signals.contains_key(&(target_step, layer)))
            {
                return Err(SimError::invalid_config(
                    "router signal target was recorded more than once",
                ));
            }
            let charge = router_signal_metadata_bytes(scores.len())?;
            let next_current = current_metadata_bytes
                .checked_add(charge)
                .ok_or(SimError::CounterOverflow("current router metadata"))?;
            let next_recorded = recorded_metadata_bytes
                .checked_add(charge)
                .ok_or(SimError::CounterOverflow("recorded router metadata"))?;
            if next_current > *metadata_limit_bytes || next_recorded > *metadata_limit_bytes {
                return Err(SimError::invalid_config(
                    "router metadata exceeded its precomputed ceiling",
                ));
            }
            active
                .entry(request)
                .or_default()
                .insert((target_step, layer), ActiveSignal { scores });
            *current_metadata_bytes = next_current;
            *peak_metadata_bytes = (*peak_metadata_bytes).max(next_current);
            *recorded_metadata_bytes = next_recorded;
        }
        Ok(())
    }

    pub(crate) fn active_score(&self, request: u64, step: u64, page: &PageDescriptor) -> u32 {
        let PolicyStateKind::Router { active, .. } = &self.kind else {
            return 0;
        };
        let PageClass::Expert { layer, expert, .. } = &page.class else {
            return 0;
        };
        let Some(signal) = active
            .get(&request)
            .and_then(|signals| signals.get(&(step, *layer)))
        else {
            return 0;
        };
        signal.scores.get(expert).copied().unwrap_or(0)
    }

    pub(crate) fn router_config(&self) -> Option<&RouterPolicyConfig> {
        match &self.kind {
            PolicyStateKind::Router { config, .. } => Some(config),
            _ => None,
        }
    }

    pub(crate) fn on_hit(
        &self,
        residents: &mut BTreeMap<PageId, ResidentEntry>,
        page: PageId,
        touch: u64,
    ) {
        let Some(entry) = residents.get_mut(&page) else {
            return;
        };
        entry.last_touch = touch;
        let protected_target = match self.kind {
            PolicyStateKind::Slru { protected_target }
            | PolicyStateKind::Router {
                protected_target, ..
            } => Some(protected_target),
            _ => None,
        };
        if let Some(protected_target) = protected_target
            && entry.segment == Segment::Probation
            && protected_target > 0
            && entry.charge_bytes <= protected_target
        {
            entry.segment = Segment::Protected;
            Self::rebalance_protected(residents, protected_target, touch);
        }
    }

    fn rebalance_protected(
        residents: &mut BTreeMap<PageId, ResidentEntry>,
        protected_target: u64,
        demotion_touch: u64,
    ) {
        loop {
            let protected_bytes: u64 = residents
                .values()
                .filter(|entry| entry.segment == Segment::Protected)
                .map(|entry| entry.charge_bytes)
                .sum();
            if protected_bytes <= protected_target {
                break;
            }
            let victim = residents
                .values()
                .filter(|entry| entry.segment == Segment::Protected)
                .min_by_key(|entry| (entry.last_touch, entry.page))
                .map(|entry| entry.page)
                .expect("positive protected bytes imply an entry");
            let demoted = residents.get_mut(&victim).expect("selected resident");
            demoted.segment = Segment::Probation;
            demoted.last_touch = demotion_touch;
        }
    }

    fn segment_rank(&self, entry: &ResidentEntry) -> u8 {
        match self.kind {
            PolicyStateKind::Slru { .. } | PolicyStateKind::Router { .. } => match entry.segment {
                Segment::Probation => 0,
                Segment::Protected => 1,
                Segment::Lru => 2,
            },
            _ => 0,
        }
    }

    pub(crate) fn plan_demand_victims(
        &self,
        catalog: PageCatalog<'_>,
        residents: &BTreeMap<PageId, ResidentEntry>,
        plan: DemandPlan<'_>,
    ) -> Result<Option<Vec<PageId>>, SimError> {
        let DemandPlan {
            resident_bytes,
            capacity_bytes,
            candidate,
            request,
            step,
        } = plan;
        if candidate.charge_bytes > capacity_bytes || self.is_no_cache() {
            return Ok(None);
        }
        let available = capacity_bytes - resident_bytes;
        if candidate.charge_bytes <= available {
            return Ok(Some(Vec::new()));
        }
        let needed = candidate.charge_bytes - available;
        let mut ordered: Vec<&ResidentEntry> = residents.values().collect();
        ordered.sort_by_key(|entry| {
            let score = self.active_score(request, step, catalog.page(entry.page));
            (
                self.segment_rank(entry),
                score,
                entry.last_touch,
                entry.page,
            )
        });
        let mut victims = Vec::new();
        let mut bytes = 0_u64;
        for entry in ordered {
            victims.push(entry.page);
            bytes = bytes
                .checked_add(entry.charge_bytes)
                .ok_or(SimError::CounterOverflow("victim plan bytes"))?;
            if bytes >= needed {
                break;
            }
        }
        if bytes < needed {
            return Ok(None);
        }
        if !self.admit_demand(catalog, residents, candidate, &victims, request, step)? {
            return Ok(None);
        }
        Ok(Some(victims))
    }

    fn admit_demand(
        &self,
        catalog: PageCatalog<'_>,
        residents: &BTreeMap<PageId, ResidentEntry>,
        candidate: &PageDescriptor,
        victims: &[PageId],
        request: u64,
        step: u64,
    ) -> Result<bool, SimError> {
        match &self.kind {
            PolicyStateKind::TinyLfu(tiny) if !victims.is_empty() => {
                let candidate_frequency = u128::from(tiny.estimate(candidate.id));
                let victim_charge = victims.iter().try_fold(0_u128, |total, id| {
                    total
                        .checked_add(u128::from(residents[id].charge_bytes))
                        .ok_or(SimError::CounterOverflow("TinyLFU victim charge"))
                })?;
                let victim_frequency = victims.iter().try_fold(0_u128, |total, id| {
                    total
                        .checked_add(u128::from(tiny.estimate(*id)))
                        .ok_or(SimError::CounterOverflow("TinyLFU victim frequency"))
                })?;
                Ok(candidate_frequency
                    .checked_mul(victim_charge)
                    .ok_or(SimError::CounterOverflow("TinyLFU candidate density"))?
                    > victim_frequency
                        .checked_mul(u128::from(candidate.charge_bytes))
                        .ok_or(SimError::CounterOverflow("TinyLFU victim density"))?)
            }
            PolicyStateKind::Router { active, .. } if !active.is_empty() => {
                let candidate_score = u128::from(self.active_score(request, step, candidate));
                let victim_charge = victims.iter().try_fold(0_u128, |total, id| {
                    total
                        .checked_add(u128::from(residents[id].charge_bytes))
                        .ok_or(SimError::CounterOverflow("router victim charge"))
                })?;
                let victim_score = victims.iter().try_fold(0_u128, |total, id| {
                    let score = self.active_score(request, step, catalog.page(*id));
                    total
                        .checked_add(u128::from(score))
                        .ok_or(SimError::CounterOverflow("router victim score"))
                })?;
                if candidate_score == 0 && victim_score == 0 {
                    return Ok(true);
                }
                Ok(candidate_score
                    .checked_mul(victim_charge)
                    .ok_or(SimError::CounterOverflow("router candidate density"))?
                    > victim_score
                        .checked_mul(u128::from(candidate.charge_bytes))
                        .ok_or(SimError::CounterOverflow("router victim density"))?)
            }
            _ => Ok(true),
        }
    }

    pub(crate) fn plan_prefetch_victims(
        &self,
        catalog: PageCatalog<'_>,
        residents: &BTreeMap<PageId, ResidentEntry>,
        plan: PrefetchPlan<'_>,
    ) -> Result<Option<Vec<PageId>>, SimError> {
        let PrefetchPlan {
            resident_bytes,
            capacity_bytes,
            absent_pages,
            request,
            target_step,
            candidate_score,
        } = plan;
        let required = absent_pages.iter().try_fold(0_u64, |total, id| {
            total
                .checked_add(catalog.page(*id).charge_bytes)
                .ok_or(SimError::CounterOverflow("prefetch group charge"))
        })?;
        if required > capacity_bytes {
            return Ok(None);
        }
        let available = capacity_bytes - resident_bytes;
        if required <= available {
            return Ok(Some(Vec::new()));
        }
        let needed = required - available;
        let mut probation: Vec<&ResidentEntry> = residents
            .values()
            .filter(|entry| entry.segment == Segment::Probation)
            .collect();
        probation.sort_by_key(|entry| {
            (
                self.active_score(request, target_step, catalog.page(entry.page)),
                entry.last_touch,
                entry.page,
            )
        });
        let mut victims = Vec::new();
        let mut victim_bytes = 0_u64;
        let mut weighted_victim_score = 0_u128;
        for entry in probation {
            let score = self.active_score(request, target_step, catalog.page(entry.page));
            victims.push(entry.page);
            victim_bytes = victim_bytes
                .checked_add(entry.charge_bytes)
                .ok_or(SimError::CounterOverflow("prefetch victim bytes"))?;
            weighted_victim_score = weighted_victim_score
                .checked_add(u128::from(score) * u128::from(entry.charge_bytes))
                .ok_or(SimError::CounterOverflow("prefetch victim score"))?;
            if victim_bytes >= needed {
                break;
            }
        }
        if victim_bytes < needed {
            return Ok(None);
        }
        let candidate_value = u128::from(candidate_score)
            .checked_mul(u128::from(required))
            .ok_or(SimError::CounterOverflow("prefetch candidate score"))?;
        if weighted_victim_score > 0 && candidate_value <= weighted_victim_score {
            return Ok(None);
        }
        Ok(Some(victims))
    }

    pub(crate) fn finish_metadata(&self) -> Result<(u64, u64), SimError> {
        match &self.kind {
            PolicyStateKind::TinyLfu(tiny) => {
                let bytes = tiny.metadata_bytes();
                Ok((bytes, bytes))
            }
            PolicyStateKind::Router {
                active,
                current_metadata_bytes,
                peak_metadata_bytes,
                recorded_metadata_bytes,
                metadata_limit_bytes,
                ..
            } => {
                let mut recomputed_current = 0_u64;
                for signals in active.values() {
                    if signals.is_empty() {
                        return Err(SimError::invalid_config(
                            "router metadata retained an empty request map",
                        ));
                    }
                    for signal in signals.values() {
                        if signal.scores.is_empty() {
                            return Err(SimError::invalid_config(
                                "router metadata retained an empty signal",
                            ));
                        }
                        recomputed_current = recomputed_current
                            .checked_add(router_signal_metadata_bytes(signal.scores.len())?)
                            .ok_or(SimError::CounterOverflow("recomputed router metadata"))?;
                    }
                }
                if recomputed_current != *current_metadata_bytes {
                    return Err(SimError::invalid_config(
                        "router metadata ledger does not match live state",
                    ));
                }
                if current_metadata_bytes > peak_metadata_bytes
                    || peak_metadata_bytes > metadata_limit_bytes
                {
                    return Err(SimError::invalid_config(
                        "router metadata peak/ceiling accounting failed",
                    ));
                }
                if recorded_metadata_bytes != metadata_limit_bytes {
                    return Err(SimError::invalid_config(
                        "router metadata did not consume its precomputed signal ceiling",
                    ));
                }
                Ok((*peak_metadata_bytes, *metadata_limit_bytes))
            }
            _ => Ok((0, 0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tinylfu_is_deterministic_and_ages() {
        let config = TinyLfuConfig {
            sketch_depth: 4,
            sketch_width: 64,
            sample_accesses: 8,
        };
        let mut first = TinyLfu::new(&config).unwrap();
        let mut second = TinyLfu::new(&config).unwrap();
        for page in [0, 1, 0, 2, 0, 1, 3, 0, 4, 0] {
            first.observe(PageId(page));
            second.observe(PageId(page));
        }
        assert_eq!(first.counters, second.counters);
        assert_eq!(first.doorkeeper, second.doorkeeper);
        assert!(first.estimate(PageId(0)) >= first.estimate(PageId(4)));
    }

    #[test]
    fn invalid_tinylfu_bounds_fail_before_allocation() {
        let error = TinyLfu::new(&TinyLfuConfig {
            sketch_depth: 9,
            sketch_width: usize::MAX,
            sample_accesses: 0,
        })
        .unwrap_err();
        assert!(error.to_string().contains("sketch_depth"));
    }
}
