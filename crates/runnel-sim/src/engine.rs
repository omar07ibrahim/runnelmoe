//! Shared simulation kernel and byte-accounting state transitions.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::model::RESULT_SCHEMA;
use crate::oracle::simulate_belady;
use crate::policy::{
    DemandPlan, PolicyState, PrefetchPlan, ResidentEntry, router_signal_metadata_bytes,
};
use crate::{
    ExpertPrediction, PageDescriptor, PageId, PolicySpec, SimError, SimulationConfig,
    SimulationMetrics, SimulationResult, TraceEvent, ValidatedTrace,
};

struct Kernel<'a> {
    trace: &'a ValidatedTrace,
    capacity_bytes: u64,
    policy: PolicyState,
    residents: BTreeMap<PageId, ResidentEntry>,
    resident_bytes: u64,
    metrics: SimulationMetrics,
    decisions: Sha256,
    touch: u64,
}

struct PrefetchGroup<'a> {
    sequence: u64,
    request: u64,
    target_step: u64,
    prediction: &'a ExpertPrediction,
    page_ids: &'a [PageId],
}

struct PrefetchBudget {
    remaining_pages: usize,
    remaining_bytes: u64,
}

impl<'a> Kernel<'a> {
    fn new(trace: &'a ValidatedTrace, config: &SimulationConfig) -> Result<Self, SimError> {
        let metadata_limit_bytes = router_metadata_limit(trace, &config.policy)?;
        let policy = PolicyState::new(&config.policy, config.capacity_bytes, metadata_limit_bytes)?;
        Ok(Self {
            trace,
            capacity_bytes: config.capacity_bytes,
            policy,
            residents: BTreeMap::new(),
            resident_bytes: 0,
            metrics: SimulationMetrics::default(),
            decisions: Sha256::new(),
            touch: 0,
        })
    }

    fn add(target: &mut u64, value: u64, name: &'static str) -> Result<(), SimError> {
        SimulationMetrics::add(target, value, name)
    }

    fn next_touch(&mut self) -> Result<u64, SimError> {
        self.touch = self
            .touch
            .checked_add(1)
            .ok_or(SimError::CounterOverflow("policy touch clock"))?;
        Ok(self.touch)
    }

    fn hash_event(&mut self, sequence: u64, tag: &[u8], page: Option<PageId>, victims: &[PageId]) {
        self.decisions.update(sequence.to_le_bytes());
        self.decisions.update(
            u64::try_from(tag.len())
                .expect("static label fits")
                .to_le_bytes(),
        );
        self.decisions.update(tag);
        match page {
            Some(id) => {
                self.decisions.update([1]);
                self.decisions.update(id.0.to_le_bytes());
            }
            None => self.decisions.update([0]),
        }
        self.decisions.update(
            u64::try_from(victims.len())
                .expect("bounded trace")
                .to_le_bytes(),
        );
        for victim in victims {
            self.decisions.update(victim.0.to_le_bytes());
        }
    }

    fn hash_page_list(&mut self, pages: &[PageId]) {
        self.decisions.update(
            u64::try_from(pages.len())
                .expect("bounded trace")
                .to_le_bytes(),
        );
        for page in pages {
            self.decisions.update(page.0.to_le_bytes());
        }
    }

    fn hash_prefetch_decision(
        &mut self,
        sequence: u64,
        tag: &[u8],
        offered: &[PageId],
        redundant: &[PageId],
        admitted_or_dropped: &[PageId],
        victims: &[PageId],
    ) {
        self.decisions.update(sequence.to_le_bytes());
        self.decisions.update(
            u64::try_from(tag.len())
                .expect("static label fits")
                .to_le_bytes(),
        );
        self.decisions.update(tag);
        self.hash_page_list(offered);
        self.hash_page_list(redundant);
        self.hash_page_list(admitted_or_dropped);
        self.hash_page_list(victims);
    }

    fn account_wasted_prefetch(&mut self, entry: &ResidentEntry) -> Result<(), SimError> {
        if entry.prefetched_unused {
            Self::add(&mut self.metrics.prefetch_wasted, 1, "prefetch_wasted")?;
            Self::add(
                &mut self.metrics.prefetch_wasted_bytes,
                entry.logical_bytes,
                "prefetch_wasted_bytes",
            )?;
        }
        Ok(())
    }

    fn evict(&mut self, page: PageId) -> Result<(), SimError> {
        let entry = self
            .residents
            .remove(&page)
            .ok_or_else(|| SimError::invalid_config("victim plan referenced a nonresident page"))?;
        self.account_wasted_prefetch(&entry)?;
        self.resident_bytes = self
            .resident_bytes
            .checked_sub(entry.charge_bytes)
            .ok_or_else(|| SimError::invalid_config("resident-byte ledger underflow"))?;
        Self::add(&mut self.metrics.evictions, 1, "evictions")?;
        Self::add(
            &mut self.metrics.evicted_charge_bytes,
            entry.charge_bytes,
            "evicted_charge_bytes",
        )?;
        Ok(())
    }

    fn insert(
        &mut self,
        page: &PageDescriptor,
        touch: u64,
        prefetched_unused: bool,
    ) -> Result<(), SimError> {
        let new_resident_bytes = self
            .resident_bytes
            .checked_add(page.charge_bytes)
            .ok_or(SimError::CounterOverflow("resident charge"))?;
        if new_resident_bytes > self.capacity_bytes {
            return Err(SimError::invalid_config(
                "atomic victim plan did not free enough capacity",
            ));
        }
        let replaced = self.residents.insert(
            page.id,
            ResidentEntry {
                page: page.id,
                charge_bytes: page.charge_bytes,
                logical_bytes: page.logical_bytes,
                last_touch: touch,
                segment: self.policy.initial_segment(),
                prefetched_unused,
            },
        );
        if replaced.is_some() {
            return Err(SimError::invalid_config(
                "attempted to insert an already resident page",
            ));
        }
        self.resident_bytes = new_resident_bytes;
        self.metrics.peak_resident_charge_bytes = self
            .metrics
            .peak_resident_charge_bytes
            .max(self.resident_bytes);
        Self::add(&mut self.metrics.admissions, 1, "admissions")?;
        Ok(())
    }

    fn demand(
        &mut self,
        sequence: u64,
        request: u64,
        step: u64,
        page_id: PageId,
    ) -> Result<(), SimError> {
        let page = self.trace.page(page_id).clone();
        self.policy.enter_demand_step(request, step)?;
        self.policy.observe_demand(page_id);
        Self::add(&mut self.metrics.demand_accesses, 1, "demand_accesses")?;
        Self::add(
            &mut self.metrics.demand_logical_bytes,
            page.logical_bytes,
            "demand_logical_bytes",
        )?;
        let touch = self.next_touch()?;

        if let Some(entry) = self.residents.get_mut(&page_id) {
            if entry.prefetched_unused {
                entry.prefetched_unused = false;
                Self::add(
                    &mut self.metrics.useful_prefetch_hits,
                    1,
                    "useful_prefetch_hits",
                )?;
                Self::add(
                    &mut self.metrics.useful_prefetch_hit_bytes,
                    page.logical_bytes,
                    "useful_prefetch_hit_bytes",
                )?;
                Self::add(&mut self.metrics.prefetch_useful, 1, "prefetch_useful")?;
                Self::add(
                    &mut self.metrics.prefetch_useful_bytes,
                    page.logical_bytes,
                    "prefetch_useful_bytes",
                )?;
                self.hash_event(sequence, b"demand-useful-prefetch-hit", Some(page_id), &[]);
            } else {
                Self::add(
                    &mut self.metrics.ordinary_demand_hits,
                    1,
                    "ordinary_demand_hits",
                )?;
                Self::add(
                    &mut self.metrics.ordinary_demand_hit_bytes,
                    page.logical_bytes,
                    "ordinary_demand_hit_bytes",
                )?;
                self.hash_event(sequence, b"demand-hit", Some(page_id), &[]);
            }
            self.policy.on_hit(&mut self.residents, page_id, touch);
            return Ok(());
        }

        Self::add(&mut self.metrics.demand_misses, 1, "demand_misses")?;
        Self::add(
            &mut self.metrics.demand_miss_bytes,
            page.logical_bytes,
            "demand_miss_bytes",
        )?;
        Self::add(
            &mut self.metrics.demand_load_bytes,
            page.logical_bytes,
            "demand_load_bytes",
        )?;

        let victims = self.policy.plan_demand_victims(
            self.trace.catalog(),
            &self.residents,
            DemandPlan {
                resident_bytes: self.resident_bytes,
                capacity_bytes: self.capacity_bytes,
                candidate: &page,
                request,
                step,
            },
        )?;
        let Some(victims) = victims else {
            Self::add(&mut self.metrics.bypasses, 1, "bypasses")?;
            self.hash_event(sequence, b"demand-miss-bypass", Some(page_id), &[]);
            return Ok(());
        };
        for victim in &victims {
            self.evict(*victim)?;
        }
        self.insert(&page, touch, false)?;
        self.hash_event(sequence, b"demand-miss-admit", Some(page_id), &victims);
        Ok(())
    }

    fn prefetch_group(
        &mut self,
        group: PrefetchGroup<'_>,
        budget: &mut PrefetchBudget,
    ) -> Result<(), SimError> {
        let PrefetchGroup {
            sequence,
            request,
            target_step,
            prediction,
            page_ids,
        } = group;
        for id in page_ids {
            let page = self.trace.page(*id);
            Self::add(&mut self.metrics.prefetch_offered, 1, "prefetch_offered")?;
            Self::add(
                &mut self.metrics.prefetch_offered_bytes,
                page.logical_bytes,
                "prefetch_offered_bytes",
            )?;
        }
        let mut absent = Vec::new();
        let mut redundant = Vec::new();
        for id in page_ids {
            let page = self.trace.page(*id);
            if self.residents.contains_key(id) {
                redundant.push(*id);
                Self::add(
                    &mut self.metrics.prefetch_redundant,
                    1,
                    "prefetch_redundant",
                )?;
                Self::add(
                    &mut self.metrics.prefetch_redundant_bytes,
                    page.logical_bytes,
                    "prefetch_redundant_bytes",
                )?;
            } else {
                absent.push(*id);
            }
        }
        if absent.is_empty() {
            self.hash_prefetch_decision(
                sequence,
                b"prefetch-redundant",
                page_ids,
                &redundant,
                &[],
                &[],
            );
            return Ok(());
        }
        if absent.len() > budget.remaining_pages {
            self.drop_prefetches(&absent)?;
            self.hash_prefetch_decision(
                sequence,
                b"prefetch-drop-page-limit",
                page_ids,
                &redundant,
                &absent,
                &[],
            );
            return Ok(());
        }
        let load_bytes = absent.iter().try_fold(0_u64, |total, id| {
            total
                .checked_add(self.trace.page(*id).logical_bytes)
                .ok_or(SimError::CounterOverflow("prefetch group load"))
        })?;
        if load_bytes > budget.remaining_bytes {
            self.drop_prefetches(&absent)?;
            self.hash_prefetch_decision(
                sequence,
                b"prefetch-drop-byte-limit",
                page_ids,
                &redundant,
                &absent,
                &[],
            );
            return Ok(());
        }
        let victims = self.policy.plan_prefetch_victims(
            self.trace.catalog(),
            &self.residents,
            PrefetchPlan {
                resident_bytes: self.resident_bytes,
                capacity_bytes: self.capacity_bytes,
                absent_pages: &absent,
                request,
                target_step,
                candidate_score: prediction.score_ppm,
            },
        )?;
        let Some(victims) = victims else {
            self.drop_prefetches(&absent)?;
            self.hash_prefetch_decision(
                sequence,
                b"prefetch-drop-admission",
                page_ids,
                &redundant,
                &absent,
                &[],
            );
            return Ok(());
        };
        for victim in &victims {
            self.evict(*victim)?;
        }
        for id in &absent {
            let page = self.trace.page(*id).clone();
            let touch = self.next_touch()?;
            self.insert(&page, touch, true)?;
            Self::add(&mut self.metrics.prefetch_admitted, 1, "prefetch_admitted")?;
            Self::add(
                &mut self.metrics.prefetch_load_bytes,
                page.logical_bytes,
                "prefetch_load_bytes",
            )?;
        }
        budget.remaining_bytes = budget
            .remaining_bytes
            .checked_sub(load_bytes)
            .ok_or_else(|| SimError::invalid_config("prefetch byte budget underflow"))?;
        budget.remaining_pages -= absent.len();
        self.hash_prefetch_decision(
            sequence,
            b"prefetch-admit",
            page_ids,
            &redundant,
            &absent,
            &victims,
        );
        Ok(())
    }

    fn drop_prefetches(&mut self, pages: &[PageId]) -> Result<(), SimError> {
        for id in pages {
            let page = self.trace.page(*id);
            Self::add(&mut self.metrics.prefetch_dropped, 1, "prefetch_dropped")?;
            Self::add(
                &mut self.metrics.prefetch_dropped_bytes,
                page.logical_bytes,
                "prefetch_dropped_bytes",
            )?;
        }
        Ok(())
    }

    fn signal(
        &mut self,
        sequence: u64,
        request: u64,
        target_step: u64,
        layer: u32,
        predictions: &[ExpertPrediction],
    ) -> Result<(), SimError> {
        let Some(config) = self.policy.router_config().cloned() else {
            self.hash_event(sequence, b"signal-ignored", None, &[]);
            return Ok(());
        };
        let mut selected: Vec<ExpertPrediction> = predictions
            .iter()
            .filter(|prediction| prediction.score_ppm >= config.minimum_score_ppm)
            .cloned()
            .collect();
        selected
            .sort_by_key(|prediction| (std::cmp::Reverse(prediction.score_ppm), prediction.expert));
        selected.truncate(config.max_experts_per_signal);
        if selected.is_empty() {
            self.hash_event(sequence, b"signal-empty", None, &[]);
            return Ok(());
        }
        self.policy
            .record_signal(request, target_step, layer, &selected)?;
        if !self.policy.prefetch_enabled() {
            self.hash_event(sequence, b"signal-admission-only", None, &[]);
            return Ok(());
        }
        let mut budget = PrefetchBudget {
            remaining_pages: config.max_pages_per_signal,
            remaining_bytes: config.max_prefetch_bytes_per_signal,
        };
        for prediction in &selected {
            if budget.remaining_pages == 0 {
                break;
            }
            let page_ids = self.trace.expert_pages(layer, prediction.expert).to_vec();
            self.prefetch_group(
                PrefetchGroup {
                    sequence,
                    request,
                    target_step,
                    prediction,
                    page_ids: &page_ids,
                },
                &mut budget,
            )?;
        }
        Ok(())
    }

    fn run(mut self) -> Result<(SimulationMetrics, String), SimError> {
        for event in self.trace.events() {
            match event {
                TraceEvent::Demand {
                    sequence,
                    request,
                    step,
                    page,
                } => self.demand(*sequence, *request, *step, *page)?,
                TraceEvent::RouterSignal {
                    sequence,
                    request,
                    target_step,
                    layer,
                    predictions,
                } => self.signal(*sequence, *request, *target_step, *layer, predictions)?,
            }
        }
        let unfinished: Vec<ResidentEntry> = self
            .residents
            .values()
            .filter(|entry| entry.prefetched_unused)
            .cloned()
            .collect();
        for entry in &unfinished {
            self.account_wasted_prefetch(entry)?;
        }
        self.metrics.final_resident_charge_bytes = self.resident_bytes;
        let (metadata_bytes, metadata_limit_bytes) = self.policy.finish_metadata()?;
        self.metrics.policy_metadata_bytes = metadata_bytes;
        self.metrics.policy_metadata_limit_bytes = metadata_limit_bytes;
        self.metrics.finish()?;
        Ok((self.metrics, hex::encode(self.decisions.finalize())))
    }
}

/// Compute the accounting-only ceiling before replay without exposing the
/// event stream to the online policy state.
fn router_metadata_limit(trace: &ValidatedTrace, spec: &PolicySpec) -> Result<u64, SimError> {
    let config = match spec {
        PolicySpec::RouterAdmit { config } | PolicySpec::RouterPrefetch { config } => config,
        _ => return Ok(0),
    };
    let mut limit = 0_u64;
    for event in trace.events() {
        let TraceEvent::RouterSignal { predictions, .. } = event else {
            continue;
        };
        let selected = predictions
            .iter()
            .filter(|prediction| prediction.score_ppm >= config.minimum_score_ppm)
            .count()
            .min(config.max_experts_per_signal);
        if selected == 0 {
            continue;
        }
        limit = limit
            .checked_add(router_signal_metadata_bytes(selected)?)
            .ok_or(SimError::CounterOverflow("router metadata ceiling"))?;
    }
    Ok(limit)
}

/// Replay a validated trace under one deterministic policy.
pub fn simulate(
    trace: &ValidatedTrace,
    config: &SimulationConfig,
) -> Result<SimulationResult, SimError> {
    if matches!(config.policy, PolicySpec::Belady) {
        let replay = simulate_belady(trace, config.capacity_bytes)?;
        return Ok(SimulationResult {
            schema: RESULT_SCHEMA.to_owned(),
            trace_id: trace.header().trace_id.clone(),
            trace_sha256: trace.sha256().to_owned(),
            policy: config.policy.name().to_owned(),
            policy_spec: config.policy.clone(),
            capacity_bytes: config.capacity_bytes,
            oracle_optimal: replay.oracle_optimal,
            metrics: replay.metrics,
            decision_sha256: replay.decision_sha256,
        });
    }
    let (metrics, decision_sha256) = Kernel::new(trace, config)?.run()?;
    Ok(SimulationResult {
        schema: RESULT_SCHEMA.to_owned(),
        trace_id: trace.header().trace_id.clone(),
        trace_sha256: trace.sha256().to_owned(),
        policy: config.policy.name().to_owned(),
        policy_spec: config.policy.clone(),
        capacity_bytes: config.capacity_bytes,
        oracle_optimal: false,
        metrics,
        decision_sha256,
    })
}
