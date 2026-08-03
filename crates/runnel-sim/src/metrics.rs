//! Checked byte and event accounting.

use serde::{Deserialize, Serialize};

use crate::SimError;

/// Raw counters emitted for every simulation.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationMetrics {
    pub demand_accesses: u64,
    pub demand_logical_bytes: u64,
    pub ordinary_demand_hits: u64,
    pub ordinary_demand_hit_bytes: u64,
    pub useful_prefetch_hits: u64,
    pub useful_prefetch_hit_bytes: u64,
    pub demand_misses: u64,
    pub demand_miss_bytes: u64,
    pub demand_load_bytes: u64,
    pub prefetch_offered: u64,
    pub prefetch_offered_bytes: u64,
    pub prefetch_admitted: u64,
    pub prefetch_load_bytes: u64,
    pub prefetch_useful: u64,
    pub prefetch_useful_bytes: u64,
    pub prefetch_wasted: u64,
    pub prefetch_wasted_bytes: u64,
    pub prefetch_redundant: u64,
    pub prefetch_redundant_bytes: u64,
    pub prefetch_dropped: u64,
    pub prefetch_dropped_bytes: u64,
    pub admissions: u64,
    pub bypasses: u64,
    pub evictions: u64,
    pub evicted_charge_bytes: u64,
    pub final_resident_charge_bytes: u64,
    pub peak_resident_charge_bytes: u64,
    pub policy_metadata_bytes: u64,
    pub policy_metadata_limit_bytes: u64,
    pub total_physical_load_bytes: u64,
}

impl SimulationMetrics {
    pub(crate) fn add(target: &mut u64, value: u64, name: &'static str) -> Result<(), SimError> {
        *target = target
            .checked_add(value)
            .ok_or(SimError::CounterOverflow(name))?;
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> Result<(), SimError> {
        self.total_physical_load_bytes = self
            .demand_load_bytes
            .checked_add(self.prefetch_load_bytes)
            .ok_or(SimError::CounterOverflow("total_physical_load_bytes"))?;
        let hits = self
            .ordinary_demand_hits
            .checked_add(self.useful_prefetch_hits)
            .and_then(|value| value.checked_add(self.demand_misses))
            .ok_or(SimError::CounterOverflow("demand_accesses identity"))?;
        if hits != self.demand_accesses {
            return Err(SimError::invalid_config(
                "demand hit/miss accounting identity failed",
            ));
        }
        let bytes = self
            .ordinary_demand_hit_bytes
            .checked_add(self.useful_prefetch_hit_bytes)
            .and_then(|value| value.checked_add(self.demand_miss_bytes))
            .ok_or(SimError::CounterOverflow("demand_logical_bytes identity"))?;
        if bytes != self.demand_logical_bytes {
            return Err(SimError::invalid_config(
                "demand byte accounting identity failed",
            ));
        }
        if self.demand_load_bytes != self.demand_miss_bytes {
            return Err(SimError::invalid_config(
                "every modeled demand miss must account one physical fill",
            ));
        }
        let classified_offers = self
            .prefetch_admitted
            .checked_add(self.prefetch_redundant)
            .and_then(|value| value.checked_add(self.prefetch_dropped))
            .ok_or(SimError::CounterOverflow("prefetch offer identity"))?;
        if classified_offers != self.prefetch_offered {
            return Err(SimError::invalid_config(
                "prefetch offer classification identity failed",
            ));
        }
        let classified_offer_bytes = self
            .prefetch_load_bytes
            .checked_add(self.prefetch_redundant_bytes)
            .and_then(|value| value.checked_add(self.prefetch_dropped_bytes))
            .ok_or(SimError::CounterOverflow("prefetch offer byte identity"))?;
        if classified_offer_bytes != self.prefetch_offered_bytes {
            return Err(SimError::invalid_config(
                "prefetch offer byte classification identity failed",
            ));
        }
        let classified_prefetches = self
            .prefetch_useful
            .checked_add(self.prefetch_wasted)
            .ok_or(SimError::CounterOverflow("prefetch count identity"))?;
        if classified_prefetches != self.prefetch_admitted {
            return Err(SimError::invalid_config(
                "admitted prefetch classification identity failed",
            ));
        }
        let classified_prefetch_bytes = self
            .prefetch_useful_bytes
            .checked_add(self.prefetch_wasted_bytes)
            .ok_or(SimError::CounterOverflow("prefetch byte identity"))?;
        if classified_prefetch_bytes != self.prefetch_load_bytes {
            return Err(SimError::invalid_config(
                "prefetch byte classification identity failed",
            ));
        }
        Ok(())
    }
}
