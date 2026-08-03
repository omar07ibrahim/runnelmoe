//! Offline reference algorithms.
//!
//! The production simulator uses the uniform-geometry Bélády replay as a
//! lower bound.  A deliberately small dynamic program is also exposed for
//! differential tests and for examples where page sizes or miss costs differ.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use sha2::{Digest, Sha256};

use crate::{PageId, SimError, SimLimits, SimulationMetrics, TraceEvent, ValidatedTrace};

const MAX_EXACT_PAGES: usize = 18;
const MAX_EXACT_DEMANDS: usize = 200;
const DECISION_SCHEMA: &[u8] = b"runnel.belady-decisions/1\n";

/// Complete output of the uniform-geometry offline replay.
///
/// This remains crate-private because the public simulator wraps it in the
/// same [`crate::SimulationResult`] shape as every online policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BeladyReplay {
    pub(crate) metrics: SimulationMetrics,
    pub(crate) decision_sha256: String,
    pub(crate) oracle_optimal: bool,
}

/// Replay exact Bélády/MIN for a uniform page geometry.
///
/// Router signals are intentionally absent from the algorithm.  On a full
/// miss, the demanded candidate participates in victim selection.  A tie at
/// the same next-use position evicts the candidate first (a stable bypass),
/// then the greatest page identifier.
pub(crate) fn simulate_belady(
    trace: &ValidatedTrace,
    capacity_bytes: u64,
) -> Result<BeladyReplay, SimError> {
    let (uniform_charge, _) = uniform_geometry(trace)?;
    let slot_capacity = usize::try_from(capacity_bytes / uniform_charge)
        .unwrap_or(usize::MAX)
        .min(trace.pages().len());

    let mut future = future_demands(trace);
    let mut residents = BTreeSet::new();
    let mut resident_charge = 0_u64;
    let mut metrics = SimulationMetrics::default();
    let mut decisions = Sha256::new();
    decisions.update(DECISION_SCHEMA);
    let mut demand_ordinal = 0_u64;

    for event in trace.events() {
        let TraceEvent::Demand { page, .. } = event else {
            // Offline replacement must not learn from router-policy signals.
            continue;
        };

        pop_current_demand(&mut future, *page)?;
        let descriptor = trace.page(*page);
        SimulationMetrics::add(&mut metrics.demand_accesses, 1, "demand_accesses")?;
        SimulationMetrics::add(
            &mut metrics.demand_logical_bytes,
            descriptor.logical_bytes,
            "demand_logical_bytes",
        )?;

        if residents.contains(page) {
            SimulationMetrics::add(&mut metrics.ordinary_demand_hits, 1, "ordinary_demand_hits")?;
            SimulationMetrics::add(
                &mut metrics.ordinary_demand_hit_bytes,
                descriptor.logical_bytes,
                "ordinary_demand_hit_bytes",
            )?;
            update_decision(&mut decisions, demand_ordinal, *page, "hit", None);
        } else {
            SimulationMetrics::add(&mut metrics.demand_misses, 1, "demand_misses")?;
            SimulationMetrics::add(
                &mut metrics.demand_miss_bytes,
                descriptor.logical_bytes,
                "demand_miss_bytes",
            )?;
            SimulationMetrics::add(
                &mut metrics.demand_load_bytes,
                descriptor.logical_bytes,
                "demand_load_bytes",
            )?;

            if slot_capacity == 0 {
                SimulationMetrics::add(&mut metrics.bypasses, 1, "bypasses")?;
                update_decision(&mut decisions, demand_ordinal, *page, "bypass", None);
            } else if residents.len() < slot_capacity {
                admit(
                    &mut residents,
                    &mut resident_charge,
                    &mut metrics,
                    *page,
                    uniform_charge,
                )?;
                update_decision(&mut decisions, demand_ordinal, *page, "admit", None);
            } else {
                let victim = farthest_victim(&residents, *page, &future);
                if victim == *page {
                    SimulationMetrics::add(&mut metrics.bypasses, 1, "bypasses")?;
                    update_decision(&mut decisions, demand_ordinal, *page, "bypass", None);
                } else {
                    let removed = residents.remove(&victim);
                    debug_assert!(removed, "selected Bélády victim must be resident");
                    resident_charge = resident_charge.checked_sub(uniform_charge).ok_or(
                        SimError::CounterOverflow("belady resident charge subtraction"),
                    )?;
                    SimulationMetrics::add(&mut metrics.evictions, 1, "evictions")?;
                    SimulationMetrics::add(
                        &mut metrics.evicted_charge_bytes,
                        uniform_charge,
                        "evicted_charge_bytes",
                    )?;
                    admit(
                        &mut residents,
                        &mut resident_charge,
                        &mut metrics,
                        *page,
                        uniform_charge,
                    )?;
                    update_decision(&mut decisions, demand_ordinal, *page, "admit", Some(victim));
                }
            }
        }

        if resident_charge > capacity_bytes {
            return Err(SimError::invalid_config(
                "Bélády resident charge exceeded capacity",
            ));
        }
        demand_ordinal = demand_ordinal
            .checked_add(1)
            .ok_or(SimError::CounterOverflow("belady demand ordinal"))?;
    }

    metrics.final_resident_charge_bytes = resident_charge;
    metrics.finish()?;
    Ok(BeladyReplay {
        metrics,
        decision_sha256: hex::encode(decisions.finalize()),
        oracle_optimal: true,
    })
}

/// Return the exact minimum demand-load cost for a tiny variable-byte trace.
///
/// This dynamic program supports nonuniform capacity charges and nonuniform
/// logical miss costs.  Signals are ignored.  It accepts at most 18 distinct
/// demanded pages and 200 demands, and consumes at most
/// [`SimLimits::max_exact_oracle_states`] transition states.  Exceeding any
/// bound returns [`SimError::ExactOracleLimit`] instead of approximating.
pub fn exact_variable_byte_cost(
    trace: &ValidatedTrace,
    capacity_bytes: u64,
    limits: SimLimits,
) -> Result<u64, SimError> {
    if limits.max_exact_oracle_states == 0 {
        return Err(SimError::ExactOracleLimit(
            "max_exact_oracle_states must be positive".to_owned(),
        ));
    }

    let demands: Vec<PageId> = trace
        .events()
        .iter()
        .filter_map(|event| match event {
            TraceEvent::Demand { page, .. } => Some(*page),
            TraceEvent::RouterSignal { .. } => None,
        })
        .collect();
    if demands.len() > MAX_EXACT_DEMANDS {
        return Err(SimError::ExactOracleLimit(format!(
            "{} demands exceed the {MAX_EXACT_DEMANDS}-demand limit",
            demands.len()
        )));
    }

    let demanded_pages: BTreeSet<PageId> = demands.iter().copied().collect();
    if demanded_pages.len() > MAX_EXACT_PAGES {
        return Err(SimError::ExactOracleLimit(format!(
            "{} distinct demanded pages exceed the {MAX_EXACT_PAGES}-page limit",
            demanded_pages.len()
        )));
    }
    if demands.is_empty() {
        return Ok(0);
    }

    let ordered_pages: Vec<PageId> = demanded_pages.into_iter().collect();
    let bit_for_page: BTreeMap<PageId, usize> = ordered_pages
        .iter()
        .copied()
        .enumerate()
        .map(|(bit, page)| (page, bit))
        .collect();
    let charges: Vec<u64> = ordered_pages
        .iter()
        .map(|page| trace.page(*page).charge_bytes)
        .collect();
    let logical_costs: Vec<u64> = ordered_pages
        .iter()
        .map(|page| trace.page(*page).logical_bytes)
        .collect();

    // A state is the resident subset after the preceding demand.  Voluntary
    // eviction can be deferred until an admission, so a hit leaves its state
    // unchanged; on a miss, every fitting retained submask plus the candidate
    // is considered, along with bypass.
    let mut frontier = BTreeMap::from([(0_u32, 0_u64)]);
    let mut explored_states = 1_usize;
    for page in demands {
        let bit = bit_for_page[&page];
        let page_mask = 1_u32 << bit;
        let mut next = BTreeMap::<u32, u64>::new();

        for (state, cost) in &frontier {
            if state & page_mask != 0 {
                consume_state_budget(&mut explored_states, limits.max_exact_oracle_states)?;
                insert_minimum(&mut next, *state, *cost);
                continue;
            }

            let miss_cost = cost
                .checked_add(logical_costs[bit])
                .ok_or(SimError::CounterOverflow("exact variable-byte cost"))?;

            consume_state_budget(&mut explored_states, limits.max_exact_oracle_states)?;
            insert_minimum(&mut next, *state, miss_cost);

            if charges[bit] <= capacity_bytes {
                let mut retained = *state;
                loop {
                    consume_state_budget(&mut explored_states, limits.max_exact_oracle_states)?;
                    let retained_charge = subset_charge(retained, &charges)?;
                    if retained_charge
                        .checked_add(charges[bit])
                        .is_some_and(|charge| charge <= capacity_bytes)
                    {
                        insert_minimum(&mut next, retained | page_mask, miss_cost);
                    }
                    if retained == 0 {
                        break;
                    }
                    retained = (retained - 1) & state;
                }
            }
        }
        frontier = next;
    }

    frontier
        .values()
        .copied()
        .min()
        .ok_or_else(|| SimError::invalid_config("exact oracle produced no terminal state"))
}

fn uniform_geometry(trace: &ValidatedTrace) -> Result<(u64, u64), SimError> {
    let Some(first) = trace.pages().first() else {
        // A validated empty trace performs no division and has a zero optimum.
        return Ok((1, 0));
    };
    if first.charge_bytes == 0 {
        return Err(SimError::UnsupportedOracleGeometry(
            "Bélády requires a positive uniform charge".to_owned(),
        ));
    }
    for page in &trace.pages()[1..] {
        if page.charge_bytes != first.charge_bytes {
            return Err(SimError::UnsupportedOracleGeometry(format!(
                "page {} charge {} differs from uniform charge {}",
                page.id.0, page.charge_bytes, first.charge_bytes
            )));
        }
        if page.logical_bytes != first.logical_bytes {
            return Err(SimError::UnsupportedOracleGeometry(format!(
                "page {} logical cost {} differs from uniform cost {}",
                page.id.0, page.logical_bytes, first.logical_bytes
            )));
        }
    }
    Ok((first.charge_bytes, first.logical_bytes))
}

fn future_demands(trace: &ValidatedTrace) -> BTreeMap<PageId, VecDeque<u64>> {
    let mut demand_ordinal = 0_u64;
    let mut future = BTreeMap::<PageId, VecDeque<u64>>::new();
    for event in trace.events() {
        if let TraceEvent::Demand { page, .. } = event {
            future.entry(*page).or_default().push_back(demand_ordinal);
            demand_ordinal = demand_ordinal.saturating_add(1);
        }
    }
    future
}

fn pop_current_demand(
    future: &mut BTreeMap<PageId, VecDeque<u64>>,
    page: PageId,
) -> Result<(), SimError> {
    let popped = future.get_mut(&page).and_then(VecDeque::pop_front);
    if popped.is_none() {
        return Err(SimError::invalid_config(
            "Bélády future-demand index became inconsistent",
        ));
    }
    Ok(())
}

fn farthest_victim(
    residents: &BTreeSet<PageId>,
    candidate: PageId,
    future: &BTreeMap<PageId, VecDeque<u64>>,
) -> PageId {
    residents
        .iter()
        .copied()
        .chain(std::iter::once(candidate))
        .max_by_key(|page| {
            let next_use = future
                .get(page)
                .and_then(|positions| positions.front().copied())
                .unwrap_or(u64::MAX);
            // Candidate-first ties preserve the existing resident set when
            // admitting the candidate cannot improve the exact cost.
            (next_use, *page == candidate, page.0)
        })
        .expect("candidate guarantees a nonempty victim set")
}

fn admit(
    residents: &mut BTreeSet<PageId>,
    resident_charge: &mut u64,
    metrics: &mut SimulationMetrics,
    page: PageId,
    charge: u64,
) -> Result<(), SimError> {
    let inserted = residents.insert(page);
    debug_assert!(inserted, "a Bélády admission must insert an absent page");
    *resident_charge = resident_charge
        .checked_add(charge)
        .ok_or(SimError::CounterOverflow("belady resident charge"))?;
    SimulationMetrics::add(&mut metrics.admissions, 1, "admissions")?;
    metrics.peak_resident_charge_bytes = metrics.peak_resident_charge_bytes.max(*resident_charge);
    Ok(())
}

fn update_decision(
    digest: &mut Sha256,
    ordinal: u64,
    page: PageId,
    action: &str,
    victim: Option<PageId>,
) {
    let victim = victim.map_or_else(|| "-".to_owned(), |id| id.0.to_string());
    digest.update(
        format!(
            "demand={ordinal};page={};action={action};victim={victim}\n",
            page.0
        )
        .as_bytes(),
    );
}

fn consume_state_budget(explored: &mut usize, maximum: usize) -> Result<(), SimError> {
    *explored = explored
        .checked_add(1)
        .ok_or_else(|| SimError::ExactOracleLimit("state budget overflowed".to_owned()))?;
    if *explored > maximum {
        return Err(SimError::ExactOracleLimit(format!(
            "explored more than {maximum} transition states"
        )));
    }
    Ok(())
}

fn insert_minimum(frontier: &mut BTreeMap<u32, u64>, state: u32, cost: u64) {
    frontier
        .entry(state)
        .and_modify(|current| *current = (*current).min(cost))
        .or_insert(cost);
}

fn subset_charge(mask: u32, charges: &[u64]) -> Result<u64, SimError> {
    let mut remaining = mask;
    let mut total = 0_u64;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as usize;
        total = total
            .checked_add(charges[bit])
            .ok_or(SimError::CounterOverflow("exact resident charge"))?;
        remaining &= remaining - 1;
    }
    Ok(total)
}
