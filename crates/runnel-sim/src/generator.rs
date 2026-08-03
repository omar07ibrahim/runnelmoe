//! Deterministic synthetic MoE trace generator.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{PREFETCH_MODEL, TRACE_SCHEMA};
use crate::{
    ExpertPrediction, PageClass, PageDescriptor, PageId, SimError, SimLimits, TraceEvent,
    TraceHeader, ValidatedTrace, parse_trace, serialize_trace,
};

const EXPERT_COUNT: usize = 128;
const PAGES_PER_EXPERT: usize = 3;
const PAGE_BYTES: u64 = 65_536;
const BURN_IN_STEPS: usize = 512;
const PREDICTOR_AGE_INTERVAL: u64 = 256;
const PREDICTOR_MIN_SUPPORT: u64 = 8;
const PREDICTOR_MIN_SCORE_PPM: u32 = 100_000;
const SCORE_SCALE: u64 = 1_000_000;
const LAYER: u32 = 0;

/// Preregistered number of measured route steps in a full M3 trace.
pub const DEFAULT_MEASURED_STEPS: usize = 4_096;

/// One of the six preregistered, openly generated workload families.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceFamily {
    /// Stationary harmonic expert popularity.
    StationaryZipf,
    /// A small hot set interrupted by sequential cold scans.
    ScanPollution,
    /// A 16-expert hot set that changes every 256 routes.
    PhaseShift,
    /// A deterministic all-expert cycle, intentionally pathological for LRU.
    CyclicPressure,
    /// Locally skewed traffic whose 8-expert cluster follows a Markov process.
    MarkovClusters,
    /// Independent uniform routes, used as the predictability negative control.
    IidUniform,
}

impl TraceFamily {
    /// All families in stable evidence-matrix order.
    pub const ALL: [Self; 6] = [
        Self::StationaryZipf,
        Self::ScanPollution,
        Self::PhaseShift,
        Self::CyclicPressure,
        Self::MarkovClusters,
        Self::IidUniform,
    ];

    /// Stable snake-case identifier used by evidence manifests.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StationaryZipf => "stationary_zipf",
            Self::ScanPollution => "scan_pollution",
            Self::PhaseShift => "phase_shift",
            Self::CyclicPressure => "cyclic_pressure",
            Self::MarkovClusters => "markov_clusters",
            Self::IidUniform => "iid_uniform",
        }
    }

    fn trace_slug(self) -> &'static str {
        match self {
            Self::StationaryZipf => "stationary-zipf",
            Self::ScanPollution => "scan-pollution",
            Self::PhaseShift => "phase-shift",
            Self::CyclicPressure => "cyclic-pressure",
            Self::MarkovClusters => "markov-clusters",
            Self::IidUniform => "iid-uniform",
        }
    }
}

impl fmt::Display for TraceFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for TraceFamily {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|family| family.as_str() == value)
            .ok_or_else(|| format!("unknown trace family {value:?}"))
    }
}

/// A generated trace plus the provenance digests needed by the evidence ledger.
#[derive(Clone, Debug)]
pub struct GeneratedTrace {
    /// Workload family.
    pub family: TraceFamily,
    /// Paired replicate index.
    pub replicate: u32,
    /// Number of measured route steps, excluding burn-in.
    pub measured_steps: usize,
    /// Validated policy-neutral trace.
    pub trace: ValidatedTrace,
    /// Canonical JSONL bytes from which `trace` was parsed.
    pub canonical_bytes: Vec<u8>,
    /// Domain-separated SHA-256 seed derivation, encoded as lowercase hex.
    pub seed_sha256: String,
    /// SHA-256 of all burn-in and measured routes in order.
    pub full_route_sha256: String,
    /// SHA-256 of measured routes only.
    pub measured_route_sha256: String,
}

/// Generate one deterministic synthetic trace.
///
/// The seed intentionally excludes `measured_steps`, so asking for a longer
/// trace cannot alter any route or router signal in an already generated
/// prefix. At most [`DEFAULT_MEASURED_STEPS`] are accepted to keep canonical
/// traces within the parser's five-MiB ceiling.
pub fn generate_trace(
    family: TraceFamily,
    replicate: u32,
    measured_steps: usize,
) -> Result<GeneratedTrace, SimError> {
    if measured_steps == 0 || measured_steps > DEFAULT_MEASURED_STEPS {
        return Err(SimError::invalid_config(format!(
            "measured_steps must be in 1..={DEFAULT_MEASURED_STEPS}, got {measured_steps}"
        )));
    }

    let seed = derive_seed(family, replicate);
    let seed_sha256 = hex::encode(seed);
    let mut routes = RouteGenerator::new(family, seed);
    let mut predictor = CausalPredictor::default();
    let mut previous = None;

    let mut full_digest = route_digest(
        b"runnel-m3-full-routes-v1\0",
        BURN_IN_STEPS + measured_steps,
    );
    let mut measured_digest = route_digest(b"runnel-m3-measured-routes-v1\0", measured_steps);

    let mut prediction_for_first_step = Vec::new();
    for absolute_step in 0..BURN_IN_STEPS {
        let route = routes.next_route(absolute_step);
        digest_route(&mut full_digest, route);
        if let Some(prior) = previous {
            predictor.observe(prior, route);
        }
        previous = Some(route);
        prediction_for_first_step = predictor.predict(route);
    }

    let pages = page_catalog();
    let maximum_events = measured_steps
        .checked_mul(7)
        .ok_or_else(|| SimError::invalid_config("measured event count overflows usize"))?;
    let mut events = Vec::with_capacity(maximum_events);
    let mut sequence = 0_u64;
    if !prediction_for_first_step.is_empty() {
        push_signal(&mut events, &mut sequence, 0, prediction_for_first_step)?;
    }

    for measured_step in 0..measured_steps {
        let absolute_step = BURN_IN_STEPS + measured_step;
        let route = routes.next_route(absolute_step);
        digest_route(&mut full_digest, route);
        digest_route(&mut measured_digest, route);
        push_demands(&mut events, &mut sequence, measured_step, route)?;

        let prior = previous.expect("the fixed burn-in always establishes a prior route");
        predictor.observe(prior, route);
        previous = Some(route);

        if measured_step + 1 < measured_steps {
            let predictions = predictor.predict(route);
            if !predictions.is_empty() {
                push_signal(&mut events, &mut sequence, measured_step + 1, predictions)?;
            }
        }
    }

    let header = TraceHeader {
        kind: "header".to_owned(),
        schema: TRACE_SCHEMA.to_owned(),
        trace_id: format!(
            "m3-{}-r{replicate:02}-n{measured_steps}",
            family.trace_slug()
        ),
        page_count: pages.len(),
        event_count: events.len(),
        charge_quantum: PAGE_BYTES,
        prefetch_model: PREFETCH_MODEL.to_owned(),
    };
    let limits = SimLimits::default();
    let canonical_bytes = serialize_trace(&header, &pages, &events, limits)?;
    let trace = parse_trace(&canonical_bytes, limits)?;

    Ok(GeneratedTrace {
        family,
        replicate,
        measured_steps,
        trace,
        canonical_bytes,
        seed_sha256,
        full_route_sha256: hex::encode(full_digest.finalize()),
        measured_route_sha256: hex::encode(measured_digest.finalize()),
    })
}

fn derive_seed(family: TraceFamily, replicate: u32) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"runnel.m3-trace/v1\0");
    digest.update(family.as_str().as_bytes());
    digest.update([0]);
    digest.update(replicate.to_string().as_bytes());
    digest.finalize().into()
}

fn route_digest(domain: &[u8], route_count: usize) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(
        u64::try_from(route_count)
            .expect("bounded measured route count fits u64")
            .to_le_bytes(),
    );
    digest
}

fn digest_route(digest: &mut Sha256, route: [u32; 2]) {
    digest.update(route[0].to_le_bytes());
    digest.update(route[1].to_le_bytes());
}

fn page_catalog() -> Vec<PageDescriptor> {
    let mut pages = Vec::with_capacity(EXPERT_COUNT * PAGES_PER_EXPERT);
    for expert in 0..EXPERT_COUNT {
        for ordinal in 0..PAGES_PER_EXPERT {
            let id = expert * PAGES_PER_EXPERT + ordinal;
            pages.push(PageDescriptor {
                id: PageId(u32::try_from(id).expect("the fixed page catalog fits u32")),
                logical_bytes: PAGE_BYTES,
                charge_bytes: PAGE_BYTES,
                class: PageClass::Expert {
                    layer: LAYER,
                    expert: u32::try_from(expert).expect("the fixed expert count fits u32"),
                    ordinal: u32::try_from(ordinal).expect("three page ordinals fit u32"),
                },
            });
        }
    }
    pages
}

fn push_signal(
    events: &mut Vec<TraceEvent>,
    sequence: &mut u64,
    target_step: usize,
    predictions: Vec<ExpertPrediction>,
) -> Result<(), SimError> {
    events.push(TraceEvent::RouterSignal {
        sequence: *sequence,
        request: 0,
        target_step: u64::try_from(target_step)
            .map_err(|_| SimError::invalid_config("target step does not fit u64"))?,
        layer: LAYER,
        predictions,
    });
    *sequence = sequence
        .checked_add(1)
        .ok_or_else(|| SimError::invalid_config("event sequence overflow"))?;
    Ok(())
}

fn push_demands(
    events: &mut Vec<TraceEvent>,
    sequence: &mut u64,
    measured_step: usize,
    route: [u32; 2],
) -> Result<(), SimError> {
    let step = u64::try_from(measured_step)
        .map_err(|_| SimError::invalid_config("measured step does not fit u64"))?;
    for expert in route {
        for ordinal in 0..PAGES_PER_EXPERT {
            let page = usize::try_from(expert)
                .expect("generated expert identifier fits usize")
                .checked_mul(PAGES_PER_EXPERT)
                .and_then(|base| base.checked_add(ordinal))
                .ok_or_else(|| SimError::invalid_config("generated page identifier overflow"))?;
            events.push(TraceEvent::Demand {
                sequence: *sequence,
                request: 0,
                step,
                page: PageId(
                    u32::try_from(page).map_err(|_| {
                        SimError::invalid_config("page identifier does not fit u32")
                    })?,
                ),
            });
            *sequence = sequence
                .checked_add(1)
                .ok_or_else(|| SimError::invalid_config("event sequence overflow"))?;
        }
    }
    Ok(())
}

struct RouteGenerator {
    family: TraceFamily,
    rng: Xoshiro256StarStar,
    permutation: [u32; EXPERT_COUNT],
    markov_cluster: usize,
}

impl RouteGenerator {
    fn new(family: TraceFamily, seed: [u8; 32]) -> Self {
        let mut rng = Xoshiro256StarStar::from_seed(seed);
        let mut permutation = std::array::from_fn(|index| {
            u32::try_from(index).expect("the fixed expert count fits u32")
        });
        for high in (1..EXPERT_COUNT).rev() {
            let selected = usize::try_from(
                rng.bounded(u64::try_from(high + 1).expect("expert permutation bound fits u64")),
            )
            .expect("bounded expert index fits usize");
            permutation.swap(high, selected);
        }
        let markov_cluster = if family == TraceFamily::MarkovClusters {
            usize::try_from(rng.bounded(16)).expect("bounded cluster index fits usize")
        } else {
            0
        };
        Self {
            family,
            rng,
            permutation,
            markov_cluster,
        }
    }

    fn next_route(&mut self, absolute_step: usize) -> [u32; 2] {
        let ranks = match self.family {
            TraceFamily::StationaryZipf => self.harmonic_pair(0, EXPERT_COUNT),
            TraceFamily::ScanPollution => self.scan_pollution(absolute_step),
            TraceFamily::PhaseShift => {
                let group = (absolute_step / 256) % 8;
                self.harmonic_pair(group * 16, 16)
            }
            TraceFamily::CyclicPressure => {
                let offset = (absolute_step * 2) % EXPERT_COUNT;
                [offset, (offset + 1) % EXPERT_COUNT]
            }
            TraceFamily::MarkovClusters => self.markov_pair(absolute_step),
            TraceFamily::IidUniform => self.uniform_pair(EXPERT_COUNT),
        };
        [self.permutation[ranks[0]], self.permutation[ranks[1]]]
    }

    fn scan_pollution(&mut self, absolute_step: usize) -> [usize; 2] {
        let phase = absolute_step % 64;
        if phase < 48 {
            self.harmonic_pair(0, 16)
        } else {
            let epoch = absolute_step / 64;
            let scan = phase - 48;
            let first = (epoch * 32 + scan * 2) % 112;
            [16 + first, 16 + (first + 1) % 112]
        }
    }

    fn markov_pair(&mut self, absolute_step: usize) -> [usize; 2] {
        if absolute_step != 0 {
            let transition = self.rng.bounded(100);
            if (85..95).contains(&transition) {
                self.markov_cluster = (self.markov_cluster + 1) % 16;
            } else if transition >= 95 {
                let other = usize::try_from(self.rng.bounded(14))
                    .expect("bounded cluster offset fits usize");
                self.markov_cluster = (self.markov_cluster + 2 + other) % 16;
            }
        }
        self.harmonic_pair(self.markov_cluster * 8, 8)
    }

    fn uniform_pair(&mut self, count: usize) -> [usize; 2] {
        let count_u64 = u64::try_from(count).expect("fixed family size fits u64");
        let first =
            usize::try_from(self.rng.bounded(count_u64)).expect("bounded expert rank fits usize");
        let second_without_first = usize::try_from(self.rng.bounded(count_u64 - 1))
            .expect("bounded expert rank fits usize");
        let second = if second_without_first >= first {
            second_without_first + 1
        } else {
            second_without_first
        };
        [first, second]
    }

    fn harmonic_pair(&mut self, start: usize, count: usize) -> [usize; 2] {
        let first = self.draw_harmonic(count, None);
        let second = self.draw_harmonic(count, Some(first));
        [start + first, start + second]
    }

    fn draw_harmonic(&mut self, count: usize, excluded: Option<usize>) -> usize {
        const WEIGHT_NUMERATOR: u64 = 1_000_000;
        let total = (0..count)
            .filter(|index| Some(*index) != excluded)
            .map(|index| WEIGHT_NUMERATOR / (u64::try_from(index).unwrap() + 1))
            .sum();
        let mut draw = self.rng.bounded(total);
        for index in 0..count {
            if Some(index) == excluded {
                continue;
            }
            let weight = WEIGHT_NUMERATOR / (u64::try_from(index).unwrap() + 1);
            if draw < weight {
                return index;
            }
            draw -= weight;
        }
        unreachable!("the harmonic draw is strictly below its summed weight")
    }
}

#[derive(Default)]
struct CausalPredictor {
    transitions: Vec<u16>,
    observed_routes: u64,
}

impl CausalPredictor {
    fn observe(&mut self, prior: [u32; 2], current: [u32; 2]) {
        if self.transitions.is_empty() {
            self.transitions = vec![0; EXPERT_COUNT * EXPERT_COUNT];
        }
        for source in prior {
            for destination in current {
                let index = usize::try_from(source).expect("generated expert fits usize")
                    * EXPERT_COUNT
                    + usize::try_from(destination).expect("generated expert fits usize");
                self.transitions[index] = self.transitions[index].saturating_add(1);
            }
        }
        self.observed_routes += 1;
        if self.observed_routes.is_multiple_of(PREDICTOR_AGE_INTERVAL) {
            for count in &mut self.transitions {
                *count = (*count).div_ceil(2);
            }
        }
    }

    fn predict(&self, current: [u32; 2]) -> Vec<ExpertPrediction> {
        if self.transitions.is_empty() {
            return Vec::new();
        }
        let mut candidates = Vec::with_capacity(EXPERT_COUNT);
        let mut total = 0_u64;
        for expert in 0..EXPERT_COUNT {
            let count = current
                .iter()
                .map(|source| {
                    let row = usize::try_from(*source).expect("generated expert fits usize");
                    u64::from(self.transitions[row * EXPERT_COUNT + expert])
                })
                .sum::<u64>();
            total += count;
            if count != 0 {
                candidates.push((count, expert));
            }
        }
        if total < PREDICTOR_MIN_SUPPORT {
            return Vec::new();
        }

        candidates.sort_unstable_by(|left, right| {
            right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1))
        });
        candidates
            .into_iter()
            .take(2)
            .filter_map(|(count, expert)| {
                let score = count
                    .checked_mul(SCORE_SCALE)?
                    .checked_div(total)
                    .and_then(|value| u32::try_from(value).ok())?;
                (score >= PREDICTOR_MIN_SCORE_PPM).then(|| ExpertPrediction {
                    expert: u32::try_from(expert).expect("the fixed expert count fits u32"),
                    score_ppm: score,
                })
            })
            .collect()
    }
}

// Independent encoding of Blackman and Vigna's published xoshiro256**
// transition. Algorithm and reference-license provenance are recorded in
// docs/PRIOR_ART.md; no reference implementation source is reused here.
struct Xoshiro256StarStar {
    state: [u64; 4],
}

impl Xoshiro256StarStar {
    fn from_seed(seed: [u8; 32]) -> Self {
        let mut state = [0_u64; 4];
        for (word, bytes) in state.iter_mut().zip(seed.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().expect("SHA-256 chunks have eight bytes"));
        }
        if state == [0; 4] {
            state[0] = 1;
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let temporary = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= temporary;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn bounded(&mut self, bound: u64) -> u64 {
        assert_ne!(bound, 0, "bounded draws require a nonzero range");
        let rejection_threshold = bound.wrapping_neg() % bound;
        loop {
            let draw = self.next_u64();
            if draw >= rejection_threshold {
                return draw % bound;
            }
        }
    }
}
