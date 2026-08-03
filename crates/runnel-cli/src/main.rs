use std::{collections::BTreeMap, error::Error, io, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand, ValueEnum};
use runnel_fixture::{FixtureArtifact, FixtureIdentity, MultiPageFixture, PAGE_SIZE};
use runnel_format::{Artifact, Limits, Manifest, TensorRecord};
use runnel_runtime::{EOS_TOKEN, RuntimeError, TinyModel, TinyTokenizer};
use runnel_store::{
    AccessReason, ArtifactSource, AsyncReader, AsyncReaderConfig, CacheConfig, Control,
    MetricsSnapshot, PageCache, PageSpec, RssSample, StoredArtifact, TraceEvent, TraceOutcome,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(name = "runnel", version, about = "RunnelMoE reference runtime")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Materialize the deterministic tiny RMOA fixture in a new directory.
    Fixture {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Generate tokens with a verified RMOA artifact.
    Generate {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
        #[arg(long, value_enum, default_value_t = Strategy::Greedy)]
        strategy: Strategy,
        #[arg(long)]
        json: bool,
    },
    /// Generate and verify the tiny artifact in a temporary directory, then run it.
    Demo {
        #[arg(long, default_value = "moe")]
        prompt: String,
        #[arg(long, default_value_t = 4)]
        max_new_tokens: usize,
        #[arg(long)]
        json: bool,
    },
    /// Verify synchronous and cached data-plane parity on disposable fixtures.
    DataPlaneDemo {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Strategy {
    Greedy,
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("greedy")
    }
}

#[derive(Debug, Serialize)]
struct AdapterOutput<'a> {
    id: &'a str,
    version: u64,
}

#[derive(Debug, Serialize)]
struct GenerationOutput<'a> {
    schema_version: u64,
    artifact_id: String,
    adapter: AdapterOutput<'a>,
    input_token_count: usize,
    generated_ids: Vec<u32>,
    text: String,
    stop_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct FixtureOutput {
    schema_version: u64,
    artifact_id: String,
    object_digest: String,
    object_length: u64,
    page_table_digest: String,
    page_table_length: u64,
}

#[derive(Debug, Serialize)]
struct DataPlaneFixtureOutput {
    artifact_id: String,
    object_digest: String,
    object_length: u64,
    page_table_digest: String,
    page_table_length: u64,
}

impl From<FixtureIdentity> for DataPlaneFixtureOutput {
    fn from(identity: FixtureIdentity) -> Self {
        Self {
            artifact_id: identity.artifact_id.to_string(),
            object_digest: identity.object_digest.to_string(),
            object_length: identity.object_length,
            page_table_digest: identity.page_table_digest.to_string(),
            page_table_length: identity.page_table_length,
        }
    }
}

#[derive(Debug, Serialize)]
struct DataPlaneFixturesOutput {
    tiny: DataPlaneFixtureOutput,
    multi_page: DataPlaneFixtureOutput,
}

#[derive(Debug, Serialize)]
struct DemandTraceOutput {
    access: &'static str,
    page_indices: [u64; 5],
    page_lengths: [u64; 3],
    event_count: u64,
    outcomes: TraceOutcomeCountsOutput,
    events: Vec<NormalizedTraceEventOutput>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
struct NormalizedTraceEventOutput {
    sequence: u64,
    outcome: &'static str,
    reason: &'static str,
    object_digest: String,
    page_size: u64,
    page_index: u64,
    logical_bytes: u64,
}

#[derive(Debug, Default, Serialize)]
struct TraceOutcomeCountsOutput {
    hit: u64,
    miss: u64,
    load_started: u64,
    load_coalesced: u64,
    late_prefetch: u64,
    prefetch_coalesced: u64,
    admitted: u64,
    evicted: u64,
    retired: u64,
    load_failed: u64,
    cancelled: u64,
    prefetch_useful: u64,
    prefetch_wasted: u64,
    prefetch_redundant: u64,
    prefetch_dropped: u64,
}

#[derive(Debug, Serialize)]
struct ExactBytesPerTokenOutput {
    numerator_bytes: u64,
    denominator_tokens: u64,
}

#[derive(Debug, Serialize)]
struct GenerationCacheOutput {
    completed_generated_tokens: u64,
    physical_read_bytes: u64,
    bytes_per_generated_token: ExactBytesPerTokenOutput,
}

#[derive(Debug, Serialize)]
struct ForcedEvictionPageOutput {
    object_digest: String,
    page_size: u64,
    page_index: u64,
    logical_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ForcedEvictionMetricsOutput {
    demand_bytes: u64,
    physical_read_bytes: u64,
    hits: u64,
    misses: u64,
    admissions: u64,
    evictions: u64,
    coalesced_demands: u64,
    prefetch: PrefetchMetricsOutput,
    accounted: AccountedGaugesOutput,
    trace_events_dropped: u64,
}

#[derive(Debug, Serialize)]
struct ForcedEvictionTraceOutput {
    access: &'static str,
    event_count: u64,
    outcomes: TraceOutcomeCountsOutput,
    events: Vec<NormalizedTraceEventOutput>,
}

#[derive(Debug, Serialize)]
struct ForcedEvictionGenerationOutput {
    full_generation_parity: bool,
    cache_capacity_bytes: u64,
    tensor_count: u64,
    tensor_page_accesses: u64,
    interference_page_accesses: u64,
    tensor_page: ForcedEvictionPageOutput,
    interference_page: ForcedEvictionPageOutput,
    metrics: ForcedEvictionMetricsOutput,
    trace: ForcedEvictionTraceOutput,
}

#[derive(Debug, Serialize)]
struct PrefetchMetricsOutput {
    bytes: u64,
    coalesced: u64,
    late: u64,
    useful: u64,
    wasted: u64,
    redundant: u64,
    dropped: u64,
}

#[derive(Debug, Serialize)]
struct AccountedGaugesOutput {
    active_loads: u64,
    page_pool_bytes: u64,
    inflight_bytes: u64,
    resident_bytes: u64,
    retiring_bytes: u64,
    leases: u64,
}

#[derive(Debug, Serialize)]
struct RssOutput {
    resident_bytes: u64,
    peak_bytes: u64,
}

impl From<RssSample> for RssOutput {
    fn from(sample: RssSample) -> Self {
        Self {
            resident_bytes: sample.resident_bytes,
            peak_bytes: sample.peak_bytes,
        }
    }
}

#[derive(Debug, Serialize)]
struct DataPlaneMetricsOutput {
    demand_bytes: u64,
    physical_read_bytes: u64,
    hits: u64,
    misses: u64,
    admissions: u64,
    evictions: u64,
    coalesced_demands: u64,
    prefetch: PrefetchMetricsOutput,
    wait_nanoseconds: u64,
    io_nanoseconds: u64,
    accounted: AccountedGaugesOutput,
    observed_rss: Option<RssOutput>,
    trace_events_dropped: u64,
}

impl From<MetricsSnapshot> for DataPlaneMetricsOutput {
    fn from(metrics: MetricsSnapshot) -> Self {
        Self {
            demand_bytes: metrics.demand_bytes,
            physical_read_bytes: metrics.physical_read_bytes,
            hits: metrics.hits,
            misses: metrics.misses,
            admissions: metrics.admissions,
            evictions: metrics.evictions,
            coalesced_demands: metrics.coalesced_demands,
            prefetch: PrefetchMetricsOutput {
                bytes: metrics.prefetch_bytes,
                coalesced: metrics.coalesced_prefetches,
                late: metrics.late_prefetches,
                useful: metrics.useful_prefetches,
                wasted: metrics.wasted_prefetches,
                redundant: metrics.redundant_prefetches,
                dropped: metrics.dropped_prefetches,
            },
            wait_nanoseconds: metrics.wait_nanoseconds,
            io_nanoseconds: metrics.io_nanoseconds,
            accounted: AccountedGaugesOutput {
                active_loads: metrics.active_loads,
                page_pool_bytes: metrics.page_pool_bytes,
                inflight_bytes: metrics.inflight_bytes,
                resident_bytes: metrics.resident_bytes,
                retiring_bytes: metrics.retiring_bytes,
                leases: metrics.leases,
            },
            observed_rss: metrics.rss.map(Into::into),
            trace_events_dropped: metrics.trace_events_dropped,
        }
    }
}

#[derive(Debug, Serialize)]
struct DataPlaneDemoOutput {
    schema_version: u64,
    fixtures: DataPlaneFixturesOutput,
    prompt: &'static str,
    max_new_tokens: usize,
    generated_ids: Vec<u32>,
    generated_text: String,
    parity: bool,
    generation_cache: GenerationCacheOutput,
    forced_eviction_generation: ForcedEvictionGenerationOutput,
    trace: DemandTraceOutput,
    cache_capacity_bytes: u64,
    metrics: DataPlaneMetricsOutput,
}

fn main() -> ExitCode {
    match run(Arguments::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn run(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    match arguments.command {
        Command::Fixture { output, json } => {
            let fixture = FixtureArtifact::build();
            let identity = fixture.write_new(output)?;
            print_fixture(identity, json)?;
        }
        Command::Generate {
            artifact,
            prompt,
            max_new_tokens,
            strategy: Strategy::Greedy,
            json,
        } => {
            let artifact = Artifact::open(artifact, Limits::default())?;
            print_generation(&artifact, &prompt, max_new_tokens, json)?;
        }
        Command::Demo {
            prompt,
            max_new_tokens,
            json,
        } => {
            let temporary = tempfile::tempdir()?;
            let root = temporary.path().join("tiny-rmoa");
            FixtureArtifact::build().write_new(&root)?;
            let artifact = Artifact::open(root, Limits::default())?;
            print_generation(&artifact, &prompt, max_new_tokens, json)?;
        }
        Command::DataPlaneDemo { json } => print_data_plane_demo(json)?,
    }
    Ok(())
}

fn print_data_plane_demo(json: bool) -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_time()
        .build()?;
    let output = runtime.block_on(build_data_plane_demo())?;
    if json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("data-plane parity: {}", output.parity);
        println!("generated IDs: {:?}", output.generated_ids);
        println!(
            "generation cache: {} physical bytes / {} completed tokens",
            output.generation_cache.physical_read_bytes,
            output.generation_cache.completed_generated_tokens,
        );
        println!(
            "forced-eviction parity: {} ({} evictions)",
            output.forced_eviction_generation.full_generation_parity,
            output.forced_eviction_generation.metrics.evictions,
        );
        println!("trace pages: {:?}", output.trace.page_indices);
        println!(
            "cache: {} hit, {} misses, {} admissions, {} evictions, {} physical bytes",
            output.metrics.hits,
            output.metrics.misses,
            output.metrics.admissions,
            output.metrics.evictions,
            output.metrics.physical_read_bytes,
        );
    }
    Ok(())
}

async fn build_data_plane_demo() -> Result<DataPlaneDemoOutput, Box<dyn Error>> {
    const PROMPT: &str = "moe";
    const MAX_NEW_TOKENS: usize = 4;
    const DEMAND_TRACE: [u64; 5] = [0, 1, 0, 2, 2];

    let temporary = tempfile::tempdir()?;
    let tiny_root = temporary.path().join("tiny");
    let multi_root = temporary.path().join("multi-page");
    let tiny_fixture = FixtureArtifact::build();
    let multi_fixture = MultiPageFixture::build();
    let tiny_identity = tiny_fixture.write_new(&tiny_root)?;
    let multi_identity = multi_fixture.write_new(&multi_root)?;

    let tiny_source = ArtifactSource::open(&tiny_root)?;
    let tiny_stored = tiny_source.open_with_expected_id(
        Limits::default(),
        tiny_identity.artifact_id,
        &Control::unbounded(),
    )?;
    let sync_tensors = collect_sync_tensor_bytes(&tiny_stored)?;
    let sync_model = model_from_verified_tensor_bytes(tiny_stored.manifest(), sync_tensors)?;

    let tiny_async = AsyncReader::new(tiny_stored.reader(), AsyncReaderConfig::new(2, 4)?)?;
    let tiny_cache = PageCache::new(
        tiny_async,
        CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE))?,
    )?;
    let cached_tensors = collect_cached_tensor_bytes(&tiny_stored, &tiny_cache).await?;
    let cached_model = model_from_verified_tensor_bytes(tiny_stored.manifest(), cached_tensors)?;

    let prompt_ids = TinyTokenizer.encode(PROMPT)?;
    let sync_generation = sync_model.generate_greedy(&prompt_ids, MAX_NEW_TOKENS)?;
    let cached_generation = cached_model.generate_greedy(&prompt_ids, MAX_NEW_TOKENS)?;
    let cached_generation_parity = sync_generation == cached_generation;
    if !cached_generation_parity {
        return Err(io::Error::other("synchronous and cached generation differ").into());
    }
    let generated_text = TinyTokenizer.decode(&sync_generation.generated_tokens)?;
    let generated_ids = sync_generation.generated_tokens.clone();
    let completed_generated_tokens = u64::try_from(generated_ids.len())?;
    if completed_generated_tokens == 0 {
        return Err(io::Error::other("tiny generation produced no completed tokens").into());
    }
    let tiny_metrics = tiny_cache.metrics();
    if tiny_metrics.physical_read_bytes != tiny_identity.object_length {
        return Err(io::Error::other("tiny cached load physical byte count changed").into());
    }
    tiny_cache.shutdown();

    let multi_source = ArtifactSource::open(&multi_root)?;
    let multi_stored = multi_source.open_with_expected_id(
        Limits::default(),
        multi_identity.artifact_id,
        &Control::unbounded(),
    )?;
    let mut specifications = BTreeMap::new();
    for specification in multi_stored.page_specs() {
        let specification = specification?;
        if specifications
            .insert(specification.key().index(), specification)
            .is_some()
        {
            return Err(io::Error::other("multi-page fixture repeated a page index").into());
        }
    }
    let page_lengths = [
        page_specification(&specifications, 0)?.length(),
        page_specification(&specifications, 1)?.length(),
        page_specification(&specifications, 2)?.length(),
    ];
    if page_lengths != [u64::from(PAGE_SIZE), u64::from(PAGE_SIZE), 17] {
        return Err(io::Error::other("multi-page fixture geometry changed").into());
    }

    // PageSpec retains its authenticated object descriptor, while SyncReader is
    // deliberately stateless. One global cache can therefore exercise pages
    // from both retained artifacts without reopening either pathname.
    let interference_page = page_specification(&specifications, 0)?.clone();
    let forced_reader = AsyncReader::new(tiny_stored.reader(), AsyncReaderConfig::new(2, 4)?)?;
    let forced_cache = PageCache::new(
        forced_reader,
        CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE))?
            .with_max_inflight_bytes(u64::from(PAGE_SIZE))
            .with_max_loads(1)
            .with_trace_capacity(64),
    )?;
    let (forced_tensors, tensor_page_accesses, interference_page_accesses) =
        collect_cached_tensor_bytes_with_interference(
            &tiny_stored,
            &forced_cache,
            &interference_page,
        )
        .await?;
    let forced_model = model_from_verified_tensor_bytes(tiny_stored.manifest(), forced_tensors)?;
    let forced_generation = forced_model.generate_greedy(&prompt_ids, MAX_NEW_TOKENS)?;
    let forced_generation_parity = sync_generation == forced_generation;
    if !forced_generation_parity {
        return Err(io::Error::other(
            "forced-eviction cached generation differs from synchronous generation",
        )
        .into());
    }
    let tensor_count = u64::try_from(tiny_stored.manifest().tensors.len())?;
    let forced_metrics = forced_cache.metrics();
    validate_forced_eviction_metrics(
        forced_metrics,
        tensor_count,
        tensor_page_accesses,
        interference_page_accesses,
        tiny_identity.object_length,
        interference_page.length(),
    )?;
    let forced_trace_events = forced_cache.trace_sink().drain();
    let forced_trace_outcomes = count_trace_outcomes(&forced_trace_events)?;
    let normalized_forced_trace = normalize_trace_events(&forced_trace_events)?;
    let tiny_object_digest = tiny_identity.object_digest.to_string();
    let multi_object_digest = multi_identity.object_digest.to_string();
    validate_forced_eviction_trace(
        &normalized_forced_trace,
        &forced_trace_outcomes,
        &tiny_object_digest,
        &multi_object_digest,
    )?;
    forced_cache.shutdown();

    let parity = cached_generation_parity && forced_generation_parity;

    let trace_reader = AsyncReader::new(multi_stored.reader(), AsyncReaderConfig::new(2, 4)?)?;
    let trace_cache = PageCache::new(
        trace_reader,
        CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE))?
            .with_max_inflight_bytes(u64::from(PAGE_SIZE))
            .with_max_loads(1)
            .with_trace_capacity(64),
    )?;
    let synchronous_reader = multi_stored.reader();
    for page_index in DEMAND_TRACE {
        let specification = page_specification(&specifications, page_index)?;
        let (expected, _stats) = synchronous_reader.read(specification, &Control::unbounded())?;
        let actual = trace_cache
            .get(specification.clone(), Control::unbounded())
            .await?;
        if actual.bytes() != expected.bytes() {
            return Err(io::Error::other("cached page differs from synchronous page").into());
        }
    }

    let metrics = trace_cache.metrics();
    validate_trace_metrics(metrics, page_lengths)?;
    let trace_events = trace_cache.trace_sink().drain();
    let trace_outcomes = count_trace_outcomes(&trace_events)?;
    validate_trace_outcomes(&trace_outcomes, trace_events.len())?;
    let normalized_trace = normalize_trace_events(&trace_events)?;
    validate_normalized_trace(&normalized_trace, &multi_object_digest)?;
    let output = DataPlaneDemoOutput {
        schema_version: 2,
        fixtures: DataPlaneFixturesOutput {
            tiny: tiny_identity.into(),
            multi_page: multi_identity.into(),
        },
        prompt: PROMPT,
        max_new_tokens: MAX_NEW_TOKENS,
        generated_ids,
        generated_text,
        parity,
        generation_cache: GenerationCacheOutput {
            completed_generated_tokens,
            physical_read_bytes: tiny_metrics.physical_read_bytes,
            bytes_per_generated_token: ExactBytesPerTokenOutput {
                numerator_bytes: tiny_metrics.physical_read_bytes,
                denominator_tokens: completed_generated_tokens,
            },
        },
        forced_eviction_generation: ForcedEvictionGenerationOutput {
            full_generation_parity: forced_generation_parity,
            cache_capacity_bytes: u64::from(PAGE_SIZE),
            tensor_count,
            tensor_page_accesses,
            interference_page_accesses,
            tensor_page: ForcedEvictionPageOutput {
                object_digest: tiny_object_digest,
                page_size: u64::from(PAGE_SIZE),
                page_index: 0,
                logical_bytes: tiny_identity.object_length,
            },
            interference_page: ForcedEvictionPageOutput {
                object_digest: interference_page.key().object().to_string(),
                page_size: interference_page.key().page_size(),
                page_index: interference_page.key().index(),
                logical_bytes: interference_page.length(),
            },
            metrics: ForcedEvictionMetricsOutput {
                demand_bytes: forced_metrics.demand_bytes,
                physical_read_bytes: forced_metrics.physical_read_bytes,
                hits: forced_metrics.hits,
                misses: forced_metrics.misses,
                admissions: forced_metrics.admissions,
                evictions: forced_metrics.evictions,
                coalesced_demands: forced_metrics.coalesced_demands,
                prefetch: PrefetchMetricsOutput {
                    bytes: forced_metrics.prefetch_bytes,
                    coalesced: forced_metrics.coalesced_prefetches,
                    late: forced_metrics.late_prefetches,
                    useful: forced_metrics.useful_prefetches,
                    wasted: forced_metrics.wasted_prefetches,
                    redundant: forced_metrics.redundant_prefetches,
                    dropped: forced_metrics.dropped_prefetches,
                },
                accounted: AccountedGaugesOutput {
                    active_loads: forced_metrics.active_loads,
                    page_pool_bytes: forced_metrics.page_pool_bytes,
                    inflight_bytes: forced_metrics.inflight_bytes,
                    resident_bytes: forced_metrics.resident_bytes,
                    retiring_bytes: forced_metrics.retiring_bytes,
                    leases: forced_metrics.leases,
                },
                trace_events_dropped: forced_metrics.trace_events_dropped,
            },
            trace: ForcedEvictionTraceOutput {
                access: "demand",
                event_count: u64::try_from(forced_trace_events.len())?,
                outcomes: forced_trace_outcomes,
                events: normalized_forced_trace,
            },
        },
        trace: DemandTraceOutput {
            access: "demand",
            page_indices: DEMAND_TRACE,
            page_lengths,
            event_count: u64::try_from(trace_events.len())?,
            outcomes: trace_outcomes,
            events: normalized_trace,
        },
        cache_capacity_bytes: u64::from(PAGE_SIZE),
        metrics: metrics.into(),
    };
    trace_cache.shutdown();
    Ok(output)
}

fn normalize_trace_events(
    events: &[TraceEvent],
) -> Result<Vec<NormalizedTraceEventOutput>, Box<dyn Error>> {
    events
        .iter()
        .map(|event| {
            Ok(NormalizedTraceEventOutput {
                sequence: event.sequence,
                outcome: trace_outcome_name(event.outcome)?,
                reason: access_reason_name(event.reason),
                object_digest: event.key.object().to_string(),
                page_size: event.key.page_size(),
                page_index: event.key.index(),
                logical_bytes: event.bytes,
            })
        })
        .collect()
}

fn trace_outcome_name(outcome: TraceOutcome) -> Result<&'static str, Box<dyn Error>> {
    Ok(match outcome {
        TraceOutcome::Hit => "hit",
        TraceOutcome::Miss => "miss",
        TraceOutcome::LoadStarted => "load_started",
        TraceOutcome::LoadCoalesced => "load_coalesced",
        TraceOutcome::LatePrefetch => "late_prefetch",
        TraceOutcome::PrefetchCoalesced => "prefetch_coalesced",
        TraceOutcome::Admitted => "admitted",
        TraceOutcome::Evicted => "evicted",
        TraceOutcome::Retired => "retired",
        TraceOutcome::LoadFailed => "load_failed",
        TraceOutcome::Cancelled => "cancelled",
        TraceOutcome::PrefetchUseful => "prefetch_useful",
        TraceOutcome::PrefetchWasted => "prefetch_wasted",
        TraceOutcome::PrefetchRedundant => "prefetch_redundant",
        TraceOutcome::PrefetchDropped => "prefetch_dropped",
        _ => return Err(io::Error::other("trace contained an unsupported outcome").into()),
    })
}

const fn access_reason_name(reason: AccessReason) -> &'static str {
    match reason {
        AccessReason::Demand => "demand",
        AccessReason::Prefetch => "prefetch",
    }
}

fn validate_normalized_trace(
    events: &[NormalizedTraceEventOutput],
    object_digest: &str,
) -> Result<(), Box<dyn Error>> {
    const EXPECTED: [(&str, u64, u64); 16] = [
        ("miss", 0, 65_536),
        ("load_started", 0, 65_536),
        ("admitted", 0, 65_536),
        ("miss", 1, 65_536),
        ("evicted", 0, 65_536),
        ("load_started", 1, 65_536),
        ("admitted", 1, 65_536),
        ("miss", 0, 65_536),
        ("evicted", 1, 65_536),
        ("load_started", 0, 65_536),
        ("admitted", 0, 65_536),
        ("miss", 2, 17),
        ("evicted", 0, 65_536),
        ("load_started", 2, 17),
        ("admitted", 2, 17),
        ("hit", 2, 17),
    ];
    if events.len() != EXPECTED.len() {
        return Err(io::Error::other("fixed demand trace length changed").into());
    }
    for (sequence, (event, (outcome, page_index, logical_bytes))) in
        events.iter().zip(EXPECTED).enumerate()
    {
        let sequence = u64::try_from(sequence)?;
        if event.sequence != sequence
            || event.outcome != outcome
            || event.reason != "demand"
            || event.object_digest != object_digest
            || event.page_size != u64::from(PAGE_SIZE)
            || event.page_index != page_index
            || event.logical_bytes != logical_bytes
        {
            return Err(io::Error::other("fixed normalized trace changed").into());
        }
    }
    Ok(())
}

fn validate_forced_eviction_trace(
    events: &[NormalizedTraceEventOutput],
    outcomes: &TraceOutcomeCountsOutput,
    tensor_object_digest: &str,
    interference_object_digest: &str,
) -> Result<(), Box<dyn Error>> {
    const PREFIX: [(&str, bool, u64); 11] = [
        ("miss", false, 7_904),
        ("load_started", false, 7_904),
        ("admitted", false, 7_904),
        ("miss", true, 65_536),
        ("evicted", false, 7_904),
        ("load_started", true, 65_536),
        ("admitted", true, 65_536),
        ("miss", false, 7_904),
        ("evicted", true, 65_536),
        ("load_started", false, 7_904),
        ("admitted", false, 7_904),
    ];
    if events.len() != 31 {
        return Err(io::Error::other("forced-eviction trace length changed").into());
    }
    for (sequence, event) in events.iter().enumerate() {
        let (outcome, uses_interference, logical_bytes) = if sequence < PREFIX.len() {
            PREFIX[sequence]
        } else {
            ("hit", false, 7_904)
        };
        let expected_digest = if uses_interference {
            interference_object_digest
        } else {
            tensor_object_digest
        };
        if event.sequence != u64::try_from(sequence)?
            || event.outcome != outcome
            || event.reason != "demand"
            || event.object_digest != expected_digest
            || event.page_size != u64::from(PAGE_SIZE)
            || event.page_index != 0
            || event.logical_bytes != logical_bytes
        {
            return Err(io::Error::other("forced-eviction normalized trace changed").into());
        }
    }
    let exact_outcomes = outcomes.hit == 20
        && outcomes.miss == 3
        && outcomes.load_started == 3
        && outcomes.admitted == 3
        && outcomes.evicted == 2
        && outcomes.load_coalesced == 0
        && outcomes.late_prefetch == 0
        && outcomes.prefetch_coalesced == 0
        && outcomes.retired == 0
        && outcomes.load_failed == 0
        && outcomes.cancelled == 0
        && outcomes.prefetch_useful == 0
        && outcomes.prefetch_wasted == 0
        && outcomes.prefetch_redundant == 0
        && outcomes.prefetch_dropped == 0;
    if !exact_outcomes {
        return Err(io::Error::other("forced-eviction trace outcomes changed").into());
    }
    Ok(())
}

fn count_trace_outcomes(events: &[TraceEvent]) -> Result<TraceOutcomeCountsOutput, Box<dyn Error>> {
    let mut counts = TraceOutcomeCountsOutput::default();
    for event in events {
        let count = match event.outcome {
            TraceOutcome::Hit => &mut counts.hit,
            TraceOutcome::Miss => &mut counts.miss,
            TraceOutcome::LoadStarted => &mut counts.load_started,
            TraceOutcome::LoadCoalesced => &mut counts.load_coalesced,
            TraceOutcome::LatePrefetch => &mut counts.late_prefetch,
            TraceOutcome::PrefetchCoalesced => &mut counts.prefetch_coalesced,
            TraceOutcome::Admitted => &mut counts.admitted,
            TraceOutcome::Evicted => &mut counts.evicted,
            TraceOutcome::Retired => &mut counts.retired,
            TraceOutcome::LoadFailed => &mut counts.load_failed,
            TraceOutcome::Cancelled => &mut counts.cancelled,
            TraceOutcome::PrefetchUseful => &mut counts.prefetch_useful,
            TraceOutcome::PrefetchWasted => &mut counts.prefetch_wasted,
            TraceOutcome::PrefetchRedundant => &mut counts.prefetch_redundant,
            TraceOutcome::PrefetchDropped => &mut counts.prefetch_dropped,
            _ => return Err(io::Error::other("trace contained an unsupported outcome").into()),
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| io::Error::other("trace outcome count overflow"))?;
    }
    Ok(counts)
}

fn validate_trace_outcomes(
    outcomes: &TraceOutcomeCountsOutput,
    event_count: usize,
) -> Result<(), Box<dyn Error>> {
    let exact = event_count == 16
        && outcomes.hit == 1
        && outcomes.miss == 4
        && outcomes.load_started == 4
        && outcomes.admitted == 4
        && outcomes.evicted == 3
        && outcomes.load_coalesced == 0
        && outcomes.late_prefetch == 0
        && outcomes.prefetch_coalesced == 0
        && outcomes.retired == 0
        && outcomes.load_failed == 0
        && outcomes.cancelled == 0
        && outcomes.prefetch_useful == 0
        && outcomes.prefetch_wasted == 0
        && outcomes.prefetch_redundant == 0
        && outcomes.prefetch_dropped == 0;
    if !exact {
        return Err(io::Error::other("fixed demand trace outcomes changed").into());
    }
    Ok(())
}

fn collect_sync_tensor_bytes(
    stored: &StoredArtifact,
) -> Result<BTreeMap<u64, Vec<u8>>, Box<dyn Error>> {
    let mut tensors = BTreeMap::new();
    for tensor in &stored.manifest().tensors {
        let mut bytes = Vec::with_capacity(tensor_length(tensor)?);
        for specification in stored.tensor_page_specs(tensor.id)? {
            let specification = specification?;
            let (page, _stats) = stored
                .reader()
                .read(&specification, &Control::unbounded())?;
            append_intersection(&mut bytes, tensor, &specification, page.bytes())?;
        }
        require_tensor_length(tensor, &bytes)?;
        tensors.insert(tensor.id, bytes);
    }
    Ok(tensors)
}

async fn collect_cached_tensor_bytes(
    stored: &StoredArtifact,
    cache: &PageCache,
) -> Result<BTreeMap<u64, Vec<u8>>, Box<dyn Error>> {
    let mut tensors = BTreeMap::new();
    for tensor in &stored.manifest().tensors {
        let mut bytes = Vec::with_capacity(tensor_length(tensor)?);
        for specification in stored.tensor_page_specs(tensor.id)? {
            let specification = specification?;
            let lease = cache
                .get(specification.clone(), Control::unbounded())
                .await?;
            append_intersection(&mut bytes, tensor, &specification, lease.bytes())?;
        }
        require_tensor_length(tensor, &bytes)?;
        tensors.insert(tensor.id, bytes);
    }
    Ok(tensors)
}

async fn collect_cached_tensor_bytes_with_interference(
    stored: &StoredArtifact,
    cache: &PageCache,
    interference_page: &PageSpec,
) -> Result<(BTreeMap<u64, Vec<u8>>, u64, u64), Box<dyn Error>> {
    let mut tensors = BTreeMap::new();
    let mut tensor_page_accesses = 0_u64;
    let mut interference_page_accesses = 0_u64;
    for (tensor_index, tensor) in stored.manifest().tensors.iter().enumerate() {
        let mut bytes = Vec::with_capacity(tensor_length(tensor)?);
        for specification in stored.tensor_page_specs(tensor.id)? {
            let specification = specification?;
            let lease = cache
                .get(specification.clone(), Control::unbounded())
                .await?;
            append_intersection(&mut bytes, tensor, &specification, lease.bytes())?;
            drop(lease);
            tensor_page_accesses = tensor_page_accesses
                .checked_add(1)
                .ok_or_else(|| io::Error::other("tensor-page access count overflow"))?;
        }
        require_tensor_length(tensor, &bytes)?;
        tensors.insert(tensor.id, bytes);

        if tensor_index == 0 {
            let interference = cache
                .get(interference_page.clone(), Control::unbounded())
                .await?;
            if u64::try_from(interference.bytes().len())? != interference_page.length() {
                return Err(io::Error::other("interference page length changed").into());
            }
            drop(interference);
            interference_page_accesses = interference_page_accesses
                .checked_add(1)
                .ok_or_else(|| io::Error::other("interference-page access count overflow"))?;
        }
    }
    Ok((tensors, tensor_page_accesses, interference_page_accesses))
}

fn validate_forced_eviction_metrics(
    metrics: MetricsSnapshot,
    tensor_count: u64,
    tensor_page_accesses: u64,
    interference_page_accesses: u64,
    tensor_page_bytes: u64,
    interference_page_bytes: u64,
) -> Result<(), Box<dyn Error>> {
    if tensor_count < 2 || tensor_page_accesses != tensor_count || interference_page_accesses != 1 {
        return Err(io::Error::other("forced-eviction access schedule changed").into());
    }
    let expected_demands = tensor_count
        .checked_add(interference_page_accesses)
        .ok_or_else(|| io::Error::other("forced-eviction access count overflow"))?;
    let expected_demand_bytes = tensor_page_bytes
        .checked_mul(tensor_count)
        .and_then(|bytes| bytes.checked_add(interference_page_bytes))
        .ok_or_else(|| io::Error::other("forced-eviction byte count overflow"))?;
    let expected_physical_bytes = tensor_page_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(interference_page_bytes))
        .ok_or_else(|| io::Error::other("forced-eviction physical byte count overflow"))?;
    let expected_hits = expected_demands
        .checked_sub(3)
        .ok_or_else(|| io::Error::other("forced-eviction hit count underflow"))?;
    let final_tensor_charge = tensor_page_bytes
        .div_ceil(64)
        .checked_mul(64)
        .ok_or_else(|| io::Error::other("forced-eviction tensor charge overflow"))?;
    let exact = metrics.demand_bytes == expected_demand_bytes
        && metrics.physical_read_bytes == expected_physical_bytes
        && metrics.hits == expected_hits
        && metrics.misses == 3
        && metrics.admissions == 3
        && metrics.evictions == 2
        && metrics.coalesced_demands == 0
        && metrics.prefetch_bytes == 0
        && metrics.coalesced_prefetches == 0
        && metrics.late_prefetches == 0
        && metrics.useful_prefetches == 0
        && metrics.wasted_prefetches == 0
        && metrics.redundant_prefetches == 0
        && metrics.dropped_prefetches == 0
        && metrics.active_loads == 0
        && metrics.inflight_bytes == 0
        && metrics.retiring_bytes == 0
        && metrics.leases == 0
        && metrics.page_pool_bytes == final_tensor_charge
        && metrics.resident_bytes == final_tensor_charge
        && metrics.trace_events_dropped == 0;
    if !exact {
        return Err(io::Error::other("forced-eviction cache accounting changed").into());
    }
    Ok(())
}

fn append_intersection(
    output: &mut Vec<u8>,
    tensor: &TensorRecord,
    specification: &PageSpec,
    page: &[u8],
) -> Result<(), Box<dyn Error>> {
    let tensor_end = tensor
        .offset
        .checked_add(tensor.length)
        .ok_or_else(|| io::Error::other("tensor range overflow"))?;
    let page_end = specification
        .offset()
        .checked_add(specification.length())
        .ok_or_else(|| io::Error::other("page range overflow"))?;
    let start = tensor.offset.max(specification.offset());
    let end = tensor_end.min(page_end);
    if start >= end {
        return Err(io::Error::other("tensor page does not intersect tensor range").into());
    }
    let local_start = usize::try_from(start - specification.offset())?;
    let local_end = usize::try_from(end - specification.offset())?;
    let intersection = page
        .get(local_start..local_end)
        .ok_or_else(|| io::Error::other("verified page is shorter than its specification"))?;
    output.extend_from_slice(intersection);
    Ok(())
}

fn tensor_length(tensor: &TensorRecord) -> Result<usize, Box<dyn Error>> {
    Ok(usize::try_from(tensor.length)?)
}

fn require_tensor_length(tensor: &TensorRecord, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if bytes.len() != tensor_length(tensor)? {
        return Err(io::Error::other("trimmed tensor bytes have the wrong length").into());
    }
    Ok(())
}

fn model_from_verified_tensor_bytes(
    manifest: &Manifest,
    mut tensors: BTreeMap<u64, Vec<u8>>,
) -> Result<TinyModel, RuntimeError> {
    TinyModel::from_verified_tensor_bytes(manifest, |descriptor| {
        tensors.remove(&descriptor.id).ok_or_else(|| {
            RuntimeError::InvalidArtifact(format!(
                "verified storage omitted tensor {}",
                descriptor.role
            ))
        })
    })
}

fn page_specification(
    specifications: &BTreeMap<u64, PageSpec>,
    page_index: u64,
) -> Result<&PageSpec, Box<dyn Error>> {
    specifications
        .get(&page_index)
        .ok_or_else(|| io::Error::other("multi-page fixture omitted a requested page").into())
}

fn validate_trace_metrics(
    metrics: MetricsSnapshot,
    page_lengths: [u64; 3],
) -> Result<(), Box<dyn Error>> {
    const PAGE_BUFFER_CAPACITY_QUANTUM: u64 = 64;

    let expected_demand_bytes = page_lengths[0]
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(page_lengths[2].checked_mul(2)?))
        .ok_or_else(|| io::Error::other("trace demand byte count overflow"))?;
    let expected_physical_bytes = page_lengths[0]
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(page_lengths[2]))
        .ok_or_else(|| io::Error::other("trace physical byte count overflow"))?;
    let expected_retained_bytes = page_lengths[2]
        .div_ceil(PAGE_BUFFER_CAPACITY_QUANTUM)
        .checked_mul(PAGE_BUFFER_CAPACITY_QUANTUM)
        .ok_or_else(|| io::Error::other("trace retained byte count overflow"))?;
    let prefetch_total = metrics
        .prefetch_bytes
        .saturating_add(metrics.coalesced_prefetches)
        .saturating_add(metrics.late_prefetches)
        .saturating_add(metrics.useful_prefetches)
        .saturating_add(metrics.wasted_prefetches)
        .saturating_add(metrics.redundant_prefetches)
        .saturating_add(metrics.dropped_prefetches);
    let exact = metrics.demand_bytes == expected_demand_bytes
        && metrics.physical_read_bytes == expected_physical_bytes
        && metrics.hits == 1
        && metrics.misses == 4
        && metrics.admissions == 4
        && metrics.evictions == 3
        && metrics.coalesced_demands == 0
        && prefetch_total == 0
        && metrics.active_loads == 0
        && metrics.page_pool_bytes == expected_retained_bytes
        && metrics.inflight_bytes == 0
        && metrics.resident_bytes == expected_retained_bytes
        && metrics.retiring_bytes == 0
        && metrics.leases == 0
        && metrics.trace_events_dropped == 0;
    if !exact {
        return Err(io::Error::other("fixed demand trace metrics changed").into());
    }
    Ok(())
}

fn print_fixture(identity: FixtureIdentity, json: bool) -> Result<(), Box<dyn Error>> {
    let output = FixtureOutput {
        schema_version: 1,
        artifact_id: identity.artifact_id.to_string(),
        object_digest: identity.object_digest.to_string(),
        object_length: identity.object_length,
        page_table_digest: identity.page_table_digest.to_string(),
        page_table_length: identity.page_table_length,
    };
    if json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("artifact: {}", output.artifact_id);
        println!(
            "object: {} ({} bytes)",
            output.object_digest, output.object_length
        );
        println!(
            "page table: {} ({} bytes)",
            output.page_table_digest, output.page_table_length
        );
    }
    Ok(())
}

fn print_generation(
    artifact: &Artifact,
    prompt: &str,
    max_new_tokens: usize,
    json: bool,
) -> Result<(), Box<dyn Error>> {
    let tokenizer = TinyTokenizer;
    let input_ids = tokenizer.encode(prompt)?;
    let model = TinyModel::from_artifact(artifact)?;
    let generation = model.generate_greedy(&input_ids, max_new_tokens)?;
    let stop_reason = if generation.generated_tokens.last() == Some(&EOS_TOKEN) {
        "eos"
    } else {
        "max_new_tokens"
    };
    let output = GenerationOutput {
        schema_version: 1,
        artifact_id: artifact.artifact_id().to_string(),
        adapter: AdapterOutput {
            id: &artifact.manifest().adapter.id,
            version: artifact.manifest().adapter.version,
        },
        input_token_count: input_ids.len(),
        text: tokenizer.decode(&generation.generated_tokens)?,
        generated_ids: generation.generated_tokens,
        stop_reason,
    };

    if json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("artifact: {}", output.artifact_id);
        println!("generated IDs: {:?}", output.generated_ids);
        println!("text: {}", output.text);
        println!("stop reason: {}", output.stop_reason);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_contract_is_stable() {
        let fixture = FixtureArtifact::build();
        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        let model = TinyModel::from_artifact(&artifact).unwrap();
        let input = TinyTokenizer.encode("moe").unwrap();
        let generated = model.generate_greedy(&input, 4).unwrap();
        assert_eq!(generated.generated_tokens, [15, 11, 20, 9]);
        assert_eq!(
            TinyTokenizer.decode(&generated.generated_tokens).unwrap(),
            "njsh"
        );
    }
}
