use runnel_fixture::{MultiPageFixture, PAGE_SIZE};
use runnel_sim::{
    PageClass, PageDescriptor, PageId, PolicySpec, SimLimits, SimulationConfig, TraceEvent,
    TraceHeader, parse_trace, serialize_trace, simulate,
};
use runnel_store::{
    ArtifactSource, AsyncReader, AsyncReaderConfig, CacheConfig, Control, PageCache,
};

const ALLOCATION_ALIGNMENT_BYTES: u64 = 64;
const DEMAND_SEQUENCE: [usize; 5] = [0, 1, 0, 2, 2];

fn aligned_allocation_bytes(logical_bytes: u64) -> u64 {
    logical_bytes
        .div_ceil(ALLOCATION_ALIGNMENT_BYTES)
        .checked_mul(ALLOCATION_ALIGNMENT_BYTES)
        .expect("fixture page allocation fits in u64")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lru_matches_production_aligned_page_pool_allocation_accounting() {
    let temporary = tempfile::tempdir().expect("temporary artifact root");
    let root = temporary.path().join("artifact");
    let fixture = MultiPageFixture::build();
    let identity = fixture.write_new(&root).expect("write multi-page fixture");
    let source = ArtifactSource::open(&root).expect("open descriptor-safe source");
    let artifact = source
        .open_with_expected_id(
            Default::default(),
            identity.artifact_id,
            &Control::unbounded(),
        )
        .expect("open stored artifact");
    let specs = artifact
        .page_specs()
        .collect::<Result<Vec<_>, _>>()
        .expect("collect page specifications");
    let full_page_bytes = u64::from(PAGE_SIZE);
    assert_eq!(
        specs
            .iter()
            .map(runnel_store::PageSpec::length)
            .collect::<Vec<_>>(),
        [full_page_bytes, full_page_bytes, 17]
    );

    let reader = AsyncReader::new(
        artifact.reader(),
        AsyncReaderConfig::new(1, 8).expect("async reader configuration"),
    )
    .expect("async reader");
    let cache = PageCache::new(
        reader,
        CacheConfig::new(full_page_bytes, full_page_bytes)
            .expect("one-full-page cache configuration")
            .with_max_inflight_bytes(full_page_bytes)
            .with_max_loads(1),
    )
    .expect("production page cache");

    let mut production_peak_resident_allocation_bytes = 0;
    let mut production_peak_page_pool_allocation_bytes = 0;
    for page_index in DEMAND_SEQUENCE {
        let lease = cache
            .get(specs[page_index].clone(), Control::unbounded())
            .await
            .expect("production cache demand");
        assert_eq!(
            u64::try_from(lease.len()).expect("fixture page length fits u64"),
            specs[page_index].length()
        );
        let snapshot = cache.metrics();
        production_peak_resident_allocation_bytes =
            production_peak_resident_allocation_bytes.max(snapshot.resident_bytes);
        production_peak_page_pool_allocation_bytes =
            production_peak_page_pool_allocation_bytes.max(snapshot.page_pool_bytes);
        drop(lease);
    }
    let production = cache.metrics();

    let pages = specs
        .iter()
        .enumerate()
        .map(|(index, spec)| PageDescriptor {
            id: PageId(u32::try_from(index).expect("three fixture pages fit u32")),
            logical_bytes: spec.length(),
            charge_bytes: aligned_allocation_bytes(spec.length()),
            class: PageClass::Shared,
        })
        .collect::<Vec<_>>();
    assert_eq!(pages[2].charge_bytes, ALLOCATION_ALIGNMENT_BYTES);
    let events = DEMAND_SEQUENCE
        .iter()
        .enumerate()
        .map(|(sequence, page_index)| TraceEvent::Demand {
            sequence: u64::try_from(sequence).expect("small event sequence fits u64"),
            request: 0,
            step: u64::try_from(sequence).expect("small event step fits u64"),
            page: PageId(u32::try_from(*page_index).expect("fixture page index fits u32")),
        })
        .collect::<Vec<_>>();
    let header = TraceHeader {
        kind: "header".to_owned(),
        schema: "runnel.cache-trace/1".to_owned(),
        trace_id: "production-store-crosscheck".to_owned(),
        page_count: pages.len(),
        event_count: events.len(),
        charge_quantum: ALLOCATION_ALIGNMENT_BYTES,
        prefetch_model: "instant-between-events-v1".to_owned(),
    };
    let limits = SimLimits::default();
    let canonical =
        serialize_trace(&header, &pages, &events, limits).expect("serialize canonical trace");
    let trace = parse_trace(&canonical, limits).expect("parse canonical trace");
    let simulated = simulate(
        &trace,
        &SimulationConfig {
            capacity_bytes: full_page_bytes,
            policy: PolicySpec::Lru,
        },
    )
    .expect("simulate production access sequence");
    let metrics = simulated.metrics;

    assert_eq!(production.hits, 1);
    assert_eq!(production.misses, 4);
    assert_eq!(production.admissions, 4);
    assert_eq!(production.evictions, 3);
    assert_eq!(production.demand_bytes, 3 * full_page_bytes + 2 * 17);
    assert_eq!(production.physical_read_bytes, 3 * full_page_bytes + 17);
    assert_eq!(metrics.ordinary_demand_hits, production.hits);
    assert_eq!(metrics.demand_misses, production.misses);
    assert_eq!(metrics.admissions, production.admissions);
    assert_eq!(metrics.evictions, production.evictions);
    assert_eq!(metrics.demand_logical_bytes, production.demand_bytes);
    assert_eq!(metrics.demand_load_bytes, production.physical_read_bytes);

    assert_eq!(production.resident_bytes, ALLOCATION_ALIGNMENT_BYTES);
    assert_eq!(production.page_pool_bytes, production.resident_bytes);
    assert_eq!(
        metrics.final_resident_charge_bytes,
        production.resident_bytes
    );
    assert_eq!(production_peak_resident_allocation_bytes, full_page_bytes);
    assert_eq!(
        production_peak_page_pool_allocation_bytes,
        production_peak_resident_allocation_bytes
    );
    assert_eq!(
        metrics.peak_resident_charge_bytes,
        production_peak_resident_allocation_bytes
    );
}
