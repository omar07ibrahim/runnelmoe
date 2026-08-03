use std::sync::Arc;

use runnel_fixture::MultiPageFixture;
use runnel_format::Limits;
use runnel_store::{
    ArtifactSource, AsyncReader, AsyncReaderConfig, CancellationToken, Control, ErrorCategory,
    PageSpec,
};
use tempfile::TempDir;
use tokio::sync::Barrier;

struct OpenFixture {
    _temporary: TempDir,
    specs: Vec<PageSpec>,
}

fn open_fixture() -> OpenFixture {
    let temporary = tempfile::tempdir().expect("temporary root");
    let root = temporary.path().join("artifact");
    let fixture = MultiPageFixture::build();
    let identity = fixture.write_new(&root).expect("write fixture");
    let source = ArtifactSource::open(&root).expect("open descriptor-safe source");
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn asynchronous_reads_match_the_authoritative_reader() {
    let fixture = open_fixture();
    let synchronous = runnel_store::SyncReader::new();
    let asynchronous = AsyncReader::new(
        synchronous.clone(),
        AsyncReaderConfig::new(2, 2).expect("configuration"),
    )
    .expect("async reader");

    for specification in &fixture.specs {
        let (expected, expected_stats) = synchronous
            .read(specification, &Control::unbounded())
            .expect("sync read");
        let (actual, actual_stats) = asynchronous
            .read(specification.clone(), Control::unbounded())
            .await
            .expect("async read");
        assert_eq!(actual.key(), expected.key());
        assert_eq!(actual.bytes(), expected.bytes());
        assert_eq!(
            actual_stats.physical_bytes(),
            expected_stats.physical_bytes()
        );
        assert!(actual_stats.io_nanoseconds() > 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_before_admission_and_shutdown_are_terminal() {
    let fixture = open_fixture();
    let reader = AsyncReader::new(
        runnel_store::SyncReader::new(),
        AsyncReaderConfig::new(1, 1).expect("configuration"),
    )
    .expect("async reader");
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = reader
        .read(fixture.specs[0].clone(), Control::with_cancellation(token))
        .await
        .expect_err("cancelled read");
    assert_eq!(cancelled.category(), ErrorCategory::Cancelled);

    reader.shutdown();
    assert!(reader.is_shutdown());
    let shut_down = reader
        .read(fixture.specs[0].clone(), Control::unbounded())
        .await
        .expect_err("shutdown read");
    assert_eq!(shut_down.category(), ErrorCategory::Shutdown);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bounded_queue_reports_saturation_without_losing_completions() {
    const REQUESTS: usize = 512;

    let fixture = open_fixture();
    let reader = AsyncReader::new(
        runnel_store::SyncReader::new(),
        AsyncReaderConfig::new(1, 1).expect("configuration"),
    )
    .expect("async reader");
    let barrier = Arc::new(Barrier::new(REQUESTS + 1));
    let mut tasks = Vec::with_capacity(REQUESTS);

    for _ in 0..REQUESTS {
        let task_reader = reader.clone();
        let specification = fixture.specs[0].clone();
        let task_barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            task_barrier.wait().await;
            task_reader.read(specification, Control::unbounded()).await
        }));
    }
    barrier.wait().await;

    let mut succeeded = 0;
    let mut saturated = 0;
    for task in tasks {
        match task.await.expect("reader task") {
            Ok((_page, _stats)) => succeeded += 1,
            Err(error) if error.category() == ErrorCategory::ResourceExhausted => saturated += 1,
            Err(error) => panic!("unexpected reader error: {error}"),
        }
    }
    assert!(succeeded >= 1);
    assert!(saturated >= 1);
    assert_eq!(succeeded + saturated, REQUESTS);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_race_preserves_all_worker_owned_results() {
    const REQUESTS: usize = 128;

    let fixture = open_fixture();
    let reader = AsyncReader::new(
        runnel_store::SyncReader::new(),
        AsyncReaderConfig::new(2, 2).expect("configuration"),
    )
    .expect("async reader");
    let barrier = Arc::new(Barrier::new(REQUESTS + 1));
    let mut tasks = Vec::with_capacity(REQUESTS);
    for _ in 0..REQUESTS {
        let task_reader = reader.clone();
        let specification = fixture.specs[1].clone();
        let task_barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            task_barrier.wait().await;
            task_reader.read(specification, Control::unbounded()).await
        }));
    }
    barrier.wait().await;
    reader.shutdown();

    for task in tasks {
        match task.await.expect("reader task") {
            Ok((page, stats)) => {
                assert_eq!(page.len(), 65_536);
                assert_eq!(stats.physical_bytes(), 65_536);
            }
            Err(error) => assert!(matches!(
                error.category(),
                ErrorCategory::ResourceExhausted | ErrorCategory::Shutdown
            )),
        }
    }
}
