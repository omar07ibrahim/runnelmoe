use std::collections::BTreeMap;

use runnel_fixture::{FixtureArtifact, MultiPageFixture, PAGE_SIZE};
use runnel_format::{Artifact, Limits, TensorRecord};
use runnel_runtime::{BackendRequest, RuntimeError, TinyModel, TinyTokenizer};
use runnel_store::{
    ArtifactSource, AsyncReader, AsyncReaderConfig, CacheConfig, Control, PageCache, PageSpec,
};

fn append_intersection(
    output: &mut Vec<u8>,
    tensor: &TensorRecord,
    specification: &PageSpec,
    page: &[u8],
) {
    let tensor_end = tensor.offset.checked_add(tensor.length).unwrap();
    let page_end = specification
        .offset()
        .checked_add(specification.length())
        .unwrap();
    let start = tensor.offset.max(specification.offset());
    let end = tensor_end.min(page_end);
    assert!(start < end);
    let local_start = usize::try_from(start - specification.offset()).unwrap();
    let local_end = usize::try_from(end - specification.offset()).unwrap();
    output.extend_from_slice(&page[local_start..local_end]);
}

fn model_from_bytes(
    manifest: &runnel_format::Manifest,
    mut tensors: BTreeMap<u64, Vec<u8>>,
    request: BackendRequest,
) -> TinyModel {
    TinyModel::from_verified_tensor_bytes_with_backend(manifest, request, |descriptor| {
        tensors.remove(&descriptor.id).ok_or_else(|| {
            RuntimeError::InvalidArtifact(format!(
                "verified storage omitted tensor {}",
                descriptor.role
            ))
        })
    })
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_and_async_cached_loads_preserve_numerical_results() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("artifact");
    let fixture = FixtureArtifact::build();
    let identity = fixture.write_new(&root).unwrap();

    let eager =
        Artifact::open_with_expected_id(&root, Limits::default(), identity.artifact_id).unwrap();
    let eager_model = TinyModel::from_artifact(&eager).unwrap();

    let source = ArtifactSource::open(&root).unwrap();
    let stored = source
        .open_with_expected_id(Limits::default(), identity.artifact_id, &Control::default())
        .unwrap();

    let mut sync_tensors = BTreeMap::new();
    for tensor in &stored.manifest().tensors {
        let mut bytes = Vec::with_capacity(usize::try_from(tensor.length).unwrap());
        for specification in stored
            .tensor_page_specs(tensor.id)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
        {
            let (page, _) = stored
                .reader()
                .read(&specification, &Control::default())
                .unwrap();
            append_intersection(&mut bytes, tensor, &specification, page.bytes());
        }
        assert_eq!(bytes.len(), usize::try_from(tensor.length).unwrap());
        sync_tensors.insert(tensor.id, bytes);
    }
    let sync_model = model_from_bytes(stored.manifest(), sync_tensors, BackendRequest::Auto);

    let async_reader =
        AsyncReader::new(stored.reader(), AsyncReaderConfig::new(2, 4).unwrap()).unwrap();
    let cache = PageCache::new(
        async_reader,
        CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE)).unwrap(),
    )
    .unwrap();
    let mut async_tensors = BTreeMap::new();
    for tensor in &stored.manifest().tensors {
        let mut bytes = Vec::with_capacity(usize::try_from(tensor.length).unwrap());
        for specification in stored
            .tensor_page_specs(tensor.id)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
        {
            let lease = cache
                .get(specification.clone(), Control::default())
                .await
                .unwrap();
            append_intersection(&mut bytes, tensor, &specification, lease.bytes());
        }
        assert_eq!(bytes.len(), usize::try_from(tensor.length).unwrap());
        async_tensors.insert(tensor.id, bytes);
    }
    let async_model = model_from_bytes(stored.manifest(), async_tensors, BackendRequest::Auto);

    let prompt = TinyTokenizer.encode("moe").unwrap();
    let expected = eager_model.generate_greedy(&prompt, 4).unwrap();
    assert_eq!(sync_model.generate_greedy(&prompt, 4).unwrap(), expected);
    assert_eq!(async_model.generate_greedy(&prompt, 4).unwrap(), expected);

    let metrics = cache.metrics();
    assert_eq!(metrics.misses, 1);
    assert_eq!(metrics.admissions, 1);
    assert_eq!(metrics.physical_read_bytes, identity.object_length);
    assert_eq!(metrics.hits, stored.manifest().tensors.len() as u64 - 1);
    assert_eq!(
        metrics.resident_bytes,
        identity.object_length.div_ceil(64) * 64
    );
    cache.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_dtype_v2_bytes_and_scalar_results_match_eager_sync_and_async_loads() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("artifact-v2");
    let fixture = FixtureArtifact::build_v2();
    let identity = fixture.write_new(&root).unwrap();
    assert_eq!(identity.object_length, 5_600);

    let eager =
        Artifact::open_with_expected_id(&root, Limits::default(), identity.artifact_id).unwrap();
    let eager_model =
        TinyModel::from_artifact_with_backend(&eager, BackendRequest::Scalar).unwrap();

    let source = ArtifactSource::open(&root).unwrap();
    let stored = source
        .open_with_expected_id(Limits::default(), identity.artifact_id, &Control::default())
        .unwrap();

    let mut sync_tensors = BTreeMap::new();
    for tensor in &stored.manifest().tensors {
        let mut bytes = Vec::with_capacity(usize::try_from(tensor.length).unwrap());
        for specification in stored
            .tensor_page_specs(tensor.id)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
        {
            let (page, _) = stored
                .reader()
                .read(&specification, &Control::default())
                .unwrap();
            append_intersection(&mut bytes, tensor, &specification, page.bytes());
        }
        assert_eq!(
            bytes,
            eager.tensor_bytes(tensor.id).unwrap(),
            "sync bytes differ for {}",
            tensor.role
        );
        sync_tensors.insert(tensor.id, bytes);
    }
    let sync_model = model_from_bytes(stored.manifest(), sync_tensors, BackendRequest::Scalar);

    let async_reader =
        AsyncReader::new(stored.reader(), AsyncReaderConfig::new(2, 4).unwrap()).unwrap();
    let cache = PageCache::new(
        async_reader,
        CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE)).unwrap(),
    )
    .unwrap();
    let mut async_tensors = BTreeMap::new();
    for tensor in &stored.manifest().tensors {
        let mut bytes = Vec::with_capacity(usize::try_from(tensor.length).unwrap());
        for specification in stored
            .tensor_page_specs(tensor.id)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
        {
            let lease = cache
                .get(specification.clone(), Control::default())
                .await
                .unwrap();
            append_intersection(&mut bytes, tensor, &specification, lease.bytes());
        }
        assert_eq!(
            bytes,
            eager.tensor_bytes(tensor.id).unwrap(),
            "async bytes differ for {}",
            tensor.role
        );
        async_tensors.insert(tensor.id, bytes);
    }
    let async_model = model_from_bytes(stored.manifest(), async_tensors, BackendRequest::Scalar);

    let prompt = TinyTokenizer.encode("moe").unwrap();
    let expected = eager_model.generate_greedy(&prompt, 4).unwrap();
    assert_eq!(sync_model.generate_greedy(&prompt, 4).unwrap(), expected);
    assert_eq!(async_model.generate_greedy(&prompt, 4).unwrap(), expected);

    let metrics = cache.metrics();
    assert_eq!(metrics.misses, 1);
    assert_eq!(metrics.admissions, 1);
    assert_eq!(metrics.hits, 21);
    assert_eq!(metrics.physical_read_bytes, 5_600);
    assert_eq!(metrics.resident_bytes, 5_632);
    cache.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_generation_parity_survives_forced_eviction_between_tensors() {
    let temporary = tempfile::tempdir().unwrap();
    let tiny_root = temporary.path().join("tiny");
    let multi_root = temporary.path().join("multi");
    let tiny_identity = FixtureArtifact::build().write_new(&tiny_root).unwrap();
    let multi_identity = MultiPageFixture::build().write_new(&multi_root).unwrap();

    let eager =
        Artifact::open_with_expected_id(&tiny_root, Limits::default(), tiny_identity.artifact_id)
            .unwrap();
    let expected_model = TinyModel::from_artifact(&eager).unwrap();

    let tiny_source = ArtifactSource::open(&tiny_root).unwrap();
    let tiny = tiny_source
        .open_with_expected_id(
            Limits::default(),
            tiny_identity.artifact_id,
            &Control::default(),
        )
        .unwrap();
    let multi_source = ArtifactSource::open(&multi_root).unwrap();
    let multi = multi_source
        .open_with_expected_id(
            Limits::default(),
            multi_identity.artifact_id,
            &Control::default(),
        )
        .unwrap();
    let interference = multi.page_specs().next().unwrap().unwrap();
    assert_eq!(interference.length(), u64::from(PAGE_SIZE));

    let async_reader =
        AsyncReader::new(tiny.reader(), AsyncReaderConfig::new(2, 4).unwrap()).unwrap();
    let cache = PageCache::new(
        async_reader,
        CacheConfig::new(u64::from(PAGE_SIZE), u64::from(PAGE_SIZE))
            .unwrap()
            .with_max_inflight_bytes(u64::from(PAGE_SIZE))
            .with_max_loads(1),
    )
    .unwrap();

    let mut tensors = BTreeMap::new();
    for (tensor_index, tensor) in tiny.manifest().tensors.iter().enumerate() {
        let mut bytes = Vec::with_capacity(usize::try_from(tensor.length).unwrap());
        let specifications = tiny
            .tensor_page_specs(tensor.id)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(specifications.len(), 1);
        for specification in specifications {
            let lease = cache
                .get(specification.clone(), Control::default())
                .await
                .unwrap();
            append_intersection(&mut bytes, tensor, &specification, lease.bytes());
        }
        assert_eq!(bytes.len(), usize::try_from(tensor.length).unwrap());
        tensors.insert(tensor.id, bytes);

        if tensor_index == 0 {
            let interference_lease = cache
                .get(interference.clone(), Control::default())
                .await
                .unwrap();
            assert_eq!(
                interference_lease.bytes().len(),
                usize::try_from(interference.length()).unwrap()
            );
        }
    }
    let forced_model = model_from_bytes(tiny.manifest(), tensors, BackendRequest::Auto);

    let prompt = TinyTokenizer.encode("moe").unwrap();
    let expected = expected_model.generate_greedy(&prompt, 4).unwrap();
    let actual = forced_model.generate_greedy(&prompt, 4).unwrap();
    assert_eq!(actual, expected);

    let tensor_count = u64::try_from(tiny.manifest().tensors.len()).unwrap();
    assert_eq!(tensor_count, 22);
    let metrics = cache.metrics();
    assert_eq!(metrics.hits, 20);
    assert_eq!(metrics.misses, 3);
    assert_eq!(metrics.admissions, 3);
    assert_eq!(metrics.evictions, 2);
    assert_eq!(metrics.demand_bytes, 239_424);
    assert_eq!(metrics.physical_read_bytes, 81_344);
    assert_eq!(metrics.resident_bytes, 7_936);
    assert_eq!(metrics.page_pool_bytes, 7_936);
    cache.shutdown();
}
