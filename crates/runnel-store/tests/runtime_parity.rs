use std::collections::BTreeMap;

use runnel_fixture::{FixtureArtifact, PAGE_SIZE};
use runnel_format::{Artifact, Limits, TensorRecord};
use runnel_runtime::{RuntimeError, TinyModel, TinyTokenizer};
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
) -> TinyModel {
    TinyModel::from_verified_tensor_bytes(manifest, |descriptor| {
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
    let sync_model = model_from_bytes(stored.manifest(), sync_tensors);

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
    let async_model = model_from_bytes(stored.manifest(), async_tensors);

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
