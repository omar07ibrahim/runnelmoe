use std::collections::BTreeMap;
use std::fs;

use runnel_format::{
    Artifact, ArtifactBytes, Digest, FormatError, Limits, Manifest, ObjectRecord, PageTable,
};

const PAGE_SIZE: u32 = 65_536;

struct Fixture {
    manifest: Vec<u8>,
    object_digest: Digest,
    object: Vec<u8>,
    page_table_digest: Digest,
    page_table: Vec<u8>,
}

impl Fixture {
    fn parts(&self) -> ArtifactBytes {
        ArtifactBytes {
            manifest: self.manifest.clone(),
            objects: BTreeMap::from([(self.object_digest, self.object.clone())]),
            page_tables: BTreeMap::from([(self.page_table_digest, self.page_table.clone())]),
        }
    }

    fn record(&self) -> ObjectRecord {
        Manifest::parse(&self.manifest).unwrap().objects[0].clone()
    }
}

fn fixture() -> Fixture {
    fixture_for_object((1_u8..=8).collect())
}

fn fixture_for_object(object: Vec<u8>) -> Fixture {
    let object_digest = Digest::of(&object);
    let page_table = make_page_table(&object, object_digest, PAGE_SIZE);
    let page_table_digest = Digest::of(&page_table);
    let tensor = format!(
        "{{\"dtype\":\"u8\",\"id\":0,\"length\":{},\"object\":\"{}\",\"offset\":0,\"role\":\"weight\",\"shape\":[{}]}}",
        object.len(),
        object_digest,
        object.len()
    );
    let manifest = make_manifest(
        object_digest,
        object.len() as u64,
        page_table_digest,
        page_table.len() as u64,
        &tensor,
    );
    Fixture {
        manifest,
        object_digest,
        object,
        page_table_digest,
        page_table,
    }
}

fn make_manifest(
    object_digest: Digest,
    object_length: u64,
    page_table_digest: Digest,
    page_table_length: u64,
    tensors: &str,
) -> Vec<u8> {
    format!(
        concat!(
            "{{",
            "\"adapter\":{{\"id\":\"runnel.tiny-causal-moe\",\"version\":1}},",
            "\"format\":\"rmoa\",",
            "\"model\":{{",
            "\"context_length\":16,",
            "\"expert_hidden_size\":12,",
            "\"hidden_size\":8,",
            "\"num_experts\":4,",
            "\"num_heads\":2,",
            "\"num_layers\":1,",
            "\"top_k\":2,",
            "\"vocab_size\":32",
            "}},",
            "\"objects\":[{{",
            "\"digest\":\"{}\",",
            "\"length\":{},",
            "\"page_size\":65536,",
            "\"page_table\":\"{}\",",
            "\"page_table_length\":{}",
            "}}],",
            "\"tensors\":[{}],",
            "\"tokenizer\":{{\"id\":\"runnel.ascii32\",\"version\":1,\"vocab_size\":32}},",
            "\"version\":1",
            "}}\n"
        ),
        object_digest, object_length, page_table_digest, page_table_length, tensors
    )
    .into_bytes()
}

fn make_page_table(object: &[u8], object_digest: Digest, page_size: u32) -> Vec<u8> {
    let page_count = object.len().div_ceil(page_size as usize);
    let mut bytes = Vec::with_capacity(64 + page_count * 32);
    bytes.extend_from_slice(b"RMOAPG1\n");
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&page_size.to_le_bytes());
    bytes.extend_from_slice(&(object.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(page_count as u64).to_le_bytes());
    bytes.extend_from_slice(object_digest.as_bytes());
    for page in object.chunks(page_size as usize) {
        bytes.extend_from_slice(Digest::of(page).as_bytes());
    }
    bytes
}

#[test]
fn valid_artifact_exposes_only_verified_tensor_bytes() {
    let fixture = fixture();
    let manifest = Manifest::parse(&fixture.manifest).unwrap();
    assert_eq!(manifest.canonical_bytes(), fixture.manifest);
    assert_eq!(manifest.artifact_id(), Digest::of(&fixture.manifest));
    assert_eq!(manifest.model.hidden_size, 8);
    assert_eq!(manifest.tokenizer.vocab_size, 32);

    let artifact = Artifact::from_bytes(fixture.parts(), Limits::default()).unwrap();
    let tensor = artifact.tensor_by_role("weight").unwrap();
    assert_eq!(tensor.descriptor.id, 0);
    assert_eq!(tensor.bytes, fixture.object);
    assert_eq!(artifact.tensor_bytes(0), Some(fixture.object.as_slice()));
    assert_eq!(artifact.load_tensor("weight").unwrap(), fixture.object);
    assert_eq!(
        artifact
            .page_table(fixture.object_digest)
            .unwrap()
            .page_hash(0),
        Some(*Digest::of(&fixture.object).as_bytes())
    );
    assert!(matches!(
        artifact.load_tensor("missing"),
        Err(FormatError::UnknownTensorRole { .. })
    ));
}

#[test]
fn rejects_duplicate_keys_at_any_object_depth() {
    let bytes = br#"{"adapter":{"id":"runnel.tiny-causal-moe","id":"duplicate","version":1}}
"#;
    assert!(matches!(
        Manifest::parse(bytes),
        Err(FormatError::DuplicateKey { ref key }) if key == "id"
    ));
}

#[test]
fn rejects_noncanonical_encoding_and_missing_final_lf() {
    let fixture = fixture();
    let mut no_lf = fixture.manifest.clone();
    assert_eq!(no_lf.pop(), Some(b'\n'));
    assert!(matches!(
        Manifest::parse(&no_lf),
        Err(FormatError::NonCanonicalManifest)
    ));

    let mut whitespace = fixture.manifest.clone();
    whitespace.insert(1, b' ');
    assert!(matches!(
        Manifest::parse(&whitespace),
        Err(FormatError::NonCanonicalManifest)
    ));

    let mut extra_lf = fixture.manifest;
    extra_lf.push(b'\n');
    assert!(matches!(
        Manifest::parse(&extra_lf),
        Err(FormatError::NonCanonicalManifest)
    ));
}

#[test]
fn every_manifest_truncation_is_rejected() {
    let fixture = fixture();
    for length in 0..fixture.manifest.len() {
        Manifest::parse(&fixture.manifest[..length]).expect_err("truncation must fail closed");
    }
}

#[test]
fn manifest_size_and_nesting_are_bounded_before_schema_allocation() {
    let limits = Limits {
        manifest_bytes: 16,
        ..Limits::default()
    };
    assert!(matches!(
        Manifest::parse_with_limits(&[b' '; 17], limits),
        Err(FormatError::ManifestTooLarge {
            actual: 17,
            limit: 16
        })
    ));

    let mut nested = String::new();
    for _ in 0..13 {
        nested.push('[');
    }
    nested.push('0');
    for _ in 0..13 {
        nested.push(']');
    }
    assert!(matches!(
        Manifest::parse(nested.as_bytes()),
        Err(FormatError::NestingTooDeep { limit: 12 })
    ));
}

#[test]
fn rejects_disallowed_json_primitives_and_unicode() {
    let valid = String::from_utf8(fixture().manifest).unwrap();
    let null = valid.replacen("\"format\":\"rmoa\"", "\"format\":null", 1);
    assert!(matches!(
        Manifest::parse(null.as_bytes()),
        Err(FormatError::NullNotAllowed)
    ));

    let float = valid.replacen("\"context_length\":16", "\"context_length\":1.5", 1);
    assert!(matches!(
        Manifest::parse(float.as_bytes()),
        Err(FormatError::FloatingPointNotAllowed)
    ));

    let negative = valid.replacen("\"context_length\":16", "\"context_length\":-1", 1);
    assert!(matches!(
        Manifest::parse(negative.as_bytes()),
        Err(FormatError::IntegerOutOfRange { .. })
    ));

    let too_large = valid.replacen(
        "\"context_length\":16",
        "\"context_length\":9007199254740992",
        1,
    );
    assert!(matches!(
        Manifest::parse(too_large.as_bytes()),
        Err(FormatError::IntegerOutOfRange { .. })
    ));

    let unicode = valid.replacen("runnel.ascii32", "runnel.asciié", 1);
    assert!(matches!(
        Manifest::parse(unicode.as_bytes()),
        Err(FormatError::NonAsciiString)
    ));
}

#[test]
fn rejects_schema_extensions_and_invalid_adapter_invariants() {
    let valid = String::from_utf8(fixture().manifest).unwrap();
    let unknown = valid.replacen("\"version\":1}\n", "\"version\":1,\"x\":1}\n", 1);
    assert!(matches!(
        Manifest::parse(unknown.as_bytes()),
        Err(FormatError::Schema { .. })
    ));

    let top_k = valid.replacen("\"top_k\":2", "\"top_k\":5", 1);
    assert!(matches!(
        Manifest::parse(top_k.as_bytes()),
        Err(FormatError::Schema { ref path, .. }) if path == "$.model.top_k"
    ));

    let heads = valid.replacen("\"num_heads\":2", "\"num_heads\":3", 1);
    assert!(matches!(
        Manifest::parse(heads.as_bytes()),
        Err(FormatError::Schema { ref path, .. }) if path == "$.model.hidden_size"
    ));

    let vocab = valid.replacen(
        "\"id\":\"runnel.ascii32\",\"version\":1,\"vocab_size\":32",
        "\"id\":\"runnel.ascii32\",\"version\":1,\"vocab_size\":31",
        1,
    );
    assert!(matches!(
        Manifest::parse(vocab.as_bytes()),
        Err(FormatError::Schema { ref path, .. }) if path == "$.tokenizer.vocab_size"
    ));
}

#[test]
fn adapter_versions_are_closed_independently_of_the_rmoa_version() {
    let version_one = String::from_utf8(fixture().manifest).unwrap();
    assert_eq!(
        Manifest::parse(version_one.as_bytes())
            .unwrap()
            .adapter
            .version,
        1
    );

    let version_two = version_one.replacen(
        "\"adapter\":{\"id\":\"runnel.tiny-causal-moe\",\"version\":1}",
        "\"adapter\":{\"id\":\"runnel.tiny-causal-moe\",\"version\":2}",
        1,
    );
    assert_eq!(
        Manifest::parse(version_two.as_bytes())
            .unwrap()
            .adapter
            .version,
        2
    );

    let zero = version_one.replacen(
        "\"adapter\":{\"id\":\"runnel.tiny-causal-moe\",\"version\":1}",
        "\"adapter\":{\"id\":\"runnel.tiny-causal-moe\",\"version\":0}",
        1,
    );
    assert!(matches!(
        Manifest::parse(zero.as_bytes()),
        Err(FormatError::Schema { ref path, .. }) if path == "$.adapter.version"
    ));

    let unsupported = version_one.replacen(
        "\"adapter\":{\"id\":\"runnel.tiny-causal-moe\",\"version\":1}",
        "\"adapter\":{\"id\":\"runnel.tiny-causal-moe\",\"version\":3}",
        1,
    );
    assert!(matches!(
        Manifest::parse(unsupported.as_bytes()),
        Err(FormatError::Schema { ref path, .. }) if path == "$.adapter.version"
    ));

    let format_two = version_one.replacen("\"version\":1}\n", "\"version\":2}\n", 1);
    assert!(matches!(
        Manifest::parse(format_two.as_bytes()),
        Err(FormatError::Schema { ref path, .. }) if path == "$.version"
    ));
}

#[test]
fn digest_parser_requires_exact_lowercase_sha256_form() {
    let digest = Digest::of(b"x");
    assert_eq!(digest.to_string().parse::<Digest>().unwrap(), digest);
    assert!(digest.path_component().parse::<Digest>().is_err());
    assert!(
        format!("sha256:{}", digest.path_component().to_uppercase())
            .parse::<Digest>()
            .is_err()
    );
    assert!("sha256:00".parse::<Digest>().is_err());
}

#[test]
fn rejects_tensor_shape_range_alignment_and_coverage_corruption() {
    let fixture = fixture();
    let invalid_shape = format!(
        "{{\"dtype\":\"u8\",\"id\":0,\"length\":8,\"object\":\"{}\",\"offset\":0,\"role\":\"weight\",\"shape\":[7]}}",
        fixture.object_digest
    );
    let manifest = make_manifest(
        fixture.object_digest,
        8,
        fixture.page_table_digest,
        96,
        &invalid_shape,
    );
    assert!(matches!(
        Manifest::parse(&manifest),
        Err(FormatError::Schema { ref path, .. }) if path.ends_with(".length")
    ));

    let gap = format!(
        "{{\"dtype\":\"u8\",\"id\":0,\"length\":4,\"object\":\"{}\",\"offset\":4,\"role\":\"weight\",\"shape\":[4]}}",
        fixture.object_digest
    );
    let manifest = make_manifest(
        fixture.object_digest,
        8,
        fixture.page_table_digest,
        96,
        &gap,
    );
    assert!(matches!(
        Manifest::parse(&manifest),
        Err(FormatError::TensorGap { .. })
    ));

    let trailing = format!(
        "{{\"dtype\":\"u8\",\"id\":0,\"length\":4,\"object\":\"{}\",\"offset\":0,\"role\":\"weight\",\"shape\":[4]}}",
        fixture.object_digest
    );
    let manifest = make_manifest(
        fixture.object_digest,
        8,
        fixture.page_table_digest,
        96,
        &trailing,
    );
    assert!(matches!(
        Manifest::parse(&manifest),
        Err(FormatError::TensorCoverage { .. })
    ));

    let overlap = format!(
        concat!(
            "{{\"dtype\":\"u8\",\"id\":0,\"length\":8,\"object\":\"{}\",\"offset\":0,\"role\":\"a\",\"shape\":[8]}},",
            "{{\"dtype\":\"u8\",\"id\":1,\"length\":4,\"object\":\"{}\",\"offset\":4,\"role\":\"b\",\"shape\":[4]}}"
        ),
        fixture.object_digest, fixture.object_digest
    );
    let manifest = make_manifest(
        fixture.object_digest,
        8,
        fixture.page_table_digest,
        96,
        &overlap,
    );
    assert!(matches!(
        Manifest::parse(&manifest),
        Err(FormatError::TensorOverlap { .. })
    ));

    let unaligned = format!(
        "{{\"dtype\":\"f32-le\",\"id\":0,\"length\":4,\"object\":\"{}\",\"offset\":2,\"role\":\"weight\",\"shape\":[1]}}",
        fixture.object_digest
    );
    let manifest = make_manifest(
        fixture.object_digest,
        8,
        fixture.page_table_digest,
        96,
        &unaligned,
    );
    assert!(matches!(
        Manifest::parse(&manifest),
        Err(FormatError::Schema { ref path, .. }) if path.ends_with(".offset")
    ));
}

#[test]
fn checked_shape_and_page_count_arithmetic_rejects_overflow() {
    let fixture = fixture();
    let overflowing_shape = format!(
        "{{\"dtype\":\"f32-le\",\"id\":0,\"length\":8,\"object\":\"{}\",\"offset\":0,\"role\":\"weight\",\"shape\":[2147483647,2147483647,2147483647]}}",
        fixture.object_digest
    );
    let manifest = make_manifest(
        fixture.object_digest,
        8,
        fixture.page_table_digest,
        96,
        &overflowing_shape,
    );
    assert!(matches!(
        Manifest::parse(&manifest),
        Err(FormatError::ArithmeticOverflow {
            context: "tensor shape product"
        })
    ));

    let record = ObjectRecord {
        digest: fixture.object_digest,
        length: u64::MAX,
        page_size: PAGE_SIZE,
        page_table: fixture.page_table_digest,
        page_table_length: 96,
    };
    assert!(matches!(
        record.page_count(),
        Err(FormatError::ArithmeticOverflow {
            context: "page count"
        })
    ));

    let zero_page_size = ObjectRecord {
        page_size: 0,
        length: 1,
        ..record
    };
    assert!(matches!(
        zero_page_size.page_count(),
        Err(FormatError::ArithmeticOverflow {
            context: "nonzero page size"
        })
    ));
}

#[test]
fn every_page_table_truncation_is_rejected() {
    let fixture = fixture();
    let record = fixture.record();
    for length in 0..fixture.page_table.len() {
        let error = PageTable::parse_verified(&record, fixture.page_table[..length].to_vec())
            .expect_err("truncation must fail closed");
        assert!(matches!(error, FormatError::BlobLengthMismatch { .. }));
    }
}

#[test]
fn page_table_digest_is_checked_before_header_fields() {
    let fixture = fixture();
    let record = fixture.record();
    let mut corrupted = fixture.page_table;
    corrupted[0] ^= 1;
    assert!(matches!(
        PageTable::parse_verified(&record, corrupted),
        Err(FormatError::BlobDigestMismatch {
            kind: "page table",
            ..
        })
    ));
}

#[test]
fn verified_but_malformed_page_table_headers_are_rejected() {
    let fixture = fixture();
    let base_record = fixture.record();
    let mutations: &[fn(&mut [u8])] = &[
        |bytes| bytes[0] ^= 1,
        |bytes| bytes[8..12].copy_from_slice(&2_u32.to_le_bytes()),
        |bytes| bytes[12..16].copy_from_slice(&(PAGE_SIZE * 2).to_le_bytes()),
        |bytes| bytes[16..24].copy_from_slice(&7_u64.to_le_bytes()),
        |bytes| bytes[24..32].copy_from_slice(&2_u64.to_le_bytes()),
        |bytes| bytes[32] ^= 1,
    ];
    for mutate in mutations {
        let mut table = fixture.page_table.clone();
        mutate(&mut table);
        let mut record = base_record.clone();
        record.page_table = Digest::of(&table);
        assert!(matches!(
            PageTable::parse_verified(&record, table),
            Err(FormatError::PageTable { .. })
        ));
    }
}

#[test]
fn every_object_truncation_and_whole_object_corruption_is_rejected() {
    let fixture = fixture();
    let record = fixture.record();
    let table = PageTable::parse_verified(&record, fixture.page_table.clone()).unwrap();
    for length in 0..fixture.object.len() {
        assert!(matches!(
            table.verify_object(&record, &fixture.object[..length]),
            Err(FormatError::BlobLengthMismatch { kind: "object", .. })
        ));
    }
    let mut corrupted = fixture.object;
    corrupted[0] ^= 1;
    assert!(matches!(
        table.verify_object(&record, &corrupted),
        Err(FormatError::BlobDigestMismatch { kind: "object", .. })
    ));
}

#[test]
fn wrong_and_reordered_page_hashes_are_rejected() {
    let fixture = fixture();
    let mut record = fixture.record();
    let mut table_bytes = fixture.page_table;
    table_bytes[64] ^= 1;
    record.page_table = Digest::of(&table_bytes);
    let table = PageTable::parse_verified(&record, table_bytes).unwrap();
    assert!(matches!(
        table.verify_object(&record, &fixture.object),
        Err(FormatError::PageHashMismatch { page_index: 0, .. })
    ));

    let object: Vec<u8> = (0..PAGE_SIZE as usize + 17)
        .map(|index| (index % 251) as u8)
        .collect();
    let object_digest = Digest::of(&object);
    let mut table_bytes = make_page_table(&object, object_digest, PAGE_SIZE);
    let (header_and_first, second_and_rest) = table_bytes.split_at_mut(96);
    header_and_first[64..96].swap_with_slice(&mut second_and_rest[..32]);
    let record = ObjectRecord {
        digest: object_digest,
        length: object.len() as u64,
        page_size: PAGE_SIZE,
        page_table: Digest::of(&table_bytes),
        page_table_length: table_bytes.len() as u64,
    };
    let table = PageTable::parse_verified(&record, table_bytes).unwrap();
    assert!(matches!(
        table.verify_object(&record, &object),
        Err(FormatError::PageHashMismatch { page_index: 0, .. })
    ));
}

#[test]
fn expected_id_limits_memory_and_missing_blobs_fail_closed() {
    let fixture = fixture();
    assert!(matches!(
        Manifest::parse_with_expected_id(
            &fixture.manifest,
            Limits::default(),
            Digest::of(b"wrong")
        ),
        Err(FormatError::ArtifactIdMismatch { .. })
    ));

    let mut wrong_id_without_blobs = fixture.parts();
    wrong_id_without_blobs.objects.clear();
    wrong_id_without_blobs.page_tables.clear();
    assert!(matches!(
        Artifact::from_bytes_with_expected_id(
            wrong_id_without_blobs,
            Limits::default(),
            Digest::of(b"wrong")
        ),
        Err(FormatError::ArtifactIdMismatch { .. })
    ));

    let mut no_object = fixture.parts();
    no_object.objects.clear();
    assert!(matches!(
        Artifact::from_bytes(no_object, Limits::default()),
        Err(FormatError::MissingBlob { kind: "object", .. })
    ));

    let mut no_table = fixture.parts();
    no_table.page_tables.clear();
    assert!(matches!(
        Artifact::from_bytes(no_table, Limits::default()),
        Err(FormatError::MissingBlob {
            kind: "page table",
            ..
        })
    ));

    let limits = Limits {
        eager_memory_bytes: 1,
        ..Limits::default()
    };
    assert!(matches!(
        Artifact::from_bytes(fixture.parts(), limits),
        Err(FormatError::MemoryBudgetExceeded { .. })
    ));

    let limits = Limits {
        objects: 0,
        ..Limits::default()
    };
    assert!(matches!(
        Manifest::parse_with_limits(&fixture.manifest, limits),
        Err(FormatError::CountLimit {
            field: "objects",
            ..
        })
    ));
}

#[test]
fn portable_open_reads_the_fixed_digest_layout_and_rechecks_lengths() {
    let fixture = fixture();
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("objects/sha256")).unwrap();
    fs::create_dir_all(root.join("page-tables/sha256")).unwrap();
    fs::write(root.join("manifest.json"), &fixture.manifest).unwrap();
    fs::write(
        root.join("objects/sha256")
            .join(fixture.object_digest.path_component()),
        &fixture.object,
    )
    .unwrap();
    fs::write(
        root.join("page-tables/sha256")
            .join(fixture.page_table_digest.path_component()),
        &fixture.page_table,
    )
    .unwrap();

    let artifact = Artifact::open(root, Limits::default()).unwrap();
    assert_eq!(artifact.load_tensor("weight").unwrap(), fixture.object);

    let object_path = root
        .join("objects/sha256")
        .join(fixture.object_digest.path_component());
    let mut extra = fixture.object;
    extra.push(0);
    fs::write(object_path, extra).unwrap();
    assert!(matches!(
        Artifact::open(root, Limits::default()),
        Err(FormatError::BlobLengthMismatch { kind: "object", .. })
    ));

    let replacement = vec![0_u8; 8];
    fs::write(
        root.join("objects/sha256")
            .join(fixture.object_digest.path_component()),
        replacement,
    )
    .unwrap();
    assert!(matches!(
        Artifact::open(root, Limits::default()),
        Err(FormatError::BlobDigestMismatch { kind: "object", .. })
    ));
}

#[cfg(unix)]
#[test]
fn portable_open_rejects_fifo_and_leaf_symlink_inputs() {
    use std::os::unix::fs::symlink;
    use std::process::Command;

    let directory = tempfile::tempdir().unwrap();
    let fifo_root = directory.path().join("fifo");
    fs::create_dir(&fifo_root).unwrap();
    let fifo = fifo_root.join("manifest.json");
    let status = Command::new("mkfifo").arg(&fifo).status().unwrap();
    assert!(status.success());
    assert!(matches!(
        Artifact::open(&fifo_root, Limits::default()),
        Err(FormatError::NotRegularFile { kind: "manifest" })
    ));

    let symlink_root = directory.path().join("symlink");
    fs::create_dir(&symlink_root).unwrap();
    let target = symlink_root.join("manifest-target.json");
    fs::write(&target, fixture().manifest).unwrap();
    symlink(&target, symlink_root.join("manifest.json")).unwrap();
    assert!(matches!(
        Artifact::open(&symlink_root, Limits::default()),
        Err(FormatError::Io {
            operation: "opening artifact file",
            ..
        })
    ));
}
