//! Deterministic synthetic model data used by tests and demos.
//!
//! The recipe is source data, not a trained checkpoint. Every value is an
//! exactly representable binary fraction derived from its tensor ID and flat
//! row-major index.

use std::{
    fmt::Write as _,
    fs::{self, OpenOptions},
    io::Write as _,
    mem::size_of,
    path::{Path, PathBuf},
};

use runnel_format::{ArtifactBytes, Digest};
use runnel_kernels::Bf16;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const VOCAB_SIZE: usize = 32;
pub const HIDDEN_SIZE: usize = 8;
pub const NUM_HEADS: usize = 2;
pub const NUM_LAYERS: usize = 1;
pub const NUM_EXPERTS: usize = 4;
pub const TOP_K: usize = 2;
pub const EXPERT_HIDDEN_SIZE: usize = 12;
pub const CONTEXT_LENGTH: usize = 16;
pub const V3_CONTEXT_LENGTH: usize = 1_024;
pub const RMS_EPSILON: f32 = 1.0 / 4096.0;
pub const ATTENTION_SCALE: f32 = 0.5;
pub const PAGE_SIZE: u32 = 65_536;
pub const MULTI_PAGE_OBJECT_LENGTH: usize = 2 * PAGE_SIZE as usize + 17;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TinySpec {
    pub adapter_id: String,
    pub adapter_version: u64,
    pub attention_scale_binary: String,
    pub context_length: usize,
    pub expert_hidden_size: usize,
    pub formula: String,
    pub hidden_size: usize,
    pub max_new_tokens: usize,
    pub num_experts: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub prompt: String,
    pub rms_epsilon_binary: String,
    pub tokenizer_id: String,
    pub tokenizer_version: u64,
    pub top_k: usize,
    pub vocab_size: usize,
}

impl Default for TinySpec {
    fn default() -> Self {
        Self {
            adapter_id: "runnel.tiny-causal-moe".into(),
            adapter_version: 1,
            attention_scale_binary: "2^-1".into(),
            context_length: CONTEXT_LENGTH,
            expert_hidden_size: EXPERT_HIDDEN_SIZE,
            formula: "q=((37*(tensor_id+1)+17*(flat_index+1)) mod 29)-14; norm=1+q/64; other=q/32"
                .into(),
            hidden_size: HIDDEN_SIZE,
            max_new_tokens: 4,
            num_experts: NUM_EXPERTS,
            num_heads: NUM_HEADS,
            num_layers: NUM_LAYERS,
            prompt: "moe".into(),
            rms_epsilon_binary: "2^-12".into(),
            tokenizer_id: "runnel.ascii32".into(),
            tokenizer_version: 1,
            top_k: TOP_K,
            vocab_size: VOCAB_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRecipe {
    pub id: u64,
    pub role: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TinyAdapterVersion {
    V1,
    V2,
    V3,
}

impl TinyAdapterVersion {
    const fn number(self) -> u64 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }

    const fn tensor_dtype(self, tensor_id: u64) -> FixtureDType {
        if matches!(self, Self::V2 | Self::V3) && matches!(tensor_id, 8..=19) {
            FixtureDType::Bf16
        } else {
            FixtureDType::F32
        }
    }

    const fn context_length(self) -> usize {
        match self {
            Self::V1 | Self::V2 => CONTEXT_LENGTH,
            Self::V3 => V3_CONTEXT_LENGTH,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixtureDType {
    F32,
    Bf16,
}

impl FixtureDType {
    const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32-le",
            Self::Bf16 => "bf16-le",
        }
    }
}

#[derive(Debug, Clone)]
struct EncodedTensor {
    recipe: TensorRecipe,
    dtype: FixtureDType,
    offset: usize,
    length: usize,
}

/// All bytes needed to materialize the deterministic tiny RMOA artifact.
///
/// The artifact is kept as generated source data rather than committed model
/// weights. Its identity is stable because every byte is derived from the
/// frozen formula and canonical manifest contract.
#[derive(Debug, Clone)]
pub struct FixtureArtifact {
    manifest: Vec<u8>,
    object: Vec<u8>,
    object_digest: Digest,
    page_table: Vec<u8>,
    page_table_digest: Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixtureIdentity {
    pub artifact_id: Digest,
    pub object_digest: Digest,
    pub object_length: u64,
    pub page_table_digest: Digest,
    pub page_table_length: u64,
}

/// Deterministic three-page RMOA used to exercise storage behavior that the
/// one-page numerical fixture cannot reach. The final page is exactly 17
/// bytes, and no generated payload is committed to the repository.
#[derive(Debug, Clone)]
pub struct MultiPageFixture {
    manifest: Vec<u8>,
    object: Vec<u8>,
    object_digest: Digest,
    page_table: Vec<u8>,
    page_table_digest: Digest,
}

#[derive(Debug, Error)]
pub enum FixtureError {
    #[error("could not {operation}: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
}

impl FixtureArtifact {
    /// Builds the immutable all-f32 tiny-v1 fixture.
    #[must_use]
    pub fn build() -> Self {
        Self::build_for(TinyAdapterVersion::V1)
    }

    /// Builds tiny-v2, which stores only routed-expert matrices as BF16.
    #[must_use]
    pub fn build_v2() -> Self {
        Self::build_for(TinyAdapterVersion::V2)
    }

    /// Builds tiny-v3 with compact BF16 experts and a 1,024-token context cap.
    #[must_use]
    pub fn build_v3() -> Self {
        Self::build_for(TinyAdapterVersion::V3)
    }

    fn build_for(adapter_version: TinyAdapterVersion) -> Self {
        let recipes = tensor_recipes();
        let mut object = Vec::new();
        let mut descriptors = Vec::with_capacity(recipes.len());
        for recipe in recipes {
            let dtype = adapter_version.tensor_dtype(recipe.id);
            let offset = object.len();
            let bytes = recipe.encoded_bytes(dtype);
            object.extend_from_slice(&bytes);
            descriptors.push(EncodedTensor {
                recipe,
                dtype,
                offset,
                length: bytes.len(),
            });
        }

        let object_digest = Digest::of(&object);
        let page_table = build_page_table(&object, object_digest);
        let page_table_digest = Digest::of(&page_table);
        let manifest = build_manifest(
            &descriptors,
            object_digest,
            object.len(),
            page_table_digest,
            page_table.len(),
            adapter_version.number(),
            adapter_version.context_length(),
        );

        Self {
            manifest,
            object,
            object_digest,
            page_table,
            page_table_digest,
        }
    }

    #[must_use]
    pub fn identity(&self) -> FixtureIdentity {
        FixtureIdentity {
            artifact_id: Digest::of(&self.manifest),
            object_digest: self.object_digest,
            object_length: self.object.len() as u64,
            page_table_digest: self.page_table_digest,
            page_table_length: self.page_table.len() as u64,
        }
    }

    #[must_use]
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest
    }

    #[must_use]
    pub fn to_parts(&self) -> ArtifactBytes {
        ArtifactBytes {
            manifest: self.manifest.clone(),
            objects: [(self.object_digest, self.object.clone())].into(),
            page_tables: [(self.page_table_digest, self.page_table.clone())].into(),
        }
    }

    /// Materialize the fixture at a new directory, publishing the manifest
    /// last. The output root and every created leaf must be absent.
    ///
    /// This deterministic test helper is not the transactional M2 ingestion
    /// path and assumes its newly created root is not concurrently replaced.
    pub fn write_new(&self, root: impl AsRef<Path>) -> Result<FixtureIdentity, FixtureError> {
        write_artifact_new(
            root.as_ref(),
            &self.manifest,
            self.object_digest,
            &self.object,
            self.page_table_digest,
            &self.page_table,
        )?;
        Ok(self.identity())
    }
}

impl MultiPageFixture {
    #[must_use]
    pub fn build() -> Self {
        let object = (0..MULTI_PAGE_OBJECT_LENGTH)
            .map(|index| ((index as u64 * 131 + 17) % 251) as u8)
            .collect::<Vec<_>>();
        let object_digest = Digest::of(&object);
        let page_table = build_page_table(&object, object_digest);
        let page_table_digest = Digest::of(&page_table);
        let manifest = build_storage_manifest(
            object_digest,
            object.len(),
            page_table_digest,
            page_table.len(),
        );
        Self {
            manifest,
            object,
            object_digest,
            page_table,
            page_table_digest,
        }
    }

    #[must_use]
    pub fn identity(&self) -> FixtureIdentity {
        FixtureIdentity {
            artifact_id: Digest::of(&self.manifest),
            object_digest: self.object_digest,
            object_length: self.object.len() as u64,
            page_table_digest: self.page_table_digest,
            page_table_length: self.page_table.len() as u64,
        }
    }

    #[must_use]
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest
    }

    #[must_use]
    pub fn object_bytes(&self) -> &[u8] {
        &self.object
    }

    #[must_use]
    pub fn page_table_bytes(&self) -> &[u8] {
        &self.page_table
    }

    #[must_use]
    pub fn to_parts(&self) -> ArtifactBytes {
        ArtifactBytes {
            manifest: self.manifest.clone(),
            objects: [(self.object_digest, self.object.clone())].into(),
            page_tables: [(self.page_table_digest, self.page_table.clone())].into(),
        }
    }

    /// Materializes the source fixture for descriptor-safe import/read tests.
    pub fn write_new(&self, root: impl AsRef<Path>) -> Result<FixtureIdentity, FixtureError> {
        write_artifact_new(
            root.as_ref(),
            &self.manifest,
            self.object_digest,
            &self.object,
            self.page_table_digest,
            &self.page_table,
        )?;
        Ok(self.identity())
    }
}

impl Default for MultiPageFixture {
    fn default() -> Self {
        Self::build()
    }
}

impl Default for FixtureArtifact {
    fn default() -> Self {
        Self::build()
    }
}

impl TensorRecipe {
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    #[must_use]
    pub fn values(&self) -> Vec<f32> {
        (0..self.element_count())
            .map(|index| value(self.id, index))
            .collect()
    }

    #[must_use]
    pub fn little_endian_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.element_count() * size_of::<f32>());
        for item in self.values() {
            bytes.extend_from_slice(&item.to_le_bytes());
        }
        bytes
    }

    fn encoded_bytes(&self, dtype: FixtureDType) -> Vec<u8> {
        match dtype {
            FixtureDType::F32 => self.little_endian_bytes(),
            FixtureDType::Bf16 => {
                let mut bytes = Vec::with_capacity(self.element_count() * size_of::<u16>());
                for item in self.values() {
                    let word = Bf16::try_from_f32_rne(item)
                        .expect("the frozen finite fixture formula must fit BF16");
                    bytes.extend_from_slice(&word.to_bits().to_le_bytes());
                }
                bytes
            }
        }
    }
}

#[must_use]
pub fn tensor_recipes() -> Vec<TensorRecipe> {
    let mut tensors = vec![
        recipe(0, "token_embedding", &[VOCAB_SIZE, HIDDEN_SIZE]),
        recipe(1, "layers.0.attn_norm", &[HIDDEN_SIZE]),
        recipe(2, "layers.0.attn_q", &[HIDDEN_SIZE, HIDDEN_SIZE]),
        recipe(3, "layers.0.attn_k", &[HIDDEN_SIZE, HIDDEN_SIZE]),
        recipe(4, "layers.0.attn_v", &[HIDDEN_SIZE, HIDDEN_SIZE]),
        recipe(5, "layers.0.attn_out", &[HIDDEN_SIZE, HIDDEN_SIZE]),
        recipe(6, "layers.0.ffn_norm", &[HIDDEN_SIZE]),
        recipe(7, "layers.0.router", &[NUM_EXPERTS, HIDDEN_SIZE]),
    ];

    for expert in 0..NUM_EXPERTS {
        let base = 8 + expert as u64 * 3;
        tensors.push(recipe(
            base,
            &format!("layers.0.experts.{expert}.gate"),
            &[EXPERT_HIDDEN_SIZE, HIDDEN_SIZE],
        ));
        tensors.push(recipe(
            base + 1,
            &format!("layers.0.experts.{expert}.up"),
            &[EXPERT_HIDDEN_SIZE, HIDDEN_SIZE],
        ));
        tensors.push(recipe(
            base + 2,
            &format!("layers.0.experts.{expert}.down"),
            &[HIDDEN_SIZE, EXPERT_HIDDEN_SIZE],
        ));
    }

    tensors.push(recipe(20, "final_norm", &[HIDDEN_SIZE]));
    tensors.push(recipe(21, "lm_head", &[VOCAB_SIZE, HIDDEN_SIZE]));
    tensors
}

#[must_use]
pub fn value(tensor_id: u64, flat_index: usize) -> f32 {
    let centered = ((37 * (tensor_id + 1) + 17 * (flat_index as u64 + 1)) % 29) as i32 - 14;
    if matches!(tensor_id, 1 | 6 | 20) {
        1.0 + centered as f32 / 64.0
    } else {
        centered as f32 / 32.0
    }
}

fn recipe(id: u64, role: &str, shape: &[usize]) -> TensorRecipe {
    TensorRecipe {
        id,
        role: role.into(),
        shape: shape.to_vec(),
    }
}

fn build_page_table(object: &[u8], object_digest: Digest) -> Vec<u8> {
    let page_size = PAGE_SIZE as usize;
    let page_count = object.len().div_ceil(page_size);
    let mut bytes = Vec::with_capacity(64 + page_count * 32);
    bytes.extend_from_slice(b"RMOAPG1\n");
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&PAGE_SIZE.to_le_bytes());
    bytes.extend_from_slice(&(object.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(page_count as u64).to_le_bytes());
    bytes.extend_from_slice(object_digest.as_bytes());
    for page in object.chunks(page_size) {
        bytes.extend_from_slice(Digest::of(page).as_bytes());
    }
    bytes
}

fn build_manifest(
    tensors: &[EncodedTensor],
    object_digest: Digest,
    object_length: usize,
    page_table_digest: Digest,
    page_table_length: usize,
    adapter_version: u64,
    context_length: usize,
) -> Vec<u8> {
    let mut tensor_json = String::new();
    for (index, tensor) in tensors.iter().enumerate() {
        if index != 0 {
            tensor_json.push(',');
        }
        let shape = tensor
            .recipe
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        write!(
            tensor_json,
            "{{\"dtype\":\"{}\",\"id\":{},\"length\":{},\"object\":\"{object_digest}\",\"offset\":{},\"role\":\"{}\",\"shape\":[{shape}]}}",
            tensor.dtype.as_str(),
            tensor.recipe.id,
            tensor.length,
            tensor.offset,
            tensor.recipe.role,
        )
        .expect("writing to String cannot fail");
    }

    format!(
        concat!(
            "{{",
            "\"adapter\":{{\"id\":\"runnel.tiny-causal-moe\",\"version\":{}}},",
            "\"format\":\"rmoa\",",
            "\"model\":{{",
            "\"context_length\":{},",
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
        adapter_version,
        context_length,
        object_digest,
        object_length,
        page_table_digest,
        page_table_length,
        tensor_json,
    )
    .into_bytes()
}

fn build_storage_manifest(
    object_digest: Digest,
    object_length: usize,
    page_table_digest: Digest,
    page_table_length: usize,
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
            "\"tensors\":[{{",
            "\"dtype\":\"u8\",",
            "\"id\":0,",
            "\"length\":{},",
            "\"object\":\"{}\",",
            "\"offset\":0,",
            "\"role\":\"storage.payload\",",
            "\"shape\":[{}]",
            "}}],",
            "\"tokenizer\":{{\"id\":\"runnel.ascii32\",\"version\":1,\"vocab_size\":32}},",
            "\"version\":1",
            "}}\n"
        ),
        object_digest,
        object_length,
        page_table_digest,
        page_table_length,
        object_length,
        object_digest,
        object_length,
    )
    .into_bytes()
}

fn write_artifact_new(
    root: &Path,
    manifest: &[u8],
    object_digest: Digest,
    object: &[u8],
    page_table_digest: Digest,
    page_table: &[u8],
) -> Result<(), FixtureError> {
    fs::create_dir(root).map_err(|source| io("create the artifact directory", source))?;
    let object_parent = root.join("objects");
    let page_table_parent = root.join("page-tables");
    fs::create_dir(&object_parent)
        .map_err(|source| io("create the object parent directory", source))?;
    fs::create_dir(&page_table_parent)
        .map_err(|source| io("create the page-table parent directory", source))?;
    let objects = object_parent.join("sha256");
    let page_tables = page_table_parent.join("sha256");
    fs::create_dir(&objects).map_err(|source| io("create the object directory", source))?;
    fs::create_dir(&page_tables).map_err(|source| io("create the page-table directory", source))?;

    write_file(
        page_tables.join(page_table_digest.path_component()),
        page_table,
        "write the page table",
    )?;
    write_file(
        objects.join(object_digest.path_component()),
        object,
        "write the tensor object",
    )?;
    write_file(root.join("manifest.json"), manifest, "write the manifest")
}

fn write_file(path: PathBuf, bytes: &[u8], operation: &'static str) -> Result<(), FixtureError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io(operation, source))?;
    file.write_all(bytes)
        .map_err(|source| io(operation, source))
}

fn io(operation: &'static str, source: std::io::Error) -> FixtureError {
    FixtureError::Io { operation, source }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runnel_format::{Artifact, DType, Limits, Manifest};

    #[test]
    fn recipe_is_contiguous_and_complete() {
        let tensors = tensor_recipes();
        assert_eq!(tensors.len(), 22);
        for (expected, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.id, expected as u64);
            assert!(tensor.element_count() > 0);
        }
    }

    #[test]
    fn formula_is_finite_and_exactly_bounded() {
        for tensor in tensor_recipes() {
            for value in tensor.values() {
                assert!(value.is_finite());
                if matches!(tensor.id, 1 | 6 | 20) {
                    assert!((0.75..=1.25).contains(&value));
                } else {
                    assert!((-0.5..=0.5).contains(&value));
                }
            }
        }
    }

    #[test]
    fn tiny_v1_identity_and_all_f32_contract_are_frozen() {
        let fixture = FixtureArtifact::build();
        let manifest = Manifest::parse(fixture.manifest_bytes()).unwrap();
        assert_eq!(manifest.artifact_id(), fixture.identity().artifact_id);
        assert_eq!(manifest.adapter.version, 1);
        assert_eq!(manifest.tensors.len(), 22);
        assert!(
            manifest
                .tensors
                .iter()
                .all(|tensor| tensor.dtype == DType::F32Le)
        );

        let identity = fixture.identity();
        assert_eq!(fixture.manifest_bytes().len(), 4_488);
        assert_eq!(identity.object_length, 7_904);
        assert_eq!(identity.page_table_length, 96);
        assert_eq!(
            identity.artifact_id.to_string(),
            "sha256:e49321cefc980ab59cd449341edfc624ccfc4b0b703cd175184e056d296d9ed3"
        );
        assert_eq!(
            identity.object_digest.to_string(),
            "sha256:6b2b8a1bbb2854084b1e1fe1e5787a9cfdb021b397e774fc7dbef79ac9d24bf6"
        );
        assert_eq!(
            identity.page_table_digest.to_string(),
            "sha256:29383b56a150f9e5705f3666ca7707f21bc3fbd663ffd7249f4ecb938da6a62d"
        );

        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        assert_eq!(artifact.artifact_id(), fixture.identity().artifact_id);
        assert_eq!(
            artifact.tensor_by_role("lm_head").unwrap().bytes.len(),
            32 * 8 * 4
        );
    }

    #[test]
    fn tiny_v2_pins_mixed_dtype_layout_and_exact_rne_bytes() {
        let version_one =
            Artifact::from_bytes(FixtureArtifact::build().to_parts(), Limits::default()).unwrap();
        let fixture = FixtureArtifact::build_v2();
        let identity = fixture.identity();

        assert_eq!(fixture.manifest_bytes().len(), 4_500);
        assert_eq!(identity.object_length, 5_600);
        assert_eq!(identity.page_table_length, 96);
        assert_eq!(
            identity.artifact_id.to_string(),
            "sha256:606baa0c1082b369632b5dd000d30dc51ae20321b2c032ef3395aaa0bfd7c76c"
        );
        assert_eq!(
            identity.object_digest.to_string(),
            "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab"
        );
        assert_eq!(
            identity.page_table_digest.to_string(),
            "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c"
        );

        let version_two = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        assert_eq!(version_two.artifact_id(), identity.artifact_id);
        assert_eq!(version_two.manifest().adapter.version, 2);
        assert_eq!(version_two.manifest().tensors.len(), 22);

        let mut f32_tensors = 0;
        let mut bf16_tensors = 0;
        for (version_one_descriptor, version_two_descriptor) in version_one
            .manifest()
            .tensors
            .iter()
            .zip(&version_two.manifest().tensors)
        {
            assert_eq!(version_two_descriptor.id, version_one_descriptor.id);
            assert_eq!(version_two_descriptor.role, version_one_descriptor.role);
            assert_eq!(version_two_descriptor.shape, version_one_descriptor.shape);

            let version_one_bytes = version_one
                .tensor_bytes(version_one_descriptor.id)
                .expect("v1 tensor bytes are verified");
            let version_two_bytes = version_two
                .tensor_bytes(version_two_descriptor.id)
                .expect("v2 tensor bytes are verified");

            if matches!(version_two_descriptor.id, 8..=19) {
                bf16_tensors += 1;
                assert_eq!(version_two_descriptor.dtype, DType::Bf16Le);
                assert_eq!(version_two_descriptor.length, 192);
                assert_eq!(
                    version_two_descriptor.offset,
                    2_240 + 192 * (version_two_descriptor.id - 8)
                );
                assert_eq!(version_two_bytes.len() * 2, version_one_bytes.len());

                for (version_one_word, version_two_word) in version_one_bytes
                    .chunks_exact(size_of::<f32>())
                    .zip(version_two_bytes.chunks_exact(size_of::<u16>()))
                {
                    let source = f32::from_le_bytes(version_one_word.try_into().unwrap());
                    let actual = u16::from_le_bytes(version_two_word.try_into().unwrap());
                    let expected = Bf16::try_from_f32_rne(source).unwrap();
                    assert_eq!(actual, expected.to_bits());
                    assert_eq!(expected.to_f32().to_bits(), source.to_bits());
                }
            } else {
                f32_tensors += 1;
                assert_eq!(version_two_descriptor.dtype, DType::F32Le);
                assert_eq!(version_two_descriptor.length, version_one_descriptor.length);
                assert_eq!(version_two_bytes, version_one_bytes);
                if version_two_descriptor.id <= 7 {
                    assert_eq!(version_two_descriptor.offset, version_one_descriptor.offset);
                }
            }
        }

        assert_eq!(bf16_tensors, 12);
        assert_eq!(f32_tensors, 10);
        assert_eq!(version_two.manifest().tensors[20].offset, 4_544);
        assert_eq!(version_two.manifest().tensors[21].offset, 4_576);
        assert_eq!(version_two.manifest().tensors[21].length, 1_024);
    }

    #[test]
    fn tiny_v3_changes_only_adapter_identity_and_context_cap() {
        let version_two_fixture = FixtureArtifact::build_v2();
        let fixture = FixtureArtifact::build_v3();
        let identity = fixture.identity();

        assert_eq!(fixture.manifest_bytes().len(), 4_502);
        assert_eq!(identity.object_length, 5_600);
        assert_eq!(identity.page_table_length, 96);
        assert_eq!(
            identity.artifact_id.to_string(),
            "sha256:382856e13f688b5176ad1e5f06c26bcd85bcaeb719a10a60e9adc9ff387c945c"
        );
        assert_eq!(
            identity.object_digest.to_string(),
            "sha256:275f985b05a85d4f85d78fc290c10c9d39e9169c46449f4d3a3ed6513a8965ab"
        );
        assert_eq!(
            identity.page_table_digest.to_string(),
            "sha256:7d660764b861f97afbc800efb361bdd59ac38fdea5fc424c62f9fb6a30b2896c"
        );

        assert_eq!(version_two_fixture.object, fixture.object);
        assert_eq!(version_two_fixture.page_table, fixture.page_table);

        let version_two =
            Artifact::from_bytes(version_two_fixture.to_parts(), Limits::default()).unwrap();
        let version_three = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        assert_eq!(version_three.artifact_id(), identity.artifact_id);
        assert_eq!(version_three.manifest().adapter.version, 3);
        assert_eq!(
            version_three.manifest().model.context_length,
            V3_CONTEXT_LENGTH as u64
        );
        for (version_two_tensor, version_three_tensor) in version_two
            .manifest()
            .tensors
            .iter()
            .zip(&version_three.manifest().tensors)
        {
            assert_eq!(version_three_tensor.id, version_two_tensor.id);
            assert_eq!(version_three_tensor.role, version_two_tensor.role);
            assert_eq!(version_three_tensor.shape, version_two_tensor.shape);
            assert_eq!(version_three_tensor.dtype, version_two_tensor.dtype);
            assert_eq!(version_three_tensor.offset, version_two_tensor.offset);
            assert_eq!(version_three_tensor.length, version_two_tensor.length);
        }
    }

    #[test]
    fn filesystem_generation_never_overwrites() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("tiny-rmoa");
        let fixture = FixtureArtifact::build();
        fixture.write_new(&root).unwrap();
        let artifact = Artifact::open(&root, Limits::default()).unwrap();
        assert_eq!(artifact.artifact_id(), fixture.identity().artifact_id);
        assert!(fixture.write_new(&root).is_err());
    }

    #[test]
    fn multi_page_fixture_has_two_full_pages_and_a_short_tail() {
        let fixture = MultiPageFixture::build();
        assert_eq!(fixture.object_bytes().len(), MULTI_PAGE_OBJECT_LENGTH);
        assert_eq!(fixture.object_bytes().chunks(PAGE_SIZE as usize).count(), 3);
        assert_eq!(
            fixture
                .object_bytes()
                .chunks(PAGE_SIZE as usize)
                .last()
                .unwrap()
                .len(),
            17
        );
        assert_eq!(fixture.page_table_bytes().len(), 64 + 3 * 32);
        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        assert_eq!(artifact.artifact_id(), fixture.identity().artifact_id);
        assert_eq!(
            artifact.tensor_by_role("storage.payload").unwrap().bytes,
            fixture.object_bytes()
        );
    }
}
