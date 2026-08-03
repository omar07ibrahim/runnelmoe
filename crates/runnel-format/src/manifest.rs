use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use sha2::{Digest as _, Sha256};
use thiserror::Error;

use crate::error::FormatError;
use crate::json::{self, JsonValue};

pub const FORMAT_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
pub const FORMAT_OBJECTS: usize = 65_536;
pub const FORMAT_TENSORS: usize = 65_536;
pub const FORMAT_OBJECT_BYTES: u64 = 1 << 40;
pub const FORMAT_AGGREGATE_OBJECT_BYTES: u64 = 8 << 40;
pub const FORMAT_PAGE_TABLE_BYTES: u64 = 64 * 1024 * 1024;
pub const FORMAT_AGGREGATE_PAGE_TABLE_BYTES: u64 = 1024 * 1024 * 1024;

const MIN_PAGE_SIZE: u32 = 65_536;
const MAX_PAGE_SIZE: u32 = 2_097_152;
const MAX_TENSOR_DIMENSION: u64 = 2_147_483_647;

/// Operator limits for parsing M1 artifacts. Defaults are intentionally below
/// the interoperability ceilings in the format contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub manifest_bytes: usize,
    pub objects: usize,
    pub tensors: usize,
    pub object_bytes: u64,
    pub aggregate_object_bytes: u64,
    pub page_table_bytes: u64,
    pub aggregate_page_table_bytes: u64,
    /// Independent M1 budget for eagerly retained manifest, object, and page-table bytes.
    pub eager_memory_bytes: u64,
}

impl Limits {
    pub fn validate(self) -> Result<Self, FormatError> {
        check_limit(
            "manifest_bytes",
            self.manifest_bytes as u64,
            FORMAT_MANIFEST_BYTES as u64,
        )?;
        check_limit("objects", self.objects as u64, FORMAT_OBJECTS as u64)?;
        check_limit("tensors", self.tensors as u64, FORMAT_TENSORS as u64)?;
        check_limit("object_bytes", self.object_bytes, FORMAT_OBJECT_BYTES)?;
        check_limit(
            "aggregate_object_bytes",
            self.aggregate_object_bytes,
            FORMAT_AGGREGATE_OBJECT_BYTES,
        )?;
        check_limit(
            "page_table_bytes",
            self.page_table_bytes,
            FORMAT_PAGE_TABLE_BYTES,
        )?;
        check_limit(
            "aggregate_page_table_bytes",
            self.aggregate_page_table_bytes,
            FORMAT_AGGREGATE_PAGE_TABLE_BYTES,
        )?;
        Ok(self)
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            manifest_bytes: 1024 * 1024,
            objects: 4_096,
            tensors: 16_384,
            object_bytes: 2 * 1024 * 1024 * 1024,
            aggregate_object_bytes: 2 * 1024 * 1024 * 1024,
            page_table_bytes: 8 * 1024 * 1024,
            aggregate_page_table_bytes: 64 * 1024 * 1024,
            eager_memory_bytes: 256 * 1024 * 1024,
        }
    }
}

fn check_limit(field: &'static str, value: u64, ceiling: u64) -> Result<(), FormatError> {
    if value > ceiling {
        return Err(FormatError::LimitExceedsFormat {
            field,
            value,
            ceiling,
        });
    }
    Ok(())
}

/// A SHA-256 content identifier. Its textual representation is
/// `sha256:` followed by 64 lowercase hexadecimal digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        Self(digest)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    #[must_use]
    pub fn path_component(&self) -> String {
        let mut output = String::with_capacity(64);
        write_hex(&self.0, &mut output);
        output
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("sha256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for Digest {
    type Err = DigestParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let hexadecimal = value
            .strip_prefix("sha256:")
            .ok_or(DigestParseError::Prefix)?;
        if hexadecimal.len() != 64 {
            return Err(DigestParseError::Length);
        }

        let mut bytes = [0_u8; 32];
        for (index, pair) in hexadecimal.as_bytes().chunks_exact(2).enumerate() {
            let high = lowercase_hex_value(pair[0]).ok_or(DigestParseError::LowercaseHex)?;
            let low = lowercase_hex_value(pair[1]).ok_or(DigestParseError::LowercaseHex)?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DigestParseError {
    #[error("digest must begin with sha256:")]
    Prefix,
    #[error("SHA-256 digest must contain exactly 64 hexadecimal digits")]
    Length,
    #[error("SHA-256 digest must use lowercase hexadecimal digits")]
    LowercaseHex,
}

fn lowercase_hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn write_hex(bytes: &[u8], output: &mut String) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Adapter {
    pub id: String,
    pub version: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tokenizer {
    pub id: String,
    pub version: u64,
    pub vocab_size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TinyModel {
    pub context_length: u64,
    pub expert_hidden_size: u64,
    pub hidden_size: u64,
    pub num_experts: u64,
    pub num_heads: u64,
    pub num_layers: u64,
    pub top_k: u64,
    pub vocab_size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectRecord {
    pub digest: Digest,
    pub length: u64,
    pub page_size: u32,
    pub page_table: Digest,
    pub page_table_length: u64,
}

impl ObjectRecord {
    pub fn page_count(&self) -> Result<u64, FormatError> {
        page_count(self.length, self.page_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32Le,
    Bf16Le,
    U8,
    I8,
}

impl DType {
    #[must_use]
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::F32Le => 4,
            Self::Bf16Le => 2,
            Self::U8 | Self::I8 => 1,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32Le => "f32-le",
            Self::Bf16Le => "bf16-le",
            Self::U8 => "u8",
            Self::I8 => "i8",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRecord {
    pub dtype: DType,
    pub id: u64,
    pub length: u64,
    pub object: Digest,
    pub offset: u64,
    pub role: String,
    pub shape: Vec<u32>,
}

/// A schema-checked, canonical RMOA v1 manifest for the M1 tiny adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub adapter: Adapter,
    pub model: TinyModel,
    pub objects: Vec<ObjectRecord>,
    pub tensors: Vec<TensorRecord>,
    pub tokenizer: Tokenizer,
    artifact_id: Digest,
    canonical_bytes: Box<[u8]>,
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        Self::parse_with_limits(bytes, Limits::default())
    }

    pub fn parse_with_limits(bytes: &[u8], limits: Limits) -> Result<Self, FormatError> {
        let limits = limits.validate()?;
        if bytes.len() > limits.manifest_bytes {
            return Err(FormatError::ManifestTooLarge {
                actual: bytes.len(),
                limit: limits.manifest_bytes,
            });
        }

        let document = json::parse(bytes)?;
        let mut manifest = validate_schema(&document, limits)?;
        if json::canonical_bytes(&document) != bytes {
            return Err(FormatError::NonCanonicalManifest);
        }
        manifest.artifact_id = Digest::of(bytes);
        manifest.canonical_bytes = bytes.to_vec().into_boxed_slice();
        Ok(manifest)
    }

    pub fn parse_with_expected_id(
        bytes: &[u8],
        limits: Limits,
        expected: Digest,
    ) -> Result<Self, FormatError> {
        let manifest = Self::parse_with_limits(bytes, limits)?;
        manifest.require_artifact_id(expected)?;
        Ok(manifest)
    }

    #[must_use]
    pub const fn artifact_id(&self) -> Digest {
        self.artifact_id
    }

    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub fn require_artifact_id(&self, expected: Digest) -> Result<(), FormatError> {
        if self.artifact_id != expected {
            return Err(FormatError::ArtifactIdMismatch {
                expected,
                actual: self.artifact_id,
            });
        }
        Ok(())
    }

    #[must_use]
    pub fn object(&self, digest: Digest) -> Option<&ObjectRecord> {
        self.objects
            .binary_search_by_key(&digest, |record| record.digest)
            .ok()
            .map(|index| &self.objects[index])
    }

    #[must_use]
    pub fn tensor_by_role(&self, role: &str) -> Option<&TensorRecord> {
        self.tensors.iter().find(|tensor| tensor.role == role)
    }
}

fn validate_schema(document: &JsonValue, limits: Limits) -> Result<Manifest, FormatError> {
    let root = expect_object(document, "$")?;
    exact_keys(
        root,
        &[
            "adapter",
            "format",
            "model",
            "objects",
            "tensors",
            "tokenizer",
            "version",
        ],
        "$",
    )?;

    expect_literal_string(required(root, "format", "$")?, "rmoa", "$.format")?;
    expect_literal_integer(required(root, "version", "$")?, 1, "$.version")?;
    let adapter = parse_adapter(required(root, "adapter", "$")?)?;
    let model = parse_model(required(root, "model", "$")?)?;
    let tokenizer = parse_tokenizer(required(root, "tokenizer", "$")?)?;
    if tokenizer.vocab_size != model.vocab_size {
        return schema("$.tokenizer.vocab_size", "must equal $.model.vocab_size");
    }

    let objects = parse_objects(required(root, "objects", "$")?, limits)?;
    let tensors = parse_tensors(required(root, "tensors", "$")?, &objects, limits)?;

    Ok(Manifest {
        adapter,
        model,
        objects,
        tensors,
        tokenizer,
        artifact_id: Digest::from_bytes([0; 32]),
        canonical_bytes: Box::new([]),
    })
}

fn parse_adapter(value: &JsonValue) -> Result<Adapter, FormatError> {
    let object = expect_object(value, "$.adapter")?;
    exact_keys(object, &["id", "version"], "$.adapter")?;
    let id = expect_string(required(object, "id", "$.adapter")?, "$.adapter.id")?;
    validate_identifier(id, "$.adapter.id")?;
    if id != "runnel.tiny-causal-moe" {
        return schema("$.adapter.id", "unsupported M1 adapter ID");
    }
    let version = positive_integer(
        required(object, "version", "$.adapter")?,
        "$.adapter.version",
    )?;
    if version != 1 {
        return schema("$.adapter.version", "unsupported M1 adapter version");
    }
    Ok(Adapter {
        id: id.to_owned(),
        version,
    })
}

fn parse_tokenizer(value: &JsonValue) -> Result<Tokenizer, FormatError> {
    let object = expect_object(value, "$.tokenizer")?;
    exact_keys(object, &["id", "version", "vocab_size"], "$.tokenizer")?;
    let id = expect_string(required(object, "id", "$.tokenizer")?, "$.tokenizer.id")?;
    validate_identifier(id, "$.tokenizer.id")?;
    Ok(Tokenizer {
        id: id.to_owned(),
        version: positive_integer(
            required(object, "version", "$.tokenizer")?,
            "$.tokenizer.version",
        )?,
        vocab_size: positive_integer(
            required(object, "vocab_size", "$.tokenizer")?,
            "$.tokenizer.vocab_size",
        )?,
    })
}

fn parse_model(value: &JsonValue) -> Result<TinyModel, FormatError> {
    let object = expect_object(value, "$.model")?;
    exact_keys(
        object,
        &[
            "context_length",
            "expert_hidden_size",
            "hidden_size",
            "num_experts",
            "num_heads",
            "num_layers",
            "top_k",
            "vocab_size",
        ],
        "$.model",
    )?;

    let dimension =
        |key: &str| positive_integer(required(object, key, "$.model")?, &format!("$.model.{key}"));
    let model = TinyModel {
        context_length: dimension("context_length")?,
        expert_hidden_size: dimension("expert_hidden_size")?,
        hidden_size: dimension("hidden_size")?,
        num_experts: dimension("num_experts")?,
        num_heads: dimension("num_heads")?,
        num_layers: dimension("num_layers")?,
        top_k: dimension("top_k")?,
        vocab_size: dimension("vocab_size")?,
    };
    if model.top_k > model.num_experts {
        return schema("$.model.top_k", "must not exceed num_experts");
    }
    if !model.hidden_size.is_multiple_of(model.num_heads) {
        return schema("$.model.hidden_size", "must be divisible by num_heads");
    }
    Ok(model)
}

fn parse_objects(value: &JsonValue, limits: Limits) -> Result<Vec<ObjectRecord>, FormatError> {
    let values = expect_array(value, "$.objects")?;
    if values.len() > limits.objects {
        return Err(FormatError::CountLimit {
            field: "objects",
            actual: values.len(),
            limit: limits.objects,
        });
    }

    let mut records: Vec<ObjectRecord> = Vec::with_capacity(values.len());
    let mut page_tables = BTreeSet::new();
    let mut aggregate_object_bytes = 0_u64;
    let mut aggregate_page_table_bytes = 0_u64;

    for (index, value) in values.iter().enumerate() {
        let path = format!("$.objects[{index}]");
        let object = expect_object(value, &path)?;
        exact_keys(
            object,
            &[
                "digest",
                "length",
                "page_size",
                "page_table",
                "page_table_length",
            ],
            &path,
        )?;
        let digest = parse_digest(
            required(object, "digest", &path)?,
            &format!("{path}.digest"),
        )?;
        if let Some(previous) = records.last()
            && digest <= previous.digest
        {
            return schema(
                &format!("{path}.digest"),
                "object digests must be unique and strictly increasing",
            );
        }
        let length = positive_integer(
            required(object, "length", &path)?,
            &format!("{path}.length"),
        )?;
        if length > limits.object_bytes {
            return schema(
                &format!("{path}.length"),
                "exceeds the configured per-object byte limit",
            );
        }
        aggregate_object_bytes =
            aggregate_object_bytes
                .checked_add(length)
                .ok_or(FormatError::ArithmeticOverflow {
                    context: "aggregate object bytes",
                })?;
        if aggregate_object_bytes > limits.aggregate_object_bytes {
            return schema(
                "$.objects",
                "exceeds the configured aggregate object byte limit",
            );
        }

        let page_size_u64 = positive_integer(
            required(object, "page_size", &path)?,
            &format!("{path}.page_size"),
        )?;
        let page_size = u32::try_from(page_size_u64)
            .map_err(|_| schema_error(&format!("{path}.page_size"), "does not fit u32"))?;
        if !(MIN_PAGE_SIZE..=MAX_PAGE_SIZE).contains(&page_size) || !page_size.is_power_of_two() {
            return schema(
                &format!("{path}.page_size"),
                "must be a power of two from 65536 through 2097152",
            );
        }

        let page_table = parse_digest(
            required(object, "page_table", &path)?,
            &format!("{path}.page_table"),
        )?;
        if !page_tables.insert(page_table) {
            return schema(
                &format!("{path}.page_table"),
                "page-table digests must be unique",
            );
        }
        let page_table_length = positive_integer(
            required(object, "page_table_length", &path)?,
            &format!("{path}.page_table_length"),
        )?;
        let expected_table_length = expected_page_table_length(length, page_size)?;
        if page_table_length != expected_table_length {
            return schema(
                &format!("{path}.page_table_length"),
                "does not equal 64 + 32 * ceil(length / page_size)",
            );
        }
        if page_table_length > limits.page_table_bytes {
            return schema(
                &format!("{path}.page_table_length"),
                "exceeds the configured per-page-table byte limit",
            );
        }
        aggregate_page_table_bytes = aggregate_page_table_bytes
            .checked_add(page_table_length)
            .ok_or(FormatError::ArithmeticOverflow {
                context: "aggregate page-table bytes",
            })?;
        if aggregate_page_table_bytes > limits.aggregate_page_table_bytes {
            return schema(
                "$.objects",
                "exceeds the configured aggregate page-table byte limit",
            );
        }

        records.push(ObjectRecord {
            digest,
            length,
            page_size,
            page_table,
            page_table_length,
        });
    }
    Ok(records)
}

fn parse_tensors(
    value: &JsonValue,
    objects: &[ObjectRecord],
    limits: Limits,
) -> Result<Vec<TensorRecord>, FormatError> {
    let values = expect_array(value, "$.tensors")?;
    if values.len() > limits.tensors {
        return Err(FormatError::CountLimit {
            field: "tensors",
            actual: values.len(),
            limit: limits.tensors,
        });
    }

    let object_lengths: BTreeMap<_, _> = objects
        .iter()
        .map(|record| (record.digest, record.length))
        .collect();
    let mut ranges: BTreeMap<Digest, Vec<(u64, u64)>> = BTreeMap::new();
    let mut roles = BTreeSet::new();
    let mut tensors = Vec::with_capacity(values.len());

    for (index, value) in values.iter().enumerate() {
        let path = format!("$.tensors[{index}]");
        let object = expect_object(value, &path)?;
        exact_keys(
            object,
            &["dtype", "id", "length", "object", "offset", "role", "shape"],
            &path,
        )?;

        let id = expect_integer(required(object, "id", &path)?, &format!("{path}.id"))?;
        if id != index as u64 {
            return schema(
                &format!("{path}.id"),
                "tensor IDs must be contiguous from zero",
            );
        }
        let dtype = parse_dtype(required(object, "dtype", &path)?, &format!("{path}.dtype"))?;
        let length = positive_integer(
            required(object, "length", &path)?,
            &format!("{path}.length"),
        )?;
        let object_digest = parse_digest(
            required(object, "object", &path)?,
            &format!("{path}.object"),
        )?;
        let object_length = *object_lengths.get(&object_digest).ok_or_else(|| {
            schema_error(&format!("{path}.object"), "does not name a declared object")
        })?;
        let offset = expect_integer(
            required(object, "offset", &path)?,
            &format!("{path}.offset"),
        )?;
        if offset % dtype.byte_width() != 0 {
            return schema(
                &format!("{path}.offset"),
                "must be aligned to the dtype width",
            );
        }
        let role = expect_string(required(object, "role", &path)?, &format!("{path}.role"))?;
        validate_role(role, &format!("{path}.role"))?;
        if !roles.insert(role.to_owned()) {
            return schema(&format!("{path}.role"), "tensor roles must be unique");
        }

        let shape_values =
            expect_array(required(object, "shape", &path)?, &format!("{path}.shape"))?;
        if !(1..=8).contains(&shape_values.len()) {
            return schema(&format!("{path}.shape"), "rank must be from 1 through 8");
        }
        let mut shape = Vec::with_capacity(shape_values.len());
        let mut elements = 1_u64;
        for (dimension_index, dimension) in shape_values.iter().enumerate() {
            let dimension_path = format!("{path}.shape[{dimension_index}]");
            let dimension = positive_integer(dimension, &dimension_path)?;
            if dimension > MAX_TENSOR_DIMENSION {
                return schema(&dimension_path, "dimension exceeds 2147483647");
            }
            elements = elements
                .checked_mul(dimension)
                .ok_or(FormatError::ArithmeticOverflow {
                    context: "tensor shape product",
                })?;
            shape.push(u32::try_from(dimension).map_err(|_| {
                schema_error(
                    &dimension_path,
                    "dimension does not fit the M1 representation",
                )
            })?);
        }
        let expected_length =
            elements
                .checked_mul(dtype.byte_width())
                .ok_or(FormatError::ArithmeticOverflow {
                    context: "tensor logical byte length",
                })?;
        if length != expected_length {
            return schema(
                &format!("{path}.length"),
                "does not equal shape product times dtype width",
            );
        }
        let end = offset
            .checked_add(length)
            .ok_or(FormatError::ArithmeticOverflow {
                context: "tensor object range",
            })?;
        if end > object_length {
            return schema(&path, "tensor range exceeds its declared object");
        }
        ranges.entry(object_digest).or_default().push((offset, end));
        tensors.push(TensorRecord {
            dtype,
            id,
            length,
            object: object_digest,
            offset,
            role: role.to_owned(),
            shape,
        });
    }

    validate_tensor_coverage(objects, &mut ranges)?;
    Ok(tensors)
}

fn validate_tensor_coverage(
    objects: &[ObjectRecord],
    ranges: &mut BTreeMap<Digest, Vec<(u64, u64)>>,
) -> Result<(), FormatError> {
    for object in objects {
        let object_ranges = ranges.entry(object.digest).or_default();
        object_ranges.sort_unstable();
        let mut cursor = 0_u64;
        for &(offset, end) in object_ranges.iter() {
            if offset > cursor {
                return Err(FormatError::TensorGap {
                    object: object.digest,
                    expected_offset: cursor,
                    actual_offset: offset,
                });
            }
            if offset < cursor {
                return Err(FormatError::TensorOverlap {
                    object: object.digest,
                    previous_end: cursor,
                    next_offset: offset,
                });
            }
            cursor = end;
        }
        if cursor != object.length {
            return Err(FormatError::TensorCoverage {
                object: object.digest,
                covered: cursor,
                length: object.length,
            });
        }
    }
    Ok(())
}

fn page_count(object_length: u64, page_size: u32) -> Result<u64, FormatError> {
    let page_size = u64::from(page_size);
    let rounding = page_size
        .checked_sub(1)
        .ok_or(FormatError::ArithmeticOverflow {
            context: "nonzero page size",
        })?;
    object_length
        .checked_add(rounding)
        .map(|rounded| rounded / page_size)
        .ok_or(FormatError::ArithmeticOverflow {
            context: "page count",
        })
}

pub(crate) fn expected_page_table_length(
    object_length: u64,
    page_size: u32,
) -> Result<u64, FormatError> {
    page_count(object_length, page_size)?
        .checked_mul(32)
        .and_then(|hash_bytes| hash_bytes.checked_add(64))
        .ok_or(FormatError::ArithmeticOverflow {
            context: "page-table byte length",
        })
}

fn parse_dtype(value: &JsonValue, path: &str) -> Result<DType, FormatError> {
    match expect_string(value, path)? {
        "f32-le" => Ok(DType::F32Le),
        "bf16-le" => Ok(DType::Bf16Le),
        "u8" => Ok(DType::U8),
        "i8" => Ok(DType::I8),
        _ => schema(path, "unsupported dtype"),
    }
}

fn parse_digest(value: &JsonValue, path: &str) -> Result<Digest, FormatError> {
    expect_string(value, path)?
        .parse()
        .map_err(|error: DigestParseError| schema_error(path, &error.to_string()))
}

fn validate_identifier(value: &str, path: &str) -> Result<(), FormatError> {
    if value.is_empty() || value.len() > 64 {
        return schema(path, "identifier length must be from 1 through 64 bytes");
    }
    let mut bytes = value.bytes();
    if !matches!(bytes.next(), Some(b'a'..=b'z'))
        || !bytes.all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'.' | b'-'))
    {
        return schema(path, "identifier does not match [a-z][a-z0-9.-]{0,63}");
    }
    Ok(())
}

fn validate_role(value: &str, path: &str) -> Result<(), FormatError> {
    if value.is_empty() || value.len() > 128 {
        return schema(path, "role length must be from 1 through 128 bytes");
    }
    let mut bytes = value.bytes();
    if !matches!(bytes.next(), Some(b'a'..=b'z'))
        || !bytes.all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-'))
    {
        return schema(path, "role does not match [a-z][a-z0-9_.-]{0,127}");
    }
    Ok(())
}

fn required<'a>(
    object: &'a BTreeMap<String, JsonValue>,
    key: &str,
    path: &str,
) -> Result<&'a JsonValue, FormatError> {
    object
        .get(key)
        .ok_or_else(|| schema_error(path, &format!("missing required key {key:?}")))
}

fn exact_keys(
    object: &BTreeMap<String, JsonValue>,
    expected: &[&str],
    path: &str,
) -> Result<(), FormatError> {
    for key in object.keys() {
        if !expected.contains(&key.as_str()) {
            return schema(path, &format!("unknown key {key:?}"));
        }
    }
    for key in expected {
        if !object.contains_key(*key) {
            return schema(path, &format!("missing required key {key:?}"));
        }
    }
    Ok(())
}

fn expect_object<'a>(
    value: &'a JsonValue,
    path: &str,
) -> Result<&'a BTreeMap<String, JsonValue>, FormatError> {
    match value {
        JsonValue::Object(value) => Ok(value),
        _ => schema(path, "expected object"),
    }
}

fn expect_array<'a>(value: &'a JsonValue, path: &str) -> Result<&'a [JsonValue], FormatError> {
    match value {
        JsonValue::Array(value) => Ok(value),
        _ => schema(path, "expected array"),
    }
}

fn expect_string<'a>(value: &'a JsonValue, path: &str) -> Result<&'a str, FormatError> {
    match value {
        JsonValue::String(value) => Ok(value),
        _ => schema(path, "expected string"),
    }
}

fn expect_integer(value: &JsonValue, path: &str) -> Result<u64, FormatError> {
    match value {
        JsonValue::Integer(value) => Ok(*value),
        _ => schema(path, "expected unsigned integer"),
    }
}

fn positive_integer(value: &JsonValue, path: &str) -> Result<u64, FormatError> {
    let value = expect_integer(value, path)?;
    if value == 0 {
        return schema(path, "must be positive");
    }
    Ok(value)
}

fn expect_literal_string(value: &JsonValue, expected: &str, path: &str) -> Result<(), FormatError> {
    if expect_string(value, path)? != expected {
        return schema(path, &format!("must be {expected:?}"));
    }
    Ok(())
}

fn expect_literal_integer(value: &JsonValue, expected: u64, path: &str) -> Result<(), FormatError> {
    if expect_integer(value, path)? != expected {
        return schema(path, &format!("must be {expected}"));
    }
    Ok(())
}

fn schema<T>(path: &str, problem: &str) -> Result<T, FormatError> {
    Err(schema_error(path, problem))
}

fn schema_error(path: &str, problem: &str) -> FormatError {
    FormatError::Schema {
        path: path.to_owned(),
        problem: problem.to_owned(),
    }
}
