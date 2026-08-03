use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Take};
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt as _;

use crate::error::FormatError;
use crate::manifest::{Digest, Limits, Manifest, ObjectRecord, TensorRecord};

const PAGE_TABLE_MAGIC: &[u8; 8] = b"RMOAPG1\n";
const PAGE_TABLE_HEADER_BYTES: usize = 64;

/// Caller-owned byte inputs for a tiny artifact. Keys are typed digests, so a
/// manifest value can never become a path.
#[derive(Debug, Default)]
pub struct ArtifactBytes {
    pub manifest: Vec<u8>,
    pub objects: BTreeMap<Digest, Vec<u8>>,
    pub page_tables: BTreeMap<Digest, Vec<u8>>,
}

/// A page table whose complete file length and SHA-256 have already been
/// verified. The immutable backing buffer remains retained for every lookup.
#[derive(Debug)]
pub struct PageTable {
    bytes: Box<[u8]>,
    page_size: u32,
    object_length: u64,
    page_count: u64,
    object_digest: Digest,
}

impl PageTable {
    pub fn parse_verified(record: &ObjectRecord, bytes: Vec<u8>) -> Result<Self, FormatError> {
        let actual_length =
            u64::try_from(bytes.len()).map_err(|_| FormatError::ArithmeticOverflow {
                context: "page-table input length",
            })?;
        if actual_length != record.page_table_length {
            return Err(FormatError::BlobLengthMismatch {
                kind: "page table",
                digest: record.page_table,
                expected: record.page_table_length,
                actual: actual_length,
            });
        }

        // The file's own digest is deliberately checked before interpreting or
        // exposing any entry from the table.
        let actual_digest = Digest::of(&bytes);
        if actual_digest != record.page_table {
            return Err(FormatError::BlobDigestMismatch {
                kind: "page table",
                expected: record.page_table,
                actual: actual_digest,
            });
        }

        if bytes.len() < PAGE_TABLE_HEADER_BYTES {
            return page_table_error(record, "header is truncated");
        }
        if &bytes[0..8] != PAGE_TABLE_MAGIC {
            return page_table_error(record, "magic does not equal RMOAPG1\\n");
        }
        let version = read_u32(&bytes, 8);
        if version != 1 {
            return page_table_error(record, "version is not 1");
        }
        let page_size = read_u32(&bytes, 12);
        if page_size != record.page_size {
            return page_table_error(record, "page size does not match the manifest");
        }
        let object_length = read_u64(&bytes, 16);
        if object_length != record.length {
            return page_table_error(record, "object length does not match the manifest");
        }
        let page_count = read_u64(&bytes, 24);
        let expected_page_count = record.page_count()?;
        if page_count != expected_page_count {
            return page_table_error(record, "page count does not equal ceil(length / page_size)");
        }
        let header_digest = Digest::from_bytes(bytes[32..64].try_into().map_err(|_| {
            FormatError::ArithmeticOverflow {
                context: "page-table object digest",
            }
        })?);
        if header_digest != record.digest {
            return page_table_error(record, "object digest does not match the manifest");
        }

        let expected_length = page_count
            .checked_mul(32)
            .and_then(|hash_bytes| hash_bytes.checked_add(64))
            .ok_or(FormatError::ArithmeticOverflow {
                context: "page-table parsed byte length",
            })?;
        if actual_length != expected_length {
            return page_table_error(record, "byte length does not match its page count");
        }

        Ok(Self {
            bytes: bytes.into_boxed_slice(),
            page_size,
            object_length,
            page_count,
            object_digest: record.digest,
        })
    }

    #[must_use]
    pub const fn page_size(&self) -> u32 {
        self.page_size
    }

    #[must_use]
    pub const fn object_length(&self) -> u64 {
        self.object_length
    }

    #[must_use]
    pub const fn page_count(&self) -> u64 {
        self.page_count
    }

    #[must_use]
    pub const fn object_digest(&self) -> Digest {
        self.object_digest
    }

    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn page_hash(&self, index: u64) -> Option<[u8; 32]> {
        if index >= self.page_count {
            return None;
        }
        let offset = usize::try_from(index)
            .ok()?
            .checked_mul(32)?
            .checked_add(PAGE_TABLE_HEADER_BYTES)?;
        let end = offset.checked_add(32)?;
        self.bytes.get(offset..end)?.try_into().ok()
    }

    pub fn verify_object(&self, record: &ObjectRecord, bytes: &[u8]) -> Result<(), FormatError> {
        let actual_length =
            u64::try_from(bytes.len()).map_err(|_| FormatError::ArithmeticOverflow {
                context: "object input length",
            })?;
        if actual_length != record.length {
            return Err(FormatError::BlobLengthMismatch {
                kind: "object",
                digest: record.digest,
                expected: record.length,
                actual: actual_length,
            });
        }

        let actual_digest = Digest::of(bytes);
        if actual_digest != record.digest {
            return Err(FormatError::BlobDigestMismatch {
                kind: "object",
                expected: record.digest,
                actual: actual_digest,
            });
        }
        if self.object_digest != record.digest
            || self.object_length != record.length
            || self.page_size != record.page_size
        {
            return page_table_error(record, "verified table is for a different object record");
        }

        let page_size =
            usize::try_from(self.page_size).map_err(|_| FormatError::ArithmeticOverflow {
                context: "page size representation",
            })?;
        for (page_index, page) in bytes.chunks(page_size).enumerate() {
            let page_index =
                u64::try_from(page_index).map_err(|_| FormatError::ArithmeticOverflow {
                    context: "page index representation",
                })?;
            if self.page_hash(page_index) != Some(*Digest::of(page).as_bytes()) {
                return Err(FormatError::PageHashMismatch {
                    object: record.digest,
                    page_index,
                });
            }
        }
        Ok(())
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut encoded = [0_u8; 4];
    encoded.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(encoded)
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut encoded = [0_u8; 8];
    encoded.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(encoded)
}

fn page_table_error<T>(record: &ObjectRecord, problem: &str) -> Result<T, FormatError> {
    Err(FormatError::PageTable {
        object: record.digest,
        problem: problem.to_owned(),
    })
}

#[derive(Debug)]
struct VerifiedObject {
    bytes: Box<[u8]>,
    page_table: PageTable,
}

/// Immutable, completely byte-verified M1 artifact.
#[derive(Debug)]
pub struct Artifact {
    manifest: Manifest,
    objects: BTreeMap<Digest, VerifiedObject>,
    retained_payload_bytes: u64,
}

/// A tensor descriptor paired with a slice that has passed whole-object and
/// per-page verification.
#[derive(Clone, Copy, Debug)]
pub struct VerifiedTensor<'a> {
    pub descriptor: &'a TensorRecord,
    pub bytes: &'a [u8],
}

impl Artifact {
    /// Opens a tiny M1 artifact and eagerly retains every declared object.
    ///
    /// This method never turns a manifest string into a path and validates all
    /// bytes. M2 will replace its portable pathname traversal with retained
    /// directory descriptors plus `openat2`/`openat` no-follow resolution.
    pub fn open(root: impl AsRef<Path>, limits: Limits) -> Result<Self, FormatError> {
        Self::open_internal(root.as_ref(), limits, None)
    }

    pub fn open_with_expected_id(
        root: impl AsRef<Path>,
        limits: Limits,
        expected: Digest,
    ) -> Result<Self, FormatError> {
        Self::open_internal(root.as_ref(), limits, Some(expected))
    }

    fn open_internal(
        root: &Path,
        limits: Limits,
        expected: Option<Digest>,
    ) -> Result<Self, FormatError> {
        let limits = limits.validate()?;
        let manifest_path = root.join("manifest.json");
        let manifest_bytes = read_bounded_manifest(&manifest_path, limits.manifest_bytes)?;
        let manifest = match expected {
            Some(expected) => Manifest::parse_with_expected_id(&manifest_bytes, limits, expected)?,
            None => Manifest::parse_with_limits(&manifest_bytes, limits)?,
        };
        drop(manifest_bytes);
        ensure_eager_budget(&manifest, limits)?;

        let mut objects = BTreeMap::new();
        let mut page_tables = BTreeMap::new();
        for record in &manifest.objects {
            let page_table_path = root
                .join("page-tables")
                .join("sha256")
                .join(record.page_table.path_component());
            page_tables.insert(
                record.page_table,
                read_exact_blob(
                    &page_table_path,
                    record.page_table,
                    record.page_table_length,
                    "page table",
                )?,
            );

            let object_path = root
                .join("objects")
                .join("sha256")
                .join(record.digest.path_component());
            objects.insert(
                record.digest,
                read_exact_blob(&object_path, record.digest, record.length, "object")?,
            );
        }

        Self::verify_loaded(manifest, objects, page_tables, limits)
    }

    pub fn from_bytes(parts: ArtifactBytes, limits: Limits) -> Result<Self, FormatError> {
        Self::from_bytes_internal(parts, limits, None)
    }

    pub fn from_bytes_with_expected_id(
        parts: ArtifactBytes,
        limits: Limits,
        expected: Digest,
    ) -> Result<Self, FormatError> {
        Self::from_bytes_internal(parts, limits, Some(expected))
    }

    fn from_bytes_internal(
        parts: ArtifactBytes,
        limits: Limits,
        expected: Option<Digest>,
    ) -> Result<Self, FormatError> {
        let limits = limits.validate()?;
        let ArtifactBytes {
            manifest: manifest_bytes,
            objects,
            page_tables,
        } = parts;
        let manifest = match expected {
            Some(expected) => Manifest::parse_with_expected_id(&manifest_bytes, limits, expected)?,
            None => Manifest::parse_with_limits(&manifest_bytes, limits)?,
        };
        drop(manifest_bytes);
        ensure_eager_budget(&manifest, limits)?;
        Self::verify_loaded(manifest, objects, page_tables, limits)
    }

    fn verify_loaded(
        manifest: Manifest,
        mut object_bytes: BTreeMap<Digest, Vec<u8>>,
        mut page_table_bytes: BTreeMap<Digest, Vec<u8>>,
        limits: Limits,
    ) -> Result<Self, FormatError> {
        ensure_eager_budget(&manifest, limits)?;
        let mut objects = BTreeMap::new();
        for record in &manifest.objects {
            let table_bytes =
                page_table_bytes
                    .remove(&record.page_table)
                    .ok_or(FormatError::MissingBlob {
                        kind: "page table",
                        digest: record.page_table,
                    })?;
            let page_table = PageTable::parse_verified(record, table_bytes)?;
            let bytes = object_bytes
                .remove(&record.digest)
                .ok_or(FormatError::MissingBlob {
                    kind: "object",
                    digest: record.digest,
                })?;
            page_table.verify_object(record, &bytes)?;
            objects.insert(
                record.digest,
                VerifiedObject {
                    bytes: bytes.into_boxed_slice(),
                    page_table,
                },
            );
        }

        let retained_payload_bytes = required_eager_bytes(&manifest)?;
        Ok(Self {
            manifest,
            objects,
            retained_payload_bytes,
        })
    }

    #[must_use]
    pub const fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    #[must_use]
    pub const fn artifact_id(&self) -> Digest {
        self.manifest.artifact_id()
    }

    #[must_use]
    pub const fn retained_payload_bytes(&self) -> u64 {
        self.retained_payload_bytes
    }

    #[must_use]
    pub fn page_table(&self, object: Digest) -> Option<&PageTable> {
        self.objects.get(&object).map(|entry| &entry.page_table)
    }

    #[must_use]
    pub fn tensor_bytes(&self, id: u64) -> Option<&[u8]> {
        let index = usize::try_from(id).ok()?;
        let descriptor = self.manifest.tensors.get(index)?;
        if descriptor.id != id {
            return None;
        }
        let object = self.objects.get(&descriptor.object)?;
        let start = usize::try_from(descriptor.offset).ok()?;
        let end = usize::try_from(descriptor.offset.checked_add(descriptor.length)?).ok()?;
        object.bytes.get(start..end)
    }

    #[must_use]
    pub fn tensor_by_role(&self, role: &str) -> Option<VerifiedTensor<'_>> {
        let descriptor = self.manifest.tensor_by_role(role)?;
        let bytes = self.tensor_bytes(descriptor.id)?;
        Some(VerifiedTensor { descriptor, bytes })
    }

    pub fn load_tensor(&self, role: &str) -> Result<Vec<u8>, FormatError> {
        self.tensor_by_role(role)
            .map(|tensor| tensor.bytes.to_vec())
            .ok_or_else(|| FormatError::UnknownTensorRole {
                role: role.to_owned(),
            })
    }
}

fn required_eager_bytes(manifest: &Manifest) -> Result<u64, FormatError> {
    let mut total = u64::try_from(manifest.canonical_bytes().len()).map_err(|_| {
        FormatError::ArithmeticOverflow {
            context: "manifest memory accounting",
        }
    })?;
    for record in &manifest.objects {
        total = total
            .checked_add(record.length)
            .and_then(|value| value.checked_add(record.page_table_length))
            .ok_or(FormatError::ArithmeticOverflow {
                context: "eager artifact memory accounting",
            })?;
    }
    Ok(total)
}

fn ensure_eager_budget(manifest: &Manifest, limits: Limits) -> Result<(), FormatError> {
    let required = required_eager_bytes(manifest)?;
    if required > limits.eager_memory_bytes {
        return Err(FormatError::MemoryBudgetExceeded {
            required,
            limit: limits.eager_memory_bytes,
        });
    }
    Ok(())
}

fn read_bounded_manifest(path: &Path, limit: usize) -> Result<Vec<u8>, FormatError> {
    let file = open_regular(path, "manifest")?;
    let limit_u64 = u64::try_from(limit).map_err(|_| FormatError::ArithmeticOverflow {
        context: "manifest byte limit",
    })?;
    let mut bytes = read_at_most(file, limit_u64.saturating_add(1), "reading manifest")?;
    if bytes.len() > limit {
        return Err(FormatError::ManifestTooLarge {
            actual: bytes.len(),
            limit,
        });
    }
    bytes.shrink_to_fit();
    Ok(bytes)
}

fn read_exact_blob(
    path: &Path,
    digest: Digest,
    expected: u64,
    kind: &'static str,
) -> Result<Vec<u8>, FormatError> {
    let file = open_regular(path, kind)?;
    let metadata = file.metadata().map_err(|error| FormatError::Io {
        operation: "reading retained file metadata",
        message: error.to_string(),
    })?;
    if metadata.len() != expected {
        return Err(FormatError::BlobLengthMismatch {
            kind,
            digest,
            expected,
            actual: metadata.len(),
        });
    }
    let bytes = read_at_most(file, expected.saturating_add(1), "reading artifact blob")?;
    let actual = u64::try_from(bytes.len()).map_err(|_| FormatError::ArithmeticOverflow {
        context: "artifact blob input length",
    })?;
    if actual != expected {
        return Err(FormatError::BlobLengthMismatch {
            kind,
            digest,
            expected,
            actual,
        });
    }
    Ok(bytes)
}

fn open_regular(path: &Path, kind: &'static str) -> Result<File, FormatError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let file = options.open(path).map_err(|error| FormatError::Io {
        operation: "opening artifact file",
        message: error.to_string(),
    })?;
    let metadata = file.metadata().map_err(|error| FormatError::Io {
        operation: "reading retained file metadata",
        message: error.to_string(),
    })?;
    if !metadata.is_file() {
        return Err(FormatError::NotRegularFile { kind });
    }
    Ok(file)
}

fn read_at_most(file: File, maximum: u64, operation: &'static str) -> Result<Vec<u8>, FormatError> {
    let initial_capacity =
        usize::try_from(maximum.min(1024 * 1024)).map_err(|_| FormatError::ArithmeticOverflow {
            context: "bounded read capacity",
        })?;
    let mut bytes = Vec::with_capacity(initial_capacity);
    let mut reader: Take<File> = file.take(maximum);
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| FormatError::Io {
            operation,
            message: error.to_string(),
        })?;
    Ok(bytes)
}
