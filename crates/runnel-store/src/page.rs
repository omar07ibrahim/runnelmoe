use std::collections::BTreeMap;
use std::fmt;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::time::Instant;

use runnel_format::{Digest, Limits, Manifest, ObjectRecord, PageTable};
use sha2::{Digest as _, Sha256};

use crate::fs::{READ_CHUNK_BYTES, metadata_len_regular, read_exact_at, read_exact_at_observed};
use crate::{ArtifactSource, Control, StoreError};

/// Identity of one logical page. Page geometry is part of the key because one
/// immutable object may legally be described by multiple page tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageKey {
    object: Digest,
    page_size: u64,
    index: u64,
}

impl PageKey {
    #[must_use]
    pub const fn new(object: Digest, page_size: u64, index: u64) -> Self {
        Self {
            object,
            page_size,
            index,
        }
    }

    #[must_use]
    pub const fn object(self) -> Digest {
        self.object
    }

    #[must_use]
    pub const fn page_size(self) -> u64 {
        self.page_size
    }

    #[must_use]
    pub const fn index(self) -> u64 {
        self.index
    }
}

struct RetainedObject {
    descriptor: ObjectRecord,
    fd: OwnedFd,
    page_table: PageTable,
}

/// Immutable binding from a page key to a retained object descriptor, exact
/// byte range, and authenticated page-table fingerprint.
#[derive(Clone)]
pub struct PageSpec {
    key: PageKey,
    offset: u64,
    length: u64,
    fingerprint: Digest,
    object: Arc<RetainedObject>,
}

impl fmt::Debug for PageSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PageSpec")
            .field("key", &self.key)
            .field("offset", &self.offset)
            .field("length", &self.length)
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

impl PageSpec {
    fn for_object(object: Arc<RetainedObject>, index: u64) -> Result<Self, StoreError> {
        let page_count = object.page_table.page_count();
        if index >= page_count {
            return Err(StoreError::PageOutOfRange {
                index,
                count: page_count,
            });
        }
        let page_size = u64::from(object.descriptor.page_size);
        let offset = page_size
            .checked_mul(index)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "computing a page offset",
            })?;
        let remaining =
            object
                .descriptor
                .length
                .checked_sub(offset)
                .ok_or(StoreError::Invariant {
                    problem: "validated page offset exceeds object length",
                })?;
        let length = remaining.min(page_size);
        let hash = object
            .page_table
            .page_hash(index)
            .ok_or(StoreError::Invariant {
                problem: "verified page table omitted an in-range page hash",
            })?;
        Ok(Self {
            key: PageKey::new(object.descriptor.digest, page_size, index),
            offset,
            length,
            fingerprint: Digest::from_bytes(hash),
            object,
        })
    }

    #[must_use]
    pub const fn key(&self) -> PageKey {
        self.key
    }

    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }

    #[must_use]
    pub const fn fingerprint(&self) -> Digest {
        self.fingerprint
    }

    pub(crate) fn allocation_bytes(&self) -> Result<u64, StoreError> {
        crate::fs::aligned_page_buffer_bytes(self.length)
    }
}

/// A page whose exact bytes matched its retained page-table entry.
#[derive(Clone)]
pub struct VerifiedPage {
    key: PageKey,
    // Keep the reader's uniquely owned allocation behind the shared handle.
    // `Arc<[u8]>::from(Vec<u8>)` may allocate and copy the complete payload,
    // temporarily doubling page memory at the loading-to-resident boundary.
    // Moving the Vec into an Arc preserves the original payload allocation;
    // the separately bounded cache-entry metadata owns only this small Vec
    // control block and Arc header.
    bytes: Arc<Vec<u8>>,
}

impl fmt::Debug for VerifiedPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedPage")
            .field("key", &self.key)
            .field("length", &self.bytes.len())
            .finish()
    }
}

impl VerifiedPage {
    #[must_use]
    pub const fn key(&self) -> PageKey {
        self.key
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub(crate) fn allocation_bytes(&self) -> u64 {
        u64::try_from(self.bytes.capacity()).unwrap_or(u64::MAX)
    }
}

/// Physical work performed by a successful authoritative page read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadStats {
    physical_bytes: u64,
    io_nanoseconds: u64,
}

impl ReadStats {
    #[must_use]
    pub const fn new(physical_bytes: u64, io_nanoseconds: u64) -> Self {
        Self {
            physical_bytes,
            io_nanoseconds,
        }
    }

    #[must_use]
    pub const fn physical_bytes(self) -> u64 {
        self.physical_bytes
    }

    #[must_use]
    pub const fn io_nanoseconds(self) -> u64 {
        self.io_nanoseconds
    }
}

/// Descriptor-backed artifact metadata with no eagerly retained object bytes.
#[derive(Clone)]
pub struct StoredArtifact {
    manifest: Arc<Manifest>,
    objects: Arc<BTreeMap<Digest, Arc<RetainedObject>>>,
}

impl fmt::Debug for StoredArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoredArtifact")
            .field("artifact_id", &self.artifact_id())
            .field("objects", &self.objects.len())
            .finish_non_exhaustive()
    }
}

impl StoredArtifact {
    pub fn open_source(
        source: &ArtifactSource,
        limits: Limits,
        control: &Control,
    ) -> Result<Self, StoreError> {
        Self::open_source_internal(source, limits, None, control)
    }

    pub fn open_source_with_expected_id(
        source: &ArtifactSource,
        limits: Limits,
        expected: Digest,
        control: &Control,
    ) -> Result<Self, StoreError> {
        Self::open_source_internal(source, limits, Some(expected), control)
    }

    fn open_source_internal(
        source: &ArtifactSource,
        limits: Limits,
        expected: Option<Digest>,
        control: &Control,
    ) -> Result<Self, StoreError> {
        let limits = limits.validate()?;
        let manifest_bytes = source.manifest_bytes(limits.manifest_bytes, control)?;
        let manifest = match expected {
            Some(expected) => Manifest::parse_with_expected_id(&manifest_bytes, limits, expected)?,
            None => Manifest::parse_with_limits(&manifest_bytes, limits)?,
        };
        drop(manifest_bytes);

        let mut objects = BTreeMap::new();
        for descriptor in &manifest.objects {
            control.check()?;
            let page_table_fd =
                source.open_page_table(descriptor.page_table, descriptor.page_table_length)?;
            let page_table_bytes = read_exact_at(
                &page_table_fd,
                0,
                descriptor.page_table_length,
                "page table",
                control,
            )?;
            let final_table_length = metadata_len_regular(&page_table_fd, "page table")?;
            if final_table_length != descriptor.page_table_length {
                return Err(StoreError::LengthMismatch {
                    kind: "page table",
                    expected: descriptor.page_table_length,
                    actual: final_table_length,
                });
            }
            let page_table = PageTable::parse_verified(descriptor, page_table_bytes)?;
            drop(page_table_fd);

            let object_fd = source.open_object(descriptor.digest, descriptor.length)?;
            let retained = Arc::new(RetainedObject {
                descriptor: descriptor.clone(),
                fd: object_fd,
                page_table,
            });
            if objects.insert(descriptor.digest, retained).is_some() {
                return Err(StoreError::Invariant {
                    problem: "validated manifest contained a duplicate object",
                });
            }
        }
        control.check()?;

        Ok(Self {
            manifest: Arc::new(manifest),
            objects: Arc::new(objects),
        })
    }

    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    #[must_use]
    pub fn artifact_id(&self) -> Digest {
        self.manifest.artifact_id()
    }

    #[must_use]
    pub fn object(&self, digest: Digest) -> Option<&ObjectRecord> {
        self.manifest.object(digest)
    }

    pub fn page_spec(&self, key: PageKey) -> Result<PageSpec, StoreError> {
        let object = self.objects.get(&key.object()).ok_or(StoreError::Missing {
            kind: "object descriptor",
        })?;
        if key.page_size() != u64::from(object.descriptor.page_size) {
            return Err(StoreError::Integrity {
                kind: "page geometry",
            });
        }
        PageSpec::for_object(Arc::clone(object), key.index())
    }

    /// Iterates every logical page without materializing a page-spec catalog.
    pub fn page_specs(&self) -> impl Iterator<Item = Result<PageSpec, StoreError>> + '_ {
        self.objects.values().flat_map(|object| {
            let object = Arc::clone(object);
            let count = object.page_table.page_count();
            (0..count).map(move |index| PageSpec::for_object(Arc::clone(&object), index))
        })
    }

    /// Iterates pages intersecting a tensor range. The first and last page may
    /// include bytes belonging to adjacent tensors; callers use the public
    /// tensor descriptor's offset and length to trim them.
    pub fn tensor_page_specs(
        &self,
        tensor_id: u64,
    ) -> Result<impl Iterator<Item = Result<PageSpec, StoreError>> + '_, StoreError> {
        let index = usize::try_from(tensor_id).map_err(|_| StoreError::Missing {
            kind: "tensor descriptor",
        })?;
        let tensor = self
            .manifest
            .tensors
            .get(index)
            .filter(|tensor| tensor.id == tensor_id)
            .ok_or(StoreError::Missing {
                kind: "tensor descriptor",
            })?;
        let object = Arc::clone(
            self.objects
                .get(&tensor.object)
                .ok_or(StoreError::Invariant {
                    problem: "validated tensor names an absent retained object",
                })?,
        );
        let page_size = u64::from(object.descriptor.page_size);
        let end_exclusive =
            tensor
                .offset
                .checked_add(tensor.length)
                .ok_or(StoreError::ArithmeticOverflow {
                    context: "computing a tensor byte range",
                })?;
        let last_byte = end_exclusive.checked_sub(1).ok_or(StoreError::Invariant {
            problem: "validated tensor has zero length",
        })?;
        let first_page = tensor.offset / page_size;
        let last_page = last_byte / page_size;
        Ok((first_page..=last_page)
            .map(move |page| PageSpec::for_object(Arc::clone(&object), page)))
    }

    #[must_use]
    pub const fn reader(&self) -> SyncReader {
        SyncReader::new()
    }
}

/// Authoritative synchronous positional page reader.
#[derive(Clone, Debug, Default)]
pub struct SyncReader;

impl SyncReader {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub fn read(
        &self,
        specification: &PageSpec,
        control: &Control,
    ) -> Result<(VerifiedPage, ReadStats), StoreError> {
        let (result, statistics) = self.read_observed(specification, control);
        result.map(|page| (page, statistics))
    }

    /// Performs one read while preserving physical work statistics on every
    /// failure path. Unverified bytes never appear in the returned result.
    pub(crate) fn read_observed(
        &self,
        specification: &PageSpec,
        control: &Control,
    ) -> (Result<VerifiedPage, StoreError>, ReadStats) {
        let mut physical_bytes = 0;
        let mut io_nanoseconds = 0;
        let result = (|| {
            control.check()?;
            validate_specification(specification)?;

            let expected_object_length = specification.object.descriptor.length;
            let initial_length = metadata_len_regular(&specification.object.fd, "object")?;
            if initial_length != expected_object_length {
                return Err(StoreError::LengthMismatch {
                    kind: "object",
                    expected: expected_object_length,
                    actual: initial_length,
                });
            }

            let io_started = Instant::now();
            let (read_result, observed_bytes) = read_exact_at_observed(
                &specification.object.fd,
                specification.offset,
                specification.length,
                "page",
                control,
            );
            io_nanoseconds = u64::try_from(io_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            physical_bytes = observed_bytes;
            let bytes = read_result?;

            let final_length = metadata_len_regular(&specification.object.fd, "object")?;
            if final_length != expected_object_length {
                return Err(StoreError::LengthMismatch {
                    kind: "object",
                    expected: expected_object_length,
                    actual: final_length,
                });
            }

            let mut hasher = Sha256::new();
            for chunk in bytes.chunks(READ_CHUNK_BYTES) {
                control.check()?;
                hasher.update(chunk);
            }
            let actual_hash: [u8; 32] = hasher.finalize().into();
            if Digest::from_bytes(actual_hash) != specification.fingerprint {
                return Err(StoreError::Integrity { kind: "page" });
            }
            control.check()?;

            Ok(VerifiedPage {
                key: specification.key,
                bytes: Arc::new(bytes),
            })
        })();
        (result, ReadStats::new(physical_bytes, io_nanoseconds))
    }

    pub fn read_with_stats(
        &self,
        specification: &PageSpec,
        control: &Control,
    ) -> Result<(VerifiedPage, ReadStats), StoreError> {
        self.read(specification, control)
    }
}

fn validate_specification(specification: &PageSpec) -> Result<(), StoreError> {
    let object = &specification.object;
    let page_size = u64::from(object.descriptor.page_size);
    if specification.key.object() != object.descriptor.digest
        || specification.key.page_size() != page_size
        || object.page_table.object_digest() != object.descriptor.digest
        || u64::from(object.page_table.page_size()) != page_size
        || object.page_table.object_length() != object.descriptor.length
    {
        return Err(StoreError::Integrity {
            kind: "page specification",
        });
    }
    let expected_offset =
        page_size
            .checked_mul(specification.key.index())
            .ok_or(StoreError::ArithmeticOverflow {
                context: "validating a page offset",
            })?;
    let expected_length = object
        .descriptor
        .length
        .checked_sub(expected_offset)
        .map(|remaining| remaining.min(page_size))
        .ok_or(StoreError::Integrity {
            kind: "page specification",
        })?;
    let retained_hash = object
        .page_table
        .page_hash(specification.key.index())
        .map(Digest::from_bytes)
        .ok_or(StoreError::PageOutOfRange {
            index: specification.key.index(),
            count: object.page_table.page_count(),
        })?;
    if specification.offset != expected_offset
        || specification.length != expected_length
        || specification.fingerprint != retained_hash
    {
        return Err(StoreError::Integrity {
            kind: "page specification",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};

    use runnel_fixture::{MultiPageFixture, PAGE_SIZE};

    use super::PageKey;
    use crate::{ArtifactSource, Control, StoreError};

    #[test]
    fn integrity_failure_retains_observed_physical_work() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("artifact");
        let fixture = MultiPageFixture::build();
        let identity = fixture.write_new(&root).unwrap();
        let source = ArtifactSource::open(&root).unwrap();
        let artifact = source
            .open_with_expected_id(
                runnel_format::Limits::default(),
                identity.artifact_id,
                &Control::default(),
            )
            .unwrap();
        let specification = artifact
            .page_spec(PageKey::new(
                identity.object_digest,
                u64::from(PAGE_SIZE),
                0,
            ))
            .unwrap();

        let object_path = root
            .join("objects/sha256")
            .join(identity.object_digest.path_component());
        let mut object = OpenOptions::new().write(true).open(object_path).unwrap();
        object.seek(SeekFrom::Start(11)).unwrap();
        object.write_all(&[0xff]).unwrap();

        let (result, statistics) = artifact
            .reader()
            .read_observed(&specification, &Control::default());
        assert_eq!(result.unwrap_err(), StoreError::Integrity { kind: "page" });
        assert_eq!(statistics.physical_bytes(), u64::from(PAGE_SIZE));
    }

    #[test]
    fn verified_short_page_retains_one_aligned_payload_allocation() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("artifact");
        let fixture = MultiPageFixture::build();
        let identity = fixture.write_new(&root).unwrap();
        let source = ArtifactSource::open(&root).unwrap();
        let artifact = source
            .open_with_expected_id(
                runnel_format::Limits::default(),
                identity.artifact_id,
                &Control::unbounded(),
            )
            .unwrap();
        let specification = artifact
            .page_spec(PageKey::new(
                identity.object_digest,
                u64::from(PAGE_SIZE),
                2,
            ))
            .unwrap();
        let (page, _) = artifact
            .reader()
            .read(&specification, &Control::unbounded())
            .unwrap();

        assert_eq!(page.len(), 17);
        assert_eq!(specification.allocation_bytes().unwrap(), 64);
        assert_eq!(page.allocation_bytes(), 64);
    }

    #[test]
    fn cancellation_after_read_and_during_hash_never_exposes_bytes() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("artifact");
        let fixture = MultiPageFixture::build();
        let identity = fixture.write_new(&root).unwrap();
        let source = ArtifactSource::open(&root).unwrap();
        let artifact = source
            .open_with_expected_id(
                runnel_format::Limits::default(),
                identity.artifact_id,
                &Control::unbounded(),
            )
            .unwrap();
        let specification = artifact
            .page_spec(PageKey::new(
                identity.object_digest,
                u64::from(PAGE_SIZE),
                0,
            ))
            .unwrap();

        // Three checkpoints reach the final read-loop checkpoint after pread;
        // the fourth cancels before any buffer can leave the reader.
        let (after_read, after_read_stats) = artifact
            .reader()
            .read_observed(&specification, &Control::cancel_after_checks(3));
        assert_eq!(after_read.unwrap_err(), StoreError::Cancelled);
        assert_eq!(after_read_stats.physical_bytes(), u64::from(PAGE_SIZE));

        // Allow that post-read checkpoint, then cancel at the first bounded
        // hashing checkpoint. Again only the typed error and physical count
        // cross the trust boundary.
        let (during_hash, hash_stats) = artifact
            .reader()
            .read_observed(&specification, &Control::cancel_after_checks(4));
        assert_eq!(during_hash.unwrap_err(), StoreError::Cancelled);
        assert_eq!(hash_stats.physical_bytes(), u64::from(PAGE_SIZE));
    }
}
