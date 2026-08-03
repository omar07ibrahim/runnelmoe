#[cfg(test)]
use std::collections::VecDeque;
use std::collections::{BTreeMap, BTreeSet};
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;
use std::sync::{Mutex, MutexGuard, TryLockError};

use runnel_format::{Digest, Limits, Manifest, ObjectRecord, PageTable};
use sha2::{Digest as _, Sha256};

use crate::budget::{DiskBudget, DiskUsage};
use crate::fs;
use crate::layout::{Layout, LeafEntry};
use crate::stage::{BlobKind, ResumeToken};
use crate::{ArtifactSource, Control, StoreError, StoredArtifact};

const MIN_COPY_BUFFER_BYTES: usize = 4 * 1024;
const MAX_COPY_BUFFER_BYTES: usize = 2 * 1024 * 1024;
const DEFAULT_COPY_BUFFER_BYTES: usize = 64 * 1024;
const STAGE_CREATE_ATTEMPTS: usize = 16;

#[cfg(test)]
#[derive(Clone, Debug)]
enum TestRenameFault {
    Unsupported {
        cancel: Option<crate::CancellationToken>,
    },
    AmbiguousPublished,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
enum TestLinkFault {
    AmbiguousPublished,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestSyncTarget {
    StageFile(BlobKind),
    Digest(BlobKind),
    Staging,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct PublicationFaults {
    rename: Mutex<BTreeMap<BlobKind, VecDeque<TestRenameFault>>>,
    link: Mutex<BTreeMap<BlobKind, VecDeque<TestLinkFault>>>,
    link_cleanup_failures: Mutex<BTreeMap<BlobKind, u64>>,
    stage_file_sync_failures: Mutex<BTreeMap<BlobKind, u64>>,
    stage_cleanup_failure_after: Mutex<BTreeMap<BlobKind, u64>>,
    digest_sync_failures: Mutex<BTreeMap<BlobKind, u64>>,
    staging_sync_failures: Mutex<u64>,
    gc_unlink_failure_after: Mutex<BTreeMap<BlobKind, u64>>,
    gc_cancel_after_unlinks: Mutex<Option<(u64, crate::CancellationToken)>>,
    sync_events: Mutex<Vec<TestSyncTarget>>,
}

#[cfg(test)]
impl PublicationFaults {
    fn push_rename(&self, kind: BlobKind, fault: TestRenameFault) {
        self.rename
            .lock()
            .expect("test rename-fault mutex")
            .entry(kind)
            .or_default()
            .push_back(fault);
    }

    fn take_rename(&self, kind: BlobKind) -> Option<TestRenameFault> {
        self.rename
            .lock()
            .expect("test rename-fault mutex")
            .get_mut(&kind)
            .and_then(VecDeque::pop_front)
    }

    fn push_link(&self, kind: BlobKind, fault: TestLinkFault) {
        self.link
            .lock()
            .expect("test link-fault mutex")
            .entry(kind)
            .or_default()
            .push_back(fault);
    }

    fn take_link(&self, kind: BlobKind) -> Option<TestLinkFault> {
        self.link
            .lock()
            .expect("test link-fault mutex")
            .get_mut(&kind)
            .and_then(VecDeque::pop_front)
    }

    fn fail_link_cleanup(&self, kind: BlobKind, count: u64) {
        self.link_cleanup_failures
            .lock()
            .expect("test link-cleanup mutex")
            .insert(kind, count);
    }

    fn take_link_cleanup_failure(&self, kind: BlobKind) -> bool {
        take_failure(&self.link_cleanup_failures, kind)
    }

    fn fail_stage_file_sync(&self, kind: BlobKind, count: u64) {
        self.stage_file_sync_failures
            .lock()
            .expect("test stage-file-sync mutex")
            .insert(kind, count);
    }

    fn take_stage_file_sync_failure(&self, kind: BlobKind) -> bool {
        take_failure(&self.stage_file_sync_failures, kind)
    }

    fn fail_stage_cleanup_after(&self, kind: BlobKind, successful_unlinks: u64) {
        self.stage_cleanup_failure_after
            .lock()
            .expect("test stage-cleanup mutex")
            .insert(kind, successful_unlinks);
    }

    fn take_stage_cleanup_failure(&self, kind: BlobKind) -> bool {
        take_countdown_failure(&self.stage_cleanup_failure_after, kind)
    }

    fn fail_digest_sync(&self, kind: BlobKind, count: u64) {
        self.digest_sync_failures
            .lock()
            .expect("test digest-sync mutex")
            .insert(kind, count);
    }

    fn take_digest_sync_failure(&self, kind: BlobKind) -> bool {
        take_failure(&self.digest_sync_failures, kind)
    }

    fn fail_staging_sync(&self, count: u64) {
        *self
            .staging_sync_failures
            .lock()
            .expect("test staging-sync mutex") = count;
    }

    fn take_staging_sync_failure(&self) -> bool {
        let mut remaining = self
            .staging_sync_failures
            .lock()
            .expect("test staging-sync mutex");
        if *remaining == 0 {
            return false;
        }
        *remaining -= 1;
        true
    }

    fn fail_gc_unlink_after(&self, kind: BlobKind, successful_unlinks: u64) {
        self.gc_unlink_failure_after
            .lock()
            .expect("test GC-unlink mutex")
            .insert(kind, successful_unlinks);
    }

    fn take_gc_unlink_failure(&self, kind: BlobKind) -> bool {
        take_countdown_failure(&self.gc_unlink_failure_after, kind)
    }

    fn cancel_gc_after_unlinks(&self, count: u64, cancellation: crate::CancellationToken) {
        assert!(count > 0, "GC cancellation count must be nonzero");
        *self
            .gc_cancel_after_unlinks
            .lock()
            .expect("test GC-cancellation mutex") = Some((count, cancellation));
    }

    fn observe_gc_unlink(&self) {
        let cancellation = {
            let mut state = self
                .gc_cancel_after_unlinks
                .lock()
                .expect("test GC-cancellation mutex");
            let Some((remaining, _cancellation)) = state.as_mut() else {
                return;
            };
            *remaining -= 1;
            if *remaining == 0 {
                state.take().map(|(_remaining, cancellation)| cancellation)
            } else {
                None
            }
        };
        if let Some(cancellation) = cancellation {
            cancellation.cancel();
        }
    }

    fn record_sync(&self, target: TestSyncTarget) {
        self.sync_events
            .lock()
            .expect("test sync-events mutex")
            .push(target);
    }

    fn take_sync_events(&self) -> Vec<TestSyncTarget> {
        std::mem::take(&mut *self.sync_events.lock().expect("test sync-events mutex"))
    }
}

#[cfg(test)]
fn take_failure(failures: &Mutex<BTreeMap<BlobKind, u64>>, kind: BlobKind) -> bool {
    let mut failures = failures.lock().expect("test publication-fault mutex");
    let Some(remaining) = failures.get_mut(&kind) else {
        return false;
    };
    if *remaining == 0 {
        return false;
    }
    *remaining -= 1;
    true
}

#[cfg(test)]
fn take_countdown_failure(failures: &Mutex<BTreeMap<BlobKind, u64>>, kind: BlobKind) -> bool {
    let mut failures = failures.lock().expect("test countdown-fault mutex");
    let Some(remaining) = failures.get_mut(&kind) else {
        return false;
    };
    if *remaining > 0 {
        *remaining -= 1;
        return false;
    }
    failures.remove(&kind);
    true
}

/// Operator policy for one content-addressed store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CasConfig {
    pub disk_budget: DiskBudget,
    pub copy_buffer_bytes: usize,
}

impl CasConfig {
    fn validate(self) -> Result<Self, StoreError> {
        self.disk_budget.validate()?;
        if !(MIN_COPY_BUFFER_BYTES..=MAX_COPY_BUFFER_BYTES).contains(&self.copy_buffer_bytes) {
            return Err(StoreError::InvalidConfig {
                field: "copy_buffer_bytes",
                problem: "must be between 4096 and 2097152",
            });
        }
        Ok(self)
    }
}

impl Default for CasConfig {
    fn default() -> Self {
        Self {
            disk_budget: DiskBudget::default(),
            copy_buffer_bytes: DEFAULT_COPY_BUFFER_BYTES,
        }
    }
}

/// Result of an idempotent manifest-last import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportReceipt {
    pub artifact_id: Digest,
    pub published_blobs: u64,
    pub reused_blobs: u64,
    pub published_bytes: u64,
}

/// Fail-closed mark/sweep behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GcPolicy {
    pub dry_run: bool,
    pub remove_staging: bool,
}

/// Validated garbage-collection plan and completed sweep counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GcReport {
    pub candidate_blobs: u64,
    pub candidate_bytes: u64,
    pub deleted_blobs: u64,
    pub deleted_bytes: u64,
    pub dry_run: bool,
}

/// Descriptor-retaining internal content-addressed store.
#[derive(Debug)]
pub struct Cas {
    layout: Layout,
    config: CasConfig,
    local_transaction: Mutex<()>,
    #[cfg(test)]
    publication_faults: PublicationFaults,
}

impl Cas {
    /// Creates a new exact CAS layout. The final path component must not exist.
    pub fn create(root: impl AsRef<Path>, config: CasConfig) -> Result<Self, StoreError> {
        let config = config.validate()?;
        let layout = Layout::create(root.as_ref())?;
        Ok(Self {
            layout,
            config,
            local_transaction: Mutex::new(()),
            #[cfg(test)]
            publication_faults: PublicationFaults::default(),
        })
    }

    /// Opens an existing exact CAS layout and rejects any unexpected entry.
    pub fn open(root: impl AsRef<Path>, config: CasConfig) -> Result<Self, StoreError> {
        let config = config.validate()?;
        let layout = Layout::open(root.as_ref())?;
        Ok(Self {
            layout,
            config,
            local_transaction: Mutex::new(()),
            #[cfg(test)]
            publication_faults: PublicationFaults::default(),
        })
    }

    /// Returns charged final, staging, and filesystem-capacity bytes under the
    /// cross-process transaction lock.
    pub fn usage(&self) -> Result<DiskUsage, StoreError> {
        let _local = self.local_lock()?;
        let _transaction = self.layout.lock()?;
        self.layout.usage()
    }

    /// Lists strict resume tokens for every currently retained stage.
    pub fn resume_tokens(&self) -> Result<Vec<ResumeToken>, StoreError> {
        let _local = self.local_lock()?;
        let _transaction = self.layout.lock()?;
        self.layout
            .stage_entries()?
            .into_iter()
            .map(|entry| ResumeToken::from_stage_name(&entry.name))
            .collect()
    }

    /// Authenticates and imports one standalone artifact. The caller-supplied
    /// expected ID is mandatory; a merely reported source identity is never
    /// trusted.
    pub fn import(
        &self,
        source: &ArtifactSource,
        expected_id: Digest,
        limits: Limits,
        control: &Control,
    ) -> Result<ImportReceipt, StoreError> {
        self.import_with_resumes(source, expected_id, limits, &[], control)
    }

    /// Resumes exact stages whose opaque tokens bind kind, digest, length, and
    /// random stage name. Tokens not belonging to this artifact are rejected
    /// before mutation.
    pub fn import_with_resumes(
        &self,
        source: &ArtifactSource,
        expected_id: Digest,
        limits: Limits,
        resumes: &[ResumeToken],
        control: &Control,
    ) -> Result<ImportReceipt, StoreError> {
        control.check()?;
        let limits = limits.validate()?;
        let manifest_bytes = source.manifest_bytes(limits.manifest_bytes, control)?;
        let manifest = Manifest::parse_with_expected_id(&manifest_bytes, limits, expected_id)?;
        let manifest_length =
            u64::try_from(manifest_bytes.len()).map_err(|_| StoreError::ArithmeticOverflow {
                context: "representing manifest byte length",
            })?;

        let _local = self.local_lock_with_control(control)?;
        let _transaction = self.layout.lock_with_control(control)?;
        control.check()?;
        self.layout.usage()?;

        if let Ok(manifest_fd) =
            self.layout
                .open_digest(BlobKind::Manifest, expected_id, manifest_length)
        {
            let reused = self.verify_committed(&manifest, &manifest_fd, control)?;
            self.cleanup_committed_stages(&manifest)?;
            self.confirm_committed_durability()?;
            return Ok(ImportReceipt {
                artifact_id: expected_id,
                published_blobs: 0,
                reused_blobs: reused,
                published_bytes: 0,
            });
        }

        let resume_map = self.validate_resumes(&manifest, manifest_length, resumes)?;
        self.preflight_import(&manifest, manifest_length, &resume_map)?;

        let mut published_blobs = 0_u64;
        let mut reused_blobs = 0_u64;
        let mut published_bytes = 0_u64;
        let mut page_tables = BTreeMap::new();

        for record in &manifest.objects {
            control.check()?;
            let source_fd = source.open_page_table(record.page_table, record.page_table_length)?;
            let token = resume_map.get(&(BlobKind::PageTable, record.page_table));
            let publication = self.ensure_blob(
                &source_fd,
                BlobKind::PageTable,
                record.page_table,
                record.page_table_length,
                None,
                token,
                control,
                |fd, control| {
                    let bytes =
                        fs::read_exact_at(fd, 0, record.page_table_length, "page table", control)?;
                    PageTable::parse_verified(record, bytes).map_err(StoreError::from)
                },
            )?;
            account_publication(
                publication,
                record.page_table_length,
                &mut published_blobs,
                &mut reused_blobs,
                &mut published_bytes,
            )?;
            let table_fd = self.layout.open_digest(
                BlobKind::PageTable,
                record.page_table,
                record.page_table_length,
            )?;
            let table_bytes = fs::read_exact_at(
                &table_fd,
                0,
                record.page_table_length,
                "page table",
                control,
            )?;
            let table = PageTable::parse_verified(record, table_bytes)?;
            page_tables.insert(record.digest, table);
        }

        for record in &manifest.objects {
            control.check()?;
            let table = page_tables
                .get(&record.digest)
                .ok_or(StoreError::Invariant {
                    problem: "verified page table disappeared from import plan",
                })?;
            let source_fd = source.open_object(record.digest, record.length)?;
            let token = resume_map.get(&(BlobKind::Object, record.digest));
            let publication = self.ensure_blob(
                &source_fd,
                BlobKind::Object,
                record.digest,
                record.length,
                Some(record.page_size),
                token,
                control,
                |fd, control| {
                    verify_object(fd, record, table, control, self.config.copy_buffer_bytes)
                },
            )?;
            account_publication(
                publication,
                record.length,
                &mut published_blobs,
                &mut reused_blobs,
                &mut published_bytes,
            )?;
        }

        control.check()?;
        let source_manifest = source.duplicate_manifest()?;
        let token = resume_map.get(&(BlobKind::Manifest, expected_id));
        let publication = self.ensure_blob(
            &source_manifest,
            BlobKind::Manifest,
            expected_id,
            manifest_length,
            None,
            token,
            control,
            |fd, control| {
                let bytes = fs::read_exact_at(fd, 0, manifest_length, "manifest", control)?;
                Manifest::parse_with_expected_id(&bytes, limits, expected_id)
                    .map(|_| ())
                    .map_err(StoreError::from)
            },
        )?;
        account_publication(
            publication,
            manifest_length,
            &mut published_blobs,
            &mut reused_blobs,
            &mut published_bytes,
        )?;

        Ok(ImportReceipt {
            artifact_id: expected_id,
            published_blobs,
            reused_blobs,
            published_bytes,
        })
    }

    /// Opens an imported artifact by trusted ID without eager object reads.
    pub fn open_artifact(
        &self,
        artifact_id: Digest,
        limits: Limits,
        control: &Control,
    ) -> Result<StoredArtifact, StoreError> {
        control.check()?;
        let manifest = self.layout.open_digest(
            BlobKind::Manifest,
            artifact_id,
            manifest_length_for_open(&self.layout, artifact_id)?,
        )?;
        let source = ArtifactSource::from_retained(
            fs::duplicate(&self.layout.root)?,
            manifest,
            fs::duplicate(&self.layout.objects)?,
            fs::duplicate(&self.layout.page_tables)?,
        )?;
        source.open_with_expected_id(limits, artifact_id, control)
    }

    /// Builds a complete authenticated mark/deletion plan before the first
    /// unlink. Malformed or unexpected content therefore causes zero mutation.
    pub fn gc(
        &self,
        policy: GcPolicy,
        limits: Limits,
        control: &Control,
    ) -> Result<GcReport, StoreError> {
        control.check()?;
        let limits = limits.validate()?;
        let _local = self.local_lock_with_control(control)?;
        let _transaction = self.layout.lock_with_control(control)?;
        let _usage = self.layout.usage()?;

        let manifest_entries = self.layout.digest_entries(BlobKind::Manifest)?;
        let object_entries = self.layout.digest_entries(BlobKind::Object)?;
        let table_entries = self.layout.digest_entries(BlobKind::PageTable)?;
        let stage_entries = self.layout.stage_entries()?;
        let mut live_objects = BTreeSet::new();
        let mut live_tables = BTreeSet::new();

        for entry in &manifest_entries {
            control.check()?;
            let artifact_id = parse_digest_name(&entry.name, "manifest")?;
            let fd = self
                .layout
                .open_digest(BlobKind::Manifest, artifact_id, entry.length)?;
            let bytes = fs::read_bounded_at(&fd, limits.manifest_bytes, "manifest", control)?;
            let manifest = Manifest::parse_with_expected_id(&bytes, limits, artifact_id)?;
            self.verify_manifest_references(&manifest, control)?;
            for record in &manifest.objects {
                live_objects.insert(record.digest);
                live_tables.insert(record.page_table);
            }
        }

        let mut candidates = Vec::new();
        self.plan_unreferenced(
            BlobKind::Object,
            &object_entries,
            &live_objects,
            control,
            &mut candidates,
        )?;
        self.plan_unreferenced(
            BlobKind::PageTable,
            &table_entries,
            &live_tables,
            control,
            &mut candidates,
        )?;
        if policy.remove_staging {
            for entry in &stage_entries {
                control.check()?;
                ResumeToken::from_stage_name(&entry.name)?;
                candidates.push(DeleteCandidate {
                    kind: DeleteKind::Stage,
                    name: entry.name.clone(),
                    device: entry.device,
                    inode: entry.inode,
                    reclaim_bytes: 0,
                    allocated_bytes: entry.charged_bytes,
                });
            }
        }

        assign_reclaim_bytes(
            &mut candidates,
            &manifest_entries,
            &object_entries,
            &table_entries,
            &stage_entries,
        )?;

        let candidate_blobs =
            u64::try_from(candidates.len()).map_err(|_| StoreError::ArithmeticOverflow {
                context: "representing garbage-collection candidate count",
            })?;
        let candidate_bytes = candidates.iter().try_fold(0_u64, |total, candidate| {
            total
                .checked_add(candidate.reclaim_bytes)
                .ok_or(StoreError::ArithmeticOverflow {
                    context: "summing garbage-collection candidate bytes",
                })
        })?;
        let mut report = GcReport {
            candidate_blobs,
            candidate_bytes,
            deleted_blobs: 0,
            deleted_bytes: 0,
            dry_run: policy.dry_run,
        };
        control.check()?;
        if policy.dry_run {
            return Ok(report);
        }

        let mut modified = BTreeSet::new();
        let mut staging_modified = false;
        for candidate in candidates {
            let step =
                (|| {
                    control.check()?;
                    match candidate.kind {
                        DeleteKind::Digest(kind) => {
                            self.unlink_gc_digest(kind, &candidate.name)?;
                            modified.insert(kind);
                        }
                        DeleteKind::Stage => {
                            fs::unlink_leaf(&self.layout.staging, &candidate.name, "staging file")?;
                            staging_modified = true;
                        }
                    }
                    report.deleted_blobs = report.deleted_blobs.checked_add(1).ok_or(
                        StoreError::ArithmeticOverflow {
                            context: "counting deleted garbage-collection blobs",
                        },
                    )?;
                    report.deleted_bytes = report
                        .deleted_bytes
                        .checked_add(candidate.reclaim_bytes)
                        .ok_or(StoreError::ArithmeticOverflow {
                            context: "counting deleted garbage-collection bytes",
                        })?;
                    Ok(())
                })();
            if let Err(error) = step {
                self.sync_gc_directories(&modified, staging_modified)?;
                return Err(error);
            }
            #[cfg(test)]
            self.publication_faults.observe_gc_unlink();
        }
        self.sync_gc_directories(&modified, staging_modified)?;
        Ok(report)
    }

    fn sync_gc_directories(
        &self,
        modified: &BTreeSet<BlobKind>,
        staging_modified: bool,
    ) -> Result<(), StoreError> {
        let mut first_error = None;
        for kind in modified {
            if let Err(error) = self.sync_digest_dir(*kind)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        if staging_modified
            && let Err(error) = self.sync_staging()
            && first_error.is_none()
        {
            first_error = Some(error);
        }
        first_error.map_or(Ok(()), Err)
    }

    fn unlink_gc_digest(&self, kind: BlobKind, name: &str) -> Result<(), StoreError> {
        #[cfg(test)]
        if self.publication_faults.take_gc_unlink_failure(kind) {
            return Err(StoreError::Io {
                operation: "unlinking injected garbage-collection candidate",
                kind: std::io::ErrorKind::Other,
            });
        }
        fs::unlink_leaf(self.layout.digest_dir(kind), name, kind.error_name())
    }

    fn local_lock(&self) -> Result<MutexGuard<'_, ()>, StoreError> {
        self.local_transaction
            .lock()
            .map_err(|_| StoreError::Invariant {
                problem: "local transaction mutex was poisoned",
            })
    }

    fn local_lock_with_control(&self, control: &Control) -> Result<MutexGuard<'_, ()>, StoreError> {
        loop {
            control.check()?;
            match self.local_transaction.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(TryLockError::WouldBlock) => std::thread::yield_now(),
                Err(TryLockError::Poisoned(_)) => {
                    return Err(StoreError::Invariant {
                        problem: "local transaction mutex was poisoned",
                    });
                }
            }
        }
    }

    fn validate_resumes(
        &self,
        manifest: &Manifest,
        manifest_length: u64,
        resumes: &[ResumeToken],
    ) -> Result<BTreeMap<(BlobKind, Digest), ResumeToken>, StoreError> {
        let stages: BTreeMap<String, LeafEntry> = self
            .layout
            .stage_entries()?
            .into_iter()
            .map(|entry| (entry.name.clone(), entry))
            .collect();
        let mut declared = BTreeMap::new();
        declared.insert(
            (BlobKind::Manifest, manifest.artifact_id()),
            manifest_length,
        );
        for record in &manifest.objects {
            declared.insert(
                (BlobKind::PageTable, record.page_table),
                record.page_table_length,
            );
            declared.insert((BlobKind::Object, record.digest), record.length);
        }

        let mut result = BTreeMap::new();
        for token in resumes {
            let key = (token.blob_kind(), token.digest());
            let expected_length = declared
                .get(&key)
                .ok_or(StoreError::ResumeMismatch { field: "artifact" })?;
            token.validate_for(key.0, key.1, *expected_length)?;
            let entry = stages.get(token.stage_name()).ok_or(StoreError::Missing {
                kind: "staging file",
            })?;
            if entry.length > *expected_length {
                return Err(StoreError::ResumeMismatch { field: "length" });
            }
            if result.insert(key, token.clone()).is_some() {
                return Err(StoreError::InvalidResumeToken);
            }
        }
        Ok(result)
    }

    fn preflight_import(
        &self,
        manifest: &Manifest,
        manifest_length: u64,
        resumes: &BTreeMap<(BlobKind, Digest), ResumeToken>,
    ) -> Result<(), StoreError> {
        let usage = self.layout.usage()?;
        let stages: BTreeMap<String, LeafEntry> = self
            .layout
            .stage_entries()?
            .into_iter()
            .map(|entry| (entry.name.clone(), entry))
            .collect();
        let mut blobs = vec![(BlobKind::Manifest, manifest.artifact_id(), manifest_length)];
        for record in &manifest.objects {
            blobs.push((
                BlobKind::PageTable,
                record.page_table,
                record.page_table_length,
            ));
            blobs.push((BlobKind::Object, record.digest, record.length));
        }

        let mut additional = 0_u64;
        for (kind, digest, length) in blobs {
            match self.layout.open_digest(kind, digest, length) {
                Ok(_) => continue,
                Err(StoreError::Missing { .. }) => {}
                Err(error) => return Err(error),
            }
            let planned = usage.round_up(length)?;
            let staged = resumes
                .get(&(kind, digest))
                .and_then(|token| stages.get(token.stage_name()))
                .map_or(0, |entry| entry.charged_bytes);
            additional = additional
                .checked_add(planned.saturating_sub(staged))
                .ok_or(StoreError::ArithmeticOverflow {
                    context: "summing planned CAS allocation",
                })?;
        }
        self.config.disk_budget.preflight(usage, additional)
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_blob<T, Verify>(
        &self,
        source: &impl AsFd,
        kind: BlobKind,
        digest: Digest,
        expected_length: u64,
        object_page_size: Option<u32>,
        resume: Option<&ResumeToken>,
        control: &Control,
        verify: Verify,
    ) -> Result<Publication, StoreError>
    where
        Verify: Fn(&OwnedFd, &Control) -> Result<T, StoreError>,
    {
        match self.layout.open_digest(kind, digest, expected_length) {
            Ok(fd) => {
                verify_digest(
                    &fd,
                    digest,
                    expected_length,
                    kind.error_name(),
                    control,
                    self.config.copy_buffer_bytes,
                )?;
                verify(&fd, control)?;
                // A retained stage may be the only durable name for a final
                // leaf whose earlier publication was not confirmed. Confirm
                // the destination before removing any recovery aliases.
                self.confirm_digest_durability(kind)?;
                self.cleanup_matching_stages(kind, digest, expected_length)?;
                self.confirm_staging_durability(kind)?;
                return Ok(Publication::Reused);
            }
            Err(StoreError::Missing { .. }) => {}
            Err(error) => return Err(error),
        }

        let (token, stage) = match resume {
            Some(token) => {
                token.validate_for(kind, digest, expected_length)?;
                (token.clone(), self.layout.open_stage(token)?)
            }
            None => self.create_stage(kind, digest, expected_length)?,
        };
        self.copy_and_rehash(
            source,
            &stage,
            kind,
            digest,
            expected_length,
            object_page_size,
            resume.is_some(),
            control,
        )?;
        verify_digest(
            &stage,
            digest,
            expected_length,
            kind.error_name(),
            control,
            self.config.copy_buffer_bytes,
        )?;
        verify(&stage, control)?;
        self.sync_stage_file(&stage, kind)?;
        control.check()?;

        let mut ambiguous_error = None;
        let mut published = match self.rename_stage_noreplace(&token, kind, digest) {
            Ok(fs::RenameNoReplace::Published) => true,
            Ok(fs::RenameNoReplace::Unsupported) => {
                control.check()?;
                match self.link_stage_noreplace(&token, kind, digest) {
                    Ok(()) => {
                        self.finish_link_publication(&token, kind)?;
                        return Ok(Publication::Published);
                    }
                    Err(StoreError::AlreadyExists { .. }) => false,
                    Err(error) => {
                        ambiguous_error = Some(error);
                        false
                    }
                }
            }
            Err(StoreError::AlreadyExists { .. }) => false,
            Err(error) => {
                ambiguous_error = Some(error);
                false
            }
        };

        if !published {
            let final_fd = match self.layout.open_digest(kind, digest, expected_length) {
                Ok(fd) => fd,
                Err(StoreError::Missing { .. }) if ambiguous_error.is_some() => {
                    return Err(ambiguous_error.expect("checked as present"));
                }
                Err(error) => return Err(error),
            };
            verify_digest(
                &final_fd,
                digest,
                expected_length,
                kind.error_name(),
                &Control::unbounded(),
                self.config.copy_buffer_bytes,
            )?;
            verify(&final_fd, &Control::unbounded())?;
            if ambiguous_error.is_some() {
                match self.layout.open_stage(&token) {
                    Ok(stage_fd) => {
                        if self.layout.same_file(&final_fd, &stage_fd)? {
                            // A link may have become addressable even though
                            // its result was lost. Preserve the stage until the
                            // destination link is known durable.
                            self.finish_link_publication(&token, kind)?;
                            return Ok(Publication::Published);
                        }
                        self.finish_reused_stage(&token, kind)?;
                        return Ok(Publication::Reused);
                    }
                    Err(StoreError::Missing { .. }) => {
                        // A no-replace rename may have succeeded even when its
                        // result was lost. The verified final plus missing
                        // stage is the descriptor-level evidence of success.
                        published = true;
                    }
                    Err(error) => return Err(error),
                }
            } else {
                self.finish_reused_stage(&token, kind)?;
                return Ok(Publication::Reused);
            }
        }

        self.confirm_blob_durability(kind)?;
        Ok(if published {
            Publication::Published
        } else {
            Publication::Reused
        })
    }

    fn cleanup_matching_stages(
        &self,
        kind: BlobKind,
        digest: Digest,
        expected_length: u64,
    ) -> Result<(), StoreError> {
        let mut staging_modified = false;
        for entry in self.layout.stage_entries()? {
            let token = match ResumeToken::from_stage_name(&entry.name) {
                Ok(token) => token,
                Err(error) => {
                    self.sync_partial_stage_cleanup(staging_modified);
                    return Err(error);
                }
            };
            if token.blob_kind() == kind
                && token.digest() == digest
                && token.expected_length() == expected_length
            {
                if let Err(error) = self.unlink_stage_for_cleanup(&token) {
                    self.sync_partial_stage_cleanup(staging_modified);
                    return Err(error);
                }
                staging_modified = true;
            }
        }
        Ok(())
    }

    fn cleanup_committed_stages(&self, manifest: &Manifest) -> Result<(), StoreError> {
        let mut committed = BTreeSet::new();
        committed.insert((
            BlobKind::Manifest,
            manifest.artifact_id(),
            u64::try_from(manifest.canonical_bytes().len()).map_err(|_| {
                StoreError::ArithmeticOverflow {
                    context: "representing committed manifest length",
                }
            })?,
        ));
        for record in &manifest.objects {
            committed.insert((
                BlobKind::PageTable,
                record.page_table,
                record.page_table_length,
            ));
            committed.insert((BlobKind::Object, record.digest, record.length));
        }
        let mut staging_modified = false;
        for entry in self.layout.stage_entries()? {
            let token = match ResumeToken::from_stage_name(&entry.name) {
                Ok(token) => token,
                Err(error) => {
                    self.sync_partial_stage_cleanup(staging_modified);
                    return Err(error);
                }
            };
            if committed.contains(&(token.blob_kind(), token.digest(), token.expected_length())) {
                if let Err(error) = self.unlink_stage_for_cleanup(&token) {
                    self.sync_partial_stage_cleanup(staging_modified);
                    return Err(error);
                }
                staging_modified = true;
            }
        }
        Ok(())
    }

    fn unlink_stage_for_cleanup(&self, token: &ResumeToken) -> Result<(), StoreError> {
        #[cfg(test)]
        if self
            .publication_faults
            .take_stage_cleanup_failure(token.blob_kind())
        {
            return Err(StoreError::Io {
                operation: "unlinking injected redundant stage",
                kind: std::io::ErrorKind::Other,
            });
        }
        self.layout.unlink_stage(token)
    }

    fn sync_partial_stage_cleanup(&self, staging_modified: bool) {
        if staging_modified {
            // Preserve the first semantic cleanup error while still making
            // every already-completed unlink durable on a best-effort basis.
            let _ = self.sync_staging();
        }
    }

    fn confirm_blob_durability(&self, kind: BlobKind) -> Result<(), StoreError> {
        self.confirm_digest_durability(kind)?;
        self.confirm_staging_durability(kind)
    }

    fn confirm_digest_durability(&self, kind: BlobKind) -> Result<(), StoreError> {
        self.sync_digest_dir(kind)
            .map_err(|_| StoreError::PublishedButDurabilityUnconfirmed {
                kind: kind.error_name(),
            })
    }

    fn confirm_staging_durability(&self, kind: BlobKind) -> Result<(), StoreError> {
        self.sync_staging()
            .map_err(|_| StoreError::PublishedButDurabilityUnconfirmed {
                kind: kind.error_name(),
            })
    }

    fn finish_link_publication(
        &self,
        token: &ResumeToken,
        kind: BlobKind,
    ) -> Result<(), StoreError> {
        // `linkat` does not remove the durable staging name. Confirm the new
        // digest-directory name before attempting that cleanup, so a failed
        // destination sync always leaves an exact resume token behind.
        self.confirm_digest_durability(kind)?;
        // Once the destination is durable, a stale hard-link alias is safe.
        // Preserve the existing success semantics and let a later retry or GC
        // remove it if this best-effort unlink fails.
        let _ = self.unlink_stage_after_link(token, kind);
        self.confirm_staging_durability(kind)
    }

    fn finish_reused_stage(&self, token: &ResumeToken, kind: BlobKind) -> Result<(), StoreError> {
        self.confirm_digest_durability(kind)?;
        if let Err(error) = self.layout.unlink_stage(token) {
            // Preserve the unlink error, while making an ambiguously completed
            // deletion durable on a best-effort basis.
            let _ = self.sync_staging();
            return Err(error);
        }
        self.confirm_staging_durability(kind)
    }

    fn confirm_committed_durability(&self) -> Result<(), StoreError> {
        let unconfirmed = || StoreError::PublishedButDurabilityUnconfirmed { kind: "manifest" };
        self.sync_digest_dir(BlobKind::PageTable)
            .map_err(|_| unconfirmed())?;
        self.sync_digest_dir(BlobKind::Object)
            .map_err(|_| unconfirmed())?;
        self.sync_staging().map_err(|_| unconfirmed())?;
        // The manifest directory is deliberately synced last: it is the
        // artifact commit point and is never confirmed after a failed
        // dependency or staging sync.
        self.sync_digest_dir(BlobKind::Manifest)
            .map_err(|_| unconfirmed())
    }

    fn sync_stage_file(&self, stage: &impl AsFd, _kind: BlobKind) -> Result<(), StoreError> {
        #[cfg(test)]
        {
            self.publication_faults
                .record_sync(TestSyncTarget::StageFile(_kind));
            if self.publication_faults.take_stage_file_sync_failure(_kind) {
                return Err(StoreError::Io {
                    operation: "synchronizing injected staging file",
                    kind: std::io::ErrorKind::Other,
                });
            }
        }
        fs::sync_fd(stage)
    }

    fn sync_digest_dir(&self, kind: BlobKind) -> Result<(), StoreError> {
        #[cfg(test)]
        {
            self.publication_faults
                .record_sync(TestSyncTarget::Digest(kind));
            if self.publication_faults.take_digest_sync_failure(kind) {
                return Err(StoreError::Io {
                    operation: "synchronizing injected digest directory",
                    kind: std::io::ErrorKind::Other,
                });
            }
        }
        self.layout.sync_digest_dir(kind)
    }

    fn sync_staging(&self) -> Result<(), StoreError> {
        #[cfg(test)]
        {
            self.publication_faults.record_sync(TestSyncTarget::Staging);
            if self.publication_faults.take_staging_sync_failure() {
                return Err(StoreError::Io {
                    operation: "synchronizing injected staging directory",
                    kind: std::io::ErrorKind::Other,
                });
            }
        }
        self.layout.sync_staging()
    }

    fn rename_stage_noreplace(
        &self,
        token: &ResumeToken,
        kind: BlobKind,
        digest: Digest,
    ) -> Result<fs::RenameNoReplace, StoreError> {
        #[cfg(test)]
        if let Some(fault) = self.publication_faults.take_rename(kind) {
            match fault {
                TestRenameFault::Unsupported { cancel } => {
                    if let Some(cancellation) = cancel {
                        cancellation.cancel();
                    }
                    return Ok(fs::RenameNoReplace::Unsupported);
                }
                TestRenameFault::AmbiguousPublished => {
                    let result = fs::rename_noreplace(
                        &self.layout.staging,
                        token.stage_name(),
                        self.layout.digest_dir(kind),
                        &digest.path_component(),
                        kind.error_name(),
                    )?;
                    if result != fs::RenameNoReplace::Published {
                        return Err(StoreError::Invariant {
                            problem: "ambiguous-rename injection did not publish",
                        });
                    }
                    return Err(StoreError::Io {
                        operation: "injected ambiguous no-replace rename",
                        kind: std::io::ErrorKind::Other,
                    });
                }
            }
        }
        fs::rename_noreplace(
            &self.layout.staging,
            token.stage_name(),
            self.layout.digest_dir(kind),
            &digest.path_component(),
            kind.error_name(),
        )
    }

    fn link_stage_noreplace(
        &self,
        token: &ResumeToken,
        kind: BlobKind,
        digest: Digest,
    ) -> Result<(), StoreError> {
        #[cfg(test)]
        if let Some(fault) = self.publication_faults.take_link(kind) {
            match fault {
                TestLinkFault::AmbiguousPublished => {
                    fs::link_noreplace(
                        &self.layout.staging,
                        token.stage_name(),
                        self.layout.digest_dir(kind),
                        &digest.path_component(),
                        kind.error_name(),
                    )?;
                    return Err(StoreError::Io {
                        operation: "injected ambiguous no-replace link",
                        kind: std::io::ErrorKind::Other,
                    });
                }
            }
        }
        fs::link_noreplace(
            &self.layout.staging,
            token.stage_name(),
            self.layout.digest_dir(kind),
            &digest.path_component(),
            kind.error_name(),
        )
    }

    fn unlink_stage_after_link(
        &self,
        token: &ResumeToken,
        _kind: BlobKind,
    ) -> Result<(), StoreError> {
        #[cfg(test)]
        if self.publication_faults.take_link_cleanup_failure(_kind) {
            return Err(StoreError::Io {
                operation: "injected post-link staging cleanup",
                kind: std::io::ErrorKind::Other,
            });
        }
        self.layout.unlink_stage(token)
    }

    fn create_stage(
        &self,
        kind: BlobKind,
        digest: Digest,
        expected_length: u64,
    ) -> Result<(ResumeToken, OwnedFd), StoreError> {
        for _ in 0..STAGE_CREATE_ATTEMPTS {
            let mut random = [0_u8; 16];
            getrandom::fill(&mut random).map_err(|_| StoreError::Io {
                operation: "generating a staging name",
                kind: std::io::ErrorKind::Other,
            })?;
            let random_hex = lower_hex(&random);
            let token = ResumeToken::new(kind, digest, expected_length, &random_hex)?;
            match self.layout.create_stage(&token) {
                Ok(fd) => {
                    if let Err(error) = self.sync_staging() {
                        let _ = self.layout.unlink_stage(&token);
                        // Creation may or may not have reached stable storage;
                        // persist the cleanup (or retained safe stage) before
                        // returning the original creation-sync failure.
                        let _ = self.sync_staging();
                        return Err(error);
                    }
                    return Ok((token, fd));
                }
                Err(StoreError::AlreadyExists { .. }) => continue,
                Err(error) => return Err(error),
            }
        }
        Err(StoreError::AlreadyExists {
            kind: "staging file",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_and_rehash(
        &self,
        source: &impl AsFd,
        stage: &impl AsFd,
        kind: BlobKind,
        digest: Digest,
        expected_length: u64,
        object_page_size: Option<u32>,
        resumed: bool,
        control: &Control,
    ) -> Result<(), StoreError> {
        let source_length = fs::metadata_len_regular(source, kind.error_name())?;
        if source_length != expected_length {
            return Err(StoreError::LengthMismatch {
                kind: kind.error_name(),
                expected: expected_length,
                actual: source_length,
            });
        }
        let mut offset = fs::metadata_len_regular(stage, "staging file")?;
        if offset > expected_length {
            return Err(StoreError::ResumeMismatch { field: "length" });
        }
        if resumed
            && kind == BlobKind::Object
            && offset < expected_length
            && let Some(page_size) = object_page_size
        {
            let boundary = offset / u64::from(page_size) * u64::from(page_size);
            if boundary != offset {
                fs::truncate(stage, boundary)?;
                offset = boundary;
            }
        }

        let mut hasher = Sha256::new();
        let mut position = 0_u64;
        while position < offset {
            control.check()?;
            let length = (offset - position).min(self.config.copy_buffer_bytes as u64);
            let staged = fs::read_exact_at(stage, position, length, "staging prefix", control)?;
            let original = fs::read_exact_at(source, position, length, kind.error_name(), control)?;
            if staged != original {
                return Err(StoreError::Integrity {
                    kind: "staging prefix",
                });
            }
            hasher.update(&staged);
            position = position
                .checked_add(length)
                .ok_or(StoreError::ArithmeticOverflow {
                    context: "rehashing a staged prefix",
                })?;
        }

        while position < expected_length {
            control.check()?;
            let length = (expected_length - position).min(self.config.copy_buffer_bytes as u64);
            let bytes = fs::read_exact_at(source, position, length, kind.error_name(), control)?;
            fs::write_all_at(stage, position, &bytes, control)?;
            hasher.update(&bytes);
            position = position
                .checked_add(length)
                .ok_or(StoreError::ArithmeticOverflow {
                    context: "copying staged bytes",
                })?;
        }
        let actual_length = fs::metadata_len_regular(stage, "staging file")?;
        if actual_length != expected_length {
            return Err(StoreError::LengthMismatch {
                kind: "staging file",
                expected: expected_length,
                actual: actual_length,
            });
        }
        let actual = Digest::from_bytes(hasher.finalize().into());
        if actual != digest {
            return Err(StoreError::Integrity {
                kind: kind.error_name(),
            });
        }
        let final_source_length = fs::metadata_len_regular(source, kind.error_name())?;
        if final_source_length != expected_length {
            return Err(StoreError::LengthMismatch {
                kind: kind.error_name(),
                expected: expected_length,
                actual: final_source_length,
            });
        }
        Ok(())
    }

    fn verify_committed(
        &self,
        manifest: &Manifest,
        manifest_fd: &OwnedFd,
        control: &Control,
    ) -> Result<u64, StoreError> {
        verify_digest(
            manifest_fd,
            manifest.artifact_id(),
            manifest.canonical_bytes().len() as u64,
            "manifest",
            control,
            self.config.copy_buffer_bytes,
        )?;
        self.verify_manifest_references(manifest, control)?;
        let objects =
            u64::try_from(manifest.objects.len()).map_err(|_| StoreError::ArithmeticOverflow {
                context: "representing committed object count",
            })?;
        objects
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(StoreError::ArithmeticOverflow {
                context: "counting committed blobs",
            })
    }

    fn verify_manifest_references(
        &self,
        manifest: &Manifest,
        control: &Control,
    ) -> Result<(), StoreError> {
        for record in &manifest.objects {
            control.check()?;
            let table_fd = self.layout.open_digest(
                BlobKind::PageTable,
                record.page_table,
                record.page_table_length,
            )?;
            let table_bytes = fs::read_exact_at(
                &table_fd,
                0,
                record.page_table_length,
                "page table",
                control,
            )?;
            let table = PageTable::parse_verified(record, table_bytes)?;
            let object_fd =
                self.layout
                    .open_digest(BlobKind::Object, record.digest, record.length)?;
            verify_object(
                &object_fd,
                record,
                &table,
                control,
                self.config.copy_buffer_bytes,
            )?;
        }
        Ok(())
    }

    fn plan_unreferenced(
        &self,
        kind: BlobKind,
        entries: &[LeafEntry],
        live: &BTreeSet<Digest>,
        control: &Control,
        output: &mut Vec<DeleteCandidate>,
    ) -> Result<(), StoreError> {
        for entry in entries {
            control.check()?;
            let digest = parse_digest_name(&entry.name, kind.error_name())?;
            let fd = self.layout.open_digest(kind, digest, entry.length)?;
            verify_digest(
                &fd,
                digest,
                entry.length,
                kind.error_name(),
                control,
                self.config.copy_buffer_bytes,
            )?;
            if !live.contains(&digest) {
                output.push(DeleteCandidate {
                    kind: DeleteKind::Digest(kind),
                    name: entry.name.clone(),
                    device: entry.device,
                    inode: entry.inode,
                    reclaim_bytes: 0,
                    allocated_bytes: entry.charged_bytes,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Publication {
    Published,
    Reused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DeleteKind {
    Digest(BlobKind),
    Stage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeleteCandidate {
    kind: DeleteKind,
    name: String,
    device: u64,
    inode: u64,
    reclaim_bytes: u64,
    allocated_bytes: u64,
}

fn assign_reclaim_bytes(
    candidates: &mut [DeleteCandidate],
    manifests: &[LeafEntry],
    objects: &[LeafEntry],
    page_tables: &[LeafEntry],
    stages: &[LeafEntry],
) -> Result<(), StoreError> {
    let candidate_names: BTreeSet<(DeleteKind, &str)> = candidates
        .iter()
        .map(|candidate| (candidate.kind, candidate.name.as_str()))
        .collect();
    let mut retained = BTreeSet::new();
    retained.extend(manifests.iter().map(|entry| (entry.device, entry.inode)));
    retained.extend(
        objects
            .iter()
            .filter(|entry| {
                !candidate_names.contains(&(DeleteKind::Digest(BlobKind::Object), &*entry.name))
            })
            .map(|entry| (entry.device, entry.inode)),
    );
    retained.extend(
        page_tables
            .iter()
            .filter(|entry| {
                !candidate_names.contains(&(DeleteKind::Digest(BlobKind::PageTable), &*entry.name))
            })
            .map(|entry| (entry.device, entry.inode)),
    );
    retained.extend(
        stages
            .iter()
            .filter(|entry| !candidate_names.contains(&(DeleteKind::Stage, &*entry.name)))
            .map(|entry| (entry.device, entry.inode)),
    );

    let mut final_candidate = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let identity = (candidate.device, candidate.inode);
        if !retained.contains(&identity) {
            final_candidate.insert(identity, index);
        }
    }
    for (identity, index) in final_candidate {
        let allocated = candidates[index].allocated_bytes;
        if candidates
            .iter()
            .filter(|candidate| (candidate.device, candidate.inode) == identity)
            .any(|candidate| candidate.allocated_bytes != allocated)
        {
            return Err(StoreError::Invariant {
                problem: "hard links reported inconsistent allocated bytes",
            });
        }
        candidates[index].reclaim_bytes = allocated;
    }
    Ok(())
}

fn account_publication(
    publication: Publication,
    length: u64,
    published_blobs: &mut u64,
    reused_blobs: &mut u64,
    published_bytes: &mut u64,
) -> Result<(), StoreError> {
    match publication {
        Publication::Published => {
            *published_blobs =
                published_blobs
                    .checked_add(1)
                    .ok_or(StoreError::ArithmeticOverflow {
                        context: "counting published blobs",
                    })?;
            *published_bytes =
                published_bytes
                    .checked_add(length)
                    .ok_or(StoreError::ArithmeticOverflow {
                        context: "counting published bytes",
                    })?;
        }
        Publication::Reused => {
            *reused_blobs = reused_blobs
                .checked_add(1)
                .ok_or(StoreError::ArithmeticOverflow {
                    context: "counting reused blobs",
                })?;
        }
    }
    Ok(())
}

fn verify_digest(
    fd: &impl AsFd,
    expected_digest: Digest,
    expected_length: u64,
    kind: &'static str,
    control: &Control,
    buffer_bytes: usize,
) -> Result<(), StoreError> {
    let initial_length = fs::metadata_len_regular(fd, kind)?;
    if initial_length != expected_length {
        return Err(StoreError::LengthMismatch {
            kind,
            expected: expected_length,
            actual: initial_length,
        });
    }
    let mut hasher = Sha256::new();
    let mut offset = 0_u64;
    while offset < expected_length {
        control.check()?;
        let length = (expected_length - offset).min(buffer_bytes as u64);
        let bytes = fs::read_exact_at(fd, offset, length, kind, control)?;
        hasher.update(bytes);
        offset = offset
            .checked_add(length)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "hashing retained bytes",
            })?;
    }
    let final_length = fs::metadata_len_regular(fd, kind)?;
    if final_length != expected_length {
        return Err(StoreError::LengthMismatch {
            kind,
            expected: expected_length,
            actual: final_length,
        });
    }
    if Digest::from_bytes(hasher.finalize().into()) != expected_digest {
        return Err(StoreError::Integrity { kind });
    }
    Ok(())
}

fn verify_object(
    fd: &impl AsFd,
    record: &ObjectRecord,
    table: &PageTable,
    control: &Control,
    buffer_bytes: usize,
) -> Result<(), StoreError> {
    if table.object_digest() != record.digest
        || table.object_length() != record.length
        || table.page_size() != record.page_size
    {
        return Err(StoreError::Integrity { kind: "page table" });
    }
    verify_digest(
        fd,
        record.digest,
        record.length,
        "object",
        control,
        buffer_bytes,
    )?;
    let page_size = u64::from(record.page_size);
    for page_index in 0..table.page_count() {
        control.check()?;
        let offset = page_index
            .checked_mul(page_size)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "computing an object page offset",
            })?;
        let length = record.length.saturating_sub(offset).min(page_size);
        let bytes = fs::read_exact_at(fd, offset, length, "object page", control)?;
        let actual = *digest_bytes_interruptible(&bytes, control, buffer_bytes)?.as_bytes();
        if table.page_hash(page_index) != Some(actual) {
            return Err(StoreError::Integrity {
                kind: "object page",
            });
        }
    }
    Ok(())
}

fn parse_digest_name(name: &str, kind: &'static str) -> Result<Digest, StoreError> {
    format!("sha256:{name}")
        .parse()
        .map_err(|_| StoreError::UnsafeLayout { kind })
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn digest_bytes_interruptible(
    bytes: &[u8],
    control: &Control,
    buffer_bytes: usize,
) -> Result<Digest, StoreError> {
    let mut hasher = Sha256::new();
    for chunk in bytes.chunks(buffer_bytes) {
        control.check()?;
        hasher.update(chunk);
    }
    control.check()?;
    Ok(Digest::from_bytes(hasher.finalize().into()))
}

fn manifest_length_for_open(layout: &Layout, artifact_id: Digest) -> Result<u64, StoreError> {
    let name = artifact_id.path_component();
    layout
        .digest_entries(BlobKind::Manifest)?
        .into_iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.length)
        .ok_or(StoreError::Missing { kind: "manifest" })
}

#[cfg(test)]
mod publication_tests {
    use std::fs::{self, OpenOptions};
    use std::io::Write as _;
    use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};

    use runnel_fixture::FixtureArtifact;
    use runnel_format::{Digest, Limits};
    use tempfile::tempdir;

    use super::{
        BlobKind, Cas, CasConfig, GcPolicy, TestLinkFault, TestRenameFault, TestSyncTarget,
    };
    use crate::stage::ResumeToken;
    use crate::{ArtifactSource, CancellationToken, Control, DiskBudget, StoreError};

    fn fixture_and_source() -> (tempfile::TempDir, FixtureArtifact, ArtifactSource) {
        let temporary = tempdir().unwrap();
        let fixture = FixtureArtifact::build();
        fixture.write_new(temporary.path().join("source")).unwrap();
        let source = ArtifactSource::open(temporary.path().join("source")).unwrap();
        (temporary, fixture, source)
    }

    fn config(max_bytes: u64) -> CasConfig {
        CasConfig {
            disk_budget: DiskBudget::new(max_bytes, 0),
            copy_buffer_bytes: 4_096,
        }
    }

    fn exact_fixture_charge(cas: &Cas, fixture: &FixtureArtifact) -> u64 {
        let usage = cas.usage().unwrap();
        let unit = usage.allocation_unit;
        let round = |length: u64| length.div_ceil(unit) * unit;
        let identity = fixture.identity();
        round(fixture.manifest_bytes().len() as u64)
            + round(identity.page_table_length)
            + round(identity.object_length)
    }

    fn write_private(path: &std::path::Path, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
    }

    fn write_object_orphan(cas_root: &std::path::Path, bytes: &[u8]) -> std::path::PathBuf {
        let digest = Digest::of(bytes);
        let path = cas_root
            .join("objects/sha256")
            .join(digest.path_component());
        write_private(&path, bytes);
        path
    }

    fn imported_cas() -> (tempfile::TempDir, Cas, std::path::PathBuf) {
        let (temporary, fixture, source) = fixture_and_source();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        cas.import(
            &source,
            fixture.identity().artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
        (temporary, cas, cas_root)
    }

    #[test]
    fn committed_retry_reconfirms_manifest_directory_last() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas = Cas::create(temporary.path().join("cas"), config(1024 * 1024)).unwrap();
        cas.import(
            &source,
            fixture.identity().artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
        cas.publication_faults.take_sync_events();
        cas.publication_faults
            .fail_digest_sync(BlobKind::Manifest, 1);

        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "manifest" }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::Digest(BlobKind::PageTable),
                TestSyncTarget::Digest(BlobKind::Object),
                TestSyncTarget::Staging,
                TestSyncTarget::Digest(BlobKind::Manifest),
            ]
        );

        cas.publication_faults.fail_staging_sync(1);
        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "manifest" }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::Digest(BlobKind::PageTable),
                TestSyncTarget::Digest(BlobKind::Object),
                TestSyncTarget::Staging,
            ]
        );

        let receipt = cas
            .import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(receipt.reused_blobs, 3);
    }

    #[test]
    fn verified_stage_file_fsync_failure_is_resumable_and_publication_retry_is_unambiguous() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        let identity = fixture.identity();
        cas.publication_faults
            .fail_stage_file_sync(BlobKind::PageTable, 1);

        assert_eq!(
            cas.import(
                &source,
                identity.artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::Io {
                operation: "synchronizing injected staging file",
                kind: std::io::ErrorKind::Other,
            }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::Staging,
                TestSyncTarget::StageFile(BlobKind::PageTable),
            ]
        );
        assert!(
            !cas_root
                .join("page-tables/sha256")
                .join(identity.page_table_digest.path_component())
                .exists()
        );
        let resumes = cas.resume_tokens().unwrap();
        assert_eq!(resumes.len(), 1);

        // The retry publishes the already verified stage. A failure syncing
        // the now-modified staging directory cannot pretend publication did
        // not happen, and a further idempotent retry converges.
        cas.publication_faults.fail_staging_sync(1);
        assert_eq!(
            cas.import_with_resumes(
                &source,
                identity.artifact_id,
                Limits::default(),
                &resumes,
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "page-table" }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::StageFile(BlobKind::PageTable),
                TestSyncTarget::Digest(BlobKind::PageTable),
                TestSyncTarget::Staging,
            ]
        );
        assert!(
            cas_root
                .join("page-tables/sha256")
                .join(identity.page_table_digest.path_component())
                .exists()
        );
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);

        let receipt = cas
            .import(
                &source,
                identity.artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(receipt.reused_blobs, 1);
        assert_eq!(receipt.published_blobs, 2);
    }

    #[test]
    fn failed_stage_creation_sync_is_cleaned_and_resynced_without_masking_the_error() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        cas.publication_faults.fail_staging_sync(1);

        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::Io {
                operation: "synchronizing injected staging directory",
                kind: std::io::ErrorKind::Other,
            }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![TestSyncTarget::Staging, TestSyncTarget::Staging]
        );
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);
        assert_eq!(
            fs::read_dir(cas_root.join("page-tables/sha256"))
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn fallback_link_rechecks_cancellation_after_unsupported_rename() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        let cancellation = CancellationToken::new();
        cas.publication_faults.push_rename(
            BlobKind::PageTable,
            TestRenameFault::Unsupported {
                cancel: Some(cancellation.clone()),
            },
        );

        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::with_cancellation(cancellation),
            )
            .unwrap_err(),
            StoreError::Cancelled
        );
        assert_eq!(
            fs::read_dir(cas_root.join("page-tables/sha256"))
                .unwrap()
                .count(),
            0
        );
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 1);
    }

    #[test]
    fn ambiguous_rename_with_verified_final_is_successful_publication() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        cas.publication_faults
            .push_rename(BlobKind::PageTable, TestRenameFault::AmbiguousPublished);

        let receipt = cas
            .import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(receipt.published_blobs, 3);
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);
    }

    #[test]
    fn ambiguous_fallback_link_with_verified_final_is_successful_publication() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        cas.publication_faults.push_rename(
            BlobKind::PageTable,
            TestRenameFault::Unsupported { cancel: None },
        );
        cas.publication_faults
            .push_link(BlobKind::PageTable, TestLinkFault::AmbiguousPublished);

        let receipt = cas
            .import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(receipt.published_blobs, 3);
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);
    }

    #[test]
    fn hard_link_cleanup_failure_retains_synced_alias_without_failing_publication() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let bootstrap = Cas::create(&cas_root, config(u64::MAX)).unwrap();
        let exact_budget = exact_fixture_charge(&bootstrap, &fixture);
        drop(bootstrap);
        let cas = Cas::open(&cas_root, config(exact_budget)).unwrap();
        cas.publication_faults.push_rename(
            BlobKind::PageTable,
            TestRenameFault::Unsupported { cancel: None },
        );
        cas.publication_faults
            .fail_link_cleanup(BlobKind::PageTable, 1);

        let receipt = cas
            .import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(receipt.published_blobs, 3);
        let final_path = cas_root
            .join("page-tables/sha256")
            .join(fixture.identity().page_table_digest.path_component());
        let stage_path = fs::read_dir(cas_root.join("staging"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert_eq!(
            fs::metadata(&final_path).unwrap().ino(),
            fs::metadata(&stage_path).unwrap().ino()
        );
        let usage = cas.usage().unwrap();
        assert_eq!(usage.staging_bytes, 0);
        let retained = cas
            .gc(
                GcPolicy {
                    dry_run: true,
                    remove_staging: false,
                },
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(retained.candidate_blobs, 0);
        assert_eq!(retained.candidate_bytes, 0);

        let removable_alias = cas
            .gc(
                GcPolicy {
                    dry_run: true,
                    remove_staging: true,
                },
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(removable_alias.candidate_blobs, 1);
        assert_eq!(removable_alias.candidate_bytes, 0);

        let retry = cas
            .import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(retry.reused_blobs, 3);
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);
    }

    #[test]
    fn failed_ambiguous_link_sync_retains_resumable_stage_and_retry_converges() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        let identity = fixture.identity();
        cas.publication_faults.push_rename(
            BlobKind::PageTable,
            TestRenameFault::Unsupported { cancel: None },
        );
        cas.publication_faults
            .push_link(BlobKind::PageTable, TestLinkFault::AmbiguousPublished);
        cas.publication_faults
            .fail_digest_sync(BlobKind::PageTable, 1);

        assert_eq!(
            cas.import(
                &source,
                identity.artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "page-table" }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::Staging,
                TestSyncTarget::StageFile(BlobKind::PageTable),
                TestSyncTarget::Digest(BlobKind::PageTable),
            ]
        );

        let final_path = cas_root
            .join("page-tables/sha256")
            .join(identity.page_table_digest.path_component());
        let resumes = cas.resume_tokens().unwrap();
        assert_eq!(resumes.len(), 1);
        let stage_path = cas_root.join("staging").join(resumes[0].stage_name());
        assert_eq!(
            fs::metadata(final_path).unwrap().ino(),
            fs::metadata(stage_path).unwrap().ino()
        );

        let retry = cas
            .import_with_resumes(
                &source,
                identity.artifact_id,
                Limits::default(),
                &resumes,
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(retry.reused_blobs, 1);
        assert_eq!(retry.published_blobs, 2);
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);
    }

    #[test]
    fn stale_hard_link_alias_with_failed_directory_sync_reports_durability_uncertainty() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        cas.publication_faults.push_rename(
            BlobKind::PageTable,
            TestRenameFault::Unsupported { cancel: None },
        );
        cas.publication_faults
            .fail_link_cleanup(BlobKind::PageTable, 1);
        cas.publication_faults
            .fail_digest_sync(BlobKind::PageTable, 1);

        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "page-table" }
        );
        let final_path = cas_root
            .join("page-tables/sha256")
            .join(fixture.identity().page_table_digest.path_component());
        let stage_path = fs::read_dir(cas_root.join("staging"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert_eq!(
            fs::metadata(final_path).unwrap().ino(),
            fs::metadata(stage_path).unwrap().ino()
        );

        let retry = cas
            .import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(retry.reused_blobs, 1);
        assert_eq!(retry.published_blobs, 2);
    }

    #[test]
    fn committed_retry_never_syncs_manifest_after_an_earlier_sync_failure() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas = Cas::create(temporary.path().join("cas"), config(1024 * 1024)).unwrap();
        cas.import(
            &source,
            fixture.identity().artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
        cas.publication_faults.take_sync_events();

        cas.publication_faults
            .fail_digest_sync(BlobKind::PageTable, 1);
        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "manifest" }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![TestSyncTarget::Digest(BlobKind::PageTable)]
        );

        cas.publication_faults.fail_digest_sync(BlobKind::Object, 1);
        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "manifest" }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::Digest(BlobKind::PageTable),
                TestSyncTarget::Digest(BlobKind::Object),
            ]
        );

        cas.publication_faults.fail_staging_sync(1);
        assert_eq!(
            cas.import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::PublishedButDurabilityUnconfirmed { kind: "manifest" }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::Digest(BlobKind::PageTable),
                TestSyncTarget::Digest(BlobKind::Object),
                TestSyncTarget::Staging,
            ]
        );

        let retry = cas
            .import(
                &source,
                fixture.identity().artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(retry.reused_blobs, 3);
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![
                TestSyncTarget::Digest(BlobKind::PageTable),
                TestSyncTarget::Digest(BlobKind::Object),
                TestSyncTarget::Staging,
                TestSyncTarget::Digest(BlobKind::Manifest),
            ]
        );
    }

    #[test]
    fn partial_redundant_stage_cleanup_syncs_prior_unlinks_and_preserves_first_error() {
        let (_source_root, fixture, source) = fixture_and_source();
        let temporary = tempdir().unwrap();
        let cas_root = temporary.path().join("cas");
        let cas = Cas::create(&cas_root, config(1024 * 1024)).unwrap();
        cas.import(
            &source,
            fixture.identity().artifact_id,
            Limits::default(),
            &Control::unbounded(),
        )
        .unwrap();
        cas.publication_faults.take_sync_events();

        let identity = fixture.identity();
        for random in [
            "00000000000000000000000000000000",
            "ffffffffffffffffffffffffffffffff",
        ] {
            let token = ResumeToken::new(
                BlobKind::PageTable,
                identity.page_table_digest,
                identity.page_table_length,
                random,
            )
            .unwrap();
            write_private(&cas_root.join("staging").join(token.stage_name()), b"stage");
        }
        cas.publication_faults
            .fail_stage_cleanup_after(BlobKind::PageTable, 1);

        assert_eq!(
            cas.import(
                &source,
                identity.artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::Io {
                operation: "unlinking injected redundant stage",
                kind: std::io::ErrorKind::Other,
            }
        );
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![TestSyncTarget::Staging]
        );
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 1);

        let retry = cas
            .import(
                &source,
                identity.artifact_id,
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap();
        assert_eq!(retry.reused_blobs, 3);
        assert_eq!(fs::read_dir(cas_root.join("staging")).unwrap().count(), 0);
    }

    #[test]
    fn cancellation_after_one_gc_unlink_syncs_the_partial_sweep() {
        let (_temporary, cas, cas_root) = imported_cas();
        let paths = [
            write_object_orphan(&cas_root, b"partial GC cancellation orphan A"),
            write_object_orphan(&cas_root, b"partial GC cancellation orphan B"),
        ];
        cas.publication_faults.take_sync_events();
        let cancellation = CancellationToken::new();
        cas.publication_faults
            .cancel_gc_after_unlinks(1, cancellation.clone());

        assert_eq!(
            cas.gc(
                GcPolicy::default(),
                Limits::default(),
                &Control::with_cancellation(cancellation),
            )
            .unwrap_err(),
            StoreError::Cancelled
        );
        assert_eq!(paths.iter().filter(|path| path.exists()).count(), 1);
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![TestSyncTarget::Digest(BlobKind::Object)]
        );
    }

    #[test]
    fn gc_unlink_failure_after_progress_syncs_prior_deletion() {
        let (_temporary, cas, cas_root) = imported_cas();
        let paths = [
            write_object_orphan(&cas_root, b"partial GC unlink failure orphan A"),
            write_object_orphan(&cas_root, b"partial GC unlink failure orphan B"),
        ];
        cas.publication_faults.take_sync_events();
        cas.publication_faults
            .fail_gc_unlink_after(BlobKind::Object, 1);

        assert_eq!(
            cas.gc(
                GcPolicy::default(),
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::Io {
                operation: "unlinking injected garbage-collection candidate",
                kind: std::io::ErrorKind::Other,
            }
        );
        assert_eq!(paths.iter().filter(|path| path.exists()).count(), 1);
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![TestSyncTarget::Digest(BlobKind::Object)]
        );
    }

    #[test]
    fn gc_directory_sync_failure_is_reported_after_safe_deletion() {
        let (_temporary, cas, cas_root) = imported_cas();
        let path = write_object_orphan(&cas_root, b"GC directory sync failure orphan");
        cas.publication_faults.take_sync_events();
        cas.publication_faults.fail_digest_sync(BlobKind::Object, 1);

        assert_eq!(
            cas.gc(
                GcPolicy::default(),
                Limits::default(),
                &Control::unbounded(),
            )
            .unwrap_err(),
            StoreError::Io {
                operation: "synchronizing injected digest directory",
                kind: std::io::ErrorKind::Other,
            }
        );
        assert!(!path.exists());
        assert_eq!(
            cas.publication_faults.take_sync_events(),
            vec![TestSyncTarget::Digest(BlobKind::Object)]
        );
    }
}
