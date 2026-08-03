use std::fmt;
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;
use std::sync::Arc;

use runnel_format::{Digest, Limits};

use crate::fs::{
    FileIdentity, duplicate, ensure_directory, file_identity, metadata_len_regular, open_child_dir,
    open_directory_path, open_regular_leaf, read_bounded_at, require_safe_permissions,
    require_same_filesystem_owner,
};
use crate::{Control, StoreError, StoredArtifact};

/// A standalone RMOA directory anchored by retained descriptors.
///
/// Opening the source walks the caller path component by component, then
/// retains the root, manifest, object-digest directory, and page-table-digest
/// directory. Later path renames cannot redirect this handle.
#[derive(Clone)]
pub struct ArtifactSource {
    inner: Arc<SourceInner>,
}

struct SourceInner {
    root: OwnedFd,
    manifest: OwnedFd,
    objects: OwnedFd,
    page_tables: OwnedFd,
    root_identity: FileIdentity,
}

impl fmt::Debug for ArtifactSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtifactSource")
            .field("retained", &true)
            .finish_non_exhaustive()
    }
}

impl ArtifactSource {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, StoreError> {
        let root = open_directory_path(root.as_ref(), "artifact root")?;
        let root_identity = file_identity(&root, "artifact root")?;
        require_safe_permissions(root_identity, "artifact root")?;

        let objects_parent = open_child_dir(&root, "objects", "object directory")?;
        require_same_filesystem_owner(root_identity, &objects_parent, "object directory")?;
        let objects = open_child_dir(&objects_parent, "sha256", "object digest directory")?;
        require_same_filesystem_owner(root_identity, &objects, "object digest directory")?;

        let page_tables_parent = open_child_dir(&root, "page-tables", "page-table directory")?;
        require_same_filesystem_owner(root_identity, &page_tables_parent, "page-table directory")?;
        let page_tables =
            open_child_dir(&page_tables_parent, "sha256", "page-table digest directory")?;
        require_same_filesystem_owner(root_identity, &page_tables, "page-table digest directory")?;

        let manifest = open_regular_leaf(&root, "manifest.json", "manifest", None)?;
        require_same_filesystem_owner(root_identity, &manifest, "manifest")?;

        Ok(Self {
            inner: Arc::new(SourceInner {
                root,
                manifest,
                objects,
                page_tables,
                root_identity,
            }),
        })
    }

    /// Builds the same boundary from descriptors authenticated by the CAS
    /// layout, without reopening a pathname.
    pub(crate) fn from_retained(
        root: OwnedFd,
        manifest: OwnedFd,
        objects: OwnedFd,
        page_tables: OwnedFd,
    ) -> Result<Self, StoreError> {
        ensure_directory(&root, "artifact root")?;
        ensure_directory(&objects, "object digest directory")?;
        ensure_directory(&page_tables, "page-table digest directory")?;
        metadata_len_regular(&manifest, "manifest")?;
        let root_identity = file_identity(&root, "artifact root")?;
        require_safe_permissions(root_identity, "artifact root")?;
        require_same_filesystem_owner(root_identity, &manifest, "manifest")?;
        require_same_filesystem_owner(root_identity, &objects, "object digest directory")?;
        require_same_filesystem_owner(root_identity, &page_tables, "page-table digest directory")?;
        Ok(Self {
            inner: Arc::new(SourceInner {
                root,
                manifest,
                objects,
                page_tables,
                root_identity,
            }),
        })
    }

    /// Reads the retained manifest under an explicit byte ceiling.
    pub fn manifest_bytes(&self, maximum: usize, control: &Control) -> Result<Vec<u8>, StoreError> {
        read_bounded_at(&self.inner.manifest, maximum, "manifest", control)
    }

    /// Parses and opens all metadata while retaining object descriptors.
    pub fn open_artifact(
        &self,
        limits: Limits,
        control: &Control,
    ) -> Result<StoredArtifact, StoreError> {
        StoredArtifact::open_source(self, limits, control)
    }

    /// As [`Self::open_artifact`], while authenticating a caller-trusted ID.
    pub fn open_with_expected_id(
        &self,
        limits: Limits,
        expected: Digest,
        control: &Control,
    ) -> Result<StoredArtifact, StoreError> {
        StoredArtifact::open_source_with_expected_id(self, limits, expected, control)
    }

    pub(crate) fn open_object(
        &self,
        digest: Digest,
        expected_length: u64,
    ) -> Result<OwnedFd, StoreError> {
        let fd = open_regular_leaf(
            &self.inner.objects,
            &digest.path_component(),
            "object",
            Some(expected_length),
        )?;
        require_same_filesystem_owner(self.inner.root_identity, &fd, "object")?;
        Ok(fd)
    }

    pub(crate) fn open_page_table(
        &self,
        digest: Digest,
        expected_length: u64,
    ) -> Result<OwnedFd, StoreError> {
        let fd = open_regular_leaf(
            &self.inner.page_tables,
            &digest.path_component(),
            "page table",
            Some(expected_length),
        )?;
        require_same_filesystem_owner(self.inner.root_identity, &fd, "page table")?;
        Ok(fd)
    }

    #[allow(dead_code)]
    pub(crate) fn retained_root(&self) -> &impl AsFd {
        &self.inner.root
    }

    #[allow(dead_code)]
    pub(crate) fn retained_objects(&self) -> &impl AsFd {
        &self.inner.objects
    }

    #[allow(dead_code)]
    pub(crate) fn retained_page_tables(&self) -> &impl AsFd {
        &self.inner.page_tables
    }

    #[allow(dead_code)]
    pub(crate) fn duplicate_manifest(&self) -> Result<OwnedFd, StoreError> {
        duplicate(&self.inner.manifest)
    }
}
