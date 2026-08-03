use std::collections::BTreeSet;
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;

use rustix::fs::{Dir, FileType, FlockOperation, Mode, OFlags, fstat, openat};

use crate::budget::DiskUsage;
use crate::fs::{
    self, create_new_leaf, mkdir_child, open_child_dir, open_directory_path, open_regular_leaf,
};
use crate::stage::{BlobKind, ResumeToken};
use crate::{Control, StoreError};

const MAX_DIRECTORY_ENTRIES: usize = 131_072;
const DIRECTORY_MODE: Mode = Mode::RWXU;
const FILE_MODE: Mode = Mode::RUSR.union(Mode::WUSR);

#[derive(Debug)]
pub(crate) struct Layout {
    pub(crate) root: OwnedFd,
    pub(crate) lock: OwnedFd,
    manifests_parent: OwnedFd,
    page_tables_parent: OwnedFd,
    objects_parent: OwnedFd,
    pub(crate) manifests: OwnedFd,
    pub(crate) page_tables: OwnedFd,
    pub(crate) objects: OwnedFd,
    pub(crate) staging: OwnedFd,
    root_device: u64,
    root_owner: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LeafEntry {
    pub(crate) name: String,
    pub(crate) length: u64,
    pub(crate) charged_bytes: u64,
    pub(crate) device: u64,
    pub(crate) inode: u64,
}

impl Layout {
    pub(crate) fn create(path: &Path) -> Result<Self, StoreError> {
        let leaf = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or(StoreError::UnsafeLayout { kind: "CAS root" })?;
        fs::validate_component(leaf, "CAS root")?;
        let parent_path = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());
        let parent =
            open_directory_path(parent_path.unwrap_or_else(|| Path::new(".")), "CAS parent")?;
        mkdir_child(&parent, leaf, "CAS root", DIRECTORY_MODE)?;
        fs::sync_fd(&parent)?;
        let root = open_child_dir(&parent, leaf, "CAS root")?;
        Self::initialize(root)
    }

    pub(crate) fn open(path: &Path) -> Result<Self, StoreError> {
        let root = open_directory_path(path, "CAS root")?;
        Self::open_root(root)
    }

    fn initialize(root: OwnedFd) -> Result<Self, StoreError> {
        ensure_empty_directory(&root, "new CAS root")?;
        for component in ["manifests", "page-tables", "objects", "staging"] {
            mkdir_child(&root, component, "CAS directory", DIRECTORY_MODE)?;
        }
        let manifests_parent = open_child_dir(&root, "manifests", "manifest parent")?;
        let page_tables_parent = open_child_dir(&root, "page-tables", "page-table parent")?;
        let objects_parent = open_child_dir(&root, "objects", "object parent")?;
        mkdir_child(
            &manifests_parent,
            "sha256",
            "manifest digest directory",
            DIRECTORY_MODE,
        )?;
        mkdir_child(
            &page_tables_parent,
            "sha256",
            "page-table digest directory",
            DIRECTORY_MODE,
        )?;
        mkdir_child(
            &objects_parent,
            "sha256",
            "object digest directory",
            DIRECTORY_MODE,
        )?;
        fs::sync_fd(&manifests_parent)?;
        fs::sync_fd(&page_tables_parent)?;
        fs::sync_fd(&objects_parent)?;
        fs::sync_fd(&root)?;
        let lock = create_new_leaf(&root, "transaction.lock", "transaction lock", FILE_MODE)?;
        fs::sync_fd(&lock)?;
        fs::sync_fd(&root)?;
        Self::open_with_lock(root, lock)
    }

    fn open_root(root: OwnedFd) -> Result<Self, StoreError> {
        require_names(
            &root,
            "CAS root",
            &[
                "manifests",
                "objects",
                "page-tables",
                "staging",
                "transaction.lock",
            ],
        )?;
        let lock = open_regular_leaf(&root, "transaction.lock", "transaction lock", Some(0))?;
        Self::open_with_lock(root, lock)
    }

    fn open_with_lock(root: OwnedFd, lock: OwnedFd) -> Result<Self, StoreError> {
        let root_stat =
            fstat(&root).map_err(|error| StoreError::io("reading CAS root metadata", &error))?;
        if FileType::from_raw_mode(root_stat.st_mode) != FileType::Directory {
            return Err(StoreError::UnsafeLayout { kind: "CAS root" });
        }
        require_effective_owner(root_stat.st_uid, "CAS root")?;
        let root_device = root_stat.st_dev;
        let root_owner = root_stat.st_uid;

        let manifests_parent = open_child_dir(&root, "manifests", "manifest parent")?;
        let page_tables_parent = open_child_dir(&root, "page-tables", "page-table parent")?;
        let objects_parent = open_child_dir(&root, "objects", "object parent")?;
        require_names(&manifests_parent, "manifest parent", &["sha256"])?;
        require_names(&page_tables_parent, "page-table parent", &["sha256"])?;
        require_names(&objects_parent, "object parent", &["sha256"])?;
        let manifests = open_child_dir(&manifests_parent, "sha256", "manifest directory")?;
        let page_tables = open_child_dir(&page_tables_parent, "sha256", "page-table directory")?;
        let objects = open_child_dir(&objects_parent, "sha256", "object directory")?;
        let staging = open_child_dir(&root, "staging", "staging directory")?;

        let layout = Self {
            root,
            lock,
            manifests_parent,
            page_tables_parent,
            objects_parent,
            manifests,
            page_tables,
            objects,
            staging,
            root_device,
            root_owner,
        };
        layout.validate_fixed_metadata()?;
        {
            // Leaf namespaces are mutable under the project transaction lock.
            // Validate their complete snapshot only after fixed descriptors
            // and the retained lock itself have been authenticated.
            let _transaction = layout.lock()?;
            layout.validate_all_entries()?;
        }
        Ok(layout)
    }

    fn validate_fixed_metadata(&self) -> Result<(), StoreError> {
        for (fd, kind, expected_type) in [
            (&self.root, "CAS root", FileType::Directory),
            (
                &self.manifests_parent,
                "manifest parent",
                FileType::Directory,
            ),
            (
                &self.page_tables_parent,
                "page-table parent",
                FileType::Directory,
            ),
            (&self.objects_parent, "object parent", FileType::Directory),
            (&self.manifests, "manifest directory", FileType::Directory),
            (
                &self.page_tables,
                "page-table directory",
                FileType::Directory,
            ),
            (&self.objects, "object directory", FileType::Directory),
            (&self.staging, "staging directory", FileType::Directory),
            (&self.lock, "transaction lock", FileType::RegularFile),
        ] {
            self.validate_metadata(fd, kind, expected_type)?;
        }
        Ok(())
    }

    fn validate_metadata(
        &self,
        fd: &impl AsFd,
        kind: &'static str,
        expected_type: FileType,
    ) -> Result<rustix::fs::Stat, StoreError> {
        let stat = fstat(fd).map_err(|error| StoreError::io("reading CAS metadata", &error))?;
        if FileType::from_raw_mode(stat.st_mode) != expected_type
            || stat.st_dev != self.root_device
            || stat.st_uid != self.root_owner
            || stat.st_mode & 0o022 != 0
        {
            return Err(StoreError::UnsafeLayout { kind });
        }
        if expected_type == FileType::RegularFile && stat.st_mode & 0o777 != 0o600 {
            return Err(StoreError::UnsafeLayout { kind });
        }
        Ok(stat)
    }

    fn validate_all_entries(&self) -> Result<(), StoreError> {
        self.digest_entries(BlobKind::Manifest)?;
        self.digest_entries(BlobKind::PageTable)?;
        self.digest_entries(BlobKind::Object)?;
        self.stage_entries()?;
        Ok(())
    }

    pub(crate) fn digest_dir(&self, kind: BlobKind) -> &OwnedFd {
        match kind {
            BlobKind::Manifest => &self.manifests,
            BlobKind::PageTable => &self.page_tables,
            BlobKind::Object => &self.objects,
        }
    }

    pub(crate) fn open_digest(
        &self,
        kind: BlobKind,
        digest: runnel_format::Digest,
        expected_length: u64,
    ) -> Result<OwnedFd, StoreError> {
        let fd = open_regular_leaf(
            self.digest_dir(kind),
            &digest.path_component(),
            kind.error_name(),
            Some(expected_length),
        )?;
        self.validate_metadata(&fd, kind.error_name(), FileType::RegularFile)?;
        Ok(fd)
    }

    pub(crate) fn create_stage(&self, token: &ResumeToken) -> Result<OwnedFd, StoreError> {
        let fd = create_new_leaf(&self.staging, token.stage_name(), "staging file", FILE_MODE)?;
        self.validate_metadata(&fd, "staging file", FileType::RegularFile)?;
        Ok(fd)
    }

    pub(crate) fn open_stage(&self, token: &ResumeToken) -> Result<OwnedFd, StoreError> {
        let fd = openat(
            &self.staging,
            token.stage_name(),
            OFlags::RDWR | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(|error| match error {
            rustix::io::Errno::NOENT => StoreError::Missing {
                kind: "staging file",
            },
            rustix::io::Errno::LOOP => StoreError::UnsafeLayout {
                kind: "staging file",
            },
            _ => StoreError::io("opening staged data", &error),
        })?;
        self.validate_metadata(&fd, "staging file", FileType::RegularFile)?;
        Ok(fd)
    }

    pub(crate) fn unlink_stage(&self, token: &ResumeToken) -> Result<(), StoreError> {
        fs::unlink_leaf(&self.staging, token.stage_name(), "staging file")
    }

    pub(crate) fn sync_staging(&self) -> Result<(), StoreError> {
        fs::sync_fd(&self.staging)
    }

    pub(crate) fn sync_digest_dir(&self, kind: BlobKind) -> Result<(), StoreError> {
        fs::sync_fd(self.digest_dir(kind))
    }

    pub(crate) fn digest_entries(&self, kind: BlobKind) -> Result<Vec<LeafEntry>, StoreError> {
        let directory = self.digest_dir(kind);
        let names = read_names(directory, kind.error_name())?;
        let mut entries = Vec::with_capacity(names.len());
        for name in names {
            if !is_digest_name(&name) {
                return Err(StoreError::UnsafeLayout {
                    kind: kind.error_name(),
                });
            }
            entries.push(self.open_entry(directory, name, kind.error_name(), false)?);
        }
        Ok(entries)
    }

    pub(crate) fn stage_entries(&self) -> Result<Vec<LeafEntry>, StoreError> {
        let names = read_names(&self.staging, "staging directory")?;
        let mut entries = Vec::with_capacity(names.len());
        for name in names {
            ResumeToken::from_stage_name(&name).map_err(|_| StoreError::UnsafeLayout {
                kind: "staging file",
            })?;
            entries.push(self.open_entry(&self.staging, name, "staging file", true)?);
        }
        Ok(entries)
    }

    fn open_entry(
        &self,
        directory: &impl AsFd,
        name: String,
        kind: &'static str,
        writable: bool,
    ) -> Result<LeafEntry, StoreError> {
        let flags = (if writable {
            OFlags::RDWR
        } else {
            OFlags::RDONLY
        }) | OFlags::NOFOLLOW
            | OFlags::CLOEXEC
            | OFlags::NONBLOCK;
        let fd = openat(directory, name.as_str(), flags, Mode::empty()).map_err(
            |error| match error {
                rustix::io::Errno::LOOP | rustix::io::Errno::NOENT => {
                    StoreError::UnsafeLayout { kind }
                }
                _ => StoreError::io("opening CAS entry", &error),
            },
        )?;
        let stat = self.validate_metadata(&fd, kind, FileType::RegularFile)?;
        let length = u64::try_from(stat.st_size).map_err(|_| StoreError::UnsafeLayout { kind })?;
        let blocks =
            u64::try_from(stat.st_blocks).map_err(|_| StoreError::UnsafeLayout { kind })?;
        let charged_bytes = blocks
            .checked_mul(512)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "accounting allocated CAS blocks",
            })?;
        Ok(LeafEntry {
            name,
            length,
            charged_bytes,
            device: stat.st_dev,
            inode: stat.st_ino,
        })
    }

    pub(crate) fn usage(&self) -> Result<DiskUsage, StoreError> {
        let mut charged_inodes = BTreeSet::new();
        let mut final_bytes = 0_u64;
        for kind in [BlobKind::Manifest, BlobKind::PageTable, BlobKind::Object] {
            for entry in self.digest_entries(kind)? {
                if charged_inodes.insert((entry.device, entry.inode)) {
                    final_bytes = final_bytes.checked_add(entry.charged_bytes).ok_or(
                        StoreError::ArithmeticOverflow {
                            context: "summing final CAS bytes",
                        },
                    )?;
                }
            }
        }
        let mut staging_bytes = 0_u64;
        for entry in self.stage_entries()? {
            if charged_inodes.insert((entry.device, entry.inode)) {
                staging_bytes = staging_bytes.checked_add(entry.charged_bytes).ok_or(
                    StoreError::ArithmeticOverflow {
                        context: "summing staged CAS bytes",
                    },
                )?;
            }
        }
        let (available_bytes, allocation_unit) = fs::available_bytes(&self.root)?;
        let usage = DiskUsage {
            final_bytes,
            staging_bytes,
            available_bytes,
            allocation_unit,
        };
        usage.checked_total_bytes()?;
        Ok(usage)
    }

    pub(crate) fn same_file(
        &self,
        left: &impl AsFd,
        right: &impl AsFd,
    ) -> Result<bool, StoreError> {
        let left = fstat(left)
            .map_err(|error| StoreError::io("reading retained file identity", &error))?;
        let right = fstat(right)
            .map_err(|error| StoreError::io("reading retained file identity", &error))?;
        Ok(left.st_dev == right.st_dev && left.st_ino == right.st_ino)
    }

    pub(crate) fn lock(&self) -> Result<TransactionLock<'_>, StoreError> {
        fs::lock_exclusive(&self.lock)?;
        Ok(TransactionLock { fd: &self.lock })
    }

    pub(crate) fn lock_with_control(
        &self,
        control: &Control,
    ) -> Result<TransactionLock<'_>, StoreError> {
        loop {
            control.check()?;
            match rustix::fs::flock(&self.lock, FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => return Ok(TransactionLock { fd: &self.lock }),
                Err(rustix::io::Errno::AGAIN) => std::thread::yield_now(),
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => {
                    return Err(StoreError::io("locking the transaction descriptor", &error));
                }
            }
        }
    }
}

pub(crate) struct TransactionLock<'a> {
    fd: &'a OwnedFd,
}

impl Drop for TransactionLock<'_> {
    fn drop(&mut self) {
        let _ = rustix::fs::flock(self.fd, FlockOperation::Unlock);
    }
}

fn ensure_empty_directory(directory: &impl AsFd, kind: &'static str) -> Result<(), StoreError> {
    if read_names(directory, kind)?.is_empty() {
        Ok(())
    } else {
        Err(StoreError::UnsafeLayout { kind })
    }
}

fn require_names(
    directory: &impl AsFd,
    kind: &'static str,
    expected: &[&str],
) -> Result<(), StoreError> {
    let actual: BTreeSet<String> = read_names(directory, kind)?.into_iter().collect();
    let expected: BTreeSet<String> = expected.iter().map(|value| (*value).to_owned()).collect();
    if actual == expected {
        Ok(())
    } else {
        Err(StoreError::UnsafeLayout { kind })
    }
}

fn read_names(directory: &impl AsFd, kind: &'static str) -> Result<Vec<String>, StoreError> {
    let mut reader = Dir::read_from(directory)
        .map_err(|error| StoreError::io("opening retained directory stream", &error))?;
    let mut names = Vec::new();
    for entry in &mut reader {
        let entry = entry.map_err(|error| StoreError::io("reading retained directory", &error))?;
        let name = entry
            .file_name()
            .to_str()
            .map_err(|_| StoreError::UnsafeLayout { kind })?;
        if matches!(name, "." | "..") {
            continue;
        }
        if names.len() == MAX_DIRECTORY_ENTRIES {
            return Err(StoreError::ResourceExhausted {
                resource: "directory entries",
                required: MAX_DIRECTORY_ENTRIES as u64 + 1,
                limit: MAX_DIRECTORY_ENTRIES as u64,
            });
        }
        names.push(name.to_owned());
    }
    names.sort_unstable();
    Ok(names)
}

fn is_digest_name(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_effective_owner(owner: u32, kind: &'static str) -> Result<(), StoreError> {
    if owner == rustix::process::geteuid().as_raw() {
        Ok(())
    } else {
        Err(StoreError::UnsafeLayout { kind })
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use rustix::fs::FlockOperation;
    use tempfile::tempdir;

    use super::{Layout, require_effective_owner};
    use crate::StoreError;

    #[test]
    fn create_and_open_exact_layout() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("cas");
        let created = Layout::create(&path).unwrap();
        assert_eq!(created.usage().unwrap().total_bytes(), 0);
        drop(created);
        let opened = Layout::open(&path).unwrap();
        assert_eq!(opened.usage().unwrap().total_bytes(), 0);
    }

    #[test]
    fn create_refuses_existing_root() {
        let temporary = tempdir().unwrap();
        let error = Layout::create(temporary.path()).unwrap_err();
        assert_eq!(error, StoreError::AlreadyExists { kind: "CAS root" });
    }

    #[cfg(unix)]
    #[test]
    fn open_rejects_internal_symlink() {
        use std::os::unix::fs::symlink;

        let temporary = tempdir().unwrap();
        let path = temporary.path().join("cas");
        drop(Layout::create(&path).unwrap());
        fs::remove_dir(path.join("objects/sha256")).unwrap();
        symlink("../manifests/sha256", path.join("objects/sha256")).unwrap();
        assert_eq!(
            Layout::open(&path).unwrap_err(),
            StoreError::UnsafeLayout {
                kind: "object directory"
            }
        );
    }

    #[test]
    fn unexpected_root_entry_is_rejected() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("cas");
        drop(Layout::create(&path).unwrap());
        fs::write(path.join("surprise"), b"unexpected").unwrap();
        assert_eq!(
            Layout::open(&path).unwrap_err(),
            StoreError::UnsafeLayout { kind: "CAS root" }
        );
    }

    #[cfg(unix)]
    #[test]
    fn open_rejects_writable_structural_parent() {
        use std::os::unix::fs::PermissionsExt as _;

        let temporary = tempdir().unwrap();
        let path = temporary.path().join("cas");
        drop(Layout::create(&path).unwrap());
        fs::set_permissions(path.join("objects"), fs::Permissions::from_mode(0o770)).unwrap();
        assert_eq!(
            Layout::open(&path).unwrap_err(),
            StoreError::UnsafeLayout {
                kind: "object parent"
            }
        );
    }

    #[test]
    fn owner_check_is_bound_to_the_effective_process_uid() {
        let effective = rustix::process::geteuid().as_raw();
        require_effective_owner(effective, "CAS root").unwrap();
        assert_eq!(
            require_effective_owner(effective.wrapping_add(1), "CAS root").unwrap_err(),
            StoreError::UnsafeLayout { kind: "CAS root" }
        );
    }

    #[test]
    fn open_waits_for_transaction_lock_before_scanning_mutable_namespaces() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("cas");
        drop(Layout::create(&path).unwrap());
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.join("transaction.lock"))
            .unwrap();
        rustix::fs::flock(&lock, FlockOperation::LockExclusive).unwrap();

        let (started_sender, started_receiver) = mpsc::channel();
        let (result_sender, result_receiver) = mpsc::channel();
        let open_path = path.clone();
        let opener = thread::spawn(move || {
            started_sender.send(()).unwrap();
            result_sender.send(Layout::open(&open_path)).unwrap();
        });
        started_receiver.recv().unwrap();
        assert!(
            result_receiver
                .recv_timeout(Duration::from_millis(100))
                .is_err(),
            "layout open bypassed the held transaction lock"
        );

        rustix::fs::flock(&lock, FlockOperation::Unlock).unwrap();
        let opened = result_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("layout open completed after unlock")
            .expect("layout remained valid");
        drop(opened);
        opener.join().unwrap();
    }
}
