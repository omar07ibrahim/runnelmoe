//! Descriptor-relative filesystem primitives for the storage trust boundary.

use std::ffi::OsStr;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Component, Path};
use std::sync::OnceLock;

use rustix::fs::{
    AtFlags, FileType, FlockOperation, Mode, OFlags, RenameFlags, ResolveFlags, fstat, fstatvfs,
    openat,
};

use crate::{Control, StoreError};

pub(crate) const READ_CHUNK_BYTES: usize = 64 * 1024;
/// Explicit payload-allocation granularity charged by the page cache.
/// Allocator bookkeeping remains observable through RSS rather than being
/// guessed from an implementation-specific malloc size class.
pub(crate) const PAGE_BUFFER_ALIGNMENT: u64 = 64;

const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC)
    .union(OFlags::NONBLOCK);
const REGULAR_READ_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC)
    .union(OFlags::NONBLOCK);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resolver {
    OpenAt2,
    ComponentOpenAt,
}

static RESOLVER: OnceLock<Result<Resolver, StoreError>> = OnceLock::new();

const RESOLVE_FLAGS: ResolveFlags = ResolveFlags::BENEATH
    .union(ResolveFlags::NO_SYMLINKS)
    .union(ResolveFlags::NO_MAGICLINKS);

/// Opens every user-supplied path component relative to the descriptor for
/// its parent. Symlinks and `..` are never followed, including in ancestors.
pub(crate) fn open_directory_path(path: &Path, kind: &'static str) -> Result<OwnedFd, StoreError> {
    if path.as_os_str().is_empty() {
        return Err(StoreError::UnsafeLayout { kind });
    }
    let absolute = path.is_absolute();
    let anchor = if absolute { "/" } else { "." };
    let mut current = openat(rustix::fs::CWD, anchor, DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| map_open_error(error, kind))?;

    for component in path.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(component) => {
                current = open_directory_component(&current, component, kind)?;
            }
            Component::ParentDir | Component::Prefix(_) => {
                return Err(StoreError::UnsafeLayout { kind });
            }
        }
    }
    ensure_directory(&current, kind)?;
    Ok(current)
}

fn open_directory_component(
    parent: &impl AsFd,
    component: &OsStr,
    kind: &'static str,
) -> Result<OwnedFd, StoreError> {
    let fd = open_relative(parent, component, DIRECTORY_FLAGS, Mode::empty(), kind)?;
    ensure_directory(&fd, kind)?;
    Ok(fd)
}

pub(crate) fn open_child_dir(
    parent: &impl AsFd,
    component: &str,
    kind: &'static str,
) -> Result<OwnedFd, StoreError> {
    validate_component(component, kind)?;
    open_directory_component(parent, OsStr::new(component), kind)
}

pub(crate) fn open_regular_leaf(
    parent: &impl AsFd,
    component: &str,
    kind: &'static str,
    expected_length: Option<u64>,
) -> Result<OwnedFd, StoreError> {
    validate_component(component, kind)?;
    let fd = open_relative(
        parent,
        OsStr::new(component),
        REGULAR_READ_FLAGS,
        Mode::empty(),
        kind,
    )?;
    let length = metadata_len_regular(&fd, kind)?;
    if let Some(expected) = expected_length
        && length != expected
    {
        return Err(StoreError::LengthMismatch {
            kind,
            expected,
            actual: length,
        });
    }
    Ok(fd)
}

fn open_relative(
    parent: &impl AsFd,
    component: &OsStr,
    flags: OFlags,
    mode: Mode,
    kind: &'static str,
) -> Result<OwnedFd, StoreError> {
    let resolver = RESOLVER.get_or_init(probe_resolver).clone()?;
    let opened = match resolver {
        Resolver::OpenAt2 => rustix::fs::openat2(parent, component, flags, mode, RESOLVE_FLAGS),
        Resolver::ComponentOpenAt => openat(parent, component, flags, mode),
    };
    opened.map_err(|error| map_open_error(error, kind))
}

fn probe_resolver() -> Result<Resolver, StoreError> {
    match rustix::fs::openat2(
        rustix::fs::CWD,
        ".",
        DIRECTORY_FLAGS,
        Mode::empty(),
        RESOLVE_FLAGS,
    ) {
        Ok(fd) => {
            drop(fd);
            Ok(Resolver::OpenAt2)
        }
        Err(rustix::io::Errno::NOSYS | rustix::io::Errno::INVAL | rustix::io::Errno::PERM) => {
            Ok(Resolver::ComponentOpenAt)
        }
        Err(error) => Err(StoreError::io("probing descriptor resolver", &error)),
    }
}

pub(crate) fn metadata_len_regular(fd: &impl AsFd, kind: &'static str) -> Result<u64, StoreError> {
    let stat = fstat(fd).map_err(|error| StoreError::io("reading retained metadata", &error))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(StoreError::UnsafeLayout { kind });
    }
    u64::try_from(stat.st_size).map_err(|_| StoreError::UnsafeLayout { kind })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FileIdentity {
    pub(crate) device: u64,
    pub(crate) owner: u32,
    pub(crate) mode: Mode,
}

pub(crate) fn file_identity(
    fd: &impl AsFd,
    _kind: &'static str,
) -> Result<FileIdentity, StoreError> {
    let stat = fstat(fd).map_err(|error| StoreError::io("reading retained identity", &error))?;
    let device = stat.st_dev;
    let owner = stat.st_uid;
    Ok(FileIdentity {
        device,
        owner,
        mode: Mode::from_raw_mode(stat.st_mode),
    })
}

pub(crate) fn require_same_filesystem_owner(
    expected: FileIdentity,
    fd: &impl AsFd,
    kind: &'static str,
) -> Result<FileIdentity, StoreError> {
    let actual = file_identity(fd, kind)?;
    if actual.device != expected.device || actual.owner != expected.owner {
        return Err(StoreError::UnsafeLayout { kind });
    }
    require_safe_permissions(actual, kind)?;
    Ok(actual)
}

pub(crate) fn require_safe_permissions(
    identity: FileIdentity,
    kind: &'static str,
) -> Result<(), StoreError> {
    if identity.mode.intersects(Mode::WGRP | Mode::WOTH) {
        return Err(StoreError::UnsafeLayout { kind });
    }
    Ok(())
}

pub(crate) fn ensure_directory(fd: &impl AsFd, kind: &'static str) -> Result<(), StoreError> {
    let stat = fstat(fd).map_err(|error| StoreError::io("reading retained metadata", &error))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(StoreError::UnsafeLayout { kind });
    }
    Ok(())
}

pub(crate) fn read_bounded_at(
    fd: &impl AsFd,
    maximum: usize,
    kind: &'static str,
    control: &Control,
) -> Result<Vec<u8>, StoreError> {
    let length = metadata_len_regular(fd, kind)?;
    let maximum_u64 = u64::try_from(maximum).map_err(|_| StoreError::ArithmeticOverflow {
        context: "representing a bounded read limit",
    })?;
    if length > maximum_u64 {
        return Err(StoreError::ResourceExhausted {
            resource: kind,
            required: length,
            limit: maximum_u64,
        });
    }
    let bytes = read_exact_at(fd, 0, length, kind, control)?;
    let final_length = metadata_len_regular(fd, kind)?;
    if final_length != length {
        return Err(StoreError::LengthMismatch {
            kind,
            expected: length,
            actual: final_length,
        });
    }
    Ok(bytes)
}

pub(crate) fn read_exact_at(
    fd: &impl AsFd,
    offset: u64,
    length: u64,
    kind: &'static str,
    control: &Control,
) -> Result<Vec<u8>, StoreError> {
    read_exact_at_observed(fd, offset, length, kind, control).0
}

/// Returns verified-loop output plus bytes actually returned by `pread`.
/// Failed buffers remain owned and are dropped inside the result path.
pub(crate) fn read_exact_at_observed(
    fd: &impl AsFd,
    offset: u64,
    length: u64,
    kind: &'static str,
    control: &Control,
) -> (Result<Vec<u8>, StoreError>, u64) {
    let capacity = match usize::try_from(length) {
        Ok(capacity) => capacity,
        Err(_) => {
            return (
                Err(StoreError::ResourceExhausted {
                    resource: kind,
                    required: length,
                    limit: usize::MAX as u64,
                }),
                0,
            );
        }
    };
    if let Err(error) = control.check() {
        return (Err(error), 0);
    }
    let allocation_bytes = match aligned_page_buffer_bytes(length) {
        Ok(value) => value,
        Err(error) => return (Err(error), 0),
    };
    let allocation_capacity = match usize::try_from(allocation_bytes) {
        Ok(value) => value,
        Err(_) => {
            return (
                Err(StoreError::ResourceExhausted {
                    resource: kind,
                    required: allocation_bytes,
                    limit: usize::MAX as u64,
                }),
                0,
            );
        }
    };
    let mut bytes = Vec::with_capacity(allocation_capacity);
    bytes.resize(capacity, 0);
    let mut filled = 0_usize;
    while filled < bytes.len() {
        if let Err(error) = control.check() {
            return (Err(error), filled as u64);
        }
        let chunk_end = filled.saturating_add(READ_CHUNK_BYTES).min(bytes.len());
        let filled_u64 = match u64::try_from(filled) {
            Ok(value) => value,
            Err(_) => {
                return (
                    Err(StoreError::ArithmeticOverflow {
                        context: "representing a positional read offset",
                    }),
                    u64::MAX,
                );
            }
        };
        let position = match offset.checked_add(filled_u64) {
            Some(value) => value,
            None => {
                return (
                    Err(StoreError::ArithmeticOverflow {
                        context: "computing a positional read offset",
                    }),
                    filled_u64,
                );
            }
        };
        match rustix::io::pread(fd, &mut bytes[filled..chunk_end], position) {
            Ok(0) => {
                return (
                    Err(StoreError::LengthMismatch {
                        kind,
                        expected: length,
                        actual: filled_u64,
                    }),
                    filled_u64,
                );
            }
            Ok(count) => match filled.checked_add(count) {
                Some(value) => filled = value,
                None => {
                    return (
                        Err(StoreError::ArithmeticOverflow {
                            context: "accounting positional read bytes",
                        }),
                        filled_u64,
                    );
                }
            },
            Err(rustix::io::Errno::INTR) => {
                if let Err(error) = control.check() {
                    return (Err(error), filled_u64);
                }
            }
            Err(error) => {
                return (
                    Err(StoreError::io("reading retained bytes", &error)),
                    filled_u64,
                );
            }
        }
    }
    let physical_bytes = u64::try_from(filled).unwrap_or(u64::MAX);
    match control.check() {
        Ok(()) => (Ok(bytes), physical_bytes),
        Err(error) => (Err(error), physical_bytes),
    }
}

pub(crate) fn aligned_page_buffer_bytes(length: u64) -> Result<u64, StoreError> {
    if length == 0 {
        return Ok(0);
    }
    length
        .checked_add(PAGE_BUFFER_ALIGNMENT - 1)
        .map(|value| value / PAGE_BUFFER_ALIGNMENT * PAGE_BUFFER_ALIGNMENT)
        .ok_or(StoreError::ArithmeticOverflow {
            context: "aligning a page-buffer allocation",
        })
}

pub(crate) fn duplicate(fd: &impl AsFd) -> Result<OwnedFd, StoreError> {
    rustix::io::dup(fd).map_err(|error| StoreError::io("duplicating retained descriptor", &error))
}

pub(crate) fn validate_component(component: &str, kind: &'static str) -> Result<(), StoreError> {
    if component.is_empty()
        || component == "."
        || component == ".."
        || component.len() > 255
        || !component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(StoreError::UnsafeLayout { kind });
    }
    Ok(())
}

fn map_open_error(error: rustix::io::Errno, kind: &'static str) -> StoreError {
    match error {
        rustix::io::Errno::NOENT => StoreError::Missing { kind },
        rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR => StoreError::UnsafeLayout { kind },
        _ => StoreError::io("opening retained entry", &error),
    }
}

// The following small mutation primitives are shared by the CAS vertical. All
// names still pass the same single-component validator used by readers.

#[allow(dead_code)]
pub(crate) fn mkdir_child(
    parent: &impl AsFd,
    component: &str,
    kind: &'static str,
    mode: Mode,
) -> Result<(), StoreError> {
    validate_component(component, kind)?;
    rustix::fs::mkdirat(parent, component, mode).map_err(|error| match error {
        rustix::io::Errno::EXIST => StoreError::AlreadyExists { kind },
        _ => StoreError::io("creating retained directory", &error),
    })
}

#[allow(dead_code)]
pub(crate) fn create_new_leaf(
    parent: &impl AsFd,
    component: &str,
    kind: &'static str,
    mode: Mode,
) -> Result<OwnedFd, StoreError> {
    validate_component(component, kind)?;
    let flags = OFlags::RDWR
        | OFlags::CREATE
        | OFlags::EXCL
        | OFlags::NOFOLLOW
        | OFlags::CLOEXEC
        | OFlags::NONBLOCK;
    let fd = openat(parent, component, flags, mode).map_err(|error| match error {
        rustix::io::Errno::EXIST => StoreError::AlreadyExists { kind },
        rustix::io::Errno::LOOP => StoreError::UnsafeLayout { kind },
        _ => StoreError::io("creating retained leaf", &error),
    })?;
    let length = metadata_len_regular(&fd, kind)?;
    if length != 0 {
        return Err(StoreError::Invariant {
            problem: "exclusive creation returned a nonempty leaf",
        });
    }
    Ok(fd)
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RenameNoReplace {
    Published,
    Unsupported,
}

#[allow(dead_code)]
pub(crate) fn rename_noreplace(
    old_parent: &impl AsFd,
    old_component: &str,
    new_parent: &impl AsFd,
    new_component: &str,
    kind: &'static str,
) -> Result<RenameNoReplace, StoreError> {
    validate_component(old_component, kind)?;
    validate_component(new_component, kind)?;
    match rustix::fs::renameat_with(
        old_parent,
        old_component,
        new_parent,
        new_component,
        RenameFlags::NOREPLACE,
    ) {
        Ok(()) => Ok(RenameNoReplace::Published),
        Err(rustix::io::Errno::EXIST) => Err(StoreError::AlreadyExists { kind }),
        Err(rustix::io::Errno::NOSYS | rustix::io::Errno::INVAL) => {
            Ok(RenameNoReplace::Unsupported)
        }
        Err(error) => Err(StoreError::io("publishing retained leaf", &error)),
    }
}

#[allow(dead_code)]
pub(crate) fn link_noreplace(
    old_parent: &impl AsFd,
    old_component: &str,
    new_parent: &impl AsFd,
    new_component: &str,
    kind: &'static str,
) -> Result<(), StoreError> {
    validate_component(old_component, kind)?;
    validate_component(new_component, kind)?;
    rustix::fs::linkat(
        old_parent,
        old_component,
        new_parent,
        new_component,
        AtFlags::empty(),
    )
    .map_err(|error| match error {
        rustix::io::Errno::EXIST => StoreError::AlreadyExists { kind },
        _ => StoreError::io("linking retained leaf", &error),
    })
}

#[allow(dead_code)]
pub(crate) fn unlink_leaf(
    parent: &impl AsFd,
    component: &str,
    kind: &'static str,
) -> Result<(), StoreError> {
    validate_component(component, kind)?;
    rustix::fs::unlinkat(parent, component, AtFlags::empty())
        .map_err(|error| StoreError::io("unlinking retained leaf", &error))
}

#[allow(dead_code)]
pub(crate) fn sync_fd(fd: &impl AsFd) -> Result<(), StoreError> {
    rustix::fs::fsync(fd).map_err(|error| StoreError::io("synchronizing retained entry", &error))
}

#[allow(dead_code)]
pub(crate) fn lock_exclusive(fd: &impl AsFd) -> Result<(), StoreError> {
    rustix::fs::flock(fd, FlockOperation::LockExclusive)
        .map_err(|error| StoreError::io("locking the transaction descriptor", &error))
}

#[allow(dead_code)]
pub(crate) fn available_bytes(fd: &impl AsFd) -> Result<(u64, u64), StoreError> {
    let stat =
        fstatvfs(fd).map_err(|error| StoreError::io("reading filesystem capacity", &error))?;
    let available =
        stat.f_bavail
            .checked_mul(stat.f_frsize)
            .ok_or(StoreError::ArithmeticOverflow {
                context: "computing available filesystem bytes",
            })?;
    Ok((available, stat.f_frsize))
}

#[allow(dead_code)]
pub(crate) fn truncate(fd: &impl AsFd, length: u64) -> Result<(), StoreError> {
    rustix::fs::ftruncate(fd, length)
        .map_err(|error| StoreError::io("truncating retained data", &error))
}

#[allow(dead_code)]
pub(crate) fn write_all_at(
    fd: &impl AsFd,
    mut offset: u64,
    mut bytes: &[u8],
    control: &Control,
) -> Result<(), StoreError> {
    while !bytes.is_empty() {
        control.check()?;
        let chunk_length = bytes.len().min(READ_CHUNK_BYTES);
        match rustix::io::pwrite(fd, &bytes[..chunk_length], offset) {
            Ok(0) => {
                return Err(StoreError::Io {
                    operation: "writing retained bytes",
                    kind: std::io::ErrorKind::WriteZero,
                });
            }
            Ok(count) => {
                let count_u64 =
                    u64::try_from(count).map_err(|_| StoreError::ArithmeticOverflow {
                        context: "representing a positional write count",
                    })?;
                offset = offset
                    .checked_add(count_u64)
                    .ok_or(StoreError::ArithmeticOverflow {
                        context: "computing a positional write offset",
                    })?;
                bytes = &bytes[count..];
            }
            Err(rustix::io::Errno::INTR) => control.check()?,
            Err(error) => return Err(StoreError::io("writing retained bytes", &error)),
        }
    }
    control.check()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{
        aligned_page_buffer_bytes, open_directory_path, open_regular_leaf, read_exact_at,
        read_exact_at_observed,
    };
    use crate::{Control, StoreError};

    #[test]
    fn parent_components_are_rejected() {
        let error = open_directory_path(Path::new("../outside"), "root").unwrap_err();
        assert_eq!(error, StoreError::UnsafeLayout { kind: "root" });
    }

    #[test]
    fn retained_leaf_reads_positionally() {
        let temporary = tempdir().unwrap();
        fs::write(temporary.path().join("leaf"), b"abcdef").unwrap();
        let root = open_directory_path(temporary.path(), "root").unwrap();
        let leaf = open_regular_leaf(&root, "leaf", "test leaf", Some(6)).unwrap();
        let bytes = read_exact_at(&leaf, 2, 3, "test leaf", &Control::default()).unwrap();
        assert_eq!(bytes, b"cde");
    }

    #[test]
    fn short_read_reports_actual_physical_bytes_without_a_buffer() {
        let temporary = tempdir().unwrap();
        fs::write(temporary.path().join("leaf"), b"abcdef").unwrap();
        let root = open_directory_path(temporary.path(), "root").unwrap();
        let leaf = open_regular_leaf(&root, "leaf", "test leaf", Some(6)).unwrap();
        let (result, physical_bytes) =
            read_exact_at_observed(&leaf, 0, 10, "test leaf", &Control::default());
        assert!(matches!(result, Err(StoreError::LengthMismatch { .. })));
        assert_eq!(physical_bytes, 6);
    }

    #[test]
    fn page_buffer_charge_includes_explicit_alignment_padding() {
        assert_eq!(aligned_page_buffer_bytes(0).unwrap(), 0);
        assert_eq!(aligned_page_buffer_bytes(1).unwrap(), 64);
        assert_eq!(aligned_page_buffer_bytes(64).unwrap(), 64);
        assert_eq!(aligned_page_buffer_bytes(65).unwrap(), 128);
        assert!(aligned_page_buffer_bytes(u64::MAX).is_err());
    }

    use std::path::Path;
}
