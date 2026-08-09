//! Bounded canonical encoding and durable publication for actor-race evidence.
//!
//! The closed capture structs declare their keys in ADR order. This layer keeps
//! that order, uses serde_json's compact formatter, rejects noncanonical byte
//! spellings, and publishes only through descriptor-relative no-replace
//! operations.

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Write};
use std::os::fd::OwnedFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};

use rustix::fs::{
    AtFlags, FileType, Mode, OFlags, fchmod, fstat, fsync, linkat, openat, statat, unlinkat,
};
use serde::{Deserialize, Serialize, de::IgnoredAny};

/// Frozen maximum for the complete canonical capture, including its final LF.
pub(super) const MAX_ACTOR_RACE_CAPTURE_BYTES: usize = 33_554_432;

const CAPTURE_ENVIRONMENT_VARIABLE: &str = "RUNNEL_ACTOR_RACE_CAPTURE";
#[cfg(test)]
const LEGACY_STAGE_PREFIX: &str = ".runnel-actor-race-stage-";
const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC)
    .union(OFlags::NONBLOCK);
const ANONYMOUS_STAGE_FLAGS: OFlags = OFlags::WRONLY
    .union(OFlags::TMPFILE)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC)
    .union(OFlags::NONBLOCK);
const STAGE_MODE: Mode = Mode::RUSR.union(Mode::WUSR);

type CaptureFileResult<T> = Result<T, String>;

/// Encodes one capture with compact JSON and exactly one trailing LF.
///
/// The serializer's type is responsible for its closed schema and key order.
/// This writer makes every allocation fallible, counts the final LF inside the
/// frozen bound, and returns no partial representation on any failure.
pub(super) fn encode_actor_race_capture<T: Serialize>(capture: &T) -> CaptureFileResult<Vec<u8>> {
    encode_actor_race_capture_with_limit_and_deadline(capture, MAX_ACTOR_RACE_CAPTURE_BYTES, None)
}

/// Encodes while checking the full-command deadline at bounded write points.
pub(super) fn encode_actor_race_capture_before<T: Serialize>(
    capture: &T,
    deadline: tokio::time::Instant,
) -> CaptureFileResult<Vec<u8>> {
    encode_actor_race_capture_with_limit_and_deadline(
        capture,
        MAX_ACTOR_RACE_CAPTURE_BYTES,
        Some(deadline),
    )
}

fn encode_actor_race_capture_with_limit<T: Serialize>(
    capture: &T,
    maximum_bytes: usize,
) -> CaptureFileResult<Vec<u8>> {
    encode_actor_race_capture_with_limit_and_deadline(capture, maximum_bytes, None)
}

fn encode_actor_race_capture_with_limit_and_deadline<T: Serialize>(
    capture: &T,
    maximum_bytes: usize,
    deadline: Option<tokio::time::Instant>,
) -> CaptureFileResult<Vec<u8>> {
    require_unexpired_publication_deadline(deadline, "before encoding")?;
    let mut writer = BoundedCanonicalWriter::new_with_deadline(maximum_bytes, deadline)?;
    serde_json::to_writer(&mut writer, capture)
        .map_err(|error| format!("cannot encode bounded actor-race evidence: {error}"))?;
    let bytes = writer.finish()?;
    validate_actor_race_capture_bytes(&bytes, maximum_bytes)?;
    require_unexpired_publication_deadline(deadline, "after encoding")?;
    Ok(bytes)
}

/// Validates and, when requested by the environment, durably publishes bytes.
///
/// An absent environment variable deliberately remains validate-only. A
/// present variable must name one normal leaf below a component-wise retained,
/// no-symlink parent path. The immediate parent must be owned by the effective
/// user, exclude group/other writes, and support inode-bound Linux `O_TMPFILE`
/// publication.
pub(super) fn publish_actor_race_capture_from_env(
    bytes: &[u8],
    deadline: tokio::time::Instant,
) -> CaptureFileResult<()> {
    validate_actor_race_capture_bytes(bytes, MAX_ACTOR_RACE_CAPTURE_BYTES)?;
    require_unexpired_publication_deadline(Some(deadline), "before publication")?;
    let Some(destination) = std::env::var_os(CAPTURE_ENVIRONMENT_VARIABLE) else {
        return Ok(());
    };
    publish_actor_race_capture_to_path_validated(bytes, Path::new(&destination), Some(deadline))
}

fn publish_actor_race_capture_to_path(bytes: &[u8], destination: &Path) -> CaptureFileResult<()> {
    validate_actor_race_capture_bytes(bytes, MAX_ACTOR_RACE_CAPTURE_BYTES)?;
    publish_actor_race_capture_to_path_validated(bytes, destination, None)
}

fn publish_actor_race_capture_to_path_validated(
    bytes: &[u8],
    destination: &Path,
    deadline: Option<tokio::time::Instant>,
) -> CaptureFileResult<()> {
    require_unexpired_publication_deadline(deadline, "before staging")?;
    let (parent, leaf) = retain_destination_parent(destination)?;
    require_rollback_safe_parent(&parent)?;
    let stage_fd = create_anonymous_stage(&parent)?;

    fchmod(&stage_fd, STAGE_MODE)
        .map_err(|error| format!("cannot set actor-race capture stage mode: {error}"))?;
    let identity = require_anonymous_regular_mode_0600(&stage_fd)?;

    let stage = write_and_sync_stage(stage_fd, bytes)?;
    require_unexpired_publication_deadline(deadline, "before the no-replace transition")?;

    linkat(&stage, "", &parent, leaf.as_os_str(), AtFlags::EMPTY_PATH).map_err(|error| {
        format!("cannot publish actor-race capture without replacement: {error}")
    })?;
    let mut published = OwnedPublishedLeaf::new(&parent, &stage, leaf, identity);
    if let Err(error) = require_linked_regular_mode_0600(&stage) {
        return Err(published.rollback_after(error));
    }

    finish_published_leaf(published, deadline)
}

fn write_and_sync_stage(stage_fd: OwnedFd, bytes: &[u8]) -> CaptureFileResult<File> {
    let mut stage_file = File::from(stage_fd);
    stage_file
        .write_all(bytes)
        .map_err(|error| format!("cannot write actor-race capture stage: {error}"))?;
    stage_file
        .sync_all()
        .map_err(|error| format!("cannot sync actor-race capture stage: {error}"))?;
    Ok(stage_file)
}

fn finish_published_leaf(
    mut published: OwnedPublishedLeaf<'_>,
    deadline: Option<tokio::time::Instant>,
) -> CaptureFileResult<()> {
    if let Err(error) = fsync(published.parent()) {
        return Err(
            published.rollback_after(format!("cannot sync actor-race capture parent: {error}"))
        );
    }
    if let Err(expired) =
        require_unexpired_publication_deadline(deadline, "after durable publication")
    {
        return Err(published.rollback_after(expired));
    }
    published.disarm();
    Ok(())
}

fn require_unexpired_publication_deadline(
    deadline: Option<tokio::time::Instant>,
    phase: &str,
) -> CaptureFileResult<()> {
    if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
        return Err(format!("actor-race capture deadline expired {phase}"));
    }
    Ok(())
}

fn validate_actor_race_capture_bytes(bytes: &[u8], maximum_bytes: usize) -> CaptureFileResult<()> {
    if bytes.len() > maximum_bytes {
        return Err("actor-race capture exceeds its frozen byte bound".to_owned());
    }
    let Some(body) = bytes.strip_suffix(b"\n") else {
        return Err("actor-race capture does not end in exactly one LF".to_owned());
    };
    if body.is_empty() {
        return Err("actor-race capture JSON body is empty".to_owned());
    }
    validate_canonical_body(body)?;

    let mut deserializer = serde_json::Deserializer::from_slice(body);
    IgnoredAny::deserialize(&mut deserializer)
        .map_err(|error| format!("actor-race capture is not complete JSON: {error}"))?;
    deserializer
        .end()
        .map_err(|error| format!("actor-race capture has trailing JSON data: {error}"))
}

fn validate_canonical_body(body: &[u8]) -> CaptureFileResult<()> {
    let mut inside_string = false;
    validate_canonical_body_chunk(body, &mut inside_string)
}

fn validate_canonical_body_chunk(chunk: &[u8], inside_string: &mut bool) -> CaptureFileResult<()> {
    if chunk.iter().any(|byte| !byte.is_ascii()) {
        return Err("actor-race capture contains a non-ASCII byte".to_owned());
    }
    if chunk.contains(&b'\\') {
        return Err("actor-race capture contains a noncanonical backslash escape".to_owned());
    }
    if chunk.iter().any(u8::is_ascii_control) {
        return Err("actor-race capture contains a control byte before its final LF".to_owned());
    }
    for byte in chunk {
        if *byte == b'"' {
            *inside_string = !*inside_string;
        } else if !*inside_string && byte.is_ascii_whitespace() {
            return Err(
                "actor-race capture contains noncanonical whitespace outside a string".to_owned(),
            );
        }
    }
    Ok(())
}

struct BoundedCanonicalWriter {
    bytes: Vec<u8>,
    body_limit: usize,
    deadline: Option<tokio::time::Instant>,
    inside_string: bool,
    total_limit: usize,
}

impl BoundedCanonicalWriter {
    fn new(total_limit: usize) -> CaptureFileResult<Self> {
        Self::new_with_deadline(total_limit, None)
    }

    fn new_with_deadline(
        total_limit: usize,
        deadline: Option<tokio::time::Instant>,
    ) -> CaptureFileResult<Self> {
        let body_limit = total_limit
            .checked_sub(1)
            .ok_or_else(|| "actor-race capture bound cannot hold its final LF".to_owned())?;
        Ok(Self {
            bytes: Vec::new(),
            body_limit,
            deadline,
            inside_string: false,
            total_limit,
        })
    }

    fn finish(mut self) -> CaptureFileResult<Vec<u8>> {
        require_unexpired_publication_deadline(self.deadline, "before the final encoded LF")?;
        let final_len = self
            .bytes
            .len()
            .checked_add(1)
            .ok_or_else(|| "actor-race capture length overflowed".to_owned())?;
        if final_len > self.total_limit {
            return Err("actor-race capture exceeds its frozen byte bound".to_owned());
        }
        self.bytes
            .try_reserve_exact(1)
            .map_err(|_| "cannot allocate the final actor-race capture LF".to_owned())?;
        self.bytes.push(b'\n');
        Ok(self.bytes)
    }
}

impl Write for BoundedCanonicalWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        require_unexpired_publication_deadline(self.deadline, "during encoding")
            .map_err(io::Error::other)?;
        let mut next_string_state = self.inside_string;
        validate_canonical_body_chunk(buffer, &mut next_string_state).map_err(io::Error::other)?;
        let new_len = self
            .bytes
            .len()
            .checked_add(buffer.len())
            .ok_or_else(|| io::Error::other("actor-race capture length overflowed"))?;
        if new_len > self.body_limit {
            return Err(io::Error::other(
                "actor-race capture exceeds its frozen byte bound",
            ));
        }
        self.bytes
            .try_reserve(buffer.len())
            .map_err(|_| io::Error::other("cannot allocate bounded actor-race capture bytes"))?;
        self.bytes.extend_from_slice(buffer);
        self.inside_string = next_string_state;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn retain_destination_parent(destination: &Path) -> CaptureFileResult<(OwnedFd, OsString)> {
    let leaf = match destination.components().next_back() {
        Some(Component::Normal(leaf)) => leaf,
        _ => {
            return Err("RUNNEL_ACTOR_RACE_CAPTURE must end in exactly one normal leaf".to_owned());
        }
    };
    if leaf.as_bytes().is_empty() || leaf.as_bytes().len() > 255 || leaf.as_bytes().contains(&0) {
        return Err("RUNNEL_ACTOR_RACE_CAPTURE has an invalid leaf".to_owned());
    }
    let leaf = leaf.to_os_string();
    let parent_path = destination.parent().unwrap_or_else(|| Path::new(""));
    let anchor = if destination.is_absolute() { "/" } else { "." };
    let mut parent = openat(rustix::fs::CWD, anchor, DIRECTORY_FLAGS, Mode::empty())
        .map_err(|error| format!("cannot retain actor-race capture path anchor: {error}"))?;

    for component in parent_path.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(component) => {
                parent = openat(&parent, component, DIRECTORY_FLAGS, Mode::empty()).map_err(
                    |error| {
                        format!(
                            "cannot retain a no-symlink actor-race capture parent component: {error}"
                        )
                    },
                )?;
                require_directory(&parent)?;
            }
            Component::ParentDir | Component::Prefix(_) => {
                return Err(
                    "RUNNEL_ACTOR_RACE_CAPTURE parent contains an unsafe component".to_owned(),
                );
            }
        }
    }
    require_directory(&parent)?;
    Ok((parent, leaf))
}

fn require_directory(fd: &OwnedFd) -> CaptureFileResult<()> {
    let stat =
        fstat(fd).map_err(|error| format!("cannot inspect actor-race capture parent: {error}"))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err("actor-race capture parent is not a retained directory".to_owned());
    }
    Ok(())
}

/// Requires a parent in which unrelated users cannot race rollback paths.
///
/// Linux has descriptor-based creation and linking but no conditional
/// unlink-by-inode operation. Keeping the retained directory owned by this
/// effective user and not group/world-writable makes the subsequent
/// identity-check-plus-unlink rollback safe against external users. We fail
/// closed instead of falling back to a pathname-addressed staging file.
fn require_rollback_safe_parent(parent: &OwnedFd) -> CaptureFileResult<()> {
    let stat = fstat(parent)
        .map_err(|error| format!("cannot inspect actor-race capture parent: {error}"))?;
    if stat.st_uid != rustix::process::geteuid().as_raw() {
        return Err(
            "actor-race capture parent must be owned by the effective publication user".to_owned(),
        );
    }
    if stat.st_mode & 0o022 != 0 {
        return Err(
            "actor-race capture parent must not be writable by group or other users".to_owned(),
        );
    }
    Ok(())
}

fn create_anonymous_stage(parent: &OwnedFd) -> CaptureFileResult<OwnedFd> {
    openat(parent, ".", ANONYMOUS_STAGE_FLAGS, STAGE_MODE).map_err(|error| {
        format!(
            "cannot create inode-bound actor-race capture stage; the destination filesystem must support Linux O_TMPFILE: {error}"
        )
    })
}

fn require_anonymous_regular_mode_0600(fd: &OwnedFd) -> CaptureFileResult<FileIdentity> {
    let stat =
        fstat(fd).map_err(|error| format!("cannot inspect actor-race capture stage: {error}"))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err("actor-race capture stage is not a regular file".to_owned());
    }
    if stat.st_mode & 0o777 != 0o600 {
        return Err("actor-race capture stage is not mode 0600".to_owned());
    }
    if stat.st_nlink != 0 {
        return Err("actor-race capture stage unexpectedly has a filesystem name".to_owned());
    }
    Ok(FileIdentity::from_stat(&stat))
}

fn require_linked_regular_mode_0600(fd: &File) -> CaptureFileResult<()> {
    let stat =
        fstat(fd).map_err(|error| format!("cannot inspect linked actor-race capture: {error}"))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_mode & 0o777 != 0o600
        || stat.st_nlink != 1
    {
        return Err(
            "linked actor-race capture lost its regular mode-0600 single-link invariant".to_owned(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

impl FileIdentity {
    fn from_stat(stat: &rustix::fs::Stat) -> Self {
        Self {
            device: stat.st_dev,
            inode: stat.st_ino,
        }
    }
}

/// Rollback authority for a destination created by this publication attempt.
///
/// It becomes armed only after the exact anonymous source inode is linked by a
/// no-replace operation. The protected-parent precondition excludes unrelated
/// writers, and rollback still compares device/inode identity before unlinking.
/// There is intentionally no `Drop` cleanup: every post-link failure invokes
/// rollback explicitly so a cleanup failure is preserved alongside its cause.
struct OwnedPublishedLeaf<'a> {
    active: bool,
    component: OsString,
    identity: FileIdentity,
    parent: &'a OwnedFd,
    source: &'a File,
}

impl<'a> OwnedPublishedLeaf<'a> {
    fn new(
        parent: &'a OwnedFd,
        source: &'a File,
        component: OsString,
        identity: FileIdentity,
    ) -> Self {
        Self {
            active: true,
            component,
            identity,
            parent,
            source,
        }
    }

    fn parent(&self) -> &OwnedFd {
        self.parent
    }

    fn disarm(&mut self) {
        self.active = false;
    }

    fn rollback_and_sync(&mut self) -> CaptureFileResult<()> {
        if !self.active {
            return Ok(());
        }

        let source_stat = fstat(self.source).map_err(|error| {
            format!("cannot recheck actor-race capture source identity: {error}")
        })?;
        if FileIdentity::from_stat(&source_stat) != self.identity {
            return Err("actor-race capture source identity changed before rollback".to_owned());
        }

        let leaf_stat = statat(
            self.parent,
            self.component.as_os_str(),
            AtFlags::SYMLINK_NOFOLLOW,
        )
        .map_err(|error| format!("cannot inspect the newly owned capture leaf: {error}"))?;
        if FileIdentity::from_stat(&leaf_stat) != self.identity
            || FileType::from_raw_mode(leaf_stat.st_mode) != FileType::RegularFile
        {
            return Err(
                "refusing to roll back an actor-race capture leaf with a different identity"
                    .to_owned(),
            );
        }

        unlinkat(self.parent, self.component.as_os_str(), AtFlags::empty()).map_err(|error| {
            format!("cannot remove the newly owned actor-race capture leaf: {error}")
        })?;
        self.active = false;
        fsync(self.parent).map_err(|error| {
            format!("cannot sync actor-race capture parent after rollback: {error}")
        })?;
        Ok(())
    }

    fn rollback_after(&mut self, primary: String) -> String {
        match self.rollback_and_sync() {
            Ok(()) => primary,
            Err(cleanup) => format!("{primary}; {cleanup}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, DirBuilder};
    use std::io;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt, symlink};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde::{Serialize, ser::SerializeSeq};

    use super::*;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    #[derive(Serialize)]
    struct SmallCapture<'a> {
        a: u64,
        b: &'a str,
    }

    struct DelayedCapture;

    impl Serialize for DelayedCapture {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut sequence = serializer.serialize_seq(Some(2))?;
            sequence.serialize_element(&0_u64)?;
            std::thread::sleep(std::time::Duration::from_millis(20));
            sequence.serialize_element(&1_u64)?;
            sequence.end()
        }
    }

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            for _ in 0..128 {
                let nonce = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "runnel-race-capture-{label}-{}-{nonce}",
                    std::process::id()
                ));
                let mut builder = DirBuilder::new();
                builder.mode(0o700);
                match builder.create(&path) {
                    Ok(()) => return Self { path },
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(error) => panic!("cannot create test directory: {error}"),
                }
            }
            panic!("cannot choose a unique test directory");
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn canonical_test_bytes() -> Vec<u8> {
        encode_actor_race_capture(&SmallCapture {
            a: 1,
            b: "request slot count",
        })
        .expect("small capture must encode")
    }

    fn assert_no_legacy_stages(directory: &Path) {
        let stages = fs::read_dir(directory)
            .expect("test directory must remain readable")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(LEGACY_STAGE_PREFIX)
            })
            .count();
        assert_eq!(stages, 0, "owned staging names must be cleaned up");
    }

    #[test]
    fn encoder_uses_compact_ascii_and_one_final_lf() {
        let bytes = canonical_test_bytes();
        assert_eq!(bytes, b"{\"a\":1,\"b\":\"request slot count\"}\n");
        assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 1);
        assert_eq!(bytes.last(), Some(&b'\n'));
        assert!(!bytes.contains(&b'\\'));
        assert!(bytes[..bytes.len() - 1].iter().all(u8::is_ascii));
    }

    #[test]
    fn canonical_validator_rejects_structural_whitespace_but_allows_string_spaces() {
        assert!(validate_actor_race_capture_bytes(b"{ \"a\":1}\n", 64).is_err());
        assert!(validate_actor_race_capture_bytes(b"{\"a\" :1}\n", 64).is_err());
        validate_actor_race_capture_bytes(b"{\"a\":\"space is data\"}\n", 64)
            .expect("ordinary spaces inside a string remain canonical data");
    }

    #[test]
    fn absent_capture_environment_is_validate_only() {
        if std::env::var_os(CAPTURE_ENVIRONMENT_VARIABLE).is_some() {
            // A gated full-capture run owns a present destination. This unit
            // test must neither publish first nor mutate process-wide state.
            return;
        }
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
        publish_actor_race_capture_from_env(&canonical_test_bytes(), deadline)
            .expect("an absent capture environment must remain validate-only");
        assert!(publish_actor_race_capture_from_env(b"{}\n\n", deadline).is_err());

        let expired = tokio::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("the monotonic clock can represent one second ago");
        let error = publish_actor_race_capture_from_env(&canonical_test_bytes(), expired)
            .expect_err("validate-only mode must enforce the full deadline");
        assert!(error.contains("deadline expired before publication"));
    }

    #[test]
    fn past_deadline_creates_neither_destination_nor_stage() {
        let directory = TestDirectory::new("past-deadline");
        let destination = directory.path().join("capture.json");
        let deadline = tokio::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("the monotonic clock can represent one second ago");

        let error = publish_actor_race_capture_to_path_validated(
            &canonical_test_bytes(),
            &destination,
            Some(deadline),
        )
        .expect_err("an expired deadline must reject publication");
        assert!(error.contains("deadline expired"));
        assert!(!destination.exists());
        assert_no_legacy_stages(directory.path());
    }

    #[test]
    fn writer_accepts_the_exact_frozen_cap_boundary() {
        let mut writer = BoundedCanonicalWriter::new(MAX_ACTOR_RACE_CAPTURE_BYTES)
            .expect("the frozen bound can hold an LF");
        let chunk = [b'a'; 64 * 1024];
        let mut remaining = MAX_ACTOR_RACE_CAPTURE_BYTES - 1;
        while remaining != 0 {
            let amount = remaining.min(chunk.len());
            writer
                .write_all(&chunk[..amount])
                .expect("exactly bounded body bytes must fit");
            remaining -= amount;
        }
        let bytes = writer.finish().expect("the reserved final LF must fit");
        assert_eq!(bytes.len(), MAX_ACTOR_RACE_CAPTURE_BYTES);
        assert_eq!(bytes.last(), Some(&b'\n'));

        let exact_small = encode_actor_race_capture_with_limit(&"abc", 6)
            .expect("JSON string plus LF exactly meets the test bound");
        assert_eq!(exact_small, b"\"abc\"\n");
    }

    #[test]
    fn writer_rejects_overflow_without_a_partial_write() {
        let mut writer = BoundedCanonicalWriter::new(8).expect("test bound is valid");
        writer.write_all(b"123456").expect("initial body must fit");
        assert!(writer.write_all(b"78").is_err());
        assert_eq!(writer.bytes, b"123456");
        assert!(encode_actor_race_capture_with_limit(&"abc", 5).is_err());
    }

    #[test]
    fn encoder_rejects_an_expired_full_command_deadline() {
        let expired = tokio::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .expect("the monotonic clock can represent one second ago");
        let error = encode_actor_race_capture_before(
            &SmallCapture {
                a: 1,
                b: "request slot count",
            },
            expired,
        )
        .expect_err("encoding must not bypass an expired full-command deadline");
        assert!(error.contains("deadline expired before encoding"));
    }

    #[test]
    fn encoder_rechecks_a_deadline_between_serializer_writes() {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(5);
        let error = encode_actor_race_capture_before(&DelayedCapture, deadline)
            .expect_err("a serializer that crosses the deadline must fail closed");
        assert!(error.contains("deadline expired"));
    }

    #[test]
    fn existing_destination_is_never_replaced() {
        let directory = TestDirectory::new("existing");
        let destination = directory.path().join("capture.json");
        fs::write(&destination, b"existing").expect("destination fixture must be written");

        assert!(publish_actor_race_capture_to_path(&canonical_test_bytes(), &destination).is_err());
        assert_eq!(
            fs::read(&destination).expect("destination must remain readable"),
            b"existing"
        );
        assert_no_legacy_stages(directory.path());
    }

    #[test]
    fn published_capture_is_mode_0600_and_complete() {
        let directory = TestDirectory::new("mode");
        let destination = directory.path().join("capture.json");
        let bytes = canonical_test_bytes();

        publish_actor_race_capture_to_path(&bytes, &destination)
            .expect("capture publication must succeed");

        let metadata = fs::symlink_metadata(&destination).expect("capture must exist");
        assert!(metadata.file_type().is_file());
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        assert_eq!(metadata.nlink(), 1);
        assert_eq!(
            fs::read(&destination).expect("capture must be readable"),
            bytes
        );
        assert_no_legacy_stages(directory.path());
    }

    #[test]
    fn symlink_parent_is_rejected_without_publication() {
        let directory = TestDirectory::new("parent-symlink");
        let real_parent = directory.path().join("real");
        fs::create_dir(&real_parent).expect("real parent must be created");
        let linked_parent = directory.path().join("linked");
        symlink(&real_parent, &linked_parent).expect("parent symlink must be created");
        let destination = linked_parent.join("capture.json");

        assert!(publish_actor_race_capture_to_path(&canonical_test_bytes(), &destination).is_err());
        assert!(!real_parent.join("capture.json").exists());
        assert_no_legacy_stages(&real_parent);
    }

    #[test]
    fn symlink_leaf_is_rejected_without_following_or_replacement() {
        let directory = TestDirectory::new("leaf-symlink");
        let target = directory.path().join("target.json");
        fs::write(&target, b"target").expect("symlink target must be written");
        let destination = directory.path().join("capture.json");
        symlink("target.json", &destination).expect("leaf symlink must be created");

        assert!(publish_actor_race_capture_to_path(&canonical_test_bytes(), &destination).is_err());
        assert_eq!(
            fs::read(&target).expect("target must remain readable"),
            b"target"
        );
        assert!(
            fs::symlink_metadata(&destination)
                .expect("leaf symlink must remain")
                .file_type()
                .is_symlink()
        );
        assert_no_legacy_stages(directory.path());
    }

    #[test]
    fn failed_publication_leaves_no_named_stage() {
        let directory = TestDirectory::new("cleanup");
        let destination = directory.path().join("capture.json");
        fs::write(&destination, b"occupied").expect("destination fixture must be written");

        let error = publish_actor_race_capture_to_path(&canonical_test_bytes(), &destination)
            .expect_err("occupied destination must reject publication");
        assert!(error.contains("without replacement"));
        assert_no_legacy_stages(directory.path());
    }

    #[test]
    fn writable_by_other_users_parent_is_rejected_before_staging() {
        let directory = TestDirectory::new("unsafe-parent-mode");
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o720))
            .expect("test parent permissions must change");
        let destination = directory.path().join("capture.json");

        let error = publish_actor_race_capture_to_path(&canonical_test_bytes(), &destination)
            .expect_err("externally writable parent must fail closed");
        assert!(error.contains("must not be writable by group or other users"));
        assert!(!destination.exists());
        assert_no_legacy_stages(directory.path());
    }

    #[test]
    fn rollback_refuses_a_replaced_leaf_and_composes_the_primary_error() {
        let directory = TestDirectory::new("rollback-identity");
        let destination = directory.path().join("capture.json");
        let (parent, leaf) =
            retain_destination_parent(&destination).expect("test parent must be retained");
        require_rollback_safe_parent(&parent).expect("test parent must be rollback-safe");
        let stage_fd = create_anonymous_stage(&parent).expect("anonymous stage must be created");
        fchmod(&stage_fd, STAGE_MODE).expect("anonymous stage mode must be set");
        let identity = require_anonymous_regular_mode_0600(&stage_fd)
            .expect("anonymous stage must have its closed invariants");
        let stage = write_and_sync_stage(stage_fd, &canonical_test_bytes())
            .expect("anonymous stage must be synced");
        linkat(&stage, "", &parent, leaf.as_os_str(), AtFlags::EMPTY_PATH)
            .expect("anonymous stage must be linked");
        let mut published = OwnedPublishedLeaf::new(&parent, &stage, leaf, identity);

        fs::remove_file(&destination).expect("owned leaf must be removed by the test adversary");
        fs::write(&destination, b"replacement").expect("replacement fixture must be created");
        let error = published.rollback_after("forced primary failure".to_owned());

        assert!(error.contains("forced primary failure"));
        assert!(error.contains("different identity"));
        assert_eq!(
            fs::read(&destination).expect("replacement must remain readable"),
            b"replacement"
        );
    }

    #[test]
    fn explicit_rollback_unlinks_only_the_retained_inode() {
        let directory = TestDirectory::new("rollback-owned");
        let destination = directory.path().join("capture.json");
        let (parent, leaf) =
            retain_destination_parent(&destination).expect("test parent must be retained");
        require_rollback_safe_parent(&parent).expect("test parent must be rollback-safe");
        let stage_fd = create_anonymous_stage(&parent).expect("anonymous stage must be created");
        fchmod(&stage_fd, STAGE_MODE).expect("anonymous stage mode must be set");
        let identity = require_anonymous_regular_mode_0600(&stage_fd)
            .expect("anonymous stage must have its closed invariants");
        let stage = write_and_sync_stage(stage_fd, &canonical_test_bytes())
            .expect("anonymous stage must be synced");
        linkat(&stage, "", &parent, leaf.as_os_str(), AtFlags::EMPTY_PATH)
            .expect("anonymous stage must be linked");
        let mut published = OwnedPublishedLeaf::new(&parent, &stage, leaf, identity);

        published
            .rollback_and_sync()
            .expect("the retained published inode must roll back");

        assert!(!destination.exists());
        assert_no_legacy_stages(directory.path());
    }
}
