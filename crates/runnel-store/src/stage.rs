use std::fmt;
use std::str::FromStr;

use runnel_format::Digest;

use crate::StoreError;

const TOKEN_PREFIX: &str = "rmoa-resume-v1";
const STAGE_PREFIX: &str = "rmoa-stage-v1";
const RANDOM_HEX_BYTES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum BlobKind {
    Manifest,
    PageTable,
    Object,
}

impl BlobKind {
    pub(crate) const fn token_name(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::PageTable => "page-table",
            Self::Object => "object",
        }
    }

    pub(crate) const fn error_name(self) -> &'static str {
        self.token_name()
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "manifest" => Some(Self::Manifest),
            "page-table" => Some(Self::PageTable),
            "object" => Some(Self::Object),
            _ => None,
        }
    }
}

/// Opaque authorization to continue one exact, descriptor-relative stage.
///
/// The textual grammar is deliberately closed and canonical:
/// `rmoa-resume-v1|kind|64-lowercase-hex|length|stage-name`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResumeToken {
    kind: BlobKind,
    digest: Digest,
    expected_length: u64,
    stage_name: String,
}

impl ResumeToken {
    pub(crate) fn new(
        kind: BlobKind,
        digest: Digest,
        expected_length: u64,
        random_hex: &str,
    ) -> Result<Self, StoreError> {
        if expected_length == 0 || !is_lower_hex(random_hex, RANDOM_HEX_BYTES) {
            return Err(StoreError::InvalidResumeToken);
        }
        let stage_name = format!(
            "{STAGE_PREFIX}-{}-{}-{expected_length}-{random_hex}",
            kind.token_name(),
            digest.path_component(),
        );
        let token = Self {
            kind,
            digest,
            expected_length,
            stage_name,
        };
        token.validate_stage_name()?;
        Ok(token)
    }

    #[must_use]
    pub const fn digest(&self) -> Digest {
        self.digest
    }

    #[must_use]
    pub const fn expected_length(&self) -> u64 {
        self.expected_length
    }

    #[must_use]
    pub fn kind(&self) -> &'static str {
        self.kind.token_name()
    }

    #[must_use]
    pub fn stage_name(&self) -> &str {
        &self.stage_name
    }

    pub(crate) fn blob_kind(&self) -> BlobKind {
        self.kind
    }

    pub(crate) fn from_stage_name(stage_name: &str) -> Result<Self, StoreError> {
        let remainder = stage_name
            .strip_prefix(&format!("{STAGE_PREFIX}-"))
            .ok_or(StoreError::InvalidResumeToken)?;
        let (kind, remainder) = [BlobKind::Manifest, BlobKind::PageTable, BlobKind::Object]
            .into_iter()
            .find_map(|kind| {
                remainder
                    .strip_prefix(&format!("{}-", kind.token_name()))
                    .map(|rest| (kind, rest))
            })
            .ok_or(StoreError::InvalidResumeToken)?;
        if remainder.len() < 64 + 1 + 1 + 1 + RANDOM_HEX_BYTES {
            return Err(StoreError::InvalidResumeToken);
        }
        let (digest_hex, remainder) = remainder.split_at(64);
        let remainder = remainder
            .strip_prefix('-')
            .ok_or(StoreError::InvalidResumeToken)?;
        let (encoded_length, random_hex) = remainder
            .rsplit_once('-')
            .ok_or(StoreError::InvalidResumeToken)?;
        if !is_lower_hex(digest_hex, 64)
            || !is_lower_hex(random_hex, RANDOM_HEX_BYTES)
            || encoded_length.is_empty()
            || (encoded_length.len() > 1 && encoded_length.starts_with('0'))
        {
            return Err(StoreError::InvalidResumeToken);
        }
        let expected_length = encoded_length
            .parse::<u64>()
            .map_err(|_| StoreError::InvalidResumeToken)?;
        if expected_length == 0 || expected_length.to_string() != encoded_length {
            return Err(StoreError::InvalidResumeToken);
        }
        let digest = format!("sha256:{digest_hex}")
            .parse()
            .map_err(|_| StoreError::InvalidResumeToken)?;
        let token = Self {
            kind,
            digest,
            expected_length,
            stage_name: stage_name.to_owned(),
        };
        token.validate_stage_name()?;
        Ok(token)
    }

    pub(crate) fn validate_for(
        &self,
        kind: BlobKind,
        digest: Digest,
        expected_length: u64,
    ) -> Result<(), StoreError> {
        self.validate_stage_name()?;
        if self.kind != kind {
            return Err(StoreError::ResumeMismatch { field: "kind" });
        }
        if self.digest != digest {
            return Err(StoreError::ResumeMismatch { field: "digest" });
        }
        if self.expected_length != expected_length {
            return Err(StoreError::ResumeMismatch { field: "length" });
        }
        Ok(())
    }

    fn validate_stage_name(&self) -> Result<(), StoreError> {
        let prefix = format!(
            "{STAGE_PREFIX}-{}-{}-{}-",
            self.kind.token_name(),
            self.digest.path_component(),
            self.expected_length,
        );
        let random = self
            .stage_name
            .strip_prefix(&prefix)
            .ok_or(StoreError::InvalidResumeToken)?;
        if !is_lower_hex(random, RANDOM_HEX_BYTES) {
            return Err(StoreError::InvalidResumeToken);
        }
        Ok(())
    }
}

impl fmt::Display for ResumeToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{TOKEN_PREFIX}|{}|{}|{}|{}",
            self.kind.token_name(),
            self.digest.path_component(),
            self.expected_length,
            self.stage_name,
        )
    }
}

impl FromStr for ResumeToken {
    type Err = StoreError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut fields = value.split('|');
        if fields.next() != Some(TOKEN_PREFIX) {
            return Err(StoreError::InvalidResumeToken);
        }
        let kind = fields
            .next()
            .and_then(BlobKind::parse)
            .ok_or(StoreError::InvalidResumeToken)?;
        let digest_hex = fields.next().ok_or(StoreError::InvalidResumeToken)?;
        if !is_lower_hex(digest_hex, 64) {
            return Err(StoreError::InvalidResumeToken);
        }
        let digest = format!("sha256:{digest_hex}")
            .parse()
            .map_err(|_| StoreError::InvalidResumeToken)?;
        let encoded_length = fields.next().ok_or(StoreError::InvalidResumeToken)?;
        if encoded_length.is_empty()
            || (encoded_length.len() > 1 && encoded_length.starts_with('0'))
        {
            return Err(StoreError::InvalidResumeToken);
        }
        let expected_length = encoded_length
            .parse::<u64>()
            .map_err(|_| StoreError::InvalidResumeToken)?;
        if expected_length == 0 || expected_length.to_string() != encoded_length {
            return Err(StoreError::InvalidResumeToken);
        }
        let stage_name = fields
            .next()
            .filter(|name| !name.is_empty())
            .ok_or(StoreError::InvalidResumeToken)?
            .to_owned();
        if fields.next().is_some() {
            return Err(StoreError::InvalidResumeToken);
        }
        let token = Self {
            kind,
            digest,
            expected_length,
            stage_name,
        };
        token.validate_stage_name()?;
        Ok(token)
    }
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use runnel_format::Digest;

    use super::{BlobKind, ResumeToken};
    use crate::StoreError;

    #[test]
    fn resume_token_round_trips_canonically() {
        let digest = Digest::of(b"object");
        let token = ResumeToken::new(
            BlobKind::Object,
            digest,
            65_537,
            "0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        assert_eq!(ResumeToken::from_str(&token.to_string()).unwrap(), token);
    }

    #[test]
    fn token_binds_every_stage_property() {
        let digest = Digest::of(b"object");
        let token = ResumeToken::new(
            BlobKind::Object,
            digest,
            65_537,
            "0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        assert_eq!(
            token
                .validate_for(BlobKind::PageTable, digest, 65_537)
                .unwrap_err(),
            StoreError::ResumeMismatch { field: "kind" }
        );
        assert_eq!(
            token
                .validate_for(BlobKind::Object, Digest::of(b"other"), 65_537)
                .unwrap_err(),
            StoreError::ResumeMismatch { field: "digest" }
        );
        assert_eq!(
            token
                .validate_for(BlobKind::Object, digest, 65_536)
                .unwrap_err(),
            StoreError::ResumeMismatch { field: "length" }
        );
    }

    #[test]
    fn token_rejects_noncanonical_or_tampered_names() {
        let digest = Digest::of(b"object").path_component();
        let malformed = [
            format!(
                "rmoa-resume-v1|object|{digest}|01|rmoa-stage-v1-object-{digest}-1-0123456789abcdef0123456789abcdef"
            ),
            format!(
                "rmoa-resume-v1|object|{digest}|1|rmoa-stage-v1-object-{digest}-2-0123456789abcdef0123456789abcdef"
            ),
            format!(
                "rmoa-resume-v1|object|{digest}|1|rmoa-stage-v1-object-{digest}-1-ABCDEF0123456789abcdef0123456789"
            ),
        ];
        for value in malformed {
            assert_eq!(
                ResumeToken::from_str(&value).unwrap_err(),
                StoreError::InvalidResumeToken
            );
        }
    }
}
