//! Checked scheduler-owned identities.
//!
//! Every issuer permanently exhausts after returning `u64::MAX`.  The zero
//! value is never observable and published identities are never recycled.

use std::fmt;
use std::num::NonZeroU64;

/// Opaque identity assigned to an accepted request.
///
/// Values are monotonically increasing within one scheduler engine.  Callers
/// may retain and compare an identity, but cannot manufacture one through the
/// public API.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(NonZeroU64);

impl RequestId {
    /// Returns the stable integer representation used in traces and events.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    #[cfg(test)]
    pub(crate) fn try_from_raw_for_test(value: u64) -> Result<Self, &'static str> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or("test request identity must be nonzero")
    }
}

impl fmt::Debug for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("RequestId")
            .field(&self.0.get())
            .finish()
    }
}

/// The scheduler identity space that permanently exhausted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IdentityKind {
    Request,
    EngineTransaction,
    SlotGeneration,
    RoundEpoch,
}

/// A checked identity issuer has no unused value remaining.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IdentityExhausted {
    kind: IdentityKind,
}

impl IdentityExhausted {
    const fn new(kind: IdentityKind) -> Self {
        Self { kind }
    }

    pub(crate) const fn kind(self) -> IdentityKind {
        self.kind
    }
}

/// Shared no-wrap issuer mechanics. `None` is a permanent exhausted state.
#[derive(Debug)]
struct NonZeroIssuer {
    next: Option<NonZeroU64>,
}

impl NonZeroIssuer {
    const fn new() -> Self {
        Self {
            // SAFETY: expressed without `new_unchecked`; one is nonzero.
            next: NonZeroU64::new(1),
        }
    }

    #[cfg(test)]
    const fn from_next(next: u64) -> Self {
        Self {
            next: NonZeroU64::new(next),
        }
    }

    fn issue(&mut self, kind: IdentityKind) -> Result<NonZeroU64, IdentityExhausted> {
        let issued = self.next.ok_or_else(|| IdentityExhausted::new(kind))?;
        self.next = issued.get().checked_add(1).and_then(NonZeroU64::new);
        Ok(issued)
    }

    fn ensure_available(&self, count: usize, kind: IdentityKind) -> Result<(), IdentityExhausted> {
        if count == 0 {
            return Ok(());
        }
        let Some(first) = self.next else {
            return Err(IdentityExhausted::new(kind));
        };
        let Ok(count) = u64::try_from(count) else {
            return Err(IdentityExhausted::new(kind));
        };
        if first.get().checked_add(count - 1).is_none() {
            return Err(IdentityExhausted::new(kind));
        }
        Ok(())
    }

    const fn peek(&self, kind: IdentityKind) -> Result<NonZeroU64, IdentityExhausted> {
        match self.next {
            Some(next) => Ok(next),
            None => Err(IdentityExhausted::new(kind)),
        }
    }
}

macro_rules! checked_identity {
    ($identity:ident, $issuer:ident, $kind:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $identity(NonZeroU64);

        impl $identity {
            #[allow(dead_code, reason = "uniform checked-identity API across issuer kinds")]
            pub(crate) const fn get(self) -> u64 {
                self.0.get()
            }
        }

        impl fmt::Debug for $identity {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($identity))
                    .field(&self.0.get())
                    .finish()
            }
        }

        #[derive(Debug)]
        pub(crate) struct $issuer(NonZeroIssuer);

        impl $issuer {
            pub(crate) const fn new() -> Self {
                Self(NonZeroIssuer::new())
            }

            #[cfg(test)]
            pub(crate) const fn from_next(next: u64) -> Self {
                Self(NonZeroIssuer::from_next(next))
            }

            pub(crate) fn issue(&mut self) -> Result<$identity, IdentityExhausted> {
                self.0.issue(IdentityKind::$kind).map($identity)
            }

            #[allow(dead_code, reason = "uniform checked-identity API across issuer kinds")]
            pub(crate) const fn peek(&self) -> Result<$identity, IdentityExhausted> {
                match self.0.peek(IdentityKind::$kind) {
                    Ok(next) => Ok($identity(next)),
                    Err(error) => Err(error),
                }
            }

            #[allow(dead_code, reason = "uniform checked-identity API across issuer kinds")]
            pub(crate) fn ensure_available(&self, count: usize) -> Result<(), IdentityExhausted> {
                self.0.ensure_available(count, IdentityKind::$kind)
            }
        }
    };
}

/// Issues accepted request identities in retained ingress order.
#[derive(Debug)]
pub(crate) struct RequestIdIssuer(NonZeroIssuer);

impl RequestIdIssuer {
    pub(crate) const fn new() -> Self {
        Self(NonZeroIssuer::new())
    }

    #[cfg(test)]
    const fn from_next(next: u64) -> Self {
        Self(NonZeroIssuer::from_next(next))
    }

    pub(crate) fn issue(&mut self) -> Result<RequestId, IdentityExhausted> {
        self.0.issue(IdentityKind::Request).map(RequestId)
    }

    #[cfg(test)]
    pub(crate) const fn peek(&self) -> Result<RequestId, IdentityExhausted> {
        match self.0.peek(IdentityKind::Request) {
            Ok(next) => Ok(RequestId(next)),
            Err(error) => Err(error),
        }
    }

    pub(crate) fn ensure_available(&self, count: usize) -> Result<(), IdentityExhausted> {
        self.0.ensure_available(count, IdentityKind::Request)
    }
}

checked_identity!(
    EngineTransactionId,
    EngineTransactionIdIssuer,
    EngineTransaction
);
checked_identity!(SlotGeneration, SlotGenerationIssuer, SlotGeneration);
checked_identity!(RoundEpoch, RoundEpochIssuer, RoundEpoch);

/// Generation-tagged index into the scheduler's fixed request-slot table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SlotKey {
    index: usize,
    generation: SlotGeneration,
}

impl SlotKey {
    pub(crate) const fn new(index: usize, generation: SlotGeneration) -> Self {
        Self { index, generation }
    }

    pub(crate) const fn index(self) -> usize {
        self.index
    }

    #[cfg(test)]
    pub(crate) const fn generation(self) -> SlotGeneration {
        self.generation
    }
}

#[cfg(test)]
pub(crate) fn request_id_for_test(value: u64) -> RequestId {
    RequestId::try_from_raw_for_test(value).expect("test request identity must be valid")
}

#[cfg(test)]
pub(crate) fn slot_generation_for_test(value: u64) -> SlotGeneration {
    SlotGeneration(NonZeroU64::new(value).expect("test slot generation must be nonzero"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_issuer_returns_max_once_then_stays_exhausted() {
        let mut issuer = RequestIdIssuer::from_next(u64::MAX - 1);

        assert_eq!(
            issuer.peek().expect("penultimate identity").get(),
            u64::MAX - 1
        );
        assert_eq!(
            issuer.issue().expect("penultimate identity").get(),
            u64::MAX - 1
        );
        assert_eq!(issuer.issue().expect("final identity").get(), u64::MAX);

        for _ in 0..3 {
            let error = issuer.issue().expect_err("issuer must remain exhausted");
            assert_eq!(error.kind(), IdentityKind::Request);
        }
        assert_eq!(
            issuer.peek().expect_err("no value remains").kind(),
            IdentityKind::Request
        );
    }

    #[test]
    fn every_internal_issuer_has_the_same_no_wrap_boundary() {
        let mut transactions = EngineTransactionIdIssuer::from_next(u64::MAX);
        let mut generations = SlotGenerationIssuer::from_next(u64::MAX);
        let mut epochs = RoundEpochIssuer::from_next(u64::MAX);

        assert_eq!(
            transactions.issue().expect("final transaction").get(),
            u64::MAX
        );
        assert_eq!(
            generations.issue().expect("final generation").get(),
            u64::MAX
        );
        assert_eq!(epochs.issue().expect("final epoch").get(), u64::MAX);
        assert_eq!(
            transactions
                .issue()
                .expect_err("transaction exhaustion")
                .kind(),
            IdentityKind::EngineTransaction
        );
        assert_eq!(
            generations
                .issue()
                .expect_err("generation exhaustion")
                .kind(),
            IdentityKind::SlotGeneration
        );
        assert_eq!(
            epochs.issue().expect_err("epoch exhaustion").kind(),
            IdentityKind::RoundEpoch
        );
    }

    #[test]
    fn range_preflight_does_not_consume_or_partially_issue() {
        let mut issuer = RequestIdIssuer::from_next(u64::MAX - 1);
        assert_eq!(
            issuer
                .ensure_available(3)
                .expect_err("only two remain")
                .kind(),
            IdentityKind::Request
        );
        assert_eq!(
            issuer.peek().expect("preflight is nonmutating").get(),
            u64::MAX - 1
        );
        issuer.ensure_available(2).expect("exact remaining range");
        assert_eq!(issuer.issue().expect("first").get(), u64::MAX - 1);
        assert_eq!(issuer.issue().expect("second").get(), u64::MAX);
        issuer
            .ensure_available(0)
            .expect("empty range is always available");
    }

    #[test]
    fn slot_key_distinguishes_reuse_of_one_index() {
        let first = SlotKey::new(7, slot_generation_for_test(41));
        let second = SlotKey::new(7, slot_generation_for_test(42));

        assert_ne!(first, second);
        assert_eq!(first.index(), second.index());
        assert_ne!(first.generation(), second.generation());
    }

    #[test]
    fn debug_contains_only_identity_metadata() {
        assert_eq!(format!("{:?}", request_id_for_test(9)), "RequestId(9)");
        assert_eq!(
            format!("{:?}", EngineTransactionIdIssuer::new()),
            "EngineTransactionIdIssuer(NonZeroIssuer { next: Some(1) })"
        );
    }
}
