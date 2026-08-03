use std::fmt;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::StoreError;

/// A cheap, clonable cancellation signal shared by storage operations.
///
/// Cancellation is cooperative: bounded read and hash loops call
/// [`Control::check`] between chunks. Dropping a token does not cancel work.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

/// Cooperative cancellation and deadline state for one storage request.
#[derive(Clone, Debug, Default)]
pub struct Control {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
    #[cfg(test)]
    successful_checks_remaining: Option<Arc<AtomicUsize>>,
}

impl Control {
    #[must_use]
    pub fn new(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            deadline,
            #[cfg(test)]
            successful_checks_remaining: None,
        }
    }

    #[must_use]
    pub fn unbounded() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_cancellation(cancellation: CancellationToken) -> Self {
        Self::new(cancellation, None)
    }

    #[must_use]
    pub fn with_deadline(deadline: Instant) -> Self {
        Self::new(CancellationToken::new(), Some(deadline))
    }

    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    #[must_use]
    pub const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Deterministic unit-test seam: allow exactly `count` successful
    /// checkpoints, then make the next checkpoint observe cancellation.
    #[cfg(test)]
    pub(crate) fn cancel_after_checks(count: usize) -> Self {
        Self {
            cancellation: CancellationToken::new(),
            deadline: None,
            successful_checks_remaining: Some(Arc::new(AtomicUsize::new(count))),
        }
    }

    /// Fails if cancellation was requested or the deadline has elapsed.
    /// Cancellation wins when both conditions are already true.
    pub fn check(&self) -> Result<(), StoreError> {
        #[cfg(test)]
        if let Some(remaining) = &self.successful_checks_remaining
            && remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_sub(1)
                })
                .is_err()
        {
            self.cancellation.cancel();
        }
        if self.cancellation.is_cancelled() {
            return Err(StoreError::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(StoreError::DeadlineExceeded);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{CancellationToken, Control};
    use crate::StoreError;

    #[test]
    fn cloned_token_observes_cancellation() {
        let token = CancellationToken::new();
        let control = Control::with_cancellation(token.clone());
        assert!(control.check().is_ok());
        token.cancel();
        assert_eq!(control.check().unwrap_err(), StoreError::Cancelled);
    }

    #[test]
    fn expired_deadline_is_rejected() {
        let control = Control::with_deadline(Instant::now() - Duration::from_nanos(1));
        assert_eq!(control.check().unwrap_err(), StoreError::DeadlineExceeded);
    }

    #[test]
    fn deterministic_checkpoint_cancellation_is_exact() {
        let control = Control::cancel_after_checks(2);
        assert!(control.check().is_ok());
        assert!(control.check().is_ok());
        assert_eq!(control.check().unwrap_err(), StoreError::Cancelled);
        assert_eq!(control.check().unwrap_err(), StoreError::Cancelled);
    }
}
