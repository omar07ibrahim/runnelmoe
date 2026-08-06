//! Sealed deterministic scheduler checkpoint instrumentation.
//!
//! The public surface is feature-gated and deliberately concrete: scheduler
//! code never invokes caller-provided callbacks or retains instrumentation.

#![cfg_attr(
    not(any(test, feature = "deterministic-checkpoint-instrumentation")),
    allow(
        dead_code,
        reason = "the concrete plan is exported only by the opt-in instrumentation feature"
    )
)]

use crate::{
    RequestId,
    control::{ControlBinding, ControlDomain},
    error::{SchedulerError, SchedulerResult},
    id::SlotKey,
    request::CancelDisposition,
};
use std::{fmt, mem::size_of};

/// Maximum number of prevalidated directives in one instrumentation plan.
pub const MAX_CHECKPOINT_PLAN_ENTRIES: usize = 64;

/// Stable transactional boundary addressable by deterministic instrumentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointPoint {
    /// Router output and task envelopes are complete; no expert task has run.
    PostRouterPreExpert,
    /// Contributions are finished and validated; publication is not planned.
    ReadyToCommitPrePlan,
    /// Adapter and service permits are validated; the final snapshot is next.
    CompositePermitPreFinalSnapshot,
    /// The final clock and control snapshot is fixed for this position.
    PostFinalSnapshot,
}

/// Closed mutation performed when a matching checkpoint is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointAction {
    /// Record the boundary without changing request control or logical time.
    ObserveOnly,
    /// Publish generation-checked cancellation.
    Cancel,
    /// Advance the engine clock to this request's inclusive deadline.
    ExpireDeadline,
    /// Publish cancellation first, then advance to the inclusive deadline.
    CancelAndExpireDeadline,
}

impl CheckpointAction {
    pub(crate) const fn expires_deadline(self) -> bool {
        matches!(self, Self::ExpireDeadline | Self::CancelAndExpireDeadline)
    }

    const fn cancels(self) -> bool {
        matches!(self, Self::Cancel | Self::CancelAndExpireDeadline)
    }
}

/// One requested deterministic action.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CheckpointDirective {
    request_id: RequestId,
    position: usize,
    point: CheckpointPoint,
    action: CheckpointAction,
}

impl CheckpointDirective {
    #[must_use]
    pub const fn new(
        request_id: RequestId,
        position: usize,
        point: CheckpointPoint,
        action: CheckpointAction,
    ) -> Self {
        Self {
            request_id,
            position,
            point,
            action,
        }
    }

    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn position(self) -> usize {
        self.position
    }

    #[must_use]
    pub const fn point(self) -> CheckpointPoint {
        self.point
    }

    #[must_use]
    pub const fn action(self) -> CheckpointAction {
        self.action
    }
}

impl fmt::Debug for CheckpointDirective {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckpointDirective")
            .field("target", &"<redacted>")
            .field("point", &self.point)
            .field("action", &self.action)
            .finish()
    }
}

/// Whether an expiry action moved the global logical clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeadlineExpirationDisposition {
    AdvancedToDeadline,
    AlreadyAtOrPastDeadline,
}

/// Immutable result of one fired directive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointEffect {
    fire_ordinal: usize,
    cancellation: Option<CancelDisposition>,
    deadline: Option<DeadlineExpirationDisposition>,
}

impl CheckpointEffect {
    #[must_use]
    pub const fn fire_ordinal(self) -> usize {
        self.fire_ordinal
    }

    #[must_use]
    pub const fn cancellation(self) -> Option<CancelDisposition> {
        self.cancellation
    }

    #[must_use]
    pub const fn deadline(self) -> Option<DeadlineExpirationDisposition> {
        self.deadline
    }
}

/// Borrow-independent inspection row synthesized from a bounded plan.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CheckpointRecord {
    directive: CheckpointDirective,
    effect: Option<CheckpointEffect>,
}

impl CheckpointRecord {
    #[must_use]
    pub const fn directive(self) -> CheckpointDirective {
        self.directive
    }

    #[must_use]
    pub const fn effect(self) -> Option<CheckpointEffect> {
        self.effect
    }
}

impl fmt::Debug for CheckpointRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckpointRecord")
            .field("target", &"<redacted>")
            .field("point", &self.directive.point)
            .field("action", &self.directive.action)
            .field("effect", &self.effect)
            .finish()
    }
}

struct BoundCheckpoint {
    directive: CheckpointDirective,
    slot: SlotKey,
    deadline_ns: Option<u64>,
    effect: Option<CheckpointEffect>,
}

/// One engine-bound, allocation-stable deterministic checkpoint plan.
pub struct CheckpointPlan {
    domain: ControlDomain,
    entries: Vec<BoundCheckpoint>,
    fired: usize,
}

impl fmt::Debug for CheckpointPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckpointPlan")
            .field("entry_count", &self.entries.len())
            .field("fired_count", &self.fired)
            .field("targets", &"<redacted>")
            .finish()
    }
}

impl CheckpointPlan {
    pub(crate) fn try_bind(
        domain: ControlDomain,
        bindings: impl ExactSizeIterator<
            Item = SchedulerResult<(CheckpointDirective, SlotKey, Option<u64>)>,
        >,
    ) -> SchedulerResult<Self> {
        let count = bindings.len();
        if count > MAX_CHECKPOINT_PLAN_ENTRIES {
            return Err(SchedulerError::invalid_request(
                "checkpoint directives",
                "count exceeds the instrumentation ceiling",
            ));
        }
        let mut entries = Vec::new();
        entries.try_reserve_exact(count).map_err(|_| {
            SchedulerError::allocation_failure(
                "checkpoint plan entries",
                u64::try_from(count).unwrap_or(u64::MAX).saturating_mul(
                    u64::try_from(size_of::<BoundCheckpoint>()).unwrap_or(u64::MAX),
                ),
            )
        })?;
        for binding in bindings {
            let (directive, slot, deadline_ns) = binding?;
            entries.push(BoundCheckpoint {
                directive,
                slot,
                deadline_ns,
                effect: None,
            });
        }
        entries.sort_unstable_by_key(|entry| {
            (
                entry.directive.request_id,
                entry.directive.position,
                entry.directive.point,
            )
        });
        if entries.windows(2).any(|pair| {
            pair[0].directive.request_id == pair[1].directive.request_id
                && pair[0].directive.position == pair[1].directive.position
                && pair[0].directive.point == pair[1].directive.point
        }) {
            return Err(SchedulerError::invalid_request(
                "checkpoint directives",
                "contain a duplicate target boundary",
            ));
        }
        Ok(Self {
            domain,
            entries,
            fired: 0,
        })
    }

    pub(crate) fn belongs_to(&self, domain: &ControlDomain) -> bool {
        self.domain.same_table(domain)
    }

    #[must_use]
    pub const fn fired_count(&self) -> usize {
        self.fired
    }

    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.fired == self.entries.len()
    }

    pub fn ensure_complete(&self) -> SchedulerResult<()> {
        if self.is_complete() {
            Ok(())
        } else {
            Err(SchedulerError::invalid_request(
                "checkpoint plan",
                "did not reach every directive",
            ))
        }
    }

    pub fn records(
        &self,
    ) -> impl ExactSizeIterator<Item = CheckpointRecord> + DoubleEndedIterator + '_ {
        self.entries.iter().map(|entry| CheckpointRecord {
            directive: entry.directive,
            effect: entry.effect,
        })
    }

    #[cfg(test)]
    pub(crate) fn allocation_fingerprint_for_test(&self) -> (usize, usize) {
        (self.entries.as_ptr() as usize, self.entries.capacity())
    }
}

pub(crate) struct CheckpointContext<'control> {
    pub(crate) point: CheckpointPoint,
    pub(crate) slot: SlotKey,
    pub(crate) request_id: RequestId,
    pub(crate) position: usize,
    pub(crate) control: &'control ControlBinding,
    pub(crate) deadline_ns: Option<u64>,
    pub(crate) monotonic_ns: u64,
}

pub(crate) trait CheckpointDriver {
    fn fire(&mut self, context: CheckpointContext<'_>) -> SchedulerResult<u64>;
}

pub(crate) struct NoCheckpoints;

impl CheckpointDriver for NoCheckpoints {
    #[inline(always)]
    fn fire(&mut self, context: CheckpointContext<'_>) -> SchedulerResult<u64> {
        Ok(context.monotonic_ns)
    }
}

impl CheckpointDriver for CheckpointPlan {
    fn fire(&mut self, context: CheckpointContext<'_>) -> SchedulerResult<u64> {
        let CheckpointContext {
            point,
            slot,
            request_id,
            position,
            control,
            deadline_ns,
            monotonic_ns,
        } = context;
        let Some(entry) = self.entries.iter_mut().find(|entry| {
            entry.directive.request_id == request_id
                && entry.directive.position == position
                && entry.directive.point == point
        }) else {
            return Ok(monotonic_ns);
        };
        if entry.slot != slot || entry.deadline_ns != deadline_ns {
            return Err(SchedulerError::invalid_request(
                "checkpoint plan",
                "target binding is stale",
            ));
        }
        if entry.effect.is_some() {
            return Err(SchedulerError::internal(
                "deterministic checkpoint fired more than once",
            ));
        }

        let cancellation = if entry.directive.action.cancels() {
            Some(control.cancel()?)
        } else {
            None
        };
        let (next_ns, deadline) = if entry.directive.action.expires_deadline() {
            let deadline_ns = deadline_ns.ok_or_else(|| {
                SchedulerError::internal("checkpoint expiry target lost its deadline")
            })?;
            if monotonic_ns < deadline_ns {
                (
                    deadline_ns,
                    Some(DeadlineExpirationDisposition::AdvancedToDeadline),
                )
            } else {
                (
                    monotonic_ns,
                    Some(DeadlineExpirationDisposition::AlreadyAtOrPastDeadline),
                )
            }
        } else {
            (monotonic_ns, None)
        };
        let fire_ordinal = self.fired;
        self.fired = self
            .fired
            .checked_add(1)
            .ok_or_else(|| SchedulerError::internal("checkpoint fire count overflows"))?;
        entry.effect = Some(CheckpointEffect {
            fire_ordinal,
            cancellation,
            deadline,
        });
        Ok(next_ns)
    }
}
