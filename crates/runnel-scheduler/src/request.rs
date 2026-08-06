use std::fmt;

use runnel_runtime::SamplingPolicy;

use crate::{error::ErrorCategory, id::RequestId};

/// Deadline policy for one request offered through two-phase batch admission.
///
/// Absolute deadlines retain [`RequestSpec`] semantics. Release-relative
/// deadlines are resolved exactly once, from the `release_ns` passed to
/// `PreparedAdmission::commit_prepared_batch`; preparation never substitutes
/// an earlier clock observation for that boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BatchDeadline {
    None,
    Absolute(u64),
    AfterRelease(u64),
}

impl fmt::Debug for BatchDeadline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "None",
            Self::Absolute(_) => "Absolute(<redacted>)",
            Self::AfterRelease(_) => "AfterRelease(<redacted>)",
        })
    }
}

/// Borrowed request description validated before any prompt allocation.
#[derive(Clone, Copy, PartialEq)]
pub struct RequestSpec<'a> {
    prompt: &'a [u32],
    max_new_tokens: usize,
    sampling: SamplingPolicy,
    deadline_ns: Option<u64>,
}

impl fmt::Debug for RequestSpec<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestSpec")
            .field("prompt_len", &self.prompt.len())
            .field("max_new_tokens", &self.max_new_tokens)
            .field(
                "sampling",
                &match self.sampling {
                    SamplingPolicy::Greedy => "greedy",
                    SamplingPolicy::Sample(_) => "sampled-redacted",
                },
            )
            .field("has_deadline", &self.deadline_ns.is_some())
            .finish()
    }
}

impl<'a> RequestSpec<'a> {
    #[must_use]
    pub const fn new(
        prompt: &'a [u32],
        max_new_tokens: usize,
        sampling: SamplingPolicy,
        deadline_ns: Option<u64>,
    ) -> Self {
        Self {
            prompt,
            max_new_tokens,
            sampling,
            deadline_ns,
        }
    }

    #[must_use]
    pub const fn prompt(self) -> &'a [u32] {
        self.prompt
    }

    #[must_use]
    pub const fn max_new_tokens(self) -> usize {
        self.max_new_tokens
    }

    #[must_use]
    pub const fn sampling(self) -> SamplingPolicy {
        self.sampling
    }

    #[must_use]
    pub const fn deadline_ns(self) -> Option<u64> {
        self.deadline_ns
    }
}

/// Borrowed request description for atomic two-phase batch admission.
///
/// Use [`Self::absolute`] to preserve the deadline carried by an ordinary
/// [`RequestSpec`], or [`Self::release_relative`] when every admitted request
/// must receive an exact deadline relative to the batch release boundary.
#[derive(Clone, Copy, PartialEq)]
pub struct BatchRequestSpec<'a> {
    request: RequestSpec<'a>,
    deadline: BatchDeadline,
}

impl fmt::Debug for BatchRequestSpec<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BatchRequestSpec")
            .field("request", &self.request)
            .field("deadline", &self.deadline)
            .finish()
    }
}

impl<'a> BatchRequestSpec<'a> {
    /// Converts an ordinary request without changing its absolute deadline.
    #[must_use]
    pub const fn absolute(request: RequestSpec<'a>) -> Self {
        let deadline = match request.deadline_ns() {
            Some(deadline) => BatchDeadline::Absolute(deadline),
            None => BatchDeadline::None,
        };
        Self { request, deadline }
    }

    /// Creates a request whose deadline is exactly
    /// `release_ns + deadline_after_release_ns`.
    #[must_use]
    pub const fn release_relative(
        prompt: &'a [u32],
        max_new_tokens: usize,
        sampling: SamplingPolicy,
        deadline_after_release_ns: u64,
    ) -> Self {
        Self::with_release_relative_deadline(
            RequestSpec::new(prompt, max_new_tokens, sampling, None),
            deadline_after_release_ns,
        )
    }

    /// Reuses an ordinary request description but resolves its deadline from
    /// the later batch release boundary. Any absolute deadline in `request` is
    /// replaced; a zero relative duration is rejected during preparation.
    #[must_use]
    pub const fn with_release_relative_deadline(
        request: RequestSpec<'a>,
        deadline_after_release_ns: u64,
    ) -> Self {
        Self {
            request: RequestSpec::new(
                request.prompt(),
                request.max_new_tokens(),
                request.sampling(),
                None,
            ),
            deadline: BatchDeadline::AfterRelease(deadline_after_release_ns),
        }
    }

    #[must_use]
    pub const fn request(self) -> RequestSpec<'a> {
        self.request
    }

    #[must_use]
    pub const fn deadline(self) -> BatchDeadline {
        self.deadline
    }
}

impl<'a> From<RequestSpec<'a>> for BatchRequestSpec<'a> {
    fn from(request: RequestSpec<'a>) -> Self {
        Self::absolute(request)
    }
}

/// One accepted offer returned by an atomic batch-admission commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcceptedAdmission {
    offered_index: usize,
    request_id: RequestId,
    admitted_ns: u64,
}

impl AcceptedAdmission {
    pub(crate) const fn new(offered_index: usize, request_id: RequestId, admitted_ns: u64) -> Self {
        Self {
            offered_index,
            request_id,
            admitted_ns,
        }
    }

    #[must_use]
    pub const fn offered_index(self) -> usize {
        self.offered_index
    }

    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn admitted_ns(self) -> u64 {
        self.admitted_ns
    }
}

/// One capacity-rejected offer returned by an atomic batch-admission commit.
pub struct RejectedAdmission {
    offered_index: usize,
    error: crate::SchedulerError,
}

impl fmt::Debug for RejectedAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RejectedAdmission")
            .field("offered_index", &self.offered_index)
            .field("error", &self.error)
            .finish()
    }
}

impl RejectedAdmission {
    pub(crate) const fn new(offered_index: usize, error: crate::SchedulerError) -> Self {
        Self {
            offered_index,
            error,
        }
    }

    #[must_use]
    pub const fn offered_index(&self) -> usize {
        self.offered_index
    }

    #[must_use]
    pub const fn error(&self) -> &crate::SchedulerError {
        &self.error
    }
}

/// Compact result of one atomic two-phase batch-admission commit.
///
/// FIFO-prefix acceptance makes accepted IDs contiguous and every rejected
/// suffix item share one pressure error. The result therefore owns no heap
/// buffers; [`Self::accepted`] and [`Self::rejected`] synthesize exact-size
/// iterators and may be retained independently of later engine admissions.
#[must_use = "batch admission outcomes must be observed"]
pub struct BatchAdmission {
    release_ns: u64,
    offered_count: usize,
    first_accepted_id: RequestId,
    accepted_count: usize,
    rejection: crate::SchedulerError,
}

impl fmt::Debug for BatchAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BatchAdmission")
            .field("release_ns", &self.release_ns)
            .field("offered_count", &self.offered_count)
            .field("accepted_count", &self.accepted_count)
            .field("rejected_count", &self.rejected_count())
            .finish()
    }
}

impl BatchAdmission {
    pub(crate) fn new(
        release_ns: u64,
        offered_count: usize,
        first_accepted_id: Option<RequestId>,
        accepted_count: usize,
        rejection: Option<crate::SchedulerError>,
    ) -> Self {
        debug_assert_eq!(first_accepted_id.is_some(), accepted_count != 0);
        debug_assert_eq!(rejection.is_some(), accepted_count != offered_count);
        Self {
            release_ns,
            offered_count,
            first_accepted_id: first_accepted_id.unwrap_or_else(RequestId::first),
            accepted_count,
            rejection: rejection.unwrap_or_else(|| {
                crate::SchedulerError::internal("batch admission has no rejected suffix")
            }),
        }
    }

    #[must_use]
    pub const fn release_ns(&self) -> u64 {
        self.release_ns
    }

    #[must_use]
    pub const fn offered_count(&self) -> usize {
        self.offered_count
    }

    #[must_use]
    pub const fn accepted_count(&self) -> usize {
        self.accepted_count
    }

    #[must_use]
    pub const fn rejected_count(&self) -> usize {
        self.offered_count - self.accepted_count
    }

    /// Iterates the accepted FIFO prefix without retaining a heap allocation.
    pub fn accepted(
        &self,
    ) -> impl ExactSizeIterator<Item = AcceptedAdmission> + DoubleEndedIterator + '_ {
        (0..self.accepted_count).map(move |offered_index| {
            let request_id = self.first_accepted_id.prevalidated_offset(offered_index);
            AcceptedAdmission::new(offered_index, request_id, self.release_ns)
        })
    }

    /// Iterates the capacity-rejected FIFO suffix. Every item carries the same
    /// first-pressure error because no later offer may bypass that boundary.
    pub fn rejected(
        &self,
    ) -> impl ExactSizeIterator<Item = RejectedAdmission> + DoubleEndedIterator + '_ {
        (self.accepted_count..self.offered_count)
            .map(move |offered_index| RejectedAdmission::new(offered_index, self.rejection.clone()))
    }
}

/// Externally observable scheduler phase without transaction payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestPhase {
    Queued,
    Ready,
    Preparing,
    ExpertOwned,
    ReadyToCommit,
    OutputBlocked,
    Terminal,
}

/// Idempotent cancellation result for an existing request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelDisposition {
    Requested,
    AlreadyRequested,
    AlreadyTerminal,
}

/// Why a request can no longer commit model work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalOutcome {
    Completed,
    Cancelled,
    DeadlineExceeded,
    Failed { category: ErrorCategory },
}

/// One committed generated token.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OutputEvent {
    request_id: RequestId,
    output_index: usize,
    token: u32,
}

impl fmt::Debug for OutputEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OutputEvent")
            .field("request_id", &self.request_id)
            .field("output_index", &self.output_index)
            .field("token", &"<redacted>")
            .finish()
    }
}

impl OutputEvent {
    pub(crate) const fn new(request_id: RequestId, output_index: usize, token: u32) -> Self {
        Self {
            request_id,
            output_index,
            token,
        }
    }

    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn output_index(self) -> usize {
        self.output_index
    }

    /// Returns the committed token. Callers must not place this value in logs
    /// or metrics unless prompt/output disclosure was explicitly requested.
    #[must_use]
    pub const fn token(self) -> u32 {
        self.token
    }
}

/// Retained terminal value, independent of the bounded output queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalResult {
    request_id: RequestId,
    outcome: TerminalOutcome,
    committed_positions: usize,
    emitted_tokens: usize,
}

impl TerminalResult {
    pub(crate) const fn new(
        request_id: RequestId,
        outcome: TerminalOutcome,
        committed_positions: usize,
        emitted_tokens: usize,
    ) -> Self {
        Self {
            request_id,
            outcome,
            committed_positions,
            emitted_tokens,
        }
    }

    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn outcome(self) -> TerminalOutcome {
        self.outcome
    }

    #[must_use]
    pub const fn committed_positions(self) -> usize {
        self.committed_positions
    }

    #[must_use]
    pub const fn emitted_tokens(self) -> usize {
        self.emitted_tokens
    }
}

/// Aggregate result of one deterministic engine step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepReport {
    pub promoted_requests: usize,
    pub waves: usize,
    pub selected_positions: usize,
    pub expert_tasks: usize,
    pub expert_groups: usize,
    pub committed_positions: usize,
    pub terminal_decisions: usize,
}

/// Allocation-free control-plane snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineSnapshot {
    pub monotonic_ns: u64,
    pub closed: bool,
    pub queued_requests: usize,
    pub active_requests: usize,
    pub output_blocked_requests: usize,
    pub retained_terminal_results: usize,
    pub ledger_used_bytes: usize,
    pub ledger_peak_bytes: usize,
}

/// Final ownership counts after destructive engine shutdown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShutdownReport {
    pub terminated_requests: usize,
    pub discarded_output_events: usize,
    pub released_request_bytes: usize,
    pub remaining_shared_bytes: usize,
}

#[cfg(test)]
mod tests {
    use runnel_runtime::{SampleConfig, SamplingPolicy};

    use super::*;

    #[test]
    fn request_and_output_debug_redact_tokens_seed_and_deadline() {
        let request = RequestSpec::new(
            &[123_456_789],
            7,
            SamplingPolicy::Sample(SampleConfig {
                seed: 987_654_321,
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
            }),
            Some(456_789_123),
        );
        let request_debug = format!("{request:?}");
        assert!(request_debug.contains("sampled-redacted"));
        for secret in ["123456789", "987654321", "456789123"] {
            assert!(!request_debug.contains(secret));
        }

        let id = crate::id::request_id_for_test(1);
        let event = OutputEvent::new(id, 0, 234_567_891);
        let event_debug = format!("{event:?}");
        assert!(event_debug.contains("<redacted>"));
        assert!(!event_debug.contains("234567891"));

        let relative = BatchRequestSpec::release_relative(
            &[345_678_912],
            3,
            SamplingPolicy::Sample(SampleConfig {
                seed: 876_543_219,
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
            }),
            765_432_198,
        );
        let relative_debug = format!("{relative:?}");
        assert!(relative_debug.contains("AfterRelease(<redacted>)"));
        for secret in ["345678912", "876543219", "765432198"] {
            assert!(!relative_debug.contains(secret));
        }

        let absolute = BatchRequestSpec::absolute(RequestSpec::new(
            &[456_789_123],
            1,
            SamplingPolicy::Greedy,
            Some(654_321_987),
        ));
        let absolute_debug = format!("{absolute:?}");
        assert!(absolute_debug.contains("Absolute(<redacted>)"));
        for secret in ["456789123", "654321987"] {
            assert!(!absolute_debug.contains(secret));
        }
    }

    #[test]
    fn compact_batch_result_reconstructs_terminal_identity_range_and_suffix() {
        let accepted = BatchAdmission::new(
            17,
            2,
            Some(crate::id::request_id_for_test(u64::MAX - 1)),
            2,
            None,
        );
        assert_eq!(
            accepted
                .accepted()
                .map(|entry| (entry.offered_index(), entry.request_id().get()))
                .collect::<Vec<_>>(),
            [(0, u64::MAX - 1), (1, u64::MAX)]
        );
        assert_eq!(accepted.rejected_count(), 0);

        let rejected = BatchAdmission::new(
            18,
            2,
            None,
            0,
            Some(crate::SchedulerError::resource_exhausted(
                "queued request count",
                1,
                0,
            )),
        );
        assert_eq!(rejected.accepted_count(), 0);
        assert_eq!(
            rejected
                .rejected()
                .map(|entry| (entry.offered_index(), entry.error().category()))
                .collect::<Vec<_>>(),
            [
                (0, crate::ErrorCategory::ResourceExhausted),
                (1, crate::ErrorCategory::ResourceExhausted),
            ]
        );
    }
}
