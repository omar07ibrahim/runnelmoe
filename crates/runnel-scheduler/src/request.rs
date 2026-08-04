use std::fmt;

use runnel_runtime::SamplingPolicy;

use crate::{error::ErrorCategory, id::RequestId};

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
    }
}
