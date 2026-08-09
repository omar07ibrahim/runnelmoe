//! Fixed-size bridges from validated scheduler geometry to ledger plans.

use std::fmt;

use runnel_runtime::{AdapterExecutionLayout, StateLayoutAccounting};

use crate::config::SchedulerConfig;
use crate::error::{SchedulerError, SchedulerResult};
use crate::ledger::{LedgerCategory, LedgerCharge, LedgerError, LedgerScope};

const PROMPT_TOKEN_BYTES: u64 = 4;
const REQUEST_RECORD_BYTES: u64 = 512;
const REQUEST_SLOT_BYTES: u64 = 64;
const OUTPUT_EVENT_BYTES: u64 = 64;
const TERMINAL_SLOT_BYTES: u64 = 64;

pub(crate) const SHARED_STATIC_PLAN_LEN: usize = 9;
pub(crate) const ADMISSION_BASE_PLAN_LEN: usize = 5;
pub(crate) const ACTIVE_PLAN_LEN: usize = 2;

/// A complete fixed-size plan passed to one atomic ledger operation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChargePlan<const N: usize> {
    charges: [LedgerCharge; N],
    total_bytes: u64,
}

impl<const N: usize> ChargePlan<N> {
    fn new(charges: [LedgerCharge; N]) -> SchedulerResult<Self> {
        let total_bytes = charges.iter().try_fold(0_u64, |total, charge| {
            total
                .checked_add(charge.bytes())
                .ok_or_else(|| SchedulerError::internal("ledger charge-plan total overflows"))
        })?;
        Ok(Self {
            charges,
            total_bytes,
        })
    }

    #[must_use]
    pub(crate) const fn as_slice(&self) -> &[LedgerCharge] {
        &self.charges
    }

    #[must_use]
    pub(crate) const fn total_bytes(self) -> u64 {
        self.total_bytes
    }

    #[must_use]
    pub(crate) fn category_bytes(self, category: LedgerCategory) -> u64 {
        self.charges
            .iter()
            .filter(|charge| charge.category() == category)
            .map(|charge| charge.bytes())
            .sum()
    }
}

impl<const N: usize> fmt::Debug for ChargePlan<N> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChargePlan")
            .field("charge_count", &N)
            .field("semantic_capacities", &Redacted)
            .finish()
    }
}

/// Produces the complete static shared partition acquired at construction.
pub(crate) fn shared_static_plan(
    config: &SchedulerConfig,
) -> SchedulerResult<ChargePlan<SHARED_STATIC_PLAN_LEN>> {
    let shared = config.shared_static_charges();
    build_shared_static_plan(
        SharedStaticValues {
            actor_command: shared.actor_command_bytes(),
            actor_control: shared.actor_control_bytes(),
            worker_scratch: shared.worker_scratch_bytes(),
            sampling_scratch: shared.sampling_scratch_bytes(),
            coalesced_batch: shared.coalesced_batch_bytes(),
            model_resident: shared.model_resident_bytes(),
            page_pool: shared.page_pool_bytes(),
            trace: shared.trace_bytes(),
            admission_reserve: shared.admission_reserve_bytes(),
        },
        config.shared_static_charge_bytes(),
    )
}

/// Produces one atomic pre-admission reservation. It contains capacities that
/// exist before adapter state allocation and can later be split by category as
/// prompt storage is released before retained result metadata.
pub(crate) fn admission_base_plan(
    config: &SchedulerConfig,
    prompt_tokens: usize,
) -> SchedulerResult<ChargePlan<ADMISSION_BASE_PLAN_LEN>> {
    let prompt_tokens = u64::try_from(prompt_tokens).map_err(|_| {
        SchedulerError::invalid_request(
            "prompt_tokens",
            "does not fit scheduler logical accounting",
        )
    })?;
    let maximum = u64::try_from(config.max_prompt_tokens()).map_err(|_| {
        SchedulerError::internal("validated prompt ceiling does not fit scheduler accounting")
    })?;
    let output_capacity = u64::try_from(config.output_capacity_per_request()).map_err(|_| {
        SchedulerError::internal("validated output capacity does not fit scheduler accounting")
    })?;
    let plan = build_admission_base_plan(prompt_tokens, maximum, output_capacity)?;
    for (category, expected) in [
        (
            LedgerCategory::RequestRecord,
            config.request_record_charge_bytes(),
        ),
        (
            LedgerCategory::RequestSlot,
            config.request_slot_charge_bytes(),
        ),
        (LedgerCategory::Output, config.output_queue_charge_bytes()),
        (
            LedgerCategory::Terminal,
            config.terminal_slot_charge_bytes(),
        ),
    ] {
        if plan.category_bytes(category) != expected {
            return Err(SchedulerError::internal(
                "admission ledger categories diverge from validated configuration",
            ));
        }
    }
    Ok(plan)
}

/// Checked category-wise sum of accepted admission plans for one provisional
/// batch ledger transaction.
pub(crate) fn aggregate_admission_base_plans(
    plans: &[ChargePlan<ADMISSION_BASE_PLAN_LEN>],
) -> SchedulerResult<ChargePlan<ADMISSION_BASE_PLAN_LEN>> {
    let mut totals = [0_u64; ADMISSION_BASE_PLAN_LEN];
    let categories = [
        LedgerCategory::PromptStorage,
        LedgerCategory::RequestRecord,
        LedgerCategory::RequestSlot,
        LedgerCategory::Output,
        LedgerCategory::Terminal,
    ];
    for plan in plans {
        for (index, category) in categories.iter().copied().enumerate() {
            totals[index] = totals[index]
                .checked_add(plan.category_bytes(category))
                .ok_or_else(|| SchedulerError::internal("batch admission ledger plan overflows"))?;
        }
    }
    ChargePlan::new([
        aligned_charge(LedgerCategory::PromptStorage, totals[0])?,
        aligned_charge(LedgerCategory::RequestRecord, totals[1])?,
        aligned_charge(LedgerCategory::RequestSlot, totals[2])?,
        aligned_charge(LedgerCategory::Output, totals[3])?,
        aligned_charge(LedgerCategory::Terminal, totals[4])?,
    ])
}

/// Produces the state and transaction-buffer reservation acquired atomically
/// at FIFO promotion. The actual adapter state layout must remain within the
/// layout envelope authenticated by [`SchedulerConfig`].
pub(crate) fn active_plan<L: StateLayoutAccounting>(
    config: &SchedulerConfig,
    state_layout: L,
) -> SchedulerResult<ChargePlan<ACTIVE_PLAN_LEN>> {
    let state_payload = usize_to_u64(
        state_layout.payload_bytes(),
        "adapter state payload does not fit scheduler accounting",
    )?;
    let state_charge = usize_to_u64(
        state_layout.charge_bytes(),
        "adapter state charge does not fit scheduler accounting",
    )?;
    let maximum = config.worst_case_state_layout();
    let plan = build_active_plan(
        config.execution_layout(),
        state_payload,
        state_charge,
        maximum.payload_bytes(),
        maximum.charge_bytes(),
    )?;
    if plan.category_bytes(LedgerCategory::PendingTransaction)
        != config.pending_transaction_charge_bytes()
    {
        return Err(SchedulerError::internal(
            "pending transaction charge diverges from validated configuration",
        ));
    }
    Ok(plan)
}

/// Maps ledger failures to the scheduler's stable, payload-free taxonomy.
pub(crate) fn map_ledger_error(error: LedgerError) -> SchedulerError {
    match error {
        LedgerError::CapacityExceeded {
            scope,
            required,
            limit,
        } => SchedulerError::resource_exhausted(scope_resource(scope), required, limit),
        LedgerError::ChargeRoundingOverflow { .. } => {
            SchedulerError::internal("validated ledger charge rounding overflowed")
        }
        LedgerError::UnalignedCharge { .. } => {
            SchedulerError::internal("validated ledger charge is not 64-byte aligned")
        }
        LedgerError::ArithmeticOverflow { .. } => {
            SchedulerError::internal("ledger accounting arithmetic overflowed")
        }
        LedgerError::ReleaseUnderflow { .. } => {
            SchedulerError::internal("ledger reservation release underflowed")
        }
        LedgerError::StaleProvisionalAcquisition => {
            SchedulerError::internal("provisional ledger acquisition is stale")
        }
        LedgerError::MutationIdentityExhausted => {
            SchedulerError::internal("ledger mutation identity space is exhausted")
        }
        LedgerError::ReservationPartitionMismatch => {
            SchedulerError::internal("ledger reservation partitions do not match aggregate")
        }
    }
}

#[derive(Clone, Copy)]
struct SharedStaticValues {
    actor_command: u64,
    actor_control: u64,
    worker_scratch: u64,
    sampling_scratch: u64,
    coalesced_batch: u64,
    model_resident: u64,
    page_pool: u64,
    trace: u64,
    admission_reserve: u64,
}

fn build_shared_static_plan(
    values: SharedStaticValues,
    expected_total: u64,
) -> SchedulerResult<ChargePlan<SHARED_STATIC_PLAN_LEN>> {
    let plan = ChargePlan::new([
        aligned_charge(LedgerCategory::ActorCommand, values.actor_command)?,
        aligned_charge(LedgerCategory::ActorControl, values.actor_control)?,
        aligned_charge(LedgerCategory::WorkerScratch, values.worker_scratch)?,
        aligned_charge(LedgerCategory::SamplingScratch, values.sampling_scratch)?,
        aligned_charge(LedgerCategory::CoalescedBatch, values.coalesced_batch)?,
        aligned_charge(LedgerCategory::ModelResident, values.model_resident)?,
        aligned_charge(LedgerCategory::PagePool, values.page_pool)?,
        aligned_charge(LedgerCategory::Trace, values.trace)?,
        aligned_charge(LedgerCategory::AdmissionReserve, values.admission_reserve)?,
    ])?;
    if plan.total_bytes() != expected_total {
        return Err(SchedulerError::internal(
            "shared static ledger categories do not conserve the validated total",
        ));
    }
    Ok(plan)
}

fn build_admission_base_plan(
    prompt_tokens: u64,
    maximum_prompt_tokens: u64,
    output_capacity: u64,
) -> SchedulerResult<ChargePlan<ADMISSION_BASE_PLAN_LEN>> {
    if prompt_tokens == 0 {
        return Err(SchedulerError::invalid_request(
            "prompt_tokens",
            "must be nonzero",
        ));
    }
    if prompt_tokens > maximum_prompt_tokens {
        return Err(SchedulerError::invalid_request(
            "prompt_tokens",
            "exceeds the configured prompt ceiling",
        ));
    }
    let prompt_payload = prompt_tokens
        .checked_mul(PROMPT_TOKEN_BYTES)
        .ok_or_else(|| {
            SchedulerError::invalid_request("prompt_tokens", "prompt storage byte count overflows")
        })?;
    let output_bytes = output_capacity
        .checked_mul(OUTPUT_EVENT_BYTES)
        .ok_or_else(|| SchedulerError::internal("validated output queue byte count overflows"))?;

    ChargePlan::new([
        payload_charge(LedgerCategory::PromptStorage, prompt_payload)?,
        aligned_charge(LedgerCategory::RequestRecord, REQUEST_RECORD_BYTES)?,
        aligned_charge(LedgerCategory::RequestSlot, REQUEST_SLOT_BYTES)?,
        aligned_charge(LedgerCategory::Output, output_bytes)?,
        aligned_charge(LedgerCategory::Terminal, TERMINAL_SLOT_BYTES)?,
    ])
}

fn build_active_plan(
    execution: AdapterExecutionLayout,
    state_payload: u64,
    state_charge: u64,
    maximum_state_payload: u64,
    maximum_state_charge: u64,
) -> SchedulerResult<ChargePlan<ACTIVE_PLAN_LEN>> {
    if state_charge < state_payload {
        return Err(SchedulerError::internal(
            "adapter state charge is smaller than its semantic payload",
        ));
    }
    if state_payload > maximum_state_payload || state_charge > maximum_state_charge {
        return Err(SchedulerError::internal(
            "adapter state charge exceeds the validated configuration envelope",
        ));
    }
    let prepared_payload = usize_to_u64(
        execution.prepared_payload_bytes(),
        "prepared transaction payload does not fit scheduler accounting",
    )?;
    let pending_payload = usize_to_u64(
        execution.pending_payload_bytes(),
        "pending transaction payload does not fit scheduler accounting",
    )?;
    let transaction_payload = prepared_payload
        .checked_add(pending_payload)
        .ok_or_else(|| {
            SchedulerError::internal("combined transaction payload byte count overflows")
        })?;

    ChargePlan::new([
        aligned_charge(LedgerCategory::ActiveState, state_charge)?,
        payload_charge(LedgerCategory::PendingTransaction, transaction_payload)?,
    ])
}

fn payload_charge(category: LedgerCategory, payload: u64) -> SchedulerResult<LedgerCharge> {
    LedgerCharge::from_payload(category, payload).map_err(map_ledger_error)
}

fn aligned_charge(category: LedgerCategory, bytes: u64) -> SchedulerResult<LedgerCharge> {
    LedgerCharge::from_aligned(category, bytes).map_err(map_ledger_error)
}

fn usize_to_u64(value: usize, problem: &'static str) -> SchedulerResult<u64> {
    u64::try_from(value).map_err(|_| SchedulerError::internal(problem))
}

const fn scope_resource(scope: LedgerScope) -> &'static str {
    match scope {
        LedgerScope::Category(category) => category.as_str(),
        LedgerScope::RequestOwned => "request-owned logical memory",
        LedgerScope::Shared => "shared logical memory",
        LedgerScope::Aggregate => "aggregate logical memory",
    }
}

struct Redacted;

impl fmt::Debug for Redacted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use crate::error::ErrorCategory;
    use crate::ledger::{CapacityLedger, LEDGER_CATEGORY_COUNT, LedgerOwnership};
    use runnel_fixture::FixtureArtifact;
    use runnel_format::{Artifact, Limits};
    use runnel_runtime::{BackendRequest, TinyModel};

    use super::*;

    #[derive(Clone, Copy)]
    struct TestStateLayout {
        payload: usize,
        charge: usize,
    }

    impl StateLayoutAccounting for TestStateLayout {
        fn payload_bytes(self) -> usize {
            self.payload
        }

        fn charge_bytes(self) -> usize {
            self.charge
        }
    }

    fn execution(prepared: usize, pending: usize) -> AdapterExecutionLayout {
        AdapterExecutionLayout::new(2, prepared, 32, 32, pending, 192, 5_600).unwrap()
    }

    fn shared_values() -> SharedStaticValues {
        SharedStaticValues {
            actor_command: 2_048,
            actor_control: 64,
            worker_scratch: 192,
            sampling_scratch: 1_024,
            coalesced_batch: 2_048,
            model_resident: 5_632,
            page_pool: 0,
            trace: 1_048_576,
            admission_reserve: 1_048_576,
        }
    }

    #[test]
    fn validated_v3_config_and_actual_state_conserve_all_three_plans() {
        let fixture = FixtureArtifact::build_v3();
        let artifact = Artifact::from_bytes(fixture.to_parts(), Limits::default()).unwrap();
        let model =
            TinyModel::from_artifact_with_backend(&artifact, BackendRequest::Scalar).unwrap();
        let config =
            SchedulerConfig::new(&model, crate::config::SchedulerLimits::evidence()).unwrap();

        let shared = shared_static_plan(&config).unwrap();
        let base = admission_base_plan(&config, 1).unwrap();
        let state = model.state_layout(1, config.state_page_tokens()).unwrap();
        let active = active_plan(&config, state).unwrap();
        assert_eq!(shared.total_bytes(), 2_226_944);
        assert_eq!(base.total_bytes(), 4_800);
        assert_eq!(active.total_bytes(), 1_472);
        assert_eq!(
            base.total_bytes() + active.total_bytes(),
            config.minimum_request_charge_bytes()
        );
        assert_eq!(
            shared.total_bytes() + base.total_bytes() + active.total_bytes(),
            config.minimum_total_charge_bytes()
        );
    }

    #[test]
    fn shared_categories_conserve_the_validated_total_exactly() {
        let values = shared_values();
        let expected = 2_108_160;
        let plan = build_shared_static_plan(values, expected).unwrap();
        assert_eq!(plan.total_bytes(), expected);
        assert_eq!(plan.as_slice().len(), SHARED_STATIC_PLAN_LEN);
        assert_eq!(plan.category_bytes(LedgerCategory::ActorCommand), 2_048);
        assert_eq!(plan.category_bytes(LedgerCategory::ActorControl), 64);
        assert_eq!(plan.category_bytes(LedgerCategory::PagePool), 0);
        for charge in plan.as_slice() {
            assert_eq!(charge.category().ownership(), LedgerOwnership::Shared);
        }

        let error = build_shared_static_plan(values, expected - 64).unwrap_err();
        assert_eq!(error.category(), ErrorCategory::Internal);
    }

    #[test]
    fn unaligned_shared_partition_fails_without_building_a_partial_plan() {
        let mut values = shared_values();
        values.worker_scratch += 1;
        let error = build_shared_static_plan(values, 0).unwrap_err();
        assert_eq!(error.category(), ErrorCategory::Internal);
        assert!(!error.to_string().contains("prompt"));
    }

    #[test]
    fn admission_base_rounds_prompt_once_and_accounts_fixed_slots() {
        let one = build_admission_base_plan(1, 17, 3).unwrap();
        let sixteen = build_admission_base_plan(16, 17, 3).unwrap();
        let seventeen = build_admission_base_plan(17, 17, 3).unwrap();
        assert_eq!(one.category_bytes(LedgerCategory::PromptStorage), 64);
        assert_eq!(sixteen.category_bytes(LedgerCategory::PromptStorage), 64);
        assert_eq!(seventeen.category_bytes(LedgerCategory::PromptStorage), 128);
        assert_eq!(one.category_bytes(LedgerCategory::RequestRecord), 512);
        assert_eq!(one.category_bytes(LedgerCategory::RequestSlot), 64);
        assert_eq!(one.category_bytes(LedgerCategory::Output), 192);
        assert_eq!(one.category_bytes(LedgerCategory::Terminal), 64);
        assert_eq!(one.total_bytes(), 896);
        assert_eq!(seventeen.total_bytes(), 960);
    }

    #[test]
    fn admission_boundaries_are_rejected_with_sanitized_errors() {
        for result in [
            build_admission_base_plan(0, 17, 3),
            build_admission_base_plan(18, 17, 3),
            build_admission_base_plan(u64::MAX, u64::MAX, 3),
        ] {
            let error = result.unwrap_err();
            assert_eq!(error.category(), ErrorCategory::InvalidRequest);
            let debug = format!("{error:?}");
            assert!(!debug.contains("token-contents-sentinel"));
            assert!(!debug.contains("seed-sentinel"));
        }
    }

    #[test]
    fn active_plan_rounds_combined_transaction_payload_once() {
        let layout = TestStateLayout {
            payload: 96,
            charge: 128,
        };
        let plan = build_active_plan(
            execution(1, 63),
            u64::try_from(layout.payload_bytes()).unwrap(),
            u64::try_from(layout.charge_bytes()).unwrap(),
            1_024,
            1_088,
        )
        .unwrap();
        assert_eq!(plan.category_bytes(LedgerCategory::ActiveState), 128);
        assert_eq!(plan.category_bytes(LedgerCategory::PendingTransaction), 64);
        assert_eq!(plan.total_bytes(), 192);
    }

    #[test]
    fn invalid_or_out_of_envelope_state_accounting_fails_closed() {
        for (payload, charge, maximum_payload, maximum_charge) in [
            (65, 64, 128, 128),
            (64, 65, 128, 128),
            (129, 192, 128, 192),
            (128, 256, 128, 192),
        ] {
            let error = build_active_plan(
                execution(1, 63),
                payload,
                charge,
                maximum_payload,
                maximum_charge,
            )
            .unwrap_err();
            assert_eq!(error.category(), ErrorCategory::Internal);
        }
    }

    #[test]
    fn base_and_active_plans_conserve_ledger_categories_and_release_to_zero() {
        let base = build_admission_base_plan(17, 17, 3).unwrap();
        let active = build_active_plan(execution(1, 63), 96, 128, 1_024, 1_088).unwrap();
        let expected = base.total_bytes() + active.total_bytes();
        let mut ledger = CapacityLedger::new(expected, [u64::MAX; LEDGER_CATEGORY_COUNT]);

        let base_reservation = ledger.acquire(base.as_slice()).unwrap();
        let active_reservation = ledger.acquire(active.as_slice()).unwrap();
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.total_used(), expected);
        assert_eq!(snapshot.request_used(), expected);
        assert_eq!(snapshot.shared_used(), 0);
        assert_eq!(
            snapshot.category(LedgerCategory::PendingTransaction).used(),
            64
        );

        ledger.release(active_reservation).unwrap();
        ledger.release(base_reservation).unwrap();
        assert!(ledger.snapshot().current_is_zero());
        assert_eq!(ledger.snapshot().total_peak(), expected);
    }

    #[test]
    fn ledger_errors_map_without_payload_or_identity_strings() {
        let capacity = map_ledger_error(LedgerError::CapacityExceeded {
            scope: LedgerScope::Category(LedgerCategory::Output),
            required: 128,
            limit: 64,
        });
        assert_eq!(capacity.category(), ErrorCategory::ResourceExhausted);
        assert!(capacity.to_string().contains("output"));

        for error in [
            LedgerError::ArithmeticOverflow {
                scope: LedgerScope::Aggregate,
            },
            LedgerError::ReleaseUnderflow {
                scope: LedgerScope::RequestOwned,
                releasing: 64,
                available: 0,
            },
            LedgerError::StaleProvisionalAcquisition,
            LedgerError::MutationIdentityExhausted,
        ] {
            let mapped = map_ledger_error(error);
            assert_eq!(mapped.category(), ErrorCategory::Internal);
            let rendered = format!("{mapped:?} {mapped}");
            assert!(!rendered.contains("prompt-contents-sentinel"));
            assert!(!rendered.contains("token-sentinel"));
            assert!(!rendered.contains("seed-sentinel"));
        }
    }

    #[test]
    fn charge_plan_debug_redacts_semantic_capacities() {
        let plan = build_admission_base_plan(17, 17, 3).unwrap();
        let debug = format!("{plan:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("960"));
        assert!(!debug.contains("17"));
    }

    #[test]
    fn plan_total_overflow_is_checked() {
        let huge =
            LedgerCharge::from_aligned(LedgerCategory::PromptStorage, u64::MAX - 63).unwrap();
        let one = LedgerCharge::from_aligned(LedgerCategory::RequestRecord, 64).unwrap();
        let error = ChargePlan::new([huge, one]).unwrap_err();
        assert_eq!(error.category(), ErrorCategory::Internal);
    }
}
