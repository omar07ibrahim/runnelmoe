use std::sync::{
    Mutex, MutexGuard, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use runnel_scheduler::{
    ActorControlCasWitness, ActorDisconnectDisposition, ActorReceiveWitness, ActorTryPopKind,
    ActorTryPopWitness, ActorWakeWitness, CancelDisposition, OutputEvent, RequestCancellation,
    RequestHandle, SchedulerError, SchedulerResult, SubmitCommandWitness, TryRecvOutput,
};
use serde::{Serialize, Serializer, ser::SerializeSeq};

use super::common::{
    ACTION_COUNT, HarnessResult, REQUEST_COUNT, action_ordinal, action_producer, require, to_u64,
    validate_producer_assignment,
};
use super::{Action, actions as generated_actions};

const PRODUCER_COUNT: usize = 2;
const ACTION_COUNTER_FINAL: u64 = 2 * ACTION_COUNT as u64;
const PRODUCER_ACTION_COUNTS: [usize; PRODUCER_COUNT] = [518, 506];
const COMMAND_SLOT_COUNT: u64 = 8;
const REQUEST_SLOT_COUNT: u64 = 16;
const MAX_CONTROL_GENERATION: u64 = u64::MAX >> 3;

const CANCELLED_FLAG: u64 = 1 << 0;
const DISCONNECTED_FLAG: u64 = 1 << 1;
const TERMINAL_FLAG: u64 = 1 << 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RaceKind {
    Submit,
    Cancel,
    ReceiverDrop,
    Drain,
    Wake,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RaceResult {
    SubmitAccepted,
    SubmitOfferExhausted,
    CancelRequested,
    CancelAlreadyRequested,
    CancelAlreadyTerminal,
    ReceiverDropped,
    DrainOutput,
    DrainEmpty,
    DrainEof,
    WakeSignaled,
    TargetUnavailable,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RaceErrorCode {
    RequestNotFound,
    ResourceExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RaceErrorCategory {
    InvalidRequest,
    ResourceExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct RaceError(
    RaceErrorCode,
    RaceErrorCategory,
    Option<&'static str>,
    Option<u64>,
    Option<u64>,
);

impl RaceError {
    const fn request_not_found() -> Self {
        Self(
            RaceErrorCode::RequestNotFound,
            RaceErrorCategory::InvalidRequest,
            None,
            None,
            None,
        )
    }

    const fn request_slots_exhausted() -> Self {
        Self(
            RaceErrorCode::ResourceExhausted,
            RaceErrorCategory::ResourceExhausted,
            Some("request slot count"),
            Some(REQUEST_SLOT_COUNT),
            Some(REQUEST_SLOT_COUNT),
        )
    }

    fn normalize(error: &SchedulerError) -> HarnessResult<Self> {
        match error {
            SchedulerError::RequestNotFound => Ok(Self::request_not_found()),
            SchedulerError::ResourceExhausted {
                resource: "request slot count",
                required: REQUEST_SLOT_COUNT,
                limit: REQUEST_SLOT_COUNT,
            } => Ok(Self::request_slots_exhausted()),
            _ => Err(format!(
                "actor race reached an uncapturable scheduler error: {error}"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct RaceOutput(u64, u64, u64);

impl RaceOutput {
    fn from_event(event: OutputEvent) -> HarnessResult<Self> {
        let request_id = event.request_id().get();
        require(request_id != 0, "race output request ID is zero")?;
        Ok(Self(
            request_id,
            to_u64(event.output_index(), "race output index")?,
            u64::from(event.token()),
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Hex64(u64);

impl Serialize for Hex64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(&format_args!("{:016x}", self.0))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct CommandWitnessCapture(bool, u64, u64, u64);

impl CommandWitnessCapture {
    const fn sentinel() -> Self {
        Self(false, 0, 0, 0)
    }

    fn normalize(witness: SubmitCommandWitness, expected_boundary: bool) -> HarnessResult<Self> {
        Self::normalize_parts(
            witness.command_slot(),
            witness.ticket(),
            witness.ready_commit_sequence(),
            expected_boundary,
        )
    }

    fn normalize_parts(
        slot: Option<usize>,
        ticket: u64,
        ready_sequence: u64,
        expected_boundary: bool,
    ) -> HarnessResult<Self> {
        if !expected_boundary {
            require(slot.is_none(), "command sentinel retained a slot")?;
            require(ticket == 0, "command sentinel retained a ticket")?;
            require(
                ready_sequence == 0,
                "command sentinel retained a ready sequence",
            )?;
            return Ok(Self::sentinel());
        }

        let slot = slot.ok_or_else(|| "reached command omitted its slot".to_owned())?;
        let slot = to_u64(slot, "command slot")?;
        require(slot < COMMAND_SLOT_COUNT, "command slot is out of range")?;
        require(ticket != 0, "reached command ticket is zero")?;
        require(
            ready_sequence != 0,
            "reached command ready sequence is zero",
        )?;
        Ok(Self(true, slot, ticket, ready_sequence))
    }

    const fn boundary_reached(self) -> bool {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct AcceptedWitnessCapture(bool, u64, u64, u64, u64, u64);

impl AcceptedWitnessCapture {
    const fn sentinel() -> Self {
        Self(false, 0, 0, 0, 0, 0)
    }

    const fn from_identity(identity: AcceptedIdentity) -> Self {
        Self(
            true,
            identity.request_id,
            identity.control_slot,
            identity.control_generation,
            identity.endpoint_slot,
            identity.endpoint_generation,
        )
    }

    const fn boundary_reached(self) -> bool {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ControlOperationCapture {
    None,
    Cancel,
    Disconnect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ControlDispositionCapture {
    Requested,
    AlreadyRequested,
    AlreadyTerminal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ControlOutcome {
    Disposition(ControlDispositionCapture),
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct ControlWitnessCapture(
    ControlOperationCapture,
    bool,
    u64,
    u64,
    Hex64,
    Hex64,
    Option<ControlDispositionCapture>,
);

impl ControlWitnessCapture {
    const fn sentinel() -> Self {
        Self(
            ControlOperationCapture::None,
            false,
            0,
            0,
            Hex64(0),
            Hex64(0),
            None,
        )
    }

    fn normalize_cancel(
        result: SchedulerResult<CancelDisposition>,
        witness: ActorControlCasWitness,
        identity: AcceptedIdentity,
    ) -> HarnessResult<(RaceResult, Option<RaceError>, Self)> {
        let (race_result, error, outcome) = match result {
            Ok(CancelDisposition::Requested) => (
                RaceResult::CancelRequested,
                None,
                ControlOutcome::Disposition(ControlDispositionCapture::Requested),
            ),
            Ok(CancelDisposition::AlreadyRequested) => (
                RaceResult::CancelAlreadyRequested,
                None,
                ControlOutcome::Disposition(ControlDispositionCapture::AlreadyRequested),
            ),
            Ok(CancelDisposition::AlreadyTerminal) => (
                RaceResult::CancelAlreadyTerminal,
                None,
                ControlOutcome::Disposition(ControlDispositionCapture::AlreadyTerminal),
            ),
            Err(error) => (
                RaceResult::Error,
                Some(RaceError::normalize(&error)?),
                ControlOutcome::Stale,
            ),
        };
        let capture = Self::normalize_parts(
            ControlOperationCapture::Cancel,
            witness.boundary_reached(),
            to_u64(witness.slot_index(), "cancel control slot")?,
            witness.expected_generation(),
            witness.loaded_word(),
            witness.resulting_word(),
            outcome,
        )?;
        capture.require_identity(identity)?;
        Ok((race_result, error, capture))
    }

    fn normalize_disconnect(
        result: SchedulerResult<ActorDisconnectDisposition>,
        witness: ActorControlCasWitness,
        identity: AcceptedIdentity,
    ) -> HarnessResult<(RaceResult, Option<RaceError>, Self)> {
        let outcome = match result {
            Ok(ActorDisconnectDisposition::Requested) => {
                ControlOutcome::Disposition(ControlDispositionCapture::Requested)
            }
            Ok(ActorDisconnectDisposition::AlreadyRequested) => {
                ControlOutcome::Disposition(ControlDispositionCapture::AlreadyRequested)
            }
            Ok(ActorDisconnectDisposition::AlreadyTerminal) => {
                ControlOutcome::Disposition(ControlDispositionCapture::AlreadyTerminal)
            }
            Err(error) => {
                return Err(format!(
                    "owned race receiver reached an unexpected disconnect error: {error}"
                ));
            }
        };
        let capture = Self::normalize_parts(
            ControlOperationCapture::Disconnect,
            witness.boundary_reached(),
            to_u64(witness.slot_index(), "disconnect control slot")?,
            witness.expected_generation(),
            witness.loaded_word(),
            witness.resulting_word(),
            outcome,
        )?;
        capture.require_identity(identity)?;
        Ok((RaceResult::ReceiverDropped, None, capture))
    }

    fn require_identity(self, identity: AcceptedIdentity) -> HarnessResult<()> {
        require(
            self.2 == identity.control_slot,
            "control witness slot differs from the accepted identity",
        )?;
        require(
            self.3 == identity.control_generation,
            "control witness generation differs from the accepted identity",
        )
    }

    fn validate(self) -> HarnessResult<()> {
        if self.0 == ControlOperationCapture::None {
            return require(
                self == Self::sentinel(),
                "unreached control witness is not its sentinel",
            );
        }
        let outcome = self
            .6
            .map_or(ControlOutcome::Stale, ControlOutcome::Disposition);
        let normalized =
            Self::normalize_parts(self.0, self.1, self.2, self.3, self.4.0, self.5.0, outcome)?;
        require(
            normalized == self,
            "control witness differs from its normalized transition",
        )
    }

    fn normalize_parts(
        operation: ControlOperationCapture,
        boundary: bool,
        slot: u64,
        expected_generation: u64,
        loaded_word: u64,
        resulting_word: u64,
        outcome: ControlOutcome,
    ) -> HarnessResult<Self> {
        require(
            operation != ControlOperationCapture::None,
            "reached control operation is none",
        )?;
        require(boundary, "control operation omitted its boundary")?;
        require(slot < REQUEST_SLOT_COUNT, "control slot is out of range")?;
        require(
            expected_generation != 0 && expected_generation <= MAX_CONTROL_GENERATION,
            "control expected generation is invalid",
        )?;
        Self::validate_packed_word(loaded_word, "loaded")?;
        Self::validate_packed_word(resulting_word, "resulting")?;
        let loaded_generation = loaded_word >> 3;
        let resulting_generation = resulting_word >> 3;
        require(
            loaded_generation != 0 && loaded_generation <= MAX_CONTROL_GENERATION,
            "loaded control generation is invalid",
        )?;
        require(
            resulting_generation != 0 && resulting_generation <= MAX_CONTROL_GENERATION,
            "resulting control generation is invalid",
        )?;

        let disposition = match outcome {
            ControlOutcome::Stale => {
                require(
                    loaded_generation > expected_generation,
                    "stale control outcome did not observe a later generation",
                )?;
                require(
                    resulting_word == loaded_word,
                    "stale control outcome modified its loaded word",
                )?;
                None
            }
            ControlOutcome::Disposition(disposition) => {
                require(
                    loaded_generation == expected_generation,
                    "successful control outcome observed another generation",
                )?;
                require(
                    resulting_generation == expected_generation,
                    "successful control outcome changed generation",
                )?;
                Self::validate_transition(operation, disposition, loaded_word, resulting_word)?;
                Some(disposition)
            }
        };

        Ok(Self(
            operation,
            true,
            slot,
            expected_generation,
            Hex64(loaded_word),
            Hex64(resulting_word),
            disposition,
        ))
    }

    fn validate_transition(
        operation: ControlOperationCapture,
        disposition: ControlDispositionCapture,
        loaded: u64,
        resulting: u64,
    ) -> HarnessResult<()> {
        let cancelled = loaded & CANCELLED_FLAG != 0;
        let disconnected = loaded & DISCONNECTED_FLAG != 0;
        let terminal = loaded & TERMINAL_FLAG != 0;
        match (operation, disposition) {
            (ControlOperationCapture::Cancel, ControlDispositionCapture::Requested) => {
                require(!cancelled && !terminal, "invalid requested-cancel source")?;
                require(
                    resulting == loaded | CANCELLED_FLAG,
                    "requested cancel has an invalid resulting word",
                )
            }
            (ControlOperationCapture::Cancel, ControlDispositionCapture::AlreadyRequested) => {
                require(cancelled && !terminal, "invalid repeated-cancel source")?;
                require(
                    resulting == loaded,
                    "repeated cancel changed its control word",
                )
            }
            (ControlOperationCapture::Cancel, ControlDispositionCapture::AlreadyTerminal) => {
                require(terminal, "terminal cancel omitted the terminal flag")?;
                require(
                    resulting == loaded,
                    "terminal cancel changed its control word",
                )
            }
            (ControlOperationCapture::Disconnect, ControlDispositionCapture::Requested) => {
                require(
                    !disconnected && !terminal,
                    "invalid requested-disconnect source",
                )?;
                require(
                    resulting == loaded | CANCELLED_FLAG | DISCONNECTED_FLAG,
                    "requested disconnect has an invalid resulting word",
                )
            }
            (ControlOperationCapture::Disconnect, ControlDispositionCapture::AlreadyRequested) => {
                require(
                    disconnected && !terminal,
                    "invalid repeated-disconnect source",
                )?;
                require(
                    resulting == loaded,
                    "repeated disconnect changed its control word",
                )
            }
            (ControlOperationCapture::Disconnect, ControlDispositionCapture::AlreadyTerminal) => {
                require(terminal, "terminal disconnect omitted the terminal flag")?;
                let expected = if disconnected {
                    loaded
                } else {
                    loaded | CANCELLED_FLAG | DISCONNECTED_FLAG
                };
                require(
                    resulting == expected,
                    "terminal disconnect has an invalid resulting word",
                )
            }
            (ControlOperationCapture::None, _) => {
                Err("sentinel control operation reached transition validation".to_owned())
            }
        }
    }

    fn validate_packed_word(word: u64, label: &str) -> HarnessResult<()> {
        let message = match label {
            "loaded" => "loaded control word disconnects without cancellation",
            "resulting" => "resulting control word disconnects without cancellation",
            _ => "control word disconnects without cancellation",
        };
        require(
            word & DISCONNECTED_FLAG == 0 || word & CANCELLED_FLAG != 0,
            message,
        )
    }

    const fn boundary_reached(self) -> bool {
        self.1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum PopKindCapture {
    Primary,
    OpportunisticEof,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct PopWitnessCapture(PopKindCapture, bool, u64, u64, u64, u64, Option<RaceOutput>);

impl PopWitnessCapture {
    const fn sentinel(kind: PopKindCapture) -> Self {
        Self(kind, false, 0, 0, 0, 0, None)
    }

    fn normalize(
        expected_kind: PopKindCapture,
        witness: ActorTryPopWitness,
        identity: AcceptedIdentity,
    ) -> HarnessResult<Self> {
        let observed_kind = match witness.kind() {
            ActorTryPopKind::Primary => PopKindCapture::Primary,
            ActorTryPopKind::OpportunisticEof => PopKindCapture::OpportunisticEof,
        };
        require(
            observed_kind == expected_kind,
            "endpoint pop witness kind changed",
        )?;
        if !witness.boundary_reached() {
            require(witness.slot_index() == 0, "pop sentinel retained a slot")?;
            require(
                witness.slot_generation() == 0,
                "pop sentinel retained a generation",
            )?;
            require(
                witness.drained_before() == 0 && witness.drained_after() == 0,
                "pop sentinel retained drain counts",
            )?;
            require(
                witness.consumed_output().is_none(),
                "pop sentinel retained an output",
            )?;
            return Ok(Self::sentinel(expected_kind));
        }

        let slot = to_u64(witness.slot_index(), "endpoint pop slot")?;
        require(
            slot == identity.endpoint_slot,
            "endpoint pop slot differs from the accepted identity",
        )?;
        require(
            witness.slot_generation() == identity.endpoint_generation,
            "endpoint pop generation differs from the accepted identity",
        )?;
        Ok(Self(
            expected_kind,
            true,
            slot,
            witness.slot_generation(),
            to_u64(witness.drained_before(), "endpoint drained-before count")?,
            to_u64(witness.drained_after(), "endpoint drained-after count")?,
            witness
                .consumed_output()
                .map(RaceOutput::from_event)
                .transpose()?,
        ))
    }

    const fn boundary_reached(self) -> bool {
        self.1
    }

    fn validate_position(self, expected_kind: PopKindCapture) -> HarnessResult<()> {
        require(self.0 == expected_kind, "endpoint pop tuple kind changed")?;
        if !self.boundary_reached() {
            return require(
                self == Self::sentinel(expected_kind),
                "unreached endpoint pop is not its sentinel",
            );
        }
        require(
            self.2 < REQUEST_SLOT_COUNT,
            "endpoint pop slot is out of range",
        )?;
        require(self.3 != 0, "endpoint pop generation is zero")?;
        if let Some(output) = self.6 {
            require(output.0 != 0, "endpoint pop output request ID is zero")?;
            require(
                output.2 <= u64::from(u32::MAX),
                "endpoint pop output token is out of range",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NormalizedReceive {
    result: RaceResult,
    output: Option<RaceOutput>,
    primary: PopWitnessCapture,
    opportunistic_eof: PopWitnessCapture,
    cached_eof: bool,
}

impl NormalizedReceive {
    fn normalize(
        result: SchedulerResult<TryRecvOutput>,
        witness: ActorReceiveWitness,
        identity: AcceptedIdentity,
    ) -> HarnessResult<Self> {
        let result = result.map_err(|error| {
            format!("race drain reached an unexpected scheduler error: {error}")
        })?;
        let primary =
            PopWitnessCapture::normalize(PopKindCapture::Primary, witness.primary(), identity)?;
        let opportunistic_eof = PopWitnessCapture::normalize(
            PopKindCapture::OpportunisticEof,
            witness.opportunistic_eof(),
            identity,
        )?;
        let cached_eof = witness.cached_eof();

        let (race_result, output) = match result {
            TryRecvOutput::Output(event) => {
                require(!cached_eof, "output receive claimed cached EOF")?;
                let output = RaceOutput::from_event(event)?;
                require(
                    output.0 == identity.request_id,
                    "drained output request identity changed",
                )?;
                require(
                    primary.boundary_reached() && primary.6 == Some(output),
                    "output receive lacks its exact primary witness",
                )?;
                require(
                    primary.5
                        == primary
                            .4
                            .checked_add(1)
                            .ok_or_else(|| "primary endpoint drain count overflowed".to_owned())?,
                    "output receive did not advance one drain position",
                )?;
                (RaceResult::DrainOutput, Some(output))
            }
            TryRecvOutput::Empty => {
                require(!cached_eof, "empty receive claimed cached EOF")?;
                require(
                    primary.boundary_reached() && primary.6.is_none(),
                    "empty receive has an invalid primary witness",
                )?;
                require(
                    primary.4 == primary.5,
                    "empty receive changed its drain count",
                )?;
                (RaceResult::DrainEmpty, None)
            }
            TryRecvOutput::Eof => {
                if cached_eof {
                    require(
                        !primary.boundary_reached() && !opportunistic_eof.boundary_reached(),
                        "cached EOF entered an endpoint boundary",
                    )?;
                } else {
                    require(
                        primary.boundary_reached() && primary.6.is_none(),
                        "direct EOF has an invalid primary witness",
                    )?;
                    require(primary.4 == primary.5, "direct EOF changed its drain count")?;
                    require(
                        !opportunistic_eof.boundary_reached(),
                        "direct EOF entered an opportunistic boundary",
                    )?;
                }
                (RaceResult::DrainEof, None)
            }
        };

        if opportunistic_eof.boundary_reached() {
            require(
                race_result == RaceResult::DrainOutput,
                "opportunistic EOF followed a non-output receive",
            )?;
            require(
                opportunistic_eof.6.is_none(),
                "opportunistic EOF consumed an output",
            )?;
            require(
                opportunistic_eof.4 == opportunistic_eof.5,
                "opportunistic EOF changed its drain count",
            )?;
            require(
                opportunistic_eof.4 == primary.5,
                "opportunistic EOF did not follow the primary drain",
            )?;
        } else if race_result != RaceResult::DrainEof || !cached_eof {
            require(
                opportunistic_eof == PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
                "unreached opportunistic EOF witness is not its sentinel",
            )?;
        }

        Ok(Self {
            result: race_result,
            output,
            primary,
            opportunistic_eof,
            cached_eof,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct WakeWitnessCapture(bool, bool, u64, u64);

impl WakeWitnessCapture {
    const fn sentinel() -> Self {
        Self(false, false, 0, 0)
    }

    fn normalize(witness: ActorWakeWitness) -> HarnessResult<Self> {
        require(
            witness.after_park_epoch > witness.before_park_epoch,
            "wake did not acknowledge a later park epoch",
        )?;
        Ok(Self(
            true,
            witness.dirty_was_set,
            witness.before_park_epoch,
            witness.after_park_epoch,
        ))
    }

    const fn boundary_reached(self) -> bool {
        self.0
    }
}

/// Exact width-stable action tuple for `runnel.actor-race-history/1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct RaceActionCapture(
    u64,
    u64,
    RaceKind,
    Option<u64>,
    Option<u64>,
    u64,
    u64,
    RaceResult,
    Option<RaceError>,
    Option<u64>,
    Option<RaceOutput>,
    CommandWitnessCapture,
    AcceptedWitnessCapture,
    ControlWitnessCapture,
    PopWitnessCapture,
    PopWitnessCapture,
    bool,
    WakeWitnessCapture,
);

impl RaceActionCapture {
    fn validate(self) -> HarnessResult<()> {
        let Self(
            ordinal,
            producer,
            kind,
            submit_attempt,
            client_index,
            invocation,
            response,
            result,
            error,
            request_id,
            output,
            command,
            accepted,
            control,
            primary,
            opportunistic,
            cached_eof,
            wake,
        ) = self;

        require(ordinal < 1_024, "race action ordinal is out of range")?;
        require(
            producer < PRODUCER_COUNT as u64,
            "race action producer is out of range",
        )?;
        require(
            (1..=2_047).contains(&invocation),
            "race action invocation is out of range",
        )?;
        require(
            (2..=2_048).contains(&response) && response > invocation,
            "race action response interval is invalid",
        )?;
        if let Some(request_id) = request_id {
            require(request_id != 0, "race action request ID is zero")?;
        }
        require(
            error.is_some() == (result == RaceResult::Error),
            "race action error/result presence differs",
        )?;
        require(
            output.is_some() == (result == RaceResult::DrainOutput),
            "race action output/result presence differs",
        )?;
        if let Some(output) = output {
            require(
                Some(output.0) == request_id,
                "race action output differs from its target request",
            )?;
            require(
                output.2 <= u64::from(u32::MAX),
                "race action output token is out of range",
            )?;
        }

        control.validate()?;
        primary.validate_position(PopKindCapture::Primary)?;
        opportunistic.validate_position(PopKindCapture::OpportunisticEof)?;

        let command_expected = kind == RaceKind::Submit
            && submit_attempt.is_some_and(|attempt| attempt < REQUEST_COUNT as u64);
        require(
            command.boundary_reached() == command_expected,
            "race action command boundary differs from its submit attempt",
        )?;
        if command_expected {
            require(
                command.1 < COMMAND_SLOT_COUNT && command.2 != 0 && command.3 != 0,
                "race action retained an invalid reached command witness",
            )?;
        } else {
            require(
                command == CommandWitnessCapture::sentinel(),
                "non-submit action retained a command witness",
            )?;
        }
        require(
            accepted.boundary_reached() == (result == RaceResult::SubmitAccepted),
            "race action accepted boundary differs from its result",
        )?;
        if accepted.boundary_reached() {
            require(
                Some(accepted.1) == request_id,
                "race action accepted request identity differs",
            )?;
            require(
                accepted.2 < REQUEST_SLOT_COUNT
                    && accepted.3 != 0
                    && accepted.3 <= MAX_CONTROL_GENERATION
                    && accepted.4 < REQUEST_SLOT_COUNT
                    && accepted.5 != 0,
                "race action retained an invalid accepted identity",
            )?;
        } else {
            require(
                accepted == AcceptedWitnessCapture::sentinel(),
                "non-accepted action retained an accepted witness",
            )?;
        }

        match kind {
            RaceKind::Submit => {
                let attempt =
                    submit_attempt.ok_or_else(|| "submit action omitted its attempt".to_owned())?;
                require(attempt <= 205, "submit attempt is out of range")?;
                require(
                    producer == attempt % PRODUCER_COUNT as u64,
                    "submit producer assignment changed",
                )?;
                if attempt < REQUEST_COUNT as u64 {
                    require(
                        client_index == Some(attempt),
                        "in-range submit client index changed",
                    )?;
                    match result {
                        RaceResult::SubmitAccepted => {
                            require(error.is_none(), "accepted submit retained an error")?;
                            require(
                                request_id.is_some(),
                                "accepted submit omitted its request ID",
                            )?;
                        }
                        RaceResult::Error => {
                            require(
                                error == Some(RaceError::request_slots_exhausted()),
                                "rejected submit did not retain exact saturation",
                            )?;
                            require(
                                request_id.is_none(),
                                "rejected submit retained a request ID",
                            )?;
                        }
                        _ => return Err("in-range submit has an invalid result".to_owned()),
                    }
                } else {
                    require(
                        client_index.is_none(),
                        "exhausted submit retained a client index",
                    )?;
                    require(
                        result == RaceResult::SubmitOfferExhausted,
                        "exhausted submit has an invalid result",
                    )?;
                    require(
                        request_id.is_none(),
                        "exhausted submit retained a request ID",
                    )?;
                }
                Self::require_noncontrol_sentinels(
                    control,
                    primary,
                    opportunistic,
                    cached_eof,
                    wake,
                )
            }
            RaceKind::Cancel => {
                let client = Self::validate_targeted_action(
                    ordinal,
                    producer,
                    submit_attempt,
                    client_index,
                    true,
                )?;
                require(
                    client < REQUEST_COUNT as u64,
                    "cancel target is out of range",
                )?;
                require(
                    matches!(
                        result,
                        RaceResult::CancelRequested
                            | RaceResult::CancelAlreadyRequested
                            | RaceResult::CancelAlreadyTerminal
                            | RaceResult::TargetUnavailable
                            | RaceResult::Error
                    ),
                    "cancel action has an invalid result",
                )?;
                if result == RaceResult::TargetUnavailable {
                    require(
                        request_id.is_none(),
                        "unavailable cancel retained a request ID",
                    )?;
                    require(
                        control == ControlWitnessCapture::sentinel(),
                        "unavailable cancel retained a control witness",
                    )?;
                } else {
                    require(request_id.is_some(), "called cancel omitted its request ID")?;
                    require(
                        control.0 == ControlOperationCapture::Cancel && control.boundary_reached(),
                        "called cancel omitted its control boundary",
                    )?;
                    Self::validate_cancel_result(result, error, control.6)?;
                }
                Self::require_endpoint_and_wake_sentinels(primary, opportunistic, cached_eof, wake)
            }
            RaceKind::ReceiverDrop => {
                Self::validate_targeted_action(
                    ordinal,
                    producer,
                    submit_attempt,
                    client_index,
                    false,
                )?;
                require(
                    matches!(
                        result,
                        RaceResult::ReceiverDropped | RaceResult::TargetUnavailable
                    ),
                    "receiver-drop action has an invalid result",
                )?;
                if result == RaceResult::TargetUnavailable {
                    require(
                        control == ControlWitnessCapture::sentinel(),
                        "unavailable drop retained a control witness",
                    )?;
                } else {
                    require(request_id.is_some(), "called drop omitted its request ID")?;
                    require(
                        control.0 == ControlOperationCapture::Disconnect
                            && control.boundary_reached(),
                        "called drop omitted its disconnect boundary",
                    )?;
                    require(
                        error.is_none() && control.6.is_some(),
                        "successful drop omitted its disposition",
                    )?;
                }
                Self::require_endpoint_and_wake_sentinels(primary, opportunistic, cached_eof, wake)
            }
            RaceKind::Drain => {
                Self::validate_targeted_action(
                    ordinal,
                    producer,
                    submit_attempt,
                    client_index,
                    false,
                )?;
                require(
                    matches!(
                        result,
                        RaceResult::DrainOutput
                            | RaceResult::DrainEmpty
                            | RaceResult::DrainEof
                            | RaceResult::TargetUnavailable
                    ),
                    "drain action has an invalid result",
                )?;
                require(
                    control == ControlWitnessCapture::sentinel(),
                    "drain retained a control witness",
                )?;
                require(
                    wake == WakeWitnessCapture::sentinel(),
                    "drain retained a wake",
                )?;
                match result {
                    RaceResult::TargetUnavailable => {
                        require(
                            primary == PopWitnessCapture::sentinel(PopKindCapture::Primary)
                                && opportunistic
                                    == PopWitnessCapture::sentinel(
                                        PopKindCapture::OpportunisticEof,
                                    )
                                && !cached_eof,
                            "unavailable drain retained endpoint evidence",
                        )?;
                    }
                    RaceResult::DrainOutput => {
                        require(request_id.is_some(), "output drain omitted its request ID")?;
                        require(
                            primary.boundary_reached() && primary.6 == output && !cached_eof,
                            "output drain has invalid primary evidence",
                        )?;
                        require(
                            primary.5
                                == primary.4.checked_add(1).ok_or_else(|| {
                                    "primary endpoint drain count overflowed".to_owned()
                                })?,
                            "output drain did not advance one position",
                        )?;
                        Self::validate_opportunistic_after_output(primary, opportunistic)?;
                    }
                    RaceResult::DrainEmpty => {
                        require(request_id.is_some(), "empty drain omitted its request ID")?;
                        require(
                            primary.boundary_reached()
                                && primary.6.is_none()
                                && primary.4 == primary.5
                                && opportunistic
                                    == PopWitnessCapture::sentinel(
                                        PopKindCapture::OpportunisticEof,
                                    )
                                && !cached_eof,
                            "empty drain has invalid endpoint evidence",
                        )?;
                    }
                    RaceResult::DrainEof => {
                        require(request_id.is_some(), "EOF drain omitted its request ID")?;
                        let direct = primary.boundary_reached()
                            && primary.6.is_none()
                            && primary.4 == primary.5
                            && opportunistic
                                == PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof)
                            && !cached_eof;
                        let cached = primary
                            == PopWitnessCapture::sentinel(PopKindCapture::Primary)
                            && opportunistic
                                == PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof)
                            && cached_eof;
                        require(direct || cached, "EOF drain has invalid endpoint evidence")?;
                    }
                    _ => unreachable!("drain result was closed above"),
                }
                Ok(())
            }
            RaceKind::Wake => {
                require(submit_attempt.is_none(), "wake retained a submit attempt")?;
                let selector =
                    client_index.ok_or_else(|| "wake omitted its consumed selector".to_owned())?;
                require(
                    selector < REQUEST_COUNT as u64,
                    "wake selector is out of range",
                )?;
                require(
                    producer == ordinal % PRODUCER_COUNT as u64,
                    "wake producer assignment changed",
                )?;
                require(result == RaceResult::WakeSignaled, "wake result changed")?;
                require(request_id.is_none(), "wake retained a request ID")?;
                require(wake.boundary_reached(), "wake omitted its boundary")?;
                require(
                    wake.3 > wake.2,
                    "wake did not acknowledge a later park epoch",
                )?;
                require(
                    control == ControlWitnessCapture::sentinel()
                        && primary == PopWitnessCapture::sentinel(PopKindCapture::Primary)
                        && opportunistic
                            == PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof)
                        && !cached_eof,
                    "wake retained a non-wake witness",
                )
            }
        }
    }

    fn validate_targeted_action(
        _ordinal: u64,
        producer: u64,
        submit_attempt: Option<u64>,
        client_index: Option<u64>,
        opposite: bool,
    ) -> HarnessResult<u64> {
        require(
            submit_attempt.is_none(),
            "targeted action retained a submit attempt",
        )?;
        let client = client_index.ok_or_else(|| "targeted action omitted its index".to_owned())?;
        require(
            client < REQUEST_COUNT as u64,
            "targeted action is out of range",
        )?;
        let home = client % PRODUCER_COUNT as u64;
        let expected = if opposite { 1 - home } else { home };
        require(producer == expected, "targeted producer assignment changed")?;
        Ok(client)
    }

    fn validate_cancel_result(
        result: RaceResult,
        error: Option<RaceError>,
        disposition: Option<ControlDispositionCapture>,
    ) -> HarnessResult<()> {
        let expected = match result {
            RaceResult::CancelRequested => Some(ControlDispositionCapture::Requested),
            RaceResult::CancelAlreadyRequested => Some(ControlDispositionCapture::AlreadyRequested),
            RaceResult::CancelAlreadyTerminal => Some(ControlDispositionCapture::AlreadyTerminal),
            RaceResult::Error => {
                require(
                    error == Some(RaceError::request_not_found()),
                    "failed cancel did not retain exact stale error",
                )?;
                None
            }
            _ => return Err("called cancel has an invalid result".to_owned()),
        };
        require(
            disposition == expected,
            "cancel result differs from its control disposition",
        )
    }

    fn validate_opportunistic_after_output(
        primary: PopWitnessCapture,
        opportunistic: PopWitnessCapture,
    ) -> HarnessResult<()> {
        if !opportunistic.boundary_reached() {
            return require(
                opportunistic == PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
                "unreached opportunistic EOF is not its sentinel",
            );
        }
        require(
            opportunistic.2 == primary.2 && opportunistic.3 == primary.3,
            "opportunistic EOF endpoint identity differs from its primary",
        )?;
        require(
            opportunistic.6.is_none(),
            "opportunistic EOF retained an output",
        )?;
        require(
            opportunistic.4 == primary.5 && opportunistic.5 == opportunistic.4,
            "opportunistic EOF drain counts are not chained",
        )
    }

    fn require_noncontrol_sentinels(
        control: ControlWitnessCapture,
        primary: PopWitnessCapture,
        opportunistic: PopWitnessCapture,
        cached_eof: bool,
        wake: WakeWitnessCapture,
    ) -> HarnessResult<()> {
        require(
            control == ControlWitnessCapture::sentinel(),
            "action retained a nonapplicable control witness",
        )?;
        Self::require_endpoint_and_wake_sentinels(primary, opportunistic, cached_eof, wake)
    }

    fn require_endpoint_and_wake_sentinels(
        primary: PopWitnessCapture,
        opportunistic: PopWitnessCapture,
        cached_eof: bool,
        wake: WakeWitnessCapture,
    ) -> HarnessResult<()> {
        require(
            primary == PopWitnessCapture::sentinel(PopKindCapture::Primary)
                && opportunistic == PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof)
                && !cached_eof
                && wake == WakeWitnessCapture::sentinel(),
            "action retained a nonapplicable endpoint or wake witness",
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RaceActionKey {
    ordinal: u64,
    producer: u8,
    kind: RaceKind,
    submit_attempt: Option<u64>,
    client_index: Option<u64>,
}

impl RaceActionKey {
    /// Authenticates static action fields before either producer reaches the
    /// start barrier. The post-barrier path carries only this fixed-width key.
    fn authenticate(action: &Action) -> HarnessResult<Self> {
        validate_producer_assignment(action)?;
        let ordinal = u64::from(action_ordinal(action));
        let producer = u8::try_from(action_producer(action))
            .map_err(|_| "race producer does not fit u8".to_owned())?;
        require(
            usize::from(producer) < PRODUCER_COUNT,
            "race producer is out of range",
        )?;

        let (kind, submit_attempt, client_index) = match action {
            Action::Submit {
                exhausted,
                request_index,
                submit_attempt,
                ..
            } => {
                let attempt = u64::from(*submit_attempt);
                require(attempt <= 205, "race submit attempt is out of range")?;
                let expected_index = (attempt < REQUEST_COUNT as u64).then_some(attempt);
                require(
                    request_index.0.map(u64::from) == expected_index,
                    "race submit index differs from its attempt",
                )?;
                require(
                    *exhausted == expected_index.is_none(),
                    "race exhausted flag differs from its submit attempt",
                )?;
                (RaceKind::Submit, Some(attempt), expected_index)
            }
            Action::Cancel { request_index, .. } => {
                (RaceKind::Cancel, None, Some(u64::from(*request_index)))
            }
            Action::Drop { request_index, .. } => (
                RaceKind::ReceiverDrop,
                None,
                Some(u64::from(*request_index)),
            ),
            Action::Drain { request_index, .. } => {
                (RaceKind::Drain, None, Some(u64::from(*request_index)))
            }
            Action::Wake { selector_index, .. } => {
                (RaceKind::Wake, None, Some(u64::from(*selector_index)))
            }
        };
        let key = Self {
            ordinal,
            producer,
            kind,
            submit_attempt,
            client_index,
        };
        key.validate()?;
        Ok(key)
    }

    fn validate(self) -> HarnessResult<()> {
        require(
            self.ordinal < ACTION_COUNT as u64,
            "race key ordinal is out of range",
        )?;
        require(
            usize::from(self.producer) < PRODUCER_COUNT,
            "race key producer is out of range",
        )?;
        match self.kind {
            RaceKind::Submit => {
                let attempt = self
                    .submit_attempt
                    .ok_or_else(|| "race submit key omitted its attempt".to_owned())?;
                require(attempt <= 205, "race submit key attempt is out of range")?;
                require(
                    u64::from(self.producer) == attempt % PRODUCER_COUNT as u64,
                    "race submit key producer changed",
                )?;
                let expected_index = (attempt < REQUEST_COUNT as u64).then_some(attempt);
                require(
                    self.client_index == expected_index,
                    "race submit key index changed",
                )
            }
            RaceKind::Cancel | RaceKind::ReceiverDrop | RaceKind::Drain => {
                require(
                    self.submit_attempt.is_none(),
                    "targeted race key retained a submit attempt",
                )?;
                let client_index = self
                    .client_index
                    .ok_or_else(|| "targeted race key omitted its index".to_owned())?;
                require(
                    client_index < REQUEST_COUNT as u64,
                    "targeted race key index is out of range",
                )?;
                let home = client_index % PRODUCER_COUNT as u64;
                let expected = if self.kind == RaceKind::Cancel {
                    1 - home
                } else {
                    home
                };
                require(
                    u64::from(self.producer) == expected,
                    "targeted race key producer changed",
                )
            }
            RaceKind::Wake => {
                require(
                    self.submit_attempt.is_none(),
                    "wake race key retained a submit attempt",
                )?;
                require(
                    self.client_index
                        .is_some_and(|index| index < REQUEST_COUNT as u64),
                    "wake race key selector is invalid",
                )?;
                require(
                    u64::from(self.producer) == self.ordinal % PRODUCER_COUNT as u64,
                    "wake race key producer changed",
                )
            }
        }
    }

    fn capture(
        self,
        invocation: u64,
        response: u64,
        evidence: RaceActionEvidence,
    ) -> RaceActionCapture {
        RaceActionCapture(
            self.ordinal,
            u64::from(self.producer),
            self.kind,
            self.submit_attempt,
            self.client_index,
            invocation,
            response,
            evidence.result,
            evidence.error,
            evidence.request_id,
            evidence.output,
            evidence.command,
            evidence.accepted,
            evidence.control,
            evidence.primary,
            evidence.opportunistic,
            evidence.cached_eof,
            evidence.wake,
        )
    }

    fn matches_capture(self, capture: RaceActionCapture) -> bool {
        capture.0 == self.ordinal
            && capture.1 == u64::from(self.producer)
            && capture.2 == self.kind
            && capture.3 == self.submit_attempt
            && capture.4 == self.client_index
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RaceActionEvidence {
    result: RaceResult,
    error: Option<RaceError>,
    request_id: Option<u64>,
    output: Option<RaceOutput>,
    command: CommandWitnessCapture,
    accepted: AcceptedWitnessCapture,
    control: ControlWitnessCapture,
    primary: PopWitnessCapture,
    opportunistic: PopWitnessCapture,
    cached_eof: bool,
    wake: WakeWitnessCapture,
}

impl RaceActionEvidence {
    const fn with_result(result: RaceResult) -> Self {
        Self {
            result,
            error: None,
            request_id: None,
            output: None,
            command: CommandWitnessCapture::sentinel(),
            accepted: AcceptedWitnessCapture::sentinel(),
            control: ControlWitnessCapture::sentinel(),
            primary: PopWitnessCapture::sentinel(PopKindCapture::Primary),
            opportunistic: PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
            cached_eof: false,
            wake: WakeWitnessCapture::sentinel(),
        }
    }
}

struct RaceHistorySlot {
    key: RaceActionKey,
    started: AtomicBool,
    invocation: OnceLock<u64>,
    published: OnceLock<RaceActionCapture>,
    complete: OnceLock<RaceActionCapture>,
}

impl RaceHistorySlot {
    const fn new(key: RaceActionKey) -> Self {
        Self {
            key,
            started: AtomicBool::new(false),
            invocation: OnceLock::new(),
            published: OnceLock::new(),
            complete: OnceLock::new(),
        }
    }
}

struct RaceHistory {
    counter: AtomicU64,
    slots: Box<[RaceHistorySlot]>,
}

struct RaceInvocationToken {
    ordinal: usize,
    producer: u8,
    invocation: u64,
}

struct PublishedRaceAction {
    ordinal: usize,
    producer: u8,
    invocation: u64,
    capture: RaceActionCapture,
}

impl RaceHistory {
    fn new(actions: &[Action]) -> HarnessResult<Self> {
        require(
            actions.len() == ACTION_COUNT,
            "race history action count changed",
        )?;
        require(
            actions == generated_actions(ACTION_COUNT).as_slice(),
            "race history actions differ from the authenticated corpus",
        )?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(ACTION_COUNT)
            .map_err(|_| "race history allocation failed".to_owned())?;
        let mut producer_counts = [0_usize; PRODUCER_COUNT];
        for (expected_ordinal, action) in actions.iter().enumerate() {
            let key = RaceActionKey::authenticate(action)?;
            require(
                usize::try_from(key.ordinal).ok() == Some(expected_ordinal),
                "race history action order is not consecutive",
            )?;
            producer_counts[usize::from(key.producer)] += 1;
            slots.push(RaceHistorySlot::new(key));
        }
        require(
            producer_counts == PRODUCER_ACTION_COUNTS,
            "race history producer partition changed",
        )?;
        Ok(Self {
            counter: AtomicU64::new(0),
            slots: slots.into_boxed_slice(),
        })
    }

    /// Called immediately before target lookup or scheduler API entry.
    fn begin_action(&self, ordinal: usize, producer: u8) -> HarnessResult<RaceInvocationToken> {
        let slot = self
            .slots
            .get(ordinal)
            .ok_or_else(|| "race action ordinal is out of range".to_owned())?;
        require(
            slot.key.producer == producer,
            "race action reached the wrong producer",
        )?;
        slot.started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "race action was begun more than once".to_owned())?;
        let invocation = self.reserve_counter()?;
        slot.invocation
            .set(invocation)
            .map_err(|_| "race invocation was published more than once".to_owned())?;
        require(
            invocation < ACTION_COUNTER_FINAL,
            "race invocation counter is out of range",
        )?;
        Ok(RaceInvocationToken {
            ordinal,
            producer,
            invocation,
        })
    }

    /// Publishes normalized result and witness evidence before reserving the
    /// response endpoint. The returned token is deliberately non-cloneable.
    fn publish_evidence(
        &self,
        token: RaceInvocationToken,
        evidence: RaceActionEvidence,
    ) -> HarnessResult<PublishedRaceAction> {
        let slot = self
            .slots
            .get(token.ordinal)
            .ok_or_else(|| "published race ordinal is out of range".to_owned())?;
        require(
            slot.key.producer == token.producer,
            "published race producer changed",
        )?;
        require(
            slot.started.load(Ordering::SeqCst),
            "race evidence was published before invocation",
        )?;
        require(
            slot.invocation.get() == Some(&token.invocation),
            "race evidence invocation differs from its reservation",
        )?;
        let provisional_response = token
            .invocation
            .checked_add(1)
            .ok_or_else(|| "race provisional response overflowed".to_owned())?;
        let capture = slot
            .key
            .capture(token.invocation, provisional_response, evidence);
        capture.validate()?;
        require(
            slot.key.matches_capture(capture),
            "published race evidence changed its static action",
        )?;
        slot.published
            .set(capture)
            .map_err(|_| "race evidence was published more than once".to_owned())?;
        Ok(PublishedRaceAction {
            ordinal: token.ordinal,
            producer: token.producer,
            invocation: token.invocation,
            capture,
        })
    }

    fn finish_action(&self, published: PublishedRaceAction) -> HarnessResult<()> {
        let slot = self
            .slots
            .get(published.ordinal)
            .ok_or_else(|| "completed race ordinal is out of range".to_owned())?;
        require(
            slot.key.producer == published.producer,
            "completed race producer changed",
        )?;
        require(
            slot.published.get() == Some(&published.capture),
            "completed race evidence differs from its publication",
        )?;
        let response = self.reserve_counter()?;
        require(
            response <= ACTION_COUNTER_FINAL && response > published.invocation,
            "race response counter is out of range",
        )?;
        let mut complete = published.capture;
        complete.6 = response;
        complete.validate()?;
        require(
            slot.key.matches_capture(complete),
            "completed race record changed its static action",
        )?;
        slot.complete
            .set(complete)
            .map_err(|_| "race action was completed more than once".to_owned())
    }

    fn record_action(
        &self,
        token: RaceInvocationToken,
        evidence: RaceActionEvidence,
    ) -> HarnessResult<()> {
        let published = self.publish_evidence(token, evidence)?;
        self.finish_action(published)
    }

    fn reserve_counter(&self) -> HarnessResult<u64> {
        self.counter
            .fetch_add(1, Ordering::SeqCst)
            .checked_add(1)
            .ok_or_else(|| "race action counter overflowed".to_owned())
    }

    fn counter_value(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    fn into_complete(self) -> HarnessResult<CompleteRaceHistory> {
        let mut seen_counter_values = [false; ACTION_COUNTER_FINAL as usize + 1];
        let mut producer_counts = [0_usize; PRODUCER_COUNT];
        let mut previous = [None; PRODUCER_COUNT];
        for (ordinal, slot) in self.slots.iter().enumerate() {
            require(
                slot.started.load(Ordering::SeqCst),
                "race history retained a vacant action",
            )?;
            let published = slot
                .published
                .get()
                .copied()
                .ok_or_else(|| "race history retained an invoked-only action".to_owned())?;
            require(
                slot.invocation.get() == Some(&published.5),
                "race history invocation publication changed",
            )?;
            let complete = slot
                .complete
                .get()
                .copied()
                .ok_or_else(|| "race history retained an evidence-only action".to_owned())?;
            require(
                usize::try_from(complete.0).ok() == Some(ordinal),
                "race history tuple order differs from ordinal",
            )?;
            complete.validate()?;
            require(
                slot.key.matches_capture(complete),
                "race history differs from its authenticated action",
            )?;
            let mut expected_publication = complete;
            expected_publication.6 = expected_publication
                .5
                .checked_add(1)
                .ok_or_else(|| "race publication interval overflowed".to_owned())?;
            require(
                published == expected_publication,
                "race completion changed published evidence",
            )?;

            let producer = usize::from(slot.key.producer);
            producer_counts[producer] += 1;
            if let Some(previous_response) = previous[producer] {
                require(
                    previous_response < complete.5,
                    "race producer program order was violated",
                )?;
            }
            previous[producer] = Some(complete.6);
            for value in [complete.5, complete.6] {
                let value = usize::try_from(value)
                    .map_err(|_| "race counter value does not fit usize".to_owned())?;
                require(
                    value != 0 && value <= ACTION_COUNTER_FINAL as usize,
                    "race counter value is out of range",
                )?;
                require(!seen_counter_values[value], "race counter value was reused")?;
                seen_counter_values[value] = true;
            }
        }
        require(
            producer_counts == PRODUCER_ACTION_COUNTS,
            "complete race producer partition changed",
        )?;
        require(
            self.counter_value() == ACTION_COUNTER_FINAL,
            "race action counter did not finish at 2048",
        )?;
        require(
            seen_counter_values[1..].iter().all(|seen| *seen),
            "race action counter domain has a gap",
        )?;
        validate_global_submission_structure(&self.slots)?;
        let overlap_pair = select_overlap_pair(&self.slots)?;
        Ok(CompleteRaceHistory {
            slots: self.slots,
            overlap_pair,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum RaceBoundary {
    Command,
    Control,
    PrimaryEndpoint,
    OpportunisticEndpoint,
    Wake,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundaryMask(u8);

impl BoundaryMask {
    const COMMAND: u8 = 1 << 0;
    const CONTROL: u8 = 1 << 1;
    const PRIMARY_ENDPOINT: u8 = 1 << 2;
    const OPPORTUNISTIC_ENDPOINT: u8 = 1 << 3;
    const WAKE: u8 = 1 << 4;

    fn from_capture(capture: RaceActionCapture) -> Self {
        let mut mask = 0;
        if capture.11.boundary_reached() {
            mask |= Self::COMMAND;
        }
        if capture.13.boundary_reached() {
            mask |= Self::CONTROL;
        }
        if capture.14.boundary_reached() {
            mask |= Self::PRIMARY_ENDPOINT;
        }
        if capture.15.boundary_reached() {
            mask |= Self::OPPORTUNISTIC_ENDPOINT;
        }
        if capture.17.boundary_reached() {
            mask |= Self::WAKE;
        }
        Self(mask)
    }

    const fn first(self) -> Option<RaceBoundary> {
        if self.0 & Self::COMMAND != 0 {
            Some(RaceBoundary::Command)
        } else if self.0 & Self::CONTROL != 0 {
            Some(RaceBoundary::Control)
        } else if self.0 & Self::PRIMARY_ENDPOINT != 0 {
            Some(RaceBoundary::PrimaryEndpoint)
        } else if self.0 & Self::OPPORTUNISTIC_ENDPOINT != 0 {
            Some(RaceBoundary::OpportunisticEndpoint)
        } else if self.0 & Self::WAKE != 0 {
            Some(RaceBoundary::Wake)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct RaceOverlapPair(u64, RaceBoundary, u64, RaceBoundary);

struct CompleteRaceHistory {
    slots: Box<[RaceHistorySlot]>,
    overlap_pair: RaceOverlapPair,
}

impl CompleteRaceHistory {
    fn actions(&self) -> impl ExactSizeIterator<Item = RaceActionCapture> + '_ {
        self.slots.iter().map(|slot| {
            *slot
                .complete
                .get()
                .expect("complete race history was validated")
        })
    }

    const fn overlap_pair(&self) -> RaceOverlapPair {
        self.overlap_pair
    }
}

impl Serialize for CompleteRaceHistory {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(ACTION_COUNT))?;
        for action in self.actions() {
            sequence.serialize_element(&action)?;
        }
        sequence.end()
    }
}

fn validate_global_submission_structure(slots: &[RaceHistorySlot]) -> HarnessResult<()> {
    let mut ready_seen = [false; REQUEST_COUNT + 1];
    let mut tickets_seen = [[false; REQUEST_COUNT + 1]; COMMAND_SLOT_COUNT as usize];
    let mut ticket_counts = [0_usize; COMMAND_SLOT_COUNT as usize];
    let mut accepted_by_ready = [None; REQUEST_COUNT + 1];
    let mut command_count = 0_usize;

    for slot in slots {
        let action = *slot
            .complete
            .get()
            .ok_or_else(|| "global race validation found an incomplete action".to_owned())?;
        if !action.11.boundary_reached() {
            continue;
        }
        command_count += 1;
        let command_slot = usize::try_from(action.11.1)
            .map_err(|_| "command slot does not fit usize".to_owned())?;
        let ticket = usize::try_from(action.11.2)
            .map_err(|_| "command ticket does not fit usize".to_owned())?;
        let ready = usize::try_from(action.11.3)
            .map_err(|_| "ready sequence does not fit usize".to_owned())?;
        require(
            ticket != 0 && ticket <= REQUEST_COUNT,
            "command ticket exceeds the closed workload",
        )?;
        require(
            ready != 0 && ready <= REQUEST_COUNT,
            "ready sequence exceeds the closed workload",
        )?;
        require(!ready_seen[ready], "ready sequence was duplicated")?;
        ready_seen[ready] = true;
        require(
            !tickets_seen[command_slot][ticket],
            "command slot ticket was duplicated",
        )?;
        tickets_seen[command_slot][ticket] = true;
        ticket_counts[command_slot] += 1;

        if action.12.boundary_reached() {
            accepted_by_ready[ready] = Some(action.12);
        }
    }

    require(
        command_count == REQUEST_COUNT,
        "race history did not reach exactly 64 commands",
    )?;
    require(
        ready_seen[1..].iter().all(|seen| *seen),
        "race ready sequence is not exactly 1 through 64",
    )?;
    for (slot, count) in ticket_counts.into_iter().enumerate() {
        require(
            tickets_seen[slot][1..1 + count].iter().all(|seen| *seen),
            "command slot tickets have a gap",
        )?;
    }

    let mut last_control_generation = [0_u64; REQUEST_SLOT_COUNT as usize];
    let mut last_endpoint_generation = [0_u64; REQUEST_SLOT_COUNT as usize];
    for (expected_request_id, accepted) in
        (1_u64..).zip(accepted_by_ready.into_iter().skip(1).flatten())
    {
        require(
            accepted.1 == expected_request_id,
            "accepted request IDs differ from ready-FIFO order",
        )?;
        let control_slot = usize::try_from(accepted.2)
            .map_err(|_| "accepted control slot does not fit usize".to_owned())?;
        let endpoint_slot = usize::try_from(accepted.4)
            .map_err(|_| "accepted endpoint slot does not fit usize".to_owned())?;
        require(
            accepted.3 == last_control_generation[control_slot] + 1,
            "accepted control generations do not increase in ready order",
        )?;
        require(
            accepted.5 == last_endpoint_generation[endpoint_slot] + 1,
            "accepted endpoint generations do not increase in ready order",
        )?;
        last_control_generation[control_slot] = accepted.3;
        last_endpoint_generation[endpoint_slot] = accepted.5;
    }
    Ok(())
}

fn select_overlap_pair(slots: &[RaceHistorySlot]) -> HarnessResult<RaceOverlapPair> {
    for left_ordinal in 0..slots.len() {
        let left = *slots[left_ordinal]
            .complete
            .get()
            .ok_or_else(|| "overlap scan found an incomplete left action".to_owned())?;
        let Some(left_boundary) = BoundaryMask::from_capture(left).first() else {
            continue;
        };
        for right_slot in slots.iter().skip(left_ordinal + 1) {
            let right = *right_slot
                .complete
                .get()
                .ok_or_else(|| "overlap scan found an incomplete right action".to_owned())?;
            if left.1 == right.1 || !(left.5 < right.6 && right.5 < left.6) {
                continue;
            }
            let Some(right_boundary) = BoundaryMask::from_capture(right).first() else {
                continue;
            };
            return Ok(RaceOverlapPair(
                left.0,
                left_boundary,
                right.0,
                right_boundary,
            ));
        }
    }
    Err("race history lacks a qualifying cross-producer overlap".to_owned())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AcceptedIdentity {
    request_id: u64,
    control_slot: u64,
    control_generation: u64,
    endpoint_slot: u64,
    endpoint_generation: u64,
}

impl AcceptedIdentity {
    fn from_stored_handle(handle: &RequestHandle) -> HarnessResult<Self> {
        let witness = handle.accepted_request_stress_witness();
        Self::normalize(
            handle.request_id().get(),
            Self {
                request_id: witness.request_id().get(),
                control_slot: to_u64(witness.control_slot_index(), "accepted control slot")?,
                control_generation: witness.control_generation(),
                endpoint_slot: to_u64(witness.endpoint_slot_index(), "accepted endpoint slot")?,
                endpoint_generation: witness.endpoint_generation(),
            },
            REQUEST_SLOT_COUNT,
        )
    }

    fn normalize(
        expected_request_id: u64,
        candidate: Self,
        slot_limit: u64,
    ) -> HarnessResult<Self> {
        require(expected_request_id != 0, "accepted request ID is zero")?;
        require(
            candidate.request_id == expected_request_id,
            "accepted witness request ID differs from its handle",
        )?;
        require(slot_limit != 0, "accepted identity slot limit is zero")?;
        require(
            candidate.control_slot < slot_limit,
            "accepted control slot is out of range",
        )?;
        require(
            candidate.control_generation != 0
                && candidate.control_generation <= MAX_CONTROL_GENERATION,
            "accepted control generation is invalid",
        )?;
        require(
            candidate.endpoint_slot < slot_limit,
            "accepted endpoint slot is out of range",
        )?;
        require(
            candidate.endpoint_generation != 0,
            "accepted endpoint generation is zero",
        )?;
        Ok(candidate)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReceiverOwnership {
    Owned { producer: u8 },
    Consumed,
}

struct AcceptedEntry {
    identity: AcceptedIdentity,
    authority: Option<RequestCancellation>,
    receiver: ReceiverOwnership,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AcceptedMetadata {
    identity: AcceptedIdentity,
    receiver: ReceiverOwnership,
    authority_present: bool,
}

impl AcceptedEntry {
    fn metadata(&self) -> AcceptedMetadata {
        AcceptedMetadata {
            identity: self.identity,
            receiver: self.receiver,
            authority_present: self.authority.is_some(),
        }
    }
}

#[derive(Clone)]
struct CancelTarget {
    identity: AcceptedIdentity,
    authority: RequestCancellation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReceiverTarget {
    identity: AcceptedIdentity,
    ownership: ReceiverOwnership,
}

struct TakenAuthority {
    client_index: usize,
    identity: AcceptedIdentity,
    authority: RequestCancellation,
}

type AcceptedTable = [Option<AcceptedEntry>; REQUEST_COUNT];

struct RaceRegistry {
    entries: Mutex<AcceptedTable>,
}

impl RaceRegistry {
    fn new() -> Self {
        Self {
            entries: Mutex::new(std::array::from_fn(|_| None)),
        }
    }

    fn lock(&self) -> HarnessResult<MutexGuard<'_, AcceptedTable>> {
        self.entries
            .lock()
            .map_err(|_| "accepted registry is poisoned".to_owned())
    }

    /// The caller stores the handle in its producer-local table before this
    /// complete shared publication and before reserving the action response.
    fn publish_stored(
        &self,
        client_index: usize,
        producer: u8,
        handle: &RequestHandle,
    ) -> HarnessResult<AcceptedIdentity> {
        require(
            client_index < REQUEST_COUNT,
            "accepted client index is out of range",
        )?;
        require(
            usize::from(producer) < PRODUCER_COUNT,
            "accepted producer is out of range",
        )?;
        require(
            client_index % PRODUCER_COUNT == usize::from(producer),
            "accepted receiver is not owned by its home producer",
        )?;

        let identity = AcceptedIdentity::from_stored_handle(handle)?;
        let entry = AcceptedEntry {
            identity,
            authority: Some(handle.cancellation()),
            receiver: ReceiverOwnership::Owned { producer },
        };

        let mut entries = self.lock()?;
        require(
            entries[client_index].is_none(),
            "race client was accepted more than once",
        )?;
        for accepted in entries.iter().flatten() {
            require(
                accepted.identity.request_id != identity.request_id,
                "accepted request ID was duplicated",
            )?;
            require(
                (
                    accepted.identity.control_slot,
                    accepted.identity.control_generation,
                ) != (identity.control_slot, identity.control_generation),
                "accepted control identity was duplicated",
            )?;
            require(
                (
                    accepted.identity.endpoint_slot,
                    accepted.identity.endpoint_generation,
                ) != (identity.endpoint_slot, identity.endpoint_generation),
                "accepted endpoint identity was duplicated",
            )?;
        }
        entries[client_index] = Some(entry);
        Ok(identity)
    }

    /// Clones under the publication mutex and returns before any control API
    /// is entered.
    fn lookup_cancel(&self, client_index: usize) -> HarnessResult<Option<CancelTarget>> {
        require(
            client_index < REQUEST_COUNT,
            "cancellation client index is out of range",
        )?;
        let target = {
            let entries = self.lock()?;
            let Some(entry) = entries[client_index].as_ref() else {
                return Ok(None);
            };
            let authority = entry.authority.as_ref().cloned().ok_or_else(|| {
                format!("accepted client {client_index} cancellation authority was already taken")
            })?;
            CancelTarget {
                identity: entry.identity,
                authority,
            }
        };
        Ok(Some(target))
    }

    /// Returns only metadata; the sole receiver remains producer-local.
    fn lookup_receiver(
        &self,
        client_index: usize,
        producer: u8,
    ) -> HarnessResult<Option<ReceiverTarget>> {
        require(
            client_index < REQUEST_COUNT,
            "receiver client index is out of range",
        )?;
        require(
            usize::from(producer) < PRODUCER_COUNT,
            "receiver producer is out of range",
        )?;
        require(
            client_index % PRODUCER_COUNT == usize::from(producer),
            "receiver lookup is not on its home producer",
        )?;
        Ok(self.lock()?[client_index]
            .as_ref()
            .map(|entry| ReceiverTarget {
                identity: entry.identity,
                ownership: entry.receiver,
            }))
    }

    /// Called only after the producer-local handle was actually consumed.
    fn mark_receiver_consumed(
        &self,
        client_index: usize,
        producer: u8,
        expected: AcceptedIdentity,
    ) -> HarnessResult<()> {
        require(
            client_index < REQUEST_COUNT,
            "consumed receiver client index is out of range",
        )?;
        require(
            usize::from(producer) < PRODUCER_COUNT,
            "consumed receiver producer is out of range",
        )?;
        require(
            client_index % PRODUCER_COUNT == usize::from(producer),
            "receiver was consumed by a non-home producer",
        )?;
        let mut entries = self.lock()?;
        let entry = entries[client_index]
            .as_mut()
            .ok_or_else(|| format!("client {client_index} receiver was never accepted"))?;
        require(entry.identity == expected, "race receiver identity changed")?;
        require(
            entry.receiver == ReceiverOwnership::Owned { producer },
            "race receiver was already consumed",
        )?;
        entry.receiver = ReceiverOwnership::Consumed;
        Ok(())
    }

    fn snapshot(&self) -> HarnessResult<RaceRegistrySnapshot> {
        let entries = self.lock()?;
        Ok(RaceRegistrySnapshot {
            entries: std::array::from_fn(|index| {
                entries[index].as_ref().map(AcceptedEntry::metadata)
            }),
        })
    }

    /// Validates every entry before mutation. The returned authorities retain
    /// ascending client order and are always dropped outside the mutex.
    fn take_authorities(&self) -> HarnessResult<Vec<TakenAuthority>> {
        let mut taken = Vec::new();
        taken
            .try_reserve_exact(REQUEST_COUNT)
            .map_err(|_| "cancellation-authority collection allocation failed".to_owned())?;

        let mut entries = self.lock()?;
        for entry in entries.iter().flatten() {
            require(
                entry.authority.is_some(),
                "race cancellation authority was already taken",
            )?;
        }
        for (client_index, entry) in entries.iter_mut().enumerate() {
            let Some(entry) = entry else {
                continue;
            };
            let authority = entry
                .authority
                .take()
                .ok_or_else(|| format!("validated client {client_index} authority disappeared"))?;
            taken.push(TakenAuthority {
                client_index,
                identity: entry.identity,
                authority,
            });
        }
        Ok(taken)
    }
}

#[derive(Clone, Debug)]
struct RaceRegistrySnapshot {
    entries: [Option<AcceptedMetadata>; REQUEST_COUNT],
}

impl RaceRegistrySnapshot {
    fn get(&self, client_index: usize) -> Option<AcceptedMetadata> {
        self.entries.get(client_index).copied().flatten()
    }

    fn accepted_count(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    fn live_receiver_count(&self) -> usize {
        self.entries
            .iter()
            .flatten()
            .filter(|entry| matches!(entry.receiver, ReceiverOwnership::Owned { .. }))
            .count()
    }

    fn accepted_ids(&self) -> [u64; REQUEST_COUNT] {
        std::array::from_fn(|index| {
            self.entries[index].map_or(0, |entry| entry.identity.request_id)
        })
    }

    fn require_all_authorities_present(&self) -> HarnessResult<()> {
        require(
            self.entries
                .iter()
                .flatten()
                .all(|entry| entry.authority_present),
            "accepted registry snapshot has a missing authority",
        )
    }

    fn require_all_authorities_absent(&self) -> HarnessResult<()> {
        require(
            self.entries
                .iter()
                .flatten()
                .all(|entry| !entry.authority_present),
            "accepted registry retained an authority after take",
        )
    }
}

#[cfg(test)]
mod tests {
    use runnel_scheduler::{ActorRequestDropWitnessSink, RequestHandle};

    use super::*;
    use crate::common::{
        DropWitnessContext, authenticated_workload, drop_receiver_with_preallocated_witness,
        finish_receiver, receive_once_with_witness, request_spec, spawn_fresh_actor,
        validate_shutdown,
    };

    fn valid_identity() -> AcceptedIdentity {
        AcceptedIdentity {
            request_id: 1,
            control_slot: 0,
            control_generation: 1,
            endpoint_slot: 0,
            endpoint_generation: 1,
        }
    }

    fn control_word(generation: u64, flags: u64) -> u64 {
        (generation << 3) | flags
    }

    fn synthetic_evidence(key: RaceActionKey) -> RaceActionEvidence {
        match key.kind {
            RaceKind::Submit => {
                let attempt = key.submit_attempt.expect("submit attempt");
                if attempt >= REQUEST_COUNT as u64 {
                    return RaceActionEvidence::with_result(RaceResult::SubmitOfferExhausted);
                }
                let identity = AcceptedIdentity {
                    request_id: attempt + 1,
                    control_slot: attempt % REQUEST_SLOT_COUNT,
                    control_generation: attempt / REQUEST_SLOT_COUNT + 1,
                    endpoint_slot: attempt % REQUEST_SLOT_COUNT,
                    endpoint_generation: attempt / REQUEST_SLOT_COUNT + 1,
                };
                let mut evidence = RaceActionEvidence::with_result(RaceResult::SubmitAccepted);
                evidence.request_id = Some(identity.request_id);
                evidence.command = CommandWitnessCapture(
                    true,
                    attempt % COMMAND_SLOT_COUNT,
                    attempt / COMMAND_SLOT_COUNT + 1,
                    attempt + 1,
                );
                evidence.accepted = AcceptedWitnessCapture::from_identity(identity);
                evidence
            }
            RaceKind::Cancel | RaceKind::ReceiverDrop | RaceKind::Drain => {
                RaceActionEvidence::with_result(RaceResult::TargetUnavailable)
            }
            RaceKind::Wake => {
                let mut evidence = RaceActionEvidence::with_result(RaceResult::WakeSignaled);
                evidence.wake = WakeWitnessCapture(true, false, key.ordinal + 1, key.ordinal + 2);
                evidence
            }
        }
    }

    fn record_synthetic(history: &RaceHistory, ordinal: usize) {
        let key = history.slots[ordinal].key;
        let token = history
            .begin_action(ordinal, key.producer)
            .expect("synthetic invocation");
        history
            .record_action(token, synthetic_evidence(key))
            .expect("synthetic completion");
    }

    fn complete_synthetic_history() -> RaceHistory {
        let workload = authenticated_workload().expect("authenticated workload");
        let history = RaceHistory::new(&workload.actions).expect("race history");

        // Ordinal zero is a producer-zero unavailable drain. Complete it, then
        // deliberately overlap the first producer-zero and producer-one
        // commands without weakening either producer's program order.
        record_synthetic(&history, 0);
        let left_key = history.slots[1].key;
        let right_key = history.slots[2].key;
        let left = history
            .begin_action(1, left_key.producer)
            .expect("left invocation");
        let right = history
            .begin_action(2, right_key.producer)
            .expect("right invocation");
        let left = history
            .publish_evidence(left, synthetic_evidence(left_key))
            .expect("left evidence publication");
        let right = history
            .publish_evidence(right, synthetic_evidence(right_key))
            .expect("right evidence publication");
        history.finish_action(left).expect("left completion");
        history.finish_action(right).expect("right completion");

        for ordinal in 3..ACTION_COUNT {
            record_synthetic(&history, ordinal);
        }
        history
    }

    fn overlap_slot(key: RaceActionKey, capture: RaceActionCapture) -> RaceHistorySlot {
        let slot = RaceHistorySlot::new(key);
        slot.complete.set(capture).expect("overlap capture");
        slot
    }

    #[test]
    fn complete_history_freezes_counter_program_order_and_overlap() {
        let history = complete_synthetic_history();
        assert_eq!(history.counter_value(), ACTION_COUNTER_FINAL);
        let complete = history.into_complete().expect("complete race history");
        assert_eq!(complete.actions().len(), ACTION_COUNT);
        assert_eq!(complete.actions().next().expect("first action").0, 0);
        assert_eq!(complete.actions().last().expect("last action").0, 1_023);
        assert_eq!(
            complete.overlap_pair(),
            RaceOverlapPair(1, RaceBoundary::Command, 2, RaceBoundary::Command)
        );
        let encoded = serde_json::to_value(&complete).expect("history JSON");
        assert_eq!(
            encoded.as_array().expect("action array").len(),
            ACTION_COUNT
        );
    }

    #[test]
    fn global_validation_allows_unused_command_slots_without_panicking() {
        let mut history = complete_synthetic_history();
        let mut next_ticket = 0_u64;
        for slot in &mut history.slots {
            if !slot
                .complete
                .get()
                .expect("complete action")
                .11
                .boundary_reached()
            {
                continue;
            }
            next_ticket += 1;
            let ready_sequence = slot.complete.get().expect("complete action").11.3;
            slot.published.get_mut().expect("published action").11 =
                CommandWitnessCapture(true, 0, next_ticket, ready_sequence);
            slot.complete.get_mut().expect("complete action").11 =
                CommandWitnessCapture(true, 0, next_ticket, ready_sequence);
        }
        assert_eq!(next_ticket, REQUEST_COUNT as u64);
        history
            .into_complete()
            .expect("single command-slot history remains valid");
    }

    #[test]
    fn evidence_publication_precedes_response_reservation() {
        let workload = authenticated_workload().expect("authenticated workload");
        let history = RaceHistory::new(&workload.actions).expect("race history");
        let key = history.slots[0].key;
        let invocation = history.begin_action(0, key.producer).expect("invocation");
        assert_eq!(history.counter_value(), 1);
        let published = history
            .publish_evidence(invocation, synthetic_evidence(key))
            .expect("evidence publication");
        assert!(history.slots[0].published.get().is_some());
        assert!(history.slots[0].complete.get().is_none());
        assert_eq!(history.counter_value(), 1);
        history
            .finish_action(published)
            .expect("response completion");
        assert_eq!(history.counter_value(), 2);
        assert!(history.slots[0].complete.get().is_some());
    }

    #[test]
    fn duplicate_begin_fails_before_consuming_another_counter_value() {
        let workload = authenticated_workload().expect("authenticated workload");
        let history = RaceHistory::new(&workload.actions).expect("race history");
        let producer = history.slots[0].key.producer;
        let _token = history.begin_action(0, producer).expect("first invocation");
        assert_eq!(history.counter_value(), 1);
        assert!(history.begin_action(0, producer).is_err());
        assert_eq!(history.counter_value(), 1);
    }

    #[test]
    fn collection_rejects_every_incomplete_slot_phase() {
        let workload = authenticated_workload().expect("authenticated workload");
        assert!(
            RaceHistory::new(&workload.actions)
                .expect("vacant history")
                .into_complete()
                .is_err()
        );

        let invoked = RaceHistory::new(&workload.actions).expect("invoked history");
        let key = invoked.slots[0].key;
        let _token = invoked
            .begin_action(0, key.producer)
            .expect("invoked action");
        assert!(invoked.into_complete().is_err());

        let published = RaceHistory::new(&workload.actions).expect("published history");
        let key = published.slots[0].key;
        let token = published
            .begin_action(0, key.producer)
            .expect("published invocation");
        let _published = published
            .publish_evidence(token, synthetic_evidence(key))
            .expect("published evidence");
        assert!(published.into_complete().is_err());
    }

    #[test]
    fn history_authenticates_static_actions_and_counter_domain() {
        let workload = authenticated_workload().expect("authenticated workload");
        let mut reordered = workload.actions.clone();
        reordered.swap(0, 1);
        assert!(RaceHistory::new(&reordered).is_err());
        let mut altered_selector = workload.actions.clone();
        let Action::Drain { request_index, .. } = &mut altered_selector[0] else {
            panic!("frozen first action changed");
        };
        *request_index = 44;
        assert!(RaceHistory::new(&altered_selector).is_err());

        let overflow = RaceHistory::new(&workload.actions).expect("overflow history");
        overflow
            .counter
            .store(ACTION_COUNTER_FINAL, Ordering::SeqCst);
        let producer = overflow.slots[0].key.producer;
        assert!(overflow.begin_action(0, producer).is_err());
        assert_eq!(overflow.counter_value(), ACTION_COUNTER_FINAL + 1);

        let maximum = RaceHistory::new(&workload.actions).expect("maximum history");
        maximum.counter.store(u64::MAX, Ordering::SeqCst);
        let producer = maximum.slots[0].key.producer;
        assert!(maximum.begin_action(0, producer).is_err());
        assert_eq!(maximum.counter_value(), 0);

        let mut corrupted = complete_synthetic_history();
        corrupted.slots[7].key.ordinal = 8;
        assert!(corrupted.into_complete().is_err());

        let mut wrong_generation = complete_synthetic_history();
        let first_submit = wrong_generation
            .slots
            .iter_mut()
            .find(|slot| slot.key.submit_attempt == Some(0))
            .expect("first submit");
        first_submit
            .published
            .get_mut()
            .expect("published submit")
            .12
            .3 = 2;
        first_submit
            .complete
            .get_mut()
            .expect("complete submit")
            .12
            .3 = 2;
        assert!(wrong_generation.into_complete().is_err());

        let mut duplicate_counter = complete_synthetic_history();
        let duplicate_response = duplicate_counter.slots[1]
            .complete
            .get()
            .expect("left complete")
            .6;
        duplicate_counter.slots[2]
            .complete
            .get_mut()
            .expect("right complete")
            .6 = duplicate_response;
        assert!(duplicate_counter.into_complete().is_err());
    }

    #[test]
    fn overlap_requires_opposite_producers_strict_intervals_and_boundaries() {
        let workload = authenticated_workload().expect("authenticated workload");
        let left_key = RaceActionKey::authenticate(&workload.actions[1]).expect("left key");
        let right_key = RaceActionKey::authenticate(&workload.actions[2]).expect("right key");
        let same_key = RaceActionKey::authenticate(&workload.actions[3]).expect("same key");
        let left = left_key.capture(1, 4, synthetic_evidence(left_key));
        let right_nonoverlap = right_key.capture(4, 6, synthetic_evidence(right_key));
        assert!(
            select_overlap_pair(&[
                overlap_slot(left_key, left),
                overlap_slot(right_key, right_nonoverlap),
            ])
            .is_err()
        );

        let same = same_key.capture(2, 5, synthetic_evidence(same_key));
        assert!(
            select_overlap_pair(&[overlap_slot(left_key, left), overlap_slot(same_key, same),])
                .is_err()
        );

        let sentinel_key = RaceActionKey::authenticate(&workload.actions[0]).expect("sentinel key");
        let sentinel = sentinel_key.capture(1, 4, synthetic_evidence(sentinel_key));
        let right = right_key.capture(2, 5, synthetic_evidence(right_key));
        assert!(
            select_overlap_pair(&[
                overlap_slot(sentinel_key, sentinel),
                overlap_slot(right_key, right),
            ])
            .is_err()
        );

        let pair =
            select_overlap_pair(&[overlap_slot(left_key, left), overlap_slot(right_key, right)])
                .expect("qualifying overlap");
        assert_eq!(
            pair,
            RaceOverlapPair(1, RaceBoundary::Command, 2, RaceBoundary::Command)
        );
    }

    #[test]
    fn boundary_mask_uses_the_frozen_precedence() {
        let workload = authenticated_workload().expect("authenticated workload");
        let key = RaceActionKey::authenticate(&workload.actions[1]).expect("action key");
        let mut capture = key.capture(1, 2, synthetic_evidence(key));
        capture.13 = ControlWitnessCapture(
            ControlOperationCapture::Cancel,
            true,
            0,
            1,
            Hex64(control_word(1, 0)),
            Hex64(control_word(1, CANCELLED_FLAG)),
            Some(ControlDispositionCapture::Requested),
        );
        capture.14 = PopWitnessCapture(PopKindCapture::Primary, true, 0, 1, 0, 0, None);
        capture.15 = PopWitnessCapture(PopKindCapture::OpportunisticEof, true, 0, 1, 0, 0, None);
        capture.17 = WakeWitnessCapture(true, false, 1, 2);
        assert_eq!(
            BoundaryMask::from_capture(capture).first(),
            Some(RaceBoundary::Command)
        );
        capture.11 = CommandWitnessCapture::sentinel();
        assert_eq!(
            BoundaryMask::from_capture(capture).first(),
            Some(RaceBoundary::Control)
        );
        capture.13 = ControlWitnessCapture::sentinel();
        assert_eq!(
            BoundaryMask::from_capture(capture).first(),
            Some(RaceBoundary::PrimaryEndpoint)
        );
        capture.14 = PopWitnessCapture::sentinel(PopKindCapture::Primary);
        assert_eq!(
            BoundaryMask::from_capture(capture).first(),
            Some(RaceBoundary::OpportunisticEndpoint)
        );
        capture.15 = PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof);
        assert_eq!(
            BoundaryMask::from_capture(capture).first(),
            Some(RaceBoundary::Wake)
        );
        capture.17 = WakeWitnessCapture::sentinel();
        assert_eq!(BoundaryMask::from_capture(capture).first(), None);
    }

    #[test]
    fn action_tuple_and_sentinels_have_exact_canonical_spellings() {
        assert_eq!(
            serde_json::to_string(&Hex64(0)).expect("hex"),
            "\"0000000000000000\""
        );
        assert_eq!(
            serde_json::to_string(&Hex64(1)).expect("hex"),
            "\"0000000000000001\""
        );
        assert_eq!(
            serde_json::to_string(&Hex64(u64::MAX)).expect("hex"),
            "\"ffffffffffffffff\""
        );
        assert_eq!(
            serde_json::to_string(&CommandWitnessCapture::sentinel()).expect("command sentinel"),
            "[false,0,0,0]"
        );
        assert_eq!(
            serde_json::to_string(&AcceptedWitnessCapture::sentinel()).expect("accepted sentinel"),
            "[false,0,0,0,0,0]"
        );
        assert_eq!(
            serde_json::to_string(&ControlWitnessCapture::sentinel()).expect("control sentinel"),
            "[\"none\",false,0,0,\"0000000000000000\",\"0000000000000000\",null]"
        );
        assert_eq!(
            serde_json::to_string(&PopWitnessCapture::sentinel(PopKindCapture::Primary))
                .expect("primary sentinel"),
            "[\"primary\",false,0,0,0,0,null]"
        );
        assert_eq!(
            serde_json::to_string(&PopWitnessCapture::sentinel(
                PopKindCapture::OpportunisticEof,
            ))
            .expect("opportunistic sentinel"),
            "[\"opportunistic_eof\",false,0,0,0,0,null]"
        );
        assert_eq!(
            serde_json::to_string(&WakeWitnessCapture::sentinel()).expect("wake sentinel"),
            "[false,false,0,0]"
        );

        let identity = valid_identity();
        let action = RaceActionCapture(
            7,
            1,
            RaceKind::Submit,
            Some(7),
            Some(7),
            1,
            2,
            RaceResult::SubmitAccepted,
            None,
            Some(1),
            None,
            CommandWitnessCapture(true, 0, 1, 1),
            AcceptedWitnessCapture::from_identity(identity),
            ControlWitnessCapture::sentinel(),
            PopWitnessCapture::sentinel(PopKindCapture::Primary),
            PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
            false,
            WakeWitnessCapture::sentinel(),
        );
        let encoded = serde_json::to_string(&action).expect("action tuple");
        action.validate().expect("valid action tuple");
        let value: serde_json::Value = serde_json::from_str(&encoded).expect("action JSON");
        assert_eq!(value.as_array().expect("action array").len(), 18);
        assert_eq!(
            encoded,
            "[7,1,\"submit\",7,7,1,2,\"submit_accepted\",null,1,null,[true,0,1,1],[true,1,0,1,0,1],[\"none\",false,0,0,\"0000000000000000\",\"0000000000000000\",null],[\"primary\",false,0,0,0,0,null],[\"opportunistic_eof\",false,0,0,0,0,null],false,[false,false,0,0]]"
        );
        assert!(action.11.boundary_reached());
        assert!(action.12.boundary_reached());
        assert!(!action.13.boundary_reached());
        assert!(!action.14.boundary_reached());
        assert!(!action.15.boundary_reached());
        assert!(!action.17.boundary_reached());

        let mut invalid = action;
        invalid.0 = 1_024;
        assert!(invalid.validate().is_err());
        invalid = action;
        invalid.1 = 2;
        assert!(invalid.validate().is_err());
        invalid = action;
        invalid.5 = 0;
        assert!(invalid.validate().is_err());
        invalid = action;
        invalid.6 = invalid.5;
        assert!(invalid.validate().is_err());
        invalid = action;
        invalid.8 = Some(RaceError::request_not_found());
        assert!(invalid.validate().is_err());
        invalid = action;
        invalid.10 = Some(RaceOutput(1, 0, u64::from(u32::MAX)));
        assert!(invalid.validate().is_err());

        let exhausted = RaceActionCapture(
            10,
            0,
            RaceKind::Submit,
            Some(64),
            None,
            3,
            4,
            RaceResult::SubmitOfferExhausted,
            None,
            None,
            None,
            CommandWitnessCapture::sentinel(),
            AcceptedWitnessCapture::sentinel(),
            ControlWitnessCapture::sentinel(),
            PopWitnessCapture::sentinel(PopKindCapture::Primary),
            PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
            false,
            WakeWitnessCapture::sentinel(),
        );
        exhausted.validate().expect("valid exhausted submit");
        let mut invalid_exhausted = exhausted;
        invalid_exhausted.4 = Some(64);
        assert!(invalid_exhausted.validate().is_err());

        let wake = RaceActionCapture(
            3,
            1,
            RaceKind::Wake,
            None,
            Some(63),
            5,
            6,
            RaceResult::WakeSignaled,
            None,
            None,
            None,
            CommandWitnessCapture::sentinel(),
            AcceptedWitnessCapture::sentinel(),
            ControlWitnessCapture::sentinel(),
            PopWitnessCapture::sentinel(PopKindCapture::Primary),
            PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
            false,
            WakeWitnessCapture(true, false, 9, 10),
        );
        wake.validate().expect("valid wake action");

        assert_eq!(
            serde_json::to_value([
                RaceKind::Submit,
                RaceKind::Cancel,
                RaceKind::ReceiverDrop,
                RaceKind::Drain,
                RaceKind::Wake,
            ])
            .expect("kind spellings"),
            serde_json::json!(["submit", "cancel", "receiver_drop", "drain", "wake"])
        );
        assert_eq!(
            serde_json::to_value([
                RaceResult::SubmitAccepted,
                RaceResult::SubmitOfferExhausted,
                RaceResult::CancelRequested,
                RaceResult::CancelAlreadyRequested,
                RaceResult::CancelAlreadyTerminal,
                RaceResult::ReceiverDropped,
                RaceResult::DrainOutput,
                RaceResult::DrainEmpty,
                RaceResult::DrainEof,
                RaceResult::WakeSignaled,
                RaceResult::TargetUnavailable,
                RaceResult::Error,
            ])
            .expect("result spellings"),
            serde_json::json!([
                "submit_accepted",
                "submit_offer_exhausted",
                "cancel_requested",
                "cancel_already_requested",
                "cancel_already_terminal",
                "receiver_dropped",
                "drain_output",
                "drain_empty",
                "drain_eof",
                "wake_signaled",
                "target_unavailable",
                "error"
            ])
        );
    }

    #[test]
    fn only_exact_protocol_errors_are_capturable() {
        assert_eq!(
            RaceError::normalize(&SchedulerError::request_not_found()).expect("stale"),
            RaceError::request_not_found()
        );
        assert_eq!(
            serde_json::to_string(&RaceError::request_not_found()).expect("stale JSON"),
            "[\"request_not_found\",\"invalid_request\",null,null,null]"
        );
        assert_eq!(
            RaceError::normalize(&SchedulerError::resource_exhausted(
                "request slot count",
                REQUEST_SLOT_COUNT,
                REQUEST_SLOT_COUNT,
            ))
            .expect("saturation"),
            RaceError::request_slots_exhausted()
        );
        assert_eq!(
            serde_json::to_string(&RaceError::request_slots_exhausted()).expect("saturation JSON"),
            "[\"resource_exhausted\",\"resource_exhausted\",\"request slot count\",16,16]"
        );
        assert!(
            RaceError::normalize(&SchedulerError::invalid_request("request", "unknown")).is_err()
        );
        assert!(
            RaceError::normalize(&SchedulerError::resource_exhausted(
                "request slot count",
                REQUEST_SLOT_COUNT - 1,
                REQUEST_SLOT_COUNT,
            ))
            .is_err()
        );
        assert!(
            RaceError::normalize(&SchedulerError::resource_exhausted(
                "another resource",
                REQUEST_SLOT_COUNT,
                REQUEST_SLOT_COUNT,
            ))
            .is_err()
        );
        assert!(
            RaceError::normalize(&SchedulerError::allocation_failure(
                "request slot count",
                REQUEST_SLOT_COUNT,
            ))
            .is_err()
        );
    }

    #[test]
    fn command_and_wake_normalization_reject_partial_boundaries() {
        assert_eq!(
            CommandWitnessCapture::normalize_parts(None, 0, 0, false).expect("sentinel"),
            CommandWitnessCapture::sentinel()
        );
        assert_eq!(
            CommandWitnessCapture::normalize_parts(Some(7), 9, 11, true).expect("reached"),
            CommandWitnessCapture(true, 7, 9, 11)
        );
        assert!(CommandWitnessCapture::normalize_parts(Some(0), 1, 1, false).is_err());
        assert!(CommandWitnessCapture::normalize_parts(None, 1, 0, false).is_err());
        assert!(CommandWitnessCapture::normalize_parts(Some(8), 1, 1, true).is_err());
        assert!(CommandWitnessCapture::normalize_parts(Some(0), 0, 1, true).is_err());
        assert!(CommandWitnessCapture::normalize_parts(Some(0), 1, 0, true).is_err());

        assert_eq!(
            WakeWitnessCapture::normalize(ActorWakeWitness {
                dirty_was_set: true,
                before_park_epoch: 7,
                after_park_epoch: 8,
            })
            .expect("wake"),
            WakeWitnessCapture(true, true, 7, 8)
        );
        assert!(
            WakeWitnessCapture::normalize(ActorWakeWitness {
                dirty_was_set: false,
                before_park_epoch: 8,
                after_park_epoch: 8,
            })
            .is_err()
        );
        assert!(
            WakeWitnessCapture::normalize(ActorWakeWitness {
                dirty_was_set: false,
                before_park_epoch: 8,
                after_park_epoch: 7,
            })
            .is_err()
        );
    }

    #[test]
    fn packed_control_algebra_is_closed_and_generation_bound() {
        let valid = [
            (
                ControlOperationCapture::Cancel,
                ControlDispositionCapture::Requested,
                control_word(1, 0),
                control_word(1, CANCELLED_FLAG),
            ),
            (
                ControlOperationCapture::Cancel,
                ControlDispositionCapture::AlreadyRequested,
                control_word(1, CANCELLED_FLAG),
                control_word(1, CANCELLED_FLAG),
            ),
            (
                ControlOperationCapture::Cancel,
                ControlDispositionCapture::AlreadyTerminal,
                control_word(1, TERMINAL_FLAG),
                control_word(1, TERMINAL_FLAG),
            ),
            (
                ControlOperationCapture::Disconnect,
                ControlDispositionCapture::Requested,
                control_word(1, 0),
                control_word(1, CANCELLED_FLAG | DISCONNECTED_FLAG),
            ),
            (
                ControlOperationCapture::Disconnect,
                ControlDispositionCapture::AlreadyRequested,
                control_word(1, CANCELLED_FLAG | DISCONNECTED_FLAG),
                control_word(1, CANCELLED_FLAG | DISCONNECTED_FLAG),
            ),
            (
                ControlOperationCapture::Disconnect,
                ControlDispositionCapture::AlreadyTerminal,
                control_word(1, TERMINAL_FLAG),
                control_word(1, CANCELLED_FLAG | DISCONNECTED_FLAG | TERMINAL_FLAG),
            ),
            (
                ControlOperationCapture::Disconnect,
                ControlDispositionCapture::AlreadyTerminal,
                control_word(1, CANCELLED_FLAG | DISCONNECTED_FLAG | TERMINAL_FLAG),
                control_word(1, CANCELLED_FLAG | DISCONNECTED_FLAG | TERMINAL_FLAG),
            ),
        ];
        for (operation, disposition, loaded, resulting) in valid {
            let capture = ControlWitnessCapture::normalize_parts(
                operation,
                true,
                15,
                1,
                loaded,
                resulting,
                ControlOutcome::Disposition(disposition),
            )
            .expect("valid control transition");
            assert!(capture.boundary_reached());
        }

        let stale = ControlWitnessCapture::normalize_parts(
            ControlOperationCapture::Cancel,
            true,
            0,
            1,
            control_word(2, TERMINAL_FLAG),
            control_word(2, TERMINAL_FLAG),
            ControlOutcome::Stale,
        )
        .expect("stale transition");
        assert_eq!(stale.6, None);

        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::None,
                true,
                0,
                1,
                control_word(1, 0),
                control_word(1, 0),
                ControlOutcome::Stale,
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                false,
                0,
                1,
                control_word(1, 0),
                control_word(1, 0),
                ControlOutcome::Stale,
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                true,
                REQUEST_SLOT_COUNT,
                1,
                control_word(1, 0),
                control_word(1, 0),
                ControlOutcome::Stale,
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                true,
                0,
                0,
                control_word(1, 0),
                control_word(1, 0),
                ControlOutcome::Stale,
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                true,
                0,
                2,
                control_word(1, 0),
                control_word(1, 0),
                ControlOutcome::Stale,
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                true,
                0,
                1,
                control_word(1, 0),
                control_word(1, 0),
                ControlOutcome::Stale,
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                true,
                0,
                1,
                control_word(2, 0),
                control_word(2, CANCELLED_FLAG),
                ControlOutcome::Stale,
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                true,
                0,
                1,
                control_word(1, DISCONNECTED_FLAG),
                control_word(1, DISCONNECTED_FLAG),
                ControlOutcome::Disposition(ControlDispositionCapture::AlreadyRequested),
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Cancel,
                true,
                0,
                1,
                control_word(1, 0),
                control_word(1, 0),
                ControlOutcome::Disposition(ControlDispositionCapture::Requested),
            )
            .is_err()
        );
        assert!(
            ControlWitnessCapture::normalize_parts(
                ControlOperationCapture::Disconnect,
                true,
                0,
                1,
                control_word(1, 0),
                control_word(2, CANCELLED_FLAG | DISCONNECTED_FLAG),
                ControlOutcome::Disposition(ControlDispositionCapture::Requested),
            )
            .is_err()
        );
    }

    #[test]
    fn whole_action_validation_cannot_bypass_control_or_endpoint_normalization() {
        let cancel_control = ControlWitnessCapture::normalize_parts(
            ControlOperationCapture::Cancel,
            true,
            0,
            1,
            control_word(1, 0),
            control_word(1, CANCELLED_FLAG),
            ControlOutcome::Disposition(ControlDispositionCapture::Requested),
        )
        .expect("cancel transition");
        let cancel = RaceActionCapture(
            1,
            1,
            RaceKind::Cancel,
            None,
            Some(0),
            1,
            2,
            RaceResult::CancelRequested,
            None,
            Some(1),
            None,
            CommandWitnessCapture::sentinel(),
            AcceptedWitnessCapture::sentinel(),
            cancel_control,
            PopWitnessCapture::sentinel(PopKindCapture::Primary),
            PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
            false,
            WakeWitnessCapture::sentinel(),
        );
        cancel.validate().expect("valid cancel action");

        let mut invalid_cancel = cancel;
        invalid_cancel.13.2 = REQUEST_SLOT_COUNT;
        assert!(invalid_cancel.validate().is_err());
        invalid_cancel = cancel;
        invalid_cancel.13.3 = 0;
        assert!(invalid_cancel.validate().is_err());
        invalid_cancel = cancel;
        invalid_cancel.13.4 = Hex64(control_word(1, DISCONNECTED_FLAG));
        invalid_cancel.13.5 = invalid_cancel.13.4;
        invalid_cancel.13.6 = Some(ControlDispositionCapture::AlreadyRequested);
        invalid_cancel.7 = RaceResult::CancelAlreadyRequested;
        assert!(invalid_cancel.validate().is_err());
        invalid_cancel = cancel;
        invalid_cancel.13 = ControlWitnessCapture(
            ControlOperationCapture::Cancel,
            true,
            0,
            1,
            Hex64(control_word(1, 0)),
            Hex64(control_word(1, 0)),
            None,
        );
        invalid_cancel.7 = RaceResult::Error;
        invalid_cancel.8 = Some(RaceError::request_not_found());
        assert!(invalid_cancel.validate().is_err());

        let output = RaceOutput(1, 7, u64::from(u32::MAX));
        let primary = PopWitnessCapture(PopKindCapture::Primary, true, 3, 4, 7, 8, Some(output));
        let drain = RaceActionCapture(
            2,
            0,
            RaceKind::Drain,
            None,
            Some(2),
            3,
            4,
            RaceResult::DrainOutput,
            None,
            Some(1),
            Some(output),
            CommandWitnessCapture::sentinel(),
            AcceptedWitnessCapture::sentinel(),
            ControlWitnessCapture::sentinel(),
            primary,
            PopWitnessCapture::sentinel(PopKindCapture::OpportunisticEof),
            false,
            WakeWitnessCapture::sentinel(),
        );
        drain.validate().expect("valid output drain");

        let mut invalid_drain = drain;
        invalid_drain.14.0 = PopKindCapture::OpportunisticEof;
        assert!(invalid_drain.validate().is_err());
        invalid_drain = drain;
        invalid_drain.14.2 = REQUEST_SLOT_COUNT;
        assert!(invalid_drain.validate().is_err());
        invalid_drain = drain;
        invalid_drain.14.3 = 0;
        assert!(invalid_drain.validate().is_err());
        invalid_drain = drain;
        invalid_drain.14.5 = invalid_drain.14.4;
        assert!(invalid_drain.validate().is_err());
        invalid_drain = drain;
        invalid_drain.10 = Some(RaceOutput(1, 7, u64::from(u32::MAX) + 1));
        invalid_drain.14.6 = invalid_drain.10;
        assert!(invalid_drain.validate().is_err());
        invalid_drain = drain;
        invalid_drain.15 =
            PopWitnessCapture(PopKindCapture::OpportunisticEof, true, 4, 4, 8, 8, None);
        assert!(invalid_drain.validate().is_err());
        invalid_drain = drain;
        invalid_drain.15 =
            PopWitnessCapture(PopKindCapture::OpportunisticEof, true, 3, 4, 7, 8, None);
        assert!(invalid_drain.validate().is_err());
    }

    #[test]
    fn accepted_identity_validation_is_closed_and_bounded() {
        let valid = valid_identity();
        assert_eq!(
            AcceptedIdentity::normalize(1, valid, REQUEST_SLOT_COUNT).expect("valid identity"),
            valid
        );
        assert!(AcceptedIdentity::normalize(0, valid, REQUEST_SLOT_COUNT).is_err());
        assert!(AcceptedIdentity::normalize(2, valid, REQUEST_SLOT_COUNT).is_err());
        assert!(AcceptedIdentity::normalize(1, valid, 0).is_err());

        let mut invalid = valid;
        invalid.control_slot = REQUEST_SLOT_COUNT;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid = valid;
        invalid.control_generation = 0;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid.control_generation = MAX_CONTROL_GENERATION + 1;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid = valid;
        invalid.endpoint_slot = REQUEST_SLOT_COUNT;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
        invalid = valid;
        invalid.endpoint_generation = 0;
        assert!(AcceptedIdentity::normalize(1, invalid, REQUEST_SLOT_COUNT).is_err());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn registry_publishes_one_complete_entry_and_consumes_receiver_once() {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let workload = authenticated_workload().expect("authenticated workload");
            let fresh = spawn_fresh_actor(&workload.actor_config)
                .await
                .expect("fresh actor");
            let actor = fresh.actor;
            let probe = fresh.probe;
            let client = actor.client();
            let request = request_spec(&workload.descriptors[0]).expect("request spec");
            let (submission, command) = client.try_submit_with_witness(request);
            assert!(
                CommandWitnessCapture::normalize(command, true)
                    .expect("command witness")
                    .boundary_reached()
            );
            let handle = submission
                .expect("submit command")
                .wait()
                .await
                .expect("engine admission");

            let registry = RaceRegistry::new();
            assert!(registry.lookup_cancel(0).expect("lookup").is_none());
            assert!(registry.lookup_receiver(0, 0).expect("lookup").is_none());
            assert!(
                registry
                    .mark_receiver_consumed(0, 0, valid_identity())
                    .is_err()
            );
            assert!(registry.publish_stored(0, 1, &handle).is_err());

            let mut handles = std::iter::repeat_with(|| None)
                .take(REQUEST_COUNT)
                .collect::<Vec<Option<RequestHandle>>>();
            handles[0] = Some(handle);
            let stored = handles[0].as_ref().expect("stored receiver");
            let identity = registry
                .publish_stored(0, 0, stored)
                .expect("complete publication");
            assert_eq!(identity.request_id, 1);
            assert!(identity.control_generation <= MAX_CONTROL_GENERATION);
            assert_eq!(
                AcceptedWitnessCapture::from_identity(identity).1,
                identity.request_id
            );
            assert!(registry.publish_stored(0, 0, stored).is_err());
            assert!(registry.publish_stored(2, 0, stored).is_err());

            let before = registry.snapshot().expect("snapshot");
            before
                .require_all_authorities_present()
                .expect("complete authorities");
            assert_eq!(before.accepted_count(), 1);
            assert_eq!(before.live_receiver_count(), 1);
            assert_eq!(before.accepted_ids()[0], 1);
            assert_eq!(
                before.get(0).expect("metadata").receiver,
                ReceiverOwnership::Owned { producer: 0 }
            );

            let receiver = registry
                .lookup_receiver(0, 0)
                .expect("receiver lookup")
                .expect("receiver target");
            assert_eq!(receiver.identity, identity);
            assert_eq!(receiver.ownership, ReceiverOwnership::Owned { producer: 0 });
            assert!(registry.lookup_receiver(0, 1).is_err());
            let cancel = registry
                .lookup_cancel(0)
                .expect("cancel lookup")
                .expect("cancel target");
            assert_eq!(cancel.identity, identity);
            let (cancel_result, cancel_witness) = cancel.authority.cancel_with_stress_witness();
            let (cancel_result, cancel_error, cancel_capture) =
                ControlWitnessCapture::normalize_cancel(cancel_result, cancel_witness, identity)
                    .expect("cancel witness");
            assert!(matches!(
                cancel_result,
                RaceResult::CancelRequested
                    | RaceResult::CancelAlreadyRequested
                    | RaceResult::CancelAlreadyTerminal
            ));
            assert!(cancel_error.is_none());
            assert!(cancel_capture.boundary_reached());

            let handle = handles[0].take().expect("owned receiver");
            let sink = ActorRequestDropWitnessSink::new();
            let (disconnect_result, disconnect_witness) =
                drop_receiver_with_preallocated_witness(handle, sink, DropWitnessContext::Script)
                    .expect("receiver drop witness");
            let (drop_result, drop_error, disconnect_capture) =
                ControlWitnessCapture::normalize_disconnect(
                    disconnect_result,
                    disconnect_witness.expect("disconnect boundary"),
                    identity,
                )
                .expect("disconnect normalization");
            assert_eq!(drop_result, RaceResult::ReceiverDropped);
            assert!(drop_error.is_none());
            assert!(disconnect_capture.boundary_reached());
            let mut wrong = identity;
            wrong.endpoint_generation += 1;
            assert!(registry.mark_receiver_consumed(0, 0, wrong).is_err());
            registry
                .mark_receiver_consumed(0, 0, identity)
                .expect("receiver consumption");
            assert!(registry.mark_receiver_consumed(0, 0, identity).is_err());

            let consumed = registry.snapshot().expect("consumed snapshot");
            assert_eq!(consumed.accepted_count(), 1);
            assert_eq!(consumed.live_receiver_count(), 0);
            assert_eq!(consumed.accepted_ids(), before.accepted_ids());
            assert_eq!(
                consumed.get(0).expect("metadata").receiver,
                ReceiverOwnership::Consumed
            );

            let authorities = registry.take_authorities().expect("authority take");
            assert_eq!(authorities.len(), 1);
            assert_eq!(authorities[0].client_index, 0);
            assert_eq!(authorities[0].identity, identity);
            registry
                .snapshot()
                .expect("post-take snapshot")
                .require_all_authorities_absent()
                .expect("authorities absent");
            assert!(registry.take_authorities().is_err());
            drop(cancel);
            assert!(authorities[0].authority.cancel().is_ok());
            drop(authorities);

            drop(handles);
            drop(client);
            let pre_shutdown = probe
                .wait_quiescent()
                .await
                .expect("pre-shutdown quiescence");
            assert_eq!(pre_shutdown.outstanding_requests, 0);
            assert_eq!(pre_shutdown.request_bytes, 0);
            let report = actor.shutdown().await.expect("shutdown");
            let post_shutdown = probe.snapshot().expect("post-shutdown snapshot");
            validate_shutdown(
                report,
                1,
                0,
                post_shutdown.request_bytes,
                post_shutdown.shared_bytes,
            )
            .expect("zero-effect shutdown");
        })
        .await
        .expect("registry lifecycle test timed out");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn live_receive_and_wake_witnesses_normalize_without_invented_boundaries() {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let workload = authenticated_workload().expect("authenticated workload");
            let fresh = spawn_fresh_actor(&workload.actor_config)
                .await
                .expect("fresh actor");
            let actor = fresh.actor;
            let probe = fresh.probe;
            let client = actor.client();
            let request = request_spec(&workload.descriptors[1]).expect("request spec");
            let (submission, command) = client.try_submit_with_witness(request);
            CommandWitnessCapture::normalize(command, true).expect("command witness");
            let mut handle = submission
                .expect("submit command")
                .wait()
                .await
                .expect("engine admission");
            let identity =
                AcceptedIdentity::from_stored_handle(&handle).expect("accepted identity");

            let (result, witness) = receive_once_with_witness(&mut handle);
            let receive = NormalizedReceive::normalize(result, witness, identity)
                .expect("complete receive normalization");
            assert!(matches!(
                receive.result,
                RaceResult::DrainOutput | RaceResult::DrainEmpty | RaceResult::DrainEof
            ));
            assert_eq!(
                receive.output.is_some(),
                receive.result == RaceResult::DrainOutput
            );
            assert!(
                receive.primary.boundary_reached()
                    || (receive.result == RaceResult::DrainEof && receive.cached_eof)
            );
            if receive.opportunistic_eof.boundary_reached() {
                assert_eq!(receive.result, RaceResult::DrainOutput);
            }

            handle.cancel().expect("cleanup cancellation");
            let cleanup = finish_receiver(1, handle).await.expect("receiver cleanup");
            assert_eq!(cleanup.terminal.request_id().get(), identity.request_id);
            let wake = probe.wake_and_wait().await.expect("wake acknowledgement");
            assert!(
                WakeWitnessCapture::normalize(wake)
                    .expect("wake normalization")
                    .boundary_reached()
            );
            drop(client);
            let pre_shutdown = probe
                .wait_quiescent()
                .await
                .expect("pre-shutdown quiescence");
            assert_eq!(pre_shutdown.outstanding_requests, 0);
            let report = actor.shutdown().await.expect("shutdown");
            let post_shutdown = probe.snapshot().expect("post-shutdown snapshot");
            validate_shutdown(
                report,
                1,
                0,
                post_shutdown.request_bytes,
                post_shutdown.shared_bytes,
            )
            .expect("zero-effect shutdown");
        })
        .await
        .expect("live witness normalization test timed out");
    }

    #[test]
    fn registry_bounds_fail_before_mutation() {
        let registry = RaceRegistry::new();
        assert!(registry.lookup_cancel(REQUEST_COUNT).is_err());
        assert!(registry.lookup_receiver(REQUEST_COUNT, 0).is_err());
        assert!(registry.lookup_receiver(0, 1).is_err());
        assert!(
            registry
                .mark_receiver_consumed(REQUEST_COUNT, 0, valid_identity())
                .is_err()
        );
        assert!(
            registry
                .mark_receiver_consumed(0, 2, valid_identity())
                .is_err()
        );
        let snapshot = registry.snapshot().expect("snapshot");
        assert_eq!(snapshot.accepted_count(), 0);
        assert_eq!(snapshot.live_receiver_count(), 0);
        assert!(registry.take_authorities().expect("empty take").is_empty());
    }
}
