use std::sync::{Mutex, MutexGuard};

use runnel_scheduler::{
    ActorControlCasWitness, ActorDisconnectDisposition, ActorReceiveWitness, ActorTryPopKind,
    ActorTryPopWitness, ActorWakeWitness, CancelDisposition, OutputEvent, RequestCancellation,
    RequestHandle, SchedulerError, SchedulerResult, SubmitCommandWitness, TryRecvOutput,
};
use serde::{Serialize, Serializer};

use super::common::{HarnessResult, REQUEST_COUNT, require, to_u64};

const PRODUCER_COUNT: usize = 2;
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
