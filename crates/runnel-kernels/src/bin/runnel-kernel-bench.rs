//! Closed, machine-readable evidence child for the preregistered M4 kernel grid.

#![deny(unsafe_code, unsafe_op_in_unsafe_fn)]

use std::fmt;
use std::hint::black_box;
use std::io::{self, Read as _, Write as _};
use std::sync::{Arc, Barrier};

use runnel_kernels::{
    BackendRequest, Bf16, Bf16Matrix, Capabilities, FiniteInput, GemvWorkspace, PreparedGemv,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const CASE_SCHEMA: &str = "runnel.m4-case/1";
const CORRECTNESS_SCHEMA: &str = "runnel.m4-correctness/1";
const REQUEST_SCHEMA: &str = "runnel.m4-cell-request/1";
const WARMUP_SCHEMA: &str = "runnel.m4-warmup/1";
const OBSERVATION_SCHEMA: &str = "runnel.m4-observation/1";
const LIVE_MEMORY_CAP_BYTES: u64 = 256 * 1024 * 1024;
const MEMORY_ACCOUNTING_RESERVE_BYTES: u64 = 1024 * 1024;
const MAX_STDIN_BYTES: usize = 64 * 1024;
const MAX_STDOUT_BYTES: usize = 1024 * 1024;
const PAGE_BYTES: usize = 4096;
const WARMUP_ORDERS: [PairOrder; 5] = [
    PairOrder::BaselineCandidate,
    PairOrder::CandidateBaseline,
    PairOrder::BaselineCandidate,
    PairOrder::CandidateBaseline,
    PairOrder::BaselineCandidate,
];

const WEIGHT_DOMAIN: &[u8] = b"runnel-m4-weight-v1\0";
const INPUT_DOMAIN: &[u8] = b"runnel-m4-input-v1\0";
const EXPECTED_CASE_DIGESTS: [(&str, &str); 5] = [
    (
        "sha256:f71a97cd58a4d41d072a30643f3b93d82351969a338e6acbf67310380af76573",
        "sha256:40cad436319e56ce852d9334c8c54a93fd6d04eaf530708722224c89829f6f5c",
    ),
    (
        "sha256:44a7d48fb4016ac0724b9b28aefe1898fbe5e1b62359cb546e17c3d53a68eb6c",
        "sha256:7c61014e62916ca3848dbad1904656a21f2745f88ecc38fe138f1b64b46c4179",
    ),
    (
        "sha256:da7bf0d37157514fd134e374cb55b3c281d38f97edc2372124d11e76c40515f5",
        "sha256:b04e28a0551f0c5fca3eb0f4be9fc5498bd6542a9bb0752fab263ff4ee168dbc",
    ),
    (
        "sha256:6d17b58eaa4386cf23d8c240e120bf8e22cb0cf566960854da4f86d4616ca2e8",
        "sha256:e7e31b6745eb435841112d80958a2a388f47fbfbdb9936b6fb574f1aabdba375",
    ),
    (
        "sha256:d9daa89e789a5dfff2ba007619f8636cf7ff9f03cee41f44aeaeac76e0dc6002",
        "sha256:2c0108bea6a9e2d6cb80f7a0503b1960ef5549e444f3553d4830e5ff50eb5f62",
    ),
];

#[derive(Clone, Copy, Debug)]
struct CaseSpec {
    id: &'static str,
    rows: usize,
    columns: usize,
    calls_per_worker: usize,
    role: &'static str,
}

const CASES: [CaseSpec; 5] = [
    CaseSpec {
        id: "tail-257x513",
        rows: 257,
        columns: 513,
        calls_per_worker: 1_019,
        role: "descriptive",
    },
    CaseSpec {
        id: "l2-512x512",
        rows: 512,
        columns: 512,
        calls_per_worker: 512,
        role: "descriptive",
    },
    CaseSpec {
        id: "llc-2048x2048",
        rows: 2_048,
        columns: 2_048,
        calls_per_worker: 32,
        role: "descriptive",
    },
    CaseSpec {
        id: "stream-expand-8192x2048",
        rows: 8_192,
        columns: 2_048,
        calls_per_worker: 8,
        role: "primary",
    },
    CaseSpec {
        id: "stream-contract-2048x8192",
        rows: 2_048,
        columns: 8_192,
        calls_per_worker: 8,
        role: "primary",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Comparison {
    Avx2,
    StagedF32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    Natural,
    VectorOffset,
}

impl Layout {
    const fn logical_offset(self) -> usize {
        match self {
            Self::Natural => 0,
            Self::VectorOffset => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CellSpec {
    id: &'static str,
    case_id: &'static str,
    comparison: Comparison,
    layout: Layout,
    workers: usize,
}

const CELLS: [CellSpec; 13] = [
    CellSpec {
        id: "avx-natural-tail",
        case_id: "tail-257x513",
        comparison: Comparison::Avx2,
        layout: Layout::Natural,
        workers: 1,
    },
    CellSpec {
        id: "avx-natural-l2",
        case_id: "l2-512x512",
        comparison: Comparison::Avx2,
        layout: Layout::Natural,
        workers: 1,
    },
    CellSpec {
        id: "avx-natural-llc",
        case_id: "llc-2048x2048",
        comparison: Comparison::Avx2,
        layout: Layout::Natural,
        workers: 1,
    },
    CellSpec {
        id: "avx-natural-stream-expand",
        case_id: "stream-expand-8192x2048",
        comparison: Comparison::Avx2,
        layout: Layout::Natural,
        workers: 1,
    },
    CellSpec {
        id: "avx-natural-stream-contract",
        case_id: "stream-contract-2048x8192",
        comparison: Comparison::Avx2,
        layout: Layout::Natural,
        workers: 1,
    },
    CellSpec {
        id: "avx-offset-tail",
        case_id: "tail-257x513",
        comparison: Comparison::Avx2,
        layout: Layout::VectorOffset,
        workers: 1,
    },
    CellSpec {
        id: "avx-offset-stream-expand",
        case_id: "stream-expand-8192x2048",
        comparison: Comparison::Avx2,
        layout: Layout::VectorOffset,
        workers: 1,
    },
    CellSpec {
        id: "avx-offset-stream-contract",
        case_id: "stream-contract-2048x8192",
        comparison: Comparison::Avx2,
        layout: Layout::VectorOffset,
        workers: 1,
    },
    CellSpec {
        id: "avx-two-worker-stream-expand",
        case_id: "stream-expand-8192x2048",
        comparison: Comparison::Avx2,
        layout: Layout::Natural,
        workers: 2,
    },
    CellSpec {
        id: "avx-two-worker-stream-contract",
        case_id: "stream-contract-2048x8192",
        comparison: Comparison::Avx2,
        layout: Layout::Natural,
        workers: 2,
    },
    CellSpec {
        id: "staged-natural-llc",
        case_id: "llc-2048x2048",
        comparison: Comparison::StagedF32,
        layout: Layout::Natural,
        workers: 1,
    },
    CellSpec {
        id: "staged-natural-stream-expand",
        case_id: "stream-expand-8192x2048",
        comparison: Comparison::StagedF32,
        layout: Layout::Natural,
        workers: 1,
    },
    CellSpec {
        id: "staged-natural-stream-contract",
        case_id: "stream-contract-2048x8192",
        comparison: Comparison::StagedF32,
        layout: Layout::Natural,
        workers: 1,
    },
];

#[derive(Debug)]
struct BenchError(String);

impl BenchError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BenchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BenchError {}

impl From<io::Error> for BenchError {
    fn from(error: io::Error) -> Self {
        Self::new(format!("I/O error: {error}"))
    }
}

type Result<T> = std::result::Result<T, BenchError>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum PairOrder {
    BaselineCandidate,
    CandidateBaseline,
}

impl PairOrder {
    const fn variants(self) -> [Variant; 2] {
        match self {
            Self::BaselineCandidate => [Variant::Baseline, Variant::Candidate],
            Self::CandidateBaseline => [Variant::Candidate, Variant::Baseline],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Variant {
    Baseline,
    Candidate,
}

impl Variant {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Candidate => "candidate",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Implementation {
    ScalarBf16,
    Avx2Bf16,
    StagedF32,
}

impl Implementation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ScalarBf16 => "rust-scalar-bf16-gemv-v1",
            Self::Avx2Bf16 => "c-avx2-bf16-gemv-v1",
            Self::StagedF32 => "rust-staged-f32-gemv-v1",
        }
    }
}

fn implementation(cell: CellSpec, variant: Variant) -> Implementation {
    match (variant, cell.comparison) {
        (Variant::Baseline, _) => Implementation::ScalarBf16,
        (Variant::Candidate, Comparison::Avx2) => Implementation::Avx2Bf16,
        (Variant::Candidate, Comparison::StagedF32) => Implementation::StagedF32,
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CellRequest {
    schema: String,
    cell_id: String,
    case_id: String,
    child_sequence: u64,
    cpus: Vec<usize>,
    warmup_orders: Vec<PairOrder>,
    measured_orders: Vec<PairOrder>,
}

#[derive(Serialize)]
struct CaseRow {
    schema: &'static str,
    case_id: &'static str,
    rows: usize,
    columns: usize,
    calls_per_worker: usize,
    matrix_bytes: u64,
    input_bytes: u64,
    output_bytes: u64,
    element_products_per_worker: u64,
    role: &'static str,
    weight_sha256: String,
    input_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum RowStatus {
    Ok,
    Unsupported,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ObservationStatus {
    Ok,
    Unsupported,
    KernelError,
}

#[derive(Serialize)]
struct CorrectnessRow {
    schema: &'static str,
    check_id: String,
    kind: &'static str,
    case_id: Option<&'static str>,
    backend: &'static str,
    status: RowStatus,
    failure: Option<String>,
    metrics: KernelCorrectnessMetrics,
}

#[derive(Clone, Debug, Serialize)]
struct KernelCorrectnessMetrics {
    components: u64,
    max_abs_error: Option<f64>,
    max_error_to_bound_ratio: Option<f64>,
    worst_index: Option<u64>,
    worst_abs_error: Option<f64>,
    worst_error_bound: Option<f64>,
    cross_scalar_max_abs: Option<f64>,
    cross_scalar_max_rel: Option<f64>,
    cross_scalar_max_ulp: Option<u32>,
}

#[derive(Serialize)]
struct WarmupRow {
    schema: &'static str,
    cell_id: String,
    case_id: String,
    child_sequence: u64,
    pair_sequence: usize,
    order: PairOrder,
    baseline_status: ObservationStatus,
    candidate_status: ObservationStatus,
}

#[derive(Serialize)]
struct ObservationRow {
    schema: &'static str,
    cell_id: String,
    case_id: String,
    child_sequence: u64,
    pair_sequence: usize,
    pair_order: PairOrder,
    variant_sequence: usize,
    variant: &'static str,
    implementation: &'static str,
    status: ObservationStatus,
    failure: Option<String>,
    elapsed_ns: Option<u64>,
    expected_calls: u64,
    executed_calls: u64,
    sink_sha256: Option<String>,
    output_sha256: Option<String>,
    addresses_mod_64: Option<AddressResidues>,
    workers: Vec<WorkerMetadata>,
    resource_usage: Option<ResourceUsage>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct AddressResidues {
    weights: usize,
    input: usize,
    output: usize,
}

#[derive(Clone, Debug, Serialize)]
struct WorkerMetadata {
    worker_index: usize,
    cpu: usize,
    cpu_before: i32,
    cpu_after: i32,
    affinity: Vec<usize>,
    mxcsr_before: u32,
    mxcsr_after: u32,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ResourceUsage {
    user_cpu_ns: u64,
    system_cpu_ns: u64,
    minor_page_faults: u64,
    major_page_faults: u64,
    voluntary_context_switches: u64,
    involuntary_context_switches: u64,
}

#[derive(Clone, Copy, Debug)]
struct ResourceSnapshot {
    user_cpu_ns: u64,
    system_cpu_ns: u64,
    minor_page_faults: u64,
    major_page_faults: u64,
    voluntary_context_switches: u64,
    involuntary_context_switches: u64,
}

#[derive(Debug)]
struct GeneratedCase {
    spec: CaseSpec,
    matrix: Bf16Matrix,
    input: F32Storage,
    weight_sha256: String,
    input_sha256: String,
}

#[derive(Debug)]
struct F32Storage {
    storage: Vec<f32>,
    logical_offset: usize,
}

impl F32Storage {
    fn from_values(values: Vec<f32>, layout: Layout) -> Self {
        let logical_offset = layout.logical_offset();
        if logical_offset == 0 {
            return Self {
                storage: values,
                logical_offset,
            };
        }
        let mut storage = Vec::with_capacity(values.len() + logical_offset);
        storage.resize(logical_offset, 0.0);
        storage.extend(values);
        Self {
            storage,
            logical_offset,
        }
    }

    fn zeros(length: usize, layout: Layout) -> Self {
        let logical_offset = layout.logical_offset();
        Self {
            storage: vec![0.0; length + logical_offset],
            logical_offset,
        }
    }

    fn as_slice(&self) -> &[f32] {
        &self.storage[self.logical_offset..]
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.storage[self.logical_offset..]
    }

    fn pointer_mod_64(&self) -> usize {
        self.as_slice().as_ptr() as usize % 64
    }
}

#[derive(Debug)]
struct WorkerBuffers {
    output: F32Storage,
    workspace: GemvWorkspace,
    staged_matrix: Vec<f32>,
    staged_temporary: Vec<f32>,
}

impl WorkerBuffers {
    fn new(spec: CaseSpec, cell: CellSpec) -> Result<Self> {
        let staged_elements = if cell.comparison == Comparison::StagedF32 {
            checked_elements(spec)?
        } else {
            0
        };
        let staged_rows = if staged_elements == 0 { 0 } else { spec.rows };
        Ok(Self {
            output: F32Storage::zeros(spec.rows, cell.layout),
            workspace: GemvWorkspace::try_new(spec.rows).map_err(|error| {
                BenchError::new(format!("workspace allocation failed: {error}"))
            })?,
            staged_matrix: vec![0.0; staged_elements],
            staged_temporary: vec![0.0; staged_rows],
        })
    }
}

#[derive(Debug)]
enum PreparedExecution<'case> {
    Bf16(PreparedGemv<'case, 'case>),
    StagedF32,
}

#[derive(Debug)]
struct BatchCore {
    executed_calls: usize,
    sink: u64,
}

#[derive(Debug)]
struct ExecutionAttempt {
    core: BatchCore,
    failures: Vec<String>,
}

impl ExecutionAttempt {
    fn new() -> Self {
        Self {
            core: BatchCore {
                executed_calls: 0,
                sink: 0,
            },
            failures: Vec::new(),
        }
    }

    fn failed(error: impl fmt::Display) -> Self {
        let mut attempt = Self::new();
        attempt.failures.push(error.to_string());
        attempt
    }
}

#[derive(Debug)]
struct BatchSuccess {
    elapsed_ns: u64,
    expected_calls: u64,
    executed_calls: u64,
    sink_sha256: String,
    output_sha256: String,
    addresses: AddressResidues,
    workers: Vec<WorkerMetadata>,
    resource_usage: ResourceUsage,
}

#[derive(Debug)]
struct BatchFailure {
    failure: String,
    elapsed_ns: Option<u64>,
    executed_calls: u64,
    sink_sha256: Option<String>,
    output_sha256: Option<String>,
    addresses: Option<AddressResidues>,
    workers: Vec<WorkerMetadata>,
    resource_usage: Option<ResourceUsage>,
}

impl BatchFailure {
    fn before_start(error: impl fmt::Display) -> Box<Self> {
        Box::new(Self {
            failure: bounded_failure([error.to_string()]),
            elapsed_ns: None,
            executed_calls: 0,
            sink_sha256: None,
            output_sha256: None,
            addresses: None,
            workers: Vec::new(),
            resource_usage: None,
        })
    }
}

type BatchRunResult = std::result::Result<BatchSuccess, Box<BatchFailure>>;

#[derive(Debug)]
enum BatchOutcome {
    Success(BatchSuccess),
    Unsupported(String),
    Failed(BatchFailure),
}

impl BatchOutcome {
    fn status(&self) -> ObservationStatus {
        match self {
            Self::Success(_) => ObservationStatus::Ok,
            Self::Unsupported(_) => ObservationStatus::Unsupported,
            Self::Failed(_) => ObservationStatus::KernelError,
        }
    }
}

fn main() {
    if let Err(error) = run_cli(std::env::args().skip(1)) {
        let _ = writeln!(io::stderr().lock(), "runnel-kernel-bench: {error}");
        std::process::exit(2);
    }
}

fn run_cli(arguments: impl IntoIterator<Item = String>) -> Result<()> {
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    match arguments.as_slice() {
        [command] if command == "cases" => command_cases(),
        [command] if command == "correctness" => command_correctness(),
        [command] if command == "run-cell" => command_run_cell(),
        _ => Err(BenchError::new(
            "expected exactly one subcommand: cases, correctness, or run-cell",
        )),
    }
}

fn command_cases() -> Result<()> {
    let mut rows = Vec::with_capacity(CASES.len());
    for specification in CASES {
        enforce_memory_cap(specification, Layout::Natural, 1, Comparison::Avx2)?;
        let generated = generate_case(specification, Layout::Natural)?;
        rows.push(CaseRow {
            schema: CASE_SCHEMA,
            case_id: specification.id,
            rows: specification.rows,
            columns: specification.columns,
            calls_per_worker: specification.calls_per_worker,
            matrix_bytes: bytes_for::<u16>(checked_elements(specification)?)?,
            input_bytes: bytes_for::<f32>(specification.columns)?,
            output_bytes: bytes_for::<f32>(specification.rows)?,
            element_products_per_worker: element_products_per_worker(specification)?,
            role: specification.role,
            weight_sha256: generated.weight_sha256,
            input_sha256: generated.input_sha256,
        });
    }
    emit_json_lines(&rows)
}

fn command_correctness() -> Result<()> {
    let mut rows = Vec::with_capacity(CASES.len() * 2 + 3);
    let mut published_scalar_outputs = Vec::with_capacity(CASES.len());
    for specification in CASES {
        enforce_memory_cap(specification, Layout::Natural, 1, Comparison::StagedF32)?;
        let generated = generate_case(specification, Layout::Natural)?;
        let scalar = correctness_output(&generated, Implementation::ScalarBf16);
        let scalar_output = scalar.as_ref().ok().map(|value| value.0.as_slice());
        published_scalar_outputs.push((specification.id, scalar_output.map(<[f32]>::to_vec)));
        rows.push(correctness_row(
            generated.spec,
            "scalar",
            scalar.as_ref().map(|value| (&value.0, &value.1)),
            scalar_output,
        ));

        if Capabilities::detected().avx2_available() {
            let avx2 = correctness_output(&generated, Implementation::Avx2Bf16);
            rows.push(correctness_row(
                generated.spec,
                "avx2",
                avx2.as_ref().map(|value| (&value.0, &value.1)),
                scalar_output,
            ));
        } else {
            rows.push(CorrectnessRow {
                schema: CORRECTNESS_SCHEMA,
                check_id: format!("{}-avx2", generated.spec.id),
                kind: "kernel",
                case_id: Some(generated.spec.id),
                backend: "avx2",
                status: RowStatus::Unsupported,
                failure: Some("backend_unavailable".to_owned()),
                metrics: empty_correctness_metrics(generated.spec.rows),
            });
        }
    }

    for case_id in [
        "llc-2048x2048",
        "stream-expand-8192x2048",
        "stream-contract-2048x8192",
    ] {
        let specification = find_case(case_id)
            .ok_or_else(|| BenchError::new(format!("missing staged correctness case {case_id}")))?;
        enforce_memory_cap(specification, Layout::Natural, 1, Comparison::StagedF32)?;
        let generated = generate_case(specification, Layout::Natural)?;
        let scalar_output = published_scalar_outputs
            .iter()
            .find(|(published_case, _)| *published_case == case_id)
            .ok_or_else(|| {
                BenchError::new(format!(
                    "missing published scalar correctness result for {case_id}"
                ))
            })?
            .1
            .as_deref();
        let staged = correctness_output(&generated, Implementation::StagedF32);
        rows.push(correctness_row(
            generated.spec,
            "staged",
            staged.as_ref().map(|value| (&value.0, &value.1)),
            scalar_output,
        ));
    }
    emit_json_lines(&rows)
}

fn command_run_cell() -> Result<()> {
    let request = read_canonical_request(io::stdin().lock())?;
    let (cell, case) = validate_request(&request)?;
    let stdout = io::stdout();
    let mut stream = JsonLineStream::new(stdout.lock(), MAX_STDOUT_BYTES);
    run_cell(&request, cell, case, &mut stream)
}

fn find_case(case_id: &str) -> Option<CaseSpec> {
    CASES.iter().copied().find(|case| case.id == case_id)
}

fn find_cell(cell_id: &str) -> Option<CellSpec> {
    CELLS.iter().copied().find(|cell| cell.id == cell_id)
}

fn checked_elements(specification: CaseSpec) -> Result<usize> {
    specification
        .rows
        .checked_mul(specification.columns)
        .ok_or_else(|| BenchError::new("case element count overflows usize"))
}

fn bytes_for<T>(elements: usize) -> Result<u64> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| BenchError::new("byte count overflows usize"))?;
    u64::try_from(bytes).map_err(|_| BenchError::new("byte count does not fit u64"))
}

fn element_products_per_worker(specification: CaseSpec) -> Result<u64> {
    u64::try_from(checked_elements(specification)?)
        .ok()
        .and_then(|elements| {
            u64::try_from(specification.calls_per_worker)
                .ok()
                .and_then(|calls| elements.checked_mul(calls))
        })
        .ok_or_else(|| BenchError::new("element-product count overflows u64"))
}

fn enforce_memory_cap(
    specification: CaseSpec,
    layout: Layout,
    workers: usize,
    comparison: Comparison,
) -> Result<u64> {
    let elements = checked_elements(specification)?;
    let offset = layout.logical_offset();
    let matrix = bytes_for::<u16>(
        elements
            .checked_add(offset)
            .ok_or_else(|| BenchError::new("matrix offset accounting overflows usize"))?,
    )?;
    let input = bytes_for::<f32>(
        specification
            .columns
            .checked_add(offset)
            .ok_or_else(|| BenchError::new("input offset accounting overflows usize"))?,
    )?;
    let output = bytes_for::<f32>(
        specification
            .rows
            .checked_add(offset)
            .ok_or_else(|| BenchError::new("output offset accounting overflows usize"))?,
    )?;
    let workspace = bytes_for::<f32>(specification.rows)?;
    let staged_temporary = if comparison == Comparison::StagedF32 {
        workspace
    } else {
        0
    };
    let staged_matrix = if comparison == Comparison::StagedF32 {
        bytes_for::<f32>(elements)?
    } else {
        0
    };
    let per_worker = output
        .checked_add(workspace)
        .and_then(|value| value.checked_add(staged_temporary))
        .ok_or_else(|| BenchError::new("per-worker memory accounting overflows u64"))?;
    let worker_bytes = per_worker
        .checked_mul(
            u64::try_from(workers).map_err(|_| BenchError::new("worker count does not fit u64"))?,
        )
        .ok_or_else(|| BenchError::new("worker memory accounting overflows u64"))?;
    let total = matrix
        .checked_add(input)
        .and_then(|value| value.checked_add(worker_bytes))
        .and_then(|value| value.checked_add(staged_matrix))
        .and_then(|value| value.checked_add(MEMORY_ACCOUNTING_RESERVE_BYTES))
        .ok_or_else(|| BenchError::new("live-memory accounting overflows u64"))?;
    if total > LIVE_MEMORY_CAP_BYTES {
        return Err(BenchError::new(format!(
            "case needs {total} accounted live bytes, exceeding {LIVE_MEMORY_CAP_BYTES}"
        )));
    }
    Ok(total)
}

fn generate_case(specification: CaseSpec, layout: Layout) -> Result<GeneratedCase> {
    let elements = checked_elements(specification)?;
    let logical_offset = layout.logical_offset();
    let mut words = Vec::with_capacity(elements + logical_offset);
    words.resize(logical_offset, 0x7f80);
    fill_counter_bytes(WEIGHT_DOMAIN, specification.id, elements, |_, byte| {
        let signed = i16::from(byte) - 128;
        let value = f32::from(signed) / 128.0;
        let word = Bf16::try_from_f32_rne(value)
            .expect("closed weight stream is finite and within BF16 range");
        words.push(word.to_bits());
    })?;

    let logical_words = &words[logical_offset..];
    let weight_sha256 = sha256_words(logical_words);
    let matrix = if logical_offset == 0 {
        Bf16Matrix::from_words(specification.rows, specification.columns, words)
    } else {
        Bf16Matrix::from_storage(
            specification.rows,
            specification.columns,
            words,
            logical_offset,
        )
    }
    .map_err(|error| BenchError::new(format!("could not build matrix: {error}")))?;

    let mut input_values = Vec::with_capacity(specification.columns);
    fill_counter_bytes(
        INPUT_DOMAIN,
        specification.id,
        specification.columns,
        |index, byte| {
            let signed = i16::from(byte) - 128;
            let value = if index == 0 {
                1.0
            } else {
                f32::from(signed) / 64.0
            };
            input_values.push(value);
        },
    )?;
    let input_sha256 = sha256_f32(&input_values);
    let input = F32Storage::from_values(input_values, layout);

    if let Some(index) = CASES.iter().position(|case| case.id == specification.id) {
        let (expected_weight, expected_input) = EXPECTED_CASE_DIGESTS[index];
        if weight_sha256 != expected_weight || input_sha256 != expected_input {
            return Err(BenchError::new(format!(
                "canonical digest mismatch for case {}",
                specification.id
            )));
        }
    }

    Ok(GeneratedCase {
        spec: specification,
        matrix,
        input,
        weight_sha256,
        input_sha256,
    })
}

fn fill_counter_bytes(
    domain: &[u8],
    case_id: &str,
    count: usize,
    mut consume: impl FnMut(usize, u8),
) -> Result<()> {
    let case_length = u16::try_from(case_id.len())
        .map_err(|_| BenchError::new("case ID is too long for the counter stream"))?;
    let mut produced = 0_usize;
    let mut counter = 0_u64;
    while produced < count {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        hasher.update(case_length.to_le_bytes());
        hasher.update(case_id.as_bytes());
        hasher.update(counter.to_le_bytes());
        for byte in hasher.finalize() {
            if produced == count {
                break;
            }
            consume(produced, byte);
            produced += 1;
        }
        counter = counter
            .checked_add(1)
            .ok_or_else(|| BenchError::new("counter stream exhausted u64"))?;
    }
    Ok(())
}

fn sha256_words(words: &[u16]) -> String {
    let mut hasher = Sha256::new();
    for &word in words {
        hasher.update(word.to_le_bytes());
    }
    format_digest(hasher.finalize().as_slice())
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &value in values {
        hasher.update(value.to_le_bytes());
    }
    format_digest(hasher.finalize().as_slice())
}

fn sha256_sink(worker_sinks: &[(usize, u64, usize)]) -> String {
    let mut hasher = Sha256::new();
    for &(worker, sink, calls) in worker_sinks {
        hasher.update(u64::try_from(worker).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(sink.to_le_bytes());
        hasher.update(u64::try_from(calls).unwrap_or(u64::MAX).to_le_bytes());
    }
    format_digest(hasher.finalize().as_slice())
}

fn sha256_outputs(outputs: &[&[f32]]) -> String {
    let mut hasher = Sha256::new();
    for (worker, output) in outputs.iter().enumerate() {
        hasher.update(u64::try_from(worker).unwrap_or(u64::MAX).to_le_bytes());
        for &value in *output {
            hasher.update(value.to_le_bytes());
        }
    }
    format_digest(hasher.finalize().as_slice())
}

fn format_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(7 + bytes.len() * 2);
    output.push_str("sha256:");
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn correctness_output(
    generated: &GeneratedCase,
    implementation: Implementation,
) -> Result<(Vec<f32>, NumericalDiagnostic)> {
    let input = FiniteInput::new(generated.input.as_slice())
        .map_err(|error| BenchError::new(format!("input validation failed: {error}")))?;
    let mut output = vec![f32::NAN; generated.spec.rows];
    match implementation {
        Implementation::ScalarBf16 | Implementation::Avx2Bf16 => {
            let request = match implementation {
                Implementation::ScalarBf16 => BackendRequest::Scalar,
                Implementation::Avx2Bf16 => BackendRequest::Avx2,
                Implementation::StagedF32 => unreachable!(),
            };
            let prepared = PreparedGemv::new(&generated.matrix, input, request)
                .map_err(|error| BenchError::new(format!("dispatch failed: {error}")))?;
            let mut workspace = GemvWorkspace::try_new(generated.spec.rows).map_err(|error| {
                BenchError::new(format!("workspace allocation failed: {error}"))
            })?;
            prepared
                .run(&mut workspace, &mut output)
                .map_err(|error| BenchError::new(format!("kernel failed: {error}")))?;
        }
        Implementation::StagedF32 => {
            let mut staged = vec![0.0; checked_elements(generated.spec)?];
            stage_matrix(generated.matrix.words(), &mut staged)?;
            let mut temporary = vec![0.0; generated.spec.rows];
            f32_gemv(
                &staged,
                generated.input.as_slice(),
                generated.spec.rows,
                generated.spec.columns,
                &mut temporary,
            )?;
            output.copy_from_slice(&temporary);
        }
    }
    let diagnostic = numerical_diagnostic(generated, &output)?;
    Ok((output, diagnostic))
}

#[derive(Clone, Copy, Debug)]
struct NumericalDiagnostic {
    passed: bool,
    components: usize,
    max_abs_error: f64,
    max_error_to_bound_ratio: f64,
    worst_index: usize,
    worst_abs_error: f64,
    worst_error_bound: f64,
}

fn numerical_diagnostic(generated: &GeneratedCase, output: &[f32]) -> Result<NumericalDiagnostic> {
    if output.len() != generated.spec.rows {
        return Err(BenchError::new("correctness output length is wrong"));
    }
    let unit_roundoff = 2.0_f64.powi(-24);
    let operations = 2.0
        * f64::from(
            u32::try_from(generated.spec.columns)
                .map_err(|_| BenchError::new("column count does not fit u32"))?,
        );
    let gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff);
    let mut passed = true;
    let mut max_abs_error = -1.0_f64;
    let mut max_error_to_bound_ratio = -1.0_f64;
    let mut worst_index = 0_usize;
    let mut worst_abs_error = 0.0_f64;
    let mut worst_error_bound = 0.0_f64;

    for (row, &actual) in output.iter().enumerate() {
        if !actual.is_finite() {
            return Err(BenchError::new(format!(
                "correctness output {row} is not finite"
            )));
        }
        let start = row * generated.spec.columns;
        let words = &generated.matrix.words()[start..start + generated.spec.columns];
        let mut reference = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        for (&word, &input) in words.iter().zip(generated.input.as_slice()) {
            let product = f64::from(Bf16::from_bits(word).to_f32()) * f64::from(input);
            reference += product;
            sum_abs += product.abs();
        }
        if !reference.is_finite() || !sum_abs.is_finite() {
            return Err(BenchError::new(format!(
                "f64 reference accumulation {row} is not finite"
            )));
        }
        let error = (f64::from(actual) - reference).abs();
        let bound = gamma * sum_abs + 1.0e-7;
        if !error.is_finite() || !bound.is_finite() || bound <= 0.0 {
            return Err(BenchError::new(format!(
                "correctness diagnostic {row} is not finite and positive"
            )));
        }
        if error > bound {
            passed = false;
        }
        max_abs_error = max_abs_error.max(error);
        let ratio = error / bound;
        if ratio > max_error_to_bound_ratio {
            max_error_to_bound_ratio = ratio;
            worst_index = row;
            worst_abs_error = error;
            worst_error_bound = bound;
        }
    }

    Ok(NumericalDiagnostic {
        passed,
        components: output.len(),
        max_abs_error,
        max_error_to_bound_ratio,
        worst_index,
        worst_abs_error,
        worst_error_bound,
    })
}

fn correctness_row(
    specification: CaseSpec,
    backend: &'static str,
    result: std::result::Result<(&Vec<f32>, &NumericalDiagnostic), &BenchError>,
    scalar_output: Option<&[f32]>,
) -> CorrectnessRow {
    let check_id = format!("{}-{backend}", specification.id);
    match result {
        Ok((output, diagnostic)) => {
            let cross_scalar = scalar_output.map(|scalar| cross_diagnostic(scalar, output));
            let (status, failure) = if diagnostic.passed {
                (RowStatus::Ok, None)
            } else {
                (
                    RowStatus::Failed,
                    Some("numerical_bound_exceeded".to_owned()),
                )
            };
            CorrectnessRow {
                schema: CORRECTNESS_SCHEMA,
                check_id,
                kind: "kernel",
                case_id: Some(specification.id),
                backend,
                status,
                failure,
                metrics: KernelCorrectnessMetrics {
                    components: u64::try_from(diagnostic.components).unwrap_or(u64::MAX),
                    max_abs_error: Some(diagnostic.max_abs_error),
                    max_error_to_bound_ratio: Some(diagnostic.max_error_to_bound_ratio),
                    worst_index: Some(u64::try_from(diagnostic.worst_index).unwrap_or(u64::MAX)),
                    worst_abs_error: Some(diagnostic.worst_abs_error),
                    worst_error_bound: Some(diagnostic.worst_error_bound),
                    cross_scalar_max_abs: cross_scalar.map(|value| value.0),
                    cross_scalar_max_rel: cross_scalar.map(|value| value.1),
                    cross_scalar_max_ulp: cross_scalar.map(|value| value.2),
                },
            }
        }
        Err(_) => CorrectnessRow {
            schema: CORRECTNESS_SCHEMA,
            check_id,
            kind: "kernel",
            case_id: Some(specification.id),
            backend,
            status: RowStatus::Failed,
            failure: Some("kernel_execution_error".to_owned()),
            metrics: empty_correctness_metrics(specification.rows),
        },
    }
}

fn empty_correctness_metrics(components: usize) -> KernelCorrectnessMetrics {
    KernelCorrectnessMetrics {
        components: u64::try_from(components).unwrap_or(u64::MAX),
        max_abs_error: None,
        max_error_to_bound_ratio: None,
        worst_index: None,
        worst_abs_error: None,
        worst_error_bound: None,
        cross_scalar_max_abs: None,
        cross_scalar_max_rel: None,
        cross_scalar_max_ulp: None,
    }
}

fn cross_diagnostic(reference: &[f32], candidate: &[f32]) -> (f64, f64, u32) {
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut max_ulp = 0_u32;
    for (&left, &right) in reference.iter().zip(candidate) {
        let absolute = (f64::from(left) - f64::from(right)).abs();
        let scale = f64::from(left).abs().max(f64::from(right).abs());
        let relative = if scale == 0.0 { 0.0 } else { absolute / scale };
        max_abs = max_abs.max(absolute);
        max_rel = max_rel.max(relative);
        max_ulp = max_ulp.max(ulp_distance(left, right));
    }
    (max_abs, max_rel, max_ulp)
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    fn ordered(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}

fn read_canonical_request(mut reader: impl io::Read) -> Result<CellRequest> {
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take(u64::try_from(MAX_STDIN_BYTES + 1).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_STDIN_BYTES {
        return Err(BenchError::new(format!(
            "request exceeds {MAX_STDIN_BYTES} bytes"
        )));
    }
    if !bytes.ends_with(b"\n") || bytes.ends_with(b"\n\n") {
        return Err(BenchError::new(
            "request must end with exactly one ASCII LF",
        ));
    }
    let request: CellRequest = serde_json::from_slice(&bytes)
        .map_err(|error| BenchError::new(format!("invalid request JSON: {error}")))?;
    let canonical = canonical_json_line(&request)?;
    if canonical != bytes {
        return Err(BenchError::new(
            "request is not canonical compact key-sorted JSON plus one LF",
        ));
    }
    Ok(request)
}

fn validate_request(request: &CellRequest) -> Result<(CellSpec, CaseSpec)> {
    if request.schema != REQUEST_SCHEMA {
        return Err(BenchError::new(format!(
            "unsupported request schema {:?}",
            request.schema
        )));
    }
    let cell = find_cell(&request.cell_id)
        .ok_or_else(|| BenchError::new(format!("unknown cell ID {:?}", request.cell_id)))?;
    let case = find_case(&request.case_id)
        .ok_or_else(|| BenchError::new(format!("unknown case ID {:?}", request.case_id)))?;
    if cell.case_id != case.id {
        return Err(BenchError::new(format!(
            "cell {} requires case {}, not {}",
            cell.id, cell.case_id, case.id
        )));
    }
    if request.child_sequence >= u64::try_from(CELLS.len()).unwrap_or(u64::MAX) {
        return Err(BenchError::new("child_sequence is outside 0..13"));
    }
    if request.cpus.len() != cell.workers {
        return Err(BenchError::new(format!(
            "cell {} requires {} CPUs, got {}",
            cell.id,
            cell.workers,
            request.cpus.len()
        )));
    }
    if request.cpus.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(BenchError::new(
            "cpus must be unique and strictly increasing",
        ));
    }
    if request.warmup_orders != WARMUP_ORDERS {
        return Err(BenchError::new(
            "warmup_orders do not match the frozen five-pair sequence",
        ));
    }
    if request.measured_orders.len() != 30 {
        return Err(BenchError::new(
            "measured_orders must contain exactly 30 pairs",
        ));
    }
    let baseline_first = request
        .measured_orders
        .iter()
        .filter(|&&order| order == PairOrder::BaselineCandidate)
        .count();
    if baseline_first != 15 {
        return Err(BenchError::new(
            "measured_orders must contain 15 of each pair order",
        ));
    }
    enforce_memory_cap(case, cell.layout, cell.workers, cell.comparison)?;
    Ok((cell, case))
}

fn run_cell<W: io::Write>(
    request: &CellRequest,
    cell: CellSpec,
    case: CaseSpec,
    stream: &mut JsonLineStream<W>,
) -> Result<()> {
    let inherited_affinity = current_affinity()?;
    if inherited_affinity != request.cpus {
        return Err(BenchError::new(format!(
            "inherited affinity {inherited_affinity:?} does not equal requested CPUs {:?}",
            request.cpus
        )));
    }
    if cell.workers == 1 {
        bind_current_thread(request.cpus[0])?;
    } else {
        validate_distinct_physical_cores(&request.cpus)?;
    }

    let generated = generate_case(case, cell.layout)?;
    let mut buffers = (0..cell.workers)
        .map(|_| WorkerBuffers::new(case, cell))
        .collect::<Result<Vec<_>>>()?;
    first_touch(&generated, &mut buffers);

    for (pair_sequence, &order) in WARMUP_ORDERS.iter().enumerate() {
        let mut baseline_status = None;
        let mut candidate_status = None;
        for variant in order.variants() {
            let outcome = run_variant(&generated, &mut buffers, cell, variant, &request.cpus);
            match variant {
                Variant::Baseline => baseline_status = Some(outcome.status()),
                Variant::Candidate => candidate_status = Some(outcome.status()),
            }
        }
        stream.emit(&WarmupRow {
            schema: WARMUP_SCHEMA,
            cell_id: request.cell_id.clone(),
            case_id: request.case_id.clone(),
            child_sequence: request.child_sequence,
            pair_sequence,
            order,
            baseline_status: baseline_status
                .ok_or_else(|| BenchError::new("warmup omitted baseline"))?,
            candidate_status: candidate_status
                .ok_or_else(|| BenchError::new("warmup omitted candidate"))?,
        })?;
    }

    let mut observation_count = 0_usize;
    for (pair_sequence, &pair_order) in request.measured_orders.iter().enumerate() {
        for (variant_sequence, variant) in pair_order.variants().into_iter().enumerate() {
            let implementation = implementation(cell, variant);
            let outcome = run_variant(&generated, &mut buffers, cell, variant, &request.cpus);
            let row = observation_row(
                request,
                cell,
                case,
                pair_sequence,
                pair_order,
                variant_sequence,
                variant,
                implementation,
                outcome,
            )?;
            stream.emit(&row)?;
            observation_count += 1;
        }
    }
    if observation_count != 60 {
        return Err(BenchError::new("cell did not produce exactly 60 rows"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn observation_row(
    request: &CellRequest,
    cell: CellSpec,
    case: CaseSpec,
    pair_sequence: usize,
    pair_order: PairOrder,
    variant_sequence: usize,
    variant: Variant,
    implementation: Implementation,
    outcome: BatchOutcome,
) -> Result<ObservationRow> {
    let expected_calls = u64::try_from(case.calls_per_worker)
        .ok()
        .and_then(|calls| {
            u64::try_from(cell.workers)
                .ok()
                .and_then(|workers| calls.checked_mul(workers))
        })
        .ok_or_else(|| BenchError::new("aggregate expected calls overflow u64"))?;
    let common = |status, failure| ObservationRow {
        schema: OBSERVATION_SCHEMA,
        cell_id: request.cell_id.clone(),
        case_id: request.case_id.clone(),
        child_sequence: request.child_sequence,
        pair_sequence,
        pair_order,
        variant_sequence,
        variant: variant.as_str(),
        implementation: implementation.as_str(),
        status,
        failure,
        elapsed_ns: None,
        expected_calls,
        executed_calls: 0,
        sink_sha256: None,
        output_sha256: None,
        addresses_mod_64: None,
        workers: Vec::new(),
        resource_usage: None,
    };
    Ok(match outcome {
        BatchOutcome::Success(success) => ObservationRow {
            schema: OBSERVATION_SCHEMA,
            cell_id: request.cell_id.clone(),
            case_id: request.case_id.clone(),
            child_sequence: request.child_sequence,
            pair_sequence,
            pair_order,
            variant_sequence,
            variant: variant.as_str(),
            implementation: implementation.as_str(),
            status: ObservationStatus::Ok,
            failure: None,
            elapsed_ns: Some(success.elapsed_ns),
            expected_calls: success.expected_calls,
            executed_calls: success.executed_calls,
            sink_sha256: Some(success.sink_sha256),
            output_sha256: Some(success.output_sha256),
            addresses_mod_64: Some(success.addresses),
            workers: success.workers,
            resource_usage: Some(success.resource_usage),
        },
        BatchOutcome::Unsupported(failure) => common(ObservationStatus::Unsupported, Some(failure)),
        BatchOutcome::Failed(failure) => ObservationRow {
            schema: OBSERVATION_SCHEMA,
            cell_id: request.cell_id.clone(),
            case_id: request.case_id.clone(),
            child_sequence: request.child_sequence,
            pair_sequence,
            pair_order,
            variant_sequence,
            variant: variant.as_str(),
            implementation: implementation.as_str(),
            status: ObservationStatus::KernelError,
            failure: Some(failure.failure),
            elapsed_ns: failure.elapsed_ns,
            expected_calls,
            executed_calls: failure.executed_calls,
            sink_sha256: failure.sink_sha256,
            output_sha256: failure.output_sha256,
            addresses_mod_64: failure.addresses,
            workers: failure.workers,
            resource_usage: failure.resource_usage,
        },
    })
}

fn first_touch(generated: &GeneratedCase, buffers: &mut [WorkerBuffers]) {
    touch_u16(generated.matrix.words());
    touch_f32(generated.input.as_slice());
    for buffer in buffers {
        touch_f32(buffer.output.as_slice());
        touch_f32(&buffer.staged_matrix);
        touch_f32(&buffer.staged_temporary);
    }
}

fn touch_u16(values: &[u16]) {
    let stride = (PAGE_BYTES / std::mem::size_of::<u16>()).max(1);
    for value in values.iter().step_by(stride) {
        black_box(*value);
    }
    if let Some(last) = values.last() {
        black_box(*last);
    }
}

fn touch_f32(values: &[f32]) {
    let stride = (PAGE_BYTES / std::mem::size_of::<f32>()).max(1);
    for value in values.iter().step_by(stride) {
        black_box(*value);
    }
    if let Some(last) = values.last() {
        black_box(*last);
    }
}

fn run_variant(
    generated: &GeneratedCase,
    buffers: &mut [WorkerBuffers],
    cell: CellSpec,
    variant: Variant,
    cpus: &[usize],
) -> BatchOutcome {
    let selected = implementation(cell, variant);
    if selected == Implementation::Avx2Bf16 && !Capabilities::detected().avx2_available() {
        return BatchOutcome::Unsupported("backend_unavailable".to_owned());
    }
    let result = if cell.workers == 1 {
        run_single_worker(generated, &mut buffers[0], selected, cpus[0])
    } else {
        run_two_workers(generated, buffers, selected, cpus)
    };
    match result {
        Ok(success) => BatchOutcome::Success(success),
        Err(failure) => BatchOutcome::Failed(*failure),
    }
}

fn run_single_worker(
    generated: &GeneratedCase,
    buffer: &mut WorkerBuffers,
    implementation: Implementation,
    cpu: usize,
) -> BatchRunResult {
    let mxcsr_entry = read_mxcsr().map_err(BatchFailure::before_start)?;
    validate_mxcsr(mxcsr_entry).map_err(BatchFailure::before_start)?;
    let affinity = current_affinity().map_err(BatchFailure::before_start)?;
    if affinity != [cpu] {
        return Err(BatchFailure::before_start(BenchError::new(format!(
            "single worker affinity {affinity:?} is not [{cpu}]"
        ))));
    }
    let validated_input =
        validated_input(generated, implementation).map_err(BatchFailure::before_start)?;
    let prepared = prepare_execution(generated, implementation, validated_input)
        .map_err(BatchFailure::before_start)?;
    preflight(generated, buffer, &prepared).map_err(BatchFailure::before_start)?;
    let mxcsr_before = read_mxcsr().map_err(BatchFailure::before_start)?;
    validate_mxcsr(mxcsr_before).map_err(BatchFailure::before_start)?;
    validate_preflight_mxcsr_control(mxcsr_entry, mxcsr_before)
        .map_err(BatchFailure::before_start)?;
    let cpu_before = current_cpu().map_err(BatchFailure::before_start)?;
    if cpu_before != i32::try_from(cpu).unwrap_or(-1) {
        return Err(BatchFailure::before_start(BenchError::new(format!(
            "worker started timing on CPU {cpu_before}, expected {cpu}"
        ))));
    }
    let before_usage = resource_snapshot().map_err(BatchFailure::before_start)?;
    let started = monotonic_raw_ns().map_err(BatchFailure::before_start)?;
    let execution = execute_calls(generated, buffer, &prepared);
    let capture = SingleBatchCapture {
        cpu,
        affinity,
        mxcsr_before,
        cpu_before,
        before_usage,
        started,
        execution,
        finished: monotonic_raw_ns(),
        after_usage: resource_snapshot(),
        cpu_after: current_cpu(),
        mxcsr_after: read_mxcsr(),
    };
    finalize_single_batch(generated, buffer, capture)
}

#[derive(Debug)]
struct SingleBatchCapture {
    cpu: usize,
    affinity: Vec<usize>,
    mxcsr_before: u32,
    cpu_before: i32,
    before_usage: ResourceSnapshot,
    started: u64,
    execution: ExecutionAttempt,
    finished: Result<u64>,
    after_usage: Result<ResourceSnapshot>,
    cpu_after: Result<i32>,
    mxcsr_after: Result<u32>,
}

fn finalize_single_batch(
    generated: &GeneratedCase,
    buffer: &WorkerBuffers,
    capture: SingleBatchCapture,
) -> BatchRunResult {
    let SingleBatchCapture {
        cpu,
        affinity,
        mxcsr_before,
        cpu_before,
        before_usage,
        started,
        execution,
        finished,
        after_usage,
        cpu_after,
        mxcsr_after,
    } = capture;
    let ExecutionAttempt { core, mut failures } = execution;
    let finished = retain_result(finished, &mut failures);
    let after_usage = retain_result(after_usage, &mut failures);
    let cpu_after = retain_result(cpu_after, &mut failures);
    let mxcsr_after = retain_result(mxcsr_after, &mut failures);
    let elapsed_ns =
        finished.and_then(|finished| checked_elapsed(started, finished, &mut failures));
    let resource_usage = after_usage
        .and_then(|after| retain_result(resource_delta(before_usage, after), &mut failures));

    if let Err(error) = validate_batch_core(
        generated.spec.calls_per_worker,
        &core,
        buffer.output.as_slice(),
    ) {
        failures.push(error.to_string());
    }
    let expected_calls = usize_to_u64_or_record(
        generated.spec.calls_per_worker,
        "call count does not fit u64",
        &mut failures,
    );
    let executed_calls = usize_to_u64_or_record(
        core.executed_calls,
        "executed calls do not fit u64",
        &mut failures,
    );
    let worker = match (cpu_after, mxcsr_after) {
        (Some(cpu_after), Some(mxcsr_after)) => {
            if cpu_after != cpu_before {
                failures.push(format!(
                    "worker migrated from CPU {cpu_before} to {cpu_after}"
                ));
            }
            if mxcsr_after != mxcsr_before {
                failures.push(format!(
                    "MXCSR changed from 0x{mxcsr_before:08x} to 0x{mxcsr_after:08x}"
                ));
            }
            Some(WorkerMetadata {
                worker_index: 0,
                cpu,
                cpu_before,
                cpu_after,
                affinity,
                mxcsr_before,
                mxcsr_after,
            })
        }
        _ => None,
    };
    let sink_sha256 = sha256_sink(&[(0, core.sink, core.executed_calls)]);
    let output_sha256 = sha256_outputs(&[buffer.output.as_slice()]);
    let addresses = address_residues(generated, buffer);

    if failures.is_empty() {
        return Ok(BatchSuccess {
            elapsed_ns: elapsed_ns.expect("failure recorded for missing elapsed time"),
            expected_calls,
            executed_calls,
            sink_sha256,
            output_sha256,
            addresses,
            workers: vec![worker.expect("failure recorded for missing worker metadata")],
            resource_usage: resource_usage.expect("failure recorded for missing resource usage"),
        });
    }

    Err(Box::new(BatchFailure {
        failure: bounded_failure(failures),
        elapsed_ns,
        executed_calls,
        sink_sha256: Some(sink_sha256),
        output_sha256: Some(output_sha256),
        addresses: Some(addresses),
        workers: worker.into_iter().collect(),
        resource_usage,
    }))
}

fn run_two_workers(
    generated: &GeneratedCase,
    buffers: &mut [WorkerBuffers],
    implementation: Implementation,
    cpus: &[usize],
) -> BatchRunResult {
    if buffers.len() != 2 || cpus.len() != 2 {
        return Err(BatchFailure::before_start(BenchError::new(
            "two-worker cell requires exactly two buffers and CPUs",
        )));
    }
    if implementation == Implementation::StagedF32 {
        return Err(BatchFailure::before_start(BenchError::new(
            "staged-f32 comparison has no two-worker cell",
        )));
    }

    let batch = collect_two_worker_batch(generated, buffers, implementation, cpus);
    finalize_two_worker_batch(generated, buffers, batch)
}

fn collect_two_worker_batch(
    generated: &GeneratedCase,
    buffers: &mut [WorkerBuffers],
    implementation: Implementation,
    cpus: &[usize],
) -> TwoWorkerBatch {
    let ready_barrier = Arc::new(Barrier::new(3));
    let start_barrier = Arc::new(Barrier::new(3));
    let finish_barrier = Arc::new(Barrier::new(3));
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(2);
        for (worker_index, (buffer, &cpu)) in buffers.iter_mut().zip(cpus).enumerate() {
            let ready_barrier = Arc::clone(&ready_barrier);
            let start_barrier = Arc::clone(&start_barrier);
            let finish_barrier = Arc::clone(&finish_barrier);
            handles.push(scope.spawn(move || {
                two_worker_thread(
                    generated,
                    buffer,
                    implementation,
                    worker_index,
                    cpu,
                    &ready_barrier,
                    &start_barrier,
                    &finish_barrier,
                )
            }));
        }

        let measurements =
            measure_two_worker_interval(&ready_barrier, &start_barrier, &finish_barrier);

        let mut thread_results = Vec::with_capacity(2);
        let mut failures = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(result) => thread_results.push(result),
                Err(_) => failures.push("two-worker thread panicked".to_owned()),
            }
        }
        TwoWorkerBatch {
            measurements,
            thread_results,
            failures,
        }
    })
}

fn finalize_two_worker_batch(
    generated: &GeneratedCase,
    buffers: &[WorkerBuffers],
    batch: TwoWorkerBatch,
) -> BatchRunResult {
    let TwoWorkerBatch {
        measurements,
        mut thread_results,
        mut failures,
    } = batch;
    thread_results.sort_by_key(|result| result.worker_index);
    for result in &mut thread_results {
        failures.append(&mut result.failures);
    }
    let before_usage = retain_result(measurements.start.before_usage, &mut failures);
    let started = retain_result(measurements.start.started, &mut failures);
    let finished = retain_result(measurements.finish.finished, &mut failures);
    let after_usage = retain_result(measurements.finish.after_usage, &mut failures);
    let elapsed_ns = match (started, finished) {
        (Some(started), Some(finished)) => checked_elapsed(started, finished, &mut failures),
        _ => None,
    };
    let resource_usage = match (before_usage, after_usage) {
        (Some(before), Some(after)) => retain_result(resource_delta(before, after), &mut failures),
        _ => None,
    };
    let expected_per_worker = generated.spec.calls_per_worker;
    let expected_total = checked_usize_product_or_record(
        expected_per_worker,
        2,
        "aggregate call count overflows usize",
        &mut failures,
    );
    let executed_total = total_executed_calls(&thread_results, &mut failures);
    if executed_total != expected_total {
        failures.push(format!(
            "two workers executed {executed_total} calls, expected {expected_total}"
        ));
    }
    let sinks = thread_results
        .iter()
        .map(|result| {
            (
                result.worker_index,
                result.core.sink,
                result.core.executed_calls,
            )
        })
        .collect::<Vec<_>>();
    let outputs = buffers
        .iter()
        .map(|buffer| buffer.output.as_slice())
        .collect::<Vec<_>>();
    let workers = retain_worker_metadata(thread_results, 2, &mut failures);
    let expected_calls = usize_to_u64_or_record(
        expected_total,
        "expected calls do not fit u64",
        &mut failures,
    );
    let executed_calls = usize_to_u64_or_record(
        executed_total,
        "executed calls do not fit u64",
        &mut failures,
    );
    let sink_sha256 = sha256_sink(&sinks);
    let output_sha256 = sha256_outputs(&outputs);
    let addresses = AddressResidues {
        weights: generated.matrix.words().as_ptr() as usize % 64,
        input: generated.input.pointer_mod_64(),
        output: buffers[0].output.pointer_mod_64(),
    };

    if failures.is_empty() {
        return Ok(BatchSuccess {
            elapsed_ns: elapsed_ns.expect("failure recorded for missing elapsed time"),
            expected_calls,
            executed_calls,
            sink_sha256,
            output_sha256,
            addresses,
            workers,
            resource_usage: resource_usage.expect("failure recorded for missing resource usage"),
        });
    }

    Err(Box::new(BatchFailure {
        failure: bounded_failure(failures),
        elapsed_ns,
        executed_calls,
        sink_sha256: Some(sink_sha256),
        output_sha256: Some(output_sha256),
        addresses: Some(addresses),
        workers,
        resource_usage,
    }))
}

#[derive(Debug)]
struct ThreadBatch {
    worker_index: usize,
    core: BatchCore,
    metadata: Option<WorkerMetadata>,
    failures: Vec<String>,
}

#[derive(Debug)]
struct ThreadSetup<'case> {
    mxcsr_before: u32,
    cpu_before: i32,
    affinity: Vec<usize>,
    prepared: PreparedExecution<'case>,
}

#[derive(Debug)]
struct TwoWorkerBatch {
    measurements: IntervalMeasurements,
    thread_results: Vec<ThreadBatch>,
    failures: Vec<String>,
}

#[derive(Debug)]
struct IntervalStart {
    before_usage: Result<ResourceSnapshot>,
    started: Result<u64>,
}

#[derive(Debug)]
struct IntervalFinish {
    finished: Result<u64>,
    after_usage: Result<ResourceSnapshot>,
}

#[derive(Debug)]
struct IntervalMeasurements {
    start: IntervalStart,
    finish: IntervalFinish,
}

fn coordinate_two_worker_interval<T, U>(
    ready_barrier: &Barrier,
    start_barrier: &Barrier,
    finish_barrier: &Barrier,
    before_start_release: impl FnOnce() -> T,
    after_finish: impl FnOnce() -> U,
) -> (T, U) {
    ready_barrier.wait();
    let start_result = before_start_release();
    start_barrier.wait();
    finish_barrier.wait();
    let finish_result = after_finish();
    (start_result, finish_result)
}

fn measure_two_worker_interval(
    ready_barrier: &Barrier,
    start_barrier: &Barrier,
    finish_barrier: &Barrier,
) -> IntervalMeasurements {
    let (start, finish) = coordinate_two_worker_interval(
        ready_barrier,
        start_barrier,
        finish_barrier,
        || {
            let before_usage = resource_snapshot();
            let started = monotonic_raw_ns();
            IntervalStart {
                before_usage,
                started,
            }
        },
        || {
            let finished = monotonic_raw_ns();
            let after_usage = resource_snapshot();
            IntervalFinish {
                finished,
                after_usage,
            }
        },
    );
    IntervalMeasurements { start, finish }
}

#[allow(clippy::too_many_arguments)]
fn two_worker_thread(
    generated: &GeneratedCase,
    buffer: &mut WorkerBuffers,
    implementation: Implementation,
    worker_index: usize,
    cpu: usize,
    ready_barrier: &Barrier,
    start_barrier: &Barrier,
    finish_barrier: &Barrier,
) -> ThreadBatch {
    let setup = (|| -> Result<ThreadSetup<'_>> {
        bind_current_thread(cpu)?;
        let mxcsr_entry = read_mxcsr()?;
        validate_mxcsr(mxcsr_entry)?;
        let affinity = current_affinity()?;
        if affinity != [cpu] {
            return Err(BenchError::new(format!(
                "worker {worker_index} affinity {affinity:?} is not [{cpu}]"
            )));
        }
        let validated_input = validated_input(generated, implementation)?;
        let prepared = prepare_execution(generated, implementation, validated_input)?;
        preflight(generated, buffer, &prepared)?;
        let mxcsr_before = read_mxcsr()?;
        validate_mxcsr(mxcsr_before)?;
        validate_preflight_mxcsr_control(mxcsr_entry, mxcsr_before)?;
        let cpu_before = current_cpu()?;
        if cpu_before != i32::try_from(cpu).unwrap_or(-1) {
            return Err(BenchError::new(format!(
                "worker {worker_index} started timing on CPU {cpu_before}, expected {cpu}"
            )));
        }
        Ok(ThreadSetup {
            mxcsr_before,
            cpu_before,
            affinity,
            prepared,
        })
    })();

    ready_barrier.wait();
    start_barrier.wait();
    let core = match &setup {
        Ok(setup) => execute_calls(generated, buffer, &setup.prepared),
        Err(error) => ExecutionAttempt::failed(error),
    };
    finish_barrier.wait();
    match setup {
        Ok(setup) => finalize_thread_batch(generated, buffer, worker_index, cpu, setup, core),
        Err(_) => ThreadBatch {
            worker_index,
            core: core.core,
            metadata: None,
            failures: core.failures,
        },
    }
}

fn finalize_thread_batch(
    generated: &GeneratedCase,
    buffer: &WorkerBuffers,
    worker_index: usize,
    cpu: usize,
    setup: ThreadSetup<'_>,
    execution: ExecutionAttempt,
) -> ThreadBatch {
    let ThreadSetup {
        mxcsr_before,
        cpu_before,
        affinity,
        prepared: _,
    } = setup;
    let ExecutionAttempt { core, mut failures } = execution;
    if let Err(error) = validate_batch_core(
        generated.spec.calls_per_worker,
        &core,
        buffer.output.as_slice(),
    ) {
        failures.push(error.to_string());
    }
    let cpu_after = retain_result(current_cpu(), &mut failures);
    let mxcsr_after = retain_result(read_mxcsr(), &mut failures);
    let metadata = match (cpu_after, mxcsr_after) {
        (Some(cpu_after), Some(mxcsr_after)) => {
            if cpu_after != cpu_before {
                failures.push(format!(
                    "worker {worker_index} migrated from CPU {cpu_before} to {cpu_after}"
                ));
            }
            if mxcsr_after != mxcsr_before {
                failures.push(format!(
                    "worker {worker_index} changed MXCSR from 0x{mxcsr_before:08x} to 0x{mxcsr_after:08x}"
                ));
            }
            Some(WorkerMetadata {
                worker_index,
                cpu,
                cpu_before,
                cpu_after,
                affinity,
                mxcsr_before,
                mxcsr_after,
            })
        }
        _ => None,
    };
    ThreadBatch {
        worker_index,
        core,
        metadata,
        failures,
    }
}

fn preflight(
    generated: &GeneratedCase,
    buffer: &mut WorkerBuffers,
    prepared: &PreparedExecution<'_>,
) -> Result<()> {
    match prepared {
        PreparedExecution::Bf16(prepared) => {
            prepared
                .run(&mut buffer.workspace, buffer.output.as_mut_slice())
                .map_err(|error| BenchError::new(format!("kernel preflight failed: {error}")))?;
        }
        PreparedExecution::StagedF32 => {
            stage_matrix(generated.matrix.words(), &mut buffer.staged_matrix)?;
            f32_gemv(
                &buffer.staged_matrix,
                generated.input.as_slice(),
                generated.spec.rows,
                generated.spec.columns,
                &mut buffer.staged_temporary,
            )?;
            validate_finite(&buffer.staged_temporary)?;
            buffer
                .output
                .as_mut_slice()
                .copy_from_slice(&buffer.staged_temporary);
        }
    }
    let diagnostic = numerical_diagnostic(generated, buffer.output.as_slice())?;
    if !diagnostic.passed {
        return Err(BenchError::new(format!(
            "preflight f64 gate failed at component {}: error {}, maximum bound {}",
            diagnostic.worst_index, diagnostic.worst_abs_error, diagnostic.worst_error_bound
        )));
    }
    Ok(())
}

fn validated_input(
    generated: &GeneratedCase,
    implementation: Implementation,
) -> Result<Option<FiniteInput<'_>>> {
    match implementation {
        Implementation::ScalarBf16 | Implementation::Avx2Bf16 => {
            FiniteInput::new(generated.input.as_slice())
                .map(Some)
                .map_err(|error| BenchError::new(format!("input validation failed: {error}")))
        }
        Implementation::StagedF32 => Ok(None),
    }
}

fn prepare_execution<'case>(
    generated: &'case GeneratedCase,
    implementation: Implementation,
    validated_input: Option<FiniteInput<'case>>,
) -> Result<PreparedExecution<'case>> {
    match implementation {
        Implementation::ScalarBf16 | Implementation::Avx2Bf16 => {
            let request = if implementation == Implementation::ScalarBf16 {
                BackendRequest::Scalar
            } else {
                BackendRequest::Avx2
            };
            let input = validated_input
                .ok_or_else(|| BenchError::new("BF16 preparation omitted validated input"))?;
            PreparedGemv::new(&generated.matrix, input, request)
                .map(PreparedExecution::Bf16)
                .map_err(|error| BenchError::new(format!("dispatch failed: {error}")))
        }
        Implementation::StagedF32 => Ok(PreparedExecution::StagedF32),
    }
}

fn execute_calls(
    generated: &GeneratedCase,
    buffer: &mut WorkerBuffers,
    prepared: &PreparedExecution<'_>,
) -> ExecutionAttempt {
    match prepared {
        PreparedExecution::Bf16(prepared) => run_prepared_calls(
            prepared,
            &mut buffer.workspace,
            buffer.output.as_mut_slice(),
            generated.spec.calls_per_worker,
        ),
        PreparedExecution::StagedF32 => run_staged_calls(
            generated.matrix.words(),
            generated.input.as_slice(),
            generated.spec,
            &mut buffer.staged_matrix,
            &mut buffer.staged_temporary,
            buffer.output.as_mut_slice(),
        ),
    }
}

fn run_prepared_calls(
    prepared: &PreparedGemv<'_, '_>,
    workspace: &mut GemvWorkspace,
    output: &mut [f32],
    calls: usize,
) -> ExecutionAttempt {
    let mut attempt = ExecutionAttempt::new();
    for call_index in 0..calls {
        if let Err(error) =
            black_box(prepared).run(black_box(&mut *workspace), black_box(&mut *output))
        {
            attempt
                .failures
                .push(format!("timed kernel call failed: {error}"));
            break;
        }
        attempt.core.sink = fold_sink(attempt.core.sink, output, call_index);
        attempt.core.executed_calls += 1;
    }
    attempt
}

fn run_staged_calls(
    words: &[u16],
    input: &[f32],
    specification: CaseSpec,
    staged_matrix: &mut [f32],
    temporary: &mut [f32],
    output: &mut [f32],
) -> ExecutionAttempt {
    let mut attempt = ExecutionAttempt::new();
    if let Err(error) = stage_matrix(black_box(words), black_box(staged_matrix)) {
        attempt.failures.push(error.to_string());
        return attempt;
    }
    for call_index in 0..specification.calls_per_worker {
        if let Err(error) = f32_gemv(
            black_box(staged_matrix),
            black_box(input),
            specification.rows,
            specification.columns,
            black_box(temporary),
        ) {
            attempt.failures.push(error.to_string());
            break;
        }
        if let Err(error) = validate_finite(temporary) {
            attempt.failures.push(error.to_string());
            break;
        }
        output.copy_from_slice(temporary);
        attempt.core.sink = fold_sink(attempt.core.sink, output, call_index);
        attempt.core.executed_calls += 1;
    }
    attempt
}

fn stage_matrix(words: &[u16], staged: &mut [f32]) -> Result<()> {
    if words.len() != staged.len() {
        return Err(BenchError::new(format!(
            "staging length mismatch: {} words, {} destinations",
            words.len(),
            staged.len()
        )));
    }
    for (&word, destination) in words.iter().zip(staged) {
        *destination = Bf16::from_bits(word).to_f32();
    }
    Ok(())
}

fn f32_gemv(
    matrix: &[f32],
    input: &[f32],
    rows: usize,
    columns: usize,
    output: &mut [f32],
) -> Result<()> {
    if matrix.len() != rows.saturating_mul(columns)
        || input.len() != columns
        || output.len() != rows
    {
        return Err(BenchError::new(
            "staged f32 GEMV dimensions are inconsistent",
        ));
    }
    for (row, destination) in output.iter_mut().enumerate() {
        let start = row * columns;
        let mut sum = 0.0_f32;
        for (&weight, &value) in matrix[start..start + columns].iter().zip(input) {
            sum += weight * value;
        }
        *destination = sum;
    }
    Ok(())
}

fn validate_finite(values: &[f32]) -> Result<()> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(BenchError::new(format!(
                "output {index} is nonfinite 0x{:08x}",
                value.to_bits()
            )));
        }
    }
    Ok(())
}

fn fold_sink(previous: u64, output: &[f32], call_index: usize) -> u64 {
    let selected = output[call_index % output.len()].to_bits();
    let call = u64::try_from(call_index).unwrap_or(u64::MAX);
    previous.rotate_left(11)
        ^ u64::from(black_box(selected))
        ^ call.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn expected_sink(output: &[f32], calls: usize) -> u64 {
    (0..calls).fold(0_u64, |sink, call_index| {
        fold_sink(sink, output, call_index)
    })
}

fn validate_batch_core(expected_calls: usize, core: &BatchCore, output: &[f32]) -> Result<()> {
    if core.executed_calls != expected_calls {
        return Err(BenchError::new(format!(
            "executed {} calls, expected {expected_calls}",
            core.executed_calls
        )));
    }
    let expected = expected_sink(output, expected_calls);
    if core.sink != expected {
        return Err(BenchError::new(format!(
            "call-indexed sink 0x{:016x} does not match 0x{expected:016x}",
            core.sink
        )));
    }
    Ok(())
}

fn retain_result<T>(result: Result<T>, failures: &mut Vec<String>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            failures.push(error.to_string());
            None
        }
    }
}

fn checked_elapsed(started: u64, finished: u64, failures: &mut Vec<String>) -> Option<u64> {
    if let Some(elapsed) = finished.checked_sub(started).filter(|&elapsed| elapsed > 0) {
        Some(elapsed)
    } else {
        failures.push("CLOCK_MONOTONIC_RAW interval is not positive".to_owned());
        None
    }
}

fn usize_to_u64_or_record(value: usize, message: &str, failures: &mut Vec<String>) -> u64 {
    if let Ok(value) = u64::try_from(value) {
        value
    } else {
        failures.push(message.to_owned());
        0
    }
}

fn checked_usize_product_or_record(
    value: usize,
    multiplier: usize,
    message: &str,
    failures: &mut Vec<String>,
) -> usize {
    if let Some(product) = value.checked_mul(multiplier) {
        product
    } else {
        failures.push(message.to_owned());
        0
    }
}

fn total_executed_calls(results: &[ThreadBatch], failures: &mut Vec<String>) -> usize {
    let mut total = 0_usize;
    for result in results {
        if let Some(value) = total.checked_add(result.core.executed_calls) {
            total = value;
        } else {
            failures.push("executed call count overflows usize".to_owned());
            return 0;
        }
    }
    total
}

fn retain_worker_metadata(
    results: Vec<ThreadBatch>,
    expected: usize,
    failures: &mut Vec<String>,
) -> Vec<WorkerMetadata> {
    let mut workers = results
        .into_iter()
        .filter_map(|result| result.metadata)
        .collect::<Vec<_>>();
    workers.sort_by_key(|worker| worker.worker_index);
    if workers.len() != expected {
        failures.push(format!(
            "retained {} worker metadata records, expected {expected}",
            workers.len()
        ));
    }
    workers
}

fn address_residues(generated: &GeneratedCase, buffer: &WorkerBuffers) -> AddressResidues {
    AddressResidues {
        weights: generated.matrix.words().as_ptr() as usize % 64,
        input: generated.input.pointer_mod_64(),
        output: buffer.output.pointer_mod_64(),
    }
}

fn bounded_failure(messages: impl IntoIterator<Item = String>) -> String {
    const MAX_FAILURE_CHARACTERS: usize = 512;
    let joined = messages.into_iter().collect::<Vec<_>>().join("; ");
    let bounded = joined
        .chars()
        .take(MAX_FAILURE_CHARACTERS)
        .collect::<String>();
    if bounded.is_empty() {
        "kernel batch failed without diagnostic detail".to_owned()
    } else {
        bounded
    }
}

fn validate_preflight_mxcsr_control(entry: u32, before_timing: u32) -> Result<()> {
    const STICKY_EXCEPTION_STATUS: u32 = 0x3f;
    let control_mask = !STICKY_EXCEPTION_STATUS;
    if entry & control_mask != before_timing & control_mask {
        return Err(BenchError::new(format!(
            "preflight changed MXCSR control state from 0x{entry:08x} to 0x{before_timing:08x}"
        )));
    }
    Ok(())
}

use platform::{
    bind_current_thread, current_affinity, current_cpu, monotonic_raw_ns, read_mxcsr,
    resource_delta, resource_snapshot, validate_distinct_physical_cores, validate_mxcsr,
};

/// The benchmark-only platform probe island.
///
/// It neither changes floating-point mode nor host policy. Its unsafe calls are
/// limited to reading clocks/resource state, reading or restricting the calling
/// thread's affinity, reading CPU residency, and reading MXCSR.
#[allow(unsafe_code)]
mod platform {
    use super::{BenchError, ResourceSnapshot, ResourceUsage, Result};
    use std::io;

    #[cfg(target_os = "linux")]
    pub(super) fn monotonic_raw_ns() -> Result<u64> {
        let mut value = std::mem::MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: `value` points to writable storage for one `timespec`; the
        // kernel initializes it completely on a successful call.
        let status = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, value.as_mut_ptr()) };
        if status != 0 {
            return Err(BenchError::new(format!(
                "clock_gettime(CLOCK_MONOTONIC_RAW) failed: {}",
                io::Error::last_os_error()
            )));
        }
        // SAFETY: successful `clock_gettime` initialized the value above.
        let value = unsafe { value.assume_init() };
        if value.tv_sec < 0 || value.tv_nsec < 0 {
            return Err(BenchError::new("monotonic clock returned a negative value"));
        }
        let seconds = u64::try_from(value.tv_sec)
            .map_err(|_| BenchError::new("clock seconds do not fit u64"))?;
        let nanoseconds = u64::try_from(value.tv_nsec)
            .map_err(|_| BenchError::new("clock nanoseconds do not fit u64"))?;
        seconds
            .checked_mul(1_000_000_000)
            .and_then(|total| total.checked_add(nanoseconds))
            .ok_or_else(|| BenchError::new("monotonic clock value overflows u64"))
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn monotonic_raw_ns() -> Result<u64> {
        Err(BenchError::new(
            "CLOCK_MONOTONIC_RAW evidence timing requires Linux",
        ))
    }

    #[cfg(target_os = "linux")]
    pub(super) fn resource_snapshot() -> Result<ResourceSnapshot> {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: `usage` is writable storage for one `rusage`; the kernel fills
        // it on success and `RUSAGE_SELF` observes only this benchmark child.
        let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if status != 0 {
            return Err(BenchError::new(format!(
                "getrusage(RUSAGE_SELF) failed: {}",
                io::Error::last_os_error()
            )));
        }
        // SAFETY: successful `getrusage` initialized the value above.
        let usage = unsafe { usage.assume_init() };
        Ok(ResourceSnapshot {
            user_cpu_ns: timeval_ns(usage.ru_utime)?,
            system_cpu_ns: timeval_ns(usage.ru_stime)?,
            minor_page_faults: nonnegative_counter(usage.ru_minflt, "minor page faults")?,
            major_page_faults: nonnegative_counter(usage.ru_majflt, "major page faults")?,
            voluntary_context_switches: nonnegative_counter(
                usage.ru_nvcsw,
                "voluntary context switches",
            )?,
            involuntary_context_switches: nonnegative_counter(
                usage.ru_nivcsw,
                "involuntary context switches",
            )?,
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn resource_snapshot() -> Result<ResourceSnapshot> {
        Err(BenchError::new("resource evidence requires Linux"))
    }

    #[cfg(target_os = "linux")]
    fn timeval_ns(value: libc::timeval) -> Result<u64> {
        if value.tv_sec < 0 || value.tv_usec < 0 {
            return Err(BenchError::new("getrusage returned a negative timeval"));
        }
        let seconds = u64::try_from(value.tv_sec)
            .map_err(|_| BenchError::new("rusage seconds do not fit u64"))?;
        let microseconds = u64::try_from(value.tv_usec)
            .map_err(|_| BenchError::new("rusage microseconds do not fit u64"))?;
        seconds
            .checked_mul(1_000_000_000)
            .and_then(|total| total.checked_add(microseconds.checked_mul(1_000)?))
            .ok_or_else(|| BenchError::new("rusage timeval overflows u64"))
    }

    #[cfg(target_os = "linux")]
    fn nonnegative_counter(value: libc::c_long, name: &str) -> Result<u64> {
        u64::try_from(value).map_err(|_| BenchError::new(format!("{name} counter is negative")))
    }

    pub(super) fn resource_delta(
        before: ResourceSnapshot,
        after: ResourceSnapshot,
    ) -> Result<ResourceUsage> {
        let subtract = |later: u64, earlier: u64, name: &str| {
            later
                .checked_sub(earlier)
                .ok_or_else(|| BenchError::new(format!("{name} counter decreased")))
        };
        Ok(ResourceUsage {
            user_cpu_ns: subtract(after.user_cpu_ns, before.user_cpu_ns, "user CPU")?,
            system_cpu_ns: subtract(after.system_cpu_ns, before.system_cpu_ns, "system CPU")?,
            minor_page_faults: subtract(
                after.minor_page_faults,
                before.minor_page_faults,
                "minor page fault",
            )?,
            major_page_faults: subtract(
                after.major_page_faults,
                before.major_page_faults,
                "major page fault",
            )?,
            voluntary_context_switches: subtract(
                after.voluntary_context_switches,
                before.voluntary_context_switches,
                "voluntary context switch",
            )?,
            involuntary_context_switches: subtract(
                after.involuntary_context_switches,
                before.involuntary_context_switches,
                "involuntary context switch",
            )?,
        })
    }

    #[cfg(target_os = "linux")]
    pub(super) fn current_affinity() -> Result<Vec<usize>> {
        // SAFETY: a zeroed `cpu_set_t` is a valid empty CPU set and is immediately
        // initialized by `sched_getaffinity` before it is inspected.
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        // SAFETY: `set` names writable storage of exactly the supplied size; PID 0
        // requests the calling thread's mask.
        let status = unsafe {
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw mut set)
        };
        if status != 0 {
            return Err(BenchError::new(format!(
                "sched_getaffinity failed: {}",
                io::Error::last_os_error()
            )));
        }
        let mut cpus = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            // SAFETY: `set` was initialized above and `cpu` is within CPU_SETSIZE.
            if unsafe { libc::CPU_ISSET(cpu, &set) } {
                cpus.push(cpu);
            }
        }
        if cpus.is_empty() {
            return Err(BenchError::new("affinity mask is empty"));
        }
        Ok(cpus)
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn current_affinity() -> Result<Vec<usize>> {
        Err(BenchError::new("CPU affinity evidence requires Linux"))
    }

    #[cfg(target_os = "linux")]
    pub(super) fn bind_current_thread(cpu: usize) -> Result<()> {
        if cpu >= libc::CPU_SETSIZE as usize {
            return Err(BenchError::new(format!("CPU {cpu} exceeds CPU_SETSIZE")));
        }
        // SAFETY: a zeroed `cpu_set_t` is a valid empty set.
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        // SAFETY: `cpu` is range-checked above and `set` is valid writable storage.
        unsafe {
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
        }
        // SAFETY: `set` is a fully initialized singleton mask of the supplied size;
        // PID 0 applies it only to the calling thread.
        let status = unsafe {
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw const set)
        };
        if status != 0 {
            return Err(BenchError::new(format!(
                "could not bind to CPU {cpu}: {}",
                io::Error::last_os_error()
            )));
        }
        if current_affinity()? != [cpu] {
            return Err(BenchError::new(format!(
                "CPU {cpu} binding did not produce a singleton mask"
            )));
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn bind_current_thread(_cpu: usize) -> Result<()> {
        Err(BenchError::new("CPU binding requires Linux"))
    }

    #[cfg(target_os = "linux")]
    pub(super) fn current_cpu() -> Result<i32> {
        // SAFETY: `sched_getcpu` has no pointer arguments or caller obligations.
        let cpu = unsafe { libc::sched_getcpu() };
        if cpu < 0 {
            return Err(BenchError::new(format!(
                "sched_getcpu failed: {}",
                io::Error::last_os_error()
            )));
        }
        Ok(cpu)
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn current_cpu() -> Result<i32> {
        Err(BenchError::new("CPU residency evidence requires Linux"))
    }

    pub(super) fn validate_distinct_physical_cores(cpus: &[usize]) -> Result<()> {
        if cpus.len() != 2 {
            return Err(BenchError::new(
                "physical-core validation requires exactly two CPUs",
            ));
        }
        let first = cpu_topology(cpus[0])?;
        let second = cpu_topology(cpus[1])?;
        if first == second {
            return Err(BenchError::new(format!(
                "CPUs {} and {} share package/core {first:?}",
                cpus[0], cpus[1]
            )));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn cpu_topology(cpu: usize) -> Result<(u64, u64)> {
        let root = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let package = read_topology_integer(&format!("{root}/physical_package_id"))?;
        let core = read_topology_integer(&format!("{root}/core_id"))?;
        Ok((package, core))
    }

    #[cfg(not(target_os = "linux"))]
    fn cpu_topology(_cpu: usize) -> Result<(u64, u64)> {
        Err(BenchError::new("CPU topology validation requires Linux"))
    }

    #[cfg(target_os = "linux")]
    fn read_topology_integer(path: &str) -> Result<u64> {
        let value = std::fs::read_to_string(path)
            .map_err(|error| BenchError::new(format!("could not read CPU topology: {error}")))?;
        value
            .trim()
            .parse()
            .map_err(|error| BenchError::new(format!("invalid CPU topology integer: {error}")))
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn read_mxcsr() -> Result<u32> {
        #[allow(deprecated)]
        // SAFETY: x86_64 guarantees SSE and `_mm_getcsr` only reads this thread's
        // control/status register without dereferencing memory.
        let value = unsafe { std::arch::x86_64::_mm_getcsr() };
        Ok(value)
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub(super) fn read_mxcsr() -> Result<u32> {
        Err(BenchError::new("MXCSR evidence requires x86_64"))
    }

    pub(super) fn validate_mxcsr(value: u32) -> Result<()> {
        let rounding_control = (value >> 13) & 0b11;
        let denormals_are_zero = value & (1 << 6) != 0;
        let flush_to_zero = value & (1 << 15) != 0;
        if rounding_control != 0 || denormals_are_zero || flush_to_zero {
            return Err(BenchError::new(format!(
                "MXCSR 0x{value:08x} is not RN with DAZ/FTZ clear"
            )));
        }
        Ok(())
    }
}

fn canonical_json_line(value: &impl Serialize) -> Result<Vec<u8>> {
    let value = serde_json::to_value(value)
        .map_err(|error| BenchError::new(format!("could not encode JSON value: {error}")))?;
    let mut bytes = serde_json::to_vec(&value)
        .map_err(|error| BenchError::new(format!("could not encode JSON bytes: {error}")))?;
    bytes.push(b'\n');
    Ok(bytes)
}

struct JsonLineStream<W> {
    writer: W,
    bytes_written: usize,
    maximum_bytes: usize,
}

impl<W: io::Write> JsonLineStream<W> {
    const fn new(writer: W, maximum_bytes: usize) -> Self {
        Self {
            writer,
            bytes_written: 0,
            maximum_bytes,
        }
    }

    fn emit(&mut self, value: &impl Serialize) -> Result<()> {
        let bytes = canonical_json_line(value)?;
        let next_total = self
            .bytes_written
            .checked_add(bytes.len())
            .ok_or_else(|| BenchError::new("streamed JSONL byte count overflow"))?;
        if next_total > self.maximum_bytes {
            return Err(BenchError::new(format!(
                "streamed JSONL exceeds {} bytes",
                self.maximum_bytes
            )));
        }
        self.writer.write_all(&bytes)?;
        self.writer.flush()?;
        self.bytes_written = next_total;
        Ok(())
    }
}

fn emit_json_lines(values: &[impl Serialize]) -> Result<()> {
    let mut output = Vec::new();
    for value in values {
        let row = canonical_json_line(value)?;
        if output.len().saturating_add(row.len()) > MAX_STDOUT_BYTES {
            return Err(BenchError::new(format!(
                "JSONL response exceeds {MAX_STDOUT_BYTES} bytes"
            )));
        }
        output.extend_from_slice(&row);
    }
    io::stdout().lock().write_all(&output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct RecordingWriter {
        bytes: Vec<u8>,
        flushes: usize,
        writes: usize,
        fail_on_write: Option<usize>,
    }

    impl io::Write for RecordingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if self.fail_on_write == Some(self.writes) {
                return Err(io::Error::other("injected streaming write failure"));
            }
            self.bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    fn request_for(cell: CellSpec) -> CellRequest {
        let cpus = if cell.workers == 1 {
            vec![0]
        } else {
            vec![0, 1]
        };
        let mut measured_orders = vec![PairOrder::BaselineCandidate; 15];
        measured_orders.extend([PairOrder::CandidateBaseline; 15]);
        CellRequest {
            schema: REQUEST_SCHEMA.to_owned(),
            cell_id: cell.id.to_owned(),
            case_id: cell.case_id.to_owned(),
            child_sequence: 0,
            cpus,
            warmup_orders: WARMUP_ORDERS.to_vec(),
            measured_orders,
        }
    }

    fn unit_failure_fixture() -> (CaseSpec, CellSpec, GeneratedCase, WorkerBuffers) {
        let specification = CaseSpec {
            id: "unit-3x5",
            rows: 3,
            columns: 5,
            calls_per_worker: 4,
            role: "test",
        };
        let cell = CellSpec {
            id: "unit-cell",
            case_id: specification.id,
            comparison: Comparison::Avx2,
            layout: Layout::Natural,
            workers: 1,
        };
        let generated = generate_case(specification, Layout::Natural).unwrap();
        let buffer = WorkerBuffers::new(specification, cell).unwrap();
        (specification, cell, generated, buffer)
    }

    #[test]
    fn tables_are_closed_and_resource_bounded() {
        assert_eq!(CASES.len(), 5);
        assert_eq!(CELLS.len(), 13);
        for case in CASES {
            assert!(element_products_per_worker(case).unwrap() >= 134_000_000);
            assert!(element_products_per_worker(case).unwrap() <= 135_000_000);
        }
        for cell in CELLS {
            let case = find_case(cell.case_id).unwrap();
            let accounted =
                enforce_memory_cap(case, cell.layout, cell.workers, cell.comparison).unwrap();
            assert!(accounted <= LIVE_MEMORY_CAP_BYTES);
        }
    }

    #[test]
    fn counter_stream_is_repeatable_and_domain_separated() {
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut input = Vec::new();
        fill_counter_bytes(WEIGHT_DOMAIN, "unit-case", 97, |_, byte| first.push(byte)).unwrap();
        fill_counter_bytes(WEIGHT_DOMAIN, "unit-case", 97, |_, byte| second.push(byte)).unwrap();
        fill_counter_bytes(INPUT_DOMAIN, "unit-case", 97, |_, byte| input.push(byte)).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, input);
        assert_eq!(
            format_digest(&Sha256::digest(&first)),
            "sha256:f7dbf5fa034a2ed9cd7863902c2220f9203937d47c05d70fc44ea8f051d664d2"
        );
    }

    #[test]
    fn all_generated_cases_have_pinned_canonical_digests() {
        for (case, (weight_digest, input_digest)) in CASES.into_iter().zip(EXPECTED_CASE_DIGESTS) {
            let generated = generate_case(case, Layout::Natural).unwrap();
            assert_eq!(generated.weight_sha256, weight_digest);
            assert_eq!(generated.input_sha256, input_digest);
            assert_eq!(generated.matrix.len(), case.rows * case.columns);
            assert_eq!(generated.input.as_slice().len(), case.columns);
        }
    }

    #[test]
    fn vector_offset_input_is_exactly_four_bytes_past_its_allocation() {
        let generated = generate_case(CASES[0], Layout::VectorOffset).unwrap();
        let allocation = generated.input.storage.as_ptr() as usize;
        let logical = generated.input.as_slice().as_ptr() as usize;
        assert_eq!(logical - allocation, std::mem::size_of::<f32>());
        assert_eq!(generated.matrix.words().as_ptr() as usize % 2, 0);
    }

    #[test]
    fn request_parser_rejects_noncanonical_duplicate_and_unknown_input() {
        let request = request_for(CELLS[0]);
        let canonical = canonical_json_line(&request).unwrap();
        assert_eq!(
            read_canonical_request(canonical.as_slice())
                .unwrap()
                .cell_id,
            cell_id(&request)
        );

        let mut whitespace = canonical.clone();
        whitespace.insert(1, b' ');
        assert!(read_canonical_request(whitespace.as_slice()).is_err());

        let duplicate = String::from_utf8(canonical.clone()).unwrap().replacen(
            '{',
            "{\"schema\":\"runnel.m4-cell-request/1\",",
            1,
        );
        assert!(read_canonical_request(duplicate.as_bytes()).is_err());

        let unknown = String::from_utf8(canonical)
            .unwrap()
            .replacen('{', "{\"unknown\":0,", 1);
        assert!(read_canonical_request(unknown.as_bytes()).is_err());
    }

    #[test]
    fn canonical_json_sorts_struct_keys_recursively() {
        #[derive(Serialize)]
        struct Nested {
            zebra: u8,
            alpha: u8,
        }

        #[derive(Serialize)]
        struct Outer {
            zebra: u8,
            nested: Nested,
            alpha: u8,
        }

        let bytes = canonical_json_line(&Outer {
            zebra: 4,
            nested: Nested { zebra: 3, alpha: 2 },
            alpha: 1,
        })
        .unwrap();
        assert_eq!(
            bytes,
            b"{\"alpha\":1,\"nested\":{\"alpha\":2,\"zebra\":3},\"zebra\":4}\n"
        );
    }

    #[test]
    fn jsonl_stream_flushes_each_self_authenticating_warmup_event() {
        let event = WarmupRow {
            schema: WARMUP_SCHEMA,
            cell_id: "unit-cell".to_owned(),
            case_id: "unit-case".to_owned(),
            child_sequence: 3,
            pair_sequence: 0,
            order: PairOrder::BaselineCandidate,
            baseline_status: ObservationStatus::Ok,
            candidate_status: ObservationStatus::Ok,
        };
        let mut writer = RecordingWriter::default();
        {
            let mut stream = JsonLineStream::new(&mut writer, 4096);
            stream.emit(&event).unwrap();
            stream.emit(&event).unwrap();
        }
        let line = canonical_json_line(&event).unwrap();
        assert_eq!(writer.bytes, [line.as_slice(), line.as_slice()].concat());
        assert_eq!(writer.flushes, 2);
    }

    #[test]
    fn jsonl_stream_preserves_flushed_prefix_on_later_write_failure() {
        let event = WarmupRow {
            schema: WARMUP_SCHEMA,
            cell_id: "unit-cell".to_owned(),
            case_id: "unit-case".to_owned(),
            child_sequence: 3,
            pair_sequence: 0,
            order: PairOrder::BaselineCandidate,
            baseline_status: ObservationStatus::Ok,
            candidate_status: ObservationStatus::Ok,
        };
        let first_line = canonical_json_line(&event).unwrap();
        let mut writer = RecordingWriter {
            fail_on_write: Some(2),
            ..RecordingWriter::default()
        };
        {
            let mut stream = JsonLineStream::new(&mut writer, 4096);
            stream.emit(&event).unwrap();
            assert!(stream.emit(&event).is_err());
        }
        assert_eq!(writer.bytes, first_line);
        assert_eq!(writer.flushes, 1);
    }

    fn cell_id(request: &CellRequest) -> String {
        request.cell_id.clone()
    }

    #[test]
    fn request_semantics_reject_wrong_case_counts_and_orders() {
        let mut request = request_for(CELLS[0]);
        validate_request(&request).unwrap();
        request.case_id = CASES[1].id.to_owned();
        assert!(validate_request(&request).is_err());

        let mut request = request_for(CELLS[0]);
        request.measured_orders.pop();
        assert!(validate_request(&request).is_err());

        let mut request = request_for(CELLS[0]);
        request.measured_orders.fill(PairOrder::BaselineCandidate);
        assert!(validate_request(&request).is_err());

        let mut request = request_for(CELLS[8]);
        request.cpus = vec![0];
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn anti_elision_validation_detects_a_missing_call_and_sink() {
        let output = [1.0_f32, -2.0, 3.0];
        let valid = BatchCore {
            executed_calls: 4,
            sink: expected_sink(&output, 4),
        };
        validate_batch_core(4, &valid, &output).unwrap();

        let missing = BatchCore {
            executed_calls: 3,
            sink: expected_sink(&output, 3),
        };
        assert!(validate_batch_core(4, &missing, &output).is_err());

        let wrong_sink = BatchCore {
            executed_calls: 4,
            sink: valid.sink ^ 1,
        };
        assert!(validate_batch_core(4, &wrong_sink, &output).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn two_worker_interval_excludes_setup_and_includes_execution() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ready_barrier = Barrier::new(3);
        let start_barrier = Barrier::new(3);
        let finish_barrier = Barrier::new(3);
        let ready_workers = AtomicUsize::new(0);
        let completed_workers = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..2 {
                handles.push(scope.spawn(|| {
                    ready_workers.fetch_add(1, Ordering::SeqCst);
                    ready_barrier.wait();
                    start_barrier.wait();
                    assert_eq!(ready_workers.load(Ordering::SeqCst), 2);
                    completed_workers.fetch_add(1, Ordering::SeqCst);
                    finish_barrier.wait();
                }));
            }

            assert_eq!(completed_workers.load(Ordering::SeqCst), 0);
            let measurements =
                measure_two_worker_interval(&ready_barrier, &start_barrier, &finish_barrier);
            assert_eq!(ready_workers.load(Ordering::SeqCst), 2);
            let IntervalStart {
                before_usage,
                started,
            } = measurements.start;
            let IntervalFinish {
                finished,
                after_usage,
            } = measurements.finish;
            let before_usage = before_usage.unwrap();
            let started = started.unwrap();
            let finished = finished.unwrap();
            let after_usage = after_usage.unwrap();
            assert_eq!(completed_workers.load(Ordering::SeqCst), 2);
            assert!(finished > started);
            resource_delta(before_usage, after_usage).unwrap();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn two_worker_barriers_release_when_clock_setup_fails() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ready_barrier = Barrier::new(3);
        let start_barrier = Barrier::new(3);
        let finish_barrier = Barrier::new(3);
        let completed_workers = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..2 {
                handles.push(scope.spawn(|| {
                    ready_barrier.wait();
                    start_barrier.wait();
                    completed_workers.fetch_add(1, Ordering::SeqCst);
                    finish_barrier.wait();
                }));
            }

            let (start_result, finish_result) = coordinate_two_worker_interval(
                &ready_barrier,
                &start_barrier,
                &finish_barrier,
                || Err::<(), _>(BenchError::new("injected clock setup failure")),
                || Ok::<(), BenchError>(()),
            );
            assert!(start_result.is_err());
            finish_result.unwrap();
            assert_eq!(completed_workers.load(Ordering::SeqCst), 2);

            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn preflight_mxcsr_check_ignores_only_sticky_status_flags() {
        let entry = 0x1f80_u32;
        validate_preflight_mxcsr_control(entry, entry | 0x3f).unwrap();
        assert!(validate_preflight_mxcsr_control(entry, entry ^ (1 << 13)).is_err());
        assert!(validate_preflight_mxcsr_control(entry, entry ^ (1 << 7)).is_err());
    }

    #[test]
    fn tiny_scalar_and_staged_batches_pass_the_f64_gate() {
        let specification = CaseSpec {
            id: "unit-3x5",
            rows: 3,
            columns: 5,
            calls_per_worker: 4,
            role: "test",
        };
        let generated = generate_case(specification, Layout::Natural).unwrap();
        let (scalar, scalar_diagnostic) =
            correctness_output(&generated, Implementation::ScalarBf16).unwrap();
        let (staged, staged_diagnostic) =
            correctness_output(&generated, Implementation::StagedF32).unwrap();
        assert!(scalar_diagnostic.passed);
        assert!(staged_diagnostic.passed);
        assert_eq!(scalar, staged);

        let input = FiniteInput::new(generated.input.as_slice()).unwrap();
        let prepared = PreparedGemv::new(&generated.matrix, input, BackendRequest::Scalar).unwrap();
        let mut workspace = GemvWorkspace::try_new(specification.rows).unwrap();
        let mut output = vec![0.0; specification.rows];
        let attempt = run_prepared_calls(
            &prepared,
            &mut workspace,
            &mut output,
            specification.calls_per_worker,
        );
        assert!(attempt.failures.is_empty());
        validate_batch_core(specification.calls_per_worker, &attempt.core, &output).unwrap();
    }

    #[test]
    fn prepared_execution_is_reused_for_preflight_and_timed_calls() {
        let specification = CaseSpec {
            id: "unit-3x5",
            rows: 3,
            columns: 5,
            calls_per_worker: 4,
            role: "test",
        };
        let cell = CellSpec {
            id: "unit-cell",
            case_id: specification.id,
            comparison: Comparison::Avx2,
            layout: Layout::Natural,
            workers: 1,
        };
        let generated = generate_case(specification, Layout::Natural).unwrap();
        let mut buffer = WorkerBuffers::new(specification, cell).unwrap();
        let input = validated_input(&generated, Implementation::ScalarBf16).unwrap();
        let prepared = prepare_execution(&generated, Implementation::ScalarBf16, input).unwrap();

        preflight(&generated, &mut buffer, &prepared).unwrap();
        let attempt = execute_calls(&generated, &mut buffer, &prepared);
        assert!(attempt.failures.is_empty());
        validate_batch_core(
            specification.calls_per_worker,
            &attempt.core,
            buffer.output.as_slice(),
        )
        .unwrap();
    }

    #[test]
    fn post_start_failure_retains_observed_batch_evidence() {
        let (specification, cell, generated, buffer) = unit_failure_fixture();
        let before_usage = ResourceSnapshot {
            user_cpu_ns: 10,
            system_cpu_ns: 20,
            minor_page_faults: 1,
            major_page_faults: 0,
            voluntary_context_switches: 2,
            involuntary_context_switches: 3,
        };
        let after_usage = ResourceSnapshot {
            user_cpu_ns: 40,
            system_cpu_ns: 60,
            minor_page_faults: 5,
            major_page_faults: 1,
            voluntary_context_switches: 7,
            involuntary_context_switches: 9,
        };
        let capture = SingleBatchCapture {
            cpu: 0,
            affinity: vec![0],
            mxcsr_before: 0x1f80,
            cpu_before: 0,
            before_usage,
            started: 100,
            execution: ExecutionAttempt {
                core: BatchCore {
                    executed_calls: 2,
                    sink: 0x1234,
                },
                failures: vec!["injected timed kernel failure".to_owned()],
            },
            finished: Ok(250),
            after_usage: Ok(after_usage),
            cpu_after: Ok(0),
            mxcsr_after: Ok(0x1f80),
        };
        let failure = finalize_single_batch(&generated, &buffer, capture).unwrap_err();
        assert_eq!(failure.elapsed_ns, Some(150));
        assert_eq!(failure.executed_calls, 2);
        assert!(failure.sink_sha256.is_some());
        assert!(failure.output_sha256.is_some());
        assert!(failure.addresses.is_some());
        assert_eq!(failure.workers.len(), 1);
        assert!(failure.resource_usage.is_some());

        let request = request_for(cell);
        let row = observation_row(
            &request,
            cell,
            specification,
            0,
            PairOrder::BaselineCandidate,
            0,
            Variant::Baseline,
            Implementation::ScalarBf16,
            BatchOutcome::Failed(*failure),
        )
        .unwrap();
        assert_eq!(row.status, ObservationStatus::KernelError);
        assert_eq!(row.elapsed_ns, Some(150));
        assert_eq!(row.executed_calls, 2);
        assert!(row.addresses_mod_64.is_some());
        assert_eq!(row.workers.len(), 1);
        assert!(row.resource_usage.is_some());

        let partial = finalize_single_batch(
            &generated,
            &buffer,
            SingleBatchCapture {
                cpu: 0,
                affinity: vec![0],
                mxcsr_before: 0x1f80,
                cpu_before: 0,
                before_usage,
                started: 300,
                execution: ExecutionAttempt {
                    core: BatchCore {
                        executed_calls: 3,
                        sink: 0x5678,
                    },
                    failures: vec!["injected timed kernel failure".to_owned()],
                },
                finished: Ok(500),
                after_usage: Err(BenchError::new("injected resource probe failure")),
                cpu_after: Ok(1),
                mxcsr_after: Ok(0x1f81),
            },
        )
        .unwrap_err();
        assert_eq!(partial.elapsed_ns, Some(200));
        assert_eq!(partial.executed_calls, 3);
        assert!(partial.addresses.is_some());
        assert_eq!(partial.workers.len(), 1);
        assert!(partial.resource_usage.is_none());
        assert_eq!(partial.workers[0].cpu_after, 1);
        assert_eq!(partial.workers[0].mxcsr_after, 0x1f81);
    }

    #[test]
    fn two_worker_failure_retains_aggregate_and_per_worker_evidence() {
        let specification = CaseSpec {
            id: "unit-3x5",
            rows: 3,
            columns: 5,
            calls_per_worker: 4,
            role: "test",
        };
        let cell = CellSpec {
            id: "unit-two-worker",
            case_id: specification.id,
            comparison: Comparison::Avx2,
            layout: Layout::Natural,
            workers: 2,
        };
        let generated = generate_case(specification, Layout::Natural).unwrap();
        let buffers = (0..2)
            .map(|_| WorkerBuffers::new(specification, cell).unwrap())
            .collect::<Vec<_>>();
        let snapshot = |user_cpu_ns| ResourceSnapshot {
            user_cpu_ns,
            system_cpu_ns: user_cpu_ns,
            minor_page_faults: user_cpu_ns,
            major_page_faults: 0,
            voluntary_context_switches: user_cpu_ns,
            involuntary_context_switches: user_cpu_ns,
        };
        let thread = |worker_index, executed_calls, failures| ThreadBatch {
            worker_index,
            core: BatchCore {
                executed_calls,
                sink: u64::try_from(worker_index).unwrap(),
            },
            metadata: Some(WorkerMetadata {
                worker_index,
                cpu: worker_index,
                cpu_before: i32::try_from(worker_index).unwrap(),
                cpu_after: i32::try_from(worker_index).unwrap(),
                affinity: vec![worker_index],
                mxcsr_before: 0x1f80,
                mxcsr_after: 0x1f80,
            }),
            failures,
        };
        let batch = TwoWorkerBatch {
            measurements: IntervalMeasurements {
                start: IntervalStart {
                    before_usage: Ok(snapshot(1)),
                    started: Ok(100),
                },
                finish: IntervalFinish {
                    finished: Ok(400),
                    after_usage: Ok(snapshot(5)),
                },
            },
            thread_results: vec![
                thread(0, 2, vec!["injected worker failure".to_owned()]),
                thread(1, 4, Vec::new()),
            ],
            failures: Vec::new(),
        };

        let failure = finalize_two_worker_batch(&generated, &buffers, batch).unwrap_err();
        assert_eq!(failure.elapsed_ns, Some(300));
        assert_eq!(failure.executed_calls, 6);
        assert!(failure.sink_sha256.is_some());
        assert!(failure.output_sha256.is_some());
        assert!(failure.addresses.is_some());
        assert_eq!(failure.workers.len(), 2);
        assert!(failure.resource_usage.is_some());
        assert!(failure.failure.contains("injected worker failure"));
    }

    #[test]
    fn correctness_mismatch_retains_componentwise_diagnostics() {
        let specification = CaseSpec {
            id: "unit-3x5",
            rows: 3,
            columns: 5,
            calls_per_worker: 1,
            role: "test",
        };
        let generated = generate_case(specification, Layout::Natural).unwrap();
        let output = vec![f32::MAX; specification.rows];
        let diagnostic = numerical_diagnostic(&generated, &output).unwrap();
        assert!(!diagnostic.passed);
        assert!(diagnostic.max_error_to_bound_ratio > 1.0);

        let row = correctness_row(specification, "scalar", Ok((&output, &diagnostic)), None);
        assert_eq!(row.status, RowStatus::Failed);
        assert_eq!(row.failure.as_deref(), Some("numerical_bound_exceeded"));
        assert!(row.metrics.max_abs_error.is_some());
        assert!(row.metrics.max_error_to_bound_ratio.unwrap() > 1.0);
        assert!(row.metrics.worst_error_bound.unwrap() > 0.0);

        let execution_error = BenchError::new("test execution failure");
        let row = correctness_row(specification, "scalar", Err(&execution_error), None);
        assert_eq!(row.status, RowStatus::Failed);
        assert_eq!(row.failure.as_deref(), Some("kernel_execution_error"));
        assert!(row.metrics.max_abs_error.is_none());
        assert!(row.metrics.max_error_to_bound_ratio.is_none());
    }

    #[test]
    fn independent_backend_proof_survives_missing_scalar_cross_diagnostics() {
        let specification = CaseSpec {
            id: "unit-3x5",
            rows: 3,
            columns: 5,
            calls_per_worker: 1,
            role: "test",
        };
        let generated = generate_case(specification, Layout::Natural).unwrap();
        let (output, diagnostic) =
            correctness_output(&generated, Implementation::StagedF32).unwrap();
        let row = correctness_row(specification, "staged", Ok((&output, &diagnostic)), None);

        assert_eq!(row.status, RowStatus::Ok);
        assert!(row.failure.is_none());
        assert!(row.metrics.max_abs_error.is_some());
        assert!(row.metrics.max_error_to_bound_ratio.is_some());
        assert!(row.metrics.worst_index.is_some());
        assert!(row.metrics.worst_abs_error.is_some());
        assert!(row.metrics.worst_error_bound.is_some());
        assert!(row.metrics.cross_scalar_max_abs.is_none());
        assert!(row.metrics.cross_scalar_max_rel.is_none());
        assert!(row.metrics.cross_scalar_max_ulp.is_none());
    }

    #[test]
    fn command_parser_is_closed() {
        assert!(run_cli(["unknown".to_owned()]).is_err());
        assert!(run_cli(Vec::<String>::new()).is_err());
        assert!(run_cli(["cases".to_owned(), "extra".to_owned()]).is_err());
    }
}
