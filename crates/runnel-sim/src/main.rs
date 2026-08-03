//! Command-line entry point for deterministic cache-policy replay.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use runnel_sim::{
    PolicySpec, RouterPolicyConfig, SimError, SimLimits, SimulationConfig, SimulationResult,
    TinyLfuConfig, TraceFamily, ValidatedTrace, generate_trace, parse_trace, serialize_trace,
    simulate,
};
use rustix::fs::{FileType, Mode, OFlags, fstat, openat};
use serde::Serialize;

const PAGE_BYTES: u64 = 65_536;
const CAPACITY_PAGES: [u64; 3] = [32, 64, 128];
const TRACE_OPEN_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC)
    .union(OFlags::NONBLOCK);

#[derive(Debug, Parser)]
#[command(name = "runnel-cache-sim")]
#[command(about = "Bounded byte-accounted cache-policy simulator")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate a canonical JSONL trace and print its identity.
    Validate {
        /// Canonical trace path. Symbolic links are rejected.
        trace: PathBuf,
    },
    /// Replay one trace under one policy.
    Simulate {
        /// Canonical trace path. Symbolic links are rejected.
        trace: PathBuf,
        /// Named policy.
        #[arg(long, value_enum)]
        policy: PolicyArg,
        /// Page-pool byte capacity.
        #[arg(long)]
        capacity_bytes: u64,
    },
    /// Write one openly reproducible synthetic trace to standard output.
    Generate {
        /// Preregistered workload family.
        #[arg(long)]
        family: TraceFamily,
        /// Paired replicate index.
        #[arg(long)]
        replicate: u32,
        /// Measured route steps after generator burn-in.
        #[arg(long, default_value_t = runnel_sim::DEFAULT_MEASURED_STEPS)]
        measured_steps: usize,
    },
    /// Generate one trace and replay the complete M3 policy/capacity matrix.
    Matrix {
        /// Preregistered workload family.
        #[arg(long)]
        family: TraceFamily,
        /// Paired replicate index.
        #[arg(long)]
        replicate: u32,
        /// Measured route steps after generator burn-in.
        #[arg(long, default_value_t = runnel_sim::DEFAULT_MEASURED_STEPS)]
        measured_steps: usize,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PolicyArg {
    NoCache,
    Lru,
    Slru,
    TinyLfu,
    RouterAdmit,
    RouterPrefetch,
    Belady,
}

#[derive(Serialize)]
struct ValidationOutput<'a> {
    schema: &'static str,
    trace_id: &'a str,
    trace_sha256: &'a str,
    page_count: usize,
    event_count: usize,
}

#[derive(Serialize)]
struct MatrixOutput<'a> {
    schema: &'static str,
    family: &'a str,
    replicate: u32,
    measured_steps: usize,
    seed_sha256: &'a str,
    full_route_sha256: &'a str,
    measured_route_sha256: &'a str,
    trace_sha256: &'a str,
    results: Vec<SimulationResult>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), SimError> {
    let cli = Cli::parse();
    match cli.command {
        Command::Validate { trace } => {
            let trace = load_trace(&trace)?;
            write_json(&ValidationOutput {
                schema: "runnel.cache-validation/1",
                trace_id: &trace.header().trace_id,
                trace_sha256: trace.sha256(),
                page_count: trace.pages().len(),
                event_count: trace.events().len(),
            })?;
        }
        Command::Simulate {
            trace,
            policy,
            capacity_bytes,
        } => {
            let trace = load_trace(&trace)?;
            let policy = policy_spec(policy, &trace, capacity_bytes)?;
            let result = simulate(
                &trace,
                &SimulationConfig {
                    capacity_bytes,
                    policy,
                },
            )?;
            write_json(&result)?;
        }
        Command::Generate {
            family,
            replicate,
            measured_steps,
        } => {
            let generated = generate_trace(family, replicate, measured_steps)?;
            let canonical = serialize_trace(
                generated.trace.header(),
                generated.trace.pages(),
                generated.trace.events(),
                SimLimits::default(),
            )?;
            io::stdout().write_all(&canonical)?;
        }
        Command::Matrix {
            family,
            replicate,
            measured_steps,
        } => {
            let generated = generate_trace(family, replicate, measured_steps)?;
            let mut results = Vec::with_capacity(CAPACITY_PAGES.len() * 6);
            for capacity_pages in CAPACITY_PAGES {
                let capacity_bytes = capacity_pages
                    .checked_mul(PAGE_BYTES)
                    .ok_or(SimError::CounterOverflow("matrix capacity bytes"))?;
                for policy in [
                    PolicyArg::Lru,
                    PolicyArg::Slru,
                    PolicyArg::TinyLfu,
                    PolicyArg::RouterAdmit,
                    PolicyArg::RouterPrefetch,
                    PolicyArg::Belady,
                ] {
                    let policy = policy_spec(policy, &generated.trace, capacity_bytes)?;
                    results.push(simulate(
                        &generated.trace,
                        &SimulationConfig {
                            capacity_bytes,
                            policy,
                        },
                    )?);
                }
            }
            write_json(&MatrixOutput {
                schema: "runnel.cache-matrix/1",
                family: family.as_str(),
                replicate,
                measured_steps,
                seed_sha256: &generated.seed_sha256,
                full_route_sha256: &generated.full_route_sha256,
                measured_route_sha256: &generated.measured_route_sha256,
                trace_sha256: generated.trace.sha256(),
                results,
            })?;
        }
    }
    Ok(())
}

fn load_trace(path: &Path) -> Result<ValidatedTrace, SimError> {
    let limits = SimLimits::default();
    let (file, initial_length) = open_trace_regular(path)?;
    let bytes = read_bounded_trace(file, initial_length, limits.max_trace_bytes)?;
    parse_trace(&bytes, limits)
}

fn open_trace_regular(path: &Path) -> Result<(File, u64), SimError> {
    let descriptor =
        openat(rustix::fs::CWD, path, TRACE_OPEN_FLAGS, Mode::empty()).map_err(|error| {
            if error == rustix::io::Errno::LOOP {
                SimError::TraceInputSymlink
            } else {
                SimError::Io(error.into())
            }
        })?;
    let metadata = fstat(&descriptor).map_err(|error| SimError::Io(error.into()))?;
    if FileType::from_raw_mode(metadata.st_mode) != FileType::RegularFile {
        return Err(SimError::TraceInputNotRegular);
    }
    let length = u64::try_from(metadata.st_size).map_err(|_| SimError::TraceInputNotRegular)?;
    Ok((File::from(descriptor), length))
}

fn read_bounded_trace(
    mut file: File,
    initial_length: u64,
    maximum: usize,
) -> Result<Vec<u8>, SimError> {
    let limit = u64::try_from(maximum)
        .map_err(|_| SimError::InvalidTrace("trace byte limit does not fit u64".to_owned()))?;
    if initial_length > limit {
        return Err(SimError::TraceInputTooLarge {
            observed_at_least: initial_length,
            limit,
        });
    }
    let read_limit = limit
        .checked_add(1)
        .ok_or(SimError::CounterOverflow("trace read limit"))?;
    let initial_capacity = usize::try_from(initial_length)
        .map_err(|_| SimError::InvalidTrace("trace length does not fit usize".to_owned()))?;
    let mut bytes = Vec::with_capacity(initial_capacity);
    {
        let mut bounded = Read::by_ref(&mut file).take(read_limit);
        bounded.read_to_end(&mut bytes)?;
    }
    let bytes_read = u64::try_from(bytes.len())
        .map_err(|_| SimError::InvalidTrace("trace read length does not fit u64".to_owned()))?;
    if bytes_read > limit {
        return Err(SimError::TraceInputTooLarge {
            observed_at_least: bytes_read,
            limit,
        });
    }
    let final_metadata = fstat(&file).map_err(|error| SimError::Io(error.into()))?;
    let final_length =
        u64::try_from(final_metadata.st_size).map_err(|_| SimError::TraceInputNotRegular)?;
    if final_length > limit {
        return Err(SimError::TraceInputTooLarge {
            observed_at_least: final_length,
            limit,
        });
    }
    if initial_length != bytes_read || final_length != bytes_read {
        return Err(SimError::TraceInputChanged {
            initial: initial_length,
            bytes_read,
            final_length,
        });
    }
    Ok(bytes)
}

fn policy_spec(
    policy: PolicyArg,
    trace: &ValidatedTrace,
    capacity_bytes: u64,
) -> Result<PolicySpec, SimError> {
    let router = |max_prefetch_bytes_per_signal| RouterPolicyConfig {
        protected_fraction_ppm: 750_000,
        minimum_score_ppm: 100_000,
        max_experts_per_signal: 2,
        max_pages_per_signal: 6,
        max_prefetch_bytes_per_signal,
    };
    Ok(match policy {
        PolicyArg::NoCache => PolicySpec::NoCache,
        PolicyArg::Lru => PolicySpec::Lru,
        PolicyArg::Slru => PolicySpec::Slru {
            protected_fraction_ppm: 750_000,
        },
        PolicyArg::TinyLfu => PolicySpec::TinyLfu {
            config: TinyLfuConfig::m3(
                trace
                    .pages()
                    .iter()
                    .map(|page| page.charge_bytes)
                    .min()
                    .map_or(1, |minimum_charge| capacity_bytes / minimum_charge),
            )?,
        },
        PolicyArg::RouterAdmit => {
            let maximum_logical = trace
                .pages()
                .iter()
                .map(|page| page.logical_bytes)
                .max()
                .unwrap_or(0);
            PolicySpec::RouterAdmit {
                config: router(maximum_logical.saturating_mul(6)),
            }
        }
        PolicyArg::RouterPrefetch => {
            let maximum_logical = trace
                .pages()
                .iter()
                .map(|page| page.logical_bytes)
                .max()
                .unwrap_or(0);
            PolicySpec::RouterPrefetch {
                config: router(
                    maximum_logical
                        .checked_mul(6)
                        .ok_or(SimError::CounterOverflow("router prefetch byte limit"))?,
                ),
            }
        }
        PolicyArg::Belady => PolicySpec::Belady,
    })
}

fn write_json(value: &impl Serialize) -> Result<(), SimError> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    #[test]
    fn m3_policy_derivation_remains_frozen_for_generated_geometry() {
        let generated = generate_trace(TraceFamily::IidUniform, 0, 8).unwrap();
        let capacity_bytes = 32 * PAGE_BYTES;
        let expected_router = RouterPolicyConfig {
            protected_fraction_ppm: 750_000,
            minimum_score_ppm: 100_000,
            max_experts_per_signal: 2,
            max_pages_per_signal: 6,
            max_prefetch_bytes_per_signal: 6 * PAGE_BYTES,
        };

        assert_eq!(
            policy_spec(PolicyArg::TinyLfu, &generated.trace, capacity_bytes).unwrap(),
            PolicySpec::TinyLfu {
                config: TinyLfuConfig::m3(32).unwrap(),
            }
        );
        assert_eq!(
            policy_spec(PolicyArg::RouterAdmit, &generated.trace, capacity_bytes).unwrap(),
            PolicySpec::RouterAdmit {
                config: expected_router.clone(),
            }
        );
        assert_eq!(
            policy_spec(PolicyArg::RouterPrefetch, &generated.trace, capacity_bytes,).unwrap(),
            PolicySpec::RouterPrefetch {
                config: expected_router,
            }
        );
    }

    #[test]
    fn retained_descriptor_is_not_reopened_after_path_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("trace.jsonl");
        let original = b"original\n";
        std::fs::write(&path, original).unwrap();
        let (file, initial_length) = open_trace_regular(&path).unwrap();

        let moved = temporary.path().join("moved.jsonl");
        std::fs::rename(&path, &moved).unwrap();
        std::fs::write(&path, b"replacement\n").unwrap();

        assert_eq!(
            read_bounded_trace(file, initial_length, 64).unwrap(),
            original
        );
    }

    #[test]
    fn retained_descriptor_read_rejects_growth_at_limit_plus_one() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("trace.jsonl");
        std::fs::write(&path, b"1234").unwrap();
        let (file, initial_length) = open_trace_regular(&path).unwrap();
        let mut writer = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writer.write_all(b"56789").unwrap();
        writer.flush().unwrap();

        assert!(matches!(
            read_bounded_trace(file, initial_length, 8),
            Err(SimError::TraceInputTooLarge {
                observed_at_least: 9,
                limit: 8
            })
        ));
    }

    #[test]
    fn bounded_read_accepts_exact_limit_and_rejects_initial_oversize() {
        let temporary = tempfile::tempdir().unwrap();
        let exact = temporary.path().join("exact.jsonl");
        std::fs::write(&exact, b"12345678").unwrap();
        let (file, initial_length) = open_trace_regular(&exact).unwrap();
        assert_eq!(
            read_bounded_trace(file, initial_length, 8).unwrap(),
            b"12345678"
        );

        let oversized = temporary.path().join("oversized.jsonl");
        std::fs::File::create(&oversized)
            .unwrap()
            .set_len(9)
            .unwrap();
        let (file, initial_length) = open_trace_regular(&oversized).unwrap();
        assert!(matches!(
            read_bounded_trace(file, initial_length, 8),
            Err(SimError::TraceInputTooLarge {
                observed_at_least: 9,
                limit: 8
            })
        ));
    }
}
