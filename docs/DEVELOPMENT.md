# Development environment

## M0 host inventory

Captured 2026-08-03 without recording credentials:

| Resource | Available at bootstrap |
| --- | --- |
| OS/kernel | Ubuntu 24.04 userland, Linux 6.17 AWS x86_64 |
| CPU | 4 vCPUs, AMD EPYC 7R13, 1 NUMA node, AVX2 (no AVX-512 exposed) |
| Memory | 30 GiB total; available memory is a required live preflight |
| Swap | 8 GiB total |
| Root filesystem | ext4 on NVMe; free space is a required live preflight |
| Rust | rustc/cargo 1.97.1; rustfmt and clippy in the pinned toolchain |
| Python | CPython 3.12.3; exact CPU-oracle closure in `oracle/requirements.txt` |
| Native compiler | GCC 13.3; additional kernel tooling is not required by M0 |
| Measurement | perf 6.17 |
| Packaging | Docker 29.1; no local image build authorized by default |
| GitHub | `omar07ibrahim`, HTTPS auth, repository/workflow capability |

Reproduce the non-secret inventory with `uname -srmo`, `lscpu`, `free -h`,
`df -h /`, and each named tool's `--version`. Volatile availability values
are deliberately not project claims.

The execution environment also provides a persistent Goal and bounded parallel
review agents. These coordinate development but are not build dependencies or
evidence sources.

## Disk guard

Keep at least 2 GiB free. Before dependency installs, release builds, fuzzing,
or container work, run:

    df -h /
    du -sh target .git benchmarks fixtures 2>/dev/null

Use at most two build jobs on this host. Avoid simultaneous debug/release target
trees, disable incremental compilation for evidence builds, and remove only
project-owned reproducible artifacts when reclaiming space. Never download the
Kimi K3 checkpoint or another multi-gigabyte model.

## M1 environment

The repository contract uses only Python 3.12 standard-library modules. Rust is
pinned by `rust-toolchain.toml` and Cargo resolves exactly `Cargo.lock`. The
independent oracle is optional for ordinary CLI use and has an exact CPU-only
dependency closure:

```console
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --no-deps -r oracle/requirements.txt
```

Installing PyTorch consumes substantial temporary and installed disk space, so
run the disk guard first. No command downloads a model checkpoint.

## Complete M1 verification

From the repository root:

```console
python3 scripts/verify_repository.py
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
.venv/bin/python -m oracle.generate --check
.venv/bin/python -m unittest discover -s oracle/tests -v
cargo run --locked -p runnel -- demo --prompt moe --max-new-tokens 4 --json
```

After dependencies have been fetched once, append `--offline` to Cargo test,
Clippy, documentation, and run commands for a network-independent replay.
`oracle.generate --check` is non-mutating and uses no network.

To inspect the actual tiny RMOA layout, choose a new output directory:

```console
cargo run --locked -p runnel -- fixture --output /tmp/runnel-tiny-rmoa --json
cargo run --locked -p runnel -- generate --artifact /tmp/runnel-tiny-rmoa --prompt moe --max-new-tokens 4 --strategy greedy --json
```

The fixture command refuses an existing output root. Generated object and page
table bytes are disposable and excluded from source control.

## M2 verified-data-plane verification

The M2 store is Linux-oriented and uses only tiny generated artifacts. It does
not download a model. Run the disk guard first, then:

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
python3 -m unittest discover -s scripts/tests -v
cargo run --locked -p runnel -- data-plane-demo --json
```

Append `--offline` to the Cargo commands after the locked dependency set is
available. The storage suite covers descriptor traversal, corruption,
truncation, reordered pages, cancellation/deadline checkpoints, exact disk and
page-pool boundaries, resumable stages, no-replace publication ambiguity,
fail-closed collection, queue saturation, cache single-flight, eviction,
retiring leases, and sync/async numerical parity.

### Reproduce M2 raw evidence

Evidence must name a clean commit. Confirm `git status --short` is empty and
choose a new lowercase ID:

```console
python3 scripts/run_m2_experiment.py m2-data-plane-forced-eviction-YYYYMMDD
```

The standard-library harness performs the recorded two-job, nonincremental,
locked offline release build from that clean commit. It refuses dirty trees,
less than 2 GiB of free disk, existing result directories, absolute paths,
fewer than 30 measured repetitions, missing observations, or an incorrect
demo. It writes the experiment contract, allowlisted environment metadata,
every raw trial, and a generated statistical summary below
`benchmarks/raw/<id>/`.
Timing describes the validation command on a shared virtualized host; it is not
evidence of a performance improvement.

The accepted schema-v2 M2 evidence is committed at
`benchmarks/raw/m2-data-plane-forced-eviction-20260803/`. Its clean
implementation commit is
`99772585e23d8f1ce3459ba6397d93fd1be0fc8a`; the exact recorded invocation is:

```console
python3 scripts/run_m2_experiment.py m2-data-plane-forced-eviction-20260803 --warmups 3 --repetitions 30 --timeout-seconds 30 --bootstrap-seed 20260803 --bootstrap-resamples 10000
```

Calling the harness with only a new experiment ID uses those same default
parameters, but the record always preserves the fully expanded invocation.
The schema-v1 `benchmarks/raw/m2-data-plane-20260803/` run is retained as a
preliminary append-only record; it does not combine full-generation parity with
forced eviction. Use a new experiment ID when reproducing either procedure.

## M3 cache-policy verification

The simulator and both independent policy references use no model download.
Run the full workspace gate plus the deterministic matrix smoke test:

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --all-targets --locked --offline
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked --offline
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s oracle/tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -v
cargo run --locked --offline -p runnel-sim --bin runnel-cache-sim -- matrix --family markov_clusters --replicate 0 --measured-steps 64
```

The Rust tests include byte-ledger identities, malformed trace bytes, retained
descriptor replacement/FIFO/growth cases, exact and brute-force oracle
comparisons, a real M2 `PageCache` cross-check, prefetch classification, and
Rust-versus-Python online-policy differentials. The 64-step matrix is a
functional smoke test, not accepted research evidence or a timing result.

### Capture and verify the primary M3 matrix

Capture requires a clean implementation commit, at least 2 GiB plus the
16 MiB evidence allowance free on the repository filesystem, and a Linux
tmpfs build root with sufficient memory. The harness creates and later removes
only its own private mode-0700 child below that build root:

```console
python3 scripts/run_m3_experiment.py capture \
  --build-root /dev/shm \
  --output benchmarks/raw/m3-cache-policies-YYYYMMDD \
  --commit <full-40-character-HEAD>

python3 scripts/run_m3_experiment.py verify \
  --input benchmarks/raw/m3-cache-policies-YYYYMMDD --check
```

Capture performs a fresh two-job locked/offline release build; a caller cannot
supply a binary. For each of 180 family/replicate pairs it validates and
discards one expanded canonical trace, independently reconstructs the measured
routes and digest, then records the 18 policy/capacity observations. It checks
clean HEAD, the historical harness blob, and the executable hash again before
atomic publication. The resulting intervals are unadjusted exploratory
per-cell descriptions and support no omnibus “any policy wins” claim.
