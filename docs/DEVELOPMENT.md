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

The accepted schema-v2 record is
`benchmarks/raw/m3-cache-policies-20260803/`. It was captured from clean
implementation commit `fcbaaebb7e211a228ee75d9cab713acc2b9890ac` and published
in commit `700de98110db58e05f5db24d29bb14f8a482cffc`. The exact canonical
command, artifact hashes, independent review, and interpretation are in the
[M3 review](reviews/M3_REVIEW.md). CI runs the archival `verify --check`
command against this append-only directory. Evidence-bearing pull requests
must use a merge commit: squashing or rebasing would discard the historical
implementation commit whose harness blob the archival verifier checks.

Capture performs a fresh two-job locked/offline release build; a caller cannot
supply a binary. For each of 180 family/replicate pairs it validates and
discards one expanded canonical trace, independently reconstructs the measured
routes and digest, then records the 18 policy/capacity observations. It checks
clean HEAD, the historical harness blob, and the executable hash again before
atomic publication. The resulting intervals are unadjusted exploratory
per-cell descriptions and support no omnibus “any policy wins” claim.

## M4 compact-BF16 kernel verification

M4 adds a native object only for a native Linux x86-64 GNU build. The safe
scalar backend remains available with default features disabled. No command
downloads a model; both adapter fixtures are formula-generated. Run the disk
guard before compiling, then use:

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --all-targets --locked --offline
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked --offline
cargo clippy -p runnel-kernels --all-targets --no-default-features --locked --offline -- -D warnings
cargo test -p runnel-kernels --all-targets --no-default-features --locked --offline
cargo clippy -p runnel-runtime --all-targets --no-default-features --locked --offline -- -D warnings
cargo test -p runnel-runtime --all-targets --no-default-features --locked --offline
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s oracle/tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts.tests.test_run_m4_experiment -v
cargo run --locked --offline -p runnel --bin runnel-m4-model-check
```

CI additionally compiles and runs the direct C ABI harness under AddressSanitizer
and UndefinedBehaviorSanitizer, checks scalar-only Rust targets for native
objects, and verifies that the default-feature AArch64 cross target stays
scalar-only. The model-check command emits correctness JSONL; it is not a
timing benchmark.

### Capture and verify the M4 kernel matrix

Capture requires clean `HEAD`, a Linux tmpfs build root, at least 2 GiB plus
the evidence reserve free there, and at least 32 MiB free on the repository
filesystem. It creates a private two-job, locked/offline, nonincremental
release build and refuses caller-supplied binaries:

```console
python3 scripts/run_m4_experiment.py capture \
  --build-root /dev/shm \
  --output benchmarks/raw/m4-bf16-gemv-YYYYMMDD \
  --commit <full-40-character-HEAD>

python3 scripts/run_m4_experiment.py verify \
  --input benchmarks/raw/m4-bf16-gemv-YYYYMMDD --check
```

The accepted record is `benchmarks/raw/m4-bf16-gemv-20260803/`. It was
captured from clean implementation commit
`035d217baf0901809fa02bf0a5c11c1a490198c2` and published in commit
`ae69481def4b320ff619090ce7793ef3d65ace33`. Its exact command, hashes,
independent reviews, host caveats, and claim boundaries are in the
[M4 review](reviews/M4_REVIEW.md). CI byte-regenerates the summary and both
figures. Preserve the implementation commit with a merge commit; squashing or
rebasing would break historical harness custody.

## M5 direct atomic batch-admission verification

The direct synchronous engine batch API is not the Tokio actor command path.
Its tests cover complete-slice validation, FIFO-prefix pressure parity,
release-relative and absolute deadline rollback, exact ID/generation rollback,
semantic reserve exact-fit/one-byte-short construction, holey free-list
equivalence, stale-control races, inert permit drop, endpoint order, full
completion/cancellation/reap, and slot/control/endpoint reuse. No command below
downloads weights or records timing evidence:

```console
cargo fmt --all -- --check
cargo clippy -p runnel-scheduler --all-targets --all-features --locked --offline -- -D warnings
cargo test -p runnel-scheduler --all-targets --all-features --locked --offline
cargo clippy -p runnel-scheduler --all-targets --no-default-features --locked --offline -- -D warnings
cargo test -p runnel-scheduler --all-targets --no-default-features --locked --offline
RUSTDOCFLAGS="-D warnings" cargo doc -p runnel-scheduler --no-deps --locked --offline
```

`SchedulerEngine::<A>::required_batch_admission_reserve_bytes` reports the
adapter-typed semantic Vec-payload requirement. It intentionally excludes
allocator metadata and over-allocation; whole-process RSS remains an observed
quantity. A smaller raw reserve is rejected before shared engine allocation.

## M5 genuine actor-race verification

The feature-gated scheduler race runs exactly 32 fresh two-producer
repetitions and publishes one bounded capture into a private tmpfs directory.
The authoritative verifier consumes that Rust capture in one direction: Rust
does not receive a Python digest, transcript, accepted set, or other derived
artifact. Run the CI gate exactly with:

```console
race_dir="$(mktemp -d /dev/shm/runnel-actor-race.XXXXXX)"
test "$(stat -c %a "$race_dir")" = 700
RUNNEL_ACTOR_RACE_CAPTURE="$race_dir/capture.json" \
  cargo test -p runnel-scheduler --test actor_script \
    --no-default-features --features actor-stress-instrumentation \
    race::tests::genuine_actor_race_capture_runs_exactly_32_fresh_repetitions \
    --locked -- --exact
test "$(stat -c %a "$race_dir/capture.json")" = 600
python -m oracle.actor_race_verify "$race_dir/capture.json"
```

The default Python command enables the mandatory independent PyTorch model
gate and is the only publishable mode. `--no-model` validates structural
history for diagnosis but reports `NONPUBLISHABLE` and cannot support model
parity or token-prefix claims. The command above is a correctness gate, not an
accepted timing result or performance claim.
