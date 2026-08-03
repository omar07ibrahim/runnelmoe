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
