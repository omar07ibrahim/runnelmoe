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
| Python | CPython 3.12.3; oracle dependencies will be pinned with M1 |
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

## Baseline check

M0 needs only Python 3.12:

    python3 scripts/verify_repository.py

From M1, Rust is pinned by `rust-toolchain.toml`; the fixture generator pins
its Python dependencies in the oracle environment. Exact commands are added to
this file only after local and CI verification.
