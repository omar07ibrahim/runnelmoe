#!/usr/bin/env python3
"""Offline repository-contract checks for the design and release gates."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ONLY = {"AGENTS.md", "MISSION.md", "INITIAL_PROMPT.txt"}
SKIP_PARTS = {
    ".codex",
    ".git",
    ".runnelmoe",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
    "tmp",
}
MAX_FILE_BYTES = 5 * 1024 * 1024

REQUIRED = {
    ".editorconfig",
    ".github/CODEOWNERS",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/experiment.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/ci.yml",
    ".gitignore",
    "Cargo.lock",
    "Cargo.toml",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "benchmarks/README.md",
    "crates/README.md",
    "crates/runnel-cli/Cargo.toml",
    "crates/runnel-cli/src/main.rs",
    "crates/runnel-cli/tests/cli.rs",
    "crates/runnel-fixture/Cargo.toml",
    "crates/runnel-fixture/src/lib.rs",
    "crates/runnel-format/Cargo.toml",
    "crates/runnel-format/README.md",
    "crates/runnel-format/src/artifact.rs",
    "crates/runnel-format/src/error.rs",
    "crates/runnel-format/src/json.rs",
    "crates/runnel-format/src/lib.rs",
    "crates/runnel-format/src/manifest.rs",
    "crates/runnel-format/tests/corruption.rs",
    "crates/runnel-kernels/Cargo.toml",
    "crates/runnel-kernels/README.md",
    "crates/runnel-kernels/build.rs",
    "crates/runnel-kernels/include/runnel_kernels.h",
    "crates/runnel-kernels/native/bf16_gemv.c",
    "crates/runnel-kernels/src/lib.rs",
    "crates/runnel-kernels/tests/bf16.rs",
    "crates/runnel-kernels/tests/c/abi_sanitizer.c",
    "crates/runnel-kernels/tests/gemv.rs",
    "crates/runnel-runtime/Cargo.toml",
    "crates/runnel-runtime/src/error.rs",
    "crates/runnel-runtime/src/lib.rs",
    "crates/runnel-runtime/src/model.rs",
    "crates/runnel-runtime/src/tensor.rs",
    "crates/runnel-runtime/src/tokenizer.rs",
    "crates/runnel-runtime/tests/oracle_parity.rs",
    "crates/runnel-sim/Cargo.toml",
    "crates/runnel-sim/README.md",
    "crates/runnel-sim/src/engine.rs",
    "crates/runnel-sim/src/error.rs",
    "crates/runnel-sim/src/generator.rs",
    "crates/runnel-sim/src/lib.rs",
    "crates/runnel-sim/src/main.rs",
    "crates/runnel-sim/src/metrics.rs",
    "crates/runnel-sim/src/model.rs",
    "crates/runnel-sim/src/oracle/mod.rs",
    "crates/runnel-sim/src/policy/mod.rs",
    "crates/runnel-sim/src/trace.rs",
    "crates/runnel-sim/tests/cli.rs",
    "crates/runnel-sim/tests/generator.rs",
    "crates/runnel-sim/tests/golden_traces.rs",
    "crates/runnel-sim/tests/oracles.rs",
    "crates/runnel-sim/tests/parser_fuzz_smoke.rs",
    "crates/runnel-sim/tests/policies.rs",
    "crates/runnel-sim/tests/python_differential.rs",
    "crates/runnel-sim/tests/store_crosscheck.rs",
    "crates/runnel-store/Cargo.toml",
    "crates/runnel-store/README.md",
    "crates/runnel-store/src/async_io.rs",
    "crates/runnel-store/src/budget.rs",
    "crates/runnel-store/src/cache.rs",
    "crates/runnel-store/src/cas.rs",
    "crates/runnel-store/src/control.rs",
    "crates/runnel-store/src/error.rs",
    "crates/runnel-store/src/fs.rs",
    "crates/runnel-store/src/layout.rs",
    "crates/runnel-store/src/lib.rs",
    "crates/runnel-store/src/metrics.rs",
    "crates/runnel-store/src/page.rs",
    "crates/runnel-store/src/source.rs",
    "crates/runnel-store/src/stage.rs",
    "crates/runnel-store/src/trace.rs",
    "crates/runnel-store/tests/async_reader.rs",
    "crates/runnel-store/tests/cache_state.rs",
    "crates/runnel-store/tests/cas_budget.rs",
    "crates/runnel-store/tests/cas_gc.rs",
    "crates/runnel-store/tests/cas_import.rs",
    "crates/runnel-store/tests/runtime_parity.rs",
    "crates/runnel-store/tests/secure_filesystem.rs",
    "crates/runnel-store/tests/secure_reads.rs",
    "SECURITY.md",
    "docs/BENCHMARKING.md",
    "docs/CLAIMS.md",
    "docs/DESIGN.md",
    "docs/DEVELOPMENT.md",
    "docs/FORMAT.md",
    "docs/NAME_REVIEW.md",
    "docs/PRIOR_ART.md",
    "docs/ROADMAP.md",
    "docs/THREAT_MODEL.md",
    "docs/adr/0001-clean-room-and-system-boundaries.md",
    "docs/adr/0002-immutable-tensor-objects.md",
    "docs/adr/0003-tiny-reference-runtime.md",
    "docs/adr/0004-verified-data-plane.md",
    "docs/adr/0005-cache-policy-research.md",
    "docs/adr/0006-bf16-avx2-expert-kernel.md",
    "docs/diagrams/runtime.dot",
    "docs/reviews/M0_REVIEW.md",
    "docs/reviews/M1_REVIEW.md",
    "docs/reviews/M2_REVIEW.md",
    "docs/reviews/M3_REVIEW.md",
    "fixtures/README.md",
    "fixtures/cache/README.md",
    "fixtures/cache/m2-forced-eviction.jsonl",
    "fixtures/cache/router-perfect.jsonl",
    "fixtures/cache/variable-byte.jsonl",
    "fixtures/tiny/README.md",
    "fixtures/tiny/golden_logits.json",
    "fixtures/tiny/golden_metadata.json",
    "fixtures/tiny/golden_routes.json",
    "fixtures/tiny/golden_tokens.json",
    "fixtures/tiny/spec.json",
    "fixtures/tiny-v2/README.md",
    "fixtures/tiny-v2/golden_logits.json",
    "fixtures/tiny-v2/golden_metadata.json",
    "fixtures/tiny-v2/golden_routes.json",
    "fixtures/tiny-v2/golden_tokens.json",
    "fixtures/tiny-v2/spec.json",
    "oracle/README.md",
    "oracle/__init__.py",
    "oracle/cache_policy.py",
    "oracle/generate.py",
    "oracle/requirements.txt",
    "oracle/runnel_oracle/__init__.py",
    "oracle/runnel_oracle/model.py",
    "oracle/runnel_oracle/spec.py",
    "oracle/runnel_oracle/tokenizer.py",
    "oracle/tests/__init__.py",
    "oracle/tests/test_cache_policy.py",
    "oracle/tests/test_oracle.py",
    "rust-toolchain.toml",
    "scripts/verify_repository.py",
    "scripts/run_m2_experiment.py",
    "scripts/run_m3_experiment.py",
    "scripts/tests/test_run_m2_experiment.py",
    "scripts/tests/test_run_m3_experiment.py",
    "benchmarks/raw/m2-data-plane-20260803/environment.json",
    "benchmarks/raw/m2-data-plane-20260803/experiment.json",
    "benchmarks/raw/m2-data-plane-20260803/observations.jsonl",
    "benchmarks/raw/m2-data-plane-20260803/summary.json",
    "benchmarks/raw/m2-data-plane-forced-eviction-20260803/environment.json",
    "benchmarks/raw/m2-data-plane-forced-eviction-20260803/experiment.json",
    "benchmarks/raw/m2-data-plane-forced-eviction-20260803/observations.jsonl",
    "benchmarks/raw/m2-data-plane-forced-eviction-20260803/summary.json",
    "benchmarks/raw/m3-cache-policies-20260803/environment.json",
    "benchmarks/raw/m3-cache-policies-20260803/experiment.json",
    "benchmarks/raw/m3-cache-policies-20260803/figures/optimal-gap.svg",
    "benchmarks/raw/m3-cache-policies-20260803/figures/paired-change.svg",
    "benchmarks/raw/m3-cache-policies-20260803/figures/prefetch-accounting.svg",
    "benchmarks/raw/m3-cache-policies-20260803/observations.jsonl",
    "benchmarks/raw/m3-cache-policies-20260803/summary.json",
    "benchmarks/raw/m3-cache-policies-20260803/traces.jsonl",
}

FORBIDDEN_BYTES = {
    b"/home/" + b"ubuntu/": "private host path",
    b"/Us" + b"ers/": "private macOS host path",
    b"-----BEGIN " + b"PRIVATE KEY-----": "private key",
    b"-----BEGIN OPEN" + b"SSH PRIVATE KEY-----": "OpenSSH private key",
    b"-----BEGIN RSA " + b"PRIVATE KEY-----": "RSA private key",
    b"-----BEGIN EC " + b"PRIVATE KEY-----": "EC private key",
    b"-----BEGIN DSA " + b"PRIVATE KEY-----": "DSA private key",
    b"gh" + b"o_": "GitHub token prefix",
    b"gh" + b"p_": "GitHub token prefix",
    b"gh" + b"u_": "GitHub token prefix",
    b"gh" + b"s_": "GitHub token prefix",
    b"gh" + b"r_": "GitHub token prefix",
    b"github_" + b"pat_": "GitHub token prefix",
    b"AK" + b"IA": "AWS access-key prefix",
    b"AS" + b"IA": "AWS temporary access-key prefix",
}

FORBIDDEN_BASENAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    "credentials",
    "hosts.yml",
    "id_ed25519",
    "id_rsa",
}

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
ACTION_USE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*['\"]?([^'\"#\s]+)['\"]?\s*(?:#.*)?$",
    re.MULTILINE,
)
USES_KEY_CANDIDATE = re.compile(
    r"(?:^|[\s{,\-])(?:uses|['\"]uses['\"])\s*:"
)
EXPLICIT_MAPPING_KEY = re.compile(r"^\s*(?:-\s*)?\?(?:\s|$)")
QUOTED_MAPPING_KEY = re.compile(
    r"(?:^|[\s{,])['\"][^'\"\n]+['\"]\s*:"
)
ALIAS_MAPPING_KEY = re.compile(
    r"(?:^|[\s{,])\*[A-Za-z0-9_-]+\s*:"
)
REMOTE_ACTION = re.compile(
    r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+@[0-9a-f]{40}$"
)
PINNED_CONTAINER = re.compile(
    r"^docker://[A-Za-z0-9_.:/-]+@sha256:[0-9a-f]{64}$"
)


def repository_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if relative.name in WORKSPACE_ONLY and len(relative.parts) == 1:
            continue
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        files.append(relative)
    return sorted(files)


def check_local_links(path: Path, text: str, failures: list[str]) -> None:
    for match in MARKDOWN_LINK.finditer(text):
        destination = match.group(1).strip().strip("<>")
        if not destination or destination.startswith(("http://", "https://", "mailto:")):
            continue
        destination = destination.split("#", 1)[0]
        if not destination:
            continue
        target = (path.parent / destination).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            failures.append(f"{path}: local link escapes repository: {destination}")
            continue
        if not target.exists():
            failures.append(f"{path}: broken local link: {destination}")


def valid_action_reference(reference: str) -> bool:
    return (
        reference.startswith("./")
        or REMOTE_ACTION.fullmatch(reference) is not None
        or PINNED_CONTAINER.fullmatch(reference) is not None
    )


def action_references(
    text: str, source: str, failures: list[str]
) -> list[str]:
    references: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        if (
            EXPLICIT_MAPPING_KEY.search(line)
            or QUOTED_MAPPING_KEY.search(line)
            or ALIAS_MAPPING_KEY.search(line)
        ):
            failures.append(
                f"{source}:{line_number}: noncanonical YAML mapping key is forbidden"
            )
            continue
        if not USES_KEY_CANDIDATE.search(line):
            continue
        match = ACTION_USE.fullmatch(line)
        if match is None:
            failures.append(
                f"{source}:{line_number}: noncanonical uses mapping is forbidden"
            )
            continue
        references.append(match.group(1))
    return references


def check_action_parser(failures: list[str]) -> None:
    sha = "a" * 40
    digest = "b" * 64
    valid = {
        f"uses: actions/checkout@{sha}": f"actions/checkout@{sha}",
        f"- uses: 'github/codeql-action/init@{sha}'": (
            f"github/codeql-action/init@{sha}"
        ),
        'uses: "./.github/actions/local"': "./.github/actions/local",
        f"uses: docker://alpine@sha256:{digest}": (
            f"docker://alpine@sha256:{digest}"
        ),
    }
    invalid = {
        "uses: actions/checkout@v7",
        "uses: actions/checkout",
        "uses: docker://alpine:3.20",
        "uses: docker://alpine@sha256:abcd",
    }
    bypasses = {
        f"uses : actions/checkout@{sha}",
        f'"uses": actions/checkout@{sha}',
        f"- {{uses: actions/checkout@{sha}}}",
        f"'uses' : actions/checkout@{sha}",
        f"steps:\n  - ? uses\n    : actions/checkout@{sha}",
        f'  - "u\\u0073es": actions/checkout@{sha}',
        f"  - *action_key: actions/checkout@{sha}",
        (
            "action_key: &action_key uses\n"
            f"steps:\n  - {{*action_key: actions/checkout@{sha}}}"
        ),
    }
    for line, expected in valid.items():
        found = ACTION_USE.findall(line)
        if found != [expected] or not valid_action_reference(expected):
            failures.append(f"internal action-pin parser rejected valid case: {line}")
    for line in invalid:
        found = ACTION_USE.findall(line)
        if len(found) != 1:
            failures.append(f"internal action-pin parser missed invalid case: {line}")
        elif valid_action_reference(found[0]):
            failures.append(f"internal action-pin parser accepted invalid case: {line}")
    for line in bypasses:
        local_failures: list[str] = []
        action_references(line, "<action-parser-self-test>", local_failures)
        if not local_failures:
            failures.append(f"internal action-pin parser missed bypass case: {line}")


def main() -> int:
    failures: list[str] = []
    check_action_parser(failures)
    files = repository_files()
    present = {path.as_posix() for path in files}

    for required in sorted(REQUIRED - present):
        failures.append(f"missing required file: {required}")

    for relative in files:
        path = ROOT / relative
        if relative.name in FORBIDDEN_BASENAMES:
            failures.append(f"{relative}: forbidden credential-like filename")
        if path.is_symlink():
            failures.append(f"symbolic links are not allowed in source: {relative}")
            continue
        if (
            relative.parts[:2] == ("fixtures", "tiny")
            and any(part in {"objects", "page-tables"} for part in relative.parts)
        ):
            failures.append(
                f"{relative}: generated checkpoint bytes must not be committed"
            )
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(
                f"{relative}: {size} bytes exceeds {MAX_FILE_BYTES}-byte source limit"
            )
        data = path.read_bytes()
        for marker, label in FORBIDDEN_BYTES.items():
            if marker in data:
                failures.append(f"{relative}: contains {label}")
        if data and not data.endswith(b"\n"):
            failures.append(f"{relative}: missing final newline")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if relative.suffix == ".md":
            check_local_links(relative, text, failures)

    action_files = list((ROOT / ".github/workflows").glob("*.yml"))
    action_files += list((ROOT / ".github/workflows").glob("*.yaml"))
    action_files += list((ROOT / ".github/actions").glob("**/action.yml"))
    action_files += list((ROOT / ".github/actions").glob("**/action.yaml"))
    for action_file in action_files:
        workflow_text = action_file.read_text(encoding="utf-8")
        references = action_references(
            workflow_text, str(action_file.relative_to(ROOT)), failures
        )
        for reference in references:
            if not valid_action_reference(reference):
                failures.append(
                    f"{action_file.relative_to(ROOT)}: action is not pinned: {reference}"
                )

    prior_art = ROOT / "docs/PRIOR_ART.md"
    if prior_art.exists():
        text = prior_art.read_text(encoding="utf-8")
        pin = "85ab2cd901aa81b70caac7711f06864d594b8ff3"
        if pin not in text:
            failures.append("docs/PRIOR_ART.md: missing required prior-art pin")
        if "No upstream source code" not in text:
            failures.append("docs/PRIOR_ART.md: missing explicit no-code-reuse statement")

    claims = ROOT / "docs/CLAIMS.md"
    readme = ROOT / "README.md"
    m0_design_only = readme.exists() and "design baseline (M0)" in readme.read_text(
        encoding="utf-8"
    )
    if (
        m0_design_only
        and claims.exists()
        and "| measured |" in claims.read_text(encoding="utf-8")
    ):
        failures.append("docs/CLAIMS.md: measured claim exists during design-only M0")

    citation = ROOT / "CITATION.cff"
    if citation.exists():
        citation_text = citation.read_text(encoding="utf-8")
        for key in ("cff-version:", "title:", "authors:", "repository-code:", "license:"):
            if key not in citation_text:
                failures.append(f"CITATION.cff: missing {key}")

    if failures:
        print("repository contract: FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"repository contract: ok ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
