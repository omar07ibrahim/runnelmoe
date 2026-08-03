# Prior art and provenance

This document records what was consulted, at which revision, and what did not
cross the clean-room boundary. Links are for attribution; no third-party
implementation artifact is vendored or reproduced.

## Study record

### Kimi K3 in C

- Project: [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)
- Revision studied: [`85ab2cd901aa81b70caac7711f06864d594b8ff3`](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/85ab2cd901aa81b70caac7711f06864d594b8ff3)
- Revision date: 2026-08-01
- Pin checked: 2026-08-03; upstream `main` still resolved to the pin
- Upstream license: [Apache License 2.0](https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/85ab2cd901aa81b70caac7711f06864d594b8ff3/LICENSE)
- Pinned tree: `a55c1c452d5ce1cbae02cd7f8198e4c124737bd1`
- Design-study content: pinned `LICENSE`, `NOTICE`, `README.md`,
  `docs/ARCHITECTURE.md`, `docs/BENCHMARKING.md`,
  `docs/PERFORMANCE.md`, and `docs/ROADMAP.md`.
- Metadata reviewed: repository and commit objects plus the recursive tree
  (paths, object types, and sizes), including test path metadata. No test
  fixture or implementation file content was imported.

An independent contamination audit fetched the following pinned text files
solely to compare them with the publication candidate for accidental phrase or
layout overlap:

    .editorconfig
    .github/ISSUE_TEMPLATE/config.yml
    .github/PULL_REQUEST_TEMPLATE.md
    .github/workflows/ci.yml
    .gitignore
    CONTRIBUTING.md
    LICENSE
    NOTICE
    README.md
    SECURITY.md
    docs/API.md
    docs/ARCHITECTURE.md
    docs/BENCHMARKING.md
    docs/PERFORMANCE.md
    docs/QUICKSTART.md
    docs/README.md
    docs/ROADMAP.md
    docs/TESTING.md
    docs/TUNING.md

Files not already named as design-study content were audit-only inputs. No
upstream implementation source or test-fixture content was fetched by that
audit, and no fetched text was added to this repository.

The project is credited for making constrained-memory Kimi K3 inference a
concrete systems problem and for surfacing the categories an independent
runtime must investigate: numerical reference paths, compact expert storage,
memory placement, storage traffic, caching, and reproducible evaluation.
Those are problem requirements, not imported designs. RunnelMoE uses different
language/runtime boundaries, an original object format and cache state model,
synthetic model fixtures, independently selected interfaces, and its own
measurements.

No upstream source code, prose, equations as rendered there, fixtures, tests,
scripts, images, diagrams, visual identity, distinctive directory layout, CLI
design, benchmark claim, or generated artifact has been copied or adapted.
Shared governance/documentation paths are conventional or mission-required.
The repository was not forked or cloned.

### Moonshot AI Kimi K3

- Publisher repository:
  [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- Publisher revision recorded:
  [`7c5be9599120d7993748de66a76128614f15f210`](https://github.com/MoonshotAI/Kimi-K3/tree/7c5be9599120d7993748de66a76128614f15f210)
- Technical report:
  [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- Repository and weight terms:
  [Kimi K3 License](https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/LICENSE)

The publisher's report is the primary source for Kimi K3 architecture facts.
It describes a sparse MoE with Kimi Delta Attention, Stable LatentMoE,
attention residuals, many experts with a small active subset, very long
context, and native compact numeric formats. These facts motivate adapter
extensibility, sparse data movement, state budgeting, and compact-kernel
research. The report applies MXFP4 to routed MoE expert weights, not every
model tensor. The pinned released configuration explicitly excludes attention,
shared-expert and dense-MLP projections, the LM head, vision tower, and
multimodal projector from its compression rule. These facts do not establish
support in RunnelMoE. The project does not redistribute or acquire Kimi
weights, and no Kimi license applies to this repository's original code.

## Primary specifications recorded for future adapters

- [Kimi K3 report v1](https://arxiv.org/abs/2607.24653v1), DOI
  `10.48550/arXiv.2607.24653` — CC BY-NC-ND 4.0; citation only.
- [Pinned released configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json)
  — associated weights use the custom Kimi K3 License; metadata only was
  inspected.
- [Kimi Linear report v2](https://arxiv.org/abs/2510.26692v2) — primary KDA
  equations, CC BY-NC-ND 4.0; citation only.
- [Attention Residuals report v1](https://arxiv.org/abs/2603.15031v1) — primary
  residual architecture source, CC BY-NC-ND 4.0; citation only.
- [Safetensors specification at `6eb4dc9a…`](https://github.com/huggingface/safetensors/tree/6eb4dc9a28ebce297606e0f4836bbf28839cacef)
  — Apache-2.0; a possible future import boundary, with no reused code.
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  — Open Web Foundation Final Specification Agreement 1.0; primary source for
  any future MX-format work.
- [RFC 8785: JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
  — normative RMOA manifest serialization reference, published under the
  [IETF Trust Legal Provisions](https://trustee.ietf.org/documents/trust-legal-provisions/).

Policy papers used in later milestones will be added at the commit that first
relies on them, with version, link, and license where applicable. Results from
another implementation are never accepted as RunnelMoE measurements.

## Contribution rule

Before introducing any third-party implementation material, stop and amend
this record with exact files, revision, license obligations, notices,
modifications, and reviewer approval. Merely compatible formats and primary
published specifications should be preferred.
