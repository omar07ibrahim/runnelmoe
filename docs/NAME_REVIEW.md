# Project-name decision

- Decision date: 2026-08-03
- Final name: **RunnelMoE**
- Repository slug: `runnelmoe`
- Status: preliminary availability screen, not legal advice or trademark
  clearance

## Decision

The working name StrataMoE was rejected before publication. Although the exact
`omar07ibrahim/StrataMoE` owner slot was free, a newly public
[StrataMoE Lab at `96c3d50a…`](https://github.com/Labeeb2339/stratamoe-lab/tree/96c3d50acd7265e056dd04c1283e426bc129d753)
(Apache-2.0) already describes the same GPU/RAM/NVMe MoE placement niche. The
adjacent [Strata paper v1](https://arxiv.org/abs/2508.18572v1)
(CC BY 4.0) and
several unrelated software businesses using “Strata” make mistaken association
plausible.

RunnelMoE was selected because a runnel is a bounded channel for flow, matching
the runtime's controlled movement of expert pages through memory tiers.

## Screen performed

On 2026-08-03:

- exact GitHub owner and global repository searches found no RunnelMoE project;
- exact-name web searches found no relevant software product or research
  system;
- exact-name searches of public package indexes for Cargo, PyPI, and npm found
  no package; and
- an exact-name preliminary trademark web search returned no relevant result.

Point-in-time commands used for the screen:

    gh api repos/omar07ibrahim/runnelmoe
    gh search repos RunnelMoE --match name --limit 100
    curl -A "RunnelMoE-name-screen/1.0" -fSs "https://crates.io/api/v1/crates?q=runnelmoe&per_page=10"
    curl -fSs https://pypi.org/pypi/RunnelMoE/json
    curl -fSs https://registry.npmjs.org/runnelmoe

At the decision time, before this project's public repository was created, the
owner, PyPI, and npm lookups returned not-found; the crates.io and global
GitHub searches returned no results. Search endpoints can change behavior and
will now find this project. Exact web
queries for `"RunnelMoE"`, `"Runnel MoE"`, and
`"RunnelMoE" trademark` produced no relevant software or mark.

The word “runnel” itself is ordinary language and is used in unrelated
contexts. This screen establishes practical portfolio distinctness only.
Before commercial branding, distribution in additional jurisdictions, or
substantial marketing spend, a qualified trademark search is still required.
