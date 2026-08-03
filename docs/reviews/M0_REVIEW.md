# M0 independent review record

- Baseline commit: `fe2fd0ce723d84035af16fd276e1c18c983c36ab`
- Review completed: 2026-08-03
- Repository: [omar07ibrahim/runnelmoe](https://github.com/omar07ibrahim/runnelmoe)
- Baseline CI:
  [run 30788257437](https://github.com/omar07ibrahim/runnelmoe/actions/runs/30788257437)
  — passed `repository-contract`
- Scope: 35 publication candidates in the baseline commit

## Independent passes

Bounded reviewers did not edit the candidate:

| Review | Final result | Scope |
| --- | --- | --- |
| Architecture | pass | milestone gates, adapter/runtime boundaries, parser cases, host-realistic sequencing |
| Naming | pass | same-niche collision screen, owner/package/web availability, final-name recommendation |
| Prior art | pass | exact upstream pin/license, primary Kimi sources, independent requirements, reuse risks |
| M0 acceptance | pass | required documents, local links, interface/format/error/non-goal contract, CI pins |
| Security and format | pass | RMOA integrity, canonicalization, TOCTOU, publication, budgets, cancellation, CI supply chain |
| Clean-room and claims | pass | source record, phrase/layout comparison, licensing, capability language, private-data exclusion |

The clean-room comparison fetched only the exact documentation/configuration
paths listed in `docs/PRIOR_ART.md`. It found no matching block of nine or
more words in the public prose, no reused implementation or fixture content,
and materially distinct architecture and object-format choices.

## Findings resolved before publication

- Changed the final name from StrataMoE to RunnelMoE after a same-niche
  collision was found.
- Replaced capability language with target/contract language while no runtime
  exists.
- Made RMOA page verification scalable with digest-bound external page tables,
  page-only reads, RFC 8785 safe-integer canonical JSON, precise schema limits,
  retained verified tables, and finite operational defaults.
- Specified directory-FD traversal, no-follow and no-replace publication,
  crash/cancellation linearization, orphan accounting and garbage collection.
- Expanded memory admission to charge capacity, alignment, page-table and
  policy metadata, leases/waiters, scratch, state, and bounded channels before
  allocation or I/O submission.
- Hardened the offline verifier against secret-like files and unpinned GitHub
  Actions/containers, including restricted-YAML regression cases.
- Split design-study sources from audit-only inputs and corrected Kimi
  quantization attribution.

## Reproduction

From the baseline commit:

    python3 -m py_compile scripts/verify_repository.py
    python3 scripts/verify_repository.py
    git show --check --format=oneline fe2fd0ce723d84035af16fd276e1c18c983c36ab

The first command validates syntax, the second performs the offline 35-file
repository contract, and GitHub repeated the contract on a clean checkout.
M0 has no performance benchmark because it contains no executable runtime and
makes no performance claim.
