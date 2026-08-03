# Contributing

RunnelMoE welcomes small, evidence-backed changes. Open an issue before a
large adapter, format, scheduler, kernel, or dependency change so its
acceptance gate can be agreed first.

## Clean-room rule

Do not submit code, prose, fixtures, tests, scripts, artwork, layouts, or
benchmark data copied or adapted from the Kimi K3 C prior-art repository.
Primary specifications and papers are preferred. Any necessary third-party
implementation material must be identified by exact file/revision and reviewed
for license and notice obligations before work begins. Update
`docs/PRIOR_ART.md` in the same change.

By contributing, you certify that you have the right to license your
contribution under Apache-2.0.

## Change contract

1. State the problem and falsifiable acceptance condition.
2. Keep the scalar/reference path independently executable.
3. Add negative and boundary tests, not only a success case.
4. For an optimization, follow `docs/BENCHMARKING.md`; submit raw results and
   numerical/token parity with the claim.
5. Update the threat model, ADRs, claim ledger, and public docs when a boundary
   or promise changes.
6. Keep fixtures tiny, deterministic, generated from open synthetic data, and
   free of secrets or third-party weights.

Run the repository checks documented in `docs/DEVELOPMENT.md`. Commits should
be focused and use an imperative summary. Pull requests must disclose
AI-assisted work where required by applicable policy and must never claim
tests or measurements that were not run.
