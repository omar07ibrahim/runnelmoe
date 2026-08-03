# Security policy

## Supported versions

RunnelMoE is pre-release software. Security fixes are applied to the latest
commit on `main`; no older version is currently supported.

## Report privately

Do not open a public issue for a suspected vulnerability. Use
[GitHub's private vulnerability report](https://github.com/omar07ibrahim/runnelmoe/security/advisories/new)
and include the affected commit, impact, minimal reproduction, and any proposed
mitigation. Do not include private model data, prompts, tokens, or host
credentials.

The maintainer will acknowledge a complete report when available, validate it,
coordinate a fix and disclosure, and credit the reporter if requested.
No response-time or bounty promise is made.

## Scope

High-value reports include artifact parser/verification bypasses, path
traversal, memory-safety errors, budget bypasses, cross-request state leaks,
unintended non-loopback exposure, cancellation races, and prompt/token leakage.
Third-party service availability, the safety/content behavior of model weights,
and attacks requiring an already privileged local administrator are outside
the project's current security boundary.

See [the threat model](docs/THREAT_MODEL.md) for explicit assumptions.
