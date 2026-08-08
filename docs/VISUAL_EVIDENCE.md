# Candidate visual-evidence contract

This contract covers portfolio visuals for the accepted M1-M4 repository state.
It does not cover the unfinished M5 scheduler, and it creates no end-to-end
inference, serving, or storage-performance claim.

Generated assets are candidates until a maintainer independently reviews the
hosted artifact and explicitly approves adoption. The capture job never commits
or pushes generated files.

## Source captures

The hosted job runs these four documented commands at the exact checked-out
commit, with locked Cargo dependencies and offline execution after the build:

```console
cargo run --locked -p runnel -- demo --prompt moe --max-new-tokens 4 --json
cargo run --locked -p runnel -- data-plane-demo --json
cargo run --locked -p runnel-sim --bin runnel-cache-sim -- \
  matrix --family markov_clusters --replicate 0 --measured-steps 64
cargo run --locked -p runnel --bin runnel-m4-model-check
```

Standard output and standard error are retained separately as bounded raw
files. The renderer parses strict JSON or JSONL directly from those bytes.
Raw output is authoritative; a visual is never accepted as replacement
evidence.

The M3 command is the documented 64-step deterministic functional smoke test,
not the accepted 30-seed experiment and not a timing benchmark. The M4 command
is a correctness ledger and emits no timing. Volatile M2 wait, I/O, and RSS
fields remain available in raw stdout but are not visualized as performance
evidence.

## Candidate artifact

One immutable Actions artifact contains:

- eight raw stream files, one stdout/stderr pair per command;
- four source-backed result SVGs;
- one architecture SVG derived from committed Cargo manifests;
- one PNG rendering of the actual M1 stdout transcript;
- one four-frame GIF that presents the four captured commands in order;
- a human-readable candidate README;
- a closed JSON manifest and SHA-256 inventory.

The transcript PNG is captioned as a deterministic rendering of captured
stdout, not an operating-system screenshot. The GIF is captioned as a
deterministic four-frame derivation, not a screen recording. Neither asset
contains invented terminal output.

The architecture figure uses only the actual M1-M4 workspace crates and their
internal manifest dependencies. It deliberately excludes the scheduler node
in `docs/diagrams/runtime.dot`, because that node is not accepted M1-M4
implementation evidence.

## Provenance and reproducibility

The manifest records:

- repository, full commit SHA, Git tree SHA, clean-checkout status, and commit
  timestamp;
- GitHub run ID and attempt;
- exact argument vectors, a stable logical working directory, an allowlisted
  environment, exit status, and SHA-256/size for each raw stream;
- exact Python, Pillow, Cargo, and Rust compiler versions plus executable or
  lock-file digests;
- every Cargo manifest used for the architecture graph;
- source capture IDs and digests for every derived asset;
- canvas, content bounds, frame count, file size, and SHA-256 for every file.

The renderer is standard-library Python except for Pillow. The workflow pins
Python to an exact patch release and installs one exact Linux x86-64 Pillow
wheel by URL and SHA-256. Every external GitHub Action is pinned to a full
40-character commit SHA. No paid service, mutable action tag, package resolver,
browser, screen recorder, or unpinned system font is part of rendering.

## Bounds, integrity, and security

Capture subprocesses run without a shell, with a wall deadline and independent
one-MiB stdout/stderr limits. The artifact is created outside the checkout in a
mode-0700 directory; files are mode 0600. Capture must leave the checkout clean.

Verification rejects:

- duplicate JSON keys, non-finite values, malformed or incomplete schemas, and
  changed deterministic command results;
- an unlisted, missing, symbolic-link, oversized, or digest-mismatched file;
- private host paths, credential-like byte prefixes, private-key markers, or
  NUL bytes in any textual source;
- SVG scripts, foreign objects, images, external references, event handlers,
  data URLs, masks, or clipping;
- canvases beyond 1,600 by 2,000 pixels, content outside a 24-pixel safe frame,
  a non-full-frame GIF, an unexpected frame count, or raster content touching
  the outer border;
- missing PNG/GIF provenance metadata or a digest join that does not match the
  raw source.

The artifact-level SHA-256 reported by `actions/upload-artifact` is the final
hosted custody value. No candidate visual is linked from project documentation
or committed as an adopted asset before independent review and explicit
approval.
