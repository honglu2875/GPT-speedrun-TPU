# Publishing v4 scaling evidence

> V4 is now an immutable incomplete archive and has no scaling-law claim. This
> publisher remains pinned to its launch protocol; its closed-study release
> gate intentionally refuses the partial v4 run set. Do not use it for v5 or
> publish v5 evidence under a v4 prefix or receipt.

`scripts/publish_scaling_evidence.py` builds a closed, independently verifiable
archive for `current_budget_isoflop_v4`. It is intentionally specific to the
study launched from Git commit
`317ca1dae79cc75acf2ae48583b0392e8bb95114`; it is not a general run uploader.

Do not publish a partial study. The build requires a fitted scaling law and the
complete staged output: `fit.json`, `fit.md`, all three slice fits, all existing
per-slice learning-rate selections, and exactly these 15 files for every run:

1. `run-manifest.json`
2. `work/config.yaml`
3. `work/train.py`
4. `work/speedrun/__init__.py`
5. `work/speedrun/kernels/__init__.py`
6. `work/speedrun/kernels/autotune.py`
7. `work/speedrun/kernels/linear_cross_entropy.py`
8. `work/speedrun/kernels/pallas_linear_cross_entropy.py`
9. `work/speedrun/kernels/tpu_flash_attention.py`
10. `artifacts/training.csv`
11. `artifacts/validation.csv`
12. `artifacts/diagnostics.csv`
13. `artifacts/metrics.json`
14. `artifacts/result.json`
15. `artifacts/stability-admission.json`

An extra file, an empty extra directory, any interior symbolic/hard link, a
missing selection, or a changed source during copying fails the build. A
user-selected bundle or runs root may itself be a symlink. Standalone publish
resolves its bundle root once before credential access. Combined build+publish
resolves and recursively inventories its runs root once before building, then
requires that exact canonical path and every file/directory identity through
the pre-credential publication gate; retargeting the original alias cannot
select a second tree.
Files are copied through one read-and-hash stream; they are never hard-linked
into the archive.
The archive also contains a deterministic `git archive` of the launch commit and
a standalone `verify.py`. Every completed run must be part of the exact minimal
prospective staged state: the `c100_n124_control` is unconditional, each initial
LR grid is a contiguous low-to-high prefix through its first ineligible trial,
and geometric adaptation stops at the first valid bracket. A missing stable
initial suffix or an unnecessary post-bracket trial fails publication.

Each recomputed run is also bound to the exact checked-in public 4B manifest
(`99ac90a5...c2c14` raw; `92b21722...8dc1` canonical): both manifest hashes,
the prepared immutable revision, production identity, and all 40 shard paths,
sizes, token counts, and SHA-256s must agree.

## Local release gate

First build and verify without credentials or network access:

```bash
uv --cache-dir /tmp/uv-cache run --frozen --no-sync python scripts/publish_scaling_evidence.py build --runs runs/scaling/current-budget-isoflop-v4 --output /tmp/current_budget_isoflop_v4-evidence --dry-run
```

The dry run recomputes every run validation and admission from all CSV rows,
every learning-rate selection, every slice fit, the final fit, and rendered fit
Markdown from a private snapshot. It also requires `metrics.json` and
`result.json` to be byte-identical. It loads the scaling implementation from the
hash-pinned Git source tar, not from the current checkout. Run the bundled
verifier again with:

```bash
uv run --script /tmp/current_budget_isoflop_v4-evidence/verify.py verify --bundle /tmp/current_budget_isoflop_v4-evidence
```

The output directory must not already exist. Before creating even a missing
output parent, the builder rejects lexical or already-resolvable containment
and aliasing with either the runs tree or repository source. Final installation
is an atomic no-replace operation. These rules prevent staging artifacts from
polluting source evidence and prevent an earlier archive from being silently
mixed with a new build.

## Publication

Use a dedicated token file owned by the current user with mode `0600`:

```text
HF_TOKEN=hf_...
```

Then build to a new path and publish:

```bash
uv run --script scripts/publish_scaling_evidence.py build --runs runs/scaling/current-budget-isoflop-v4 --output /tmp/current_budget_isoflop_v4-release --token-file /absolute/private/path/hf-token --receipt-output data/manifests/scaling/current_budget_isoflop_v4.json
```

The token is read only after local verification succeeds. Upload uses a single
retained mode-read-only private snapshot and an explicit operation for each path
in its manifest-derived allowlist. Every upload object is opened once with
no-follow semantics, rehashed through that retained descriptor, kept open
through the commit, and checked for descriptor mutation afterward. It records
the current immutable parent commit and refuses a competing parent update. If
the content-derived remote
prefix already exists, publication becomes a safe resume: no overwrite is
attempted and the existing immutable prefix must pass the same complete remote
verification.

To resume from an already-built bundle without rebuilding or mixing content,
run:

```bash
uv run --script scripts/publish_scaling_evidence.py publish --bundle /tmp/current_budget_isoflop_v4-release --token-file /absolute/private/path/hf-token --receipt-output data/manifests/scaling/current_budget_isoflop_v4.json
```

The bundle is snapshotted once and fully revalidated before the token is opened.
The token path must have no linked component and must name one current-user,
single-link regular file of at most 4096 bytes with exact mode `0600`. Bundle,
evidence, token, and receipt paths must be disjoint and non-aliased. The receipt
writer refuses symlink components and never replaces an existing object. An
already-existing byte-identical current-user receipt is accepted as an
idempotent success; any different existing receipt is a hard failure and must
be reviewed and moved explicitly. Archive installation likewise uses an atomic
no-replace operation, so a concurrent creator cannot be clobbered.

After upload (or safe resume), the publisher uses an anonymous client and
unauthenticated, size-bounded streaming downloads at the exact 40-hex commit.
Anonymous repository resolution must return that same full OID; no branch,
fallback head, or mutable hex-looking reference is accepted. It then:

- require the exact closed remote tree;
- SHA-256 every remote object in full;
- verify the archive manifest;
- repeat all semantic recomputations on the downloaded copy.

Only after those checks does it write the external receipt. The receipt pins the
repository, immutable revision, archive directory, manifest hash, complete
per-file hash inventory, and every semantic verification flag. Commit that
receipt separately; it is deliberately outside the archive whose identity it
records.

The default public destination is
`quintic/gpt-tpu-speedrun-scaling-evidence/current_budget_isoflop_v4/...`.
Publication is deliberately public: the anonymous immutable-download gate
cannot authenticate a private archive.
