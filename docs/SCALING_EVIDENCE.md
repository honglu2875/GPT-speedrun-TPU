# Publishing v4 scaling evidence

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

An extra file, an empty extra directory, any symbolic/hard link, a missing
selection, or a changed source during copying fails the build. Files are copied
through one read-and-hash stream; they are never hard-linked into the archive.
The archive also contains a deterministic `git archive` of the launch commit and
a standalone `verify.py`.

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

The output directory must not already exist. This prevents an earlier archive
from being silently mixed with a new build.

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
folder commit beneath a content-derived archive directory. The publisher then
uses an anonymous client and unauthenticated streaming downloads at that exact
immutable commit to:

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
