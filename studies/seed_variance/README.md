# MoE seed-variance studies

This builds the paired browser for the 60M and 125M `reference_moe` seed
cohorts. The full-resolution artifacts live in the ignored Hugging Face staging
tree; the generator and its test are tracked here so the report is reproducible.

The clean cohorts are:

- `seed-variance-60M`: 41/64 planned seeds, 1337–1368 and 1370–1378, v6e-8
- `seed-variance-125M`: 22/64 planned seeds, 1337–1358, v4-32

The 60M seed-1369 run is excluded because it used a different, dirty
`train.py`. Both retained cohorts use recipe SHA256
`9b551600d10cb4c19f2d3e7778b600140a548157018b577eb7e26d8b2a50c65d`.
All of these logs predate the MoE AdamW bias-mask fix in commit
`102a264672c8453700a02e321495a14c585e58ea`; see the individual study cards for
the compatibility boundary and current reproduction commands.

## Build

Export each cohort from the local run store. The two named target directories
must not already exist:

```bash
rig report --runs runs --select 60m-moe-seedvar \
  --study-export-target hf-dataset --study-name seed-variance-60M

rig report --runs runs --select 125m-moe-seedvar \
  --study-export-target hf-dataset --study-name seed-variance-125M
```

Fill the generated study `README.md` files, then build the self-contained
browser:

```bash
python -m studies.seed_variance.report \
  --study-60m hf-dataset/seed-variance-60M \
  --study-125m hf-dataset/seed-variance-125M \
  --max-points 1440 --output hf-dataset/seed-variance.html
```

For each metric, the builder verifies identical columns, steps, token
accounting, and FLOP accounting within a cohort. It computes the finite-only
sample standard deviation at every exact step before LTTB thinning. The HTML
contains 429 selectable series: 141 training metrics and 288 long-form
diagnostics, shown as a synchronized 60M/125M pair.

Training-loss and gradient SD include both initialization and shuffled-batch
variation because the seed controls both. Fixed-set validation is the cleaner
measure of endpoint model variance; the per-step curves answer the narrower
question of how far the recorded training signals separate.
