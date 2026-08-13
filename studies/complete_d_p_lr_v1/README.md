# Complete(d)P learning-rate transfer v1

This is the first, deliberately one-dimensional family study. It holds the
global batch at 128 sequences, trains each 60M–500M tier for approximately five
tokens per parameter (0.25 of the 20-TPP ladder), and sweeps seven normalized
base learning rates separated by sqrt(2). The 60M, 125M, and 250M curves are the
candidate-admission range; 500M is a larger reproduction point. The 1B tier is
implemented by the family but is not part of this compute-limited first sweep.

There is one seed in the coarse pass. An apparent shared optimum should be
confirmed around the minimum with at least three seeds before treating it as a
transfer result. A minimum on either grid edge is not a successful bracket and
requires extending that edge.

The runner creates the results CSV before launching its first TPU job and
updates it atomically after every accepted run. It is resumable and does not
render a chart:

```bash
make sweep-lr
```

Live results are written beneath the gitignored `runs/studies/` tree. Chart
semantics and HTML rendering are intentionally deferred.
