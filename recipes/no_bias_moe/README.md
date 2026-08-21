# no_bias_moe — bias-free routed ablation

A controlled fork of [`reference_moe`](../reference_moe/) that removes every
additive bias from the Transformer while leaving its architecture, routing,
parameterization, optimizer, data contract, and evaluation protocol unchanged.
It exists as a separate recipe so an exploratory architecture choice never
silently rewrites the routed reference.

The removed parameters are:

- QKV projection biases;
- attention-output projection biases;
- routed expert up-projection biases; and
- routed expert down-projection biases.

The dense fallback in the training script is bias-free as well. The router,
embeddings, unembedding, and RMSNorm already had no additive bias; RMSNorm
retains its learned scale. The initialized parameter tree and AdamW policy both
fail tests if an additive bias is reintroduced.

## Comparison contract

The three standalone YAML documents are copies of `reference_moe`'s scientific
configuration. In particular, this recipe defaults to the `8k` context, uses
batch 16 and learning rate `2^-8`, and the development profile runs for 5 TPP.
The inherited dense-equivalent tier denominator is intentionally unchanged, so
paired runs process the same number of tokens and take the same number of
optimizer steps. Bias additions are elementwise operations outside the
headline algorithmic-FLOP count, so the two arms also retain the same headline
training budget.

Fresh `reference_moe` controls are required for a clean comparison. The
archived 60M and 125M MoE studies predate commit
`102a264672c8453700a02e321495a14c585e58ea`, when stacked expert biases were
incorrectly included in AdamW weight decay. Those results remain useful
historical measurements, but pairing them with a model that has no expert
biases would confound the architecture ablation with the optimizer fix.

Use the harness rather than invoking the trainer directly:

```bash
uv run --frozen --no-sync rig run no_bias_moe \
  --cluster v4-32 --profile dev --tier 60m \
  --tokens-per-parameter 5 --base-learning-rate 0.00390625 \
  --seed 1337 --checkpoint-policy none \
  --name 60m-no-bias-moe-5tpp-bs16-lr2e-8-s1337
```

For a short-context diagnostic, add `--context 1k`; that selects the aligned
1,024-token preset and its batch-128 anchor in both routed recipes.
