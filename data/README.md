# Data

No training corpus is stored in Git, and importing `rig.data` never
downloads or generates anything. Only explicit preparation—through the command
or its programmatic API—performs network access.

## Profiles

`smoke` is a tiny, deterministic synthetic stream with a 256-token vocabulary.
It is generated locally from a specified algorithm and seeds, so CPU tests work
without a tracked binary or an internet connection. Its manifest contains the
expected hashes of the generated shards.

`classic` (manifest name `fineweb10b-gpt2`) is GPT-2-tokenized FineWeb in the
same prebuilt shard convention used by llm.c and Modded-NanoGPT. The initial
selection contains:

- `fineweb_val_000000.bin`, of which the first 10,485,760 tokens form the fixed
  validation prefix;
- `fineweb_train_000001.bin` through `fineweb_train_000009.bin`, 100,000,000
  tokens each.

The prepared source repository is pinned to revision
`889765ea1f903759787add96995d81171b632d0c`. Every URL contains that revision,
and every file has the SHA-256 published as its Git LFS object ID at that exact
revision. Preparation checks the hash and the file format before making a shard
visible at its final filename.

## Choosing storage

The data root is always explicit. For example, on a machine where the
repository's `shm` symlink points at a large RAM filesystem:

```console
uv run --frozen --no-sync rig prepare --path shm/
```

The path is treated as the exact cache root; it is not silently namespaced and
is never purged. The preparer creates or replaces only shard filenames listed
by the selected manifest and their adjacent `.part` files. A normal disk path
works identically. The command checks that the destination is writable and has
enough currently known free space before transferring data.

Downloads stream into `<filename>.part`, support HTTP Range resumption, flush
data to storage, validate the 200 MB result, and atomically rename it. If a
server ignores a Range request the partial file is safely restarted. An invalid
existing final shard is not overwritten unless replacement is explicitly
requested.

For scripts and the competition harness, the equivalent programmatic API is:

```python
from rig.data import prepare

dataset = prepare(
    path="shm/",
    profile="classic",
    train_shards=9,
    offline=False,
    check_only=False,
    progress=lambda name, done, total: print(name, done, total),
)
```

Set `check_only=True` for strictly read-only verification. Set `offline=True`
to permit existing files and deterministic generation but reject missing
network shards. A user-supplied shard is supported by placing it under the
manifest filename and running verification; it is judged by header, length,
and hash exactly like a downloaded shard.

### Named scaled corpora

Corpus identity is selected directly: `classic`, `2B`, `4B`, `8B`, or `hero`.
For example, `rig prepare --dataset 8B --train-shards 79` saves and prepares
the complete 7.9B-token training prefix, while `rig dataset prepare 8B
--shards 20` prepares a smaller explicit prefix without changing saved run
settings. The published maxima are 9 classic train shards, 19 for `2B`, 39 for
`4B`, 79 for `8B`, and 749 for `hero`; every shard holds 100M tokens. Scaled
shards live under `<data-root>/fineweb-scaled/<variant>/`, so their filenames
cannot collide with classic data.

The routing layer trusts only a checked-in manifest under
`data/manifests/fineweb-scaled-gpt2/`. That manifest must pin the published
repository and commit, provide an immutable URL and SHA-256 for every exact
100M-token shard, and match the builder's source, tokenizer, temporal cutoff,
and document-disjoint validation contract. Until publication produces that
real manifest, scaled `rig prepare` fails with an explicit message; it
never manufactures a placeholder or treats a cache-local build plan as a
download contract.

Doctor, profiling, and runs resolve the same saved name and shard prefix. A
missing selection fails with the exact preparation command instead of falling
back to another corpus. Dataset choice does not determine trainer steps; the
fixed-TPP recipe plan resolves the token horizon independently.

## Binary format

All shards use the llm.c GPT-2 v1 layout:

1. A 1,024-byte header containing 256 little-endian signed 32-bit integers.
2. Header element 0 is magic `20240520`.
3. Header element 1 is version `1`.
4. Header element 2 is the number of tokens.
5. The payload contains exactly that many little-endian `uint16` token IDs.

Validation rejects an incorrect magic/version, a short header, a non-positive or
unexpected token count, trailing bytes, a truncated payload, an unexpected
declared size, or a hash mismatch.

The upstream raw corpus is
[HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
configuration `sample-10BT`. The pretokenized cache is
[kjj0/fineweb10B-gpt2](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2),
whose documented generation matches llm.c: GPT-2 `tiktoken`, with token 50256
inserted before each document.

The raw FineWeb dataset is distributed under ODC-By 1.0 and remains subject to
the source Common Crawl terms. Its dataset card warns that harmful material and
personal information may remain. The repository's Apache-2.0 license covers
this project's code; it does not replace the corpus terms. The prepared shard
bytes are pinned exactly, but the prepared repository does not identify an
immutable raw-corpus or preprocessing-code revision, so that earlier lineage is
recorded as upstream-claimed rather than independently reconstructed here.

The nested 2B, 4B, 8B, and 75B prefixes are already built and published, and
each manifest pins an immutable URL and SHA-256 per shard, so `rig` downloads
and verifies them without any local build step. The builder and its entrypoint
that produced those bytes are published beside the data under `provenance/` in
the dataset repository, and remain in this repository's history at `1d73eb5`.

## Fresh10 diagnostic

Official preparation also downloads `fresh10`, a diagnostic-only temporal
out-of-distribution set hosted in the public
[`quintic/fresh10`](https://huggingface.co/datasets/quintic/fresh10) dataset
repository. Its checked-in manifest pins an immutable Hub revision and the
SHA-256 of every domain shard. A shard contains four independent recent
documents; each span stores one GPT-2 end-of-text context token followed by
exactly 2,048 scored tokens. Evaluation therefore scores 8,192 targets per
domain and 81,920 overall without allowing a target to cross a source boundary.

The ten fixed domains are science, medicine, software, history, fiction,
government, legal, economics, climate, and education. Per-document metadata in
the manifest includes title, publisher, source and license URLs, publication and
retrieval dates, raw/canonical text hashes, and exact token offsets. Licenses
remain per source—this mixed collection has no single replacement license—and
the repository's Apache-2.0 license covers code only. The Hub dataset card lists
attribution and reuse obligations. Fresh10 loss and perplexity are reported per
domain and as a macro diagnostic; they never affect qualification.
