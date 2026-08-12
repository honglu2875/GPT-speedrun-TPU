# Data

No training corpus is stored in Git, and importing `speedrun.data` never
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
uv run speedrun prepare --path shm/
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
from speedrun.data import prepare

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
