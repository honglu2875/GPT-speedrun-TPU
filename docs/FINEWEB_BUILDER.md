# Scaled FineWeb preparation

`scripts/prepare_fineweb.py` builds four nested GPT-2-tokenized datasets in
the existing llm.c binary format:

| folder | validation | training | physical prefix |
|---|---:|---:|---:|
| `2B/` | 100M | 1.9B | 2B tokens |
| `4B/` | 100M | 3.9B | 4B tokens |
| `8B/` | 100M | 7.9B | 8B tokens |
| `hero/` | 100M | 74.9B | exactly 75B tokens |

The folders contain hard links to a single private shard pool. They therefore
look self-contained to training and upload tools without consuming 89B tokens
of duplicate local storage. Copying the folders to different filesystems will
materialize the duplicates; inspect physical use with `du`, not by summing
apparent file sizes.

## Reproducibility and isolation

The input is the globally shuffled
`HuggingFaceFW/fineweb_100BT-shuffled` dataset at immutable revision
`ee8552966e3d6a5fee2f317f2ae0b342be03d998`. Its dataset card says documents
were globally shuffled with seed 42. The builder records and validates the LFS
SHA-256 and byte length of all 100 Parquet files.

Rows are retained only when their source date is strictly before 2024-04-01.
Missing or malformed dates are excluded. The 40 normalized Fresh10 source URLs
and canonical-text hashes are also excluded defensively. Its raw artifact
hashes are compared opportunistically to UTF-8 source-row bytes, but are not
treated as normalized-content hashes. This
preserves the intended temporal separation from Fresh10 even though the
shuffled source repository includes later crawls. Each completed prefix
records exclusion counts and the exact exclusion-policy hash.

The source `token_count` column is never trusted as GPT-2 tokenization. Text is
re-tokenized with `tiktoken==0.11.0`, and GPT-2 EOT token 50256 is inserted
before every retained document. The first exact 100M output tokens become
validation; every subsequent exact 100M shard is training data. A document may
cross ordinary shard boundaries. The remainder of the one document that fills
validation is discarded before training begins, making validation and training
document-disjoint. Its exact discarded-token count and a SHA-256 of its source
document identifier are recorded without exposing the identifier.

## Storage and memory policy

Pass one explicit root on the large RAM filesystem. The script forces
`HF_HOME`, datasets/Hub caches, `TIKTOKEN_CACHE_DIR`, `XDG_CACHE_HOME`, and all
temporary directories below `<root>/.cache`. Put uv's cache there too because
uv resolves the script dependencies before Python can set environment
variables:

```console
export FW_ROOT=/dev/shm/fineweb-scaled
export UV_CACHE_DIR="$FW_ROOT/.cache/uv"
```

Only one compressed source Parquet (about 3–4 GB) is present at a time. It is
downloaded resumably, checked against its LFS SHA-256, processed in small
PyArrow batches, and removed after its final row. The tokenizer never holds a
100M-token output shard in memory; tokens stream into an adjacent `.part`.
The default maximum source document is 16 MiB and the default safety reserve is
16 GiB.

A 100B output needs about 200 GB before the active Parquet, output `.part`, and
memory reserve. It is not safe on the current 201 GiB `/dev/shm`. The default
75B hero needs about 150 GB plus working space. Preparation fails before source
downloads when the requested target does not fit.

## Probe, plan, and build

First run the 1M-token prefix probe as a speed smoke test. The upstream stream
is already globally shuffled; this command examines its beginning rather than
drawing an independent random sample. It downloads and verifies the first
source Parquet, measures actual GPT-2 throughput and pre-cutoff token retention,
reports JSON, and deletes the source Parquet after success:

```console
uv run --script scripts/prepare_fineweb.py \
  --root "$FW_ROOT" \
  --probe-tokens 1M
```

Before approving a hero build, repeat with `--probe-tokens 100M`. The report
includes a conservative cap: 90% of observed retention times the nominal 100B
source, rounded down to a 100M shard. Proceed with 75B only when that cap is at
least 75B. The 1M smoke estimate is too small to be the hero supply gate.

Create all four folders, pin provenance, and check capacity without downloading
the corpus:

```console
uv run --script scripts/prepare_fineweb.py \
  --root "$FW_ROOT" \
  --through 8B \
  --plan-only
```

Build and finalize the 2B, 4B, and 8B manifests first:

```console
uv run --script scripts/prepare_fineweb.py \
  --root "$FW_ROOT" \
  --through 8B
```

Then continue the same stream to the 75B hero in the background:

```console
nohup env UV_CACHE_DIR="$UV_CACHE_DIR" \
  uv run --script scripts/prepare_fineweb.py \
  --root "$FW_ROOT" \
  --through hero \
  >"$FW_ROOT/hero-build.log" 2>&1 &
```

Rerun the same command after interruption. Every completed output shard has an
atomic checkpoint containing the next Parquet file, row group, row, and any
unconsumed token tail from the current document. An incomplete `.part` is
rewritten from the previous checkpoint. In the narrow crash window where an
output shard was renamed but its checkpoint was not, the builder reproduces
that one shard and accepts it only if its SHA-256 matches.

Each completed folder has a normal `manifest.json`. The publisher's dry-run is
the release-gate verifier; selecting `8B` also validates its required 2B and 4B
predecessors and the exact nested-prefix relationship:

```console
uv run --script scripts/publish_fineweb.py \
  --root "$FW_ROOT" \
  --variants 8B \
  --dry-run
```

Do not train from a folder until `manifest.json` appears. `BUILD_PLAN.json`
exists from the beginning, while an incomplete hero contains only the prefix
hard links produced so far.

The cache-local manifest above is suitable for direct verification but has no
download URLs. It is therefore not used as an implicit fallback by the normal
preparation wizard: peer TPU VMs have independent RAM filesystems and cannot
bootstrap from the controller's local file. After publication, commit the
publisher-produced immutable URL-bearing manifests at
`data/manifests/fineweb-scaled-gpt2/{2B,4B,8B,hero}.json`. Then an official
preparation budget above the classic 900M capacity automatically downloads or
checks the smallest fitting variant beneath
`<data-path>/fineweb-scaled/<variant>/`. This routing is preparation-only and
does not modify the official run contract.

## Publish to Hugging Face

The publisher defaults to the public dataset repository
`quintic/fineweb-scaled-gpt2` and to the completed 2B, 4B, and 8B variants.
It reads `.env.hf` as data—never as a shell file—and requires it to be a
current-user-owned, non-symlink regular file with no group/world permissions
and exactly one `HF_TOKEN=...` assignment. The token is never printed.

Validate every local shard and show the upload plan without reading the token
or changing the Hub:

```console
uv run --script scripts/publish_fineweb.py \
  --root "$FW_ROOT" \
  --dry-run
```

Publish the three small variants first:

```console
uv run --script scripts/publish_fineweb.py \
  --root "$FW_ROOT" \
  --variants 2B 4B 8B
```

After the background build completes, publish only hero:

```console
uv run --script scripts/publish_fineweb.py \
  --root "$FW_ROOT" \
  --variants hero
```

Before any upload, each variant directory must contain exactly the manifest's
fixed shard names plus `manifest.json`, `source.json`, `exclusions.json`, and
`BUILD_PLAN.json`: no extras, links, or subdirectories are accepted. The bulk
uploader receives the corresponding closed list of exact names, so it cannot
upload `.cache`, `.fineweb-build`, logs, or credentials. The exact frozen
builder and entrypoint are uploaded separately under `provenance/`. Xet
high-performance mode is enabled and identical prefix bytes are expected to be
content-deduplicated by the Hub. Each variant then receives a separate manifest
whose file URLs pin the immutable shard commit. The publisher writes
per-variant commit receipts after every completed variant, so re-running after
an interruption resumes against Hub content.

For a public repo it finally verifies without authentication:

- the exact published manifest hash;
- the exact closed remote tree and Hub LFS size/SHA-256 metadata for every
  shard; and
- the llm.c header fetched by HTTP Range for those selected shards.

Publication receipts, including shard and manifest commits, are stored at
`<root>/.fineweb-build/publication.json`; that private work directory is never
part of an upload. Verified URL-bearing manifests are staged at
`<root>/.fineweb-build/staged-manifests/{variant}.json` (or the explicit
`--manifest-output` directory) for review and check-in under
`data/manifests/fineweb-scaled-gpt2/`.
