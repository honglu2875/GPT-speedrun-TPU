---
pretty_name: FineWeb Scaled GPT-2 Prefixes
language:
- en
license: odc-by
task_categories:
- text-generation
tags:
- pretraining
- gpt2
- llm.c
---

# FineWeb Scaled GPT-2 Prefixes

This repository contains nested 2B, 4B, 8B, and hero-scale token prefixes for
controlled language-model scaling experiments. The binary shards use the
llm.c GPT-2 v1 format and are directly consumable by the GPT TPU Rig
trainer.

## Dataset structure

Each folder is independently usable after its `manifest.json` is present:

| folder | validation tokens | training tokens |
|---|---:|---:|
| `2B/` | 100,000,000 | 1,900,000,000 |
| `4B/` | 100,000,000 | 3,900,000,000 |
| `8B/` | 100,000,000 | 7,900,000,000 |
| `hero/` | 100,000,000 | 74,900,000,000 |

The variants are exact nested prefixes: the validation shard and early
training shards have identical SHA-256 values across folders. Repeated Hub
paths are expected to deduplicate at the content-storage layer.

Every `.bin` file has a 1,024-byte header of 256 little-endian signed int32
values. Header slots 0, 1, and 2 contain magic `20240520`, version `1`, and the
token count. The remaining payload is exactly 100,000,000 little-endian
`uint16` GPT-2 token IDs.

## Source and preparation

The source is
[`HuggingFaceFW/fineweb_100BT-shuffled`](https://huggingface.co/datasets/HuggingFaceFW/fineweb_100BT-shuffled)
at revision `ee8552966e3d6a5fee2f317f2ae0b342be03d998`. Its card describes a
global document shuffle with seed 42. Each variant includes `source.json` with
the immutable revision plus the byte length and LFS SHA-256 of all 100 source
Parquet files.

To maintain temporal separation from the Fresh10 diagnostic corpus, source
rows are retained only when their crawl/source date is strictly before
2024-04-01; missing or invalid dates are rejected. The 40 normalized Fresh10
source URLs and canonical-text hashes are excluded defensively. Fresh10 raw
artifact hashes are also compared opportunistically with UTF-8 source-row
bytes, but are not treated as normalized-content hashes.
`exclusions.json` contains the exact policy, and each manifest records observed
exclusion counts at that prefix boundary.

Retained text is re-tokenized—not inferred from the source `token_count`
column—with GPT-2 `tiktoken==0.11.0`. Token 50256 is inserted before every
document. The first exact 100M tokens are validation and all remaining shards
are training. Ordinary training-shard boundaries may split a document. At the
validation boundary only, the remainder of the crossing document is discarded
so validation and training are document-disjoint; the manifest records the
discarded-token count and a hash-safe document identifier.

The preparation code uses one verified source Parquet at a time, bounded
PyArrow batches, streaming output writes, exact row-level checkpoints, atomic
shard installation, and SHA-256 verification. The generated manifests are the
authoritative per-file integrity inventory.

The publication includes the exact frozen preparation sources at
[`provenance/fineweb_builder.py`](provenance/fineweb_builder.py) and
[`provenance/prepare_fineweb.py`](provenance/prepare_fineweb.py). Their
SHA-256 values are respectively
`26c61bc921af290e6beb28596feb2c50cac5b15a56a2f3adf921682317f6f109`
and `3a676241de10c3ac7cf36ed19ccbd1c0e419bb90de960d4e14be51a1f225bd5c`.
The same frozen sources are independently retrievable from immutable Git commit
[`c6acab32cea6e48260d139be1774b3e3286d7afd`](https://github.com/honglu2875/GPT-speedrun-TPU/tree/c6acab32cea6e48260d139be1774b3e3286d7afd).
Every production variant pins source-inventory SHA-256
`02ddc6361cc2f8a3d23b0d8b823c7eb7e2b1663ad3d0eff63e83b373456fc12b`,
exclusion-policy SHA-256
`ab25cabd0781b1046b7ad7b281b4147ff6e27d36977f4e842b8c92573399ad77`,
and preparation-core SHA-256
`4bbdcb76da837276f6f337b805d37a74e3272b476e01fd198f416097abe19241`.

## Intended use

These variants are intended for controlled pretraining and IsoFLOP/scaling-law
experiments where data prefixes, tokenizer, split, and binary layout must be
held constant. They are not intended as a benchmark test set or as a source of
factual ground truth.

## Limitations and responsible use

FineWeb derives from Common Crawl. Despite upstream filtering, it can contain
harmful, biased, inaccurate, copyrighted, or personally identifying material.
The temporal cutoff and Fresh10 exclusions address evaluation leakage, not all
possible duplication or contamination. Users remain responsible for suitable
content controls, legal review, and downstream safety evaluation.

## License and attribution

FineWeb is distributed under
[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and remains subject
to the [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use) and
underlying source terms. The GPT TPU Rig preparation code is
Apache-2.0; that code license does not replace the corpus license. Please cite
and attribute the upstream
[FineWeb dataset and paper](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
and the shuffled source's requested SmolData citation:

```bibtex
@misc{niklaus2026smoldata,
  title        = {SmolData},
  author       = {Joel Niklaus and Hynek Kydl{\'\i}{\v{c}}ek},
  year         = {2026},
  publisher    = {Hugging Face},
  journal      = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/collections/HuggingFaceFW/smol-data}}
}
```

Retain this preparation provenance when redistributing shards.
