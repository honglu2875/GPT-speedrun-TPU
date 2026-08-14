# Corpus producer tooling

These scripts build and publish the scaled FineWeb corpora described by
`data/manifests/fineweb-scaled-gpt2/`. Nothing on the training path imports
them — a normal `rig run` consumes the checked-in manifests and the prepared
cache, so you only need this directory when minting a *new* corpus.

| Script | Purpose |
|---|---|
| `prepare_fineweb.py` | build a nested GPT-2-tokenized corpus under a cache root |
| `publish_fineweb.py` | upload a completed variant and emit its URL-bearing manifest |

## Two files are byte-frozen

`prepare_fineweb.py` and `rig/fineweb_builder.py` are provenance for roughly
25 GB of already-published data. Their SHA-256 digests are recorded inside
every published manifest as `entrypoint_sha256` and `builder_module_sha256`,
and `rig/data_routing.py` checks both against `SCALED_ENTRYPOINT_SHA256` and
`SCALED_BUILDER_SHA256` whenever a scaled manifest is resolved — which happens
on every non-smoke run.

```text
scripts/prepare_fineweb.py   3a676241de10c3ac7cf36ed19ccbd1c0e419bb90de960d4e14be51a1f225bd5c
rig/fineweb_builder.py       26c61bc921af290e6beb28596feb2c50cac5b15a56a2f3adf921682317f6f109
```

Editing either one changes the hash a rebuild would stamp into its new
manifest, and that manifest would then be rejected. Both were therefore left
untouched by the `speedrun` -> `rig` rename, and both still spell their imports
with the package's former name.

## Running the frozen entrypoint

Because `prepare_fineweb.py` still says `from speedrun.fineweb_builder import
...`, that name has to resolve. `rig.frozen` registers it:

```bash
uv run --with pyarrow==19.0.1 --with tiktoken==0.11.0 python -c \
  "import rig.frozen, runpy; runpy.run_path('scripts/prepare_fineweb.py', run_name='__main__')" \
  -- --help
```

`publish_fineweb.py` is not frozen and imports `rig.frozen` itself, so it runs
directly.

## Deliberately re-pinning

If you ever do need to change the builder, treat it as minting a new corpus
rather than editing an old one: update the two constants in
`rig/data_routing.py` to the new digests, rebuild, and publish fresh manifests.
Never rewrite the `preparation` block of an existing manifest to match new
code — that would claim the published bytes were produced by a builder that
never touched them.
