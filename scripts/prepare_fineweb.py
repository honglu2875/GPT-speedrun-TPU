# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pyarrow==19.0.1",
#   "tiktoken==0.11.0",
# ]
# ///
"""Prepare nested, GPT-2-tokenized FineWeb datasets under an SHM root."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from speedrun.fineweb_builder import (  # noqa: E402
    BuildConfig,
    DEFAULT_HERO_TOKENS,
    DEFAULT_RESERVE_BYTES,
    FineWebBuildError,
    ParquetDocumentSource,
    SourceFileCache,
    TiktokenGPT2Encoder,
    build_fineweb,
    configure_cache_root,
    ensure_build_work_directory,
    load_fresh10_exclusion_policy,
    load_or_fetch_inventory,
    probe_fineweb,
)


def token_count(value: str) -> int:
    normalized = value.strip().replace("_", "").upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    multiplier = multipliers.get(normalized[-1:], 1)
    number = normalized[:-1] if multiplier != 1 else normalized
    try:
        parsed = float(number) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid token count: {value!r}") from exc
    if parsed <= 0 or not parsed.is_integer():
        raise argparse.ArgumentTypeError("token count must be a positive integer")
    return int(parsed)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build nested 2B/4B/8B/hero llm.c GPT-2 shards from the pinned, "
            "globally shuffled FineWeb 100BT source."
        )
    )
    result.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Exact SHM-rooted output/cache directory (for example shm/fineweb-scaled)",
    )
    result.add_argument(
        "--through",
        choices=("2B", "4B", "8B", "hero"),
        default="hero",
        help="Stop once this nested prefix is complete",
    )
    result.add_argument(
        "--hero-tokens",
        type=token_count,
        default=DEFAULT_HERO_TOKENS,
        help="Hero total including 100M validation tokens (default: 75B)",
    )
    result.add_argument(
        "--reserve-gib",
        type=float,
        default=DEFAULT_RESERVE_BYTES / 1024**3,
        help="Free SHM space reserved for caches/process memory (default: 16 GiB)",
    )
    result.add_argument("--batch-rows", type=int, default=256)
    result.add_argument(
        "--tokenizer-threads",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    result.add_argument("--max-document-mib", type=int, default=16)
    result.add_argument("--download-timeout", type=float, default=300.0)
    result.add_argument(
        "--fresh10-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "manifests" / "fresh10.json",
        help="Pinned 40-document manifest used for defensive URL/hash exclusions",
    )
    result.add_argument(
        "--plan-only",
        action="store_true",
        help="Fetch provenance, create all folders, and preflight without downloading data",
    )
    result.add_argument(
        "--probe-tokens",
        type=token_count,
        help=(
            "Instead of writing shards, tokenize this many sampled GPT-2 tokens and "
            "report pre-2024 retention/throughput"
        ),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.reserve_gib < 0:
        raise SystemExit("--reserve-gib cannot be negative")

    caches = configure_cache_root(args.root)
    root = Path(args.root).expanduser().resolve(strict=True)
    work = ensure_build_work_directory(root)

    try:
        inventory = load_or_fetch_inventory(work / "source.json")
        exclusions = load_fresh10_exclusion_policy(args.fresh10_manifest)
        last_print = 0.0

        def download_progress(name: str, completed: int, total: int) -> None:
            nonlocal last_print
            now = time.monotonic()
            if now - last_print >= 5 or completed == total:
                print(
                    f"source {Path(name).name}: {completed / 1024**3:.2f}/"
                    f"{total / 1024**3:.2f} GiB",
                    flush=True,
                )
                last_print = now

        source_cache = SourceFileCache(
            caches["source"],
            inventory,
            timeout=args.download_timeout,
            progress=download_progress,
        )
        source = ParquetDocumentSource(inventory, source_cache)
        encoder = TiktokenGPT2Encoder(args.tokenizer_threads)

        if args.probe_tokens is not None:
            started = time.monotonic()
            probe = probe_fineweb(
                source,
                encoder,
                exclusions,
                examined_token_target=args.probe_tokens,
                batch_rows=args.batch_rows,
                max_document_bytes=args.max_document_mib * 1024**2,
            )
            elapsed = time.monotonic() - started
            source_cache.release_active()
            report = {
                "mode": "probe",
                "seconds": elapsed,
                "examined_documents": probe.examined_documents,
                "accepted_documents": probe.accepted_documents,
                "examined_gpt2_tokens": probe.examined_gpt2_tokens,
                "accepted_gpt2_tokens": probe.accepted_gpt2_tokens,
                "estimated_token_retention": probe.estimated_token_retention,
                "conservative_hero_cap_tokens": int(
                    probe.estimated_token_retention
                    * 100_000_000_000
                    * 0.9
                    // 100_000_000
                    * 100_000_000
                ),
                "conservative_cap_policy": (
                    "floor to 100M of 90% * observed retention * nominal 100B source"
                ),
                "examined_tokens_per_second": probe.examined_gpt2_tokens / elapsed,
                "accepted_tokens_per_second": probe.accepted_gpt2_tokens / elapsed,
                "exclusion_documents": probe.exclusion_documents,
                "exclusion_gpt2_tokens": probe.exclusion_gpt2_tokens,
                "source_cursor": probe.next_cursor,
                "source_inventory_sha256": inventory.digest,
                "exclusion_policy_sha256": exclusions.digest,
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        config = BuildConfig(
            root=root,
            hero_tokens=args.hero_tokens,
            reserve_bytes=int(args.reserve_gib * 1024**3),
            max_document_bytes=args.max_document_mib * 1024**2,
            batch_rows=args.batch_rows,
            tokenizer_threads=args.tokenizer_threads,
        )
        result = build_fineweb(
            config,
            inventory,
            exclusions,
            source,
            encoder,
            through=args.through,
            stop_after_shards=0 if args.plan_only else None,
            progress=lambda message: print(message, flush=True),
        )
        print(
            json.dumps(
                {
                    "root": str(result.root),
                    "completed_shards": result.completed_shards,
                    "completed_tokens": result.completed_tokens,
                    "complete_variants": result.complete_variants,
                    "plan_only": args.plan_only,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except FineWebBuildError as exc:
        print(f"fineweb preparation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "\ninterrupted safely; rerun the same command to resume from the last "
            "completed shard",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
