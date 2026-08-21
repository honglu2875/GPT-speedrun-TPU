"""Evaluation passes and the schedules that decide when they run.

Every recipe scores the same way: a deterministic prefix of the validation
split for the headline number, packed fixed-shape batches for a downstream
domain, and sparse probes on a cadence during training. None of that varies
with the model, so none of it belongs in an entry program.

Both passes take plain shapes rather than a recipe's config, and both are
deliberately outside the timed training region -- ``train_seconds`` measures
training, and an evaluation that crept inside it would flatter the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from rig.mesh import (
    finite_metric,
    local_batch_size,
    local_device_get,
    put_host_local_array,
    rank_local_slice,
)
from rig.runlog import ValidationRow
from rig.tokens import DownstreamDomain, TokenDataset, downstream_batches


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One token-weighted loss measurement and its execution accounting."""

    loss: float
    scored_tokens: int
    seconds: float

    def __post_init__(self) -> None:
        finite_metric("evaluation loss", self.loss)
        if self.scored_tokens <= 0:
            raise ValueError("an evaluation must score at least one token")
        finite_metric("evaluation seconds", self.seconds, positive=True)

    @property
    def perplexity(self) -> float:
        return perplexity_from_loss(self.loss)

    def validation_row(
        self,
        *,
        step: int,
        tokens_processed: int,
        kind: str,
        domain: str,
        canonical: bool,
    ) -> ValidationRow:
        """Represent this measurement in the shared validation artifact."""

        return ValidationRow(
            step=step,
            tokens_processed=tokens_processed,
            kind=kind,
            domain=domain,
            validation_tokens=self.scored_tokens,
            validation_loss=self.loss,
            perplexity=self.perplexity,
            validation_seconds=self.seconds,
            canonical=canonical,
        )

    def metadata(
        self, *, canonical: bool | None = None
    ) -> dict[str, float | int | bool]:
        """Return the stable result.json representation of one measurement."""

        result: dict[str, float | int | bool] = {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "scored_tokens": self.scored_tokens,
            "seconds": self.seconds,
        }
        if canonical is not None:
            result["canonical"] = canonical
        return result


@dataclass(frozen=True, slots=True)
class DomainEvaluation:
    """A named downstream domain and its measured result."""

    name: str
    result: EvaluationResult

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a downstream evaluation needs a domain name")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Canonical and optional downstream results in every persisted form.

    This is intentionally a value object rather than an evaluation lifecycle.
    Recipes still decide when each pass runs and which executable and data it
    receives. The report owns only the fixed aggregation and serialization
    contract shared by every recipe.
    """

    canonical: EvaluationResult
    downstream: tuple[DomainEvaluation, ...] = ()
    canonical_name: str = "fineweb"
    downstream_name: str = "fresh10"

    def __post_init__(self) -> None:
        object.__setattr__(self, "downstream", tuple(self.downstream))
        if not self.canonical_name or not self.downstream_name:
            raise ValueError("evaluation suite names cannot be empty")
        names = [entry.name for entry in self.downstream]
        if len(names) != len(set(names)):
            raise ValueError("downstream evaluation domain names must be unique")

    @property
    def macro(self) -> EvaluationResult | None:
        """Unweighted domain mean, matching the established Fresh10 score."""

        if not self.downstream:
            return None
        return EvaluationResult(
            loss=finite_metric(
                f"{self.downstream_name} macro loss",
                float(np.mean([entry.result.loss for entry in self.downstream])),
            ),
            scored_tokens=sum(entry.result.scored_tokens for entry in self.downstream),
            seconds=finite_metric(
                f"{self.downstream_name} seconds",
                sum(entry.result.seconds for entry in self.downstream),
                positive=True,
            ),
        )

    def validation_rows(
        self, *, step: int, tokens_processed: int
    ) -> tuple[ValidationRow, ...]:
        """Return canonical, domain, then macro rows in artifact order."""

        rows = [
            self.canonical.validation_row(
                step=step,
                tokens_processed=tokens_processed,
                kind="fineweb",
                domain=self.canonical_name,
                canonical=True,
            )
        ]
        rows.extend(
            entry.result.validation_row(
                step=step,
                tokens_processed=tokens_processed,
                kind="downstream",
                domain=entry.name,
                canonical=False,
            )
            for entry in self.downstream
        )
        macro = self.macro
        if macro is not None:
            rows.append(
                macro.validation_row(
                    step=step,
                    tokens_processed=tokens_processed,
                    kind="downstream_macro",
                    domain=f"{self.downstream_name}_macro",
                    canonical=False,
                )
            )
        return tuple(rows)

    def metadata(self) -> dict[str, Any]:
        """Return the complete evaluations section for result.json."""

        result: dict[str, Any] = {
            self.canonical_name: self.canonical.metadata(canonical=True)
        }
        macro = self.macro
        if macro is not None:
            result[self.downstream_name] = {
                "domains": {
                    entry.name: entry.result.metadata() for entry in self.downstream
                },
                "macro_loss": macro.loss,
                "macro_perplexity": macro.perplexity,
                "scored_tokens": macro.scored_tokens,
                "seconds": macro.seconds,
            }
        return result


def should_run_validation_probe(step: int, *, every: int, final_step: int) -> bool:
    """Return whether this step gets a non-canonical fixed-prefix probe."""

    return every > 0 and step < final_step and step % every == 0


def should_run_diagnostics(step: int, *, every: int, final_step: int) -> bool:
    """Capture the first/final updates plus the configured sparse cadence."""

    return every > 0 and (step == 1 or step == final_step or step % every == 0)


def evaluate_validation_prefix(
    params: Any,
    dataset: TokenDataset,
    compiled_eval: Any,
    data_sharding: NamedSharding,
    *,
    batch_size: int,
    seq_len: int,
    semantic_vocab_size: int,
    batches: int,
    mesh: Mesh | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> EvaluationResult:
    """Synchronously evaluate batches ``0..batches-1`` of the fixed prefix."""

    if batches <= 0:
        raise ValueError("validation batch count must be positive")
    started = time.perf_counter()
    loss_sum = 0.0
    scored_tokens = 0
    if process_count > 1 and mesh is None:
        raise ValueError("a global mesh is required for multi-process evaluation")
    local_batch = local_batch_size(batch_size, process_count)
    mask_host = np.ones((local_batch, seq_len), dtype=np.float32)
    if mesh is None:
        mask = jax.device_put(mask_host, data_sharding)
    else:
        mask = put_host_local_array(
            mask_host, mesh, P("data", None), data_sharding, process_count
        )
    for eval_index in range(batches):
        eval_x_host, eval_y_host = dataset.validation_batch(
            eval_index,
            batch_size,
            seq_len,
            semantic_vocab_size,
        )
        eval_x_host = rank_local_slice(eval_x_host, process_index, process_count)
        eval_y_host = rank_local_slice(eval_y_host, process_index, process_count)
        if mesh is None:
            eval_x = jax.device_put(eval_x_host, data_sharding)
            eval_y = jax.device_put(eval_y_host, data_sharding)
        else:
            eval_x = put_host_local_array(
                eval_x_host, mesh, P("data", None), data_sharding, process_count
            )
            eval_y = put_host_local_array(
                eval_y_host, mesh, P("data", None), data_sharding, process_count
            )
        batch_loss_sum, batch_scored = local_device_get(
            compiled_eval(params, eval_x, eval_y, mask)
        )
        loss_sum += float(batch_loss_sum)
        scored_tokens += int(batch_scored)
    elapsed = max(time.perf_counter() - started, 1.0e-12)
    expected_tokens = batches * batch_size * seq_len
    if scored_tokens != expected_tokens:
        raise RuntimeError(
            f"validation executable scored {scored_tokens:,} tokens; expected "
            f"{expected_tokens:,}"
        )
    return EvaluationResult(
        loss=finite_metric("validation_loss", loss_sum / scored_tokens),
        scored_tokens=scored_tokens,
        seconds=finite_metric("validation_seconds", elapsed, positive=True),
    )


def evaluate_downstream_domain(
    params: Any,
    domain: DownstreamDomain,
    compiled_eval: Any,
    data_sharding: NamedSharding,
    *,
    batch_size: int,
    seq_len: int,
    mesh: Mesh | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> EvaluationResult:
    """Evaluate one domain with exact masking and the shared eval executable."""

    started = time.perf_counter()
    loss_sum = 0.0
    scored_tokens = 0
    if process_count > 1 and mesh is None:
        raise ValueError("a global mesh is required for multi-process evaluation")
    for x_host, y_host, mask_host in downstream_batches(
        domain, seq_len=seq_len, batch_size=batch_size
    ):
        x_host = rank_local_slice(x_host, process_index, process_count)
        y_host = rank_local_slice(y_host, process_index, process_count)
        mask_host = rank_local_slice(mask_host, process_index, process_count)
        if mesh is None:
            x = jax.device_put(x_host, data_sharding)
            y = jax.device_put(y_host, data_sharding)
            mask = jax.device_put(mask_host, data_sharding)
        else:
            x = put_host_local_array(
                x_host, mesh, P("data", None), data_sharding, process_count
            )
            y = put_host_local_array(
                y_host, mesh, P("data", None), data_sharding, process_count
            )
            mask = put_host_local_array(
                mask_host, mesh, P("data", None), data_sharding, process_count
            )
        batch_loss_sum, batch_scored = local_device_get(
            compiled_eval(params, x, y, mask)
        )
        loss_sum += float(batch_loss_sum)
        scored_tokens += int(batch_scored)
    elapsed = finite_metric(
        f"downstream {domain.name} seconds",
        max(time.perf_counter() - started, 1.0e-12),
        positive=True,
    )
    if scored_tokens != domain.scored_tokens:
        raise RuntimeError(
            f"downstream {domain.name} scored {scored_tokens:,} tokens; expected "
            f"{domain.scored_tokens:,}"
        )
    loss = finite_metric(f"downstream {domain.name} loss", loss_sum / scored_tokens)
    return EvaluationResult(loss=loss, scored_tokens=scored_tokens, seconds=elapsed)


def evaluate_downstream_domains(
    params: Any,
    domains: Sequence[DownstreamDomain],
    compiled_eval: Any,
    data_sharding: NamedSharding,
    *,
    batch_size: int,
    seq_len: int,
    mesh: Mesh | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> tuple[DomainEvaluation, ...]:
    """Evaluate manifest-ordered domains with the shared executable."""

    return tuple(
        DomainEvaluation(
            domain.name,
            evaluate_downstream_domain(
                params,
                domain,
                compiled_eval,
                data_sharding,
                batch_size=batch_size,
                seq_len=seq_len,
                mesh=mesh,
                process_index=process_index,
                process_count=process_count,
            ),
        )
        for domain in domains
    )


def perplexity_from_loss(loss: float) -> float:
    try:
        perplexity = math.exp(loss)
    except OverflowError as exc:
        raise FloatingPointError(f"loss {loss!r} overflows perplexity") from exc
    return finite_metric("perplexity", perplexity, positive=True)
