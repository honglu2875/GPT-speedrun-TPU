from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from rig import evaluation
from rig.evaluation import DomainEvaluation, EvaluationReport, EvaluationResult
from rig.tokens import DocumentSpan, DownstreamDomain


class EvaluationReportTests(unittest.TestCase):
    def test_one_report_owns_rows_aggregation_and_result_metadata(self) -> None:
        canonical = EvaluationResult(loss=1.5, scored_tokens=100, seconds=0.25)
        report = EvaluationReport(
            canonical,
            (
                DomainEvaluation(
                    "books", EvaluationResult(loss=2.0, scored_tokens=20, seconds=0.1)
                ),
                DomainEvaluation(
                    "code", EvaluationResult(loss=4.0, scored_tokens=30, seconds=0.2)
                ),
            ),
        )

        macro = report.macro
        self.assertIsNotNone(macro)
        assert macro is not None
        self.assertEqual(macro.loss, 3.0)
        self.assertEqual(macro.scored_tokens, 50)
        self.assertAlmostEqual(macro.seconds, 0.3)
        self.assertAlmostEqual(macro.perplexity, math.exp(3.0))

        rows = report.validation_rows(step=7, tokens_processed=896)
        self.assertEqual(
            [(row.kind, row.domain, row.canonical) for row in rows],
            [
                ("fineweb", "fineweb", True),
                ("downstream", "books", False),
                ("downstream", "code", False),
                ("downstream_macro", "fresh10_macro", False),
            ],
        )
        self.assertTrue(all(row.step == 7 for row in rows))
        self.assertTrue(all(row.tokens_processed == 896 for row in rows))

        metadata = report.metadata()
        self.assertEqual(
            metadata,
            {
                "fineweb": {
                    "loss": 1.5,
                    "perplexity": math.exp(1.5),
                    "scored_tokens": 100,
                    "seconds": 0.25,
                    "canonical": True,
                },
                "fresh10": {
                    "domains": {
                        "books": {
                            "loss": 2.0,
                            "perplexity": math.exp(2.0),
                            "scored_tokens": 20,
                            "seconds": 0.1,
                        },
                        "code": {
                            "loss": 4.0,
                            "perplexity": math.exp(4.0),
                            "scored_tokens": 30,
                            "seconds": 0.2,
                        },
                    },
                    "macro_loss": 3.0,
                    "macro_perplexity": math.exp(3.0),
                    "scored_tokens": 50,
                    "seconds": 0.30000000000000004,
                },
            },
        )

    def test_canonical_only_report_has_no_downstream_summary(self) -> None:
        report = EvaluationReport(
            EvaluationResult(loss=1.0, scored_tokens=64, seconds=0.5)
        )

        self.assertIsNone(report.macro)
        self.assertEqual(len(report.validation_rows(step=1, tokens_processed=64)), 1)
        self.assertEqual(set(report.metadata()), {"fineweb"})

    def test_report_rejects_duplicate_domain_names(self) -> None:
        result = EvaluationResult(loss=1.0, scored_tokens=8, seconds=0.1)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            EvaluationReport(
                result,
                (DomainEvaluation("same", result), DomainEvaluation("same", result)),
            )

    def test_result_rejects_empty_or_nonfinite_measurements(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one token"):
            EvaluationResult(loss=1.0, scored_tokens=0, seconds=0.1)
        with self.assertRaises(FloatingPointError):
            EvaluationResult(loss=float("nan"), scored_tokens=1, seconds=0.1)

    def test_downstream_domain_orchestration_preserves_manifest_order(self) -> None:
        domains = (
            DownstreamDomain(
                "books",
                np.arange(5, dtype=np.int32),
                (DocumentSpan(0, 5, 1, 4),),
            ),
            DownstreamDomain(
                "code",
                np.arange(3, dtype=np.int32),
                (DocumentSpan(0, 3, 1, 2),),
            ),
        )

        def compiled_eval(params, x, y, mask):
            del params, x, y
            scored = mask.sum()
            return np.asarray(scored * 2.0), np.asarray(scored)

        with patch.object(
            evaluation.jax, "device_put", side_effect=lambda value, _sharding: value
        ):
            results = evaluation.evaluate_downstream_domains(
                object(),
                domains,
                compiled_eval,
                object(),
                batch_size=2,
                seq_len=4,
            )

        self.assertEqual([entry.name for entry in results], ["books", "code"])
        self.assertEqual([entry.result.loss for entry in results], [2.0, 2.0])
        self.assertEqual([entry.result.scored_tokens for entry in results], [4, 2])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
