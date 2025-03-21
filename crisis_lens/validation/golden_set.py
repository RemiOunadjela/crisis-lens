"""Golden set evaluation framework.

The golden set is the ground truth for measuring classifier quality.
It's a curated collection of labeled examples that represent the full
taxonomy of crisis types, including tricky edge cases (satire, news
reporting about crises vs. actual crisis content, coded language).

In T&S, golden sets are maintained by senior analysts and updated
quarterly. They serve two purposes:
1. Regression testing -- did the latest prompt change break anything?
2. Boosted prompt refinement -- systematic error analysis drives
   prompt improvements.

The evaluation produces precision, recall, F1 at configurable
confidence thresholds, plus a detailed false-negative audit that
feeds directly into the boosted prompt loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from crisis_lens.config import IncidentType


class GoldenExample(BaseModel):
    """A single labeled example in the golden set."""

    example_id: str
    text: str
    source: str = "golden_set"
    language: str = "en"
    true_labels: list[IncidentType]
    true_severity: str = "P4"
    notes: str = ""


class PredictionEntry(BaseModel):
    """A prediction to compare against the golden set."""

    example_id: str
    predicted_labels: list[IncidentType] = Field(default_factory=list)
    predicted_severity: str = "P4"
    confidence: float = 0.0


class ConfusionStats(BaseModel):
    """Per-label confusion matrix stats."""

    label: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class FalseNegativeEntry(BaseModel):
    """Detailed record of a missed detection for audit."""

    example_id: str
    text_preview: str
    missed_labels: list[IncidentType]
    predicted_labels: list[IncidentType]
    confidence: float


class EvaluationResult(BaseModel):
    """Complete evaluation output."""

    total_examples: int
    per_label_stats: dict[str, ConfusionStats] = Field(default_factory=dict)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    severity_accuracy: float = 0.0
    false_negatives: list[FalseNegativeEntry] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Golden Set Evaluation ({self.total_examples} examples)",
            f"  Macro Precision: {self.macro_precision:.3f}",
            f"  Macro Recall:    {self.macro_recall:.3f}",
            f"  Macro F1:        {self.macro_f1:.3f}",
            f"  Severity Acc:    {self.severity_accuracy:.3f}",
            "",
            "  Per-label F1:",
        ]
        for label, stats in sorted(self.per_label_stats.items()):
            lines.append(
                f"    {label:30s}  P={stats.precision:.3f}"
                f"  R={stats.recall:.3f}  F1={stats.f1:.3f}"
            )

        if self.false_negatives:
            lines.extend(["", f"  False Negatives: {len(self.false_negatives)}"])
            for fn in self.false_negatives[:5]:
                missed = [lb.value for lb in fn.missed_labels]
                lines.append(f"    - {fn.example_id}: missed {missed}")

        if self.failure_modes:
            lines.extend(["", "  Identified Failure Modes:"])
            for fm in self.failure_modes:
                lines.append(f"    - {fm}")

        return "\n".join(lines)


class GoldenSetEvaluator:
    """Evaluates predictions against a golden set of labeled examples."""

    def __init__(self, golden_set: list[GoldenExample] | None = None):
        self._golden: dict[str, GoldenExample] = {}
        if golden_set:
            for ex in golden_set:
                self._golden[ex.example_id] = ex

    @classmethod
    def from_file(cls, path: str | Path) -> GoldenSetEvaluator:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        examples = [GoldenExample(**item) for item in data]
        return cls(golden_set=examples)

    def add_example(self, example: GoldenExample) -> None:
        self._golden[example.example_id] = example

    @property
    def size(self) -> int:
        return len(self._golden)

    def evaluate(
        self,
        predictions: list[PredictionEntry],
        confidence_threshold: float = 0.0,
    ) -> EvaluationResult:
        """Run full evaluation at a given confidence threshold."""
        all_labels: set[str] = set()
        for ex in self._golden.values():
            for inc_type in ex.true_labels:
                all_labels.add(inc_type.value)

        stats: dict[str, ConfusionStats] = {
            lbl: ConfusionStats(label=lbl) for lbl in all_labels
        }

        pred_map = {p.example_id: p for p in predictions}
        severity_correct = 0
        false_negatives: list[FalseNegativeEntry] = []

        for ex_id, golden in self._golden.items():
            pred = pred_map.get(ex_id)
            if pred is None:
                # Missing prediction = all false negatives
                for true_label in golden.true_labels:
                    stats[true_label.value].false_negatives += 1
                false_negatives.append(
                    FalseNegativeEntry(
                        example_id=ex_id,
                        text_preview=golden.text[:100],
                        missed_labels=golden.true_labels,
                        predicted_labels=[],
                        confidence=0.0,
                    )
                )
                continue

            # Filter by confidence threshold
            if pred.confidence < confidence_threshold:
                pred_labels: set[str] = set()
            else:
                pred_labels = {lb.value for lb in pred.predicted_labels}

            true_labels = {lb.value for lb in golden.true_labels}

            # Per-label scoring
            for label in all_labels:
                in_true = label in true_labels
                in_pred = label in pred_labels

                if in_true and in_pred:
                    stats[label].true_positives += 1
                elif in_pred and not in_true:
                    stats[label].false_positives += 1
                elif in_true and not in_pred:
                    stats[label].false_negatives += 1
                else:
                    stats[label].true_negatives += 1

            # Track false negatives for audit
            missed = true_labels - pred_labels
            if missed:
                false_negatives.append(
                    FalseNegativeEntry(
                        example_id=ex_id,
                        text_preview=golden.text[:100],
                        missed_labels=[IncidentType(lb) for lb in missed],
                        predicted_labels=pred.predicted_labels,
                        confidence=pred.confidence,
                    )
                )

            # Severity comparison
            if pred.predicted_severity == golden.true_severity:
                severity_correct += 1

        # Compute macro averages
        label_stats = {k: v for k, v in stats.items()}
        precisions = [
            s.precision for s in label_stats.values()
            if (s.true_positives + s.false_positives) > 0
        ]
        recalls = [
            s.recall for s in label_stats.values()
            if (s.true_positives + s.false_negatives) > 0
        ]

        macro_p = sum(precisions) / len(precisions) if precisions else 0.0
        macro_r = sum(recalls) / len(recalls) if recalls else 0.0
        macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

        total = len(self._golden)
        sev_acc = severity_correct / total if total > 0 else 0.0

        # Identify systematic failure modes
        failure_modes = self._identify_failure_modes(false_negatives, label_stats)

        return EvaluationResult(
            total_examples=total,
            per_label_stats=label_stats,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            severity_accuracy=sev_acc,
            false_negatives=false_negatives,
            failure_modes=failure_modes,
        )

    def _identify_failure_modes(
        self,
        false_negatives: list[FalseNegativeEntry],
        stats: dict[str, ConfusionStats],
    ) -> list[str]:
        """Heuristic identification of systematic error patterns."""
        modes: list[str] = []

        # Labels with recall below 0.5
        for label, s in stats.items():
            if s.recall < 0.5 and (s.true_positives + s.false_negatives) >= 2:
                modes.append(
                    f"Low recall on '{label}' ({s.recall:.2f}) -- "
                    f"model may be under-detecting this category"
                )

        # Labels with precision below 0.5
        for label, s in stats.items():
            if s.precision < 0.5 and (s.true_positives + s.false_positives) >= 2:
                modes.append(
                    f"Low precision on '{label}' ({s.precision:.2f}) -- "
                    f"model may be over-triggering on this category"
                )

        # High-confidence misses (confident but wrong)
        confident_misses = [fn for fn in false_negatives if fn.confidence > 0.7]
        if len(confident_misses) >= 2:
            modes.append(
                f"{len(confident_misses)} high-confidence false negatives -- "
                f"model is confident but missing labels, calibration issue"
            )

        return modes

    def evaluate_at_thresholds(
        self,
        predictions: list[PredictionEntry],
        thresholds: list[float] | None = None,
    ) -> dict[float, EvaluationResult]:
        """Evaluate at multiple confidence thresholds for threshold tuning."""
        thresholds = thresholds or [0.0, 0.3, 0.5, 0.7, 0.9]
        return {t: self.evaluate(predictions, confidence_threshold=t) for t in thresholds}
