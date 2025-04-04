"""Golden set evaluation example.

Demonstrates the validation workflow: compare classifier predictions
against a curated golden set, analyze failure modes, and generate
boosted prompt improvements.

This is the core feedback loop for improving classification quality:
1. Run predictions against golden set
2. Identify systematic errors
3. Inject corrective instructions into the prompt
4. Re-evaluate and measure improvement

Usage:
    python examples/golden_set_evaluation.py
"""

from crisis_lens.classification.classifier import CrisisClassifier
from crisis_lens.config import IncidentType
from crisis_lens.validation.golden_set import (
    GoldenExample,
    GoldenSetEvaluator,
    PredictionEntry,
)


def main() -> None:
    # Build a sample golden set
    golden_set = [
        GoldenExample(
            example_id="GS-001",
            text="Active shooter reported at shopping center, SWAT deployed",
            true_labels=[IncidentType.VIOLENT_EXTREMISM],
            true_severity="P0",
            notes="Clear active threat, no ambiguity",
        ),
        GoldenExample(
            example_id="GS-002",
            text="7.2 magnitude earthquake strikes off the coast, tsunami warning issued",
            true_labels=[IncidentType.NATURAL_DISASTER],
            true_severity="P1",
        ),
        GoldenExample(
            example_id="GS-003",
            text="Bot network detected: 5000 accounts posting identical content about election fraud",
            true_labels=[IncidentType.PLATFORM_MANIPULATION, IncidentType.MISINFORMATION],
            true_severity="P2",
        ),
        GoldenExample(
            example_id="GS-004",
            text="Satirical article: 'Local man survives earthquake while eating tacos'",
            true_labels=[],  # Not a real crisis
            true_severity="P4",
            notes="Satire -- common false positive trigger",
        ),
        GoldenExample(
            example_id="GS-005",
            text="User posted 'I can't take it anymore, ending it tonight' with self-harm hashtags",
            true_labels=[IncidentType.SELF_HARM],
            true_severity="P1",
        ),
        GoldenExample(
            example_id="GS-006",
            text="Leaked database with 2M user credentials posted on hacking forum",
            true_labels=[IncidentType.DATA_BREACH],
            true_severity="P0",
        ),
    ]

    evaluator = GoldenSetEvaluator(golden_set=golden_set)

    # Simulate predictions (in production, these come from the classifier)
    predictions = [
        PredictionEntry(
            example_id="GS-001",
            predicted_labels=[IncidentType.VIOLENT_EXTREMISM],
            predicted_severity="P0",
            confidence=0.95,
        ),
        PredictionEntry(
            example_id="GS-002",
            predicted_labels=[IncidentType.NATURAL_DISASTER],
            predicted_severity="P1",
            confidence=0.92,
        ),
        PredictionEntry(
            example_id="GS-003",
            predicted_labels=[IncidentType.PLATFORM_MANIPULATION],  # Missed misinformation label
            predicted_severity="P2",
            confidence=0.78,
        ),
        PredictionEntry(
            example_id="GS-004",
            predicted_labels=[IncidentType.NATURAL_DISASTER],  # False positive on satire
            predicted_severity="P3",
            confidence=0.45,
        ),
        PredictionEntry(
            example_id="GS-005",
            predicted_labels=[IncidentType.SELF_HARM],
            predicted_severity="P1",
            confidence=0.88,
        ),
        # GS-006 missing entirely -- simulates a gap in coverage
    ]

    # Evaluate at default threshold
    print("=" * 72)
    print("GOLDEN SET EVALUATION")
    print("=" * 72)
    result = evaluator.evaluate(predictions, confidence_threshold=0.5)
    print(result.summary())

    # Multi-threshold analysis
    print("\n" + "=" * 72)
    print("THRESHOLD ANALYSIS")
    print("=" * 72)
    threshold_results = evaluator.evaluate_at_thresholds(
        predictions, thresholds=[0.0, 0.3, 0.5, 0.7, 0.9]
    )
    for threshold, res in sorted(threshold_results.items()):
        print(f"  threshold={threshold:.1f}  P={res.macro_precision:.3f}  R={res.macro_recall:.3f}  F1={res.macro_f1:.3f}")

    # Feed failure modes into classifier boosted prompt
    print("\n" + "=" * 72)
    print("BOOSTED PROMPT REFINEMENT")
    print("=" * 72)
    classifier = CrisisClassifier()
    for fm in result.failure_modes:
        classifier.add_failure_mode(fm)
        print(f"  Added failure mode: {fm}")

    # Add specific calibration based on the false positives we found
    if any(fn.example_id == "GS-004" for fn in result.false_negatives):
        note = "Satirical content about disasters should receive confidence < 0.3"
        classifier.add_calibration_note(note)
        print(f"  Added calibration: {note}")

    print(f"\n  Boosted system prompt length: {len(classifier.system_prompt)} chars")
    print("  Ready for re-evaluation with refined prompt.")


if __name__ == "__main__":
    main()
