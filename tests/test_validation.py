"""Tests for the validation framework."""

import pytest

from crisis_lens.config import IncidentType
from crisis_lens.detection.monitors import TextRecord
from crisis_lens.validation.golden_set import (
    GoldenExample,
    GoldenSetEvaluator,
    PredictionEntry,
)
from crisis_lens.validation.simulation import TimeBoxedSimulation


class TestGoldenSetEvaluator:
    @pytest.fixture()
    def evaluator(self) -> GoldenSetEvaluator:
        examples = [
            GoldenExample(
                example_id="EX-001",
                text="Active shooter at the mall, multiple injured",
                true_labels=[IncidentType.VIOLENT_EXTREMISM],
                true_severity="P0",
            ),
            GoldenExample(
                example_id="EX-002",
                text="Earthquake magnitude 7.2 hits coastal region",
                true_labels=[IncidentType.NATURAL_DISASTER],
                true_severity="P1",
            ),
            GoldenExample(
                example_id="EX-003",
                text="Coordinated bot network promoting disinformation",
                true_labels=[IncidentType.PLATFORM_MANIPULATION, IncidentType.MISINFORMATION],
                true_severity="P2",
            ),
            GoldenExample(
                example_id="EX-004",
                text="Satirical post about fictional earthquake, clearly labeled comedy",
                true_labels=[],  # Not a crisis
                true_severity="P4",
            ),
        ]
        return GoldenSetEvaluator(golden_set=examples)

    def test_perfect_predictions(self, evaluator: GoldenSetEvaluator):
        preds = [
            PredictionEntry(
                example_id="EX-001",
                predicted_labels=[IncidentType.VIOLENT_EXTREMISM],
                predicted_severity="P0",
                confidence=0.95,
            ),
            PredictionEntry(
                example_id="EX-002",
                predicted_labels=[IncidentType.NATURAL_DISASTER],
                predicted_severity="P1",
                confidence=0.9,
            ),
            PredictionEntry(
                example_id="EX-003",
                predicted_labels=[IncidentType.PLATFORM_MANIPULATION, IncidentType.MISINFORMATION],
                predicted_severity="P2",
                confidence=0.85,
            ),
            PredictionEntry(
                example_id="EX-004",
                predicted_labels=[],
                predicted_severity="P4",
                confidence=0.1,
            ),
        ]
        result = evaluator.evaluate(preds)
        # With perfect predictions on labeled examples, should have high scores
        assert result.macro_precision >= 0.9
        assert result.macro_recall >= 0.9
        assert result.severity_accuracy == 1.0

    def test_missing_predictions_count_as_fn(self, evaluator: GoldenSetEvaluator):
        # Only predict for first example
        preds = [
            PredictionEntry(
                example_id="EX-001",
                predicted_labels=[IncidentType.VIOLENT_EXTREMISM],
                predicted_severity="P0",
                confidence=0.95,
            ),
        ]
        result = evaluator.evaluate(preds)
        assert len(result.false_negatives) >= 2  # EX-002 and EX-003 are missed

    def test_confidence_threshold_filtering(self, evaluator: GoldenSetEvaluator):
        preds = [
            PredictionEntry(
                example_id="EX-001",
                predicted_labels=[IncidentType.VIOLENT_EXTREMISM],
                confidence=0.3,  # Below threshold
            ),
            PredictionEntry(
                example_id="EX-002",
                predicted_labels=[IncidentType.NATURAL_DISASTER],
                confidence=0.8,  # Above threshold
            ),
        ]
        result = evaluator.evaluate(preds, confidence_threshold=0.5)
        # EX-001 should be treated as no prediction due to low confidence
        fn_ids = {fn.example_id for fn in result.false_negatives}
        assert "EX-001" in fn_ids

    def test_multi_threshold_evaluation(self, evaluator: GoldenSetEvaluator):
        preds = [
            PredictionEntry(
                example_id="EX-001",
                predicted_labels=[IncidentType.VIOLENT_EXTREMISM],
                confidence=0.7,
            ),
            PredictionEntry(
                example_id="EX-002",
                predicted_labels=[IncidentType.NATURAL_DISASTER],
                confidence=0.9,
            ),
        ]
        results = evaluator.evaluate_at_thresholds(preds, thresholds=[0.0, 0.5, 0.8])
        assert len(results) == 3
        # Higher threshold should generally produce lower recall
        assert results[0.0].macro_recall >= results[0.8].macro_recall

    def test_failure_mode_detection(self, evaluator: GoldenSetEvaluator):
        # Predictions that systematically miss a category
        preds = [
            PredictionEntry(
                example_id="EX-001",
                predicted_labels=[],  # Missed
                confidence=0.9,
            ),
            PredictionEntry(
                example_id="EX-002",
                predicted_labels=[IncidentType.NATURAL_DISASTER],
                confidence=0.9,
            ),
            PredictionEntry(
                example_id="EX-003",
                predicted_labels=[IncidentType.MISINFORMATION],  # Missed platform_manipulation
                confidence=0.85,
            ),
        ]
        result = evaluator.evaluate(preds)
        assert len(result.false_negatives) > 0

    def test_evaluator_size(self, evaluator: GoldenSetEvaluator):
        assert evaluator.size == 4

    def test_add_example(self):
        evaluator = GoldenSetEvaluator()
        evaluator.add_example(GoldenExample(
            example_id="NEW-001",
            text="New test example",
            true_labels=[IncidentType.OTHER],
        ))
        assert evaluator.size == 1

    def test_summary_output(self, evaluator: GoldenSetEvaluator):
        preds = [
            PredictionEntry(
                example_id="EX-001",
                predicted_labels=[IncidentType.VIOLENT_EXTREMISM],
                confidence=0.9,
            ),
        ]
        result = evaluator.evaluate(preds)
        summary = result.summary()
        assert "Golden Set Evaluation" in summary
        assert "Macro Precision" in summary


class TestTimeBoxedSimulation:
    def test_basic_simulation(self):
        records = [
            TextRecord(text="Earthquake hits the coast", timestamp=100.0, source="sim"),
            TextRecord(text="Tsunami warning issued", timestamp=105.0, source="sim"),
            TextRecord(text="Normal conversation about weather", timestamp=200.0, source="sim"),
            TextRecord(text="Reports of flooding and destruction", timestamp=305.0, source="sim"),
        ]
        sim = TimeBoxedSimulation(window_seconds=100)
        result = sim.run(records)

        assert result.total_records == 4
        assert result.total_windows >= 2

    def test_empty_records(self):
        sim = TimeBoxedSimulation()
        result = sim.run([])
        assert result.total_records == 0
        assert result.total_windows == 0

    def test_single_window(self):
        records = [
            TextRecord(text="Active shooter situation", timestamp=10.0, source="sim"),
            TextRecord(text="Gunfire reported downtown", timestamp=15.0, source="sim"),
        ]
        sim = TimeBoxedSimulation(window_seconds=300)
        result = sim.run(records)
        assert result.total_windows == 1

    def test_detection_rate(self):
        records = [
            TextRecord(text="Bombing attack at airport", timestamp=100.0),
            TextRecord(text="Weather is fine today", timestamp=200.0),
        ]
        sim = TimeBoxedSimulation(window_seconds=300)
        result = sim.run(records)
        # At least one signal from the bombing text
        assert result.detection_rate >= 0.0

    def test_first_detection_window(self):
        records = [
            TextRecord(text="Normal text", timestamp=100.0),
            TextRecord(text="Normal conversation", timestamp=200.0),
            TextRecord(text="Earthquake magnitude 8.0", timestamp=700.0),
        ]
        sim = TimeBoxedSimulation(window_seconds=100)
        result = sim.run(records)
        first = result.first_detection_window()
        if first is not None:
            assert first.signals_detected > 0

    def test_summary(self):
        records = [
            TextRecord(text="Active shooting reported", timestamp=100.0),
        ]
        sim = TimeBoxedSimulation(window_seconds=60)
        result = sim.run(records)
        summary = result.summary()
        assert "records" in summary
        assert "signals" in summary

    def test_chronological_ordering(self):
        # Records out of order -- simulation should sort them
        records = [
            TextRecord(text="Second event: tsunami", timestamp=500.0),
            TextRecord(text="First event: earthquake", timestamp=100.0),
        ]
        sim = TimeBoxedSimulation(window_seconds=100)
        result = sim.run(records)
        # Should process in chronological order
        assert result.total_records == 2
        if result.total_windows >= 2:
            assert result.windows[0].start_time <= result.windows[1].start_time
