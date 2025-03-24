"""Time-boxed simulation for backtesting detection pipelines.

Backtesting crisis detection is tricky because of lookahead bias: if you
evaluate a model on historical data without respecting temporal ordering,
you'll overestimate performance. A classifier that sees a crisis unfold
over 24 hours and then classifies the earliest signals correctly isn't
actually performing well -- it's cheating.

TimeBoxedSimulation replays historical data in chronological order with
configurable time windows, ensuring the model only sees data available
at each decision point.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crisis_lens.detection.monitors import TextRecord
from crisis_lens.detection.rules import RuleEngine, Signal


class SimulationWindow(BaseModel):
    """A single evaluation window in the simulation."""

    window_id: int
    start_time: float
    end_time: float
    records_count: int = 0
    signals_detected: int = 0
    signals: list[Signal] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Aggregated results from a time-boxed simulation run."""

    total_records: int = 0
    total_signals: int = 0
    total_windows: int = 0
    windows: list[SimulationWindow] = Field(default_factory=list)
    detection_timeline: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.total_signals / self.total_records

    def first_detection_window(self) -> SimulationWindow | None:
        """Return the first window where a signal was detected."""
        for w in self.windows:
            if w.signals_detected > 0:
                return w
        return None

    def summary(self) -> str:
        first = self.first_detection_window()
        first_str = f"window {first.window_id}" if first else "none"
        return (
            f"Simulation: {self.total_records} records, "
            f"{self.total_signals} signals, "
            f"{self.total_windows} windows, "
            f"detection rate {self.detection_rate:.3f}, "
            f"first detection at {first_str}"
        )


class TimeBoxedSimulation:
    """Replays historical data in time-ordered windows.

    Prevents lookahead bias by ensuring the detection engine only
    processes data within each window before evaluation.
    """

    def __init__(
        self,
        window_seconds: int = 300,
        engine: RuleEngine | None = None,
    ):
        self.window_seconds = window_seconds
        self.engine = engine or RuleEngine()

    def run(self, records: list[TextRecord]) -> SimulationResult:
        """Execute simulation on a list of time-ordered records."""
        if not records:
            return SimulationResult()

        # Sort by timestamp
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        windows: list[SimulationWindow] = []
        result = SimulationResult()

        current_start = sorted_records[0].timestamp
        window_id = 0
        window_records: list[TextRecord] = []

        for record in sorted_records:
            if record.timestamp >= current_start + self.window_seconds:
                # Process completed window
                w = self._process_window(window_id, current_start, window_records)
                windows.append(w)
                result.total_signals += w.signals_detected
                result.total_records += w.records_count

                # Advance window
                window_id += 1
                current_start = record.timestamp
                window_records = []

                # Reset dedup cache per window to simulate fresh state
                self.engine.clear_dedup_cache()

            window_records.append(record)

        # Process final window
        if window_records:
            w = self._process_window(window_id, current_start, window_records)
            windows.append(w)
            result.total_signals += w.signals_detected
            result.total_records += w.records_count

        result.windows = windows
        result.total_windows = len(windows)

        # Build detection timeline
        result.detection_timeline = [
            {
                "window": w.window_id,
                "start": w.start_time,
                "records": w.records_count,
                "signals": w.signals_detected,
            }
            for w in windows
        ]

        return result

    def _process_window(
        self,
        window_id: int,
        start_time: float,
        records: list[TextRecord],
    ) -> SimulationWindow:
        signals: list[Signal] = []
        for record in records:
            sig = self.engine.evaluate(
                text=record.text,
                language=record.language,
                source=record.source,
            )
            if sig is not None:
                signals.append(sig)

        end_time = records[-1].timestamp if records else start_time

        return SimulationWindow(
            window_id=window_id,
            start_time=start_time,
            end_time=end_time,
            records_count=len(records),
            signals_detected=len(signals),
            signals=signals,
        )
