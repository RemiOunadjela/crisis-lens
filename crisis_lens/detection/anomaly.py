"""Statistical anomaly detection for crisis signal streams.

Implements lightweight online anomaly detection without heavy ML dependencies.
The approach uses exponential moving averages and z-score deviation -- enough
to catch volume spikes and velocity changes that indicate emerging incidents.

For production T&S, anomaly detection is the safety net that catches crises
that don't match any keyword or pattern rule. The 2022 Itaewon crowd crush
was an example where volume anomaly detection would have flagged the incident
before keyword rules activated.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnomalyWindow:
    """Sliding window statistics for a single metric."""

    window_size: int = 100
    _values: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _sum: float = 0.0
    _sum_sq: float = 0.0

    def __post_init__(self) -> None:
        self._values = deque(maxlen=self.window_size)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self._sum / self.count

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        variance = (self._sum_sq / self.count) - (self.mean**2)
        return math.sqrt(max(variance, 0.0))

    def push(self, value: float) -> None:
        if len(self._values) == self._values.maxlen:
            evicted = self._values[0]
            self._sum -= evicted
            self._sum_sq -= evicted**2
        self._values.append(value)
        self._sum += value
        self._sum_sq += value**2

    def z_score(self, value: float) -> float:
        if self.std == 0:
            return 0.0 if value == self.mean else float("inf")
        return (value - self.mean) / self.std


@dataclass
class VolumeAnomalyDetector:
    """Detects unusual spikes in signal volume over time buckets.

    Tracks the number of signals per time bucket and flags when the current
    bucket's count deviates significantly from the rolling average.
    """

    bucket_seconds: int = 60
    z_threshold: float = 2.5
    min_samples: int = 10
    _window: AnomalyWindow = field(default_factory=lambda: AnomalyWindow(window_size=60))
    _current_bucket: int = 0
    _current_count: int = 0

    def observe(self, timestamp: float) -> dict[str, Any] | None:
        """Record a signal timestamp and return anomaly info if detected."""
        bucket = int(timestamp) // self.bucket_seconds

        if bucket != self._current_bucket:
            result = self._flush_bucket()
            self._current_bucket = bucket
            self._current_count = 1
            return result

        self._current_count += 1
        return None

    def _flush_bucket(self) -> dict[str, Any] | None:
        if self._current_count == 0:
            return None

        count = float(self._current_count)

        if self._window.count >= self.min_samples:
            z = self._window.z_score(count)
            is_anomaly = z > self.z_threshold
        else:
            z = 0.0
            is_anomaly = False

        self._window.push(count)

        if is_anomaly:
            return {
                "bucket": self._current_bucket,
                "count": self._current_count,
                "z_score": round(z, 2),
                "mean": round(self._window.mean, 2),
                "std": round(self._window.std, 2),
            }
        return None


@dataclass
class VelocityAnomalyDetector:
    """Detects rapid acceleration in signal arrival rate.

    Useful for catching coordinated attacks where volume ramps quickly
    from baseline. Measures the rate-of-change of signal volume rather
    than absolute volume.
    """

    window_size: int = 30
    z_threshold: float = 3.0
    _deltas: AnomalyWindow = field(default_factory=lambda: AnomalyWindow(window_size=30))
    _last_count: float | None = None

    def observe(self, bucket_count: float) -> dict[str, Any] | None:
        if self._last_count is not None:
            delta = bucket_count - self._last_count

            if self._deltas.count >= 5:
                z = self._deltas.z_score(delta)
                if z > self.z_threshold:
                    self._deltas.push(delta)
                    self._last_count = bucket_count
                    return {
                        "delta": delta,
                        "z_score": round(z, 2),
                        "mean_delta": round(self._deltas.mean, 2),
                    }

            self._deltas.push(delta)

        self._last_count = bucket_count
        return None
