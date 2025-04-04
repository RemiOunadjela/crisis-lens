"""Basic monitoring example.

Demonstrates how to set up a crisis detection pipeline that processes
a JSONL file of text records and outputs detected signals with severity
scores. This mirrors the typical setup for a T&S team evaluating the
tool against historical escalation data.

Usage:
    python examples/basic_monitoring.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

from crisis_lens.config import PipelineConfig
from crisis_lens.detection.monitors import TextRecord
from crisis_lens.pipeline import CrisisDetectionPipeline


SAMPLE_RECORDS = [
    {
        "text": "Breaking: 6.8 magnitude earthquake hits central Turkey, buildings collapsed in multiple provinces",
        "source": "news_feed",
        "language": "en",
    },
    {
        "text": "Users reporting coordinated harassment campaign targeting journalist @example_user, thousands of replies in minutes",
        "source": "platform_signals",
        "language": "en",
    },
    {
        "text": "Just had a great lunch at the new restaurant downtown, highly recommend!",
        "source": "social",
        "language": "en",
    },
    {
        "text": "Alerta: incendio forestal fuera de control en la sierra, evacuaciones en curso",
        "source": "news_feed",
        "language": "es",
    },
    {
        "text": "Network of bot accounts detected amplifying election misinformation, coordinated posting pattern confirmed",
        "source": "integrity_signals",
        "language": "en",
    },
    {
        "text": "Terremoto de magnitude 7.1 atinge a costa, alerta de tsunami emitido",
        "source": "news_feed",
        "language": "pt",
    },
]


async def main() -> None:
    # Create a temporary JSONL file with sample data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for record in SAMPLE_RECORDS:
            f.write(json.dumps(record) + "\n")
        data_path = f.name

    try:
        config = PipelineConfig.default()
        pipeline = CrisisDetectionPipeline(config=config)

        print("Processing sample records...")
        print("-" * 72)

        result = await pipeline.process_file(data_path)

        for signal in result.signals:
            types = ", ".join(t.value for t in signal.suggested_types)
            print(
                f"  [{signal.severity.value}] {signal.signal_id} "
                f"(score={signal.score:.2f}, types=[{types}])"
            )
            print(f"         {signal.text[:80]}...")
            print()

        print("-" * 72)
        summary = result.summary()
        print(f"Total signals: {summary['signals_detected']}")
        print(f"Escalations (P0/P1): {summary['escalations']}")

        # Generate SITREP if there are signals
        if result.sitreps:
            print("\n" + "=" * 72)
            print(result.sitreps[0].render_markdown())

    finally:
        Path(data_path).unlink()


if __name__ == "__main__":
    asyncio.run(main())
