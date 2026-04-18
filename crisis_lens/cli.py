"""Command-line interface for crisis-lens.

Exposes the core pipeline operations as CLI commands:
  - monitor: real-time signal detection from file sources
  - classify: LLM-based classification of text inputs
  - validate: golden set evaluation
  - report: generate SITREPs and handover summaries
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from crisis_lens import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="crisis-lens")
def main() -> None:
    """crisis-lens: AI-Powered Crisis Detection for Trust & Safety."""
    pass


@main.command()
@click.option("--source", required=True, type=click.Path(exists=True), help="JSONL file to monitor")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Pipeline config YAML")
@click.option("--min-severity", default="P4", type=click.Choice(["P0", "P1", "P2", "P3", "P4"]))
@click.option("--output", "-o", type=click.Path(), help="Output file for signals (JSONL)")
@click.option("--dry-run", is_flag=True, default=False, help="Parse and detect without writing output; print a summary instead")
def monitor(source: str, config_path: str | None, min_severity: str, output: str | None, dry_run: bool) -> None:
    """Monitor a text stream for crisis signals."""
    from crisis_lens.config import PipelineConfig, Severity
    from crisis_lens.pipeline import CrisisDetectionPipeline

    config = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    pipeline = CrisisDetectionPipeline(config=config)
    severity_filter = Severity(min_severity)
    severity_order = list(Severity)

    severity_colors = {
        "P0": "red bold", "P1": "red",
        "P2": "yellow", "P3": "blue",
    }

    async def _run() -> None:
        if dry_run:
            console.print("[dim]-- dry run: detection only, no output written --[/dim]")
            counts: dict[str, int] = {s.value: 0 for s in Severity}
            total_records = 0
            matching_signals = []

            async for signal in pipeline.stream_signals(source):
                total_records += 1
                if severity_order.index(signal.severity) <= severity_order.index(severity_filter):
                    counts[signal.severity.value] += 1
                    matching_signals.append(signal)

            table = Table(title="Dry Run Summary", show_header=True)
            table.add_column("Severity", style="bold")
            table.add_column("Signals", justify="right")
            for sev_value, cnt in counts.items():
                color = severity_colors.get(sev_value, "dim")
                table.add_row(f"[{color}]{sev_value}[/{color}]", str(cnt))
            console.print(table)

            total_matching = sum(counts.values())
            console.print(
                f"\n[dim]records scanned:[/dim] {total_records}  "
                f"[dim]signals that would be emitted:[/dim] {total_matching}"
            )
            if output:
                console.print(f"[dim]output would have been written to:[/dim] {output}")
            return

        out_file = open(output, "w") if output else None
        try:
            count = 0
            async for signal in pipeline.stream_signals(source):
                if severity_order.index(signal.severity) <= severity_order.index(severity_filter):
                    count += 1
                    line = json.dumps({
                        "signal_id": signal.signal_id,
                        "severity": signal.severity.value,
                        "score": round(signal.score, 3),
                        "types": [t.value for t in signal.suggested_types],
                        "rules": signal.matched_rules,
                        "text": signal.text[:200],
                    })
                    if out_file:
                        out_file.write(line + "\n")
                    else:
                        sev = signal.severity.value
                        color = severity_colors.get(sev, "dim")
                        text_preview = signal.text[:100]
                        console.print(
                            f"[{color}][{sev}][/{color}] "
                            f"{signal.signal_id} | {text_preview}"
                        )
            console.print(f"\nProcessed. {count} signals detected.", style="green")
        finally:
            if out_file:
                out_file.close()

    asyncio.run(_run())


@main.command()
@click.option(
    "--input", "input_path", required=True,
    type=click.Path(exists=True), help="JSONL file with texts",
)
@click.option("--model", default="gpt-4o-mini", help="LLM model identifier")
@click.option("--provider", default="openai", type=click.Choice(["openai", "huggingface"]))
@click.option("--output", "-o", type=click.Path(), help="Output file for classifications")
def classify(input_path: str, model: str, provider: str, output: str | None) -> None:
    """Classify text inputs using an LLM provider."""
    from crisis_lens.classification.classifier import CrisisClassifier
    from crisis_lens.classification.providers import create_provider
    from crisis_lens.config import ClassificationConfig, LLMProviderConfig
    from crisis_lens.detection.rules import Signal

    llm_config = LLMProviderConfig(provider=provider, model=model)
    cls_config = ClassificationConfig(llm=llm_config)
    llm_provider = create_provider(llm_config)
    classifier = CrisisClassifier(provider=llm_provider, config=cls_config)

    async def _run() -> None:
        results = []
        with open(input_path) as f:
            for line in f:
                data = json.loads(line.strip())
                signal = Signal(
                    signal_id=data.get("signal_id", "UNKNOWN"),
                    text=data.get("text", ""),
                    source=data.get("source", "cli"),
                    language=data.get("language", "en"),
                    score=data.get("score", 0.5),
                )
                result = await classifier.classify(signal)
                results.append(result)
                label_types = [lb.incident_type.value for lb in result.labels]
                console.print(
                    f"  {result.signal_id}: "
                    f"{result.severity.value} -> {label_types}"
                )

        if output:
            with open(output, "w") as f:
                for r in results:
                    f.write(r.model_dump_json() + "\n")
            console.print(f"Results written to {output}", style="green")

    asyncio.run(_run())


@main.command()
@click.option(
    "--golden-set", required=True,
    type=click.Path(exists=True), help="Golden set JSON file",
)
@click.option(
    "--predictions", required=True,
    type=click.Path(exists=True), help="Predictions JSONL file",
)
@click.option("--threshold", default=0.5, type=float, help="Confidence threshold")
def validate(golden_set: str, predictions: str, threshold: float) -> None:
    """Evaluate predictions against a golden set."""
    from crisis_lens.config import IncidentType
    from crisis_lens.validation.golden_set import GoldenSetEvaluator, PredictionEntry

    evaluator = GoldenSetEvaluator.from_file(golden_set)

    preds: list[PredictionEntry] = []
    with open(predictions) as f:
        for line in f:
            data = json.loads(line.strip())
            preds.append(PredictionEntry(
                example_id=data["example_id"],
                predicted_labels=[IncidentType(v) for v in data.get("predicted_labels", [])],
                predicted_severity=data.get("predicted_severity", "P4"),
                confidence=data.get("confidence", 0.0),
            ))

    result = evaluator.evaluate(preds, confidence_threshold=threshold)

    table = Table(title="Golden Set Evaluation")
    table.add_column("Label", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    for label, stats in sorted(result.per_label_stats.items()):
        table.add_row(label, f"{stats.precision:.3f}", f"{stats.recall:.3f}", f"{stats.f1:.3f}")

    console.print(table)
    console.print(f"\nMacro P={result.macro_precision:.3f}  R={result.macro_recall:.3f}  F1={result.macro_f1:.3f}")
    console.print(f"Severity accuracy: {result.severity_accuracy:.3f}")

    if result.failure_modes:
        console.print("\n[yellow]Failure modes:[/yellow]")
        for fm in result.failure_modes:
            console.print(f"  - {fm}")


@main.command()
@click.option("--incident-id", required=True, help="Incident identifier")
@click.option("--signals", type=click.Path(exists=True), help="Signals JSONL file")
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "json"]))
@click.option("--output", "-o", type=click.Path(), help="Output file")
def report(incident_id: str, signals: str | None, fmt: str, output: str | None) -> None:
    """Generate a situation report for an incident."""
    from crisis_lens.detection.rules import Signal
    from crisis_lens.reports.sitrep import SitRepGenerator

    signal_list: list[Signal] = []
    if signals:
        with open(signals) as f:
            for line in f:
                data = json.loads(line.strip())
                signal_list.append(Signal(**data))

    generator = SitRepGenerator()
    sitrep = generator.generate(incident_id=incident_id, signals=signal_list)

    if fmt == "markdown":
        content = sitrep.render_markdown(include_signals=True)
    else:
        content = sitrep.model_dump_json(indent=2)

    if output:
        Path(output).write_text(content)
        console.print(f"Report written to {output}", style="green")
    else:
        console.print(content)


if __name__ == "__main__":
    main()
