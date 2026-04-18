"""Tests for the CLI layer."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from crisis_lens.cli import main


def _write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestMonitorDryRun:
    def test_dry_run_prints_summary(self, tmp_path: Path):
        source = tmp_path / "signals.jsonl"
        _write_jsonl(
            [
                {"text": "Active shooter and bombing at the plaza, multiple casualties", "source": "test", "language": "en"},
                {"text": "Earthquake devastates coastal city, tsunami warning issued", "source": "test", "language": "en"},
                {"text": "Regular weather update for the weekend", "source": "test", "language": "en"},
            ],
            source,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["monitor", "--source", str(source), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "dry run" in result.output.lower()
        assert "records scanned" in result.output.lower()
        assert "signals that would be emitted" in result.output.lower()

    def test_dry_run_does_not_write_output_file(self, tmp_path: Path):
        source = tmp_path / "signals.jsonl"
        output = tmp_path / "out.jsonl"
        _write_jsonl(
            [{"text": "Suicide bombing attack kills dozens near the parliament", "source": "test", "language": "en"}],
            source,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["monitor", "--source", str(source), "--output", str(output), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert not output.exists(), "dry-run must not write the output file"
        assert output.name in result.output

    def test_dry_run_empty_source(self, tmp_path: Path):
        source = tmp_path / "empty.jsonl"
        source.write_text("")

        runner = CliRunner()
        result = runner.invoke(main, ["monitor", "--source", str(source), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "records scanned" in result.output.lower()

    def test_monitor_without_dry_run_writes_output(self, tmp_path: Path):
        source = tmp_path / "signals.jsonl"
        output = tmp_path / "out.jsonl"
        _write_jsonl(
            [{"text": "Suicide bombing attack kills dozens near the parliament", "source": "test", "language": "en"}],
            source,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["monitor", "--source", str(source), "--output", str(output)])

        assert result.exit_code == 0, result.output
        assert output.exists(), "non-dry-run should write the output file"
