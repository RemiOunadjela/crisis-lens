# crisis-lens

[![CI](https://github.com/RemiOunadjela/crisis-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/RemiOunadjela/crisis-lens/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AI-Powered Crisis Detection for Trust & Safety**

A system for detecting, classifying, and triaging emerging crises from text streams. Built for Trust & Safety teams managing content safety at scale.

---

## Why crisis-lens?

Trust & Safety teams at large platforms handle thousands of escalations daily. When a real crisis hits -- an active threat, a coordinated harassment campaign, a natural disaster driving platform abuse -- the difference between a 10-minute and a 2-hour response can determine whether the incident stays contained or goes viral.

The current state of crisis detection at most organizations:

- **Keyword dashboards** that generate noise but miss novel threats
- **Manual triage** by on-call analysts who burn out scanning queues
- **No structured feedback loop** between incident outcomes and detection rules
- **Shift handovers** that rely on Slack messages and tribal knowledge

crisis-lens addresses these gaps with a pipeline that combines rule-based detection, LLM-powered classification, and a validation framework that systematically improves accuracy over time.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              crisis-lens pipeline            │
                    │                                             │
  Text Streams ───►│  ┌──────────┐  ┌────────────┐  ┌─────────┐ │
  (JSONL, API)     │  │ Detection │─►│Classification│─►│ Triage  │ │
                    │  │          │  │  (LLM)     │  │         │ │
                    │  │ Keywords │  │  OpenAI    │  │ P0-P4   │ │
                    │  │ Patterns │  │  HuggingFace│  │ Routing │ │
                    │  │ Anomaly  │  │  vLLM      │  │         │ │
                    │  └──────────┘  └────────────┘  └────┬────┘ │
                    │                                     │      │
                    │       ┌──────────────────────────────┘      │
                    │       ▼                                     │
                    │  ┌──────────┐  ┌──────────────┐            │
                    │  │  SITREP  │  │   Handover   │            │
                    │  │ Reports  │  │  Summaries   │            │
                    │  └──────────┘  └──────────────┘            │
                    └─────────────────────────────────────────────┘
                              ▲                  │
                              │    Feedback       │
                    ┌─────────┴──────────────────▼────────────────┐
                    │          Validation Framework                │
                    │                                             │
                    │  Golden Set ──► Evaluate ──► Failure Modes  │
                    │                    │                        │
                    │                    ▼                        │
                    │            Boosted Prompts                  │
                    │       (Iterative Refinement)                │
                    └─────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install crisis-lens
```

For development:

```bash
git clone https://github.com/RemiOunadjela/crisis-lens.git
cd crisis-lens
pip install -e ".[dev]"
```

### Monitor a text stream

```bash
# Process a JSONL file of text records
crisis-lens monitor --source data/signals.jsonl --min-severity P2

# With custom config
crisis-lens monitor --source data/signals.jsonl --config configs/default.yaml -o detected.jsonl
```

### Classify signals with an LLM

```bash
# Using OpenAI
export OPENAI_API_KEY=sk-...
crisis-lens classify --input signals.jsonl --model gpt-4o-mini

# Using a HuggingFace model
export HF_API_KEY=hf_...
crisis-lens classify --input signals.jsonl --provider huggingface --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

### Evaluate against a golden set

```bash
crisis-lens validate --golden-set golden.json --predictions preds.jsonl --threshold 0.5
```

### Generate a situation report

```bash
crisis-lens report --incident-id INC-2024-0042 --signals detected.jsonl --format markdown
```

### Python API

```python
import asyncio
from crisis_lens.pipeline import CrisisDetectionPipeline
from crisis_lens.detection.monitors import TextRecord

pipeline = CrisisDetectionPipeline()

records = [
    TextRecord(
        text="Breaking: 6.8 magnitude earthquake hits central Turkey",
        source="news_feed",
        language="en",
    ),
]

result = asyncio.run(pipeline.process_records(records))

for signal in result.signals:
    print(f"[{signal.severity.value}] {signal.signal_id}: {signal.text[:80]}")
```

## Core Components

### Detection

The detection layer uses three complementary strategies:

- **Keyword rules** -- multilingual term matching for known crisis vocabulary (EN, ES, PT). Fast and deterministic, these catch the majority of known threat categories.
- **Pattern rules** -- regex-based matching for structural indicators like URL flooding, coordinated posting patterns, and known bad-actor naming conventions.
- **Anomaly detection** -- online statistical detection (z-score on volume/velocity) that catches emerging incidents before keywords activate. This is the safety net for unknown-unknowns.

Signals are scored 0-1 and mapped to severity tiers (P0-P4) using configurable thresholds.

### Classification

LLM-powered multi-label classification with support for:

- **OpenAI-compatible APIs** (OpenAI, Azure, vLLM, Ollama)
- **HuggingFace Inference API** (hosted models, inference endpoints)

The classifier handles 10 incident types that cover the standard T&S taxonomy: violent extremism, natural disasters, platform manipulation, coordinated harassment, child safety, self-harm, misinformation, data breaches, regulatory events, and political unrest.

### The Boosted Prompt Methodology

Standard prompt engineering is trial-and-error. Boosted prompts formalize the improvement loop:

1. **Baseline evaluation** -- Run the classifier against a golden set of labeled examples.
2. **Failure mode analysis** -- Automatically identify systematic errors: categories with low recall, high-confidence false positives, calibration drift.
3. **Prompt injection** -- Feed failure modes and calibration notes directly into the system prompt as corrective instructions.
4. **Re-evaluation** -- Measure improvement on the same golden set. Iterate until convergence.

This is analogous to boosting in ensemble methods -- each iteration targets the residual errors of the previous round. In practice, 2-3 iterations typically capture the major failure modes.

```python
from crisis_lens.classification.classifier import CrisisClassifier
from crisis_lens.validation.golden_set import GoldenSetEvaluator

classifier = CrisisClassifier()
evaluator = GoldenSetEvaluator.from_file("golden_set.json")

# Iteration 1: baseline
result = evaluator.evaluate(predictions)
for failure_mode in result.failure_modes:
    classifier.add_failure_mode(failure_mode)

# Iteration 2: re-classify with boosted prompt
# ... re-run predictions with classifier.classify_batch(signals)
# ... re-evaluate and add any new failure modes
```

### Golden Set Validation

The golden set is the single source of truth for classifier quality. It should include:

- **Clear positives** across all incident types
- **Ambiguous cases** (satire about violence, news reporting vs. direct threats)
- **Negative examples** (benign content that resembles crisis language)
- **Multilingual examples** reflecting your platform's language distribution

The evaluation framework computes per-label precision, recall, and F1 at configurable confidence thresholds, plus a false-negative audit that highlights exactly which examples the classifier is missing and why.

Why this matters: most T&S teams tune classifiers on aggregate metrics and miss that they've silently degraded on a specific category. The per-label breakdown prevents this.

### Time-Boxed Simulation

Backtesting detection pipelines requires care to avoid lookahead bias. The `TimeBoxedSimulation` replays historical data in chronological windows, ensuring the detection engine only sees data that would have been available at each decision point.

This is critical for honest evaluation: a system that classifies early signals correctly only because it saw the full incident arc is useless for real-time detection.

### Reports

Two output formats designed for T&S operations:

- **SITREPs** -- structured situation reports with severity, scope metrics, and recommended actions based on incident type. These go to cross-functional stakeholders (legal, comms, policy, engineering).
- **Handover summaries** -- shift transition documents capturing active incidents, actions taken, pending follow-ups, and emerging signals. Designed for 24/7 operations where context loss between shifts causes response delays.

## Configuration

Pipeline behavior is controlled via YAML configuration:

```yaml
detection:
  languages: [en, es, pt]
  batch_size: 64
  min_confidence: 0.3
  thresholds:
    p0_threshold: 0.95
    p1_threshold: 0.85
    p2_threshold: 0.70
    p3_threshold: 0.50

classification:
  llm:
    provider: openai
    model: gpt-4o-mini
    temperature: 0.1
  confidence_threshold: 0.6
  max_labels: 3
```

See `configs/default.yaml` for the full configuration reference.

## Development

```bash
git clone https://github.com/RemiOunadjela/crisis-lens.git
cd crisis-lens
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check crisis_lens/ tests/

# Type check
mypy crisis_lens/ --ignore-missing-imports
```

## Related Projects

- **[safetybench](https://github.com/RemiOunadjela/safetybench)** -- Benchmarking framework for evaluating content moderation models against T&S-specific metrics.
- **[metric-guard](https://github.com/RemiOunadjela/metric-guard)** -- Data quality monitoring for compliance and regulatory metrics pipelines.
- **[transparency-engine](https://github.com/RemiOunadjela/transparency-engine)** -- Regulatory-compliant transparency report generation across DSA, OSA, and custom frameworks.

## License

MIT
