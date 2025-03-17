"""Situation Report (SITREP) generation.

Auto-generates structured incident reports following the format used by
T&S operations teams. SITREPs are the primary artifact for cross-functional
communication during active incidents -- they go to legal, comms, policy,
and engineering, so clarity and structure matter more than detail.

The format is intentionally rigid: incident responders under pressure need
to find information in predictable locations, not parse free-form narratives.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from crisis_lens.classification.classifier import ClassificationResult
from crisis_lens.config import IncidentType, Severity
from crisis_lens.detection.rules import Signal


class IncidentMetrics(BaseModel):
    """Quantitative summary of incident scope."""

    signal_count: int = 0
    unique_sources: int = 0
    languages_affected: list[str] = Field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    peak_velocity: float = 0.0  # signals per minute at peak


class RecommendedAction(BaseModel):
    action: str
    priority: str  # "immediate" | "short_term" | "monitoring"
    owner: str = "on-call"  # default assignment


class SitRep(BaseModel):
    """Structured Situation Report for a crisis incident."""

    incident_id: str
    title: str
    severity: Severity
    status: str = "active"  # active | monitoring | resolved | false_positive
    incident_types: list[IncidentType] = Field(default_factory=list)
    summary: str = ""
    metrics: IncidentMetrics = Field(default_factory=IncidentMetrics)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    signals: list[Signal] = Field(default_factory=list)
    classifications: list[ClassificationResult] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = ""

    def render_markdown(self, include_signals: bool = False) -> str:
        lines = [
            f"# SITREP: {self.title}",
            "",
            f"**Incident ID:** {self.incident_id}",
            f"**Severity:** {self.severity.value}",
            f"**Status:** {self.status}",
            f"**Types:** {', '.join(t.value for t in self.incident_types)}",
            f"**Created:** {self.created_at}",
            "",
            "## Summary",
            "",
            self.summary or "_No summary available._",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Signals detected | {self.metrics.signal_count} |",
            f"| Unique sources | {self.metrics.unique_sources} |",
            f"| Languages | {', '.join(self.metrics.languages_affected) or 'N/A'} |",
            f"| First seen | {self.metrics.first_seen or 'N/A'} |",
            f"| Last seen | {self.metrics.last_seen or 'N/A'} |",
            f"| Peak velocity | {self.metrics.peak_velocity:.1f} signals/min |",
            "",
            "## Recommended Actions",
            "",
        ]

        if self.recommended_actions:
            for i, action in enumerate(self.recommended_actions, 1):
                lines.append(
                    f"{i}. **[{action.priority.upper()}]** "
                    f"{action.action} (Owner: {action.owner})"
                )
        else:
            lines.append("_No actions recommended._")

        if include_signals and self.signals:
            lines.extend(["", "## Raw Signals", ""])
            for sig in self.signals[:50]:
                lines.append(f"- `{sig.signal_id}` [{sig.severity.value}] {sig.text[:120]}...")

        lines.append("")
        return "\n".join(lines)


ACTION_TEMPLATES: dict[IncidentType, list[RecommendedAction]] = {
    IncidentType.CHILD_SAFETY: [
        RecommendedAction(
            action="Escalate to NCMEC reporting pipeline",
            priority="immediate", owner="child-safety-team",
        ),
        RecommendedAction(
            action="Freeze implicated accounts pending review",
            priority="immediate", owner="on-call",
        ),
        RecommendedAction(
            action="Engage legal for preservation requests",
            priority="short_term", owner="legal",
        ),
    ],
    IncidentType.VIOLENT_EXTREMISM: [
        RecommendedAction(
            action="Activate counter-terrorism playbook",
            priority="immediate", owner="on-call",
        ),
        RecommendedAction(
            action="Coordinate with law enforcement liaison",
            priority="immediate", owner="law-enforcement-ops",
        ),
        RecommendedAction(
            action="Deploy emergency content removal queue",
            priority="immediate", owner="content-ops",
        ),
    ],
    IncidentType.NATURAL_DISASTER: [
        RecommendedAction(
            action="Activate crisis response hub for affected region",
            priority="immediate", owner="crisis-response",
        ),
        RecommendedAction(
            action="Enable SOS features in affected geographies",
            priority="short_term", owner="product",
        ),
        RecommendedAction(
            action="Monitor for scam/fraud exploiting disaster",
            priority="monitoring", owner="on-call",
        ),
    ],
    IncidentType.PLATFORM_MANIPULATION: [
        RecommendedAction(
            action="Snapshot network graph of involved accounts",
            priority="immediate", owner="integrity",
        ),
        RecommendedAction(
            action="Rate-limit flagged account cluster",
            priority="short_term", owner="anti-abuse",
        ),
        RecommendedAction(
            action="Draft transparency report entry",
            priority="monitoring", owner="policy",
        ),
    ],
    IncidentType.COORDINATED_HARASSMENT: [
        RecommendedAction(
            action="Enable enhanced protections for targeted users",
            priority="immediate", owner="user-safety",
        ),
        RecommendedAction(
            action="Identify and action coordinating accounts",
            priority="short_term", owner="on-call",
        ),
        RecommendedAction(
            action="Notify comms team for potential media inquiry",
            priority="monitoring", owner="comms",
        ),
    ],
    IncidentType.SELF_HARM: [
        RecommendedAction(
            action="Surface crisis resources to affected users",
            priority="immediate", owner="well-being",
        ),
        RecommendedAction(
            action="Review and enforce self-harm content policies",
            priority="short_term", owner="content-ops",
        ),
    ],
}


class SitRepGenerator:
    """Builds SITREPs from collected signals and classifications."""

    def __init__(self, max_signals: int = 50):
        self.max_signals = max_signals

    def generate(
        self,
        incident_id: str,
        signals: list[Signal],
        classifications: list[ClassificationResult] | None = None,
        title: str | None = None,
    ) -> SitRep:
        if not signals:
            return SitRep(
                incident_id=incident_id,
                title=title or f"Incident {incident_id}",
                severity=Severity.P4,
                summary="No signals collected for this incident.",
            )

        classifications = classifications or []

        # Determine severity from highest-scored signal
        max_severity = max(signals, key=lambda s: s.score).severity

        # Collect incident types
        all_types: list[IncidentType] = []
        for sig in signals:
            for t in sig.suggested_types:
                if t not in all_types:
                    all_types.append(t)
        for cls_result in classifications:
            for label in cls_result.labels:
                if label.incident_type not in all_types:
                    all_types.append(label.incident_type)

        # Build metrics
        sources = {s.source for s in signals}
        languages = list({s.language for s in signals})
        metrics = IncidentMetrics(
            signal_count=len(signals),
            unique_sources=len(sources),
            languages_affected=languages,
        )

        # Auto-generate title from types if not provided
        if not title:
            type_labels = [t.value.replace("_", " ").title() for t in all_types[:3]]
            title = f"{', '.join(type_labels)} Incident" if type_labels else f"Incident {incident_id}"

        # Collect recommended actions based on types
        actions: list[RecommendedAction] = []
        for t in all_types:
            template_actions = ACTION_TEMPLATES.get(t, [])
            for action in template_actions:
                if action not in actions:
                    actions.append(action)

        # Build summary
        summary_parts = [
            f"Detected {len(signals)} signals across {len(sources)} source(s).",
            f"Languages: {', '.join(languages)}.",
            f"Incident types: {', '.join(t.value for t in all_types)}.",
        ]
        if any(s.severity in (Severity.P0, Severity.P1) for s in signals):
            summary_parts.append("HIGH SEVERITY signals detected -- immediate review required.")

        return SitRep(
            incident_id=incident_id,
            title=title,
            severity=max_severity,
            incident_types=all_types,
            summary=" ".join(summary_parts),
            metrics=metrics,
            recommended_actions=actions,
            signals=signals[: self.max_signals],
            classifications=classifications,
        )
