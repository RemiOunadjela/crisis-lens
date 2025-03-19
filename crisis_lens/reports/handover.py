"""Shift handover summary generation.

In 24/7 T&S operations, shift handovers are where incidents get dropped.
An outgoing analyst might be tracking three developing situations, and if
the incoming analyst doesn't get crisp context, response time degrades.

This module auto-generates structured handover documents that capture:
- Active incidents and their current status
- Escalation decisions made during the shift
- Emerging signals that haven't been escalated yet
- Outstanding actions waiting for resolution
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from crisis_lens.config import Severity
from crisis_lens.reports.sitrep import SitRep


class ShiftAction(BaseModel):
    """An action taken or pending during a shift."""

    description: str
    status: str  # "completed" | "pending" | "escalated" | "deferred"
    owner: str = ""
    notes: str = ""


class HandoverSummary(BaseModel):
    """Structured handover document for shift transitions."""

    shift_start: str
    shift_end: str
    analyst: str = ""
    active_incidents: list[SitRep] = Field(default_factory=list)
    resolved_incidents: list[str] = Field(default_factory=list)
    actions_taken: list[ShiftAction] = Field(default_factory=list)
    pending_actions: list[ShiftAction] = Field(default_factory=list)
    emerging_signals_summary: str = ""
    notes: str = ""

    def render_markdown(self) -> str:
        lines = [
            "# Shift Handover Summary",
            "",
            f"**Shift:** {self.shift_start} -- {self.shift_end}",
            f"**Analyst:** {self.analyst or 'N/A'}",
            "",
        ]

        # Active incidents
        lines.extend(["## Active Incidents", ""])
        if self.active_incidents:
            for inc in self.active_incidents:
                type_str = ", ".join(t.value for t in inc.incident_types)
                lines.append(
                    f"- **{inc.incident_id}** [{inc.severity.value}] "
                    f"{inc.title} ({type_str}) -- {inc.status}"
                )
                if inc.summary:
                    lines.append(f"  - {inc.summary[:200]}")
        else:
            lines.append("_No active incidents._")

        # Resolved
        lines.extend(["", "## Resolved This Shift", ""])
        if self.resolved_incidents:
            for rid in self.resolved_incidents:
                lines.append(f"- {rid}")
        else:
            lines.append("_None._")

        # Actions taken
        lines.extend(["", "## Actions Taken", ""])
        if self.actions_taken:
            for a in self.actions_taken:
                owner_str = f" ({a.owner})" if a.owner else ""
                lines.append(f"- [{a.status.upper()}] {a.description}{owner_str}")
                if a.notes:
                    lines.append(f"  - Note: {a.notes}")
        else:
            lines.append("_No actions logged._")

        # Pending actions
        lines.extend(["", "## Pending / Requires Follow-up", ""])
        if self.pending_actions:
            for a in self.pending_actions:
                owner_str = f" (Owner: {a.owner})" if a.owner else ""
                lines.append(f"- **{a.description}**{owner_str}")
                if a.notes:
                    lines.append(f"  - {a.notes}")
        else:
            lines.append("_No pending actions._")

        # Emerging signals
        if self.emerging_signals_summary:
            lines.extend([
                "",
                "## Emerging Signals",
                "",
                self.emerging_signals_summary,
            ])

        # Notes
        if self.notes:
            lines.extend(["", "## Analyst Notes", "", self.notes])

        lines.append("")
        return "\n".join(lines)


class HandoverGenerator:
    """Builds handover summaries from shift data."""

    def __init__(self, lookback_hours: int = 8):
        self.lookback_hours = lookback_hours

    def generate(
        self,
        sitreps: list[SitRep],
        resolved_ids: list[str] | None = None,
        actions: list[ShiftAction] | None = None,
        analyst: str = "",
        notes: str = "",
    ) -> HandoverSummary:
        now = datetime.now(timezone.utc)

        active = [s for s in sitreps if s.status == "active"]
        active.sort(key=lambda s: list(Severity).index(s.severity))

        all_actions = actions or []
        completed = [a for a in all_actions if a.status in ("completed", "escalated")]
        pending = [a for a in all_actions if a.status in ("pending", "deferred")]

        # Build emerging signals summary from low-severity active incidents
        emerging = [s for s in active if s.severity in (Severity.P3, Severity.P4)]
        if emerging:
            emerging_text = (
                f"{len(emerging)} low-severity signals under monitoring. "
                f"Types: {', '.join({t.value for s in emerging for t in s.incident_types})}."
            )
        else:
            emerging_text = ""

        return HandoverSummary(
            shift_start=now.isoformat(),
            shift_end=now.isoformat(),
            analyst=analyst,
            active_incidents=active,
            resolved_incidents=resolved_ids or [],
            actions_taken=completed,
            pending_actions=pending,
            emerging_signals_summary=emerging_text,
            notes=notes,
        )
