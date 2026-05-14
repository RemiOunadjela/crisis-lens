"""Webhook notification dispatcher for crisis-lens signals.

Fires a JSON HTTP POST to each configured endpoint when a signal meets the
minimum severity threshold. Supports exponential backoff retry and optional
HMAC-SHA256 request signing for payload verification.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from crisis_lens.config import NotificationsConfig, Severity
from crisis_lens.detection.rules import Signal

logger = logging.getLogger("crisis_lens.notifications")

_SEVERITY_ORDER = [Severity.P0, Severity.P1, Severity.P2, Severity.P3, Severity.P4]


class WebhookDispatcher:
    """Dispatches signal alerts to configured webhook URLs via HTTP POST."""

    def __init__(self, config: NotificationsConfig) -> None:
        self.config = config

    def _severity_passes(self, signal_severity: Severity) -> bool:
        return _SEVERITY_ORDER.index(signal_severity) <= _SEVERITY_ORDER.index(self.config.min_severity)

    @staticmethod
    def _build_payload(signal: Signal) -> dict[str, Any]:
        return {
            "alert_type": "signal",
            "signal_id": signal.signal_id,
            "severity": signal.severity.value,
            "score": round(signal.score, 4),
            "text": signal.text[:500],
            "source": signal.source,
            "language": signal.language,
            "matched_rules": signal.matched_rules,
            "suggested_types": [t.value for t in signal.suggested_types],
            "created_at": signal.created_at,
        }

    async def _post_with_retry(
        self,
        client: httpx.AsyncClient,
        webhook_index: int,
        payload: dict[str, Any],
    ) -> None:
        wh = self.config.webhooks[webhook_index]
        body = json.dumps(payload)
        headers = {**wh.headers, "Content-Type": "application/json"}
        if wh.secret:
            sig = hmac.new(wh.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            headers["X-Crisis-Lens-Signature"] = f"sha256={sig}"

        last_exc: Exception | None = None
        for attempt in range(wh.max_retries + 1):
            try:
                resp = await client.post(wh.url, content=body, headers=headers, timeout=wh.timeout_seconds)
                if resp.is_success:
                    logger.debug("webhook delivered to %s (status %d)", wh.url, resp.status_code)
                    return
                logger.warning("webhook %s returned %d (attempt %d)", wh.url, resp.status_code, attempt + 1)
                if resp.status_code < 500:
                    return  # 4xx won't improve with retries
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning("webhook %s request error on attempt %d: %s", wh.url, attempt + 1, exc)

            if attempt < wh.max_retries:
                await asyncio.sleep(wh.retry_base_delay * (2**attempt))

        if last_exc:
            logger.error("webhook %s failed after %d attempts: %s", wh.url, wh.max_retries + 1, last_exc)

    async def dispatch(self, signal: Signal) -> None:
        """POST alert payload to all configured endpoints if signal passes severity filter."""
        if not self.config.enabled or not self.config.webhooks:
            return
        if not self._severity_passes(signal.severity):
            return

        payload = self._build_payload(signal)
        async with httpx.AsyncClient() as client:
            await asyncio.gather(
                *[
                    self._post_with_retry(client, i, payload)
                    for i in range(len(self.config.webhooks))
                ],
                return_exceptions=True,
            )
