"""Tests for webhook notification dispatch."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crisis_lens.config import NotificationsConfig, Severity, WebhookConfig
from crisis_lens.detection.rules import Signal
from crisis_lens.notifications.dispatcher import WebhookDispatcher


def make_signal(severity: Severity = Severity.P1, score: float = 0.9) -> Signal:
    return Signal(
        signal_id="SIG-TEST-001",
        text="test crisis signal",
        source="test",
        language="en",
        score=score,
        severity=severity,
        matched_rules=["keyword_violence"],
    )


def make_mock_client(status_code: int = 200) -> tuple[MagicMock, MagicMock]:
    mock_response = MagicMock()
    mock_response.is_success = status_code < 400
    mock_response.status_code = status_code
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    return mock_client, mock_response


def patch_async_client(mock_client: MagicMock) -> "AbstractContextManager[MagicMock]":  # type: ignore[name-defined]
    patcher = patch("crisis_lens.notifications.dispatcher.httpx.AsyncClient")
    mock_class = patcher.start()
    mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
    return patcher


class TestWebhookDispatcherSeverityFilter:
    def test_p0_passes_with_min_p2(self):
        config = NotificationsConfig(enabled=True, min_severity=Severity.P2)
        d = WebhookDispatcher(config)
        assert d._severity_passes(Severity.P0) is True

    def test_p1_passes_with_min_p2(self):
        config = NotificationsConfig(enabled=True, min_severity=Severity.P2)
        d = WebhookDispatcher(config)
        assert d._severity_passes(Severity.P1) is True

    def test_p2_passes_with_min_p2(self):
        config = NotificationsConfig(enabled=True, min_severity=Severity.P2)
        d = WebhookDispatcher(config)
        assert d._severity_passes(Severity.P2) is True

    def test_p3_blocked_with_min_p2(self):
        config = NotificationsConfig(enabled=True, min_severity=Severity.P2)
        d = WebhookDispatcher(config)
        assert d._severity_passes(Severity.P3) is False

    def test_p4_blocked_with_min_p2(self):
        config = NotificationsConfig(enabled=True, min_severity=Severity.P2)
        d = WebhookDispatcher(config)
        assert d._severity_passes(Severity.P4) is False

    def test_only_p0_passes_with_min_p0(self):
        config = NotificationsConfig(enabled=True, min_severity=Severity.P0)
        d = WebhookDispatcher(config)
        assert d._severity_passes(Severity.P0) is True
        assert d._severity_passes(Severity.P1) is False


class TestWebhookDispatcherPayload:
    def test_payload_structure(self):
        signal = make_signal()
        payload = WebhookDispatcher._build_payload(signal)
        assert payload["alert_type"] == "signal"
        assert payload["signal_id"] == signal.signal_id
        assert payload["severity"] == "P1"
        assert "score" in payload
        assert "text" in payload
        assert "source" in payload
        assert "language" in payload
        assert "matched_rules" in payload
        assert "suggested_types" in payload
        assert "created_at" in payload

    def test_payload_text_truncated_at_500(self):
        signal = make_signal()
        signal.text = "a" * 800
        payload = WebhookDispatcher._build_payload(signal)
        assert len(payload["text"]) == 500

    def test_payload_text_under_limit_unchanged(self):
        signal = make_signal()
        signal.text = "short text"
        payload = WebhookDispatcher._build_payload(signal)
        assert payload["text"] == "short text"


class TestWebhookDispatcherDispatch:
    @pytest.mark.asyncio
    async def test_disabled_config_skips_http(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(enabled=False, webhooks=[WebhookConfig(url="https://example.com")])
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal())
            mock_client.post.assert_not_called()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_empty_webhooks_skips_http(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(enabled=True, webhooks=[])
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal())
            mock_client.post.assert_not_called()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_low_severity_signal_skips_http(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P2,
            webhooks=[WebhookConfig(url="https://example.com")],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P3))
            mock_client.post.assert_not_called()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_posts_to_webhook_url(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P2,
            webhooks=[WebhookConfig(url="https://hooks.example.com/alert")],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P1))
            mock_client.post.assert_called_once()
            url_called = mock_client.post.call_args[0][0]
            assert url_called == "https://hooks.example.com/alert"
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_payload_json_in_request_body(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P3,
            webhooks=[WebhookConfig(url="https://hooks.example.com/alert")],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            signal = make_signal(severity=Severity.P1)
            await d.dispatch(signal)
            body = json.loads(mock_client.post.call_args[1]["content"])
            assert body["severity"] == "P1"
            assert body["signal_id"] == signal.signal_id
            assert body["alert_type"] == "signal"
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_posts_to_multiple_webhooks(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P3,
            webhooks=[
                WebhookConfig(url="https://hooks-a.example.com"),
                WebhookConfig(url="https://hooks-b.example.com"),
            ],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P1))
            assert mock_client.post.call_count == 2
            urls_called = {call[0][0] for call in mock_client.post.call_args_list}
            assert "https://hooks-a.example.com" in urls_called
            assert "https://hooks-b.example.com" in urls_called
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_hmac_signature_header_present(self):
        mock_client, _ = make_mock_client()
        secret = "my-signing-secret"
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P3,
            webhooks=[WebhookConfig(url="https://hooks.example.com/alert", secret=secret)],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P1))
            headers = mock_client.post.call_args[1]["headers"]
            assert "X-Crisis-Lens-Signature" in headers
            body = mock_client.post.call_args[1]["content"]
            expected_sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            assert headers["X-Crisis-Lens-Signature"] == f"sha256={expected_sig}"
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_no_signature_header_without_secret(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P3,
            webhooks=[WebhookConfig(url="https://hooks.example.com/alert")],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P1))
            headers = mock_client.post.call_args[1]["headers"]
            assert "X-Crisis-Lens-Signature" not in headers
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_retries_on_5xx_then_succeeds(self):
        error_response = MagicMock()
        error_response.is_success = False
        error_response.status_code = 503

        success_response = MagicMock()
        success_response.is_success = True
        success_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post.side_effect = [error_response, success_response]

        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P3,
            webhooks=[WebhookConfig(url="https://hooks.example.com", max_retries=2, retry_base_delay=0.0)],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P1))
            assert mock_client.post.call_count == 2
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self):
        mock_client, _ = make_mock_client(status_code=400)
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P3,
            webhooks=[WebhookConfig(url="https://hooks.example.com", max_retries=3, retry_base_delay=0.0)],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P1))
            mock_client.post.assert_called_once()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_custom_headers_forwarded(self):
        mock_client, _ = make_mock_client()
        config = NotificationsConfig(
            enabled=True,
            min_severity=Severity.P3,
            webhooks=[WebhookConfig(
                url="https://hooks.example.com",
                headers={"Authorization": "Bearer token123", "X-Team": "safety"},
            )],
        )
        d = WebhookDispatcher(config)
        patcher = patch_async_client(mock_client)
        try:
            await d.dispatch(make_signal(severity=Severity.P1))
            headers = mock_client.post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer token123"
            assert headers["X-Team"] == "safety"
            assert headers["Content-Type"] == "application/json"
        finally:
            patcher.stop()
