"""Unit tests for telemetry/metrics.py: MetricEvent validation and the
LoggingMetricsHook reference implementation."""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from resilientforge.telemetry import LoggingMetricsHook, MetricEvent


def test_metric_event_requires_event_type_tool_name_timestamp():
    event = MetricEvent(event_type="call_result", tool_name="t", timestamp="2026-01-01T00:00:00+00:00")
    assert event.success is None  # every event-specific field defaults to None


def test_metric_event_rejects_an_unknown_event_type():
    with pytest.raises(ValidationError):
        MetricEvent(event_type="something_else", tool_name="t", timestamp="2026-01-01T00:00:00+00:00")


def test_logging_metrics_hook_logs_valid_json(caplog):
    hook = LoggingMetricsHook()
    with caplog.at_level(logging.INFO, logger="resilientforge.metrics"):
        hook(
            MetricEvent(
                event_type="call_result",
                tool_name="create_event",
                timestamp="2026-01-01T00:00:00+00:00",
                success=True,
                source="recipe",
            )
        )

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["event_type"] == "call_result"
    assert payload["tool_name"] == "create_event"
    assert payload["success"] is True
    assert payload["source"] == "recipe"
    assert "error_type" not in payload  # exclude_none — unset fields aren't emitted


def test_logging_metrics_hook_uses_a_distinct_named_logger():
    hook = LoggingMetricsHook(logger_name="my.custom.logger")
    assert hook._logger.name == "my.custom.logger"
