"""Post-stale-kill retry jitter (incident t_22b546e3, section 7.4).

A stream stale kill fires exactly 900s after the last chunk, so retrying the
instant the kill lands can re-enter the provider on the same clock phase and
die again. The first retry after a stale kill therefore waits a uniform
30-60s; every other retry path keeps its existing backoff. These tests inject
the RNG, sleeper and clock — no wall-clock sleeping, no network.
"""

import logging

import httpx
import pytest
from unittest.mock import MagicMock

from agent import chat_completion_helpers as h

pytestmark = pytest.mark.real_stale_retry_jitter


def _bare_call():
    """A ``_StreamingCall`` without the client/transport setup."""
    call = object.__new__(h._StreamingCall)
    call.agent = MagicMock()
    call.agent._interrupt_requested = False
    call.agent._consecutive_stale_streams = 0
    call.api_kwargs = {"model": "test-model"}
    call._stale_kill_pending = {"yes": False}
    call._attempt_stream_response = None
    call._stream_stale_timeout = 900.0
    call.last_chunk_time = {"t": 0.0}
    call.clients = MagicMock()
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call.deltas_were_sent = {"yes": False}
    call.first_delta_fired = {"done": False}
    call.provider_tool_in_flight = {"yes": False}
    call._request_cancelled = {"value": False}
    call.managed_stream_holder = {"stream": None}
    call._cancel_current_stream_attempt = MagicMock()
    return call


def test_stale_retry_waits_in_window_logs_and_is_one_shot(caplog):
    """The retry after a stale kill sleeps the drawn jitter once and logs it."""
    call = _bare_call()
    slept = []
    clock = {"t": 0.0}

    def sleeper(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    seen = {}

    def rng(low, high):
        seen["window"] = (low, high)
        return 42.5

    # The kill (monitor thread) arms the wait.
    call._kill_stale_stream(900.0)
    assert call._stale_kill_pending["yes"] is True

    with caplog.at_level(logging.INFO, logger=h.logger.name):
        assert call._wait_out_stale_retry_jitter(
            rng=rng, sleeper=sleeper, clock=lambda: clock["t"]) is True

    low, high = h.STALE_RETRY_JITTER_S
    assert seen["window"] == (low, high)
    assert low <= 42.5 <= high
    assert sum(slept) == pytest.approx(42.5)
    assert all(seconds <= 0.2 for seconds in slept)
    assert "stale_retry jitter=42.5s" in caplog.text

    # One-shot: a second retry (or any non-stale retry) pays nothing.
    slept.clear()
    assert call._wait_out_stale_retry_jitter(rng=rng, sleeper=sleeper) is True
    assert slept == []


def test_stale_retry_wait_aborts_promptly_on_interrupt(monkeypatch):
    """/stop during the 30-60s wait must not hang the turn."""
    call = _bare_call()
    call._stale_kill_pending["yes"] = True
    call.agent._interrupt_requested = True
    slept = []

    assert call._wait_out_stale_retry_jitter(
        rng=lambda low, high: high, sleeper=slept.append) is False
    assert slept == []

    # The retry loop turns that False into the normal interrupt result.
    call.agent._interrupt_requested = False
    call._stale_kill_pending["yes"] = True
    call._wait_out_stale_retry_jitter = lambda: False
    call._start_stream_attempt = MagicMock(return_value=0)
    call._call()
    assert isinstance(call.result["error"], InterruptedError)


def test_normal_stream_error_does_not_arm_the_wait():
    """A plain transient stream error keeps the old immediate retry."""
    call = _bare_call()
    call.agent._is_provider_stream_parse_error.return_value = False
    call.agent._stream_options_unsupported = False

    assert call._handle_stream_error(httpx.ConnectError("boom"), attempt=0, max_retries=2) is True
    assert call._stale_kill_pending["yes"] is False
    assert call._wait_out_stale_retry_jitter(
        rng=lambda low, high: high, sleeper=MagicMock()) is True
