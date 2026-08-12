"""Kanban worker session-end emission (HM1).

A kanban worker session must emit the memory-provider ``on_session_end``
lifecycle event when it finishes — on BOTH the success and failure paths,
exactly once per session — so memory providers (e.g. ``memori_byodb``, which
writes only from that hook) ingest the worker's last turns.

The dispatcher spawns workers as ``hermes ... chat -q`` subprocesses:
  - the non-goal worker takes the ``quiet=False`` single-query branch, which
    already reaches ``_finalize_single_query`` → ``_run_cleanup`` →
    ``shutdown_memory_provider`` from its ``finally`` block;
  - the goal-mode worker takes the fully-quiet ``quiet=True`` branch, which
    used to rely SOLELY on ``atexit`` firing ``_run_cleanup`` after
    ``sys.exit``. This regression test locks in that the ``-Q`` path now calls
    ``_finalize_single_query`` deterministically before exiting, so the
    memory session-end fires even when atexit is bypassed (kanban
    ``os._exit(0)`` signal handler, exit watchdog, hard kill) — exactly once,
    and safely (a raising hook never fails the worker).

Interactive-session behaviour is unchanged: ``_run_cleanup`` already fired the
memory session-end via the same ``shutdown_memory_provider`` path and is
untouched here; only the goal-mode worker's exit branch changed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import cli as cli_mod
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """A memory provider that records its lifecycle calls."""

    def __init__(self, log, *, raise_on_end=False):
        self._log = log
        self._raise = raise_on_end

    @property
    def name(self):
        return "recording"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self._log.append("initialize")

    def on_session_end(self, messages):
        self._log.append(f"on_session_end:{len(messages)}")
        if self._raise:
            raise RuntimeError("provider on_session_end boom")

    def shutdown_all(self):
        self._log.append("shutdown_all")

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return "{}"


def _build_fake_cli(log, *, fail=False, raise_on_end=False):
    """Build a FakeCLI that mimics the worker single-query surface.

    The agent carries a real MemoryManager wired to a _RecordingProvider, so
    a successful emission actually exercises the provider's on_session_end.
    """
    mm = MemoryManager()
    mm._providers = [_RecordingProvider(log, raise_on_end=raise_on_end)]
    agent = SimpleNamespace(
        session_id="worker-session",
        platform="cli",
        model="probe",
        _session_messages=[{"role": "user", "content": "hi"}],
        _memory_manager=mm,
    )

    def shutdown(msgs=None):
        log.append("shutdown_memory_provider")
        mm.on_session_end(msgs if msgs is not None else [])
        mm.shutdown_all()

    agent.shutdown_memory_provider = shutdown
    agent.run_conversation = (
        lambda user_message=None, conversation_history=None: {
            "final_response": "done",
            "failed": fail,
            "error": "boom" if fail else None,
        }
    )

    def _chat(query, images=None):
        mm.initialize_all("worker-session")
        return "done"

    cli = SimpleNamespace(
        console=SimpleNamespace(print=lambda *a, **kw: None),
        session_id="worker-session",
        conversation_history=[],
        agent=agent,
        chat=_chat,
        _claim_active_session=lambda surface, *, stderr=False: True,
        _show_security_advisories=lambda: None,
        _print_exit_summary=lambda clear_screen=True: None,
        _release_active_session=lambda: log.append("release_active_session"),
        _ensure_runtime_credentials=lambda: True,
        _init_agent=lambda **kw: True,
        _resolve_turn_agent_config=lambda q: {
            "signature": "s", "model": None, "runtime": None, "request_overrides": None,
        },
        _active_agent_route_signature="s",
    )
    return cli


def _patch_fake_cli(fake_cli):
    """Point the cli module at the fake and neutralise atexit + helpers."""
    __import__("sys").modules["cli"].HermesCLI = lambda **kw: fake_cli
    cli_mod.atexit.register = lambda *a, **kw: None
    cli_mod._active_agent_ref = fake_cli.agent
    cli_mod._cleanup_done = False
    cli_mod._single_query_finalize_attempted_session_ids = set()


def _run_quiet_worker(log, *, quiet, fail=False, raise_on_end=False):
    """Drive ``cli.main(query=...)`` for the given quiet/fail combination.

    Simulates ``sys.exit`` (catches SystemExit) but does NOT simulate atexit,
    because the fix under test is precisely that the ``-Q`` path no longer
    depends on atexit to deliver the memory session-end.
    """
    fake = _build_fake_cli(log, fail=fail, raise_on_end=raise_on_end)
    _patch_fake_cli(fake)
    try:
        try:
            cli_mod.main(query="work kanban task X", quiet=quiet, toolsets="terminal")
        except SystemExit:
            pass
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False
        cli_mod._single_query_finalize_attempted_session_ids = set()
        __import__("sys").modules["cli"].HermesCLI = cli_mod.HermesCLI


def _count_ends(log):
    return sum(1 for c in log if c.startswith("on_session_end"))


def test_worker_quiet_path_emits_exactly_once():
    """Non-goal (-q, quiet=False) worker success emits on_session_end once."""
    log = []
    _run_quiet_worker(log, quiet=False)
    assert log.count("shutdown_memory_provider") == 1
    assert _count_ends(log) == 1
    assert "on_session_end:1" in log  # forwarded the real transcript


def test_goal_quiet_worker_success_emits_exactly_once():
    """Goal-mode (-Q, quiet=True) worker success emits on_session_end once,
    deterministically via _finalize_single_query (not atexit)."""
    log = []
    _run_quiet_worker(log, quiet=True)
    assert log.count("shutdown_memory_provider") == 1
    assert _count_ends(log) == 1


def test_goal_quiet_worker_failure_emits_exactly_once():
    """Goal-mode (-Q) worker FAILURE also emits on_session_end exactly once."""
    log = []
    _run_quiet_worker(log, quiet=True, fail=True)
    assert log.count("shutdown_memory_provider") == 1
    assert _count_ends(log) == 1


@patch("hermes_cli.plugins.invoke_hook")
def test_raising_provider_hook_does_not_fail_worker(mock_invoke_hook):
    """A memory provider on_session_end that raises is swallowed (logged by
    MemoryManager) — it never fails the worker card or the dispatcher."""
    log = []
    _run_quiet_worker(log, quiet=True, raise_on_end=True)
    assert _count_ends(log) == 1  # the provider was still called
    # The worker survived the raise: cleanup ran and the session lease was
    # released (had the exception escaped the MemoryManager guard, this
    # would never be reached).
    assert log.count("shutdown_memory_provider") == 1
    assert log.count("release_active_session") >= 1


def test_run_cleanup_still_dedupes_to_exactly_once():
    """Interactive shutdown: _run_cleanup's memory session-end still fires
    exactly once (the _cleanup_done guard), so my worker-path emission cannot
    double-fire when atexit later runs the same cleanup."""
    log = []
    fake = _build_fake_cli(log)
    cli_mod._active_agent_ref = fake.agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()
        cli_mod._run_cleanup()  # second call must be a no-op
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False
    assert log.count("shutdown_memory_provider") == 1
    assert _count_ends(log) == 1
