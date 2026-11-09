"""The kanban notifier names each wake by the ``task_events`` rows it was built from (#152206).

The cursor claim already makes one batch unique, so naming the wake by its own event ids turns that
claim into an admission the fence can replay: the retry after a refused wake settles the SAME
admission instead of queueing a second copy of the same event — the retry amplification the random
``uuid4`` identity made impossible to suppress. A genuinely new event of the same task in the same
second gets a different name and still wakes: the identity is event ids, never a wall-clock window.

The name is observed at the ``deliver_wake`` seam, which is where the notifier hands it to the fence.
"""
import asyncio

from gateway import wake as wake_mod
from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    """Push-capable adapter (no supports_async_delivery attr => push)."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


class RecordingWake:
    """Stands in for ``gateway.wake.deliver_wake`` and records the token each wake carries."""

    def __init__(self, refuse_first=False):
        self.calls = []
        self._refuse = refuse_first

    async def __call__(self, adapter, *, text, session_id="", source=None,
                       notification_category="result", profile=None, dedupe=None):
        self.calls.append({"text": text, "dedupe": dedupe})
        if self._refuse:
            self._refuse = False
            raise wake_mod.WakeNotAccepted("simulated refusal", reason="runtime_draining")

    @property
    def names(self):
        return [call["dedupe"] for call in self.calls]


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _subscribed_task():
    """One task with a wake-only telegram sub."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="wake naming task", assignee="worker",
                                 session_id="agent:main:telegram:dm:chat-1")
        kbn.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="wake")
    return task_id


def _add_wake_bearing_event(task_id, reason):
    """A second ``blocked`` event for the same task, in the same second as the first."""
    with kbc.connect() as conn:
        kb.unblock_task(conn, task_id)
        kb.block_task(conn, task_id, reason=reason, kind="needs_input")


def test_a_refused_wake_retries_under_the_same_name(tmp_path, monkeypatch):
    """The wake leg failed, the claim rewound and the next tick re-delivered the SAME event: the
    retry carries the same identity, so the fence replays one admission instead of queueing two."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "retry-name.db"))
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="wake naming task", assignee="worker",
                                 session_id="agent:main:telegram:dm:chat-1")
        kbn.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="wake")
        kb.complete_task(conn, task_id, summary="done")

    recorder = RecordingWake(refuse_first=True)
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))
    assert len(recorder.names) == 1, f"one refused wake attempt, got {recorder.names}"
    first = recorder.names[0]
    assert first, "the notifier always names a kanban wake"

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))  # the rewound event is claimed again
    assert len(recorder.names) == 2, "a refused wake is retried, not dropped"
    assert recorder.names[1] == first, (
        f"a retry of one event must reuse its name, got {first!r} then {recorder.names[1]!r}"
    )


def test_two_events_of_one_task_get_two_names(tmp_path, monkeypatch):
    """Two wake-bearing events of the same task — both written in the same second — are two wakes:
    the name is the event ids, so nothing about the task or the clock can merge them."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "two-names.db"))
    kb.init_db()
    task_id = _subscribed_task()
    with kbc.connect() as conn:
        kb.block_task(conn, task_id, reason="needs input", kind="needs_input")

    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))
    assert len(recorder.names) == 1, recorder.names
    first = recorder.names[0]

    _add_wake_bearing_event(task_id, "needs input again")
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))
    assert len(recorder.names) == 2, "the second event of the same task still wakes"
    second = recorder.names[1]

    assert first != second, f"two events of one task must not share a name: {first!r} / {second!r}"
    head = [name.split(":")[:3] for name in (first, second)]
    assert head[0] == head[1], f"both names are one board and task: {first!r} / {second!r}"
    assert [parts[0] for parts in head] == ["kanban", "kanban"], head
    assert [parts[2] for parts in head] == [task_id, task_id], (
        f"both names still say which task they woke: {first!r} / {second!r}"
    )
    assert [len(name.split(":")) for name in (first, second)] == [4, 4], (
        "the token stays one short element, so the identity fits the fence's bounded request_id"
    )


def test_the_name_is_the_event_ids_not_their_order():
    """The token is a function of the claimed event ids: same ids in any order, same name; a
    different board or task, a different name — so two boards can never mute each other."""
    from gateway.kanban_watchers_notifier import _wake_dedupe_token

    class _Event:
        def __init__(self, id):
            self.id = id

    one = _wake_dedupe_token("ops", "t_1", [_Event(7), _Event(9)])
    assert one == _wake_dedupe_token("ops", "t_1", [_Event(9), _Event(7)])
    assert one.split(":")[:3] == ["kanban", "ops", "t_1"], one
    assert _wake_dedupe_token("other", "t_1", [_Event(7), _Event(9)]) != one
    assert _wake_dedupe_token("ops", "t_2", [_Event(7), _Event(9)]) != one
    assert _wake_dedupe_token("ops", "t_1", [_Event(7)]) != one
