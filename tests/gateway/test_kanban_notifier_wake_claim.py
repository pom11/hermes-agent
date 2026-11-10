"""The gateway notifier is a *writer* of the wake-once claim (#152206 §3.1, wired up).

§3.1 put ``kanban_wake_claims`` (one row per profile/task/event) and its four helpers in
the board DB; nothing wrote to it. This file pins the notifier half of the wiring: it
takes the claim immediately before it wakes, confirms it when the wake is accepted and
deletes it when the send was refused, so a second emitter that wakes the same profile for
the same ``task_events`` row has something to ask.

The two halves stay separate, and the tests keep them separate:

* §A (``test_kanban_notifier_wake_dedupe.py``) names the wake by its event ids, so a
  retry by the SAME emitter replays one admission at the fence — idempotence.
* §B (here) decides which emitter OWES the wake, so a DIFFERENT emitter stays quiet —
  suppression. A same-emitter retry must therefore still wake (its own claim was handed
  back the moment the send was refused) and must reuse the same name.

And the suppression is only as wide as what gets claimed: kinds outside ``_WAKE_KINDS``
(``spawn_failed``, a bare ``status``) are never claimed, so no row can ever mute a wake
through them. Everything unknown — absent table, helper that raises — delivers.
"""
import asyncio
import sqlite3

from gateway import wake as wake_mod
from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

WAKER = "hscc-orch-waker"  # the other emitter (#152206 §3.2, out-of-process waker)


class RecordingAdapter:
    """A push-capable stand-in: it declares nothing, so ``adapter_supports_push``
    (``gateway/wake.py:35``) defaults it to True."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


class RecordingWake:
    """Stands in for ``gateway.wake.deliver_wake``; records each wake's §A name and where it goes."""

    def __init__(self, refuse_first=False, raise_first=False):
        self.calls = []
        self._refuse = refuse_first
        self._raise = raise_first

    async def __call__(self, adapter, *, text, session_id="", source=None,
                       notification_category="result", profile=None, dedupe=None):
        self.calls.append({"text": text, "dedupe": dedupe, "session_id": session_id,
                           "chat": getattr(source, "chat_id", None)})
        if self._refuse:
            self._refuse = False
            raise wake_mod.WakeNotAccepted("simulated refusal", reason="runtime_draining")
        if self._raise:
            self._raise = False
            raise RuntimeError("simulated adapter failure")

    @property
    def names(self):
        return [call["dedupe"] for call in self.calls]

    @property
    def texts(self):
        return [call["text"] for call in self.calls]

    @property
    def destinations(self):
        return sorted(str(c["chat"] or c["session_id"] or "") for c in self.calls)


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _profile():
    """The profile an un-stamped subscription's wake lands in — the §3.1 claim key."""
    return _make_runner()._active_profile_name() or "default"


def _tick(monkeypatch):
    """One notifier tick on a fresh runner, which is what the loop needs: the tick stops
    by clearing ``_running``, so a runner is good for exactly one of them."""
    runner = _make_runner()
    asyncio.run(_one_tick(monkeypatch, runner))
    return runner


async def _one_tick(monkeypatch, runner):
    """Run exactly one notifier tick, the way the §A naming test does."""
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:  # the watcher's initial wiring delay
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _board(tmp_path, monkeypatch, name):
    """A pinned board DB (the shape a dispatcher-pinned box runs on)."""
    db = tmp_path / f"{name}.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    kb.init_db()
    return db


def _wake_subscribed_task(db, *, title="wake claim task"):
    """A task with a wake-only telegram sub, caught up so only later events are unseen."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title=title, assignee="worker",
                                 session_id="agent:main:telegram:dm:chat-1")
        kbn.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="chat-1",
                           chat_type="dm", delivery_mode="wake")
    return task_id


def _events(db, task_id, kind=None):
    with kbc.connect() as conn:
        rows = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ?"
            + (" AND kind = ?" if kind else "")
            + " ORDER BY id",
            (task_id, kind) if kind else (task_id,),
        ).fetchall()
    return [int(r["id"]) for r in rows]


def _claim_rows(db):
    with kbc.connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT profile, task_id, event_id, emitter, claimed_at, confirmed_at"
            " FROM kanban_wake_claims ORDER BY event_id, emitter").fetchall()]


def _claim_for(db, event_id):
    """The single claim row for one event; a mismatch names what was there instead."""
    rows = [r for r in _claim_rows(db) if r["event_id"] == int(event_id)]
    assert len(rows) == 1, f"expected exactly one claim row for event {event_id}, got {rows}"
    return rows[0]


def _cursor(db, task_id):
    with kbc.connect() as conn:
        return int(conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs WHERE task_id = ?",
            (task_id,)).fetchone()["last_event_id"])


def _spy(monkeypatch, op):
    """Record the notifier's calls to one claim helper while still running the real one.

    The failure paths would otherwise pass on base too: an emitter that never claims
    anything leaves nothing to release, so "the row is gone" proves nothing. The calls do.
    """
    calls: list[dict] = []
    real = getattr(kbn, op)

    def _recorded(conn, **kwargs):
        calls.append(kwargs)
        return real(conn, **kwargs)

    monkeypatch.setattr(kbn, op, _recorded)
    return calls


def test_the_wake_lands_once_and_the_claim_is_recorded_for_the_woken_profile(tmp_path, monkeypatch):
    """Control, and the key contract: the row is keyed by the PROFILE THE WAKE LANDS IN,
    stamped with this emitter's name and confirmed once the wake was accepted. The other
    emitter asks with exactly that key, so a key that disagreed would suppress nothing."""
    db = _board(tmp_path, monkeypatch, "claim-record")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done")
    (event_id,) = _events(db, task_id, "completed")

    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    runner = _tick(monkeypatch)

    assert recorder.names == [recorder.names[0]] and recorder.names[0], recorder.names
    profile = runner._kanban_notifier_profile  # the profile the wake was addressed to
    assert profile, "the notifier resolves the woken profile"
    row = _claim_for(db, event_id)
    assert row["profile"] == profile and row["task_id"] == task_id, row
    assert row["emitter"] == "gateway-notifier", row
    assert row["confirmed_at"] is not None, "an accepted wake confirms its claim"
    with kbc.connect() as conn:
        assert kbn.wake_already_claimed(conn, profile=profile, task_id=task_id,
                                        event_ids=[event_id]) == {event_id}


def test_a_cross_emitter_claim_suppresses_the_wake_and_still_settles_the_cursor(tmp_path, monkeypatch):
    """§B is what this card adds: the other emitter already woke this profile for this
    event, so this notifier sends no wake — and it does not spin either, because the
    cursor still advances (the other emitter owns the delivery now)."""
    db = _board(tmp_path, monkeypatch, "cross-emitter")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done")
    (event_id,) = _events(db, task_id, "completed")

    profile = _profile()
    with kbc.connect() as conn:  # the waker got there first, and delivered
        assert kbn.try_claim_wake(conn, profile=profile, task_id=task_id,
                                  event_id=event_id, emitter=WAKER) is True
        assert kbn.confirm_wake(conn, profile=profile, task_id=task_id,
                                event_id=event_id, emitter=WAKER) is True

    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert recorder.names == [], f"the wake was already owed by {WAKER}: {recorder.names}"
    assert _cursor(db, task_id) == event_id, "suppression settles the cursor; no tick-per-tick retry"
    row = _claim_for(db, event_id)
    assert row["emitter"] == WAKER and row["confirmed_at"] is not None, (
        f"holder-only ops leave another emitter's claim alone: {row}")


def test_one_emitters_own_claim_never_suppresses_its_own_fan_out(tmp_path, monkeypatch):
    """Two subscriptions in one profile are two destinations, so one event is two wakes.

    §3.1 suppresses a DIFFERENT emitter's duplicate; it must not arbitrate this emitter's
    fan-out, which the per-subscription cursor already settles and §A's fence already
    separates — the admission identity is ``[producer, session key, name]``, so the same
    name reaching two sessions is two admissions. A claim we hold is never our own mute.
    """
    db = _board(tmp_path, monkeypatch, "fan-out")
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="notify both", assignee="worker",
                                 session_id="agent:main:telegram:dm:chat-1")
        for chat in ("chat-1", "chat-2"):
            kbn.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id=chat,
                               chat_type="dm", delivery_mode="wake")
        kb.complete_task(conn, task_id, summary="done")
    (event_id,) = _events(db, task_id, "completed")

    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert len(recorder.names) == 2, f"both destinations wake for one event: {recorder.names}"
    assert recorder.names[0] == recorder.names[1], (
        f"one event is one name; the destination separates them: {recorder.names}")
    assert recorder.destinations == ["chat-1", "chat-2"], recorder.destinations
    row = _claim_for(db, event_id)
    assert row["emitter"] == "gateway-notifier" and row["confirmed_at"] is not None, row


def test_a_failed_send_releases_the_claim_so_the_other_emitter_can_wake(tmp_path, monkeypatch):
    """A refused send (``WakeNotAccepted``) deletes the row rather than letting the lease
    run: from the other emitter's seat the event is immediately owed again, so a notifier
    that cannot reach its destination never mutes the wake."""
    db = _board(tmp_path, monkeypatch, "release")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done")
    (event_id,) = _events(db, task_id, "completed")

    recorder = RecordingWake(refuse_first=True)
    profile = _profile()
    released = _spy(monkeypatch, "release_wake")
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert len(recorder.names) == 1, "the wake was attempted once"
    assert [c["event_id"] for c in released] == [event_id], (
        f"the notifier itself handed its claim back: {released}")
    with kbc.connect() as conn:
        assert kbn.wake_already_claimed(conn, profile=profile, task_id=task_id,
                                        event_ids=[event_id]) == set()
        assert kbn.try_claim_wake(conn, profile=profile, task_id=task_id,
                                  event_id=event_id, emitter=WAKER) is True, (
            "the refused wake is owed again to the other emitter, not leased to us")
    assert _cursor(db, task_id) < event_id, "the cursor rewound for our own retry"


def test_a_retry_of_a_refused_wake_reuses_the_same_name(tmp_path, monkeypatch):
    """Same-emitter idempotence stays §A's job: we handed our claim back, so the next tick
    re-claims the same event and rebuilds the same name — the fence replays one admission
    instead of queueing a second copy. If the release had been forgotten, this is the test
    that would show a silently dropped wake instead."""
    db = _board(tmp_path, monkeypatch, "retry-name")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done")

    recorder = RecordingWake(refuse_first=True)
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)
    first = recorder.names[0]
    _tick(monkeypatch)

    assert recorder.names == [first, first], (
        f"a retry of one event must still wake, under the same name: {recorder.names}")
    (event_id,) = _events(db, task_id, "completed")
    row = _claim_for(db, event_id)
    assert row["emitter"] == "gateway-notifier" and row["confirmed_at"] is not None, row


def test_a_wake_that_raises_releases_the_claim_too(tmp_path, monkeypatch):
    """The release is not only for ``WakeNotAccepted``: any failed wake hands the claim
    back, or the other emitter waits out the whole lease for a wake that never happened."""
    db = _board(tmp_path, monkeypatch, "raise")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done")
    (event_id,) = _events(db, task_id, "completed")

    recorder = RecordingWake(raise_first=True)
    profile = _profile()
    released = _spy(monkeypatch, "release_wake")
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert len(recorder.names) == 1, "the wake was attempted once"
    assert [c["event_id"] for c in released] == [event_id], (
        f"a failed wake hands the claim back too: {released}")
    with kbc.connect() as conn:
        assert kbn.wake_already_claimed(conn, profile=profile, task_id=task_id,
                                        event_ids=[event_id]) == set()


def test_a_partially_claimed_group_wakes_only_for_what_is_owed(tmp_path, monkeypatch):
    """A batch is not all-or-nothing. One wake-bearing event was already woken by the other
    emitter, one was not: the notifier still wakes — for the event it holds — and drops the
    claimed kind out of the wake text and out of §A's name."""
    from agent.i18n import t
    from collections import namedtuple
    from gateway.kanban_watchers_notifier import _wake_dedupe_token

    _Ev = namedtuple("_Ev", "id")  # the token reads ev.id off the claimed events

    db = _board(tmp_path, monkeypatch, "partial")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.block_task(conn, task_id, reason="needs input", kind="needs_input")
    (blocked_id,) = _events(db, task_id, "blocked")
    with kbc.connect() as conn:
        kb.unblock_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done")
    (unblocked_id,) = _events(db, task_id, "unblocked")
    (completed_id,) = _events(db, task_id, "completed")
    assert blocked_id < unblocked_id < completed_id

    profile = _profile()
    with kbc.connect() as conn:  # only the blocked event was already woken
        assert kbn.try_claim_wake(conn, profile=profile, task_id=task_id,
                                  event_id=blocked_id, emitter=WAKER) is True
        assert kbn.confirm_wake(conn, profile=profile, task_id=task_id,
                                event_id=blocked_id, emitter=WAKER) is True

    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert len(recorder.names) == 1, f"the unclaimed event still wakes: {recorder.names}"
    # The name is ``kanban:<board>:<task>:<sha1 of the event ids>``; read the board back off
    # it rather than re-deriving the slug, so the assertion is about the ids, not the pinning.
    parts = recorder.names[0].split(":")
    assert parts[0] == "kanban" and parts[2] == task_id, recorder.names[0]
    board = parts[1]
    # Pruning is by claimed EVENT, not by whole batch: the suppressed event leaves the name,
    # the silent-but-claimed ``unblocked`` row stays in it (it is part of this batch, it just
    # carries no wake of its own).
    assert recorder.names[0] == _wake_dedupe_token(
        board, task_id, [_Ev(unblocked_id), _Ev(completed_id)]), "the name covers exactly what is owed"
    assert recorder.names[0] != _wake_dedupe_token(
        board, task_id, [_Ev(blocked_id), _Ev(unblocked_id), _Ev(completed_id)]), (
        "the already-woken event must not survive in the name")
    assert recorder.names[0] != _wake_dedupe_token(board, task_id, [_Ev(completed_id)]), (
        "the non-wake-bearing rows of the batch stay in the name")
    assert t("gateway.kanban.wake.completed") in recorder.texts[0], recorder.texts[0]
    assert t("gateway.kanban.wake.blocked") not in recorder.texts[0], (
        f"the suppressed kind is out of the wake text: {recorder.texts[0]}")
    assert _claim_for(db, blocked_id)["emitter"] == WAKER
    assert _claim_for(db, completed_id)["emitter"] == "gateway-notifier"
    assert _cursor(db, task_id) == completed_id


def test_a_claim_on_a_non_wake_bearing_event_never_suppresses(tmp_path, monkeypatch):
    """The claim set is exactly the group's ``_WAKE_KINDS`` ids, so a kind outside it is
    invisible to suppression however many rows exist for it: ``spawn_failed`` is not even
    claimed by the notifier, and a ``status`` row planted for its own event id leaves the
    wake for the ``completed`` event intact."""
    db = _board(tmp_path, monkeypatch, "non-wake")
    task_id = _wake_subscribed_task(db)
    profile = _profile()
    with kbc.connect() as conn:
        kb._append_event(conn, task_id, "status", {"status": "ready"})
        kb._append_event(conn, task_id, "spawn_failed", {"error": "host gone"})
        kb.complete_task(conn, task_id, summary="done")
    (status_id,) = _events(db, task_id, "status")
    (spawn_id,) = _events(db, task_id, "spawn_failed")
    (completed_id,) = _events(db, task_id, "completed")

    with kbc.connect() as conn:  # someone claimed the non-wake-bearing events already
        for event_id in (status_id, spawn_id):
            assert kbn.try_claim_wake(conn, profile=profile, task_id=task_id,
                                      event_id=event_id, emitter=WAKER) is True
            assert kbn.confirm_wake(conn, profile=profile, task_id=task_id,
                                    event_id=event_id, emitter=WAKER) is True

    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert len(recorder.names) == 1, "rows on non-wake-bearing events mute nothing"
    assert _claim_for(db, status_id)["emitter"] == WAKER, "the notifier never claimed the status event"
    assert _claim_for(db, spawn_id)["emitter"] == WAKER, "the notifier never claimed the spawn_failed event"
    assert _claim_for(db, completed_id)["emitter"] == "gateway-notifier"


def test_an_absent_claim_table_still_delivers(tmp_path, monkeypatch):
    """Fail-open is the posture: a board that predates the table (a foreign opener, a
    half-migrated file) must not cost the user their wake."""
    db = _board(tmp_path, monkeypatch, "no-table")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done")
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE kanban_wake_claims")

    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert len(recorder.names) == 1, f"an unreadable claim table delivers, never drops: {recorder.names}"


def test_a_claim_helper_that_raises_entirely_still_delivers(tmp_path, monkeypatch):
    """The wrapper's own fail-open: an error that is not even a ``sqlite3.Error`` — a
    broken helper, a monkeypatched seam — is answered as 'nothing was claimed'."""
    db = _board(tmp_path, monkeypatch, "helper-raises")
    task_id = _wake_subscribed_task(db)
    with kbc.connect() as conn:
        kb.complete_task(conn, task_id, summary="done")

    def _boom(*args, **kwargs):
        raise RuntimeError("claim helper unavailable")

    monkeypatch.setattr(kbn, "try_claim_wake", _boom)
    recorder = RecordingWake()
    monkeypatch.setattr(wake_mod, "deliver_wake", recorder)
    _tick(monkeypatch)

    assert len(recorder.names) == 1, f"a failing claim delivers: {recorder.names}"


def test_the_claim_profile_is_the_profile_the_wake_lands_in():
    """The key both emitters must agree on, without a gateway: a stamped subscription is
    woken AS that profile; an un-stamped one is woken by this gateway's own profile; and a
    runner that cannot say which profile it is falls back to the default."""
    from gateway.kanban_watchers_notifier import _KanbanNotification

    def _named(sub_profile, active=None):
        n = _KanbanNotification.__new__(_KanbanNotification)
        n.sub_profile = sub_profile
        n.runner = type("_Runner", (), ({"_active_profile_name": lambda self: active}
                                        if active is not None else {}))()
        return n._wake_claim_profile()

    assert _named("hscc-orch") == "hscc-orch"
    assert _named("", "worker") == "worker"
    assert _named("") == "default"
