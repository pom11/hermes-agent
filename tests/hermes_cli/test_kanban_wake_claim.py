"""Wake-once claim (hermes-agent#152206 §3.1) — ``kanban_wake_claims``.

``kanban_notify_subs.last_event_id`` records what the *notifier* consumed, not
that a profile was woken, so a second emitter that wakes the same profile for
the same ``task_events`` row had nothing to ask and had to infer it. These
tests pin the record itself: one row per (profile, task, event), and the four
helpers over it.

The invariants that matter, each from the proposal's posture:

* A claim is taken once: the first emitter owes the wake, the next one does not.
* The key is event identity only — two events in the same second are two wakes,
  and a claim for one profile/task never speaks for another.
* An unconfirmed claim is a LEASE: past ``lease_s`` it stops suppressing, so a
  crashed emitter mutes one event briefly, never permanently.
* A refused send releases the claim, so failure never mutes the other emitter.
* Every unknown — absent table, half-migrated schema — fails open towards
  DELIVERING. A doubled wake is the status quo; a dropped one is a regression.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

NOTIFIER = "notifier"
WAKER = "hscc-orch-waker"
PROFILE = "hscc-orch"
SAME_SECOND = 1_760_000_000


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """An isolated HERMES_HOME with an initialized default board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task(conn) -> str:
    return kb.create_task(conn, title="wake me once", assignee="worker")


def _events(conn, task_id: str, kinds=("completed", "blocked"), *, created_at=SAME_SECOND):
    """Real ``task_events`` rows at an identical ``created_at`` — the identity
    the claim is keyed by, and the same ids a wake-dedupe token hashes."""
    ids = []
    with kb.write_txn(conn):
        for kind in kinds:
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, ?, NULL, ?)",
                (task_id, kind, created_at),
            )
            ids.append(int(conn.execute("SELECT last_insert_rowid()").fetchone()[0]))
    return ids


# --- schema / migration -----------------------------------------------------

def test_board_db_carries_the_claim_table(kanban_home):
    """A board opened by hermes has the table, and its key is event identity:
    (profile, task_id, event_id) — no clock column, no status column."""
    with kbc.connect_closing() as conn:
        cols = {c["name"] for c in conn.execute("PRAGMA table_info(kanban_wake_claims)")}
        assert cols == {
            "profile", "task_id", "event_id", "emitter", "claimed_at", "confirmed_at",
        }
        pk = [
            c["name"] for c in conn.execute("PRAGMA table_info(kanban_wake_claims)")
            if c["pk"]
        ]
        assert pk == ["profile", "task_id", "event_id"]


def test_legacy_board_gains_the_table_on_next_connect(kanban_home):
    """A board written before the migration is not left without a claim store:
    ``SCHEMA_SQL`` is re-run at init, so the additive table arrives by itself."""
    path = kb.kanban_db_path()
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute("DROP TABLE kanban_wake_claims")
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    kb.init_db()
    with kbc.connect_closing() as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kanban_wake_claims'"
        ).fetchone() is not None


# --- claim ------------------------------------------------------------------

def test_first_emitter_owes_the_wake_and_the_second_is_suppressed(kanban_home):
    """The whole point: whoever claims first owes the wake, and the other
    emitter — a different profile-keyed writer on the same board — stands down."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))

        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=WAKER,
        ) is True
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is False
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[event_id],
        ) == {event_id}


def test_claim_is_event_identity_not_wall_clock(kanban_home):
    """Two events for one task in the SAME second are two wakes. The claim is
    keyed by event id, so neither suppresses the other (#152206 §4, t_75b25a6e)."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        first, second = _events(conn, task_id, ("completed", "completed"))
        assert first != second  # distinct rows, identical created_at

        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=first, emitter=NOTIFIER,
        ) is True
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=second, emitter=NOTIFIER,
        ) is True
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[first, second],
        ) == {first, second}


def test_a_claim_never_speaks_for_another_profile_or_task(kanban_home):
    """Keyed on the RECIPIENT profile: a profile nobody woke for this event is
    still owed its wake, even while another profile's claim is live."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        other_task = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))
        (other_event,) = _events(conn, other_task, ("completed",))

        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=WAKER,
        ) is True
        # Another recipient profile, same event: still owed its wake.
        assert kbn.wake_already_claimed(
            conn, profile="someone-else", task_id=task_id, event_ids=[event_id],
        ) == set()
        assert kbn.try_claim_wake(
            conn, profile="someone-else", task_id=task_id, event_id=event_id,
            emitter=NOTIFIER,
        ) is True
        # Same profile, another task's event: also still owed.
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=other_task, event_ids=[other_event],
        ) == set()
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=other_task, event_id=other_event,
            emitter=NOTIFIER,
        ) is True


def test_claim_is_scoped_to_what_was_claimed_so_other_kinds_never_suppress(kanban_home):
    """Kind filtering belongs to the emitters: an event nobody claimed (because
    ``spawn_failed`` is not wake-bearing) is never reported as claimed."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        wake_event, spawn_failed = _events(
            conn, task_id, ("completed", "spawn_failed"),
        )
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=wake_event, emitter=NOTIFIER,
        ) is True
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id,
            event_ids=[wake_event, spawn_failed],
        ) == {wake_event}


# --- lease ------------------------------------------------------------------

def test_live_lease_suppresses_until_it_expires_then_is_reclaimable(kanban_home):
    """An unconfirmed claim stops suppressing at ``lease_s``, and the take-over
    re-stamps emitter + clock so a crash mutes for at most one lease."""
    t0 = int(time.time())
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))

        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id,
            emitter=WAKER, now=t0, lease_s=120,
        ) is True
        # Inside the lease: the waker is mid-send, the notifier stands down.
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[event_id],
            now=t0 + 119, lease_s=120,
        ) == {event_id}
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id,
            emitter=NOTIFIER, now=t0 + 119, lease_s=120,
        ) is False
        # Past the lease: the waker crashed without confirming; the event is
        # claimable again and the notifier owes the wake.
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[event_id],
            now=t0 + 120, lease_s=120,
        ) == set()
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id,
            emitter=NOTIFIER, now=t0 + 120, lease_s=120,
        ) is True
        row = conn.execute(
            "SELECT emitter, claimed_at, confirmed_at FROM kanban_wake_claims "
            "WHERE profile = ? AND task_id = ? AND event_id = ?",
            (PROFILE, task_id, event_id),
        ).fetchone()
        assert (row["emitter"], row["claimed_at"], row["confirmed_at"]) == (
            NOTIFIER, t0 + 120, None,
        )


def test_confirmed_delivery_suppresses_long_after_the_lease(kanban_home):
    """A confirmed claim is a receipt, not a lease: it never expires, because
    the wake really happened."""
    t0 = int(time.time())
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id,
            emitter=WAKER, now=t0,
        ) is True
        assert kbn.confirm_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=WAKER,
        ) is True

        much_later = t0 + 10 * 86400
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[event_id],
            now=much_later, lease_s=120,
        ) == {event_id}
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id,
            emitter=NOTIFIER, now=much_later,
        ) is False


def test_only_the_holder_may_confirm_or_release(kanban_home):
    """One emitter must never settle or drop another's claim — that would turn
    an unconfirmed lease into a fake receipt, or a receipt into silence."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=WAKER,
        ) is True

        assert kbn.confirm_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is False
        assert kbn.release_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is False
        # Still the waker's live lease: the notifier is not owed the wake.
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is False

        assert kbn.confirm_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=WAKER,
        ) is True
        # Settled once; a second confirm by the same holder changes nothing.
        assert kbn.confirm_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=WAKER,
        ) is False


# --- release on a refused send ----------------------------------------------

def test_release_after_a_refused_send_lets_the_other_emitter_wake(kanban_home):
    """``WakeNotAccepted`` must not mute the other emitter: releasing hands the
    event straight back, with no lease to wait out."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is True
        assert kbn.release_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is True
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[event_id],
        ) == set()
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=WAKER,
        ) is True
        # Nothing left to release the second time.
        assert kbn.release_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is False


# --- the read-only ask ------------------------------------------------------

def test_wake_already_claimed_partitions_a_mixed_batch(kanban_home):
    """One tick asks about a whole claimed range: confirmed, live-leased,
    lease-expired and unclaimed events must be told apart."""
    t0 = int(time.time())
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        confirmed, live, expired, unclaimed = _events(
            conn, task_id, ("completed",) * 4,
        )
        for event_id, emitter, at in (
            (confirmed, WAKER, t0 - 10 * 86400), (live, WAKER, t0 - 10),
            (expired, WAKER, t0 - 400),
        ):
            assert kbn.try_claim_wake(
                conn, profile=PROFILE, task_id=task_id, event_id=event_id,
                emitter=emitter, now=at,
            ) is True
        assert kbn.confirm_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=confirmed, emitter=WAKER,
        ) is True

        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id,
            event_ids=[confirmed, live, expired, unclaimed], now=t0,
        ) == {confirmed, live}
        # An empty batch asks nothing and never touches the DB.
        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[],
        ) == set()


def test_the_ask_works_on_a_read_only_connection(kanban_home):
    """The out-of-process waker opens the board read-only; the public hook must
    answer from a read-only connection (claims are visible through WAL)."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is True
        board_path = kb.kanban_db_path()

    ro = sqlite3.connect(board_path.resolve().as_uri() + "?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    try:
        assert kbn.wake_already_claimed(
            ro, profile=PROFILE, task_id=task_id, event_ids=[event_id],
        ) == {event_id}
    finally:
        ro.close()


# --- fail open towards delivering -------------------------------------------

def test_absent_table_fails_open_towards_delivering(kanban_home):
    """A board whose DB predates the table behaves exactly like today's: the
    ask says nothing is claimed and every claim is granted, without raising."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))
        with kb.write_txn(conn):
            conn.execute("DROP TABLE kanban_wake_claims")

        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[event_id],
        ) == set()
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is True
        assert kbn.confirm_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is False
        assert kbn.release_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is False


def test_half_migrated_schema_fails_open_towards_delivering(kanban_home):
    """A table that exists but is not the one the code expects (a partially
    applied migration) is an unknown, and an unknown means deliver."""
    with kbc.connect_closing() as conn:
        task_id = _task(conn)
        (event_id,) = _events(conn, task_id, ("completed",))
        with kb.write_txn(conn):
            conn.execute(
                "ALTER TABLE kanban_wake_claims RENAME COLUMN confirmed_at TO wrong_name"
            )

        assert kbn.wake_already_claimed(
            conn, profile=PROFILE, task_id=task_id, event_ids=[event_id],
        ) == set()
        assert kbn.try_claim_wake(
            conn, profile=PROFILE, task_id=task_id, event_id=event_id, emitter=NOTIFIER,
        ) is True
