"""Notification subscriptions consumed by the gateway kanban-notifier: per-(task, platform, chat, thread) rows with delivery metadata, unseen-event cursors and purge of stale done-task subs.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Event


# Notifier reaction to a terminal event: "notify" = passive adapter.send only
# (default); "notify+wake" = send AND wake the destination agent; "wake" = wake only.
_NOTIFY_DELIVERY_MODES = ("notify", "notify+wake", "wake")

_SCALAR_TYPES = (str, int, float, bool)

# Subscription primary key predicate; every per-row statement below binds
# ``(task_id, platform, chat_id, thread_id or "")`` against it.
_SUB_KEY_WHERE = "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?"


def _sub_key(task_id: str, platform: str, chat_id: str, thread_id: Optional[str]) -> tuple:
    return (task_id, platform, chat_id, thread_id or "")


def _encode_notify_delivery_metadata(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Serialize platform send metadata stored on notification subscriptions."""
    if not isinstance(metadata, Mapping):
        return None
    clean = {
        str(key): value
        for key, value in metadata.items()
        if value is not None and isinstance(value, _SCALAR_TYPES)
    }
    if not clean:
        return None
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _decode_notify_delivery_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, _SCALAR_TYPES)}


def add_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_id_alt: Optional[str] = None,
    chat_type: Optional[str] = None,
    notifier_profile: Optional[str] = None,
    delivery_mode: Optional[str] = None,
    delivery_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Register a gateway source wanting terminal-state notifications for
    ``task_id``; idempotent on (task, platform, chat, thread).

    ``user_id_alt`` (Signal UUID, Feishu union_id, ...) and ``chat_type`` are
    replayed on active wake: ``build_session_key`` prefers the alt id, so
    omitting it would key the wake into a different session. ``None`` keeps an
    existing row's value. ``delivery_mode``: ``None`` leaves an existing row
    untouched, an explicit valid value is last-write-wins, unknown falls back
    to ``"notify"``. ``delivery_metadata`` merges supplied routing anchors
    into an existing row so re-subscribing never discards them. New subs start
    caught up (``last_event_id`` =
    ``MAX(task_events.id)``) so the notifier never replays history at boot.
    """
    valid_mode = delivery_mode if delivery_mode in _NOTIFY_DELIVERY_MODES else None
    # api_server is stateless: the adapter has no send(), the wake self-post IS
    # the delivery. A plain 'notify' default would leave those subs with no
    # delivery mechanism at all. Explicit modes still win.
    insert_mode = valid_mode or ("notify+wake" if platform == "api_server" else "notify")
    key = _sub_key(task_id, platform, chat_id, thread_id)
    with _kb.write_txn(conn):
        existing = conn.execute(
            "SELECT delivery_metadata FROM kanban_notify_subs " + _SUB_KEY_WHERE,
            key,
        ).fetchone()
        existing_metadata = _decode_notify_delivery_metadata(existing["delivery_metadata"]) if existing else {}
        merged_metadata = dict(existing_metadata)
        if delivery_metadata:
            merged_metadata.update(delivery_metadata)
        metadata_json = _encode_notify_delivery_metadata(merged_metadata) if merged_metadata else None
        conn.execute(
            """
            INSERT OR IGNORE INTO kanban_notify_subs
                (task_id, platform, chat_id, thread_id, user_id, user_id_alt,
                 chat_type, notifier_profile, delivery_mode, delivery_metadata,
                 created_at, last_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT MAX(id) FROM task_events WHERE task_id = ?), 0))
            """,
            (
                *key, user_id, user_id_alt, chat_type or "dm", notifier_profile,
                insert_mode, metadata_json, int(time.time()), task_id,
            ),
        )
        # chat_type / delivery_mode are last-write-wins; delivery metadata
        # preserves existing routing fields while supplied fields overwrite them.
        # user_id, user_id_alt and notifier_profile only self-heal legacy rows lacking one.
        for column, value, fill_only in (
            ("chat_type", chat_type, False),
            ("user_id", user_id, True),
            ("user_id_alt", user_id_alt, True),
            ("notifier_profile", notifier_profile, True),
            ("delivery_mode", valid_mode, False),
            ("delivery_metadata", metadata_json, False),
        ):
            if not value:
                continue
            guard = f" AND ({column} IS NULL OR {column} = '')" if fill_only else ""
            conn.execute(
                f"UPDATE kanban_notify_subs SET {column} = ? " + _SUB_KEY_WHERE + guard,
                (value, *key),
            )


def _notify_profile_filter(
    notifier_profiles: Optional[Iterable[str]],
    *,
    include_unowned: bool,
) -> tuple[str, list[str]]:
    """Build an optional SQL predicate for notification profile ownership."""
    if notifier_profiles is None:
        return "", []

    profiles = sorted({str(p).strip() for p in notifier_profiles if str(p).strip()})
    clauses: list[str] = []
    params: list[str] = []
    if profiles:
        clauses.append("notifier_profile IN (" + ",".join("?" for _ in profiles) + ")")
        params.extend(profiles)
    if include_unowned:
        clauses.append("notifier_profile IS NULL OR notifier_profile = ''")
    if not clauses:
        return "0", []
    return "(" + ") OR (".join(clauses) + ")", params


def list_notify_subs(
    conn: sqlite3.Connection,
    task_id: Optional[str] = None,
    *,
    notifier_profiles: Optional[Iterable[str]] = None,
    include_unowned: bool = False,
) -> list[dict]:
    """List subscriptions, optionally restricted to notifier profile owners.

    No ``notifier_profiles`` -> all subscriptions. Gateway notifiers pass the
    profiles they own so they cannot claim another gateway's events;
    ``include_unowned`` (dispatch owner) covers legacy rows without a stamp.
    """
    owner_where, owner_params = _notify_profile_filter(
        notifier_profiles, include_unowned=include_unowned,
    )
    where: list[str] = []
    params: list[Any] = []
    if task_id is not None:
        where.append("task_id = ?")
        params.append(task_id)
    if owner_where:
        where.append(owner_where)
        params.extend(owner_params)
    sql = "SELECT * FROM kanban_notify_subs"
    if where:
        sql += " WHERE " + " AND ".join(f"({clause})" for clause in where)
    out: list[dict] = []
    for row in conn.execute(sql, params).fetchall():
        item = dict(row)
        if "delivery_metadata" in item:
            item["delivery_metadata"] = _decode_notify_delivery_metadata(item.get("delivery_metadata"))
        out.append(item)
    return out


def count_notify_subs(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
    notifier_profiles: Optional[Iterable[str]] = None,
    include_unowned: bool = False,
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> int:
    """Count ``kanban_notify_subs`` rows via a read-only connection — the
    notifier's cheap zero-subscription early exit. Unlike :func:`connect` it
    never creates the file, runs init/migration or opens writable; WAL rows are
    still visible so a fresh sub is never missed. Missing DB / missing table
    counts as zero; platform matches case-insensitively (as notifier routing),
    chat/thread exactly. Raises :class:`sqlite3.Error` if the DB exists but is
    unreadable — callers pick their own fallback.
    """
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    if not path.exists():
        return 0
    owner_where, owner_params = _notify_profile_filter(
        notifier_profiles, include_unowned=include_unowned,
    )
    clauses: list[str] = []
    params: list[Any] = []
    if owner_where:
        clauses.append(f"({owner_where})")
        params.extend(owner_params)
    for clause, value in (
        ("LOWER(platform) = LOWER(?)", platform),
        ("chat_id = ?", chat_id),
        ("thread_id = ?", thread_id),
    ):
        if value is not None:
            clauses.append(clause)
            params.append(value)
    query = "SELECT COUNT(*) FROM kanban_notify_subs"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            row = conn.execute(query, params).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return 0
            raise
        return int(row[0]) if row else 0
    finally:
        conn.close()


def remove_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> bool:
    with _kb.write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs " + _SUB_KEY_WHERE,
            _sub_key(task_id, platform, chat_id, thread_id),
        )
    return cur.rowcount > 0


def retarget_notify_subs(
    conn: sqlite3.Connection, *, platform: str, from_chat_ids: Iterable[str], to_chat_id: str,
) -> int:
    """Re-key subscriptions bound to any of ``from_chat_ids`` onto ``to_chat_id`` (cursor kept), returning
    how many moved. A row whose new key already exists is dropped: the owner's own subscription wins."""
    ids = sorted({c for c in from_chat_ids if c and c != to_chat_id})
    if not ids:
        return 0
    moved = 0
    with _kb.write_txn(conn):
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            where = f"LOWER(platform) = LOWER(?) AND chat_id IN ({','.join('?' * len(chunk))})"
            moved += conn.execute(
                f"UPDATE OR IGNORE kanban_notify_subs SET chat_id = ? WHERE {where}", (to_chat_id, platform, *chunk),
            ).rowcount
            conn.execute(f"DELETE FROM kanban_notify_subs WHERE {where}", (platform, *chunk))
    return moved


def purge_stale_done_notify_subs(conn: sqlite3.Connection, *, max_age_days: int = 30) -> int:
    """Delete notify subs whose task sat in ``done``/``blocked`` untouched for
    longer than ``max_age_days`` (``<= 0`` disables); returns rows deleted.

    Subs survive ``done`` because a reopened task must still notify its origin,
    which accumulates forever on never-archiving boards. ``blocked`` is
    abandoned (unlike ``backlog``/``ready``) so it reaps on the same clock. Age
    = latest event, else ``completed_at``, else ``created_at`` — any activity,
    including a reopen, exempts the sub.

    The notifier keeps subscriptions alive through ``done`` because a completed task can be reopened (review
    corrections, continuation) and the reopened cycle must still notify its origin session. On boards that
    never archive, that retention would otherwise accumulate subscription rows forever — each one scanned
    every notifier tick. This GC bounds that: a task that has been ``done`` with no new events for the
    retention window is treated as settled and its subscriptions are purged. ``blocked`` tasks
    (circuit-breaker trips, dead workers) are reaped on the same clock — they are abandoned, not idle,
    unlike a ``backlog``/``ready`` card that is merely waiting for pickup (#100955).
    """
    try:
        days = int(max_age_days)
    except (TypeError, ValueError):
        days = 30
    if days <= 0:
        return 0
    cutoff = int(time.time()) - days * 86400
    with _kb.write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs WHERE task_id IN ("
            " SELECT t.id FROM tasks t"
            " WHERE t.status IN ('done', 'blocked')"
            " AND COALESCE("
            "  (SELECT MAX(e.created_at) FROM task_events e"
            "   WHERE e.task_id = t.id),"
            "  t.completed_at, t.created_at, 0"
            " ) < ?)",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


def _notify_cursor(
    conn: sqlite3.Connection, task_id: str, platform: str, chat_id: str, thread_id: Optional[str],
) -> Optional[int]:
    """``last_event_id`` of one subscription row, or ``None`` when unsubscribed."""
    row = conn.execute(
        "SELECT last_event_id FROM kanban_notify_subs " + _SUB_KEY_WHERE,
        _sub_key(task_id, platform, chat_id, thread_id),
    ).fetchone()
    return None if row is None else int(row["last_event_id"])


def unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, list[Event]]:
    """Return ``(new_cursor, events)`` with ``id > last_event_id``. The cursor
    is NOT advanced here; call :func:`advance_notify_cursor` after delivery.
    """
    cursor = _notify_cursor(conn, task_id, platform, chat_id, thread_id)
    if cursor is None:
        return 0, []
    kind_list = list(kinds) if kinds else None
    q = (
        "SELECT * FROM task_events WHERE task_id = ? AND id > ? "
        + ("AND kind IN (" + ",".join("?" * len(kind_list)) + ") " if kind_list else "")
        + "ORDER BY id ASC"
    )
    params: list[Any] = [task_id, cursor]
    if kind_list:
        params.extend(kind_list)
    rows = conn.execute(q, params).fetchall()
    out = [_kb.Event.from_row(r) for r in rows]
    max_id = max([cursor, *(int(r["id"]) for r in rows)])
    return max_id, out


def claim_unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, int, list[Event]]:
    """Atomically claim unseen events for one subscription.

    Returns ``(old_cursor, new_cursor, events)``; when events are returned the
    row's ``last_event_id`` has already been advanced inside ``BEGIN IMMEDIATE``,
    so concurrent gateway watchers on the same board DB serialize on SQLite's
    writer lock and only the first claims a given event range. Callers send the
    events, then leave the cursor or call :func:`rewind_notify_cursor` on
    delivery failure.
    """
    with _kb.write_txn(conn):
        old_cursor = _notify_cursor(conn, task_id, platform, chat_id, thread_id)
        if old_cursor is None:
            return 0, 0, []
        new_cursor, events = unseen_events_for_sub(
            conn, task_id=task_id, platform=platform, chat_id=chat_id,
            thread_id=thread_id, kinds=kinds,
        )
        if not events:
            return old_cursor, old_cursor, []
        _cas_cursor(conn, _sub_key(task_id, platform, chat_id, thread_id), new_cursor, old_cursor)
        return old_cursor, new_cursor, events


def _cas_cursor(conn: sqlite3.Connection, key: tuple, new_cursor: int, expected: int) -> sqlite3.Cursor:
    """Move ``last_event_id`` only if it still equals ``expected``."""
    return conn.execute(
        "UPDATE kanban_notify_subs SET last_event_id = ? " + _SUB_KEY_WHERE + " AND last_event_id = ?",
        (int(new_cursor), *key, int(expected)),
    )


def advance_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    new_cursor: int,
) -> None:
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? " + _SUB_KEY_WHERE,
            (int(new_cursor), *_sub_key(task_id, platform, chat_id, thread_id)),
        )


def record_notify_ping(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str,
    thread_id: Optional[str] = None, event_id: int,
) -> None:
    """Checkpoint a sent ping independently of the retryable wake cursor."""
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_ping_event_id = MAX(last_ping_event_id, ?) "
            + _SUB_KEY_WHERE,
            (int(event_id), *_sub_key(task_id, platform, chat_id, thread_id)),
        )


def rewind_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    claimed_cursor: int,
    old_cursor: int,
) -> bool:
    """Undo a claim when delivery fails. The CAS guard only rewinds if no later
    notifier advanced the row, so retries never clobber newer progress.
    """
    with _kb.write_txn(conn):
        cur = _cas_cursor(conn, _sub_key(task_id, platform, chat_id, thread_id), old_cursor, claimed_cursor)
    return cur.rowcount > 0


# --- Wake-once claim (hermes-agent#152206 §3.1) ----------------------------
#
# ``kanban_notify_subs.last_event_id`` says "the notifier consumed this event",
# not "the recipient was woken", so a second emitter that wakes the same profile
# for the same ``task_events`` row has nothing to ask. These four functions are
# that record: one row per (profile, task, event) in the board DB both emitters
# already open, keyed by event identity alone. Callers that decide whether to
# deliver must treat every unknown as "deliver" — see :func:`_wake_claim_open`.
#
# Suppression is only ever as wide as what a caller claims, so a kind that must
# never wake (``spawn_failed`` and the rest outside the notifier's ``_WAKE_KINDS``)
# simply is never claimed and can never suppress anything. The kind filter stays
# with the emitters; this table holds no opinion about it.

# Lease on an unconfirmed claim: long enough to cover a slow send, short enough
# that an emitter crashing between claim and wake mutes one event briefly rather
# than permanently. Not user-configurable — a claim that suppresses only what a
# live lease or a confirmed delivery covers can never drop a wake that would
# otherwise have fired.
_WAKE_CLAIM_LEASE_S = 120

_WAKE_CLAIM_KEY_WHERE = "WHERE profile = ? AND task_id = ? AND event_id = ?"


def _wake_claim_open(exc: sqlite3.Error, op: str, fallback: Any) -> Any:
    """Fail-open verdict for a claim read/write that could not happen.

    The answer is always the one that still delivers the wake (#152206 §4): a
    doubled wake is the status quo, a dropped one is a regression. A board whose
    DB predates the table is the expected shape on an un-migrated board — every
    hermes-opened board gets it from ``SCHEMA_SQL`` at connect(), so an absent
    table means a foreign or legacy opener, not a bug; it is not worth a
    WARNING on every notifier tick.
    """
    if "no such table" in str(exc).lower():
        _kb._log.debug("kanban wake claim: %s skipped, table absent (%s)", op, exc)
    else:
        _kb._log.warning(
            "kanban wake claim: %s failed (%s); failing open towards delivering", op, exc
        )
    return fallback


def _wake_claim_row(
    conn: sqlite3.Connection, *, profile: str, task_id: str, event_id: int
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT emitter, claimed_at, confirmed_at FROM kanban_wake_claims "
        + _WAKE_CLAIM_KEY_WHERE,
        (profile, task_id, int(event_id)),
    ).fetchone()


def try_claim_wake(
    conn: sqlite3.Connection,
    *,
    profile: str,
    task_id: str,
    event_id: int,
    emitter: str,
    now: Optional[int] = None,
    lease_s: int = _WAKE_CLAIM_LEASE_S,
) -> bool:
    """Take the wake claim for one event. ``True`` = you owe this wake.

    Succeeds only when the event is unclaimed or its claim is an *unconfirmed*
    lease that has run past ``lease_s`` — the expired take-over re-stamps both
    emitter and clock, so a crashed sender mutes the event for at most
    ``lease_s`` seconds. Anything else answers ``False``: a live lease means
    somebody else already owes it, a confirmed row means it was already
    delivered. Then call :func:`confirm_wake` once the wake was accepted, or
    :func:`release_wake` when it was not.

    Callers must not already hold a ``write_txn`` (same rule as the cursor CAS
    helpers above), and must not claim a non-wake-bearing kind — see the block
    comment on this group.
    """
    moment = int(now if now is not None else time.time())
    lease = int(lease_s)
    try:
        with _kb.write_txn(conn):
            row = _wake_claim_row(conn, profile=profile, task_id=task_id, event_id=event_id)
            if row is None:
                conn.execute(
                    "INSERT INTO kanban_wake_claims"
                    " (profile, task_id, event_id, emitter, claimed_at, confirmed_at)"
                    " VALUES (?, ?, ?, ?, ?, NULL)",
                    (profile, task_id, int(event_id), emitter, moment),
                )
                return True
            if row["confirmed_at"] is not None:
                return False
            if moment - int(row["claimed_at"]) < lease:
                return False
            # Expired lease. CAS on the expiry so a concurrent take-over that
            # committed between our SELECT and here loses instead of overwriting.
            taken = conn.execute(
                "UPDATE kanban_wake_claims"
                " SET emitter = ?, claimed_at = ?, confirmed_at = NULL "
                + _WAKE_CLAIM_KEY_WHERE
                + " AND confirmed_at IS NULL AND claimed_at <= ?",
                (emitter, moment, profile, task_id, int(event_id), moment - lease),
            )
            return taken.rowcount > 0
    except sqlite3.Error as exc:
        return _wake_claim_open(exc, "try_claim_wake", True)


def confirm_wake(
    conn: sqlite3.Connection,
    *,
    profile: str,
    task_id: str,
    event_id: int,
    emitter: str,
    now: Optional[int] = None,
) -> bool:
    """Settle the claim as delivered; ``True`` when this emitter's lease settled.

    Only the holder may confirm — one emitter must not spend another's lease —
    and an already-confirmed row is left alone (first delivery wins). A failed
    confirm leaves an unconfirmed lease, which stops suppressing at ``lease_s``:
    the cost is a doubled wake, never a dropped one.
    """
    moment = int(now if now is not None else time.time())
    try:
        with _kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE kanban_wake_claims SET confirmed_at = ? "
                + _WAKE_CLAIM_KEY_WHERE
                + " AND emitter = ? AND confirmed_at IS NULL",
                (moment, profile, task_id, int(event_id), emitter),
            )
        return cur.rowcount > 0
    except sqlite3.Error as exc:
        return _wake_claim_open(exc, "confirm_wake", False)


def release_wake(
    conn: sqlite3.Connection,
    *,
    profile: str,
    task_id: str,
    event_id: int,
    emitter: str,
) -> bool:
    """Drop this emitter's claim so the wake is owed again — call it when the
    send was refused (``WakeNotAccepted``) or raised, mirroring
    :func:`rewind_notify_cursor`'s discipline for the notifier's own cursor.

    A refused send must not mute the other emitter. Only the holder may release:
    a lease that already expired belongs to whoever took it over, and deleting
    that row would hand the same event to a third sender.
    """
    try:
        with _kb.write_txn(conn):
            cur = conn.execute(
                "DELETE FROM kanban_wake_claims " + _WAKE_CLAIM_KEY_WHERE + " AND emitter = ?",
                (profile, task_id, int(event_id), emitter),
            )
        return cur.rowcount > 0
    except sqlite3.Error as exc:
        return _wake_claim_open(exc, "release_wake", False)


def wake_already_claimed(
    conn: sqlite3.Connection,
    *,
    profile: str,
    task_id: str,
    event_ids: Iterable[int],
    now: Optional[int] = None,
    lease_s: int = _WAKE_CLAIM_LEASE_S,
) -> set[int]:
    """The subset of ``event_ids`` that must NOT be woken again for ``profile``.

    Read-only (safe on a read-only connection — the out-of-process waker opens
    the board that way) and the public "was this already woken?" hook. A
    confirmed delivery suppresses forever; an unconfirmed lease only until it
    expires. Any failure returns the empty set — nothing is known to be claimed,
    so the caller delivers; that is the whole point of the fail-open posture.
    """
    ids = sorted({int(e) for e in event_ids})
    if not ids:
        return set()
    moment = int(now if now is not None else time.time())
    cutoff = moment - int(lease_s)
    placeholders = ",".join("?" * len(ids))
    try:
        rows = conn.execute(
            "SELECT event_id FROM kanban_wake_claims"
            " WHERE profile = ? AND task_id = ?"
            f" AND event_id IN ({placeholders})"
            " AND (confirmed_at IS NOT NULL OR claimed_at > ?)",
            (profile, task_id, *ids, cutoff),
        ).fetchall()
    except sqlite3.Error as exc:
        return _wake_claim_open(exc, "wake_already_claimed", set())
    return {int(r["event_id"]) for r in rows}


def wake_claim_holder(
    conn: sqlite3.Connection, *, profile: str, task_id: str, event_id: int
) -> Optional[str]:
    """The emitter whose claim currently covers this event, or ``None``.

    ``None`` means nobody holds it (no row) *or* the read could not happen — both are
    "deliver". The question exists because a refusal to claim is not always another
    service's refusal: an emitter that fans one event out to several destinations (the
    notifier, one subscription per destination) legitimately holds the claim it is being
    refused by. Suppression is what a DIFFERENT emitter's claim means; an emitter's own
    claim says nothing about whether its next destination was woken — its cursor does.
    """
    try:
        row = conn.execute(
            "SELECT emitter FROM kanban_wake_claims " + _WAKE_CLAIM_KEY_WHERE,
            (profile, task_id, int(event_id)),
        ).fetchone()
    except sqlite3.Error as exc:
        return _wake_claim_open(exc, "wake_claim_holder", None)
    return str(row["emitter"]) if row is not None else None


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
