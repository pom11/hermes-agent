"""Consented authorized-controller handoff (\"remote control\") — registry primitives.

Exclusivity is a correctness guarantee (at most one live owner per stored session,
enforced unconditionally). The handoff does NOT weaken the fence: it is a CONSENTED
transfer, proven by the owner (same pid + live_session_id) granting a ONE-SHOT nonce
that only the named controller can present. These tests exercise the three registry
steps -- request / grant / complete -- including every denial path, so a controller
can never self-approve and a third party can never ride a consent meant for someone
else. The cross-process shape (two real gateway processes, owner consents via its
client) is covered by scripts/probe_active_session_handoff.py.
"""

import os
import time

import pytest

from hermes_cli import active_sessions


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _acquire_owner(home, session_id="sess-1", live_session_id="live-owner",
                   surface="cli"):
    lease, message = active_sessions.try_acquire_active_session(
        session_id=session_id, surface=surface, config={},
        metadata={"live_session_id": live_session_id},
    )
    assert lease is not None and message is None
    return lease


def test_full_handoff_round_trip_transfers_lease(home):
    """Controller requests -> owner grants -> controller completes -> one owner remains."""
    owner_lease = _acquire_owner(home)

    # 1. Controller cannot simply acquire: still fenced by exclusivity.
    blocked, refusal = active_sessions.try_acquire_active_session(
        session_id="sess-1", surface="desktop", config={},
        metadata={"live_session_id": "live-controller"},
    )
    assert blocked is None and getattr(refusal, "reason", None) == active_sessions.SESSION_NOT_OWNED

    # 2. Controller requests the handoff; a pending ask is stamped; not yet granted.
    pending, err = active_sessions.request_active_session_handoff(
        "sess-1", controller="hscc-app")
    assert err is None and pending["granted"] is False
    assert pending["owner"]["pid"] == os.getpid()
    assert pending["owner"]["surface"] == "cli"

    # 3. The OWNER grants: only it can (same pid + live_session_id). The nonce is one-shot.
    nonce = active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner")
    assert nonce

    # 4. Controller completes with the nonce and gets a live lease.
    controller_lease, err = active_sessions.complete_active_session_handoff(
        "sess-1", controller="hscc-app", nonce=nonce, surface="desktop",
        metadata={"live_session_id": "live-controller"},
    )
    assert err is None and controller_lease is not None
    assert controller_lease.session_id == "sess-1"

    # Exactly one live owner remains, now the controller (session survives, not fenced).
    entries = active_sessions._read_entries(active_sessions._state_path(home))
    assert len(entries) == 1
    assert entries[0]["surface"] == "desktop"
    assert (entries[0].get("metadata") or {}).get("live_session_id") == "live-controller"

    # The owner's old lease object can no longer release the controller's entry:
    # release() drops only the entry matching its own (lease_id, pid) identity, and the
    # entry now carries the controller's pid, so the controller's turn is untouched.
    owner_lease.release()
    still = active_sessions._read_entries(active_sessions._state_path(home))
    assert len(still) == 1 and still[0]["surface"] == "desktop"
    assert (still[0].get("metadata") or {}).get("live_session_id") == "live-controller"

    # Controller releases at turn end: the session is free again.
    controller_lease.release()
    assert active_sessions._read_entries(active_sessions._state_path(home)) == []


def test_nonce_is_one_shot_replay_refused(home):
    """A consumed consent cannot be replayed: the second completion must refuse."""
    _acquire_owner(home)
    active_sessions.request_active_session_handoff("sess-1", controller="hscc-app")
    nonce = active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner")
    assert nonce

    first, err = active_sessions.complete_active_session_handoff(
        "sess-1", controller="hscc-app", nonce=nonce, surface="desktop")
    assert err is None and first is not None

    # The handoff record is consumed; the same nonce no longer completes.
    second, err = active_sessions.complete_active_session_handoff(
        "sess-1", controller="hscc-app", nonce=nonce, surface="desktop")
    assert second is None
    assert getattr(err, "reason", None) == active_sessions.SESSION_NOT_OWNED


def test_controller_cannot_self_approve(home):
    """Without the owner's live_session_id, a grant is refused: no consent, no takeover."""
    _acquire_owner(home)
    active_sessions.request_active_session_handoff("sess-1", controller="hscc-app")

    # The controller does not know the owner's live_session_id; even if it guesses the
    # pid, _is_same_writer needs BOTH (pid AND live_session_id) to match.
    assert active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-controller") is None

    # Nothing transferred, still a pending (ungranted) ask — the owner can still consent.
    nonce = active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner")
    assert nonce


def test_wrong_controller_cannot_ride_a_consent(home):
    """A consent granted to controller A cannot be presented by controller B."""
    _acquire_owner(home)
    active_sessions.request_active_session_handoff("sess-1", controller="hscc-app")
    nonce = active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner")
    assert nonce

    lease, err = active_sessions.complete_active_session_handoff(
        "sess-1", controller="other-app", nonce=nonce, surface="desktop")
    assert lease is None
    assert getattr(err, "reason", None) == active_sessions.SESSION_NOT_OWNED

    # The consent is intact for the named controller: it still completes.
    lease, err = active_sessions.complete_active_session_handoff(
        "sess-1", controller="hscc-app", nonce=nonce, surface="desktop")
    assert lease is not None and err is None


def test_request_with_no_live_owner_refuses(home):
    """No live owner => nothing to hand off, typed SESSION_NOT_OWNED (not a silent pass)."""
    pending, err = active_sessions.request_active_session_handoff(
        "ghost", controller="hscc-app")
    assert pending is None
    assert getattr(err, "reason", None) == active_sessions.SESSION_NOT_OWNED


def test_request_requires_controller_identity(home):
    """A nameless controller cannot even place a handoff request."""
    _acquire_owner(home)
    pending, err = active_sessions.request_active_session_handoff("sess-1", controller="")
    assert pending is None
    assert getattr(err, "reason", None) == active_sessions.SESSION_COORDINATION_UNAVAILABLE


def test_grant_requires_a_pending_request(home):
    """An owner cannot grant a handoff nobody asked for — consent answers a request."""
    owner_lease = _acquire_owner(home)
    assert active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner") is None
    owner_lease.release()


def test_grant_targets_only_the_named_controller(home):
    """A grant for controller A does nothing for a pending request from controller B."""
    _acquire_owner(home)
    active_sessions.request_active_session_handoff("sess-1", controller="hscc-app")
    assert active_sessions.grant_active_session_handoff(
        "sess-1", controller="other-app", live_session_id="live-owner") is None
    # Still pending for the right controller — it can still grant.
    assert active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner")


def test_owner_release_before_completion_cancels_handoff(home):
    """If the owner exits/relinquishes before the controller completes, the pending
    request vanishes with it — no consent can outlive its owner's lease."""
    owner_lease = _acquire_owner(home)
    active_sessions.request_active_session_handoff("sess-1", controller="hscc-app")
    nonce = active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner")
    assert nonce

    owner_lease.release()  # owner finished / revoked before handing over

    lease, err = active_sessions.complete_active_session_handoff(
        "sess-1", controller="hscc-app", nonce=nonce, surface="desktop")
    assert lease is None
    assert getattr(err, "reason", None) == active_sessions.SESSION_NOT_OWNED


def test_second_request_is_idempotent_within_ttl(home):
    """A repeated request from the same controller does not stack asks or reset a grant."""
    _acquire_owner(home)
    active_sessions.request_active_session_handoff("sess-1", controller="hscc-app")
    pending, err = active_sessions.request_active_session_handoff(
        "sess-1", controller="hscc-app")
    assert err is None and pending["granted"] is False

    nonce = active_sessions.grant_active_session_handoff(
        "sess-1", controller="hscc-app", live_session_id="live-owner")
    # A third request now reports granted=True so the controller knows it may complete.
    pending, err = active_sessions.request_active_session_handoff(
        "sess-1", controller="hscc-app")
    assert err is None and pending["granted"] is True
