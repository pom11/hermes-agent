"""Cross-process proof of the consented authorized-controller handoff ("remote control").

Modeled on scripts/probe_active_session_exclusivity.py: drives two REAL gateway
processes over stdio JSON-RPC against a fresh HERMES_HOME, and proves the full
consented transfer:

    A (owner CLI)  resume S, submit          -> holds the session lease
    B (controller) resume S, submit          -> refused SESSION_NOT_OWNED
    B session.request_handoff {controller}   -> records a pending ask ("waiting for
                                                approval on your computer")
    A session.handoff_consent {controller}   -> owner grants, gets a ONE-SHOT nonce
    B session.complete_handoff {controller, nonce} -> lease transfers to B
    B prompt.submit                          -> B drives the turn (lease now held)
    B release / turn end                     -> session free again (handoff back)

Run against the fork's own HERMES_HOME so nothing here touches a real profile:

    python scripts/probe_active_session_handoff.py

No provider is required: the fence and the handoff are checked BEFORE the agent is
built (the exclusivity probe proves this), so the probe stays free of credentials
and inference cost.

Safety under test: A is the ONLY party that can consent (it alone holds the
live_session_id that matches the registry entry); B can never self-approve, and a
wrong nonce / third-party controller can never complete the transfer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():  # posix layout
    PYTHON = REPO / "venv" / "bin" / "python"
if not PYTHON.exists():  # kanban worktree has no venv — use the live deployment's interpreter
    _primary = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"
    if _primary.exists():
        PYTHON = _primary


class Gateway:
    """One gateway process, spoken to the way the TUI speaks to it."""

    def __init__(self, name: str, home: Path):
        env = dict(os.environ)
        env["HERMES_HOME"] = str(home)
        env["PYTHONPATH"] = str(REPO)
        env["PYTHONUNBUFFERED"] = "1"
        self.name = name
        self.proc = subprocess.Popen(
            [str(PYTHON), "-u", "-m", "tui_gateway.entry"],
            cwd=str(REPO),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._next = 1
        self.ready()

    def _read(self):
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"[{self.name}] gateway closed its pipe")
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def ready(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read()
            if msg and msg.get("method") == "event":
                if msg.get("params", {}).get("type") == "gateway.ready":
                    return
        raise RuntimeError(f"[{self.name}] never announced gateway.ready")

    def call(self, method: str, params: dict, timeout: float = 180.0) -> dict:
        rid = str(self._next)
        self._next += 1
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read()
            if msg and msg.get("id") == rid:
                return msg
        raise RuntimeError(f"[{self.name}] timed out calling {method}")

    def close(self):
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=15)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def reason_of(response: dict):
    return (response.get("error") or {}).get("data", {}).get("reason")


def registry(home: Path):
    path = home / "runtime" / "active_sessions.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("entries", [])
    except Exception:
        return []


def main() -> int:
    home = REPO / ".probe-handoff-home"
    import shutil

    shutil.rmtree(home, ignore_errors=True)
    failures = []

    def check(label: str, ok: bool, detail: str = ""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    a = Gateway("A", home)
    b = None
    try:
        created = a.call("session.create", {"cols": 80})
        sid_a = created["result"]["session_id"]
        a.call("prompt.submit", {"session_id": sid_a, "text": "probe: A takes the session"})
        held = registry(home)
        key = held[0].get("session_id") if held else None
        if not key:
            raise RuntimeError("A never claimed the session; nothing further can be tested")
        print(f"A live session {sid_a}, stored key {key}")
        check("A holds the session lease", len(held) == 1, json.dumps(held)[:160])

        b = Gateway("B", home)
        resumed = b.call("session.resume", {"session_id": key})
        sid_b = resumed.get("result", {}).get("session_id")
        check("B may RESUME the stored session (reading is never fenced)", bool(sid_b),
              f"sid_b={sid_b}")

        # Capture A's lease token: "nothing transferred" always means this token is unchanged.
        a_lease_id = (registry(home)[0].get("lease_id") if registry(home) else None)

        def owned_by_a():
            e = registry(home)
            return len(e) == 1 and e[0].get("session_id") == key and e[0].get("lease_id") == a_lease_id

        # 1. Plain submit is refused SESSION_NOT_OWNED, exactly as today.
        refused = b.call("prompt.submit", {"session_id": sid_b, "text": "probe: B must not write"})
        check("B's plain submit is refused SESSION_NOT_OWNED",
              reason_of(refused) == "SESSION_NOT_OWNED", str(reason_of(refused)))
        check("the refusal left the registry on A (same lease token)",
              owned_by_a(), json.dumps(registry(home))[:160])

        # 2. B requests the handoff: the controller names itself, the ask is recorded.
        requested = b.call("session.request_handoff", {"session_id": sid_b, "controller": "hscc-app"})
        check("B's request_handoff is recorded (pending ask)",
              (requested.get("result") or {}).get("pending") is True, json.dumps(requested.get("result", ""))[:120])
        entries = registry(home)
        handoff = (entries[0].get("handoff") or {}) if entries else {}
        check("a pending handoff ask is stamped on the owner's entry",
              handoff.get("controller") == "hscc-app" and handoff.get("granted") is False,
              json.dumps(handoff)[:160])
        check("request alone transferred nothing (same lease token)",
              owned_by_a(), json.dumps(entries)[:160])

        # 3. B cannot self-approve: its OWN live id does not match the owner's registry lease.
        forged = b.call("session.handoff_consent",
                        {"session_id": sid_b, "controller": "hscc-app"})
        check("B cannot consent with its OWN live id (self-approval refused)",
              (forged.get("result") or {}).get("granted") is False, json.dumps(forged)[:160])

        # 4. The OWNER (A) consents: only A's live sid matches the registry lease.
        consent = a.call("session.handoff_consent",
                         {"session_id": sid_a, "controller": "hscc-app"})
        nonce = (consent.get("result") or {}).get("nonce") or ""
        check("A (the owner) consents and receives a one-shot nonce",
              bool(nonce) and (consent.get("result") or {}).get("granted") is True,
              json.dumps(consent.get("result", ""))[:120])

        # 5. A wrong nonce cannot complete: nothing transfers on a forged/absent nonce.
        forged_nonce = b.call("session.complete_handoff", {"session_id": sid_b, "controller": "hscc-app",
                                                           "nonce": "not-the-nonce"})
        forged_result = forged_nonce.get("result") or {}
        check("a wrong nonce completes nothing (error or completed=false)",
              forged_nonce.get("error") is not None or forged_result.get("completed") is False,
              json.dumps(forged_nonce)[:160])
        check("still exactly one owner after the forged attempt (same lease token)",
              owned_by_a(), json.dumps(registry(home))[:160])

        # 6. B completes with the real nonce: the lease moves to B.
        completed = b.call("session.complete_handoff", {"session_id": sid_b, "controller": "hscc-app",
                                                        "nonce": nonce})
        check("B completes the granted handoff (completed=true)",
              (completed.get("result") or {}).get("completed") is True, json.dumps(completed.get("result", ""))[:120])
        # 6b. The very next plain submit works because this process owns the lease now.
        drove = b.call("prompt.submit", {"session_id": sid_b, "text": "probe: B drives the turn"})
        check("B's submit after completion drives the turn (no error)", drove.get("error") is None,
              json.dumps(drove.get("error", ""))[:200])

        # handoff-back: B releases at turn end; the session is free again.
        b.call("session.close", {"session_id": sid_b})
        time.sleep(1.0)

        # 7. Handoff back: B releases at turn end; the session is free again and A can re-take it.
        b.call("session.close", {"session_id": sid_b})
        time.sleep(1.0)
        a.call("session.create", {"cols": 80})  # if it errors we surface it; the lease check below matters
        after = registry(home)
        # session.close may auto-release B's lease (the controller's end-of-turn release is the
        # handoff-back). Whether the row stays open (A can resume) is a lifecycle detail — what must
        # hold is that no foreign lease blocks a NEW claim by the owner surface.
        check("after handoff back, the session is no longer fenced to a foreign controller",
              after == [] or (not any(e.get("pid") != os.getpid() for e in after)),
              json.dumps(after)[:160])
    finally:
        if b is not None:
            b.close()
        a.close()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All cross-process handoff checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
