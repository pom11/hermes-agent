"""Tests for submit_review_task (running -> review transition)."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def db_path(monkeypatch):
    """Isolated temp DB for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "kanban.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
        # connect auto-initializes on first connection
        yield db


def test_submit_review_running_task(db_path):
    """submit_review_task transitions running -> review and clears claim."""
    conn = kb.connect(db_path=db_path)
    try:
        # Create and claim a task to get it into running state
        tid = kb.create_task(
            conn,
            title="Test code change",
            body="Needs review",
        )
        # Claim it to transition ready -> running
        task = kb.claim_task(conn, tid)
        assert task is not None
        assert task.status == "running"

        # Submit to review
        ok = kb.submit_review_task(conn, tid)
        assert ok is True

        # Verify transition
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "review"
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None

        # Verify event was emitted
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
        # Should have: created, claimed, submitted_review (at minimum)
        event_kinds = [e[0] for e in events]
        assert "submitted_review" in event_kinds
    finally:
        conn.close()


def test_submit_review_non_running_task(db_path):
    """submit_review_task returns False if task is not running."""
    conn = kb.connect(db_path=db_path)
    try:
        # Create a blocked task
        tid = kb.create_task(
            conn,
            title="Blocked task",
            body="Not running",
            initial_status="blocked",
        )
        # Verify initial state
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"

        # Attempt submit to review (should fail)
        ok = kb.submit_review_task(conn, tid)
        assert ok is False

        # Verify status unchanged
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
    finally:
        conn.close()
