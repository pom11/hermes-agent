"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<repo>/.worktrees/<child-id>``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"


def test_worktree_anchor_out_of_tree_for_registered_repo(kanban_home, tmp_path, monkeypatch):
    """Repos listed under kanban.worktree.out_of_tree_repos get their worktrees
    placed OUT of the tree (under out_of_tree_root/<repo-name>), so a compiler or
    bundler rooted at the project root never walks branch code. Repos not listed
    keep the in-tree ``<repo>/.worktrees`` default."""
    repo = _make_repo(tmp_path)
    out_root = tmp_path / "out-of-tree"

    def _cfg(repos):
        return {
            "kanban": {
                "worktree": {
                    "out_of_tree_repos": repos,
                    "out_of_tree_root": str(out_root),
                }
            }
        }

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _cfg([str(repo)]),
    )
    # Registered repo → out of tree.
    assert kbw._worktree_anchor(repo) == (out_root / repo.name).resolve()

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _cfg([]),
    )
    # Not registered → in-tree .worktrees default.
    assert kbw._worktree_anchor(repo) == (repo / ".worktrees").resolve()


def test_resolve_worktree_places_registered_repo_out_of_tree(kanban_home, tmp_path, monkeypatch):
    """End-to-end: a worktree-kind task on an out-of-tree-registered repo is
    materialized under ``<out_of_tree_root>/<repo-name>/<task-id>``, not inside
    the project root."""
    repo = _make_repo(tmp_path)
    out_root = tmp_path / "out-of-tree"
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "kanban": {
                "worktree": {
                    "out_of_tree_repos": [str(repo)],
                    "out_of_tree_root": str(out_root),
                }
            }
        },
    )

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="task on a compiler-root repo",
            workspace_kind="worktree",
            workspace_path=str(repo),  # anchor = repo root
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == (out_root / repo.name / tid).resolve()
    assert branch == f"wt/{tid}"
    # The project root itself must NOT contain the worktree.
    assert not (repo / ".worktrees").exists()
    assert (workspace / "README.md").exists()




