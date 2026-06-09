from hermes_cli import kanban_decompose as kd


def test_pair_adds_review_for_code_role():
    children = [
        {"title": "Build API", "body": "spec", "assignee": "backend-engineer", "parents": []},
    ]
    out = kd._pair_review_tasks(children, policy={"review_roles": ["backend-engineer"],
                                                 "reviewer": "reviewer"})
    assert len(out) == 2
    impl, review = out[0], out[1]
    assert impl["assignee"] == "backend-engineer"
    assert review["assignee"] == "reviewer"
    assert review["parents"] == [0]
    assert "review" in review["title"].lower()


def test_pair_skips_non_code_role():
    children = [
        {"title": "Write spec", "body": "", "assignee": "product-manager", "parents": []},
    ]
    out = kd._pair_review_tasks(children, policy={"review_roles": ["backend-engineer"],
                                                 "reviewer": "reviewer"})
    assert len(out) == 1


def test_pair_never_reviews_a_reviewer_task():
    children = [
        {"title": "Review X", "body": "", "assignee": "reviewer", "parents": []},
    ]
    out = kd._pair_review_tasks(children, policy={"review_roles": ["reviewer"],
                                                 "reviewer": "reviewer"})
    assert len(out) == 1


def test_pair_preserves_impl_indices_for_existing_parents():
    children = [
        {"title": "A", "body": "", "assignee": "coder", "parents": []},
        {"title": "B", "body": "", "assignee": "coder", "parents": [0]},
    ]
    out = kd._pair_review_tasks(children, policy={"review_roles": ["coder"],
                                                 "reviewer": "reviewer"})
    assert len(out) == 4
    assert out[1]["parents"] == [0]
    reviews = [c for c in out if c["assignee"] == "reviewer"]
    assert {tuple(r["parents"]) for r in reviews} == {(0,), (1,)}


def test_pair_empty_policy_noop():
    children = [{"title": "X", "body": "", "assignee": "coder", "parents": []}]
    assert kd._pair_review_tasks(children, policy={}) == children
    assert kd._pair_review_tasks(children, policy=None) == children


def test_review_policy_reads_config():
    cfg = {"kanban": {"auto_review": {"review_roles": ["coder", "backend-engineer"],
                                      "reviewer": "reviewer"}}}
    pol = kd._review_policy(cfg)
    assert pol["reviewer"] == "reviewer"
    assert "coder" in pol["review_roles"]


def test_review_policy_absent_returns_empty():
    assert kd._review_policy({}) == {}
    assert kd._review_policy({"kanban": {}}) == {}


def test_policy_and_transform_compose():
    """Contract the Task-3 wiring performs inside decompose_task:
    children = _pair_review_tasks(children, _review_policy(cfg))."""
    cfg = {"kanban": {"auto_review": {"review_roles": ["backend-engineer"],
                                      "reviewer": "reviewer"}}}
    children = [{"title": "Build API", "body": "spec",
                 "assignee": "backend-engineer", "parents": []}]
    out = kd._pair_review_tasks(children, kd._review_policy(cfg))
    assert len(out) == 2
    assert out[1]["assignee"] == "reviewer"
    assert out[1]["parents"] == [0]


def test_decompose_no_longer_autopairs_reviews():
    """Phase 3: built-in review path replaces Phase 2 auto-pairing.
    auto_review is removed from config, so _review_policy returns {}."""
    from hermes_cli import kanban_decompose as kd
    from hermes_cli.config import load_config
    assert kd._review_policy(load_config()) == {}
