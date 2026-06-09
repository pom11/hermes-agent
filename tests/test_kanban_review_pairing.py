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
