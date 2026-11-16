"""The committed documents must equal the templates rendered from the committed results.

This is what guarantees that every number in README.md, REPORT.md and INTERVIEW_PREP.md
was produced by the pipeline: change a result without re-rendering (or edit a number by
hand) and this test fails.
"""

from __future__ import annotations

import re

import pytest

from overfitlab.config import REPO_ROOT, load_config
from overfitlab.reporting.render import PLACEHOLDER, TEMPLATE_DIR, build_documents

RESULTS = REPO_ROOT / "reports" / "results"
DOCS = [
    "README.md",
    "REPORT.md",
    "DECISIONS.md",
    "LIMITATIONS.md",
    "LEARNING.md",
    "INTERVIEW_PREP.md",
    "NEXT_STEPS.md",
]


@pytest.mark.skipif(not (RESULTS / "e3.json").exists(), reason="no committed results")
def test_rendered_documents_match_committed_results():
    status = build_documents(load_config(REPO_ROOT / "configs" / "full.yaml"), check=True)
    assert set(status) == {"README.md", "REPORT.md", "INTERVIEW_PREP.md"}
    stale = [name for name, ok in status.items() if not ok]
    assert not stale, f"run `make docs`: out of date -> {stale}"


def test_all_seven_documents_exist():
    for name in DOCS:
        path = REPO_ROOT / name
        assert path.exists() and len(path.read_text(encoding="utf-8")) > 1500, name


def test_every_figure_referenced_exists():
    pattern = re.compile(r"!\[[^\]]*\]\(([^)]+\.png)\)")
    for name in ("README.md", "REPORT.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        figures = pattern.findall(text)
        assert figures, f"{name} references no figures"
        for fig in figures:
            assert (REPO_ROOT / fig).exists(), f"{name} references missing figure {fig}"


def test_no_unrendered_placeholders_left():
    for tmpl in TEMPLATE_DIR.glob("*.tmpl"):
        target = REPO_ROOT / tmpl.name.removesuffix(".tmpl")
        assert PLACEHOLDER.search(target.read_text(encoding="utf-8")) is None, target.name
