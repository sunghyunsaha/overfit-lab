"""Template rendering: placeholders, formats, strictness."""

from __future__ import annotations

import pytest

from overfitlab.reporting.render import lookup, md_table, render


def test_placeholders_and_formats():
    data = {"e3": {"n": 5000, "pbo": 0.2719, "label": "REAL", "rows": [{"x": 1.5}], "y": "inf"}}
    text = (
        "N={{e3.n:,}} PBO={{ e3.pbo:.2f }} {{e3.label}} {{e3.rows.0.x:.1f}} {{e3.pbo:.0%}} {{e3.y}}"
    )
    assert render(text, data) == "N=5,000 PBO=0.27 REAL 1.5 27% infinite"


def test_latex_braces_are_left_alone():
    text = r"\sqrt{\frac{1}{T-1}} and $x_{t}$ and {not a placeholder}"
    assert render(text, {}) == text


def test_missing_keys_fail_loudly():
    with pytest.raises(KeyError):
        render("{{e9.missing}}", {"e3": {}})
    assert lookup({"a": [10, 20]}, "a.1") == 20


def test_md_table():
    assert md_table(["a", "b"], [["1", "2"]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_optional_section_is_kept_only_when_its_data_exists():
    template = "a\n{{#if extra}}\nvalue {{ extra.x:.1f }}\n{{/if}}\nb\n"
    assert render(template, {"extra": {"x": 2.0}}) == "a\nvalue 2.0\nb\n"
    # Without the data the block disappears, including its strict placeholders.
    assert render(template, {}) == "a\nb\n"
