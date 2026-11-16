"""Turn results JSON into documents, so no number is ever typed by hand.

Templates live in ``docs/templates/*.tmpl``. A placeholder ``{{ e3.net.pbo.pbo:.2f }}``
is replaced by the value found at that dotted path in the combined results, formatted
with the given Python format spec (optional). Rendering is strict: an unknown path
raises an error instead of leaving a hole in the document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from overfitlab.config import REPO_ROOT, resolve_path

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*(?::([^{}]+?))?\s*\}\}")
TEMPLATE_DIR = REPO_ROOT / "docs" / "templates"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """A GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def lookup(data: dict, path: str) -> Any:
    """Follow a dotted path through nested dicts and lists."""
    node: Any = data
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(f"template placeholder {path!r}: no key {part!r}")
    return node


def render(template: str, data: dict) -> str:
    """Substitute every ``{{ path[:format] }}`` placeholder in ``template``."""

    def substitute(match: re.Match) -> str:
        value = lookup(data, match.group(1))
        spec = match.group(2)
        if value == "inf":
            return "infinite"
        if spec:
            return format(value, spec.strip())
        return str(value)

    return PLACEHOLDER.sub(substitute, template)


def load_results(results_dir: Path) -> dict:
    """Load every ``<name>.json`` in the results directory as ``{name: content}``."""
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(results_dir.glob("*.json"))
    }


def build_documents(cfg: dict, check: bool = False) -> dict[str, bool]:
    """Render every template to the repository root.

    With ``check=True`` nothing is written; the return value says, per document, whether
    the file on disk already equals the rendered template (used by the test-suite to
    guarantee the committed documents match the committed results).
    """
    data = load_results(resolve_path(cfg, "output_dir") / "results")
    status = {}
    for tmpl in sorted(TEMPLATE_DIR.glob("*.tmpl")):
        target = REPO_ROOT / tmpl.name.removesuffix(".tmpl")
        text = render(tmpl.read_text(encoding="utf-8"), data)
        if check:
            status[target.name] = target.exists() and target.read_text(encoding="utf-8") == text
        else:
            target.write_text(text, encoding="utf-8")
            status[target.name] = True
    return status
