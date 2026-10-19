"""MkDocs hook: pages built from files outside docs/.

A page whose Markdown is a single marker line is replaced:

    <!-- include: CHANGELOG.md -->   the file from the repository root, with
                                      its relative links rewritten (below)
    <!-- examples -->                 one section per script in the table of
                                      examples/README.md, with its code

Links in an included file are relative to the repository root, or full
GitHub links (the README's, for PyPI). A link to a page of the site
(docs/x.md, README.md, CHANGELOG.md, ROADMAP.md) points to that page; any
other file points to it on GitHub.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
GITHUB = "https://github.com/dvillacis/fastprox/blob/master/"
SITE_PAGES = {"README.md": "index.md", "CHANGELOG.md": "changelog.md", "ROADMAP.md": "roadmap.md"}
# A relative link, or a full link to a file on master (one pass, so that a
# rewritten link is not rewritten again).
LINK = re.compile(r"\]\((?:" + re.escape(GITHUB) + r"|(?!https?://|#|mailto:))([^)\s]+)\)")


def _site_link(target: str) -> str:
    path, _, anchor = target.partition("#")
    anchor = "#" + anchor if anchor else ""
    if path.startswith("docs/") and path.endswith(".md"):
        return path[len("docs/"):] + anchor
    if path in SITE_PAGES:
        return SITE_PAGES[path] + anchor
    return GITHUB + path + anchor


def _include(name: str) -> str:
    text = (ROOT / name).read_text(encoding="utf-8")
    text = LINK.sub(lambda m: "](" + _site_link(m.group(1)) + ")", text)
    return text.replace("](https://github.com/dvillacis/fastprox/tree/master/examples)", "](examples.md)")


def _examples() -> str:
    rows = re.findall(r"^\| \[(\w+\.py)\]\(\w+\.py\) \| (.+) \|$",
                      (ROOT / "examples" / "README.md").read_text(encoding="utf-8"), re.M)
    out = []
    for script, shows in rows:
        code = (ROOT / "examples" / script).read_text(encoding="utf-8").rstrip()
        out.append(f"## {script}\n\n{shows}\n\n"
                   f"[On GitHub]({GITHUB}examples/{script}) · "
                   f"`uv run python examples/{script}`\n\n"
                   f'??? example "Source"\n\n'
                   + "\n".join("    " + line if line else "" for line in
                               ["```python", *code.splitlines(), "```"])
                   + "\n")
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:
    line = markdown.strip()
    m = re.fullmatch(r"<!-- include: ([\w./-]+) -->", line)
    if m:
        return _include(m.group(1))
    if line.endswith("<!-- examples -->"):
        return markdown.replace("<!-- examples -->", _examples())
    return markdown
