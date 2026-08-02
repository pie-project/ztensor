#!/usr/bin/env python3
"""Builds the docs site into `_site/`.

A documentation site and nothing more: a nav bar, a sidebar, the page. The
docs are the content — there is no landing page, because `intro.md` carries
`slug: /` and is the front page, which is how this site was always arranged.

No generator, no node_modules; the output is HTML, CSS and two SVGs.

    python website/build.py            # -> website/_site
    python website/build.py --serve    # and serve it on :8000
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

try:
    import markdown
except ImportError:  # pragma: no cover - the message is the point
    sys.exit("this needs `pip install markdown`")

HERE = Path(__file__).parent
DOCS = HERE / "docs"
OUT = HERE / "_site"

# Reading order, which is also the sidebar order. `intro` has `slug: /`, so it
# is the front page.
#
# The specification is read from `spec/` rather than copied into `docs/`. It
# was a copy once, and the copy drifted from the normative text, which is the
# one kind of staleness a format specification cannot afford.
SPEC = Path(__file__).parent.parent / "spec" / "ztensor-v2-spec.md"

PAGES = [
    ("intro.md", "Introduction", "index.html"),
    ("guide.md", "Guide", "guide.html"),
    (SPEC, "Specification", "spec.html"),
    ("benchmarks.md", "Benchmarks", "benchmarks.html"),
]

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

CSS = """
:root {
  --bg: #ffffff;
  --bg-alt: #f6f7f8;
  --fg: #1c1e21;
  --fg-dim: #525860;
  --border: #e3e5e8;
  --link: #2b6cb0;
  --code-bg: #f6f7f8;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sidebar: 15rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1b1b1d;
    --bg-alt: #242526;
    --fg: #e3e3e3;
    --fg-dim: #a8a8a8;
    --border: #333438;
    --link: #78a9e0;
    --code-bg: #242526;
  }
}
:root[data-theme="dark"] {
  --bg: #1b1b1d; --bg-alt: #242526; --fg: #e3e3e3; --fg-dim: #a8a8a8;
  --border: #333438; --link: #78a9e0; --code-bg: #242526;
}
:root[data-theme="light"] {
  --bg: #ffffff; --bg-alt: #f6f7f8; --fg: #1c1e21; --fg-dim: #525860;
  --border: #e3e5e8; --link: #2b6cb0; --code-bg: #f6f7f8;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.65;
}
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; }

.navbar {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  height: 3.5rem;
  padding-inline: 1.25rem;
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  background: var(--bg);
  z-index: 10;
}
.navbar .brand { font-weight: 600; color: var(--fg); font-size: 1.05rem; }
.navbar .version { color: var(--fg-dim); font-size: 0.8rem; font-family: var(--mono); }
.navbar .links { margin-left: auto; display: flex; gap: 1.25rem; font-size: 0.9rem; }
.navbar .links a { color: var(--fg-dim); }
.navbar .links a:hover { color: var(--fg); text-decoration: none; }

.layout { display: grid; grid-template-columns: 1fr; }
@media (min-width: 60rem) {
  .layout { grid-template-columns: var(--sidebar) minmax(0, 1fr); }
}

.sidebar { border-bottom: 1px solid var(--border); padding: 1rem 1.25rem; }
@media (min-width: 60rem) {
  .sidebar {
    border-bottom: 0;
    border-right: 1px solid var(--border);
    padding: 1.5rem 1rem;
    position: sticky;
    top: 3.5rem;
    height: calc(100vh - 3.5rem);
    overflow-y: auto;
  }
}
.sidebar ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.15rem; }
.sidebar a {
  display: block;
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  color: var(--fg-dim);
  font-size: 0.92rem;
}
.sidebar a:hover { background: var(--bg-alt); color: var(--fg); text-decoration: none; }
.sidebar a[aria-current="page"] { background: var(--bg-alt); color: var(--link); font-weight: 500; }

main { padding: 2rem 1.25rem 4rem; min-width: 0; }
article { max-width: 46rem; margin-inline: auto; }
article > * + * { margin-top: 1rem; }

h1 { font-size: 2rem; line-height: 1.2; margin: 0 0 1.25rem; }
h2 { font-size: 1.4rem; margin-top: 2.5rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border); }
h3 { font-size: 1.15rem; margin-top: 1.75rem; }
h4 { font-size: 1rem; margin-top: 1.25rem; }
p, li { color: var(--fg); }
ul, ol { padding-left: 1.4rem; }
li + li { margin-top: 0.25rem; }
blockquote { margin: 0; padding-left: 1rem; border-left: 3px solid var(--border); color: var(--fg-dim); }
hr { border: 0; border-top: 1px solid var(--border); margin-block: 2rem; }
img { max-width: 100%; height: auto; }

code {
  font-family: var(--mono);
  font-size: 0.875em;
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0.1em 0.3em;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.85rem 1rem;
  overflow-x: auto;
  line-height: 1.55;
}
pre code { background: none; border: 0; padding: 0; font-size: 0.85rem; }

.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.5rem 0.85rem; border: 1px solid var(--border); }
thead th { background: var(--bg-alt); font-weight: 600; white-space: nowrap; }

.pagination { display: flex; gap: 1rem; margin-top: 3rem; padding-top: 1.25rem; border-top: 1px solid var(--border); font-size: 0.9rem; }
.pagination .next { margin-left: auto; }
"""

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<style>{css}</style>
</head>
<body>
<nav class="navbar">
  <a class="brand" href="index.html">zTensor</a>
  <span class="version">v2</span>
  <span class="links">
    <a href="https://github.com/pie-project/ztensor">GitHub</a>
  </span>
</nav>
<div class="layout">
  <aside class="sidebar">
    <ul>
{sidebar}
    </ul>
  </aside>
  <main>
    <article>
{body}
      <nav class="pagination">{prev}{next}</nav>
    </article>
  </main>
</div>
</body>
</html>
"""


def first_paragraph(text: str) -> str:
    for block in text.split("\n\n"):
        block = block.strip()
        if block and not block.startswith(("#", "|", "```", "---", ":::")):
            return re.sub(r"[*`\[\]]|\(https?://[^)]+\)", "", block).replace("\n", " ")[:160]
    return "zTensor documentation."


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    if (HERE / "static").exists():
        shutil.copytree(HERE / "static", OUT / "static")
    # Pages would otherwise run Jekyll over this and eat `_`-prefixed paths.
    (OUT / ".nojekyll").write_text("")

    sidebar = "\n".join(
        f'      <li><a href="{slug}"{{current_{i}}}>{name}</a></li>'
        for i, (_, name, slug) in enumerate(PAGES)
    )

    for i, (md_name, title, slug) in enumerate(PAGES):
        path = md_name if isinstance(md_name, Path) else DOCS / md_name
        source = FRONTMATTER.sub("", path.read_text())
        html = markdown.markdown(
            source,
            extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
        )
        # Cross-doc links are written as `./other.md`; here they are pages.
        # A page may be reached by its source filename or by its slug — the
        # spec is written as `./spec.md` but lives in `spec/` under its own
        # name — so both spellings resolve.
        for other_md, _, other_slug in PAGES:
            stems = {Path(other_md).stem, Path(other_slug).stem}
            for stem in stems:
                html = re.sub(
                    rf'href="\.?/?{re.escape(stem)}\.md(#[^"]*)?"',
                    rf'href="{other_slug}\1"',
                    html,
                )
        # The docs are written to be read in the repo, where they sit one
        # directory below `static/`. On the built site they are at the root, so
        # `../static/` would climb out of it.
        html = html.replace('src="../static/', 'src="static/')
        # Tables scroll inside their own box rather than widening the page.
        html = html.replace("<table>", '<div class="table-scroll"><table>').replace(
            "</table>", "</table></div>"
        )

        marks = {f"current_{j}": ' aria-current="page"' if j == i else "" for j in range(len(PAGES))}
        (OUT / slug).write_text(
            SHELL.format(
                title=f"{title} · zTensor" if i else "zTensor",
                description=first_paragraph(source).replace('"', "&quot;"),
                css=CSS,
                sidebar=sidebar.format(**marks),
                body=html,
                prev=f'<a href="{PAGES[i - 1][2]}">← {PAGES[i - 1][1]}</a>' if i else "",
                next=(
                    f'<a class="next" href="{PAGES[i + 1][2]}">{PAGES[i + 1][1]} →</a>'
                    if i + 1 < len(PAGES)
                    else ""
                ),
            )
        )

    built = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    print(f"built {len(built)} files into {OUT}")
    for page in built:
        print("  ", page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="serve _site on :8000")
    args = parser.parse_args()
    build()
    if args.serve:
        import functools
        import http.server

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
        print("serving http://localhost:8000")
        http.server.HTTPServer(("", 8000), handler).serve_forever()


if __name__ == "__main__":
    main()
