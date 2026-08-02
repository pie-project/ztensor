#!/usr/bin/env python3
"""Builds the static site into `_site/`.

The landing page is hand-written HTML; the docs are markdown. Both end up
looking like one site because the doc template borrows the landing page's own
`<style>` block rather than keeping a second copy of the design — edit
`index.html` and the docs follow.

No site generator, no node_modules: the output is HTML, CSS and two SVGs, which
is all this site has ever needed.

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

# Order is the reading order, and it is also the nav order.
PAGES = [
    ("intro.md", "Introduction"),
    ("guide.md", "Guide"),
    ("spec.md", "Specification"),
    ("benchmarks.md", "Benchmarks"),
]

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# Prose styles the landing page has no use for. Everything here is expressed in
# the same tokens, so both themes come along for free.
PROSE_CSS = """
.doc { padding-block: clamp(2rem, 5vw, 3.5rem) 5rem; }
.doc .inner { max-width: 64rem; margin-inline: auto; display: grid; gap: clamp(2rem, 5vw, 3.5rem); }
@media (min-width: 62rem) {
  .doc .inner { grid-template-columns: 13rem minmax(0, 1fr); align-items: start; }
}
.toc { position: sticky; top: 2rem; display: grid; gap: 0.4rem; font-family: var(--mono); font-size: var(--step--1); }
.toc .toc-label { color: var(--fg-faint); letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.35rem; }
.toc a { text-decoration: none; color: var(--fg-dim); padding: 0.15rem 0; border-left: 2px solid transparent; padding-left: 0.7rem; }
.toc a:hover { color: var(--fg); border-left-color: var(--rule-firm); }
.toc a[aria-current="page"] { color: var(--accent); border-left-color: var(--accent); }

.prose { min-width: 0; }
.prose > * + * { margin-top: 1.1rem; }
.prose h1 { font-family: var(--mono); font-size: var(--step-3); line-height: 1.05; letter-spacing: -0.045em; margin-bottom: 0.4rem; }
.prose h2 { font-family: var(--mono); font-size: var(--step-2); letter-spacing: -0.02em; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--rule); }
.prose h3 { font-family: var(--mono); font-size: var(--step-1); letter-spacing: -0.01em; margin-top: 2rem; }
.prose h4 { font-family: var(--mono); font-size: var(--step-0); margin-top: 1.5rem; }
.prose p, .prose li { max-width: 42rem; }
.prose ul, .prose ol { padding-left: 1.3rem; display: grid; gap: 0.4rem; }
.prose li::marker { color: var(--fg-faint); }
.prose strong { font-weight: 600; }
.prose blockquote { margin: 0; border-left: 2px solid var(--accent); padding-left: 1rem; color: var(--fg-dim); }
.prose code { background: var(--bg-sunk); padding: 0.1em 0.35em; border: 1px solid var(--rule); }
.prose pre { background: var(--bg-panel); border: 1px solid var(--rule); padding: 0.9rem; overflow-x: auto; line-height: 1.6; }
.prose pre code { background: none; border: 0; padding: 0; }
.prose table { display: block; overflow-x: auto; border-collapse: collapse; border: 1px solid var(--rule); background: var(--bg-panel); width: max-content; max-width: 100%; }
.prose thead th { font-family: var(--mono); font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; font-size: var(--step--1); color: var(--fg-faint); background: var(--bg-sunk); white-space: nowrap; }
.prose th, .prose td { text-align: left; padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--rule); font-size: var(--step--1); }
.prose tbody tr:last-child td, .prose tbody tr:last-child th { border-bottom: 0; }
.prose hr { border: 0; border-top: 1px solid var(--rule); margin-block: 2rem; }
.prose img { max-width: 100%; height: auto; }
.prose a { color: var(--accent); text-decoration-color: color-mix(in srgb, var(--accent) 40%, transparent); }

.doc-nav { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--rule); font-family: var(--mono); font-size: var(--step--1); }
.doc-nav a { text-decoration: none; color: var(--fg-dim); }
.doc-nav a:hover { color: var(--accent); }
.doc-nav .next { margin-left: auto; }
"""

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — zTensor</title>
<meta name="description" content="{description}">
<style>{css}{prose}</style>
</head>
<body>
<header class="band masthead">
  <a class="mark" href="{root}index.html" style="text-decoration: none">zTensor<span>.</span></a>
  <span style="color: var(--fg-faint)">format v2 &middot; spec draft 2</span>
  <nav>{nav}
    <a href="https://github.com/pie-project/ztensor">GitHub</a>
  </nav>
</header>
<main class="band doc">
  <div class="inner">
    <aside class="toc">
      <div class="toc-label">Docs</div>
{toc}
    </aside>
    <article class="prose">
{body}
      <nav class="doc-nav">{prev}{next}</nav>
    </article>
  </div>
</main>
<footer class="band colophon rule-top">
  <div class="inner">
    <span>MIT licensed</span>
    <a href="https://github.com/pie-project/ztensor">github.com/pie-project/ztensor</a>
  </div>
</footer>
</body>
</html>
"""


def landing_css() -> str:
    """The landing page's own stylesheet, so there is only ever one."""
    html = (HERE / "index.html").read_text()
    return html[html.index("<style>") + len("<style>") : html.index("</style>")]


def first_paragraph(text: str) -> str:
    for block in text.split("\n\n"):
        block = block.strip()
        if block and not block.startswith(("#", "|", "```", "---", ":::")):
            return re.sub(r"[*`\[\]]|\(https?://[^)]+\)", "", block).replace("\n", " ")[:180]
    return "The zTensor container format."


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    css = landing_css()
    slugs = [(md.replace(".md", ".html"), title) for md, title in PAGES]

    # The landing page links to `docs/*.md`; on a served site those are pages.
    index = (HERE / "index.html").read_text()
    for md, _ in PAGES:
        index = index.replace(f'href="docs/{md}"', f'href="docs/{md[:-3]}.html"')
    (OUT / "index.html").write_text(index)

    if (HERE / "static").exists():
        shutil.copytree(HERE / "static", OUT / "static")
    (OUT / "docs").mkdir()
    # Tell GitHub Pages not to run Jekyll over this; it would eat `_`-prefixed
    # paths and rewrite things we have already rendered.
    (OUT / ".nojekyll").write_text("")

    for i, (md_name, title) in enumerate(PAGES):
        source = (DOCS / md_name).read_text()
        description = first_paragraph(FRONTMATTER.sub("", source))
        html = markdown.markdown(
            FRONTMATTER.sub("", source),
            extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
        )
        # Links between docs are written as `./other.md`; they are pages here.
        html = re.sub(r'href="\.?/?([a-z]+)\.md(#[^"]*)?"', r'href="\1.html\2"', html)

        toc = "\n".join(
            f'      <a href="{slug}"{" aria-current=\"page\"" if slug == md_name[:-3] + ".html" else ""}>{name}</a>'
            for slug, name in slugs
        )
        nav = "".join(f'\n    <a href="{s}">{n}</a>' for s, n in slugs)
        prev_link = (
            f'<a href="{slugs[i - 1][0]}">&larr; {slugs[i - 1][1]}</a>' if i > 0 else ""
        )
        next_link = (
            f'<a class="next" href="{slugs[i + 1][0]}">{slugs[i + 1][1]} &rarr;</a>'
            if i + 1 < len(slugs)
            else ""
        )

        (OUT / "docs" / f"{md_name[:-3]}.html").write_text(
            SHELL.format(
                title=title,
                description=description.replace('"', "&quot;"),
                css=css,
                prose=PROSE_CSS,
                root="../",
                nav=nav,
                toc=toc,
                body=html,
                prev=prev_link,
                next=next_link,
            )
        )

    pages = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    print(f"built {len(pages)} files into {OUT}")
    for page in pages:
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
