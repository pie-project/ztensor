#!/usr/bin/env python3
"""Checks the hand-authored SVG figures for the three ways they go wrong.

All three have happened, and none is visible in the source:

1. **Text past the frame.** The figures place text by computing its width, and
   a width computed for a monospace font is wrong for the sans one next to it,
   so on a wide system font (DejaVu Sans, say) a caption runs off the canvas.
   One figure was 175 px past its own edge.

2. **Text painted over.** SVG has no z-index: later elements cover earlier
   ones. A band drawn after the text that overflowed into it will quietly hide
   that text, which looks like the text was never there.

3. **Text running into text.** The code listings are assembled from runs
   placed at computed x positions, so editing one run's content without
   moving the next leaves them overlapping: two strings printed on top of
   each other, which reads as garbage rather than as a mistake.

Widths here are deliberate overestimates, so a figure that passes has room on
the widest font a reader is likely to have.

    python website/check_diagrams.py
"""

from __future__ import annotations

import glob
import re
import sys

# Conservative advance per character, as a fraction of the font size.
MONO_EM, SANS_EM = 0.602, 0.62
EDGE = 2  # keep this much clear of the canvas


def boxes(source: str):
    """Every text and every filled shape, in document (paint) order."""
    texts, shapes = [], []
    for m in re.finditer(r"<(?:text|rect)[^>]*?(?:/>|>.*?</text>)", source, re.S):
        el, pos = m.group(0), m.start()
        if el.startswith("<text"):
            body = re.sub(r"<[^>]+>", "", el)
            x = float(re.search(r'\bx="([-\d.]+)"', el).group(1))
            y = float(re.search(r'\by="([-\d.]+)"', el).group(1))
            size = float(re.search(r'font-size="([\d.]+)"', el).group(1))
            width = len(body) * size * (MONO_EM if 'class="mono' in el else SANS_EM)
            anchor = (re.search(r'text-anchor="(\w+)"', el) or [None, "start"])[1]
            left = x if anchor == "start" else (x - width if anchor == "end" else x - width / 2)
            texts.append((pos, left, y - size * 0.78, left + width, y + size * 0.22, body, anchor, y))
        elif 'fill="none"' not in el:
            try:
                x = float(re.search(r'\bx="([-\d.]+)"', el).group(1))
                y = float(re.search(r'\by="([-\d.]+)"', el).group(1))
                w = float(re.search(r'width="([\d.]+)"', el).group(1))
                h = float(re.search(r'height="([\d.]+)"', el).group(1))
            except AttributeError:
                continue  # a shape without a box of its own; nothing to test
            shapes.append((pos, x, y, x + w, y + h))
    return texts, shapes


def check(path: str) -> list[str]:
    source = open(path).read()
    view = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', source)
    if not view:
        return [f"{path}: no viewBox"]
    vw, vh = (float(v) for v in view.groups())
    texts, shapes = boxes(source)
    found = []
    for pos, x0, y0, x1, y1, body, _, _ in texts:
        if x1 > vw - EDGE or y1 > vh - EDGE or x0 < 0 or y0 < 0:
            found.append(f"{path}: runs past the frame: {body[:44]!r}")
        for spos, sx0, sy0, sx1, sy1 in shapes:
            if spos > pos and x0 < sx1 and x1 > sx0 and y0 < sy1 and y1 > sy0:
                found.append(f"{path}: hidden behind a later shape: {body[:44]!r}")
                break

    # Runs sharing a baseline are one line of text; consecutive ones must not
    # collide. Only left-anchored runs take part: a centred or right-anchored
    # label is positioned against something else, not against its neighbour.
    lines = {}
    for _, x0, _, x1, _, body, anchor, base in texts:
        if anchor == "start":
            lines.setdefault(base, []).append((x0, x1, body))
    for base, runs in sorted(lines.items()):
        runs.sort()
        for (_, end, body), (next_start, _, next_body) in zip(runs, runs[1:]):
            if end > next_start + 0.5:
                found.append(
                    f"{path}: {body[:28]!r} runs into {next_body[:28]!r} at y={base}"
                )
    return found


def main() -> None:
    files = sorted(glob.glob("website/static/diagrams/*.svg"))
    if not files:
        sys.exit("no diagrams found; run this from the repository root")
    problems = [p for f in files for p in check(f)]
    for f in files:
        texts, shapes = boxes(open(f).read())
        mark = "!" if any(f in p for p in problems) else "ok"
        print(f"  {mark:3} {f.split('/')[-1]:20} {len(texts):>3} texts, {len(shapes):>3} shapes")
    if problems:
        print()
        for p in problems:
            print(f"::error::{p}")
        sys.exit(1)
    print(f"\n{len(files)} figures: nothing off the frame, nothing painted over")


if __name__ == "__main__":
    main()
