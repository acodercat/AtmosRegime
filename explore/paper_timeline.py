"""docs/PAPER_TIMELINE.md: every dated amendment / execution record of every pre-registration, with the commit that first added
its heading (git log -S, earliest commit). Supplementary table; regenerate before submission.

    python -m explore.paper_timeline --out docs/PAPER_TIMELINE.md

A row is a level-2 or level-3 heading that contains a date (YYYY-MM-DD) in one of the documents below, in file order. The ID is
the record tag in the title (X-41, DM-23, OPD-17, DT-3, DOSE-4, D-2a, T-1, §13, …) when there is one. Titles are verbatim.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("PREREG_MAIN.md", "PREREG_X.md", "PREREG_DOSE.md", "PREREG_DOSE_TEXT.md", "PREREG_OPD.md", "PREREG_DOSE_MINISTRAL.md",
        "PREREG_PAPER_ANALYSES.md", "PREREG_REPLICATION.md", "PREREG_E7.md", "PREREG_E0_CONFIRM.md", "PREREG_REWORD.md",
        "PREREG_TEMPLATE_SCREENING.md", "PREREG_TEMPLATE_RL.md", "PREREG_GIVEN_VERSIONS.md")
DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
TAG = re.compile(r"\b((?:X|D|T|DM|DT|DOSE|OPD)-\d+[a-z]?)\b|(§\s?\d+)")


def first_commit(path: Path, heading: str) -> str:
    out = subprocess.run(["git", "log", "--reverse", "--format=%h %ci", "-S", heading, "--", str(path.relative_to(ROOT))],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return "uncommitted"
    h, date, time, _tz = out[0].split()
    return f"`{h}` {date} {time[:5]}"


def rows(doc: str) -> list[tuple[str, str, str, str, str]]:
    path = ROOT / "docs" / doc
    out = []
    for line in path.read_text().splitlines():
        if not (line.startswith("## ") or line.startswith("### ")):
            continue
        title = line.lstrip("#").strip()
        m = DATE.search(title)
        if not m:
            continue
        t = TAG.search(title)
        tag = (t.group(1) or t.group(2)) if t else ""
        if not tag and (n := re.match(r"(\d+)\.\s", title)):   # PREREG_PAPER_ANALYSES: "13. ..." -> §13
            tag = f"§{n.group(1)}"
        out.append((doc.removesuffix(".md"), tag, m.group(1), first_commit(path, title), title.replace("|", "\\|")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "PAPER_TIMELINE.md")
    args = ap.parse_args()
    table = [r for d in DOCS if (ROOT / "docs" / d).exists() for r in rows(d)]
    head = ["# Pre-registration timeline (supplementary table, generated from git)", "",
            "Every dated amendment or execution record of every pre-registration under `docs/` (level-2 and level-3 headings that carry a date), in file order, "
            "with the commit that first added its heading (`git log -S`, earliest commit). Titles are verbatim from the documents (Chinese); "
            "the parenthetical in a title is the record's own statement of when it was written relative to the data. "
            "Sealed bodies are never edited; the records are append-only. Regenerate with `python -m explore.paper_timeline` before submission: "
            "a record not yet committed shows as \"uncommitted\".", "",
            f"Documents: {', '.join(f'`{d}`' for d in DOCS if (ROOT / 'docs' / d).exists())}. Rows: {len(table)}.", "",
            "| Document | ID | Dated | Commit (time, +08:00) | Title |", "|---|---|---|---|---|"]
    body = [f"| {d} | {tag} | {date} | {commit} | {title} |" for d, tag, date, commit, title in table]
    args.out.write_text("\n".join(head + body) + "\n")
    print(f"{len(table)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
