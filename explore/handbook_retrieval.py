"""E7 checks (docs/PREREG_E7.md, second addendum): a BM25 retriever over the 60 handbook entries.

    python -m explore.handbook_retrieval            # hit@k of the relevant entry, templates and textbook traps

The query is the problem text; documents are the handbook entries (heading + body). Pure stdlib BM25 (k1 = 1.5,
b = 0.75) over lower-cased word tokens. `retrieve(problem, k)` returns the top-k entry blocks; the relevant entry
for an item comes from `explore/handbook_map.json` (analysis only).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from explore.build_e7_input import handbook_entries

MAP = Path(__file__).with_name("handbook_map.json")
K1, B_ = 1.5, 0.75
TOKEN = re.compile(r"[a-zα-ωₐ-ₜ0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[list[str], list[str], list[Counter], dict[str, float], float]:
    entries = handbook_entries()
    titles = list(entries)
    blocks = [entries[t] for t in titles]
    docs = [Counter(tokens(b)) for b in blocks]
    n = len(docs)
    df = Counter(w for d in docs for w in d)
    idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    avg = sum(sum(d.values()) for d in docs) / n
    return titles, blocks, docs, idf, avg


def ranked(problem: str) -> list[int]:
    titles, _, docs, idf, avg = _index()
    q = Counter(tokens(problem))
    scores = []
    for i, d in enumerate(docs):
        length = sum(d.values()); s = 0.0
        for w in q:
            f = d.get(w, 0)
            if f:
                s += idf[w] * f * (K1 + 1) / (f + K1 * (1 - B_ + B_ * length / avg))
        scores.append((s, -i))
    return [-i for _, i in sorted(scores, reverse=True)]


def retrieve(problem: str, k: int) -> list[str]:
    _, blocks, *_ = _index()
    return [blocks[i] for i in ranked(problem)[:k]]


def relevant_titles() -> tuple[dict[str, str], dict[str, str]]:
    m = json.loads(MAP.read_text(encoding="utf-8"))
    tpl = {t: e["title"] for e in m["entries"] for t in e["templates"]}
    trp = {t: e["title"] for e in m["entries"] for t in e["traps"]}
    return tpl, trp


def main() -> int:
    from atmoscoder_rl import pilot
    from atmoscoder_rl.build_templates import build
    from atmoscoder_rl.datasets import load_traps
    from explore.build_e7_input import NAMES, SEED
    titles = _index()[0]
    tpl, trp = relevant_titles()
    items = [i for i in build(pilot.PAIRS, SEED, set(NAMES)) if i["variant"] == "outside"]
    traps = [t for t in load_traps() if t["id"] in trp]
    for label, rows, key in (("templates (17, outside rows)", items, lambda i: tpl[i["template"]]),
                             ("textbook regime traps (mapped)", traps, lambda t: trp[t["id"]])):
        hits = Counter()
        for r in rows:
            rank = [titles[i] for i in ranked(r["problem"])].index(key(r)) + 1
            for k in (1, 3, 5, 10):
                hits[k] += rank <= k
        print(f"{label}: n={len(rows)}  " + "  ".join(f"hit@{k} {hits[k] / len(rows):.0%}" for k in (1, 3, 5, 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
