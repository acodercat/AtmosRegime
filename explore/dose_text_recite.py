#!/usr/bin/env python
"""PREREG_DOSE_TEXT DT-3 (appended 2026-10-30 before any number of this probe was computed): the recitation probe.

DT-2 found that four question/answer rows carrying a unit's "given" clause neither raise the warm start's chance of sampling
the correct procedure nor let RL learn it.  Three readings, numbers only, say whether the clause got in at all:
  (a) recitation — ask each text-injected warm start the four training questions (explore.dose_text.FRAMES with the unit's
      standard-approach name, the training system prompt), 16 samples each at T 1.0, thinking off; a sample "recites" when
      the whitespace-normalised clause is a substring of it; also the share of the clause's word 8-grams present (recall);
      the same mechanism's not-injected seed is the control;
  (b) the clause in the prompt — the warm starts' accuracy on the unit's `given` rows of final.json (x_report's regrade),
      beside the base model's;
  (c) mention without use — among the warm starts' 64 outside samples per row (passk .texts.jsonl), the share whose 8-gram
      recall of the clause is >= 0.5, split by passed / failed.
Exploratory, added 2026-10-30 after the verbatim rate came out 0 in every observation: (a') "content" — the share of answers
containing any distinctive token of the clause (a decimal literal such as 5.35, or a capitalised name such as Kasten; mechanisms
whose clause has none are skipped), injected vs control; post hoc, numbers only.
Fixed descriptive sentences (no hypothesis test): "can recite" = injected observations with recitation >= 50 % (control < 5 %);
"uses the sentence when it is in the problem" = observations whose given accuracy >= 50 %; "mentions but does not use" = injected observations with mention share >= 10 % and
warm outside accuracy < 10 %.  Group differences resample mechanisms, 10,000 draws, random.Random("dose-text-recite:20261030").

  python -m explore.dose_text_recite sample --design results/dose_text/design.json --label sft-dtext-qwen3-8b-F1-s42 \
      --base-url http://127.0.0.1:8241/v1 --model x-recite --out $ROOT/experiments/_dose/qwen3-8b/recite.sft-dtext-qwen3-8b-F1-s42.jsonl
  python -m explore.dose_text_recite report --design results/dose_text/design.json --name qwen3-8b \
      --samples-dir $ROOT/experiments/_dose/qwen3-8b --passk-dir $ROOT/experiments/_dose/qwen3-8b --eval-dir $ROOT/experiments/_x_eval \
      --final $ROOT/experiments/_inputs/kc/final.json --inputs $ROOT/experiments/_inputs/x_eval --out results/dose_text/recite_qwen3-8b.json
"""
import argparse
import json
import os
import random
import re
from pathlib import Path
from statistics import fmean

from atmoscoder_rl.prompts import SYSTEM
from explore.dose_text import FRAMES
from explore.dose_text_report import warm_label
from explore.x_report import Family

MAX_TOKENS = 1024
N_GRAM = 8
SEED = "dose-text-recite:20261030"
DRAWS = 10_000


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def recites(text: str, clause: str) -> bool:
    return norm(clause) in norm(text)


def ngram_recall(text: str, clause: str, n: int = N_GRAM) -> float:
    cw, tw = norm(clause).split(" "), norm(text).split(" ")
    n = min(n, len(cw))
    grams = {tuple(cw[i:i + n]) for i in range(len(cw) - n + 1)}
    have = {tuple(tw[i:i + n]) for i in range(len(tw) - n + 1)}
    return len(grams & have) / len(grams) if grams else 0.0


STOP = {"Because", "Use", "The", "Since", "When", "For", "If", "This", "With", "Under", "In", "At", "Sun", "Kelvin"}


def distinctive_tokens(clause: str) -> set[str]:
    """Decimal literals and capitalised names (3+ letters, not a sentence-initial or generic word) of a clause."""
    nums = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", clause))
    names = {n for n in re.findall(r"\b[A-Z][a-z]{2,}", clause) if n not in STOP}
    return nums | names


def content_share(texts: list[str], clause: str) -> float | None:
    keys = distinctive_tokens(clause)
    if not keys or not texts:
        return None
    return sum(any(k in t for k in keys) for t in texts) / len(texts)


def parse_label(label: str) -> tuple[str, int]:
    m = re.fullmatch(r"sft-dtext-.+-(F[12])-s(\d+)", label)
    if not m:
        raise SystemExit(f"not a text warm-start label: {label}")
    return m.group(1), int(m.group(2))


def sample(args) -> int:
    import openai

    design = json.loads(args.design.read_text())
    fold, _ = parse_label(args.label)
    done = set()
    if args.out.exists():
        done = {(json.loads(l)["mechanism"], json.loads(l)["frame"]) for l in args.out.read_text().splitlines() if l.strip()}
    todo = [(m, i) for m in design["units"][fold] for i in range(len(FRAMES)) if (m, i) not in done]
    print(f"{args.label}: {len(todo)} (mechanism, frame) pairs to sample ({len(done)} done), n={args.n}", flush=True)
    client = openai.OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600.0, max_retries=0)
    extra = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("ATMOS_NO_CHAT_TEMPLATE_KWARGS") == "1":
        extra.pop("chat_template_kwargs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for m, i in todo:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FRAMES[i].format(S=design["standard"][m])}]
            resp = client.chat.completions.create(model=args.model, messages=messages, temperature=args.temperature, top_p=1.0,
                                                  max_tokens=MAX_TOKENS, n=args.n, extra_body=extra)
            texts = [c.message.content or "" for c in resp.choices]
            fh.write(json.dumps({"label": args.label, "mechanism": m, "frame": i, "n": len(texts),
                                 "truncated": sum(c.finish_reason == "length" for c in resp.choices), "texts": texts}) + "\n")
            fh.flush()
            print(f"  {m} frame {i}: recite {sum(recites(t, design['clauses'][m]) for t in texts)}/{len(texts)}", flush=True)
    return 0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mention_counts(passk_dir: Path, label: str, mechanism: str, clause: str) -> dict | None:
    """Among the warm start's outside samples of `mechanism`: n, mentions (recall >= 0.5), passed, mentions among failed."""
    texts_path, verdicts_path = passk_dir / f"passk_final_outside.{label}.texts.jsonl", passk_dir / f"passk_final_outside.{label}.jsonl"
    if not texts_path.exists() or not verdicts_path.exists():
        return None
    passed = {r["id"]: r["passed"] for r in load_jsonl(verdicts_path) if r.get("template") == mechanism and "passed" in r}
    keys = distinctive_tokens(clause)
    n = mention = npass = mention_failed = mention_passed = content = content_failed = 0
    for r in load_jsonl(texts_path):
        if r["id"] not in passed:
            continue
        for text, ok in zip(r["texts"], passed[r["id"]]):
            n += 1; npass += ok
            if ngram_recall(text, clause) >= 0.5:
                mention += 1
                if ok: mention_passed += 1
                else: mention_failed += 1
            if keys and any(k in text for k in keys):   # exploratory, as content_share: the clause's distinctive tokens in a solving sample
                content += 1; content_failed += not ok
    return {"samples": n, "passed": npass, "mentions": mention, "mentions_passed": mention_passed, "mentions_failed": mention_failed,
            "mention_share": mention / n if n else None,
            "content_share": content / n if (n and keys) else None, "content_failed": content_failed if keys else None}


def boot_diff(pairs: list[tuple[float, float]]) -> dict:
    """mean over mechanisms of (injected - control), with a mechanism-resampled interval."""
    if not pairs:
        return {"value": None}
    rng, diffs = random.Random(SEED), [a - b for a, b in pairs]
    draws = sorted(fmean(rng.choice(diffs) for _ in diffs) for _ in range(DRAWS))
    return {"value": fmean(diffs), "ci95": [draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]], "mechanisms": len(pairs)}


def report(args) -> dict:
    design = json.loads(args.design.read_text())
    args.seeds = [42, 43]
    fam = Family(args)
    base_given, base_outside = fam.accuracy(fam.base(), "given"), fam.accuracy(fam.base(), "outside")
    obs, missing = [], []
    for fold, units in design["units"].items():
        for s in (42, 43):
            label = warm_label(args.name, fold, s)
            path = args.samples_dir / f"recite.{label}.jsonl"
            recs = {(r["mechanism"], r["frame"]): r for r in load_jsonl(path)} if path.exists() else {}
            if not path.exists():
                missing.append(str(path))
            given, outside = fam.accuracy(label, "given"), fam.accuracy(label, "outside")
            for m in units:
                clause = design["clauses"][m]
                texts = [t for i in range(len(FRAMES)) for t in recs.get((m, i), {}).get("texts", [])]
                if len(texts) < len(FRAMES) * args.n:
                    missing.append(f"{label}:{m} has {len(texts)} samples")
                obs.append({"mechanism": m, "fold": fold, "seed": s, "injected": m in design["text"][fold][str(s)],
                            "recite": fmean(recites(t, clause) for t in texts) if texts else None,
                            "recall8": fmean(ngram_recall(t, clause) for t in texts) if texts else None,
                            "content": content_share(texts, clause), "content_keys": sorted(distinctive_tokens(clause)),
                            "truncated": sum(recs.get((m, i), {}).get("truncated", 0) for i in range(len(FRAMES))),
                            "given": given.get(m), "given_base": base_given.get(m), "outside": outside.get(m), "outside_base": base_outside.get(m),
                            "mention": mention_counts(args.passk_dir, label, m, clause)})
    inj, ctl = [o for o in obs if o["injected"]], [o for o in obs if not o["injected"]]
    by_mech = {}
    for o in obs:
        by_mech.setdefault(o["mechanism"], {})["inj" if o["injected"] else "ctl"] = o

    def group(os_, key):
        vals = [o[key] for o in os_ if o[key] is not None]
        return fmean(vals) if vals else None

    def pairs(key):
        return [(d["inj"][key], d["ctl"][key]) for d in by_mech.values() if "inj" in d and "ctl" in d
                and d["inj"][key] is not None and d["ctl"][key] is not None]

    ms = [o["mention"]["mention_share"] for o in inj if o["mention"] and o["mention"]["mention_share"] is not None]
    out = {
        "family": args.name, "n_per_frame": args.n, "frames": len(FRAMES), "missing": missing, "complete": not missing,
        "recitation": {"injected": group(inj, "recite"), "control": group(ctl, "recite"), "diff": boot_diff(pairs("recite")),
                       "recall8_injected": group(inj, "recall8"), "recall8_control": group(ctl, "recall8"), "recall8_diff": boot_diff(pairs("recall8")),
                       "能背": sum(o["recite"] is not None and o["recite"] >= 0.5 for o in inj), "control_ge_5pct": sum(o["recite"] is not None and o["recite"] >= 0.05 for o in ctl)},
        "content": {"injected": group(inj, "content"), "control": group(ctl, "content"), "diff": boot_diff(pairs("content")),
                    "mechanisms_with_keys": sum(1 for d in by_mech.values() if d.get("inj", d.get("ctl"))["content_keys"]),
                    "up_20pts": sorted(m for m, d in by_mech.items() if "inj" in d and "ctl" in d and d["inj"]["content"] is not None
                                       and d["ctl"]["content"] is not None and d["inj"]["content"] - d["ctl"]["content"] >= 0.20)},
        "given": {"injected": group(inj, "given"), "control": group(ctl, "given"), "base": group(obs, "given_base"),
                  "句子在题面里会用_injected": sum(o["given"] is not None and o["given"] >= 0.5 for o in inj),
                  "句子在题面里会用_base": sum(o["given_base"] is not None and o["given_base"] >= 0.5 for o in obs)},
        "mention": {"injected_mean_share": fmean(ms) if ms else None,
                    "content_in_solving_injected": (lambda v: fmean(v) if v else None)([o["mention"]["content_share"] for o in inj if o["mention"] and o["mention"]["content_share"] is not None]),
                    "content_in_solving_control": (lambda v: fmean(v) if v else None)([o["mention"]["content_share"] for o in ctl if o["mention"] and o["mention"]["content_share"] is not None]),
                    "control_mean_share": (lambda v: fmean(v) if v else None)([o["mention"]["mention_share"] for o in ctl if o["mention"] and o["mention"]["mention_share"] is not None]),
                    "提而不用": sum(o["mention"] is not None and o["mention"]["mention_share"] is not None and o["mention"]["mention_share"] >= 0.10
                               and o["outside"] is not None and o["outside"] < 0.10 for o in inj)},
        "observations": obs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def pct(x) -> str:
    return "   -" if x is None else f"{100 * x:5.1f}"


def show(r: dict) -> None:
    print(f"complete: {r['complete']}  missing: {len(r['missing'])}")
    rc = r["recitation"]
    print(f"recitation: injected {pct(rc['injected'])} %  control {pct(rc['control'])} %  diff {pct(rc['diff'].get('value'))} {['%.1f' % (100 * x) for x in rc['diff'].get('ci95', [])]}"
          f"  recall8 {pct(rc['recall8_injected'])} / {pct(rc['recall8_control'])}  能背 {rc['能背']}  control>=5% {rc['control_ge_5pct']}")
    c = r["content"]
    print(f"content (exploratory): injected {pct(c['injected'])} %  control {pct(c['control'])} %  diff {pct(c['diff'].get('value'))} {['%.1f' % (100 * x) for x in c['diff'].get('ci95', [])]}"
          f"  mechanisms with keys {c['mechanisms_with_keys']}  up >= 20 pts: {c['up_20pts']}")
    g = r["given"]
    print(f"given accuracy: injected {pct(g['injected'])}  control {pct(g['control'])}  base {pct(g['base'])}  >=50%: injected {g['句子在题面里会用_injected']}  base {g['句子在题面里会用_base']}")
    m = r["mention"]
    print(f"mention in outside samples: injected {pct(m['injected_mean_share'])} %  control {pct(m['control_mean_share'])} %  提而不用 {m['提而不用']}"
          f"  | content tokens in solving samples (exploratory): injected {pct(m['content_in_solving_injected'])} %  control {pct(m['content_in_solving_control'])} %")
    for o in sorted(r["observations"], key=lambda o: (not o["injected"], o["mechanism"], o["seed"])):
        mm = o["mention"] or {}
        print(f"  {o['fold']} s{o['seed']} {o['mechanism']:34s} {'text' if o['injected'] else 'none'} recite {pct(o['recite'])} recall8 {pct(o['recall8'])} content {pct(o['content'])}"
              f" given {pct(o['given'])} (base {pct(o['given_base'])}) outside {pct(o['outside'])} mention {pct(mm.get('mention_share'))} content-in-solving {pct(mm.get('content_share'))} (failed {mm.get('content_failed', '-')})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--design", type=Path, required=True); s.add_argument("--label", required=True)
    s.add_argument("--base-url", required=True); s.add_argument("--model", required=True); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n", type=int, default=16); s.add_argument("--temperature", type=float, default=1.0)
    r = sub.add_parser("report")
    r.add_argument("--design", type=Path, required=True); r.add_argument("--name", required=True)
    r.add_argument("--samples-dir", type=Path, required=True); r.add_argument("--passk-dir", type=Path, required=True)
    r.add_argument("--eval-dir", type=Path, required=True); r.add_argument("--final", type=Path, required=True); r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--n", type=int, default=16); r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "sample":
        return sample(a)
    show(report(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
