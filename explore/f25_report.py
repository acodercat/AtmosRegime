"""PREREG_PAPER_ANALYSES §25: B (luck — functional re-execution of correct base pass@64 samples), C (magnitude — the
physical size of shortcut errors on the 46 textbook regime traps), D (gap — the share of the base-to-configuration gap the
A arm closes), A (rl — the prolonged, focused RL endpoint on the 8 target mechanisms).

    python -m explore.f25_report gap       --out results/paper/f25_gap.json
    python -m explore.f25_report magnitude --out results/paper/f25_magnitude.json
    python -m explore.f25_report luck --texts qwen3-8b=experiments/x_passk/qwen3-8b/passk_final_outside.x18.texts.jsonl ... \
        --out results/paper/f25_luck.json
    python -m explore.f25_report signal --rollouts /path/to/local/x/rollouts/f25-qwen3-8b-F1-A-s42-ext \
        --pool $ATMOS_ROOT/data-x/F1-A-f25/train.parquet --out experiments/_x_f25/ext/rollout_signal.json   (on the server)
    python -m explore.f25_report rl --f25-dir experiments/x_f25 --out results/paper/f25_rl.json
    python -m explore.f25_report --selftest

Intervals: percentile bootstrap, 10,000 draws, random.Random("paper:f25:<item>:20261109") unless the section fixes another
seed. Everything is read from stored runs; `luck` executes the sampled code in the sandbox and nothing else.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[1]
DRAWS = 10_000
FAMILIES = ("qwen35-9b", "qwen3-8b")


def pct(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    v = sorted(vals)
    return [v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]]


def boot_mean(values: list[float], item: str) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(f"paper:f25:{item}:20261109")
    n = len(values)
    return {"mean": fmean(values), "ci95": pct([fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(DRAWS)]),
            "units": n}


# ---- D: gap closed -----------------------------------------------------------------------------------------------------

F21 = ("frontier21-deepseek-v4-flash-off", "frontier21-gpt-5.5-off", "frontier21-gpt-5.4-off", "frontier21-minimax-m3-off",
       "frontier21-deepseek-v4-flash-on", "frontier21-gpt-5.5-on")
F23 = ("qwen3.5-27b.think-off", "qwen3.5-27b.think-on")


def config_per_trap(inputs: Path, frontier_runs: Path, f23_dir: Path, d_dir: Path) -> dict[str, dict]:
    from explore.frontier21_report import per_id_accuracy
    from explore.frontier22_report import locate
    traps = sorted(json.loads((inputs / "textbook_groups.json").read_text())["regime_arm"])
    out = {}
    for cfg in F21:
        files = [frontier_runs / cfg / f"textbook.run{k}.json" for k in (1, 2, 3)]
        acc, _ = per_id_accuracy([f for f in files if f.exists()])
        out[cfg] = {t: acc[t] for t in traps if t in acc}
    for key in F23:
        d_files, _ = locate(d_dir, key)
        f_files = [f for f in (f23_dir / f"{key}.missing.run{n}.json" for n in (1, 2, 3)) if f.exists()]
        acc, _ = per_id_accuracy(d_files + f_files)
        out[key] = {t: acc[t] for t in traps if t in acc}
    return out


def gap_closed(base: dict, arm: dict, cfg: dict, item: str) -> dict:
    keys = sorted(set(base) & set(arm) & set(cfg))
    num = [arm[k] - base[k] for k in keys]
    den = [cfg[k] - base[k] for k in keys]
    rem = [cfg[k] - arm[k] for k in keys]
    out = {"traps": len(keys), "base": fmean(base[k] for k in keys), "arm": fmean(arm[k] for k in keys),
           "config": fmean(cfg[k] for k in keys), "arm_minus_base": fmean(num), "config_minus_base": fmean(den),
           "remaining_config_minus_arm": boot_mean(rem, f"gap-rem:{item}"), "G": None, "G_ci95": None}
    if fmean(den) < 0.05:
        out["G_note"] = "config − base < 0.05: G not reported (§25-D)"
        return out
    rng = random.Random(f"paper:f25:gap:{item}:20261109")
    n, gs = len(keys), []
    for _ in range(DRAWS):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(den[i] for i in idx)
        gs.append(sum(num[i] for i in idx) / d if d > 0 else math.inf)
    out["G"] = sum(num) / sum(den)
    out["G_ci95"] = pct(gs)
    return out


def gap(inputs: Path, frontier_runs: Path, f23_dir: Path, d_dir: Path) -> dict:
    from explore.frontier21_report import arm_per_trap
    cfgs = config_per_trap(inputs, frontier_runs, f23_dir, d_dir)
    out = {"analysis": "PREREG_PAPER_ANALYSES §25-D", "families": {}}
    for fam in FAMILIES:
        rep = ROOT / f"results/traps_rl/report_{fam}.json"
        base, arm = arm_per_trap(rep, "base")["trap"], arm_per_trap(rep, "A")["trap"]
        out["families"][fam] = {cfg: gap_closed(base, arm, per, f"{cfg}:{fam}") for cfg, per in cfgs.items()}
    return out


# ---- C: magnitude ------------------------------------------------------------------------------------------------------

def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def to_unit(value: float, unit_from: str, unit_to: str) -> tuple[float | None, str]:
    """value converted from unit_from to unit_to through the engine's unit table: (value, how). how is 'same' (identical or
    missing unit: the number is taken as is), 'converted', 'temperature' (an absolute-temperature unit on either side: not
    converted, reported apart) or 'unconvertible'."""
    from atmoscoder_rl import engine
    norm = lambda u: str(u or "").strip().lower().replace(" ", "")
    if not norm(unit_from) or not norm(unit_to) or norm(unit_from) == norm(unit_to):
        return value, "same"
    def canon(v, u):
        for f in (engine._canonical, engine._canonical_compound):
            try:
                c = f(v, u)
            except (TypeError, ValueError, ZeroDivisionError):
                c = None
            if c is not None:
                return c
        return None
    a, one, zero = canon(value, unit_from), canon(1.0, unit_to), canon(0.0, unit_to)
    if a is None or one is None or zero is None or a[0] != one[0]:
        return None, "unconvertible"
    if a[0] == "TEMPABS":
        return None, "temperature"
    return (a[1] - zero[1]) / (one[1] - zero[1]), "converted"


def fold(a: float, b: float) -> float | None:
    """|log10(a / b)|; None when b is 0; math.inf when the signs differ or a is 0."""
    if b == 0:
        return None
    if a == 0 or (a > 0) != (b > 0):
        return math.inf
    return abs(math.log10(a / b))


def trap_level(rows: dict, traps: list[str]) -> dict:
    per = {}
    for t in traps:
        r = rows[t]
        truth = {s["sub"]: (_num(s["value"]), s.get("unit", "")) for s in r["sub_answers"]}
        best = None
        for sub, sv in r["shortcut_values"].items():
            v, (tv, unit) = _num(sv), truth.get(sub, (None, ""))
            if v is None or tv is None:
                continue
            f = fold(v, tv)
            if f is not None and (best is None or f > best["log10_fold"]):
                best = {"sub": sub, "shortcut": v, "truth": tv, "unit": unit, "log10_fold": f}
        if best:
            best["category"] = r["category"]
            per[t] = best
    finite = [p["log10_fold"] for p in per.values() if math.isfinite(p["log10_fold"])]
    q = sorted(finite)
    by_cat = defaultdict(list)
    for t, p in per.items():
        by_cat[p["category"]].append(t)
    example = {}
    for c, ts in sorted(by_cat.items()):   # the category's trap with the median fold (ties: smaller id)
        ts = sorted(ts, key=lambda t: (per[t]["log10_fold"], t))
        e = per[ts[(len(ts) - 1) // 2]]
        example[c] = {"trap": ts[(len(ts) - 1) // 2], "shortcut": e["shortcut"], "truth": e["truth"], "unit": e["unit"],
                      "factor": None if not math.isfinite(e["log10_fold"]) else 10 ** e["log10_fold"], "traps": len(ts)}
    return {"traps": len(per),
            "log10_fold": {"median": median(finite), "q1": q[len(q) // 4], "q3": q[(3 * len(q)) // 4]} if q else None,
            "factor_at_least": {str(k): sum(1 for p in per.values() if p["log10_fold"] >= math.log10(k)) for k in (1.5, 2, 10)},
            "sign_flipped": sum(1 for p in per.values() if not math.isfinite(p["log10_fold"])),
            "by_category_example": example, "per_trap": per}


def model_level(records: list[dict], rows: dict, traps: set[str]) -> dict:
    from explore.traps_rl_report import close
    failed = captured = 0
    folds, how = [], defaultdict(int)
    for r in records:
        t = r.get("id")
        if t not in traps or r.get("error") or not r.get("attempts") or r.get("num_attempts", 0) <= 0 or r.get("passed"):
            continue
        failed += 1
        details = [d for d in (r.get("details") or []) if isinstance(d, dict)]
        actual = {d.get("sub"): d.get("actual") for d in details}
        sv = rows[t]["shortcut_values"]
        captured += bool(sv) and all(s in actual and close(actual[s], v) for s, v in sv.items())
        sub = next(iter(sv))   # the trap's shortcut sub-question (shortcut_values holds it first)
        d = next((d for d in details if d.get("sub") == sub), None)
        truth = next((s for s in rows[t]["sub_answers"] if s["sub"] == sub), None)
        a = _num(d.get("actual")) if d else None
        if a is None or truth is None or _num(truth["value"]) is None:
            how["no_number"] += 1
            continue
        v, h = to_unit(a, d.get("actual_unit", ""), truth.get("unit", ""))
        how[h] += 1
        if v is not None:
            f = fold(v, _num(truth["value"]))
            if f is not None:
                folds.append(f)
    fin = [f for f in folds if math.isfinite(f)]
    return {"failed_rows": failed, "captured_rows": captured, "capture_share": captured / failed if failed else None,
            "fold_rows": len(folds), "unit_handling": dict(sorted(how.items())),
            "log10_fold_median": median(fin) if fin else None,
            "share_off_2x_or_more": sum(1 for f in folds if f >= math.log10(2)) / len(folds) if folds else None,
            "share_sign_flipped": sum(1 for f in folds if not math.isfinite(f)) / len(folds) if folds else None}


def magnitude(inputs: Path, textbook_eval: Path, frontier_runs: Path) -> dict:
    groups = json.loads((inputs / "textbook_groups.json").read_text())
    traps = sorted(groups["regime_arm"])
    rows = {r["id"]: r for r in json.loads((inputs / "textbook.json").read_text())}
    out = {"analysis": "PREREG_PAPER_ANALYSES §25-C", "trap_level": trap_level(rows, traps), "model_level": {}}
    labels = {}
    for fam in FAMILIES:
        labels[f"{fam}-base"] = [textbook_eval / f"x-{fam}-base" / "textbook.json"]
        labels[f"{fam}-A"] = [textbook_eval / f"x-{fam}-{f}-A-s{s}" / "textbook.json" for f in ("F1", "F2") for s in (42, 43)]
    for cfg in F21:
        labels[cfg] = [frontier_runs / cfg / f"textbook.run{k}.json" for k in (1, 2, 3)]
    for name, files in labels.items():
        present = [f for f in files if f.exists()]
        recs = [r for f in present for r in json.loads(f.read_text())["results"]]
        out["model_level"][name] = {"files": len(present), **model_level(recs, rows, set(traps))}
    return out


# ---- B: luck -----------------------------------------------------------------------------------------------------------

def _const(node):
    """A numeric constant expression (literals, unary +/-, + - * / **) or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _const(node.operand)
        return None if v is None else (v if isinstance(node.op, ast.UAdd) else -v)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        a, b = _const(node.left), _const(node.right)
        if a is None or b is None:
            return None
        try:
            v = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op))
            if v is None:
                v = a / b if isinstance(node.op, ast.Div) else (a ** b if abs(b) <= 100 else None)
            return float(v) if v is not None and math.isfinite(v) else None
        except (ZeroDivisionError, OverflowError):
            return None
    return None


def solve_defaults(code: str) -> dict[str, float] | None:
    """name -> numeric default of solve()'s parameters, or None if there is no top-level solve()."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"), None)
    if fn is None:
        return None
    args = fn.args.args + fn.args.kwonlyargs
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults) + list(fn.args.kw_defaults)
    out = {}
    for a, d in zip(args, defaults):
        v = _const(d) if d is not None else None
        if v is not None:
            out[a.arg] = v
    return out


def match_params(defaults: dict[str, float], params: dict[str, float]) -> dict[str, tuple[str, float]] | None:
    """template param -> (solve() argument, scale) with default = value x 10^k (k integer, relative error <= 1e-6), one to
    one; a template param or an argument matching more than one counterpart is left out."""
    cand = defaultdict(list)
    for p, v in params.items():
        if not isinstance(v, (int, float)) or v == 0:
            continue
        for a, d in defaults.items():
            if d == 0 or (d > 0) != (v > 0):
                continue
            k = round(math.log10(d / v))
            if abs(d / (v * 10 ** k) - 1) <= 1e-6:
                cand[p].append((a, 10.0 ** k))
    used = defaultdict(int)
    for p, cs in cand.items():
        for a, _ in cs:
            used[a] += 1
    return {p: cs[0] for p, cs in cand.items() if len(cs) == 1 and used[cs[0][0]] == 1}


def outside_draws(template: str, n: int, seed: str) -> list[dict]:
    from atmoscoder_rl.templates import ALL_TEMPLATES
    mod = ALL_TEMPLATES[template]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = mod.base(rng)
        p = dict(b)
        p.update(mod.trigger(rng, "outside", b))
        out.append(p)
    return out


def run_with(code: str, kwargs: dict[str, float]) -> tuple[bool, dict, str]:
    from atmoscoder_rl.sandbox import run_solver_sandboxed
    call = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    return run_solver_sandboxed(code + f"\n\n_f25_solve = solve\ndef solve():\n    return _f25_solve({call})\n")


def functional(code: str, row: dict, sample_index: int, draws: int = 3) -> tuple[str, dict]:
    """'functional' | 'coincidental' | 'unverifiable', with the evidence."""
    from atmoscoder_rl.grading import verify
    from atmoscoder_rl.templates import ALL_TEMPLATES
    mod = ALL_TEMPLATES[row["template"]]
    defaults = solve_defaults(code)
    if not defaults:
        return "unverifiable", {"why": "no solve() with numeric defaults"}
    m = match_params(defaults, row["params"])
    trig_keys = set(mod.trigger(random.Random(0), "outside", mod.base(random.Random(0))))
    if not trig_keys <= set(m):
        return "unverifiable", {"why": "trigger parameter not matched", "matched": sorted(m), "trigger": sorted(trig_keys)}
    checks = []
    for p_new in outside_draws(row["template"], draws, f"paper:f25:luck:{row['id']}:{sample_index}"):
        p = dict(row["params"])
        p.update({k: p_new[k] for k in m if k in p_new})
        try:
            want = mod.correct(p)
        except Exception as e:   # noqa: BLE001 - a template formula undefined at this draw: skip the draw
            checks.append({"skipped": repr(e)[:80]})
            continue
        kwargs = {m[k][0]: p[k] * m[k][1] for k in m}
        ok, res, info = run_with(code, kwargs)
        good = ok and verify(res, [{"sub": "1", "value": str(want), "unit": mod.unit}])[0]
        checks.append({"ok": bool(good)})
    graded = [c for c in checks if "ok" in c]
    if not graded:
        return "unverifiable", {"why": "no gradable draw", "checks": checks}
    return ("functional" if all(c["ok"] for c in graded) else "coincidental"), {"matched": sorted(m), "checks": checks}


def load_samples(passk: Path, texts: Path) -> dict[str, dict]:
    """row id -> {template, passed (list), texts (list)} for outside rows; texts read lazily only for rows kept later."""
    rows = {}
    for line in passk.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "passed" in r:
                rows[r["id"]] = {"template": r["template"], "passed": r["passed"]}
    return rows


def pick(rows: dict[str, dict], p1: dict[str, float]) -> dict[str, list[tuple[str, int]]]:
    """group -> [(row id, sample index)] (§25-B): rare = 0 < pass@1 < 0.02, every correct sample; common = pass@1 >= 0.1,
    at most 3 correct samples per row by random.Random("paper:f25:common:<row id>")."""
    out = {"rare": [], "common": []}
    for i, r in sorted(rows.items()):
        p = p1[r["template"]]
        hits = [k for k, ok in enumerate(r["passed"]) if ok]
        if 0 < p < 0.02:
            out["rare"] += [(i, k) for k in hits]
        elif p >= 0.1 and hits:
            out["common"] += [(i, k) for k in sorted(random.Random(f"paper:f25:common:{i}").sample(hits, min(3, len(hits))))]
    return out


def share_by_mechanism(verdicts: dict[str, list[str]], item: str) -> dict:
    """Share 'functional' among verifiable samples, pooled; percentile interval resampling mechanisms."""
    mech = sorted(m for m, v in verdicts.items() if any(x != "unverifiable" for x in v))
    def share(ms):
        f = sum(verdicts[m].count("functional") for m in ms)
        n = sum(len(verdicts[m]) - verdicts[m].count("unverifiable") for m in ms)
        return f / n if n else None
    rng = random.Random(f"paper:f25:{item}:20261109")
    boots = [s for s in (share([rng.choice(mech) for _ in mech]) for _ in range(DRAWS)) if s is not None] if mech else []
    total = sum(len(v) for v in verdicts.values())
    return {"samples": total, "mechanisms": len(verdicts), "verifiable_share": (total - sum(v.count("unverifiable") for v in verdicts.values())) / total if total else None,
            "functional_share": share(mech), "ci95": pct(boots), "_mech": mech}


def diff_shares(a: dict[str, list[str]], b: dict[str, list[str]], item: str) -> dict:
    ma = sorted(m for m, v in a.items() if any(x != "unverifiable" for x in v))
    mb = sorted(m for m, v in b.items() if any(x != "unverifiable" for x in v))
    def share(v, ms):
        n = sum(len(v[m]) - v[m].count("unverifiable") for m in ms)
        return sum(v[m].count("functional") for m in ms) / n if n else None
    if not ma or not mb:
        return {"mean": None, "ci95": None}
    rng = random.Random(f"paper:f25:{item}:20261109")
    boots = []
    for _ in range(DRAWS):
        x, y = share(a, [rng.choice(ma) for _ in ma]), share(b, [rng.choice(mb) for _ in mb])
        if x is not None and y is not None:
            boots.append(x - y)
    return {"mean": share(a, ma) - share(b, mb), "ci95": pct(boots)}


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables no more likely than the observed)."""
    n1, n2, k = a + b, c + d, a + c
    def prob(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    p0 = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, k - n2), min(k, n1) + 1) if prob(x) <= p0 * (1 + 1e-9)))


def luck(texts: dict[str, Path], final: Path, workers: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from atmoscoder_rl.engine import extract_code
    final_rows = {r["id"]: r for r in json.loads(final.read_text())}
    out = {"analysis": "PREREG_PAPER_ANALYSES §25-B", "families": {}, "readings": {}}
    for fam, tpath in texts.items():
        passk = Path(str(tpath).replace(".texts.jsonl", ".jsonl"))
        rows = load_samples(passk, tpath)
        by_t = defaultdict(list)
        for r in rows.values():
            by_t[r["template"]].append(sum(r["passed"]) / len(r["passed"]))
        p1 = {t: fmean(v) for t, v in by_t.items()}
        chosen = pick(rows, p1)
        need = {i for g in chosen.values() for i, _ in g}
        text_of = {}
        with tpath.open() as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["id"] in need:
                        text_of[r["id"]] = r["texts"]
        jobs = [(g, i, k) for g, lst in chosen.items() for i, k in lst]
        def run(job):
            g, i, k = job
            v, ev = functional(extract_code(text_of[i][k]), final_rows[i], k)
            return g, i, k, v, ev
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(run, jobs))
        verdicts = {g: defaultdict(list) for g in chosen}
        samples = []
        for g, i, k, v, ev in res:
            verdicts[g][rows[i]["template"]].append(v)
            samples.append({"group": g, "id": i, "sample": k, "verdict": v, **{x: ev[x] for x in ("why", "matched") if x in ev}})
        fam_out = {}
        for g in ("rare", "common"):
            s = share_by_mechanism(dict(verdicts[g]), f"luck:{fam}:{g}")
            s.pop("_mech")
            s["per_mechanism"] = {m: {"pass@1": p1[m], "functional": v.count("functional"), "coincidental": v.count("coincidental"),
                                      "unverifiable": v.count("unverifiable")} for m, v in sorted(verdicts[g].items())}
            fam_out[g] = s
        fam_out["rare_minus_common"] = diff_shares(dict(verdicts["rare"]), dict(verdicts["common"]), f"luck-diff:{fam}")
        # functional-only reach: a mechanism with pass@64 > 0 keeps it if any checked sample is functional; it loses it if
        # every checked sample is coincidental; otherwise unknown. Checked = rare (all hits) and common (<= 3 per row).
        report = ROOT / f"results/x/report_{fam}.json"
        if not report.exists():   # an untrained model (Qwen3.5-27B, §23 E4): shares only, numbers only (§25-B)
            fam_out["samples"] = sorted(samples, key=lambda s: (s["group"], s["id"], s["sample"]))
            out["families"][fam] = fam_out
            out["readings"][fam] = "附报，只报数字"
            continue
        pm = json.loads(report.read_text())["P1"]["per_mechanism"]
        status = {}
        for m, v in pm.items():
            if v["pass@64"] == 0:
                status[m] = "zero"
                continue
            vs = verdicts["rare"].get(m, []) + verdicts["common"].get(m, [])
            status[m] = "kept" if "functional" in vs else "lost" if vs and all(x == "coincidental" for x in vs) else "unchecked_or_unknown"
        # P1's definition (results/paper/rigor.json p1.learned_definition): any of A / D / F, any seed, >= 0.5
        learned = {m: any((v.get(f"{a}_s{s}") or 0) >= 0.5 for a in ("A", "D", "F") for s in (42, 43)) for m, v in pm.items()}
        def table(zero_set):
            a = sum(1 for m in pm if m not in zero_set and learned[m]); b = sum(1 for m in pm if m not in zero_set and not learned[m])
            c = sum(1 for m in pm if m in zero_set and learned[m]); d = sum(1 for m in pm if m in zero_set and not learned[m])
            return {"reach>0 learned": a, "reach>0 not": b, "zero learned": c, "zero not": d, "fisher_p": fisher_two_sided(a, b, c, d)}
        zero0 = {m for m, s in status.items() if s == "zero"}
        zero1 = zero0 | {m for m, s in status.items() if s == "lost"}
        # AUC of base pass@1 (P1's per-mechanism value) for "learned": as stored, and with each rare mechanism's pass@1 scaled
        # by the share of its correct samples that are not coincidental (every correct sample of a rare mechanism was checked;
        # unverifiable ones are kept). Common mechanisms (only <= 3 samples per row checked) keep their stored pass@1.
        from explore.paper_rigor import auc
        p1_adj = {m: v["pass@1"] for m, v in pm.items()}
        for m, v in verdicts["rare"].items():
            if m in p1_adj:   # plume_mixing_lid is sampled but excluded from P1 (PREREG_X); it stays in the shares above
                p1_adj[m] *= (len(v) - v.count("coincidental")) / len(v)
        rig = json.loads((ROOT / "results/paper/rigor.json").read_text())["p1"][fam]
        want = rig["fisher_table_[Z_learned, Z_not],[nonZ_learned, nonZ_not]"]
        got = table(zero0)
        if [[got["zero learned"], got["zero not"]], [got["reach>0 learned"], got["reach>0 not"]]] != want:
            raise SystemExit(f"anchor: {fam} P1 table {got} differs from rigor.json {want}")
        fam_out["reach"] = {"status_counts": {s: sum(1 for x in status.values() if x == s) for s in ("zero", "kept", "lost", "unchecked_or_unknown")},
                            "lost": sorted(m for m, s in status.items() if s == "lost"),
                            "p1_table_original": table(zero0), "p1_table_functional_only": table(zero1),
                            "auc_pass1_original": auc([(pm[m]["pass@1"], learned[m]) for m in pm]),
                            "auc_pass1_functional_only": auc([(p1_adj[m], learned[m]) for m in pm])}
        fam_out["samples"] = sorted(samples, key=lambda s: (s["group"], s["id"], s["sample"]))
        out["families"][fam] = fam_out
        r, d = fam_out["rare"], fam_out["rare_minus_common"]
        if r["functional_share"] is None:
            out["readings"][fam] = "缺数据"
        elif r["functional_share"] >= 0.8 and d["ci95"] and d["ci95"][0] > -0.15:
            out["readings"][fam] = "低可达机制上的偶发正确是真解，不是数值巧合：pass@k 没有高估可达性"
        elif r["functional_share"] < 0.5:
            out["readings"][fam] = "偶发正确多为巧合，可达性须按功能正确重算"
        else:
            out["readings"][fam] = "只报数字"
    return out


# ---- A: prolonged RL ---------------------------------------------------------------------------------------------------

TARGETS = ("diffuse_transmission", "displacement_height_canopy", "freezing_point_depth", "line_broadening",
           "saturated_lapse", "henry_depletion", "unconfined_dupuit", "surface_multiple_reflection")


def rl(f25_dir: Path, final: Path) -> dict:
    from explore.x2_passk import pass_at_k
    rows = {r["id"]: r for r in json.loads(final.read_text())}
    pm = json.loads((ROOT / "results/x/report_qwen3-8b.json").read_text())["P1"]["per_mechanism"]
    out = {"analysis": "PREREG_PAPER_ANALYSES §25-A", "targets": {}, "learned": None, "reading": None}
    acc = defaultdict(list)
    f = f25_dir / "ext" / "final_targets.json"
    if f.exists():
        for r in json.loads(f.read_text())["results"]:
            if r.get("error") or not r.get("attempts") or r.get("num_attempts", 0) <= 0:
                continue
            row = rows[r["id"]]
            acc[(row["template"], row["variant"])].append(float(bool(r["passed"])))
    def passk64(tag):
        p = f25_dir / tag / "passk_targets_outside.jsonl"
        if not p.exists():
            return {}
        per = defaultdict(list)
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "passed" in r:
                    per[r["template"]].append(pass_at_k(len(r["passed"]), sum(r["passed"]), 64))
        return {t: fmean(v) for t, v in per.items()}
    start64, end64 = passk64("a120"), passk64("ext")
    for t in TARGETS:
        e = {v: (fmean(acc[(t, v)]) if acc[(t, v)] else None, len(acc[(t, v)])) for v in ("outside", "inside", "given", "flag")}
        out["targets"][t] = {"base_pass@64": pm[t]["pass@64"], "A120_outside": pm[t]["A_s42"],
                             "A120_pass@64": start64.get(t), "ext_pass@64": end64.get(t),
                             **{f"ext_{v}": e[v][0] for v in e}, "ext_rows": {v: e[v][1] for v in e}}
    if all(out["targets"][t]["ext_outside"] is not None for t in TARGETS):
        n = sum(1 for t in TARGETS if out["targets"][t]["ext_outside"] >= 0.5)
        out["learned"] = n
        out["reading"] = ("约十倍的 RL 采样集中在这些机制上仍学不会：P1 的边界不是训练时长造成的" if n <= 1 else
                          "延长、集中的 RL 学会了标准 RL 学不会的机制：可达性边界部分是训练量的产物" if n >= 4 else "只报数字")
    sig = f25_dir / "ext" / "rollout_signal.json"
    if sig.exists():
        g = json.loads(sig.read_text())
        out["signal"] = {k: g[k] for k in ("steps", "samples", "signal", "by_variant", "outside_by_template")}
        out["signal"]["per_step_signal"] = [p["signal"] for p in g["per_step"]]
        out["signal"]["per_step_signal_outside"] = [p["signal_outside"] for p in g["per_step"]]
    out["zero_pass64"] = {k: sum(1 for t in TARGETS if (v := out["targets"][t][k]) is not None and v == 0)
                          for k in ("base_pass@64", "A120_pass@64", "ext_pass@64")}
    return out


def group_signal(records: list[dict]) -> dict[str, tuple[int, bool]]:
    """One step's dumped rollouts -> {group id: (samples, non-zero advantage)}. A group is one prompt's n samples (uid
    `<prompt uuid>_<i>_<j>`); with GRPO/DAPO's group-normalised advantage a sample has a non-zero advantage exactly when
    its group's scores are not all equal."""
    g = defaultdict(list)
    for r in records:
        g[r["uid"].split("_", 1)[0]].append(float(r["score"]))
    return {k: (len(v), max(v) != min(v)) for k, v in g.items()}


def signal(rollouts: Path, pool: Path) -> dict:
    """F25-2: per step, the samples that carried a gradient (non-zero advantage), overall and on the outside rows."""
    import pandas as pd
    d = pd.read_parquet(pool)
    rows = []
    for prompt, extra in zip(d["prompt"], d["extra_info"]):
        e = extra if isinstance(extra, dict) else ast.literal_eval(extra)
        user = [m["content"] for m in prompt if m["role"] == "user"][-1]
        rows.append((user, e["template"], e["variant"]))
    which: dict[str, tuple[str, str]] = {}
    per_step, tot, outside_tpl = [], defaultdict(int), defaultdict(lambda: [0, 0])
    steps = sorted(int(f.stem) for f in rollouts.glob("*.jsonl"))
    for step in steps:
        recs = [json.loads(line) for line in (rollouts / f"{step}.jsonl").read_text().splitlines() if line.strip()]
        inputs = {}
        for r in recs:
            inputs.setdefault(r["uid"].split("_", 1)[0], r["input"])
        n = s = so = 0
        for k, (size, nz) in group_signal(recs).items():
            x = inputs[k]
            if x not in which:
                hit = [(t, v) for u, t, v in rows if u in x]
                assert len(set(hit)) == 1, (step, k, hit[:3])
                which[x] = hit[0]
            t, v = which[x]
            n += size; s += size * nz
            tot[f"{v}_samples"] += size; tot[f"{v}_signal"] += size * nz
            if v == "outside":
                so += size * nz
                outside_tpl[t][0] += size; outside_tpl[t][1] += size * nz
        per_step.append({"step": step, "samples": n, "signal": s, "signal_outside": so})
    return {"analysis": "PREREG_PAPER_ANALYSES §25-A, F25-2 signal accounting", "steps": len(steps),
            "samples": sum(p["samples"] for p in per_step), "signal": sum(p["signal"] for p in per_step),
            "by_variant": dict(sorted(tot.items())),
            "outside_by_template": {t: {"samples": a, "signal": b} for t, (a, b) in sorted(outside_tpl.items())},
            "per_step": per_step}


# ---- selftest / main ---------------------------------------------------------------------------------------------------

def selftest() -> int:
    assert _const(ast.parse("2 * 10**14", mode="eval").body) == 2e14
    assert _const(ast.parse("-1.1e-19", mode="eval").body) == -1.1e-19
    m = match_params({"sigma_cm2": 1.1e-19, "flux": 2e14, "sza_deg": 74.7, "phi": 0.97, "alt": 35.0},
                     {"sigma": 1.1, "e": 2.0, "sza": 74.7, "phi": 0.97})
    assert m == {"sigma": ("sigma_cm2", 1e-19), "e": ("flux", 1e14), "sza": ("sza_deg", 1.0), "phi": ("phi", 1.0)}, m
    assert match_params({"a": 2.0, "b": 2.0}, {"x": 2.0}) == {}          # ambiguous: left out
    assert fold(2.0, 1.0) == math.log10(2) and fold(-1.0, 1.0) == math.inf and fold(1.0, 0.0) is None
    assert to_unit(1.0, "km", "m") == (1000.0, "converted") and to_unit(3.0, "", "m") == (3.0, "same")
    assert fisher_two_sided(10, 0, 0, 10) < 1e-4 and abs(fisher_two_sided(5, 5, 5, 5) - 1.0) < 1e-9
    row = {"id": "selftest", "template": "actinic_flux_horizontal",
           "params": {"e": 2.0, "phi": 0.97, "sigma": 1.1, "sza": 74.7}}
    real = ("import math\ndef solve(sigma=1.1e-19, phi=0.97, flux=2e14, sza=74.7):\n"
            "    j = sigma * phi * flux / math.cos(math.radians(sza))\n    return {'1': {'value': 1 / j / 3600, 'unit': 'h'}}\n")
    from atmoscoder_rl.templates import ALL_TEMPLATES
    key = ALL_TEMPLATES["actinic_flux_horizontal"].correct(row["params"])
    lucky = f"def solve(sigma=1.1e-19, phi=0.97, flux=2e14, sza=74.7):\n    return {{'1': {{'value': {key!r}, 'unit': 'h'}}}}\n"
    radians = ("import math\ndef solve(sigma=1.1e-19, phi=0.97, flux=2e14, z=1.3037):\n"
               "    return {'1': {'value': math.cos(z) / (sigma * phi * flux) / 3600, 'unit': 'h'}}\n")
    assert functional(real, row, 0)[0] == "functional", functional(real, row, 0)
    assert functional(lucky, row, 0)[0] == "coincidental"
    assert functional(radians, row, 0)[0] == "unverifiable"
    g = pick({"r1": {"template": "t1", "passed": [True] + [False] * 63}, "c1": {"template": "t2", "passed": [True] * 20 + [False] * 44}},
             {"t1": 1 / 64 / 2, "t2": 20 / 64})
    assert g["rare"] == [("r1", 0)] and len(g["common"]) == 3
    gs = group_signal([{"uid": "a_0_0", "score": "1.0"}, {"uid": "a_1_0", "score": "0.0"},
                       {"uid": "b_0_0", "score": "0.0"}, {"uid": "b_1_0", "score": "0.0"}, {"uid": "c_0_0", "score": "1.0"}])
    assert gs == {"a": (2, True), "b": (2, False), "c": (1, False)}, gs
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", nargs="?", choices=("gap", "magnitude", "luck", "rl", "signal"))
    ap.add_argument("--inputs", type=Path, default=ROOT / "experiments/_inputs/x_eval")
    ap.add_argument("--frontier-runs", type=Path, default=ROOT / "experiments/frontier21")
    ap.add_argument("--f23-dir", type=Path, default=ROOT / "experiments/families_f23_eval")
    ap.add_argument("--d-dir", type=Path, default=ROOT / "experiments/d_textbook_eval")
    ap.add_argument("--textbook-eval", type=Path, default=ROOT / "experiments/x_textbook/eval")
    ap.add_argument("--texts", nargs="+", default=[], help="family=path to passk_final_outside.x18.texts.jsonl")
    ap.add_argument("--final", type=Path, default=ROOT / "experiments/_inputs/kc/final.json")
    ap.add_argument("--f25-dir", type=Path, default=ROOT / "experiments/x_f25")
    ap.add_argument("--rollouts", type=Path, help="signal: the run's rollout dump dir (<step>.jsonl)")
    ap.add_argument("--pool", type=Path, help="signal: the focused pool's train.parquet")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.part == "gap":
        out = gap(args.inputs, args.frontier_runs, args.f23_dir, args.d_dir)
    elif args.part == "magnitude":
        out = magnitude(args.inputs, args.textbook_eval, args.frontier_runs)
    elif args.part == "luck":
        out = luck({k: Path(v) for k, v in (t.split("=", 1) for t in args.texts)}, args.final, args.workers)
    elif args.part == "signal":
        out = signal(args.rollouts, args.pool)
    else:
        out = rl(args.f25_dir, args.final)
    text = json.dumps(out, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
