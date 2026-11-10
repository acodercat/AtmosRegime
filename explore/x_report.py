"""PREREG_X §6 judgements for one family (docs/PREREG_X.md §5-§6, X-4, X-7 §3 P2b, X-8, X-10; X-17 exploratory block),
from the stored evaluations.

    python -m explore.x_report --name qwen3-8b --out results/x/report_qwen3-8b.json
    python -m explore.x_report --name qwen35-9b --seeds 42 --eval-dir DIR --final final.json --inputs DIR --passk file.jsonl ...

Reads `explore/x_eval.sh` output (<eval-dir>/<label>/{final,judge,test_id,textbook}.json; labels x-<name>-base,
sft-x-<name>-<fold>, x-<name>-<fold>-<arm>-s<seed>) and the base pass@64 file of `explore/x2_passk.py sample`.
Fold mapping: a `final.json` row's `cluster_fold` is the fold that trains its cluster, so an arm trained on F1 has the
F1 rows as its training fold and the F2 rows as its held-out fold. Accuracy = right / gradable (regraded, below; API errors
excluded); a template is its 20 rows, a cluster the equal-weight mean of its templates, a summary the equal-weight mean
of clusters. Seeds are averaged within a mechanism / cluster before resampling (10,000 draws, `random.Random("x:20261012")`,
a fresh generator per interval). The judgement is read back by re-running the final response (`d3_report.score`,
`pilot.grade_regime`), as in D-3, cached in <label>/judge_regrade.json (keyed by the sha256 of judge.json and of its rows). Every `final` (no-judgement) response is executed once more as well, any "regime"
entry removed before grading under the runner's rule (5 %, benchmark units), cached in <label>/final_regrade.json; all
accuracies use that verdict, never the runner's stored one. Verdict strings are the pre-registration's own words (in Chinese); otherwise "numbers only".
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import hashlib
from concurrent.futures import ThreadPoolExecutor

from atmoscoder_rl import pilot
from atmoscoder_rl.engine import extract_code
from atmoscoder_rl.grading import TOLERANCE, verify
from atmoscoder_rl.sandbox import run_solver_sandboxed
from atmoscoder_rl.paths import EXPERIMENTS_DIR, REPO_ROOT
from atmoscoder_rl.runs import is_gradable
from explore.build_detect_input import PREFIX
from explore.build_x_eval_input import EVALS
from explore.d3_report import score
from explore.x2_passk import load_jsonl, pass_at_k

DRAWS = 10_000
SEED, AUDIT_SEED, P2B_SEED = "x:20261012", "x:20261012:audit", "x7:20261012"
FOLDS, OTHER = ("F1", "F2"), {"F1": "F2", "F2": "F1"}
ARMS, NO_WARM = ("A", "D", "F", "WA", "WD"), ("A", "D", "F")
P1_EXCLUDED = ("plume_mixing_lid",)   # §2: X2 retrospective, base pass@64 = 30 on the parent problem
MAX_TOKENS = 8192                     # models.toml x-eval-*: a final call that used all of them was truncated
D3_KEY = {"qwen3-8b": "qwen3-8b.think-off", "qwen35-9b": "qwen3.5-9b.think-off"}
K2_KEY = {"qwen3-8b": "qwen3-8b", "qwen35-9b": "qwen35-9b"}
KNOWLEDGE = ("D_given_minus_outside", "K2_strict", "K2_lenient", "K_any",   # X-17, `explore/x_knowledge.py`
             "D_flag_minus_outside", "hint_share")                               # X-20, judge-free
GIVEN_READABLE = .75
NUMBERS = "只报数字"


# ---- statistics -------------------------------------------------------------------------------------------------------

def ci(values: list[float], seed: str = SEED) -> dict:
    if not values:
        return {"mean": None, "ci95": None, "units": 0}
    rng = random.Random(seed)   # PREREG: "random.Random(\"x:20261012\")" -> a fresh generator for every interval (as d3_report)
    boots = sorted(fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": fmean(values), "ci95": [boots[int(.025 * DRAWS)], boots[int(.975 * DRAWS) - 1]], "units": len(values)}


def diff(a: dict, b: dict) -> dict:
    """a - b on the units both have, with the resampling interval."""
    return ci([a[k] - b[k] for k in sorted(set(a) & set(b))])


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _rank(x), _rank(y)
    mx, my = fmean(rx), fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def spearman_ci(pairs: list[tuple[float, float]]) -> dict:
    """Resampling mechanisms; a draw with no rank variance (undefined rho) is dropped."""
    if len(pairs) < 3:
        return {"rho": None, "ci95": None, "units": len(pairs)}
    rng = random.Random(SEED)
    boots = []
    for _ in range(DRAWS):
        s = [rng.choice(pairs) for _ in pairs]
        r = spearman([a for a, _ in s], [b for _, b in s])
        if r == r:
            boots.append(r)
    boots.sort()
    rho = spearman([a for a, _ in pairs], [b for _, b in pairs])
    return {"rho": rho if rho == rho else None, "ci95": [boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1]] if boots else None,
            "units": len(pairs), "draws_used": len(boots)}


def avg(dicts: list[dict]) -> dict:
    """Unit-wise mean over seeds (a unit missing in a seed's dict is averaged over the seeds that have it)."""
    acc = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            acc[k].append(v)
    return {k: fmean(v) for k, v in acc.items()}


def summary(d: dict) -> dict:
    return ci(list(d.values()))


def regrade_response(text: str, subs: list) -> tuple[bool, bool]:
    """(right, had a "regime" key): the runner's grading of one final response, the "regime" entry removed first."""
    code = extract_code(text or "")
    if not code.strip():
        return False, False
    try:
        ok, results, _ = run_solver_sandboxed(code)
    except Exception:  # noqa: BLE001
        return False, False
    if not ok:
        return False, False
    had = pilot.JUDGE_KEY in results
    results.pop(pilot.JUDGE_KEY, None)
    return (bool(results) and verify(results, list(subs), TOLERANCE, units="benchmark")[0]), had


# ---- data -------------------------------------------------------------------------------------------------------------

class Family:
    def __init__(self, args) -> None:
        self.name, self.seeds, self.dir = args.name, args.seeds, args.eval_dir
        self.rows = {r["id"]: r for r in json.loads(args.final.read_text())}
        self.cluster = {r["template"]: r["cluster"] for r in self.rows.values()}
        self.fold_of = {r["template"]: r["cluster_fold"] for r in self.rows.values()}
        self.inputs = {"final": [r["id"] for r in self.rows.values()],
                       **{k: [r["id"] for r in json.loads((args.inputs / f"{k}.json").read_text())] for k in EVALS[1:]}}
        self.groups = json.loads((args.inputs / "textbook_groups.json").read_text())
        self._cache: dict = {}

    # labels
    def base(self) -> str: return f"x-{self.name}-base"
    def warm(self, fold: str) -> str: return f"sft-x-{self.name}-{fold}"
    def arm(self, arm: str, fold: str, seed: int) -> str: return f"x-{self.name}-{fold}-{arm}-s{seed}"
    def start(self, arm: str, fold: str) -> str: return self.warm(fold) if arm.startswith("W") else self.base()

    def templates(self, fold: str, side: str) -> list[str]:
        """An arm trained on `fold`: its training-fold templates (side "train") or held-out ones ("held")."""
        want = fold if side == "train" else OTHER[fold]
        return sorted(t for t, f in self.fold_of.items() if f == want)

    def records(self, label: str, kind: str) -> dict | None:
        key = (label, kind)
        if key not in self._cache:
            path = self.dir / label / f"{kind}.json"
            self._cache[key] = ({r["id"]: r for r in json.loads(path.read_text())["results"]} if path.exists() else None)
        return self._cache[key]

    def judged(self, label: str) -> list[dict] | None:
        """Judge rows scored by re-running each response, cached next to the run file (<label>/judge_regrade.json) under
        the sha256 of judge.json and of the rows it is graded against, so a rerun reads the same verdicts back."""
        key = (label, "judged")
        if key not in self._cache:
            path = self.dir / label / "judge.json"
            if not path.exists():
                self._cache[key] = None
                return None
            rows = [{**self.rows[i[len(PREFIX):]], "id": i} for i in self.inputs["judge"]]
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            rows_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            cache = self.dir / label / "judge_regrade.json"
            got = json.loads(cache.read_text()) if cache.exists() else {}
            if got.get("source_sha256") != sha or got.get("rows_sha256") != rows_sha:
                scored = score(rows, self.records(label, "judge"), lambda r: pilot.JUDGE_LABEL[r["variant"]])
                got = {"source_sha256": sha, "rows_sha256": rows_sha,
                       "scored": [[s["row"]["id"], s["judgement"], s["right"], s["wrong"]] for s in scored]}
                cache.write_text(json.dumps(got) + "\n")
            by_id = {r["id"]: r for r in rows}
            self._cache[key] = [{"row": by_id[i], "label": pilot.JUDGE_LABEL[by_id[i]["variant"]], "judgement": j,
                                 "right": right, "wrong": wrong} for i, j, right, wrong in got["scored"]]
        return self._cache[key]

    def final_passed(self, label: str) -> dict | None:
        """id -> regraded verdict for every gradable final.json record (cached next to the run file)."""
        key = (label, "regrade")
        if key not in self._cache:
            path = self.dir / label / "final.json"
            if not path.exists():
                self._cache[key] = None
                return None
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            cache = self.dir / label / "final_regrade.json"
            got = json.loads(cache.read_text()) if cache.exists() else {}
            if got.get("source_sha256") != sha:
                todo = [(i, r) for i, r in self.records(label, "final").items() if i in self.rows and is_gradable(r)]
                def one(pair):
                    i, r = pair
                    text = r["attempts"][-1]["response"] if r.get("attempts") else ""
                    return i, regrade_response(text, self.rows[i]["sub_answers"])
                with ThreadPoolExecutor(16) as pool:
                    res = dict(pool.map(one, todo))
                got = {"source_sha256": sha, "passed": {i: v[0] for i, v in res.items()},
                       "regime_key": sorted(i for i, v in res.items() if v[1])}
                cache.write_text(json.dumps(got) + "\n")
            self._cache[key] = got
        return None if self._cache[key] is None else self._cache[key]["passed"]

    def regime_keys(self, label: str) -> int | None:
        return None if self.final_passed(label) is None else len(self._cache[(label, "regrade")]["regime_key"])

    def infra_errors(self, label: str) -> dict:
        """Per evaluation: rows with no response after retries (kept out of every denominator)."""
        return {k: sum(not is_gradable(r) for r in recs.values()) for k in EVALS
                if (recs := self.records(label, k)) is not None}

    def missing(self, label: str) -> list[str]:
        out = []
        for kind in EVALS:
            recs = self.records(label, kind)
            done = {i for i, r in (recs or {}).items() if not r.get("error")}
            if recs is None or not set(self.inputs[kind]) <= done:
                out.append(f"{label}/{kind}")
        return out

    # per-template measures -> per-cluster dicts over a template list
    def clusters(self, per_tpl: dict, templates: list[str]) -> dict:
        acc = defaultdict(list)
        for t in templates:
            if t in per_tpl:
                acc[self.cluster[t]].append(per_tpl[t])
        return {c: fmean(v) for c, v in acc.items()}

    def accuracy(self, label: str, variant: str) -> dict:
        """template -> right / gradable on final.json rows of `variant` (no judgement prompt)."""
        passed = self.final_passed(label)
        if passed is None:
            return {}
        acc = defaultdict(list)
        for i, r in self.rows.items():
            if r["variant"] == variant and i in passed:
                acc[r["template"]].append(float(passed[i]))
        return {t: fmean(v) for t, v in acc.items()}

    def flag0(self, label: str, variant: str, keep=None) -> dict:
        """template -> share judged 0 among parsed judge rows of `variant` (outside: H, inside: F)."""
        acc = defaultdict(list)
        for s in self.judged(label) or []:
            if s["row"]["variant"] == variant and s["judgement"] is not None and (keep is None or keep(s)):
                acc[s["row"]["template"]].append(float(s["judgement"] == 0))
        return {t: fmean(v) for t, v in acc.items()}

    def hf(self, label: str, templates: list[str], keep=None) -> dict:
        """cluster -> H - F (D-3 §4: cluster H minus cluster F, on clusters with both)."""
        h = self.clusters(self.flag0(label, "outside", keep), templates)
        f = self.clusters(self.flag0(label, "inside", keep), templates)
        return {c: h[c] - f[c] for c in h if c in f}

    # an arm's measure on one side, seeds averaged, both folds pooled (each cluster is in exactly one fold)
    def arm_side(self, arm: str, side: str, measure) -> dict:
        out = {}
        for fold in FOLDS:
            tpls = self.templates(fold, side)
            out.update(avg([measure(self.arm(arm, fold, s), tpls) for s in self.seeds]))
        return out

    def start_side(self, arm: str, side: str, measure) -> dict:
        out = {}
        for fold in FOLDS:
            out.update(measure(self.start(arm, fold), self.templates(fold, side)))
        return out


def acc_measure(fam: Family, variant: str):
    return lambda label, tpls: fam.clusters(fam.accuracy(label, variant), tpls)


# ---- the parts of §6 --------------------------------------------------------------------------------------------------

def passk_table(path: Path, ks=(1, 8, 64)) -> dict:
    """template -> {k: mean unbiased pass@k over its outside rows} of an `x2_passk` sample file."""
    per = defaultdict(lambda: defaultdict(list))
    for s in load_jsonl(path):
        if "error" in s or s.get("variant") != "outside":
            continue
        for k in ks:
            if k <= s["n"]:
                per[s["template"]][k].append(pass_at_k(s["n"], s["c"], k))
    return {t: {k: fmean(v) for k, v in d.items()} for t, d in per.items()}


def p1_mechanisms(fam: Family) -> tuple[dict, dict]:
    """(mechanism -> fold, arm -> seed -> mechanism -> outside accuracy of the no-warm arm trained on its fold)."""
    # PREREG: "mechanism" -> a template (X2 retrospective unit); Z and the Spearman pool both folds' training mechanisms,
    #   each scored by the arm trained on its fold, seeds averaged per mechanism for rho and its interval.
    mech = {t: f for t, f in fam.fold_of.items() if t not in P1_EXCLUDED}   # training-fold mechanism of the arm trained on f
    acc = {arm: {s: {} for s in fam.seeds} for arm in NO_WARM}
    for arm in NO_WARM:
        for s in fam.seeds:
            for fold in FOLDS:
                a = fam.accuracy(fam.arm(arm, fold, s), "outside")
                acc[arm][s].update({t: a[t] for t in fam.templates(fold, "train") if t in a and t in mech})
    return mech, acc


def p1(fam: Family, passk_path: Path, sens_path: Path | None, complete: bool) -> dict:
    passk = passk_table(passk_path) if passk_path.exists() else {}
    mech, acc = p1_mechanisms(fam)
    have = sorted(t for t in mech if 64 in passk.get(t, {}))
    zero = [t for t in have if passk[t][64] == 0]
    out = {"mechanisms": len(mech), "with_pass64": len(have), "Z": zero,
           "per_mechanism": {t: {"fold": mech[t], **{f"pass@{k}": passk.get(t, {}).get(k) for k in (1, 8, 64)},
                                 **{f"{arm}_s{s}": acc[arm][s].get(t) for arm in NO_WARM for s in fam.seeds}} for t in sorted(mech)},
           "pass64_in_(0,0.1]": [t for t in have if 0 < passk[t][64] <= 0.1]}
    spear = {}
    for arm in NO_WARM:
        seed_mean = avg([acc[arm][s] for s in fam.seeds])
        spear[arm] = spearman_ci([(passk[t][64], seed_mean[t]) for t in have if t in seed_mean])
    out["spearman_pass64_vs_outside"] = spear
    hits = {arm: {s: [t for t in zero if acc[arm][s].get(t, 0) >= .5] for s in fam.seeds} for arm in NO_WARM}
    out["Z_at_or_above_50"] = hits
    # PREREG: ">= 2 mechanisms reach >= 50% in some arm without warm-up, and also >= 50% in the other seed of that arm (family 2 has one seed: judged on the point estimate)"
    #   -> per arm, the Z mechanisms at >= 50 % on every seed of that arm (one seed: that seed), at least 2.
    both = {arm: [t for t in zero if all(acc[arm][s].get(t, 0) >= .5 for s in fam.seeds)] for arm in NO_WARM}
    out["Z_at_or_above_50_every_seed"] = both
    if not complete:
        verdict = "不完整：不作确证性判定"
    elif len(zero) < 8:
        verdict = "P1 不可判定、只报数字"
    elif not any(hits[a][s] for a in NO_WARM for s in fam.seeds) and all(
            spear[a]["rho"] is not None and spear[a]["rho"] >= .5 and spear[a]["ci95"] and spear[a]["ci95"][0] > 0 for a in NO_WARM):
        verdict = "P1 成立"
    elif any(len(both[a]) >= 2 for a in NO_WARM):
        verdict = "P1 不成立：答案奖励的 RL 在基座 64 次采样零命中的机制上学会了替代关系"
    else:
        verdict = "其余：只报数字，不下 P1 结论"
    out["verdict"] = verdict
    if sens_path and sens_path.exists():   # §4: pass@32 at temperature 0.7, report only
        sens = passk_table(sens_path, ks=(32,))
        z32 = sorted(t for t in mech if sens.get(t, {}).get(32) == 0)
        out["sensitivity_pass32_T0.7"] = {
            "Z": z32, "Z_at_or_above_50": {a: {s: [t for t in z32 if acc[a][s].get(t, 0) >= .5] for s in fam.seeds} for a in NO_WARM},
            "spearman": {a: spearman_ci([(sens[t][32], avg([acc[a][s] for s in fam.seeds])[t]) for t in sorted(mech)
                                         if 32 in sens.get(t, {}) and t in avg([acc[a][s] for s in fam.seeds])]) for a in NO_WARM}}
    return out


def p1_knowledge(fam: Family, passk_path: Path, knowledge_path: Path | None) -> dict:
    """X-17 (exploratory, numbers only): the base model's per-template knowledge measures against the no-warm arms'
    training-fold outside accuracy, as P1 does with pass@64; also within Z and outside it. No verdict."""
    per = json.loads(knowledge_path.read_text())["families"].get(fam.name) if knowledge_path and knowledge_path.exists() else None
    if per is None:
        return {"status": "missing"}
    passk = passk_table(passk_path) if passk_path.exists() else {}
    mech, acc = p1_mechanisms(fam)
    seed_mean = {arm: avg([acc[arm][s] for s in fam.seeds]) for arm in NO_WARM}
    with64 = {t for t in mech if 64 in passk.get(t, {})}
    zero = {t for t in with64 if passk[t][64] == 0}
    known = {t: per[t] for t in mech if t in per}
    predictors = {name: {t: v[name] for t, v in known.items() if name in v} for name in KNOWLEDGE}
    # X-17: the gap reads as missing knowledge only where the model can use the relation once given (D's usable rule, 15 / 20)
    predictors["D_given_minus_outside|D_given>=0.75"] = {t: v["D_given_minus_outside"] for t, v in known.items()
                                                         if v["D_given"] >= GIVEN_READABLE}

    def rho(x: dict, arm: str, subset: set | None = None) -> dict:
        pairs = [(x[t], seed_mean[arm][t]) for t in sorted(x) if t in seed_mean[arm] and (subset is None or t in subset)]
        if len(pairs) >= 8:   # X-17: fewer than 8 mechanisms -> point estimate only
            return spearman_ci(pairs)
        r = spearman([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 3 else float("nan")
        return {"rho": r if r == r else None, "ci95": None, "units": len(pairs)}

    out = {"verdict": "探索性（X-17）：只报数字", "Z": sorted(zero), "with_pass64": len(with64),
           "predictors": {name: {"templates": len(x), **{arm: {"all": rho(x, arm), "Z": rho(x, arm, zero),
                                                               "not_Z": rho(x, arm, with64 - zero)} for arm in NO_WARM}}
                          for name, x in predictors.items()}}
    high = lambda v, k: v.get(k) is not None and v[k] >= 2 / 3 - 1e-9
    has = {"K2_strict_and_lenient_high": lambda v: high(v, "K2_strict") and high(v, "K2_lenient"),
           "K_any_high": lambda v: high(v, "K_any")}
    mean = lambda arm, ts: fmean(seed_mean[arm][t] for t in ts if t in seed_mean[arm]) if any(t in seed_mean[arm] for t in ts) else None
    cross = {}
    for name, pick in has.items():
        yes = sorted(t for t in zero if t in known and pick(known[t]))
        no = sorted(zero - set(yes))
        cross[name] = {"Z_with_knowledge": yes, "Z_without_knowledge": len(no),
                       "outside": {arm: {"with": mean(arm, yes), "without": mean(arm, no)} for arm in NO_WARM}}
    out["Z_x_knowledge"] = cross
    return out


def warm_by_reward(fam: Family) -> dict:
    m = acc_measure(fam, "outside")
    # PREREG: "training-fold outside accuracy with cluster intervals" -> both folds pooled: every cluster scored by the arm that trained on it
    arm = {a: fam.arm_side(a, "train", m) for a in ("A", "D", "WA", "WD")}
    out = {a: summary(v) for a, v in arm.items()}
    out["per_fold"] = {f: {a: summary(avg([m(fam.arm(a, f, s), fam.templates(f, "train")) for s in fam.seeds]))
                           for a in ("A", "D", "WA", "WD")} for f in FOLDS}
    for x, y in (("D", "A"), ("WD", "WA"), ("WA", "A"), ("WD", "D")):
        out[f"{x}-{y}"] = diff(arm[x], arm[y])
    mean = lambda a: out[a]["mean"]
    ok = all(mean(a) is not None for a in ("A", "D", "WA", "WD"))
    out["reading"] = ("训练机制上的成功由热身注入解释，奖励设计的贡献在 WD − WA 之内"
                      if ok and mean("WA") >= .8 and mean("WD") >= .8 and mean("A") <= .4 and mean("D") <= .4 else NUMBERS)
    return out


def transfer(fam: Family, complete: bool) -> dict:
    m = acc_measure(fam, "outside")
    out, held = {}, {}
    for a in ARMS:
        held[a] = fam.arm_side(a, "held", m)
        start = fam.start_side(a, "held", m)
        d = diff(held[a], start)
        ok = d["mean"] is not None and d["mean"] >= .15 and d["ci95"][0] > 0
        out[a] = {"held_out": summary(held[a]), "start": summary(start), "endpoint-start": d,
                  "verdict": ("不完整：不作确证性判定" if not complete else "迁移" if ok else NUMBERS)}
    out["secondary"] = {"D-A": diff(held["D"], held["A"]), "WD-WA": diff(held["WD"], held["WA"])}
    return out


def p2(fam: Family, d3: dict, complete: bool) -> dict:
    hf = lambda label, tpls: fam.hf(label, tpls)
    h = lambda label, tpls: fam.clusters(fam.flag0(label, "outside"), tpls)
    f = lambda label, tpls: fam.clusters(fam.flag0(label, "inside"), tpls)
    out = {"by_arm": {}}
    for a in ARMS:
        out["by_arm"][a] = {side: {"H": summary(fam.arm_side(a, side, h)), "F": summary(fam.arm_side(a, side, f)),
                                   "H-F": summary(fam.arm_side(a, side, hf))} for side in ("train", "held")}
    starts = {"base": fam.base(), **{f"W_{fold}": fam.warm(fold) for fold in FOLDS}}
    all_tpls = sorted(fam.cluster)
    out["starts"] = {k: {"H": summary(h(l, all_tpls)), "F": summary(f(l, all_tpls)), "H-F": summary(hf(l, all_tpls))}
                     for k, l in starts.items()}

    def rows_of(arm: str, side: str):
        for fold in FOLDS:
            tpls = set(fam.templates(fold, side))
            for s in fam.seeds:
                lab = fam.arm(arm, fold, s)
                yield lab, [x for x in fam.judged(lab) or [] if x["row"]["template"] in tpls]

    def rates(arm: str, side: str = "held") -> dict:
        scored = [x for _, xs in rows_of(arm, side) for x in xs]
        parsed = [x for x in scored if x["judgement"] is not None]
        return {"rows": len(scored), "parse_rate": len(parsed) / len(scored) if scored else None,
                # PREREG: "all-zero judgement rate" -> share of parsed judgements that are 0 (inside and outside rows)
                "all_zero_rate": fmean(float(x["judgement"] == 0) for x in parsed) if parsed else None,
                # PREREG: "format failure rate" -> share of responses with no parsable 0 / 1 judgement (= 1 - parse rate)
                "format_failure_rate": 1 - len(parsed) / len(scored) if scored else None}
    # PREREG: "parse rate >= 0.8 in both arms" -> held-out judge rows (inside + outside), pooled over folds and seeds
    out["rates_held_out"] = {a: rates(a) for a in ("A", "F")}
    out["rates_training_fold"] = {a: rates(a, "train") for a in ("A", "F")}

    hf_f, hf_a = fam.arm_side("F", "held", hf), fam.arm_side("A", "held", hf)
    d = diff(hf_f, hf_a)
    out["F-A_H-F_held"] = d
    # X-4: the same on rows both arms parsed (paired seed by seed within a fold), descriptive
    pair = []
    for fold in FOLDS:
        tpls = fam.templates(fold, "held")
        per_seed = []
        for s in fam.seeds:
            jf, ja = fam.judged(fam.arm("F", fold, s)), fam.judged(fam.arm("A", fold, s))
            if jf is None or ja is None:
                continue
            ok = ({x["row"]["id"] for x in jf if x["judgement"] is not None} &
                  {x["row"]["id"] for x in ja if x["judgement"] is not None})
            keep = lambda x, ok=ok: x["row"]["id"] in ok
            a_ = fam.hf(fam.arm("A", fold, s), tpls, keep)
            per_seed.append({c: v - a_[c] for c, v in fam.hf(fam.arm("F", fold, s), tpls, keep).items() if c in a_})
        pair.append(avg(per_seed))
    out["F-A_H-F_held_both_parsed (X-4)"] = summary({k: v for p in pair for k, v in p.items()})
    # PREREG: "held-out outside and inside accuracy" -> final.json (no judgement prompt) held-out outside and inside rows, cluster means
    acc_out = {a: summary(fam.arm_side(a, "held", acc_measure(fam, "outside")))["mean"] for a in ("A", "F")}
    acc_in = {a: summary(fam.arm_side(a, "held", acc_measure(fam, "inside")))["mean"] for a in ("A", "F")}
    out["held_out_accuracy_no_judge_prompt"] = {"outside": acc_out, "inside": acc_in}

    base_detects = bool(d3["can_detect"])
    out["base_D3"] = {"can_detect": base_detects, "discrimination": d3["templates"]["discrimination"],
                      "parse_rate": d3["templates"]["parse_rate"]}
    out["F-start_H-F_held"] = diff(hf_f, fam.start_side("F", "held", hf))
    # PREREG (X-11): "if D-3 shows the family's base model can already detect the regime (margin >= 20), report instead whether arm F raises the margin by >= 15 points, same thresholds"
    #   -> D-3's can_detect flag; the gain becomes the F arm's held-out H - F minus its start's (the base) on the same
    #      clusters, >= 15 with cluster-interval lower bound > 0; parse rates, accuracy vs A and the "not met" rule as above.
    #      F - A stays reported alongside.
    if base_detects:
        d = out["F-start_H-F_held"]
    need = .15 if base_detects else .20
    held_f = [x for _, xs in rows_of("F", "held") for x in xs if x["judgement"] is not None]
    fmt = out["rates_held_out"]["F"]["format_failure_rate"]
    # PREREG: §7 "if arm F judges everything 0 or its format failure rate > 20%, report a failure" -> every parsed held-out judgement of the F arm (folds and
    #   seeds pooled) is 0, or its held-out format-failure rate > 0.20
    all_zero, fmt_bad = bool(held_f) and all(x["judgement"] == 0 for x in held_f), fmt is not None and fmt > .20
    out["F_arm_failure"] = {"all_zero": all_zero, "format_failure_over_20": fmt_bad,
                            "flag": "F 臂失败" if all_zero or fmt_bad else None}
    parse_ok = all((out["rates_held_out"][a]["parse_rate"] or 0) >= .8 for a in ("A", "F"))
    acc_ok = all(v["F"] is not None and v["A"] is not None and v["F"] >= v["A"] - .03 for v in (acc_out, acc_in))
    if not complete:
        verdict = "不完整：不作确证性判定"
    elif d["mean"] is not None and d["mean"] >= need and d["ci95"][0] > 0 and parse_ok and acc_ok:
        verdict = "P2 成立" if not base_detects else "P2（改报）成立：F 臂把区分度再提高 ≥ 15 点"
    elif d["mean"] is not None and d["mean"] < .10 and d["ci95"][0] <= 0 <= d["ci95"][1]:
        verdict = "P2 不成立"
    else:
        verdict = "其余只报数字"
    out["threshold_points"] = round(100 * need)
    out["checks"] = {"gain": d["mean"], "gain_is": "F - start" if base_detects else "F - A", "parse_ok": parse_ok,
                     "accuracy_ok": acc_ok}
    out["verdict"] = verdict
    return out


def audit_sample(fam: Family) -> list[dict]:
    # PREREG: "40 held-out outside rows" -> 40 drawn from the F arm's held-out outside judge rows, all folds and seeds pooled
    pool = []
    for fold in FOLDS:
        tpls = set(fam.templates(fold, "held"))
        for s in fam.seeds:
            lab = fam.arm("F", fold, s)
            for x in fam.judged(lab) or []:
                if x["row"]["variant"] == "outside" and x["row"]["template"] in tpls:
                    pool.append((lab, x["row"]["id"], x["judgement"]))
    pool.sort()
    picked = random.Random(AUDIT_SEED).sample(pool, min(40, len(pool)))
    out = []
    for lab, i, j in picked:
        resp = fam.records(lab, "judge")[i]["attempts"][-1]["response"]
        out.append({"label": lab, "id": i, "auto_judgement": j, "code": extract_code(resp or "")[-3000:]})
    return out


def p2b(fam: Family, k2_dir: Path, key: str) -> dict:
    out = {"verdict": "只报数字（X-8：K2 未启用）"}
    for grading in ("strict", "lenient"):
        path = k2_dir / f"k2_table_{grading}.json"
        if not path.exists():
            out[grading] = "missing"
            continue
        k2 = {t: v["K2"] for t, v in json.loads(path.read_text())[key]["per_template"].items()}
        part = {}
        for x, y in (("F", "A"), ("WA", "A"), ("WD", "D"), ("D", "A")):
            per_tpl = {}
            for fold in FOLDS:
                tpls = fam.templates(fold, "held")
                seeds = []
                for s in fam.seeds:   # paired within seed, then averaged over seeds
                    ax, ay = fam.accuracy(fam.arm(x, fold, s), "outside"), fam.accuracy(fam.arm(y, fold, s), "outside")
                    seeds.append({t: ax[t] - ay[t] for t in tpls if t in ax and t in ay})
                per_tpl.update(avg(seeds))
            strata = {"K_high": [t for t in per_tpl if t in k2 and k2[t] >= 2 / 3 - 1e-9],
                      "K_low": [t for t in per_tpl if t in k2 and k2[t] <= 1 / 3 + 1e-9]}
            cl = {n: fam.clusters(per_tpl, ts) for n, ts in strata.items()}
            res = {n: {"templates": len(strata[n]), "clusters": len(cl[n]),
                       **ci(list(cl[n].values()), P2B_SEED)} for n in strata}
            # high - low: resample clusters (the union), a draw's strata means from the clusters it picked
            units = sorted(set(cl["K_high"]) | set(cl["K_low"]))
            if cl["K_high"] and cl["K_low"]:
                rng = random.Random(P2B_SEED)
                boots = []
                for _ in range(DRAWS):
                    s = [rng.choice(units) for _ in units]
                    hi, lo = [cl["K_high"][c] for c in s if c in cl["K_high"]], [cl["K_low"][c] for c in s if c in cl["K_low"]]
                    if hi and lo:
                        boots.append(fmean(hi) - fmean(lo))
                boots.sort()
                res["high-low"] = {"mean": fmean(cl["K_high"].values()) - fmean(cl["K_low"].values()),
                                   "ci95": [boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1]], "draws_used": len(boots)}
            res["fewer_than_5_templates"] = any(len(v) < 5 for v in strata.values())
            res["per_template"] = per_tpl
            part[f"{x}-{y}"] = res
        out[grading] = part
    return out


def safety(fam: Family) -> dict:
    def acc(label: str) -> float | None:
        recs = fam.records(label, "test_id")
        g = [float(r["passed"]) for i, r in (recs or {}).items() if i in set(fam.inputs["test_id"]) and is_gradable(r)]
        return fmean(g) if g else None
    out = {}
    for a in ARMS:
        for fold in FOLDS:
            st = acc(fam.start(a, fold))
            for s in fam.seeds:
                e = acc(fam.arm(a, fold, s))
                out[fam.arm(a, fold, s)] = {"test_id": e, "start": st,
                                            "note": None if e is None or st is None or e >= st - .03 else "能力代价"}
    return out


def textbook(fam: Family, labels: list[str]) -> dict:
    parent = fam.groups["parent_of"]
    out = {}
    for lab in labels:
        recs = fam.records(lab, "textbook")
        if recs is None:
            continue
        ok = lambda i: is_gradable(recs.get(i))
        cell = {}
        for g in ("regime_arm", "regime_recognise_pilot"):
            ids = fam.groups[g]
            traps = [float(recs[i]["passed"]) for i in ids if ok(i)]
            pars = [float(recs[p]["passed"]) for p in sorted({parent[i] for i in ids}) if ok(p)]
            cond = [float(recs[i]["passed"]) for i in ids if ok(i) and ok(parent[i]) and recs[parent[i]]["passed"]]
            cell[g] = {"traps": len(ids), "trap_acc": fmean(traps) if traps else None,
                       "parent_acc": fmean(pars) if pars else None,
                       "trap_right_given_parent_right": [int(sum(cond)), len(cond)]}
        out[lab] = cell
    return out


def truncation(fam: Family, labels: list[str]) -> dict:
    """Per label and per arm: share of gradable rows whose graded (last) call used all MAX_TOKENS."""
    # PREREG: "report the truncation share per arm at evaluation" -> the runner keeps no finish_reason; completion_tokens >= max_tokens of the last call
    per = {}
    for lab in labels:
        for kind in ("final", "judge"):
            recs = fam.records(lab, kind)
            if recs is None:
                continue
            g = [r for r in recs.values() if is_gradable(r) and r.get("attempts")]
            n = sum(r["attempts"][-1].get("usage", {}).get("completion_tokens", 0) >= MAX_TOKENS for r in g)
            per.setdefault(lab, {})[kind] = [n, len(g)]
    by_arm = {}
    for a in ARMS:
        cells = [per[fam.arm(a, f, s)]["final"] for f in FOLDS for s in fam.seeds if "final" in per.get(fam.arm(a, f, s), {})]
        n, m = sum(c[0] for c in cells), sum(c[1] for c in cells)
        by_arm[a] = {"truncated": n, "rows": m, "share": n / m if m else None}
    return {"by_arm_final": by_arm, "per_label": per}


# ---- main -------------------------------------------------------------------------------------------------------------

def report(args) -> dict:
    fam = Family(args)
    starts = [fam.base(), *(fam.warm(f) for f in FOLDS)]
    endpoints = [fam.arm(a, f, s) for f in FOLDS for a in ARMS for s in fam.seeds]
    missing = {f: [m for lab in [fam.base(), fam.warm(f)] + [fam.arm(a, f, s) for a in ARMS for s in fam.seeds]
                   for m in fam.missing(lab)] for f in FOLDS}
    need = {i for i, r in fam.rows.items() if r["variant"] == "outside"}
    have = {s["id"] for s in load_jsonl(args.passk) if "error" not in s and s.get("n") == 64} if args.passk.exists() else set()
    if not need <= have:
        for f in FOLDS:
            missing[f].append(f"base pass@64 ({len(need - have)} of {len(need)} outside rows missing in {args.passk})")
    # PREREG: "if any planned run or evaluation is missing, no confirmatory verdict for that family and fold" -> every §6 judgement pools both folds,
    #   so one incomplete fold withholds all of them; the numbers are still computed from what exists.
    complete = not any(missing.values())
    if not args.d3.exists():
        raise SystemExit(f"D-3 report {args.d3} missing: the P2 threshold cannot be chosen")
    d3 = json.loads(args.d3.read_text()).get(args.d3_key)
    if not (isinstance(d3, dict) and isinstance(d3.get("templates"), dict) and "can_detect" in d3):
        raise SystemExit(f"D-3 report {args.d3} has no templates result for {args.d3_key!r}")
    labels = starts + endpoints
    out = {"family": args.name, "seeds": fam.seeds, "complete": complete, "missing": missing,
           "infra_errors": {lab: e for lab in labels if (e := fam.infra_errors(lab))},
           "final_responses_with_regime_key": {lab: n for lab in labels if (n := fam.regime_keys(lab)) is not None},
           "fold_mapping": {f: {"training_fold_templates": fam.templates(f, "train"),
                                "held_out_templates": fam.templates(f, "held")} for f in FOLDS},
           "P1": p1(fam, args.passk, args.passk_sens, complete),
           "P1_knowledge": p1_knowledge(fam, args.passk, getattr(args, "knowledge", None)),
           "warm_x_reward": warm_by_reward(fam),
           "transfer": transfer(fam, complete),
           "P2": p2(fam, d3, complete),
           "P2b": p2b(fam, args.k2_dir, args.k2_key),
           "safety": safety(fam),
           "textbook": textbook(fam, starts + endpoints),
           "truncation": truncation(fam, starts + endpoints),
           "accuracy_by_variant": {lab: {v: summary(fam.clusters(fam.accuracy(lab, v), sorted(fam.cluster)))["mean"]
                                         for v in ("inside", "outside", "given", "flag")}
                                   for lab in starts + endpoints if fam.records(lab, "final") is not None},
           "audit_sample": audit_sample(fam)}
    return out


def pts(x) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def show(r: dict) -> None:
    c = lambda d: f"{pts(d['mean'])} [{pts(d['ci95'][0])}, {pts(d['ci95'][1])}]" if d.get("ci95") else pts(d.get("mean"))
    print(f"== X report: {r['family']}, seeds {r['seeds']}, complete: {r['complete']}")
    errs = {lab: e for lab, e in r["infra_errors"].items() if any(e.values())}
    print(f"infrastructure-error rows (excluded): {errs or 'none'}")
    print(f"final responses with a regime key: { {k: v for k, v in r['final_responses_with_regime_key'].items() if v} or 'none'}")
    for f, m in r["missing"].items():
        if m:
            print(f"  missing {f}: {len(m)} ({', '.join(m[:4])}{' ...' if len(m) > 4 else ''})")
    p = r["P1"]
    print(f"P1: |Z| = {len(p['Z'])} of {p['with_pass64']} mechanisms with pass@64; Z >= 50 % (every seed): "
          f"{ {a: v for a, v in p['Z_at_or_above_50_every_seed'].items() if v} }")
    for a, s in p["spearman_pass64_vs_outside"].items():
        print(f"  Spearman(pass@64, {a}) = {s['rho'] if s['rho'] is None else round(s['rho'], 3)} {s.get('ci95')}")
    print(f"  -> {p['verdict']}")
    pk = r["P1_knowledge"]
    if pk.get("status") == "missing":
        print("P1 knowledge (X-17): missing")
    else:
        rnd = lambda x: "-" if x["rho"] is None else f"{x['rho']:.2f}"
        for name, v in pk["predictors"].items():
            print(f"P1 knowledge (X-17, exploratory) {name} ({v['templates']}): "
                  + "  ".join(f"{a} {rnd(v[a]['all'])} / Z {rnd(v[a]['Z'])}" for a in NO_WARM))
    w = r["warm_x_reward"]
    print("warm x reward (training fold): " + "  ".join(f"{a} {c(w[a])}" for a in ("A", "D", "WA", "WD")))
    print("  " + "  ".join(f"{k} {c(w[k])}" for k in ("D-A", "WD-WA", "WA-A", "WD-D")) + f"  -> {w['reading']}")
    t = r["transfer"]
    for a in ARMS:
        print(f"transfer {a:2s}: held-out {c(t[a]['held_out'])}  - start {c(t[a]['start'])}  = {c(t[a]['endpoint-start'])}  -> {t[a]['verdict']}")
    q = r["P2"]
    print(f"P2: F - A on held-out H - F {c(q['F-A_H-F_held'])} (threshold {q['threshold_points']}; base can detect: "
          f"{q['base_D3']['can_detect']}); both parsed (X-4) {c(q['F-A_H-F_held_both_parsed (X-4)'])}")
    for a in ("A", "F"):
        x = q["rates_held_out"][a]
        print(f"  {a}: parse {pts(x['parse_rate'])}  all-zero {pts(x['all_zero_rate'])}  format failure {pts(x['format_failure_rate'])}  "
              f"held H-F {c(q['by_arm'][a]['held']['H-F'])}")
    print(f"  F - start H - F {c(q['F-start_H-F_held'])}; gain used: {q['checks']['gain_is']}; "
          f"F arm failure (§7): {q['F_arm_failure']['flag'] or 'no'}")
    print(f"  held-out accuracy (no judge prompt) outside {q['held_out_accuracy_no_judge_prompt']['outside']} "
          f"inside {q['held_out_accuracy_no_judge_prompt']['inside']}  -> {q['verdict']}")
    print(f"P2b: {r['P2b']['verdict']}")
    costs = [k for k, v in r["safety"].items() if v["note"]]
    print(f"safety: {len(costs)} endpoints with 能力代价 {costs}")
    print("truncation (final.json): " + "  ".join(f"{a} {pts(v['share'])}" for a, v in r["truncation"]["by_arm_final"].items()))
    print(f"audit sample: {len(r['audit_sample'])} rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="family name as in the chain (qwen3-8b, qwen35-9b)")
    ap.add_argument("--seeds", type=int, nargs="+", help="RL seeds (default: 42 43 for qwen3-8b, else 42)")
    ap.add_argument("--eval-dir", type=Path, default=EXPERIMENTS_DIR / "_x_eval")
    ap.add_argument("--final", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "kc" / "final.json")
    ap.add_argument("--inputs", type=Path, default=EXPERIMENTS_DIR / "_inputs" / "x_eval")
    ap.add_argument("--passk", type=Path, help="base pass@64 samples (default experiments/_x/<name>/passk_final_outside.x18.jsonl, X-18)")
    ap.add_argument("--passk-sens", type=Path, help="§4 sensitivity resample (pass@32, T 0.7), report only")
    ap.add_argument("--d3", type=Path, default=REPO_ROOT / "results" / "d3" / "report.json")
    ap.add_argument("--d3-key"); ap.add_argument("--k2-key")
    ap.add_argument("--k2-dir", type=Path, default=REPO_ROOT / "results" / "x7")
    ap.add_argument("--knowledge", type=Path, default=REPO_ROOT / "results" / "x" / "knowledge.json",
                    help="per-template knowledge measures (explore/x_knowledge.py), X-17 exploratory block")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.seeds = args.seeds or ([42, 43] if args.name == "qwen3-8b" else [42])
    args.passk = args.passk or EXPERIMENTS_DIR / "_x" / args.name / "passk_final_outside.x18.jsonl"   # X-18: the re-sample
    args.d3_key = args.d3_key or D3_KEY.get(args.name, args.name)
    args.k2_key = args.k2_key or K2_KEY.get(args.name, args.name)
    r = report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n")
    show(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
