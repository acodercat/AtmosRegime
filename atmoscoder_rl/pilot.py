"""The training pilot (docs/PILOT_DESIGN.md): held-out split, template rows, judgement reward.

    uv run python -m atmoscoder_rl.pilot split [--out results/pilot/split.json]   # draw the held-out units

Three arms differ only in data and reward: C0 (the existing run), A (C0's data plus template rows,
answer reward) and B (A's data; on inside / outside template rows the problem also asks whether the
standard treatment is valid, and half the reward is for that judgement). Everything a later step
needs to reproduce the arms is fixed here: the units, the draw, the seeds, the instruction.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .build_templates import build
from .engine import SOLVER_TIMEOUT, extract_code
from .grading import TOLERANCE, verify
from .paths import EXPERIMENTS_DIR, REPO_ROOT
from .prompts import chat
from .reward import DEFAULT_UNIT_RULE, RewardResult, grade
from .runs import is_gradable
from .sandbox import run_solver_sandboxed
from .screening import CLUSTER_OF, CLUSTERS
from .templates import TEMPLATES

TRAIN_SEED = 20260930          # template rows the arms train on
EVAL_SEED = 20260931           # held-out template rows the arms are tested on
PAIRS = 20                     # matched pairs per template, both seeds
N_HELD_OUT = 6                 # of the 16 mechanism units
MIN_INFORMATIVE = 3            # held-out templates that can show recognition at all (below)
SPLIT_PATH = REPO_ROOT / "results" / "pilot" / "split.json"

#: The base run the split rule reads: Qwen3-8B, thinking off, on all 30 templates (seed 20260928,
#: docs/PREREG_TEMPLATE_RL.md). Existing data; the pilot's own items are drawn from new seeds.
BASE_RUN = ("prereg_tpl_rl", "qwen3-8b.think-off.run1.json")
#: A held-out template is *informative* when the base solves its inside version (>= 10 / 20, else
#: recognition has almost no denominator) and fails its outside version (<= 5 / 20, else there is no
#: deficit to remove). The primary measure is computed over the informative held-out templates.
INSIDE_MIN, OUTSIDE_MAX = 10, 5

JUDGE_KEY = "regime"
#: Appended to the problem text of B's inside / outside template rows. It names no quantity and no
#: threshold, and reads the same on both sides, so the text alone does not give the label away.
JUDGE_INSTRUCTION = (
    'In addition to the numbered entries, the dict returned by solve() must contain one more key, '
    '"regime" (the only exception to rule 2): 1 if the standard textbook treatment of this problem is '
    'valid for the values given, 0 if the values fall outside the range where it is valid.')
JUDGE_LABEL = {"inside": 1, "outside": 0}
#: The flag-only control (TODO.md, 2026-09-29): says that the standard treatment fails here and nothing
#: else. Against the standard text it measures "not noticing"; against the given text (which states the
#: replacement) it measures "not recalling the replacement".
FLAG_INSTRUCTION = ("Note: for the values given here, the standard textbook treatment of this problem is not "
                    "valid. The problem does not say what to use instead.")   # `given` rows carry no judgement: the text states the change
ANSWER_WEIGHT = 0.5


def units() -> list[str]:
    """The 16 mechanism units: each cluster, and each template in none."""
    return sorted(CLUSTERS) + sorted(t for t in TEMPLATES if t not in CLUSTER_OF)


def members(unit: str) -> tuple[str, ...]:
    return CLUSTERS.get(unit, (unit,))


def base_counts(experiments: Path = EXPERIMENTS_DIR) -> dict[str, dict[str, list[int]]]:
    """Per template and variant: [right, gradable] in the base run."""
    path = experiments / BASE_RUN[0] / BASE_RUN[1]
    counts: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for rec in json.loads(path.read_text())["results"]:
        if not is_gradable(rec):
            continue
        *_, variant = rec["id"].split("_")
        name = rec["id"].split("_", 2)[2].rsplit("_", 2)[0]      # tpl_<seed>_<name>_<nn>_<variant>
        cell = counts[name][variant]
        cell[0] += bool(rec["passed"]); cell[1] += 1
    return counts


def informative(counts: dict[str, dict[str, list[int]]], name: str) -> bool:
    return counts[name]["inside"][0] >= INSIDE_MIN and counts[name]["outside"][0] <= OUTSIDE_MAX


def draw_split(seed: int = TRAIN_SEED, experiments: Path = EXPERIMENTS_DIR) -> dict[str, Any]:
    """Draw N_HELD_OUT units until at least MIN_INFORMATIVE held-out templates are informative."""
    counts = base_counts(experiments)
    rng = random.Random(f"pilot-split:{seed}")
    pool = units()
    for draw in range(1, 1001):
        held = sorted(rng.sample(pool, N_HELD_OUT))
        held_templates = [t for u in held for t in members(u)]
        info = [t for t in held_templates if informative(counts, t)]
        if len(info) >= MIN_INFORMATIVE:
            break
    else:
        raise RuntimeError("no admissible split in 1000 draws")
    path = experiments / BASE_RUN[0] / BASE_RUN[1]
    return {
        "seed": seed, "draw": draw, "units": pool,
        "held_out_units": held, "trained_units": [u for u in pool if u not in held],
        "held_out_templates": held_templates,
        "trained_templates": [t for u in pool if u not in held for t in members(u)],
        "informative_held_out": info,
        "rule": f"held-out units drawn with random.Random('pilot-split:{seed}').sample(units, {N_HELD_OUT}), "
                f"redrawn until >= {MIN_INFORMATIVE} held-out templates have base inside >= {INSIDE_MIN}/20 "
                f"and base outside <= {OUTSIDE_MAX}/20",
        "base_run": "/".join(BASE_RUN), "base_run_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "base_counts": {t: {v: counts[t][v] for v in ("inside", "outside", "given")} for t in TEMPLATES},
    }


def load_split(path: Path = SPLIT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def template_items(names: Sequence[str], seed: int) -> list[dict[str, Any]]:
    return build(PAIRS, seed, set(names))


def judge_text(problem: str) -> str:
    return f"{problem}\n\n{JUDGE_INSTRUCTION}"


# ---- reward ---------------------------------------------------------------------------------------

def parse_judgement(entry: Any) -> int | None:
    """0 or 1 from the returned entry (bare, or in the {"value": ...} shape of rule 2); else None."""
    if isinstance(entry, dict):
        entry = entry.get("value")
    if isinstance(entry, bool):
        return int(entry)
    try:
        value = float(entry)
    except (TypeError, ValueError):
        return None
    return int(value) if value in (0.0, 1.0) else None


#: Every row returns the same keys, so verl can stack them into one batch.
def _metrics(answer: float, labelled: bool, judgement: int | None, label: int | None,
             cf: tuple[bool, bool, bool] = (False, False, False)) -> dict[str, float]:
    """`cf` = (the row has a twin, the trigger parameter was found, the solver is right at the twin)."""
    return {"answer": answer, "judge/labelled": float(labelled),
            "judge/missing": float(labelled and judgement is None),
            "judge/correct": float(labelled and judgement is not None and judgement == label),
            "cf/twinned": float(cf[0]), "cf/mapped": float(cf[1]), "cf/right": float(cf[2])}


def grade_regime(completion: str, sub_answers: Sequence[dict[str, Any]], label: int | None, *,
                 tolerance: float = TOLERANCE, timeout: float = SOLVER_TIMEOUT,
                 answer_weight: float = ANSWER_WEIGHT) -> dict[str, Any]:
    """Score one rollout. Without a label this is `reward.grade` exactly (the answer reward).

    With a label (0 / 1), the solver runs once; the "regime" entry is removed before the answer is
    graded (so it can never be matched positionally against a numbered sub), and the score is
    answer_weight x answer + (1 - answer_weight) x (judgement == label) — 0.5 in the pilot's arm B,
    0.8 in PREREG_X's arm F (answer + 0.25 x judgement up to scale). A missing or malformed
    judgement scores as wrong. Never raises, like `grade`.
    """
    if label is None:
        result = grade(completion, sub_answers)
        return {"score": result.reward, **result.trainer_metrics(),
                **_metrics(result.reward, False, None, None)}
    keys = RewardResult(0.0, False, "no_code").trainer_metrics()   # the same keys `grade` reports
    zero = {"score": 0.0, **{k: 0.0 for k in keys}, **_metrics(0.0, True, None, label)}
    code = extract_code(completion or "")
    if not code.strip():
        return zero | {"outcome/no_code": 1.0}
    try:
        ok, results, _ = run_solver_sandboxed(code, timeout=timeout)
    except Exception:  # noqa: BLE001 — must not escape into the trainer
        ok, results = False, {}
    if not ok:
        return zero | {"outcome/exec_error": 1.0}
    judgement = parse_judgement(results.pop(JUDGE_KEY, None))
    if not results:
        answer, outcome, n_subs, n_passed = 0.0, "empty_result", 0, 0
    else:
        passed, details = verify(results, list(sub_answers), tolerance, DEFAULT_UNIT_RULE)
        answer = 1.0 if passed else 0.0
        outcome = "passed" if passed else "wrong_answer"
        n_subs, n_passed = len(details), sum(1 for d in details if d.get("passed"))
    correct = judgement is not None and judgement == label
    return {"score": answer_weight * answer + (1 - answer_weight) * float(correct),
            **{k: 0.0 for k in keys}, "n_subs": n_subs, "n_passed": n_passed, f"outcome/{outcome}": 1.0,
            **_metrics(answer, True, judgement, label)}


def encode_truth(sub_answers: list[dict[str, Any]], label: int | None = None,
                 twin: dict[str, Any] | None = None, answer_weight: float | None = None) -> str:
    """verl's `ground_truth` string: the plain answer list, or an object carrying a judgement label or
    a twin (`twin_for`) for the counterfactual reward; `answer_weight` travels with a label when it is
    not the pilot's 0.5 (PREREG_X arm F: 0.8)."""
    if label is None and twin is None:
        return json.dumps(sub_answers)
    out: dict[str, Any] = {"sub_answers": sub_answers}
    if label is not None:
        out[JUDGE_KEY] = label
        if answer_weight is not None:
            out["answer_weight"] = answer_weight
    if twin is not None:
        out["twin"] = twin
    return json.dumps(out)


def decode_truth(ground_truth: str) -> tuple[list[dict[str, Any]], int | None, dict[str, Any] | None]:
    truth = json.loads(ground_truth)
    if isinstance(truth, dict):
        label = truth.get(JUDGE_KEY)
        return truth["sub_answers"], (int(label) if label is not None else None), truth.get("twin")
    return truth, None, None


def answer_weight_of(ground_truth: str) -> float:
    truth = json.loads(ground_truth)
    return float(truth.get("answer_weight", ANSWER_WEIGHT)) if isinstance(truth, dict) else ANSWER_WEIGHT


# ---- counterfactual execution (docs/REWARD_V2.md) -------------------------------------------------
#: A template item and its twin differ in one given (the trigger). A rollout's solve() takes every given
#: as a defaulted parameter, so the same program can be re-run with the trigger moved to the twin's value.
TWIN_OF = {"inside": "outside", "outside": "inside", "given": "inside"}
CF_PARTIAL = 0.5          # right on its own problem, wrong (or unmappable) at the twin's value


def twin_for(item: dict[str, Any], twin: dict[str, Any]) -> dict[str, Any]:
    """What the reward needs about the twin: its answers and the trigger's value on both sides."""
    diff = [k for k in item["params"] if item["params"][k] != twin["params"].get(k)]
    if len(diff) != 1:
        raise ValueError(f"{item['id']}: twin differs in {diff}, expected exactly one given")
    return {"sub_answers": twin["sub_answers"], "trigger": [item["params"][diff[0]], twin["params"][diff[0]]]}


def solver_defaults(code: str) -> dict[str, float]:
    """solve()'s numeric parameter defaults, by name."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            pos = node.args.args[len(node.args.args) - len(node.args.defaults):]
            pairs = list(zip(pos, node.args.defaults)) + [
                (a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None]
            out = {}
            for a, d in pairs:
                try:
                    v = ast.literal_eval(d)
                except (ValueError, SyntaxError, TypeError):
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    out[a.arg] = float(v)
            return out
    return {}


def trigger_parameter(code: str, own: float, other: float) -> tuple[str, float] | None:
    """(name, new default) of the one parameter holding the trigger, or None when none or several do.
    A default matches the stated value up to a power of ten (the solver's unit) or a 273.15 offset."""
    hits = []
    for name, d in solver_defaults(code).items():
        if d != 0 and own != 0 and d / own > 0:
            k = math.log10(d / own)
            if abs(k - round(k)) < 1e-6:
                hits.append((name, d * other / own))
                continue
        if abs(d - (own + 273.15)) < 1e-6 or abs(d - (own - 273.15)) < 1e-6:
            hits.append((name, d + other - own))
    return hits[0] if len(hits) == 1 else None


def with_trigger(code: str, own: float, other: float) -> str | None:
    """The solver, called with its trigger parameter at the twin's value; None if unmapped."""
    hit = trigger_parameter(code, own, other)
    if hit is None:
        return None
    name, value = hit
    return f"{code}\n\n_solve_as_written = solve\ndef solve():\n    return _solve_as_written({name}={value!r})\n"


def grade_cf(completion: str, sub_answers: Sequence[dict[str, Any]], twin: dict[str, Any], *,
             tolerance: float = TOLERANCE, timeout: float = SOLVER_TIMEOUT) -> dict[str, Any]:
    """1 if the solver is right on its own problem and, re-run with the trigger at the twin's value, right
    on the twin too; CF_PARTIAL if right on its own problem only (or its trigger parameter is not found);
    0 otherwise. The own-problem grading is `reward.grade` exactly. Never raises."""
    result = grade(completion, sub_answers, tolerance=tolerance, timeout=timeout)
    base = {**result.trainer_metrics()}
    if not result.passed:
        return {"score": 0.0, **base, **_metrics(0.0, False, None, None, (True, False, False))}
    own, other = twin["trigger"]
    code = with_trigger(extract_code(completion or ""), own, other)
    right = False
    if code is not None:
        try:
            ok, results, _ = run_solver_sandboxed(code, timeout=timeout)
        except Exception:  # noqa: BLE001 — must not escape into the trainer
            ok, results = False, {}
        if ok and results:
            right, _ = verify(results, list(twin["sub_answers"]), tolerance, DEFAULT_UNIT_RULE)
    return {"score": 1.0 if right else CF_PARTIAL, **base,
            **_metrics(1.0, False, None, None, (True, code is not None, bool(right)))}


# ---- warm start from the given rows (docs/REWARD_V2.md §3.2) ---------------------------------------

def strip_code(code: str) -> str | None:
    """The program without comments or docstrings (a comment such as "as the problem states" would teach,
    on a prompt that states nothing, a claim the prompt does not make). None if it does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def set_default(code: str, name: str, value: float) -> str:
    """The program with solve()'s parameter `name` defaulting to `value`."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            pos = node.args.args[len(node.args.args) - len(node.args.defaults):]
            for i, a in enumerate(pos):
                if a.arg == name:
                    node.args.defaults[i] = ast.Constant(float(f"{value:.12g}"))
            for i, a in enumerate(node.args.kwonlyargs):
                if a.arg == name and node.args.kw_defaults[i] is not None:
                    node.args.kw_defaults[i] = ast.Constant(float(f"{value:.12g}"))
    return ast.unparse(tree)


#: A comment that refers to the hint ("using the given formula") would, on a prompt that gives none,
#: teach the model to claim something the prompt does not say. Such programs are dropped, not edited.
HINT_REFERENCE = re.compile(
    r"(?i)(the problem (states|says|tells|gives|provides|specifies)|as (stated|given|specified|provided) in the problem"
    r"|we are told|provided formula|formula provided|given formula|formula given|as instructed|the hint"
    r"|the statement says|problem statement)")


def set_default_text(code: str, name: str, value: float) -> str | None:
    """`set_default` on the source text, so comments survive: the default literal of `name` is spliced
    out and replaced. None if the default spans lines or is not found."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            pos = node.args.args[len(node.args.args) - len(node.args.defaults):]
            pairs = list(zip(pos, node.args.defaults)) + [
                (a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None]
            for a, d in pairs:
                if a.arg == name and d.lineno == d.end_lineno:
                    lines = code.split("\n")
                    line = lines[d.lineno - 1].encode()
                    new = repr(float(f"{value:.12g}")).encode()
                    lines[d.lineno - 1] = (line[:d.col_offset] + new + line[d.end_col_offset:]).decode()
                    return "\n".join(lines)
    return None


def fence(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


def warmstart_rows(samples: Path, seed: int = TRAIN_SEED,
                   strip_comments: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """SFT rows from the base model's rollouts on the given rows (explore.pilot_feasibility --robust G).

    For each pair, the first rollout whose program is right on its own problem and at the inside value
    (counterfactual execution) becomes two rows, both re-verified: the *outside* problem's standard text
    with that program, and the *inside* problem's text with the program's trigger default moved to the
    inside value. The prompts carry no hint. Programs whose text refers to the hint (HINT_REFERENCE) are
    dropped. Comments are kept by default: with thinking off they are where the model reasons, and a
    warm start on comment-free programs (`strip_comments`, the first two warm starts) lowered held-out
    accuracy."""
    split = load_split()
    items = {i["id"]: i for i in template_items(split["trained_templates"], seed)}
    rows, counts = [], defaultdict(int)
    for line in samples.open():
        rec = json.loads(line)
        if rec.get("arm") != "G":
            continue
        given = items[rec["id"]]
        stem = rec["id"].rsplit("_", 1)[0]
        inside, outside = items[stem + "_inside"], items[stem + "_outside"]
        own, other = twin_for(given, inside)["trigger"]
        for completion in rec["completions"]:
            code = extract_code(completion or "")
            code = strip_code(code) if strip_comments else code
            if code is None or HINT_REFERENCE.search(code) \
                    or grade_cf(fence(code), given["sub_answers"], twin_for(given, inside))["score"] != 1.0:
                continue
            hit = trigger_parameter(code, own, other)
            if not hit:
                continue
            inside_code = (set_default if strip_comments else set_default_text)(code, hit[0], hit[1])
            if inside_code is None or not grade(fence(code), outside["sub_answers"]).passed \
                    or not grade(fence(inside_code), inside["sub_answers"]).passed:
                continue
            for item, prog in ((outside, code), (inside, inside_code)):
                rows.append({"messages": chat(item["problem"]) + [{"role": "assistant", "content": fence(prog)}],
                             "id": item["id"], "kind": "warmstart", "template": item["template"]})
            counts[given["template"]] += 1
            break
    return rows, dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="draw the held-out mechanism units")
    s.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    s.add_argument("--out", type=Path, default=SPLIT_PATH)
    w = sub.add_parser("warmstart", help="SFT rows from verified given-row rollouts, plus RAFT rows")
    w.add_argument("--samples", type=Path, required=True, help="pilot_feasibility --robust G output")
    w.add_argument("--raft", type=Path, required=True, help="sft_raft.parquet: the base's own verified solutions")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--raft-ratio", type=float, default=1.0, help="RAFT rows per template row")
    w.add_argument("--strip-comments", action="store_true", help="drop comments and docstrings (warm starts 1-2)")
    args = parser.parse_args()
    if args.cmd == "warmstart":
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows, counts = warmstart_rows(args.samples, strip_comments=args.strip_comments)
        raft = pq.read_table(args.raft).to_pylist()
        # As many of the base's own verified solutions to standard problems as template rows, so the
        # warm start does not tilt the policy towards templates alone.
        extra = random.Random(TRAIN_SEED).sample(raft, min(int(len(rows) * args.raft_ratio), len(raft)))
        extra = [{"messages": r["messages"], "id": r.get("id", ""), "kind": "raft", "template": ""} for r in extra]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows + extra), args.out)
        print(f"pairs per template: {counts}")
        print(f"wrote {len(rows)} template rows + {len(extra)} RAFT rows -> {args.out}")
        return 0
    split = draw_split(experiments=args.experiments)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=1) + "\n")
    c = split["base_counts"]
    print(f"draw {split['draw']}: held out {split['held_out_units']}")
    for t in split["held_out_templates"]:
        print(f"  {t:28s} base inside {c[t]['inside'][0]}/{c[t]['inside'][1]}  outside {c[t]['outside'][0]}/"
              f"{c[t]['outside'][1]}  {'informative' if t in split['informative_held_out'] else ''}")
    print(f"trained: {len(split['trained_templates'])} templates in {len(split['trained_units'])} units -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
