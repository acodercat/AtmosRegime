"""Emit paper/si_tables.tex: the Supplementary Tables, generated from results/.

  Table 1  the 70 training-study mechanisms with subfield, fold, base pass@64 in three models and whether each
           training route learned them
  Table 3  the 16 diagnostic configurations with the pre-registered D-1 verdict and the benchmark gap
  --frontier  Table 5 (si_table_frontier.tex): the trained 9B against frontier and open models, same harness (§21, §22)

Run from the repository root: ``uv run python paper/fig/make_si_tables.py > paper/si_tables.tex`` and
``... --frontier > paper/si_table_frontier.tex``.
"""

from __future__ import annotations

import glob
import json

from common import CATEGORY_LABEL, CONFIG_LABEL, RESULTS, d_contrasts, load


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def subfield(cat: str) -> str:
    label = CATEGORY_LABEL.get(cat, cat.replace("_", " ").capitalize())
    return "Obs. & modelling" if label == "Observation & modelling" else label


def learned(v) -> str:
    return "--" if v is None else ("yes" if v >= 0.5 else "no")


def table1() -> str:
    rows = load("paper/mechanism_table.json")
    zero_all = set(load("paper/round4_families.json")["S16a"]["zero_in_all"])
    universal = {r["template"] for r in load("paper/mechanism_examples.json")["universal_failures"]}
    out = [
        "{\\footnotesize\\setlength{\\tabcolsep}{2.7pt}\\begin{longtable}{llcrrrcccc}",
        "\\caption{\\textbf{The 70 mechanisms of the training study.} Base pass@64 on the outside version (\\%) for the three base models; fold F1 or F2; learned: trained outside accuracy $\\ge 50\\%$ after answer-reward reinforcement learning (arm A; Qwen3-8B and Qwen3.5-9B), after worked examples and reinforcement learning (Examples, Qwen3-8B, either seed; -- not in the randomised experiment) and after privileged distillation (P, Qwen3-8B); mechanisms marked $\\dagger$ have pass@64 = 0 in all three; mechanisms marked $\\ddagger$ fail (outside accuracy at most 50\\% given the inside version is solved) in every open-model configuration of the diagnostic study for which that measure is defined. Two further diagnostic mechanisms were excluded from training beforehand on physics grounds (Methods).}\\label{stab:mechanisms}\\\\",
        "\\toprule & & & \\multicolumn{3}{c}{Base pass@64 (\\%)} & \\multicolumn{4}{c}{Learned} \\\\ \\cmidrule(lr){4-6}\\cmidrule(l){7-10} Mechanism & Subfield & Fold & \\shortstack{Qwen3\\\\8B} & \\shortstack{Qwen3.5\\\\9B} & \\shortstack{Ministral\\\\3 8B} & \\shortstack{A\\\\8B} & \\shortstack{A\\\\9B} & \\shortstack{Exam-\\\\ples} & P \\\\ \\midrule \\endfirsthead",
        "\\toprule & & & \\multicolumn{3}{c}{Base pass@64 (\\%)} & \\multicolumn{4}{c}{Learned} \\\\ \\cmidrule(lr){4-6}\\cmidrule(l){7-10} Mechanism & Subfield & Fold & \\shortstack{Qwen3\\\\8B} & \\shortstack{Qwen3.5\\\\9B} & \\shortstack{Ministral\\\\3 8B} & \\shortstack{A\\\\8B} & \\shortstack{A\\\\9B} & \\shortstack{Exam-\\\\ples} & P \\\\ \\midrule \\endhead",
        "\\bottomrule \\endfoot",
    ]
    for r in sorted(rows, key=lambda r: (r["category"], r["mechanism"])):
        name = r["mechanism"].replace("_", " ") + (" $\\dagger$" if r["mechanism"] in zero_all else "") + (" $\\ddagger$" if r["mechanism"] in universal else "")
        p = [r.get("pass64_qwen3-8b"), r.get("pass64_qwen35-9b"), r.get("pass64_ministral3-8b")]
        cells = ["--" if v is None else f"{100 * v:.0f}" for v in p]
        dose = r.get("dose_qwen3-8b")
        ex = "--" if not dose else ("yes" if any(v[0] > 0 and v[1] >= 0.5 for v in dose.values()) else "no")
        outcome = [learned(r.get("A_qwen3-8b")), learned(r.get("A_qwen35-9b")), ex, learned(r.get("P"))]
        out.append(f"{name} & {esc(subfield(r['category']))} & {r['fold']} & " + " & ".join(cells + outcome) + " \\\\")
    out.append("\\end{longtable}}")
    return "\n".join(out)


def table3() -> str:
    verdicts = {}
    for path in sorted(glob.glob(str(RESULTS / "d2" / "families_report_*.json"))):
        with open(path, encoding="utf-8") as fh:
            for cfg in json.load(fh)["configurations"]:
                if cfg["status"] == "complete":
                    verdicts[cfg["key"]] = (cfg["D1"], cfg["templates"]["all"]["contrasts"]["inside_minus_outside"], cfg["textbook"]["regime_minus_geometry"])
    for cfg in load("d_execution/completed_configurations.json")["configurations"]:
        verdicts[cfg["configuration"]] = (cfg["D1"], cfg["templates_all"]["contrasts"]["inside_minus_outside"], cfg["textbook_regime_minus_geometry"])
    io69 = d_contrasts()
    verdicts = {k: (v[0], io69[k]["inside_minus_outside"], v[2], v[1]) for k, v in verdicts.items()}  # v[1]: all 72 templates
    label = {"deficit_established": "deficit established", "threshold_not_met": "threshold not met", "not_eligible": "not eligible"}
    out = [
        "\\begin{table}[H]\\centering\\small",
        "\\caption{\\textbf{Pre-registered diagnostic verdict (D-1) for each configuration.} The rule required inside $-$ outside $\\ge 20$ points on the templates (32 clusters) and a paired regime $-$ geometry gap $\\ge 15$ points on the benchmark traps (38 regime traps against the geometry traps), both intervals excluding zero; `not eligible' means a gating condition (parent accuracy or cluster count) was not met. Inside $-$ outside is given over the 69 templates whose wording matches the training study (main text) and over all 72 (the pre-registered set); the verdicts are the same on both. The template part holds for 15 of 16 configurations (Gemma~4 31B with thinking falls short); the benchmark part, with 38 traps, for one. Pooled over configurations the benchmark gap is 11.1 [$-$0.4, 22.8] points, and on an independent set of 54 traps 12.7 [4.4, 21.3] (pre-registered replication).}\\label{stab:d1}",
        "\\begin{tabular}{lrrrl}\\toprule Configuration & In $-$ out, 69 & In $-$ out, 72 & Regime $-$ geometry & Verdict \\\\ \\midrule",
    ]
    for key in sorted(verdicts, key=lambda k: -verdicts[k][1]["mean"]):
        d1, io, tb, io72 = verdicts[key]
        out.append(f"{CONFIG_LABEL.get(key, esc(key))} & {100 * io['mean']:.1f} [{100 * io['ci95'][0]:.1f}, {100 * io['ci95'][1]:.1f}] & {100 * io72['mean']:.1f} [{100 * io72['ci95'][0]:.1f}, {100 * io72['ci95'][1]:.1f}] & {100 * tb['mean']:.1f} [{100 * tb['ci95'][0]:.1f}, {100 * tb['ci95'][1]:.1f}] & {label.get(d1, esc(str(d1)))} \\\\")
    out += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(out)


FRONTIER_LABEL = {
    "frontier21-deepseek-v4-flash-off": ("DeepSeek V4 Flash", "off"),
    "frontier21-deepseek-v4-flash-on": ("DeepSeek V4 Flash", "on"),
    "frontier21-gpt-5.5-off": ("GPT-5.5", "off"),
    "frontier21-gpt-5.5-on": ("GPT-5.5", "on"),
    "frontier21-gpt-5.4-off": ("GPT-5.4", "off"),
    "frontier21-minimax-m3-off": ("MiniMax-M3", "off"),
}
OPEN_ORDER = ["gemma4-31b", "granite4.2-30b", "qwen3.5-27b", "ministral3-14b", "qwen3.5-9b", "qwen3-8b", "granite4.2-8b", "ministral3-8b", "gemma4-e4b"]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _diff(d: dict) -> str:
    return f"{100 * d['mean']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]".replace("-", "$-$")


def _independent() -> str:
    arms = load("paper/f23_e1.json")["arms"]
    a, r = arms["qwen3-8b:A"]["all"], arms["qwen3-8b:R"]["all"]
    g = lambda d: _diff(d)
    return (f"On an independent benchmark set ({a['S23a_regime_gain']['units']} regime traps, {a['geometry_gain']['units']} geometry traps, "
            f"{a['parent_gain']['units']} parent problems), template-trained Qwen3-8B (arm A, one fold and seed) gains {g(a['S23a_regime_gain'])} points on the regime traps, "
            f"{g(a['geometry_gain'])} on the geometry traps and {g(a['parent_gain'])} on the parent problems over its base model; "
            f"with a random reward the changes are {g(r['S23a_regime_gain'])}, {g(r['geometry_gain'])} and {g(r['parent_gain'])}. ")


def table_frontier() -> str:
    """Supplementary Table 4: the trained Qwen3.5-9B (arm A, thinking off) against frontier models (§21, 46 traps)
    and the diagnostic study's open configurations (§22, the 38 of those traps in both benchmark sets), same harness."""
    f21 = load("paper/frontier21.json")["configurations"]
    f22 = load("paper/frontier22.json")
    ds = f21["frontier21-deepseek-v4-flash-off"]
    a46 = ds["textbook"]["vs"]["qwen35-9b-A"]
    a38 = f22["arms_on_shared"]["qwen35-9b-A"]
    held = ds["templates"]["vs_9b_heldout"]["arm_outside"]
    out = [
        "\\begin{table}[H]\\centering\\small\\setlength{\\tabcolsep}{3pt}",
        "\\caption{\\textbf{The trained Qwen3.5-9B against frontier and open models on the benchmark regime traps, same harness.} "
        "Accuracy (\\%) on the regime traps and on their parent problems, and the paired differences $\\Delta$ = trained model $-$ comparison model "
        "(95\\% interval, bootstrap over traps). Trained model: Qwen3.5-9B, arm A, thinking off, mean of its four endpoints. Upper block: frontier models run "
        "through their APIs in the training study's evaluation harness on all \\frontierN{} traps (three repetitions); for the thinking-off runs the last two columns give "
        "their outside accuracy on the training-study templates (32 clusters), to be read against the trained model's held-out-fold outside accuracy of "
        f"{_pct(held['mean'])}\\% [{_pct(held['ci95'][0])}, {_pct(held['ci95'][1])}], and their inside $-$ outside gap on the same templates. "
        "Lower block: the diagnostic study's local runs in the same harness, which cover \\sharedN{} of these traps (three repetitions); the trained model is scored on the same \\sharedN{}. "
        "The trained model was trained on templates of the tested mechanisms; the comparison models were not. "
        "Qwen3.5-27B, an open model, is listed in the upper block because it was also run on the remaining traps. "
        + _independent() +
        "}\\label{stab:frontier}",
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrrr}\\toprule Model & Thinking & Trap acc. & Parent acc. & $\\Delta$ traps & $\\Delta$ parents & Templates outside & Template gap \\\\ \\midrule",
        f"\\multicolumn{{8}}{{l}}{{\\emph{{All {ds['textbook']['traps_covered']} regime traps}}}} \\\\",
        f"Qwen3.5-9B, trained (arm A) & off & {_pct(a46['arm_trap'])} & {_pct(ds['textbook']['parent']['mean'] + a46['parent']['mean'])} & -- & -- & {_pct(held['mean'])} & -- \\\\",
    ]
    for key, (name, think) in FRONTIER_LABEL.items():
        tb = f21[key]["textbook"]
        if f21[key]["templates"]["present"]:
            tmpl = f21[key]["templates"]
            outside, gap = _pct(tmpl["outside"]["mean"]), f"{100 * tmpl['inside_minus_outside']['mean']:.1f}"
        else:
            outside = gap = "--"
        out.append(f"{name} & {think} & {_pct(tb['trap']['mean'])} & {_pct(tb['parent']['mean'])} & {_diff(tb['vs']['qwen35-9b-A']['trap'])} & {_diff(tb['vs']['qwen35-9b-A']['parent'])} & {outside} & {gap} \\\\")
    e2 = load("paper/f23_e2.json")["configurations"]
    for think in ("off", "on"):
        tb = e2[f"qwen3.5-27b.think-{think}"]["textbook"]
        tm = e2[f"qwen3.5-27b.think-{think}"].get("templates")
        outside, gap = (_pct(tm["outside"]["mean"]), f"{100 * (tm['inside']['mean'] - tm['outside']['mean']):.1f}") if tm else ("--", "--")
        out.append(f"Qwen3.5-27B & {think} & {_pct(tb['trap']['mean'])} & {_pct(tb['parent']['mean'])} & {_diff(tb['vs']['qwen35-9b-A']['trap'])} & {_diff(tb['vs']['qwen35-9b-A']['parent'])} & {outside} & {gap} \\\\")
    out.append("\\midrule")
    out.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{The {f22['shared_traps']} traps shared with the diagnostic study}}}} \\\\")
    out.append(f"Qwen3.5-9B, trained (arm A) & off & {_pct(a38['trap'])} & {_pct(a38['parent'])} & -- & -- & -- & -- \\\\")
    for fam in OPEN_ORDER:
        for think in ("off", "on"):
            key = f"{fam}.think-{think}"
            if key not in f22["configurations"] or e2.get(key, {}).get("present"):
                continue
            tb = f22["configurations"][key]["textbook"]
            name = CONFIG_LABEL[f"{fam}.think-off"]
            out.append(f"{name} & {think} & {_pct(tb['trap']['mean'])} & {_pct(tb['parent']['mean'])} & {_diff(tb['vs']['qwen35-9b-A']['trap'])} & {_diff(tb['vs']['qwen35-9b-A']['parent'])} & -- & -- \\\\")
    out += ["\\bottomrule\\end{tabular}}\\end{table}"]
    return "\n".join(out)


def main() -> None:
    import sys

    print("% generated by paper/fig/make_si_tables.py from results/ — do not edit")
    if "--frontier" in sys.argv:
        print(table_frontier())
        return
    if "--d1" in sys.argv:
        print(table3())
        return
    print(table1())


if __name__ == "__main__":
    main()
