"""Fig. 7 — Learned mechanisms gain inside and outside the regime but not regime judgement; errors converge.

a  On mechanisms each intervention learned: change in accuracy on the inside version (the
   standard-formula branch). All learned cells are "conditional" (inside did not fall by more
   than 25 points); none is a plain swap of formula.
b  Same mechanisms: change in regime judgement, H − F (share of outside rows judged "does not
   hold" minus share of inside rows wrongly judged so).
c  Among wrong outside answers, the share equal to the standard-formula value: base model, after
   answer-reward RL (A) and after privileged distillation (P) and its controls (formerly Fig. 5c).
d  Same quantity on the held-out fold: change from base after answer-reward RL and after
   random-reward RL (formerly Fig. 5d). The benchmark-trap panel formerly here is Supplementary Fig. 6.

Data: results/paper/round4_conditional.json (explore.paper_round4 §15), results/paper/synthesis.json
(explore.paper_synthesis §13), results/paper/rigor_eval_qwen3-8b.json (explore.paper_rigor_eval §5).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C, DOUBLE, ci_err, load, label_panels, legend_above, space_panels, pct, save, style

LEVELS = [("base", "Base\nmodel"), ("A", "RL\nonly"), ("P", "Teacher\ndistillation")]
ARMS = [
    ("qwen3-8b", "A", "Qwen3-8B, RL only", C["rl"]),
    ("qwen35-9b", "A", "Qwen3.5-9B, RL only", C["rl2"]),
    ("qwen3-8b", "DOSE", "Qwen3-8B, SFT first, then RL", C["dose"]),
    ("ministral3-8b", "DOSE", "Ministral 3 8B, SFT first, then RL", C["dose2"]),
    ("qwen3-8b", "P", "Teacher distillation", C["distil"]),
]


def main() -> None:
    style()
    r4c = load("paper/round4_conditional.json")["families"]

    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.4, width_ratios=[1.2, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --- a, b: Δinside and Δ(H−F) on learned mechanisms ---------------------------------------
    y = np.arange(len(ARMS))[::-1]
    for ax, key, xlabel in [(ax_a, "d_inside", "Δ inside accuracy on learned\nmechanisms (percentage points)"), (ax_b, "d_hf", "Δ regime discrimination (H − F) on learned\nmechanisms (percentage points)")]:
        for yi, (fam, arm, label, colour) in zip(y, ARMS):
            L = r4c[fam]["interventions"][arm]["learned"]
            v = L[key]
            m = pct(v["mean"])
            ax.errorbar(m, yi, xerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt="o", color=colour, ms=3.5, elinewidth=0.7)
        ax.axvline(0, color="#999999", lw=0.6, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{l} (n = {r4c[f]['interventions'][a]['learned']['n']})" if ax is ax_a else l for f, a, l, _ in ARMS], fontsize=6)
        ax.set_xlabel(xlabel)
    ax_a.set_xlim(-5, 80)
    ax_b.set_xlim(-35, 25)
    ax_b.set_yticklabels([])  # same rows, same order as a
    ax_b.tick_params(axis="y", length=0)

    # --- c: shortcut share among wrong answers --------------------------------------------------------
    lv = load("paper/synthesis.json")["S1_error_anatomy"]["levels"]
    colours = {"base": C["base"], "A": C["rl"], "P": C["distil"], "Pk2": C["distil_bad"], "S": C["distil_bad"], "Pshuf": C["distil_bad"]}
    for i, (key, label) in enumerate(LEVELS):
        v = lv[key]["shortcut"]
        m = pct(v["mean"])
        ax_c.bar(i, m, color=colours[key], width=0.65)
        ax_c.errorbar(i, m, yerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt="none", ecolor="#333333", elinewidth=0.6)
        ax_c.text(i, 2, f"n={v['units']}", ha="center", fontsize=5, color="#222222" if colours[key] == C["distil_bad"] else "white")
    ax_c.set_xticks(range(len(LEVELS)))
    ax_c.set_xticklabels([l for _, l in LEVELS], fontsize=6)
    ax_c.set_xlim(-0.8, len(LEVELS) - 0.2)
    ax_c.set_ylabel("Wrong outside answers at the\nstandard-formula value (%)")
    ax_c.set_ylim(0, 100)
    ax_c.set_title("Qwen3-8B, trained mechanisms", fontsize=6.5, loc="left")

    # --- d: held-out fold, A vs R ---------------------------------------------------------------------
    rg = load("paper/rigor_eval_qwen3-8b.json")["shortcut_share_wrong_only_0.02"]
    base = rg["base:held+train:level"]
    # Changes relative to the base model, each with its own paired-resample interval (a level
    # interval cannot be formed by adding the base level to the interval of a change).
    items = [("A:held", "RL only", C["rl"]), ("R:held", "Random\nreward", C["random"])]
    for i, (key, label, colour) in enumerate(items):
        m, ci = pct(rg[key]["mean"]), [pct(c) for c in rg[key]["ci95"]]
        ax_d.bar(i, m, color=colour, width=0.6)
        ax_d.errorbar(i, m, yerr=ci_err(m, ci), fmt="none", ecolor="#333333", elinewidth=0.6)
    ax_d.axhline(0, color="#999999", lw=0.6)
    ax_d.set_xticks(range(2))
    ax_d.set_xticklabels([l for _, l, _ in items], fontsize=6)
    ax_d.set_xlim(-0.7, 1.7)
    ax_d.set_ylabel("Change in that share\n(percentage points)")
    ax_d.set_ylim(-20, 40)
    ax_d.text(0.98, 0.97, f"base level {pct(base['mean']):.0f} %", transform=ax_d.transAxes, ha="right", va="top", fontsize=6)
    ax_d.set_title("Qwen3-8B, held-out mechanisms", fontsize=6.5, loc="left")

    space_panels(fig)
    # Without its own row labels b would start far right of a; widen it leftwards to an 18-pt gap.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _gap = (ax_b.get_tightbbox(_r).x0 - ax_a.get_tightbbox(_r).x1) / fig.bbox.width - 18 / 72 / fig.get_size_inches()[0]
    if _gap > 0:
        _p = ax_b.get_position()
        ax_b.set_position([_p.x0 - _gap, _p.y0, _p.width + _gap, _p.height])
    label_panels(fig, [[(ax_a, "a"), (ax_b, "b")], [(ax_c, "c"), (ax_d, "d")]])
    save(fig, "fig7")


if __name__ == "__main__":
    main()
