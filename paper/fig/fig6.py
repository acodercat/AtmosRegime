"""Fig. 6 — What makes the unreachable learnable.

a  DOSE: 34 observations (17 mechanisms × 2 seeds, Qwen3-8B) randomly assigned 0, 1, 2 or 4 expert
   worked examples in the warm-up. Grey: accuracy of the warm-up model alone; coloured: after the
   identical RL. Lines join the two for each observation.
b  Learned share by dose, Qwen3-8B and Ministral 3 8B (warm-up alone vs after RL).
c  Three ways of supplying the formula, same seventeen (DOSE/text) or seven (distillation)
   mechanisms: a one-sentence statement, 1–4 executable examples, a privileged self-teacher;
   and a teacher given the wrong formula, on all templates.
d  Which reachability predicts success: AUC of the student's base pass@64 and of the teacher's
   given-version accuracy for predicting which of 65 mechanisms each method learned.

Data: results/dose/report_qwen3-8b.json (explore.dose_report), results/dose_ministral/family_compare.json
(explore.dose_family_compare), results/dose_text/dose_text_qwen3-8b.json (explore.dose_text_report),
results/opd/report_qwen3-8b_seeds42-43.json and report_ext_qwen3-8b.json (explore.opd_report*),
results/paper/synthesis.json (explore.paper_synthesis §13).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C, DOUBLE, ci_err, load, label_panels, legend_above, space_panels, pct, save, style

DOSES = [0, 1, 2, 4]


def main() -> None:
    style()
    fig = plt.figure(figsize=(DOUBLE, 4.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --- a: observations ---------------------------------------------------------------------
    obs = load("dose/report_qwen3-8b.json")["observations"]
    rng = np.random.default_rng(1)
    for o in obs:
        x = DOSES.index(o["dose"]) + rng.uniform(-0.22, 0.22)
        ax_a.plot([x, x], [pct(o["warm"]), pct(o["rl"])], color="#bbbbbb", lw=0.5, zorder=1)
        ax_a.scatter(x, pct(o["warm"]), s=9, color=C["warm"], zorder=2)
        ax_a.scatter(x, pct(o["rl"]), s=11, color=C["dose"] if o["dose"] > 0 else C["base"], zorder=3)
    ax_a.axhline(50, color="#999999", lw=0.6, ls=":")
    ax_a.set_xticks(range(len(DOSES)))
    n_dose = {d: sum(1 for o in obs if o["dose"] == d) for d in DOSES}
    ax_a.set_xticklabels([f"{d} example{'s' if d != 1 else ''}\n(n = {n_dose[d]})" for d in DOSES])
    ax_a.set_ylabel("Outside accuracy (%)")
    ax_a.set_ylim(-3, 103)
    ax_a.scatter([], [], s=9, color=C["warm"], label="warm-up alone")
    ax_a.scatter([], [], s=11, color=C["dose"], label="after RL")
    legend_above(ax_a)
    h1 = load("dose/report_qwen3-8b.json")["H1"]
    ax_a.text(0.02, 0.72, f"with − without\nexamples:\n{pct(h1['value']):.1f} [{pct(h1['ci95'][0]):.1f}, {pct(h1['ci95'][1]):.1f}]\npercentage points",
              transform=ax_a.transAxes, ha="left", va="center", fontsize=5.8)

    # --- b: by dose, two families ---------------------------------------------------------------
    fc = load("dose_ministral/family_compare.json")["dose_levels"]
    w = 0.38
    for k, (fam, label, colour) in enumerate([("qwen3-8b", "Qwen3-8B", C["dose"]), ("ministral3-8b", "Ministral 3 8B", C["dose2"])]):
        x = np.arange(len(DOSES)) + (k - 0.5) * w
        warm = [pct(fc[fam][str(d)]["warm_mean"]) for d in DOSES]
        rl = [pct(fc[fam][str(d)]["rl_mean"]) for d in DOSES]
        ax_b.bar(x, rl, w, color=colour, label=f"{label}, after RL")
        ax_b.bar(x, warm, w, color="none", edgecolor="#333333", lw=0.6, hatch="////", label=f"{label}, warm-up alone")
        for xi, d in zip(x, DOSES):
            v = fc[fam][str(d)]
            ax_b.text(xi, pct(v["rl_mean"]) + 2, f"{v['learned']}/{v['n']}", ha="center", fontsize=5, color=colour if k == 0 else "#A34A5E")
    ax_b.set_xticks(range(len(DOSES)))
    ax_b.set_xticklabels([f"{d}" for d in DOSES])
    ax_b.set_xlabel("Verified worked examples in the warm-up (randomly assigned)")
    ax_b.set_ylabel("Mean outside accuracy (%)")
    ax_b.set_ylim(0, 145)
    ax_b.text(0.02, 0.96, "labels: learned / assigned", transform=ax_b.transAxes, fontsize=5.5, color="#555555", va="top", ha="left")
    ax_b.set_yticks([0, 25, 50, 75, 100])
    legend_above(ax_b, ncol=2)

    # --- c: three forms of supply ----------------------------------------------------------------------
    text = load("dose_text/dose_text_qwen3-8b.json")["H1_text"]
    prog = h1
    # distillation contrasts within the distillation pipeline (dynamic sampling off in all its arms): P − S on the
    # main set, and Pshuf − P on all templates (the negative of contrasts.all.P-Pshuf)
    opd = load("opd/report_qwen3-8b_seeds42-43.json")["main_outcome"]["contrasts"]["main_set"]["P-S"]
    pp = load("opd/report_ext_qwen3-8b.json")["main_outcome"]["contrasts"]["all"]["P-Pshuf"]
    ext = pp
    items = [
        ("One\nsentence\nvs none\n(17 mech.)", text["value"], text["ci95"], C["text"]),
        ("Worked\nexamples\nvs none\n(17 mech.)", prog["value"], prog["ci95"], C["dose"]),
        ("Teacher:\nformula\nvs none\n(7 mech.)", opd["mean"], opd["ci95"], C["distil"]),
        ("Teacher:\nright vs wrong\nformula\n(70 mech.)", ext["mean"], ext["ci95"], C["distil"]),
    ]
    for i, (label, m, ci, colour) in enumerate(items):
        m = pct(m)
        ax_c.bar(i, m, color=colour, width=0.65)
        ax_c.errorbar(i, m, yerr=ci_err(m, [pct(c) for c in ci]), fmt="none", ecolor="#333333", elinewidth=0.6)
    ax_c.axhline(0, color="#999999", lw=0.6)
    ax_c.set_xticks(range(len(items)))
    ax_c.set_xticklabels([l for l, *_ in items], fontsize=5.5)
    ax_c.set_ylabel("Outside accuracy minus matched\ncontrol (percentage points)")
    ax_c.set_ylim(-15, 100)

    # --- d: whose reachability predicts success ---------------------------------------------------------
    s2 = load("paper/synthesis.json")["S2_reachability_source"]["per_outcome"]
    methods = [("A", "RL only"), ("P", "Teacher distillation")]
    x = np.arange(len(methods))
    stu = [s2[m]["auc_student"] for m, _ in methods]
    tea = [s2[m]["auc_teacher"] for m, _ in methods]
    ax_d.bar(x - 0.16, stu, 0.3, color=C["rl"], label="base model's own success rate")
    ax_d.bar(x + 0.16, tea, 0.3, color=C["distil"], label="teacher's success rate, given the formula")
    for xi, (m, _) in zip(x, methods):
        d, ci = abs(s2[m]["student_minus_teacher"]), sorted(abs(c) for c in s2[m]["ci95"])
        ax_d.text(xi, max(stu[xi], tea[xi]) + 0.03, f"difference {d:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", ha="center", va="bottom", fontsize=5.5)
    ax_d.axhline(0.5, color="#999999", lw=0.6, ls=":")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([l for _, l in methods])
    ax_d.set_ylabel("AUC for predicting which\nmechanisms were learned")
    ax_d.set_ylim(0, 1.45)
    ax_d.set_xlim(-0.6, 1.6)
    ax_d.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_d.legend(loc="upper left")

    space_panels(fig)
    label_panels(fig, [[(ax_a, "a"), (ax_b, "b")], [(ax_c, "c"), (ax_d, "d")]])
    save(fig, "fig6")


if __name__ == "__main__":
    main()
