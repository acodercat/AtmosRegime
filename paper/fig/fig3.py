"""Fig. 3 — The missing piece is the correct formula.

a  Intervention ladder: change in outside-regime accuracy when the prompt only flags that the
   standard formula fails, asks the model to judge validity first, asks it to recall the relevant
   formulas itself, supplies a handbook, or states the correct formula. Three models.
   Intervals: cluster bootstrap where the source analysis reports one; judge-first is a single
   paired run on 425 rows (no interval stored).
b  Knowing without doing: among wrong outside answers of DeepSeek V4 Flash, the share equal to the
   standard-formula value (cluster-weighted, cluster-bootstrap interval), split by whether the
   model itself judged that the standard formula holds.

Data: results/d_execution/completed_configurations.json and results/d2/frontier_deepseek-v4-flash-off.json
(flag, given); results/x7/x7_report_strict.json (judge first); results/x8/report.json (recall);
results/e7/report.txt (handbook, DeepSeek, 17 templates; numbers transcribed from the text report);
results/x1/x1a.json (2×2).
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

from common import C, DOUBLE, d_contrasts, RESULTS, ci_err, load, label_panels, legend_above, pct, save, space_panels, style

MODELS = [("qwen3-8b", "Qwen3-8B"), ("qwen3.5-9b", "Qwen3.5-9B"), ("deepseek", "DeepSeek V4 Flash")]
STEPS = ["flag", "judge", "recall", "handbook", "given"]
STEP_LABEL = {"flag": "Flag that it fails", "judge": "Judge validity first", "recall": "Recall formulas first", "handbook": "Handbook in context", "given": "Correct formula given"}
STEP_COLOUR = {"flag": "#DCDCDC", "judge": "#B0B0B0", "recall": "#808080", "handbook": "#454545", "given": C["given"]}  # four prompts that only point at the regime in greys; the one that supplies the formula in colour


def handbook_from_text() -> dict:
    text = (RESULTS / "e7" / "report.txt").read_text(encoding="utf-8")
    m = re.search(r"handbook-none \+([\d.]+)% \[\+([\d.]+)%, \+([\d.]+)%\]", text)
    g = re.search(r"outside acc\s+none (\d+)/(\d+)=\d+%\s+handbook (\d+)/(\d+)=\d+%\s+given (\d+)/(\d+)", text)
    none = int(g.group(1)) / int(g.group(2))
    given = int(g.group(5)) / int(g.group(6))
    return {"handbook": (float(m.group(1)), [float(m.group(2)), float(m.group(3))]), "given_same_rows": pct(given - none)}


def main() -> None:
    style()
    dex = d_contrasts()  # 69 templates (PREREG_PAPER_ANALYSES §19)
    frontier = load("d2/frontier_deepseek-v4-flash-off.json")["contrasts"]
    x7 = load("x7/x7_report_strict.json")
    x8 = load("x8/report.json")
    hb = handbook_from_text()

    def contrasts(model: str) -> dict[str, tuple[float, list[float] | None]]:
        out = {}
        d2 = frontier if model == "deepseek" else dex[f"{model}.think-off"]
        out["flag"] = (pct(d2["flag_minus_outside"]["mean"]), [pct(c) for c in d2["flag_minus_outside"]["ci95"]])
        out["given"] = (pct(d2["given_minus_outside"]["mean"]), [pct(c) for c in d2["given_minus_outside"]["ci95"]])
        key7 = {"qwen3-8b": "qwen3-8b", "qwen3.5-9b": "qwen35-9b", "deepseek": "deepseek"}[model]
        out["judge"] = (pct(x7[key7]["t_accuracy"] - x7[key7]["c_accuracy"]), None)
        key8 = {"qwen3-8b": "qwen3-8b-off", "qwen3.5-9b": "qwen3.5-9b-off", "deepseek": "deepseek-off"}[model]
        out["recall"] = (pct(x8[key8]["E-C_outside"]["mean"]), [pct(c) for c in x8[key8]["E-C_outside"]["ci95"]])
        if model == "deepseek":
            # handbook and given on the same 17 mechanisms (results/e7/report.txt), so the two bars are comparable
            out["handbook"] = hb["handbook"]
            out["given"] = (hb["given_same_rows"], None)
        return out

    fig = plt.figure(figsize=(DOUBLE, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.35)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # --- a: ladder ---------------------------------------------------------------------------
    group_w = 0.8
    bar_w = group_w / len(STEPS)
    for gi, (model, label) in enumerate(MODELS):
        con = contrasts(model)
        for si, step in enumerate(STEPS):
            if step not in con:
                continue
            m, ci = con[step]
            xpos = gi + (si - (len(STEPS) - 1) / 2) * bar_w
            ax_a.bar(xpos, m, bar_w * 0.9, color=STEP_COLOUR[step], label=STEP_LABEL[step] if gi == 2 else None)
            if ci is not None:
                ax_a.errorbar(xpos, m, yerr=ci_err(m, ci), fmt="none", ecolor="#333333", elinewidth=0.6)
            else:  # no interval stored for this contrast (judge-first; DeepSeek given on the handbook rows)
                ax_a.plot(xpos, m + 2.5, marker="v", color="#333333", ms=2.5)
    ax_a.axhline(0, color="#999999", lw=0.6)
    ax_a.set_xticks(range(len(MODELS)))
    ax_a.set_xticklabels([l for _, l in MODELS])
    ax_a.set_ylabel("Change in outside-regime accuracy\n(percentage points)")
    legend_above(ax_a, ncol=3)
    ax_a.set_ylim(-8, 90)

    # --- b: shortcut share among wrong answers, by the model's own judgement ----------------------
    # Template-weighted share with its cluster-bootstrap interval, as stored and as quoted in the text.
    x1 = load("x1/x1a.json")["deepseek-v4-flash.think-off"]
    cells = x1["wrong_cells"]
    rows = [("judged1", "Judged valid"), ("judged0", "Judged invalid")]
    for i, (key, label) in enumerate(rows):
        v = x1[f"shortcut_among_{key}_wrong"]
        m = pct(v["mean"])
        ax_b.errorbar(m, i, xerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt="o", color=C["shortcut"], ms=5, elinewidth=1.0, capsize=0)
        n = cells[f"{key}_shortcut"] + cells[f"{key}_other"]
        ax_b.text(m, i - 0.22, f"{m:.0f}% (n = {n})", va="bottom", ha="center", fontsize=6)
    ax_b.set_yticks([0, 1])
    ax_b.set_yticklabels([l for _, l in rows])
    ax_b.set_ylim(-0.8, 1.8)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Wrong outside answers equal to\nthe standard-formula value (%)")
    ax_b.set_title("DeepSeek V4 Flash", fontsize=6.5, loc="left")

    space_panels(fig)
    label_panels(fig, [[(ax_a, "a"), (ax_b, "b")]])
    save(fig, "fig3")


if __name__ == "__main__":
    main()
