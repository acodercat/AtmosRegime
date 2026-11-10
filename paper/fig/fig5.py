"""Fig. 5 — Reinforcement learning rarely repairs what the base model cannot sample.

Layout: the learning-rate curve (a) large on the left; the two per-mechanism scatters (b, c) stacked right.

a  Per mechanism: base pass@64 on the outside version against outside accuracy after RL (A arm,
   mean of two seeds), two families. Mechanisms the base model never solved in 64 samples are
   marked.
b  Share of mechanisms learned (endpoint ≥ 50 %) by bin of base pass@1, with Clopper–Pearson 95 %
   intervals, and the unscaled reference g(p) = 1 − (1 − p)^8 − p^8: the probability that a group
   of eight rollouts contains both a right and a wrong answer.
c  Base pass@64 per mechanism across three base models (Spearman ρ with 95 % intervals); the
   mechanisms no base model sampled correctly are circled. (Formerly Fig. 7d; the shortcut-share
   panels formerly here are now Fig. 7c,d.)

Data: results/x/report_qwen3-8b.json, report_qwen35-9b_s42-43.json (explore.x_report),
results/paper/pcurve.json (explore.paper_pcurve, PREREG_PAPER_ANALYSES §18),
results/paper/round4_families.json (explore.paper_round4 §16).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C, DOUBLE, FAMILY_LABEL, ci_err, load, label_panels, legend_above, space_panels, pct, save, style

FAMS = [("qwen3-8b", "report_qwen3-8b.json", C["rl"]), ("qwen35-9b", "report_qwen35-9b_s42-43.json", C["rl2"])]


def main() -> None:
    style()
    # Learning-rate curve large on the left (panel a); the two per-mechanism scatters stacked right (b, c).
    fig = plt.figure(figsize=(DOUBLE, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.88, bottom=0.11)
    ax_a = fig.add_subplot(gs[:, 0])   # pass@64 against trained accuracy (panel a)
    ax_b = fig.add_subplot(gs[0, 1])   # learned share by pass@1 bin (panel b)
    ax_d = fig.add_subplot(gs[1, 1])   # base pass@64 across models (panel c)
    r4f = load("paper/round4_families.json")

    # --- a: scatter ----------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    n_zero = []
    for fam, path, colour in FAMS:
        per = load(f"x/{path}")["P1"]["per_mechanism"]
        xs = np.array([r["pass@64"] for r in per.values()])
        ys = np.array([(r["A_s42"] + r["A_s43"]) / 2 for r in per.values()])
        jitter = rng.uniform(-0.022, 0.022, size=len(xs))
        yj = rng.uniform(-2.2, 2.2, size=len(xs))  # spread the points stacked at 0 and 100
        zero = xs == 0
        ax_a.scatter(pct(xs[~zero]) + jitter[~zero] * 100, pct(ys[~zero]) + yj[~zero], s=14, color=colour, alpha=0.85, edgecolor="white", lw=0.3, label=FAMILY_LABEL[fam])
        ax_a.scatter(pct(xs[zero]) + jitter[zero] * 100, pct(ys[zero]) + yj[zero], s=18, facecolor="none", edgecolor=colour, lw=0.8)
        n_zero.append(int(zero.sum()))
    ax_a.axhline(50, color="#999999", lw=0.6, ls=":")
    ax_a.set_xlabel("Base pass@64 on the outside version (%)")
    ax_a.set_ylabel("Outside accuracy after RL (%)")
    ax_a.text(4, 50, "learned above this line", fontsize=5.5, color="#666666", va="bottom")
    ax_a.set_xlim(-6, 106)
    ax_a.set_ylim(-6, 106)
    ax_a.set_yticks(range(0, 101, 20))
    handles, labels = ax_a.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", mfc="none", mec="#333333", mew=0.8, ls="none", ms=4))
    labels.append("never solved in 64 samples")
    legend_above(ax_a, handles, labels, ncol=3)

    # --- b: bins vs g(p) ---------------------------------------------------------------------------
    pc = load("paper/pcurve.json")
    # boundaries of the four pre-registered pass@1 bins (PREREG_PAPER_ANALYSES §18)
    for edge in pc["definition"]["bins"][1:]:
        ax_b.axvline(pct(edge[0]), color="#BBBBBB", lw=0.5, ls=":", zorder=0)
    for fam, _, colour in FAMS:
        rows = pc["families"][fam]["bins"]
        for k, r in enumerate(rows):
            if r["share"] is None:
                continue
            x = pct(r["p_median"]) if r["p_median"] else 0.045  # a bin median of p = 0 is drawn at the axis edge
            x = max(x, 0.045) * (1.0 if colour == C["rl"] else 1.25)
            ax_b.errorbar(x, pct(r["share"]), yerr=ci_err(pct(r["share"]), [pct(c) for c in r["ci95"]]), fmt="o", color=colour, ms=3.5, elinewidth=0.7, label=FAMILY_LABEL[fam] if k == 0 else None)
            if r["share"] >= 0.95 and colour != C["rl"]:
                ax_b.text(x, pct(r["share"]) + (4 if colour == C["rl"] else 9), f"{r['learned']}/{r['n']}", fontsize=5, ha="center", va="bottom", color=colour)
            else:
                ax_b.text(x * (0.72 if colour == C["rl"] else 1.4), pct(r["share"]), f"{r['learned']}/{r['n']}", fontsize=5, ha="right" if colour == C["rl"] else "left", va="center", color=colour)
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.015, 100)
    ax_b.set_ylim(-4, 118)
    ax_b.set_yticks(range(0, 101, 25))
    ax_b.set_xlabel("Base pass@1 on the outside version (%, log scale)")
    ax_b.set_ylabel("Mechanisms learned (%)")
    legend_above(ax_b, ncol=2)

    # --- c: three-family pass@64 ------------------------------------------------------------------------
    p64 = r4f["pass64"]
    pairs = [("qwen3-8b", "ministral3-8b", "Qwen3-8B", "Ministral 3 8B"), ("qwen3-8b", "qwen35-9b", "Qwen3-8B", "Qwen3.5-9B")]
    rng = np.random.default_rng(2)
    for (a, b, la, lb), colour, marker in zip(pairs, [C["rl"], C["rl2"]], ["o", "^"]):
        mechs = sorted(set(p64[a]) & set(p64[b]))
        xs = np.array([p64[a][m] for m in mechs]) + rng.uniform(-0.01, 0.01, len(mechs))
        ys = np.array([p64[b][m] for m in mechs]) + rng.uniform(-0.01, 0.01, len(mechs))
        key = f"{min(a, b)}~{max(a, b)}" if f"{min(a, b)}~{max(a, b)}" in r4f["S16a"]["pairs"] else f"{b}~{a}"
        rho = r4f["S16a"]["pairs"][key]
        ax_d.scatter(pct(xs), pct(ys), s=12, color=colour, marker=marker, alpha=0.85, edgecolor="white", lw=0.3, label=f"{lb}: ρ = {rho['rho']:.2f} [{rho['ci95'][0]:.2f}, {rho['ci95'][1]:.2f}]")
    zero_all = r4f["S16a"]["zero_in_all"]
    for m in zero_all:
        ax_d.scatter([pct(p64["qwen3-8b"][m])], [pct(p64["ministral3-8b"][m])], s=40, facecolor="none", edgecolor="black", lw=0.8, zorder=5)
    ax_d.scatter([], [], s=40, facecolor="none", edgecolor="black", lw=0.8, label="never solved by any model")
    ax_d.plot([0, 100], [0, 100], color="#bbbbbb", lw=0.6, ls=":")
    ax_d.set_xlabel("Base pass@64, Qwen3-8B (%)")
    ax_d.set_ylabel("Base pass@64,\nother family (%)")
    ax_d.set_ylim(-4, 104)  # empty band above the data holds the legend
    ax_d.set_yticks(range(0, 101, 20))
    legend_above(ax_d, ncol=2)
    nest = r4f["S16b"]["mechanisms"]

    # no space_panels: it does not handle a panel spanning two rows; spacing is set on the gridspec
    label_panels(fig, [[(ax_a, "a"), (ax_b, "b")], [(ax_d, "c")]])
    save(fig, "fig5")


if __name__ == "__main__":
    main()
