"""Fig. 2 — Where the failures are and how large they are.

a  Failure atlas: outside-regime accuracy, conditional on the inside version being solved, for every
   mechanism × configuration (grey = too few solved inside versions). Mechanisms are ordered by their
   mean over the local configurations (hardest left) and configurations by their mean (weakest top);
   subfields are not shown because they have no detectable effect (Supplementary Note 2). The mechanisms
   that fail in every local configuration (results/paper/mechanism_examples.json, PREREG_PAPER_ANALYSES §23)
   are bracketed.
b  How wrong the standard formula is on the benchmark traps with a defined deviation: distribution of the
   deviation of the shortcut answer from the certified answer (bins as in explore/paper_cost.py).
   The by-subfield share of large errors is Supplementary Fig. 3 (si_figs.py).

Data: results/conditional/d.json (explore.conditional_report), results/paper/atlas.json
(explore.paper_atlas), results/paper/cost.json (explore.paper_cost), results/paper/mechanism_examples.json.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C, CONFIG_LABEL, DOUBLE, INK, load, label_panels, pct, save, seq_cmap, style

BIN_ORDER = ["<=5%", "5-20%", "20-50%", "50-100%", "2-10x", ">10x"]
BIN_LABEL = ["≤5%", "5–20%", "20–50%", "50–100%", "2–10×", ">10×"]


def atlas_matrix():
    cond = load("conditional/d.json")
    atlas = load("paper/atlas.json")
    configs = atlas["configs"]
    per_t = atlas["per_template"]
    templates = sorted(per_t, key=lambda t: per_t[t]["mean_rec_outside_local"])  # hardest left
    M = np.full((len(templates), len(configs)), np.nan)
    for j, cfg in enumerate(configs):
        per = cond[cfg]["per_template"]
        for i, t in enumerate(templates):
            if t in per:
                M[i, j] = per[t]["outside"]
    order = np.argsort([np.nanmean(M[:, j]) for j in range(len(configs))])  # weakest configuration on top
    return templates, [configs[j] for j in order], M[:, order]


def main() -> None:
    style()
    templates, configs, M = atlas_matrix()
    universal = {r["template"] for r in load("paper/mechanism_examples.json")["universal_failures"]}
    cost = load("paper/cost.json")["traps"]["all_without_magnitude_defect"]

    fig = plt.figure(figsize=(DOUBLE, 4.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.42], hspace=0.42, left=0.15, right=0.93, top=0.9, bottom=0.1)
    top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 0.014], wspace=0.02)
    ax_a = fig.add_subplot(top[0])
    ax_cb = fig.add_subplot(top[1])
    bottom = gs[1].subgridspec(1, 2, width_ratios=[0.62, 0.38], wspace=0.05)
    ax_b = fig.add_subplot(bottom[0])

    # --- a: atlas --------------------------------------------------------------------------------
    im = ax_a.imshow(M.T, aspect="auto", cmap=seq_cmap(), vmin=0, vmax=1, interpolation="nearest")
    ax_a.set_yticks(range(len(configs)))
    ax_a.set_yticklabels([CONFIG_LABEL.get(c, c) for c in configs], fontsize=6)
    ax_a.set_xticks([])
    for sp in ax_a.spines.values():
        sp.set_visible(False)
    ux = [i for i, t in enumerate(templates) if t in universal]
    x0, x1 = min(ux) - 0.5, max(ux) + 0.5
    ax_a.plot([x0, x0, x1, x1], [-0.7, -1.3, -1.3, -0.7], color=INK, lw=0.7, clip_on=False)
    ax_a.text(x1 + 0.8, -1.0, f"fail in every local configuration ({len(ux)})", fontsize=6, va="center", ha="left")
    ax_a.set_xlabel(f"{len(templates)} mechanisms, from hardest (left) to easiest (right)")
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label("Outside accuracy\ngiven inside solved", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)

    # --- b: how far the standard-formula answer is off ------------------------------------------
    counts = np.array([cost["bins"][b] for b in BIN_ORDER])
    total = counts.sum()
    x = np.arange(len(BIN_ORDER))
    ax_b.bar(x, pct(counts / total), color=[C["warm"]] * 3 + [C["shortcut"]] * 3, width=0.68)
    for xi, c in zip(x, counts):
        ax_b.text(xi, pct(c / total) + 1, str(c), ha="center", va="bottom", fontsize=6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(BIN_LABEL, fontsize=6)
    ax_b.set_ylabel("Share of traps (%)")
    ax_b.set_xlabel(f"Deviation of the standard-formula answer from the certified one ({int(total)} traps)")
    ax_b.set_ylim(0, 48)
    ax_b.set_yticks([0, 20, 40])
    med, share = cost["R_quartiles"][1], cost["share_R_gt_50"]
    ax_b.text(1.06, 0.55, f"Median deviation: {pct(med):.0f}%\nMore than 50% off: {pct(share):.0f}% of traps",
              transform=ax_b.transAxes, va="center", ha="left", fontsize=7, linespacing=1.6)

    label_panels(fig, [[(ax_a, "a")], [(ax_b, "b")]])
    # one column of letters: put b directly under a
    la = next(tx for tx in fig.texts if tx.get_text() == "a")
    lb = next(tx for tx in fig.texts if tx.get_text() == "b")
    lb.set_x(la.get_position()[0])
    save(fig, "fig2")


if __name__ == "__main__":
    main()
