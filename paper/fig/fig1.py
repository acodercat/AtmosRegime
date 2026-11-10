"""Fig. 1 — The silent failure across models.

a  Seventeen configurations: inside − outside, given − outside and flag − outside, with 95 %
   intervals (cluster-level bootstrap, 32 clusters; 10,000 draws).
b  Share of wrong outside answers that equal the standard-formula (shortcut) value.

Data: results/paper/d_reword.json (local configurations over the 69 templates whose wording matches
the training study; explore.d_reword_sensitivity, PREREG_PAPER_ANALYSES §19),
results/d2/frontier_deepseek-v4-flash-off.json (explore.frontier_templates_report),
results/d2/report.json (S1 capture share; explore.d2_report).
"""

from __future__ import annotations

import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from common import C, CONFIG_LABEL, d_contrasts, DOUBLE, RESULTS, ci_err, load, label_panels, pct, save, space_panels, style

CONTRASTS = ["inside_minus_outside", "given_minus_outside", "flag_minus_outside"]
CONTRAST_LABEL = {
    "inside_minus_outside": "inside − outside",
    "given_minus_outside": "given − outside",
    "flag_minus_outside": "flag − outside",
}
CONTRAST_COLOUR = {
    "inside_minus_outside": C["inside"],
    "given_minus_outside": C["given"],
    "flag_minus_outside": C["flag"],
}


def local_contrasts() -> dict[str, dict]:
    """key -> {contrast: {mean, ci95}} for every local configuration, over 69 templates."""
    return d_contrasts()


def main() -> None:
    style()
    contrasts = local_contrasts()
    frontier = load("d2/frontier_deepseek-v4-flash-off.json")
    contrasts["deepseek-v4-flash.think-off"] = frontier["contrasts"]
    s1 = {k: v["S1_capture_outside_wrong"] for k, v in load("d2/report.json")["configurations"].items()}
    s1["deepseek-v4-flash.think-off"] = frontier["S1_capture_outside_wrong"]

    order = sorted(contrasts, key=lambda k: contrasts[k]["inside_minus_outside"]["mean"])
    y = np.arange(len(order))

    fig = plt.figure(figsize=(DOUBLE, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.06)
    ax_b = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1], sharey=ax_b)

    # --- a: contrasts -------------------------------------------------------------------
    for j, con in enumerate(CONTRASTS):
        dy = (j - 1) * 0.26
        for i, key in enumerate(order):
            v = contrasts[key][con]
            m = pct(v["mean"])
            frontier_marker = key.startswith("deepseek")
            ax_b.errorbar(m, y[i] + dy, xerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt="D" if frontier_marker else "o",
                          color=CONTRAST_COLOUR[con], ms=3 if frontier_marker else 2.8, elinewidth=0.7,
                          label=CONTRAST_LABEL[con] if i == 0 else None)
    ax_b.axvline(0, color="#999999", lw=0.6, ls=":")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([CONFIG_LABEL.get(k, k) for k in order], fontsize=6)
    ax_b.set_xlabel("Accuracy difference (percentage points)")
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.15, 1.0), ncol=1, handletextpad=0.3, borderaxespad=0.2)
    ax_b.set_ylim(-0.6, len(order) - 0.4)

    # --- b: shortcut capture ----------------------------------------------------------------
    for i, key in enumerate(order):
        v = s1.get(key)
        if not v or v["mean"] is None:
            continue
        m = pct(v["mean"])
        ax_c.errorbar(m, y[i], xerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt="D" if key.startswith("deepseek") else "o", color=C["shortcut"], ms=3, elinewidth=0.7)
    ax_c.set_xlabel("Wrong outside answers equal\nto the standard-formula value (%)")
    ax_c.set_xlim(0, 100)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    ax_c.tick_params(axis="y", length=0)

    space_panels(fig)
    label_panels(fig, [[(ax_b, "a"), (ax_c, "b")]])
    save(fig, "fig1")


if __name__ == "__main__":
    main()
