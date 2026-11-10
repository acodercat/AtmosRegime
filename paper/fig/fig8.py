"""Fig. 8 — Training dynamics of the template study (X): what the reward does and does not teach.

Rows: Qwen3-8B (top) and Qwen3.5-9B (bottom); two seeds averaged where both exist (R arm: one seed).
a  Training reward (verified answer) per step for the five arms and the random-reward control.
b  Policy entropy per step: answer-reward training sharpens the policy; the random reward does not.
c  Mean response length per step.
d  Training-fold probe (256 rows, every 20 steps): answer accuracy. The warm-started arms begin
   higher; all answer-reward arms climb; the random reward does not.

Data: results/curves/x-<family>-<fold>-<arm>-s<seed>.csv and results/curves/sft-x-<family>-<fold>.csv
(explore.export_curves; sources in results/curves/x_sources.json).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import C, DOUBLE, FAMILY_LABEL, RESULTS, curve, label_panels, pct, save, space_panels, style

FAMS = ["qwen3-8b", "qwen35-9b"]
ARMS = ["A", "D", "F", "WA", "WD", "R"]
# Six arms need six clearly separable hues (Paul Tol "bright"); the random-reward control stays grey and dashed.
ARM_COLOUR = {"A": "#4477AA", "D": "#66CCEE", "F": "#AA3377", "WA": "#EE6677", "WD": "#CCBB44", "R": "#888888"}
ARM_LABEL = {"A": "A: answer reward", "D": "D: counterfactual", "F": "F: answer + judgement", "WA": "WA: warm start + A", "WD": "WD: warm start + D", "R": "R: random reward"}
COLS = {
    "reward": "critic/score/mean",
    "entropy": "actor/entropy",
    "length": "response_length/mean",
    "probe": "val-aux/atmoscoder/train_probe/answer/mean@1",
}


def mean_curve(fam: str, arm: str, col: str, max_step: int = 120):
    """Mean over folds and seeds of one column, on the steps all runs share."""
    series = []
    for fold in ("F1", "F2"):
        for seed in (42, 43):
            path = RESULTS / "curves" / f"x-{fam}-{fold}-{arm}-s{seed}.csv"
            if not path.exists():
                continue
            s, v = curve(f"curves/{path.name}", col)
            keep = s <= max_step
            series.append(dict(zip(s[keep].tolist(), v[keep].tolist())))
    if not series:
        return None, None
    steps = sorted(set.intersection(*(set(d) for d in series)))
    arr = np.array([[d[st] for st in steps] for d in series])
    return np.array(steps), arr


def smooth(v: np.ndarray, k: int = 5) -> np.ndarray:
    if len(v) < k:
        return v
    kern = np.ones(k) / k
    pad = np.pad(v, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, kern, mode="valid")


def main() -> None:
    style()
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 4.3), gridspec_kw=dict(wspace=0.55, hspace=0.3))
    for r, fam in enumerate(FAMS):
        for c, (key, ylabel) in enumerate([("reward", "Training reward"), ("entropy", "Policy entropy"), ("length", "Response length (tokens)"), ("probe", "Training-fold probe accuracy (%)")]):
            ax = axes[r, c]
            lo_hi = []
            r_curve = None
            for arm in ARMS:
                steps, arr = mean_curve(fam, arm, COLS[key])
                if steps is None:
                    continue
                sm = (lambda v: v) if key == "probe" else smooth
                m, lo, hi = sm(arr.mean(0)), sm(arr.min(0)), sm(arr.max(0))
                if key == "probe":
                    m, lo, hi = pct(m), pct(lo), pct(hi)
                colour = ARM_COLOUR[arm]
                if arm == "R":
                    r_curve = (steps, m)
                    ax.plot(steps, m, color=colour, lw=1.0, ls="--", label=ARM_LABEL[arm])
                    continue
                if arr.shape[0] > 1:  # range over folds and seeds
                    ax.fill_between(steps, lo, hi, color=colour, alpha=0.13, lw=0)
                ax.plot(steps, m, color=colour, lw=1.1, label=ARM_LABEL[arm])
                lo_hi += [np.nanmin(lo), np.nanmax(hi)]
            # Keep the five reward arms readable: if the random-reward control runs far above them, let it leave
            # the frame and say so, rather than squeezing the other curves into the bottom of the panel.
            if key in ("entropy", "length") and r_curve is not None and lo_hi:
                top = max(lo_hi)
                span = top - min(lo_hi)
                if np.nanmax(r_curve[1]) > top + 0.6 * span:
                    y_lo, y_hi = min(lo_hi) - 0.08 * span, top + 0.25 * span
                    ax.set_ylim(y_lo, y_hi)
                    rs, rv = r_curve
                    exit_step = rs[np.argmax(rv > y_hi)]
                    ax.annotate("random reward,\noff scale", xy=(exit_step, y_hi), xytext=(exit_step + 12, y_hi - 0.06 * (y_hi - y_lo)),
                                fontsize=5.5, color="#666666", ha="left", va="top", annotation_clip=False)
            ax.set_xlim(0, 120)
            if r == 1:
                ax.set_xlabel("Training step")
            if key == "reward":
                ax.set_ylim(0, 1)
            if key == "probe":
                ax.set_ylim(0, 100)
            ax.set_ylabel(ylabel if r == 0 else "", fontsize=6.5)
            if c == 0:
                ax.text(0.03, 0.97, FAMILY_LABEL[fam], transform=ax.transAxes, va="top", fontsize=6.5, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=6, fontsize=6, columnspacing=1.4, handlelength=2.6)
    for h in leg.legend_handles:
        h.set_linewidth(2.2)
    space_panels(fig)
    label_panels(fig, [[(axes[0, c], letter) for c, letter in enumerate("abcd")]])
    save(fig, "fig8")


if __name__ == "__main__":
    main()
