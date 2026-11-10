"""Supplementary figures.

si1  Scale and thinking: inside − outside per family at the small and large size, thinking off
     and on (16 local configurations), with 95 % cluster-bootstrap intervals.

si6  Training reward of the benchmark-problem study (formerly Fig. 4a).
si7  Large-error share of the benchmark traps by subfield (formerly Fig. 2c).
si8  Benchmark regime traps after each intervention (formerly Fig. 7c).

Data: results/d2/families_report_*.json, results/d_execution/completed_configurations.json.
"""

from __future__ import annotations

import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from common import C, CONFIG_LABEL, d_contrasts, RESULTS, SINGLE, ci_err, curve, load, pct, save, seq_cmap, style

FAMILIES = [
    ("Qwen3", ["qwen3-8b"]),
    ("Qwen3.5", ["qwen3.5-9b", "qwen3.5-27b"]),
    ("Gemma 4", ["gemma4-e4b", "gemma4-31b"]),
    ("Granite 4.2", ["granite4.2-8b", "granite4.2-30b"]),
    ("Ministral 3", ["ministral3-8b", "ministral3-14b"]),
]


def contrasts() -> dict[str, dict]:
    return {k: v["inside_minus_outside"] for k, v in d_contrasts().items()}


def si1() -> None:
    con = contrasts()
    fig, ax = plt.subplots(figsize=(SINGLE * 1.6, 2.6))
    x = 0
    ticks, labels = [], []
    for fam, models in FAMILIES:
        for model in models:
            for mode, marker, fill in [("off", "o", True), ("on", "o", False)]:
                key = f"{model}.think-{mode}"
                if key not in con:
                    continue
                v = con[key]
                m = pct(v["mean"])
                ax.errorbar(x, m, yerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt=marker, color=C["inside"],
                            mfc=C["inside"] if fill else "white", ms=4, elinewidth=0.7)
                ticks.append(x)
                labels.append(CONFIG_LABEL[key].replace(" (thinking)", "\n(thinking)"))
                x += 1
        x += 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("Inside − outside accuracy (percentage points)")
    ax.axhline(0, color="#999999", lw=0.6, ls=":")
    ax.plot([], [], "o", color=C["inside"], label="thinking off")
    ax.plot([], [], "o", color=C["inside"], mfc="white", label="thinking on")
    ax.legend(loc="upper right")
    save(fig, "si1")



def si2() -> None:
    """Warm-up SFT loss (one epoch) for the four warm starts of the template study."""
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE * 1.6, 1.9), sharey=False)
    for ax, fam in zip(axes, ["qwen3-8b", "qwen35-9b"]):
        for fold, colour in (("F1", "#333333"), ("F2", C["rl2"])):
            s, v = curve(f"curves/sft-x-{fam}-{fold}.csv", "train/loss")
            ax.plot(s, v, color=colour, lw=0.8, label=f"fold {fold[-1]}")
        ax.set_xlabel("SFT step")
        ax.set_title({"qwen3-8b": "Qwen3-8B", "qwen35-9b": "Qwen3.5-9B"}[fam], loc="left")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Warm-up SFT loss")
    from common import label_panels, space_panels
    space_panels(fig)
    label_panels(fig, [[(axes[0], "a"), (axes[1], "b")]])
    save(fig, "si2")


def si3() -> None:
    """Outcome of every mechanism under every intervention (Qwen3-8B), sorted by base pass@64.
    Data: results/paper/synthesis.json (matrix; explore.paper_synthesis §13)."""
    import numpy as np
    from matplotlib.colors import ListedColormap
    m = load("paper/synthesis.json")["matrix"]
    cols = [("base_outside", "Base"), ("pass@64", "Base\npass@64"), ("teacher_given", "Teacher,\ngiven"), ("A", "RL A"), ("D", "RL D"), ("F", "RL F"), ("dose_rl", "Worked\nexamples"), ("P", "Distil. P"), ("S", "S"), ("Pk2", "Pk2"), ("Pk3", "Pk3"), ("Pshuf", "Pshuf")]
    mechs = sorted(m, key=lambda k: (m[k]["pass@64"] if m[k]["pass@64"] is not None else -1, m[k]["teacher_given"] or 0))
    M = np.array([[np.nan if m[k].get(c) is None else m[k][c] for c, _ in cols] for k in mechs])
    fig, ax = plt.subplots(figsize=(SINGLE * 1.7, 9.2))
    cmap = seq_cmap()
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for x in (2.5, 6.5):
        ax.axvline(x, color="white", lw=1.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([l for _, l in cols], fontsize=5.5, rotation=90)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(mechs)))
    ax.set_yticklabels([k.replace("_", " ") for k in mechs], fontsize=5.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Outside accuracy, or base pass@64 (fraction)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, "si3")


def si4() -> None:
    """Distance of wrong outside answers from the certified value, per arm (Qwen3-8B).
    Data: results/paper/synthesis_round3.json (R2_error_magnitude)."""
    r = load("paper/synthesis_round3.json")["R2_error_magnitude"]
    arms = [("base", "Base", C["base"]), ("A", "RL (A)", C["rl"]), ("P", "Distil. P", C["distil"]), ("Pk2", "Pk2", C["distil_bad"]), ("S", "S", C["distil_bad"]), ("Pshuf", "Pshuf", C["distil_bad"])]
    fig, ax = plt.subplots(figsize=(SINGLE * 1.2, 2.2))
    for i, (k, lab, col) in enumerate(arms):
        v = r[k]
        ax.plot([i, i], [v["q25"], v["q75"]], color=col, lw=4, solid_capstyle="butt")
        ax.plot(i, v["median"], "o", color="white", mec=col, ms=4, mew=1.2)
        ax.text(i, v["q75"] + 0.06, f"n={v['n']}", ha="center", fontsize=5.5, color="#555555")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([l for _, l, _ in arms])
    ax.set_ylabel("Relative error of wrong\noutside answers (median, IQR)")
    ax.set_ylim(0, max(r[k]["q75"] for k, _, _ in arms) * 1.2)
    save(fig, "si4")


def si5() -> None:
    """Pre-registered replication: paired regime − geometry gap on 54 independent benchmark traps,
    per configuration and pooled. Data: results/d5/report.json (explore.d5_report)."""
    d = load("d5/report.json")
    pc = d["per_configuration"]
    keys = sorted(pc, key=lambda k: pc[k]["mean"])
    fig, ax = plt.subplots(figsize=(SINGLE * 1.3, 3.2))
    for i, k in enumerate(keys):
        v = pc[k]; m = pct(v["mean"])
        ax.errorbar(m, i, xerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt="o", color=C["inside"], ms=3, elinewidth=0.7)
    v = d["D5-1"]; m = pct(v["mean"])
    ax.errorbar(m, len(keys) + 0.8, xerr=ci_err(m, [pct(c) for c in v["ci95"]]), fmt="D", color=C["outside"], ms=4, elinewidth=1.0)
    ax.axvline(0, color="#999999", lw=0.6, ls=":")
    ax.set_yticks(list(range(len(keys))) + [len(keys) + 0.8])
    ax.set_yticklabels([CONFIG_LABEL.get(k, k) for k in keys] + ["Pooled"], fontsize=5.5)
    ax.set_xlabel("Regime − geometry trap gap (percentage points)")
    save(fig, "si5")



def si6() -> None:
    """Training reward of the benchmark-problem study (Qwen3-8B, DAPO): four answer-reward runs and the
    random-reward control (results/curves/*.csv; formerly Fig. 4a)."""
    runs = {"dapo-qwen3-8b": "run 2", "seed1": "seed 1", "seed2": "seed 2", "seed3": "seed 3"}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    for run in runs:
        s, v = curve(f"curves/{run}.csv", "critic/score/mean")
        keep = s <= 120
        ax.plot(s[keep], v[keep], color=C["rl"], lw=0.8, alpha=0.9 if run == "dapo-qwen3-8b" else 0.45, label="answer reward" if run == "dapo-qwen3-8b" else None)
    s, v = curve("curves/ctrl-random.csv", "critic/score/mean")
    keep = s <= 120
    ax.plot(s[keep], v[keep], color=C["random"], lw=0.8, ls="--", label="random reward")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training reward (verified answer)")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    save(fig, "si6")


def si7() -> None:
    """Share of benchmark traps whose standard-formula answer is more than 50 % off, by subfield
    (results/paper/cost.json; formerly Fig. 2c)."""
    cost = load("paper/cost.json")["traps"]
    cats = cost["by_category"]
    share = cost["all_without_magnitude_defect"]["share_R_gt_50"]
    from common import CATEGORY_LABEL
    keys = sorted(cats, key=lambda k: cats[k]["share_R_gt_50"])
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    for i, k in enumerate(keys):
        v = cats[k]
        m = pct(v["share_R_gt_50"])
        ax.errorbar(m, i, xerr=ci_err(m, [pct(c) for c in v["share_R_gt_50_ci95"]]), fmt="o", color=C["rl"], ms=3.5, elinewidth=0.7)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{CATEGORY_LABEL.get(k, k)} (n = {cats[k]['defined']})" for k in keys], fontsize=6)
    ax.axvline(pct(share), color="#999999", lw=0.6, ls=":")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Traps with the standard answer > 50 % off (%)")
    save(fig, "si7")


def si8() -> None:
    """Forty-six benchmark regime traps after each intervention (results/paper/round4_traps.json,
    results/traps_rl/report_qwen3-8b.json, results/paper/frontier21.json; formerly Fig. 7c)."""
    r4t = load("paper/round4_traps.json")
    trl = load("traps_rl/report_qwen3-8b.json")
    q = r4t["families"]["qwen3-8b"]["groups"]
    mi = r4t["families"]["ministral3-8b"]["groups"]
    bars = [
        ("Base", trl["arms"]["base"]["all"], C["base"]),
        ("RL (A)", trl["arms"]["A"]["all"], C["rl"]),
        ("Warm-up + RL (WA)", trl["arms"]["WA"]["all"], C["rl"]),
        ("Warm-up + examples", q["dose_warm"]["all"], C["warm"]),
        ("Examples + RL", q["dose_rl"]["all"], C["dose"]),
        ("Distil. P", q["opd_P"]["all"], C["distil"]),
        ("Pk2", q["opd_Pk2"]["all"], C["distil_bad"]),
        ("Pshuf", q["opd_Pshuf"]["all"], C["distil_bad"]),
        ("Ministral base", mi["base"]["all"], C["base"]),
        ("Ministral examples + RL", mi["dose_rl"]["all"], C["dose2"]),
    ]
    yb = np.array([i if i < 8 else i + 0.6 for i in range(len(bars))])
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    for yi, (label, v, colour) in zip(yb, bars):
        m = pct(v["trap_acc"])
        ax.barh(yi, m, color=colour, height=0.7)
        ax.errorbar(m, yi, xerr=ci_err(m, [pct(c) for c in v["trap_acc_ci95"]]), fmt="none", ecolor="#333333", elinewidth=0.6)
    fr = load("paper/frontier21.json")["configurations"]["frontier21-deepseek-v4-flash-off"]["textbook"]
    ax.axvline(pct(fr["trap"]["mean"]), color=C["frontier"], lw=0.8, ls="-.")
    ax.text(pct(fr["trap"]["mean"]) + 1.5, yb[-1], "DeepSeek V4 Flash\n(thinking off)", fontsize=5, va="center", ha="left")
    # the comparison the text makes: examples + RL minus WA (same training without examples), paired over traps
    dw = r4t["families"]["qwen3-8b"]["vs_x_A"]["dose_rl_minus_WA"]
    ax.text(31, (yb[2] + yb[4]) / 2, f"Examples + RL − WA:\n{100 * dw['mean']:+.1f} [{100 * dw['ci95'][0]:+.1f}, {100 * dw['ci95'][1]:+.1f}] points".replace("-", "−"),
            fontsize=5.5, ha="left", va="center")
    ax.set_yticks(yb)
    ax.set_yticklabels([b for b, *_ in bars], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy on 46 benchmark regime traps (%)")
    ax.set_xlim(0, 75)
    save(fig, "si8")

def main() -> None:
    style()
    si1()
    si2()
    si3()
    si4()
    si5()
    si6()
    si7()
    si8()


if __name__ == "__main__":
    main()
