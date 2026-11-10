"""Fig. 4 — Reinforcement learning on domain problems generalises, and a trained 9B model matches larger models.

(The training reward of the benchmark-problem study is Supplementary Fig. 9, si_figs.si6.)
a  Held-out probes during training (validation reward at step 0–120): in-distribution test set,
   held-out sources, four non-atmospheric environmental domains; random-reward control dashed.
b  Gain over the base model on nine held-out probes at step 120 (mean ± s.d. over four runs) and
   under the random-reward control.
c  Benchmark regime traps (46) after training: base → warm-up → arms → random reward, two
   families; DeepSeek V4 Flash (thinking off, same harness) as a reference line.
d  Trained Qwen3.5-9B (arm A) minus each comparison model on the regime traps, paired over traps:
   frontier models through their APIs on all 46 traps (PREREG_PAPER_ANALYSES §21) and the diagnostic
   study's open configurations on the 38 traps they share (§22); filled, thinking off; open, thinking on.

Data: results/curves/{dapo-qwen3-8b,seed1,seed2,seed3,ctrl-random}.csv (explore.export_curves),
results/FINAL.md tables 1–2 (atmoscoder_rl.analysis paper / control),
results/traps_rl/report_{qwen3-8b,qwen35-9b}.json (explore.traps_rl_report),
results/paper/frontier21.json, frontier22.json (explore.frontier21_report, frontier22_report).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import C, CONFIG_LABEL, DOUBLE, align_columns, FAMILY_LABEL, RULE, ci_err, curve, load, md_table, md_tables, num, label_panels, legend_above, pct, save, space_panels, style

RUNS = {"dapo-qwen3-8b": "run 2", "seed1": "seed 1", "seed2": "seed 2", "seed3": "seed 3"}
RANDOM = "ctrl-random"
MAX_STEP = 120
PROBES = {
    "test_id": "in-distribution",
    "test_book": "held-out sources",
    "probe_cross_domain": "other environmental domains",
}
PROBE_LABEL = {
    "test_id": "In-distribution",
    "test_core_only": "In-distribution, original problems",
    "test_book (unseen textbooks)": "Held-out sources",
    "trap parents (controls)": "Parent problems of traps",
    "traps, original 67": "Benchmark traps",
    "traps, expansion clean 135": "Benchmark traps, second set",
    "scaffolding, original": "Scaffolded problems",
    "scaffolding, stripped": "Scaffolding removed",
    "cross-domain (131)": "Other environmental domains",
}
RANDOM_LABEL = {
    "test_id": "In-distribution",
    "test_book": "Held-out sources",
    "traps, expansion clean": "Benchmark traps, second set",
    "scaffolding stripped": "Scaffolding removed",
    "cross-domain": "Other environmental domains",
}
ARMS = ["base", "warm", "A", "D", "F", "WA", "WD", "R"]
ARM_LABEL = {"base": "Base model", "warm": "SFT warm-up", "A": "Answer reward (A)", "D": "Counterfactual reward (D)", "F": "Answer + judgement (F)",
             "WA": "Warm-up, then A (WA)", "WD": "Warm-up, then D (WD)", "R": "Random reward"}


def clip(steps, values, max_step=MAX_STEP):
    keep = steps <= max_step
    return steps[keep], values[keep]


def main() -> None:
    style()
    fig = plt.figure(figsize=(DOUBLE, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.75], hspace=0.42)
    top = gs[0].subgridspec(1, 2, wspace=0.42)
    bottom = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.55)
    ax_b = fig.add_subplot(top[0])      # held-out curves (panel a)
    ax_c = fig.add_subplot(top[1])      # probe gains (panel b)
    ax_d = fig.add_subplot(bottom[0])   # traps per arm (panel c)
    ax_e = fig.add_subplot(bottom[1])   # comparison models (panel d)

    # --- b: validation probes during training -------------------------------------------
    colours = {"test_id": C["rl"], "test_book": C["dose"], "probe_cross_domain": C["distil"]}
    for probe, label in PROBES.items():
        # executed-and-passed share of the probe rows (the random-reward run's "reward" column is the coin flip)
        col = f"val-aux/atmoscoder/{probe}/outcome/passed/mean@1"
        series = []
        for run in RUNS:
            s, v = clip(*curve(f"curves/{run}.csv", col))
            series.append(dict(zip(s.tolist(), v.tolist())))
        steps = sorted(set.intersection(*(set(d) for d in series)))
        arr = np.array([[d[st] for st in steps] for d in series])
        ax_b.plot(steps, pct(arr.mean(0)), color=colours[probe], lw=1.0, label=label)
        ax_b.fill_between(steps, pct(arr.min(0)), pct(arr.max(0)), color=colours[probe], alpha=0.18, lw=0)
        s, v = clip(*curve(f"curves/{RANDOM}.csv", col))
        ax_b.plot(s, pct(v), color=colours[probe], lw=0.7, ls="--", alpha=0.8)
    ax_b.plot([], [], color="#555555", lw=0.7, ls="--", label="random reward (control)")
    ax_b.set_xlabel("Training step")
    ax_b.set_ylabel("Held-out accuracy (%)")
    ax_b.set_xlim(0, MAX_STEP)
    ax_b.set_ylim(0, 110)
    ax_b.set_yticks(range(0, 101, 20))
    ax_b.legend(loc="upper left", ncol=1, labelspacing=0.25)

    # --- c: gains on nine probes at step 120 ------------------------------------------------
    table = md_table("FINAL.md", "probe")
    header = table[0]
    rows = table[1:]
    names, base, runs = [], [], []
    for r in rows:
        names.append(r[0])
        base.append(num(r[2]))
        runs.append([num(x) for x in r[3:7]])
    runs = np.array(runs)
    gains = runs - np.array(base)[:, None]
    # FINAL.md table 2: probe | real | random (same checkpoint, reward replaced by a coin flip)
    rand = {}
    for t in md_tables("FINAL.md"):
        if t[0][:3] == ["probe", "real", "random"]:
            for r in t[1:]:
                rand[r[0]] = num(r[2])
    ypos = np.arange(len(names))[::-1]
    ax_c.barh(ypos, gains.mean(1), xerr=gains.std(1, ddof=1), color=C["rl"], height=0.6, error_kw=dict(elinewidth=0.7, ), label="answer reward")
    for i, n in enumerate(names):
        key = next((k for k in rand if RANDOM_LABEL.get(k) == PROBE_LABEL.get(n)), None)
        if key is not None:
            ax_c.plot(rand[key], ypos[i], marker="D", color="#555555", ms=3.5, ls="none", label="random reward" if i == 0 or key == list(rand)[0] else None)
    ax_c.axvline(0, color="#999999", lw=0.6, ls=":")
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([PROBE_LABEL.get(n, n) for n in names])
    ax_c.set_xlabel("Gain over base model\n(percentage points)")
    ax_c.set_xlim(-6, 34)
    handles, labels = ax_c.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    legend_above(ax_c, seen.values(), seen.keys())

    # --- d: benchmark regime traps after training ---------------------------------------------
    # points with intervals, one row per training variant, in the same style as panel d
    y = -np.arange(len(ARMS), dtype=float)
    for k, fam in enumerate(["qwen3-8b", "qwen35-9b"]):
        rep = load(f"traps_rl/report_{fam}.json")
        colour = C["rl"] if k == 0 else C["rl2"]
        for yi, arm in zip(y, ARMS):
            a = rep["arms"][arm]["all"]
            m = pct(a["trap_acc"])
            ax_d.errorbar(m, yi + (0.5 - k) * 0.3, xerr=ci_err(m, [pct(c) for c in a["trap_acc_ci95"]]), fmt="o", ms=3.8, elinewidth=0.8,
                          color=colour, label=FAMILY_LABEL[fam] if arm == "base" else None)
    # same harness, all 46 traps, three repetitions (PREREG_PAPER_ANALYSES §21)
    fr = pct(load("paper/frontier21.json")["configurations"]["frontier21-deepseek-v4-flash-off"]["textbook"]["trap"]["mean"])
    ax_d.axvline(fr, color=C["frontier"], lw=0.8, ls="-.")
    ax_d.text(fr + 1.5, y[-1] - 0.05, "DeepSeek\nV4 Flash", fontsize=5.5, ha="left", va="center", color=C["frontier"])
    ax_d.axhspan(y[1] - 0.5, y[0] + 0.5, color="#F2F2F2", lw=0, zorder=0)    # before reinforcement learning
    ax_d.axhspan(y[-1] - 0.5, y[-1] + 0.5, color="#F2F2F2", lw=0, zorder=0)  # control
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([ARM_LABEL[a] for a in ARMS], fontsize=6)
    ax_d.set_xlabel("Accuracy on 46 benchmark regime traps (%)")
    ax_d.set_xlim(0, 75)
    ax_d.set_ylim(y[-1] - 0.5, y[0] + 0.5)
    legend_above(ax_d, ncol=2)

    # --- d: trained 9B against frontier and open models, paired over traps ------------------------
    f21 = load("paper/frontier21.json")["configurations"]
    f22 = load("paper/frontier22.json")["configurations"]
    names21 = {"frontier21-gpt-5.5": "GPT-5.5", "frontier21-gpt-5.4": "GPT-5.4", "frontier21-deepseek-v4-flash": "DeepSeek V4 Flash", "frontier21-minimax-m3": "MiniMax-M3"}
    rows21, rows22 = [], []
    for key, v in f21.items():
        base_key, mode = key.rsplit("-", 1)
        d = v["textbook"]["vs"]["qwen35-9b-A"]["trap"]
        rows21.append((names21[base_key], mode == "on", d))
    e2 = load("paper/f23_e2.json")["configurations"]  # Qwen3.5-27B completed to all 46 traps
    for key, v in f22.items():
        fam, mode = key.split(".think-")
        full = e2.get(key, {}).get("present")
        d = (e2[key] if full else v)["textbook"]["vs"]["qwen35-9b-A"]["trap"]
        rows22.append((CONFIG_LABEL[f"{fam}.think-off"], mode == "on", d))
    blocks = [("API models", sorted(rows21, key=lambda r: r[2]["mean"])),
              ("Open models", sorted(rows22, key=lambda r: r[2]["mean"]))]
    y, ticks, labels, heads = 0, [], [], []
    for title, rows in blocks:
        ticks.append(y)
        labels.append(title)
        heads.append(len(labels) - 1)
        y += 1
        for name, think, d in rows:
            m = pct(d["mean"])
            ax_e.errorbar(m, y, xerr=ci_err(m, [pct(c) for c in d["ci95"]]), fmt="o", ms=3.2, elinewidth=0.7,
                          color=C["frontier"], mfc="white" if think else C["frontier"], mew=0.8)
            ticks.append(y)
            labels.append(name + (" (thinking)" if think else ""))
            y += 1
        y += 0.5
    ax_e.axvline(0, color=RULE, lw=0.6, ls=":")
    ax_e.set_yticks(ticks)
    ax_e.set_yticklabels(labels, fontsize=6)
    for k in heads:
        ax_e.get_yticklabels()[k].set_fontweight("bold")
        ax_e.yaxis.get_major_ticks()[k].tick1line.set_visible(False)
    ax_e.set_ylim(y - 0.4, -1.0)
    ax_e.set_xlim(-58, 58)
    ax_e.set_xlabel("Trained Qwen3.5-9B minus comparison model\n(percentage points)")

    space_panels(fig)
    align_columns(fig, [[ax_b, ax_d], [ax_c, ax_e]])
    label_panels(fig, [[(ax_b, "a"), (ax_c, "b")], [(ax_d, "c"), (ax_e, "d")]])
    # Which study each row comes from, next to the row's first letter.
    for letter, heading in [("a", "Benchmark-problem study, Qwen3-8B"), ("c", "Template study")]:
        lt = next(tx for tx in fig.texts if tx.get_text() == letter)
        x, y = lt.get_position()
        fig.text(x + 0.022, y, heading, fontsize=7, color="#555555", va="bottom", ha="left")
    save(fig, "fig4")


if __name__ == "__main__":
    main()
