"""Emit paper/numbers.tex (script: paper/fig/make_numbers.py): every number quoted in the manuscript as a LaTeX macro, read from
results/. Run from the repository root: ``uv run python paper/fig/make_numbers.py > paper/numbers.tex``.

Each macro is one committed quantity; the comment after it names the source file and key, so a
reader can trace any sentence of the paper to the file that computed it. Nothing is typed in.
"""

from __future__ import annotations

import glob
import json
import re

from common import RESULTS, d_contrasts, load, md_tables, num

OUT: list[tuple[str, str, str]] = []  # (macro, value, source)


_DIGIT = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def put(name: str, value, source: str, fmt: str = "{:.1f}") -> None:
    """LaTeX macro names cannot contain digits, so 'Bin0' becomes 'BinZero'."""
    name = "".join(_DIGIT.get(ch, ch) for ch in name)
    if isinstance(value, float):
        value = fmt.format(value)
    OUT.append((name, str(value), source))


def ci(name: str, mean: float, lo: float, hi: float, source: str, scale: float = 100.0, fmt: str = "{:.1f}") -> None:
    put(name, mean * scale, source, fmt)
    put(name + "Lo", lo * scale, source, fmt)
    put(name + "Hi", hi * scale, source, fmt)


def main() -> None:
    # ---- diagnosis (D) --------------------------------------------------------------------------
    cons = d_contrasts()  # 16 local configurations, 69 templates (PREREG_PAPER_ANALYSES §19)
    io = [v["inside_minus_outside"]["mean"] for v in cons.values()]
    put("nLocalConfigs", len(cons), "paper/d_reword.json configurations")
    rw = load("paper/d_reword.json")
    put("nTemplatesDiag", 72 - len(rw["dropped"]), "72 minus paper/d_reword.json dropped")
    put("nClustersDiag", rw["configurations"][0]["templates_69"]["clusters"], "d_reword.json templates_69.clusters")
    put("rewordMaxChange", 100 * rw["max_abs_inside_minus_outside_change"], "d_reword.json max_abs_inside_minus_outside_change", "{:.1f}")
    d5 = load("d5/report.json")["D5-1"]
    ci("dFiveGap", d5["mean"], *d5["ci95"], "d5/report.json D5-1 (16 configurations pooled, 54 independent traps)")
    put("dOneTemplateMet", sum(1 for v in cons.values() if v["inside_minus_outside"]["mean"] >= 0.20 and v["inside_minus_outside"]["ci95"][0] > 0),
        "d_reword.json templates_69: inside − outside ≥ 20 points with lower bound > 0")
    x1 = (RESULTS / "x1" / "README.md").read_text(encoding="utf-8")
    m = re.search(r"共 (\d+) 行全部读过，(\d+) 行属实", x1)
    put("traceAuditPrecN", int(m.group(1)), "x1/README.md precision audit: traces the judge marked positive")
    put("traceAuditPrecOk", int(m.group(2)), "idem, confirmed on reading")
    m = re.search(r"合计（(\d+) 行） \| \*\*(\d+)（", x1)
    put("traceAuditN", int(m.group(1)), "x1/README.md recall audit: wrong outside traces read (Qwen3-8B and Gemma 4 31B, thinking on)")
    put("traceAuditExplicit", int(m.group(2)), "idem, traces stating explicitly that the standard relation does not apply")
    m = re.search(r"明确说了的 (\d+) 行里 \*\*(\d+) 行最后仍交", x1)
    put("traceAuditExplicitUsed", int(m.group(2)), "idem, of those, traces that still submitted the standard-relation value")
    runs = load("paper/gpu_hours.json")["runs"]
    for key, fam in [("Qwen", "x-qwen3-8b"), ("QwenB", "x-qwen35-9b")]:
        w = [r["wall_h"] for r in runs if r["run"].startswith(fam) and "-R-" in r["run"]]
        put(f"rArmHours{key}Min", min(w), "paper/gpu_hours.json random-reward runs, wall-clock hours", "{:.0f}")
        put(f"rArmHours{key}Max", max(w), "idem", "{:.0f}")
    s3 = {k: v["S3_given_minus_inside"] for k, v in load("d2/report.json")["configurations"].items()}
    within = {k: (v["ci95"][0] > -0.10 and v["ci95"][1] < 0.10) for k, v in s3.items()}
    put("sThreeN", len(s3), "d2/report.json configurations with S3_given_minus_inside (72 templates)")
    put("sThreeMet", sum(within.values()), "idem, 95% interval within ±10 points (PREREG_MAIN S3)")
    put("sThreeFailAbove", sum(1 for k, v in s3.items() if v["ci95"][1] >= 0.10), "idem, interval reaching above +10 points")
    put("sThreeFailBelow", sum(1 for k, v in s3.items() if v["ci95"][0] <= -0.10), "idem, interval reaching below -10 points")
    x7 = load("x7/x7_report_strict.json")
    for key, name in [("Qwen", "qwen3-8b"), ("QwenB", "qwen35-9b"), ("DS", "deepseek")]:
        put(f"judgeSaidNo{key}", 100 * x7[name]["said_no_rate"], f"x7_report_strict.json {name} said_no_rate (first-round rows judged invalid)")
    for key, name in [("QwenB", "qwen35-9b"), ("DS", "deepseek")]:
        v = x7[name]["T_minus_C_said_no"]
        ci(f"judgeTC{key}", v["mean"], *v["ci95"], f"x7_report_strict.json {name} T_minus_C_said_no (rows judged invalid, cluster resample)")
    reg = load("paper/cost.json")["traps"]["by_trap_type"]["regime_boundary"]
    put("costRegimeN", reg["n"], "paper/cost.json by_trap_type.regime_boundary.n")
    put("costRegimeDefined", reg["defined"], "idem defined")
    put("costRegimeMedian", 100 * reg["R_quartiles"][1], "idem R_quartiles[1]", "{:.0f}")
    put("costRegimeMedianLo", 100 * reg["R_median_ci95"][0], "idem R_median_ci95", "{:.0f}")
    put("costRegimeMedianHi", 100 * reg["R_median_ci95"][1], "idem", "{:.0f}")
    put("costRegimeShare", 100 * reg["share_R_gt_50"], "idem share_R_gt_50", "{:.0f}")
    gh = load("paper/gpu_hours.json")  # explore/gpu_hours.py
    put("computeRuns", gh["total_runs"], "paper/gpu_hours.json total_runs")
    put("computeGPUh", f"{gh['total_gpu_h']:,}", "paper/gpu_hours.json total_gpu_h")
    for tag, keys in [("Template", ["X", "X Qwen3.5-9B seed 43", "X R arm"]), ("Example", ["DOSE", "DOSE-text", "Ministral DOSE"]), ("Distil", ["OPD"])]:
        put(f"computeRuns{tag}", sum(gh["by_study"][k]["runs"] for k in keys), "paper/gpu_hours.json by_study runs: " + ", ".join(keys))
        put(f"computeGPUh{tag}", f"{sum(gh['by_study'][k]['gpu_h'] for k in keys):,}", "idem gpu_h")
    put("insideMinusOutsideMin", 100 * min(io), "contrasts.inside_minus_outside.mean over local configs", "{:.0f}")
    put("insideMinusOutsideMax", 100 * max(io), "idem", "{:.0f}")
    put("insideMinusOutsideAllPositive", int(all(v["inside_minus_outside"]["ci95"][0] > 0 for v in cons.values())), "lower CI bounds > 0")
    go = [v["given_minus_outside"]["mean"] - v["inside_minus_outside"]["mean"] for v in cons.values()]
    put("givenMinusInsideMin", 100 * min(go), "given − inside over local configs", "{:.0f}")
    put("givenMinusInsideMax", 100 * max(go), "idem", "{:.0f}")
    fo = [v["flag_minus_outside"]["mean"] for v in cons.values()]
    put("flagMinusOutsideMin", 100 * min(fo), "flag − outside over local configs", "{:.1f}")
    put("flagMinusOutsideMax", 100 * max(fo), "idem", "{:.1f}")
    fr = load("d2/frontier_deepseek-v4-flash-off.json")
    c = fr["contrasts"]["inside_minus_outside"]
    ci("frontierInsideMinusOutside", c["mean"], *c["ci95"], "d2/frontier_deepseek-v4-flash-off.json contrasts")
    s1 = [v["S1_capture_outside_wrong"]["mean"] for v in load("d2/report.json")["configurations"].values() if v["S1_capture_outside_wrong"]["mean"] is not None]
    put("shortcutShareMin", 100 * min(s1), "d2/report.json S1_capture_outside_wrong", "{:.0f}")
    put("shortcutShareMax", 100 * max(s1), "idem", "{:.0f}")
    put("frontierShortcutShare", 100 * fr["S1_capture_outside_wrong"]["mean"], "frontier S1", "{:.0f}")

    # ---- cost (traps without a magnitude defect; the set the ledger F2 reports) -------------------------
    cost = load("paper/cost.json")["traps"]["all_without_magnitude_defect"]
    put("nTrapsCost", cost["n"], "paper/cost.json traps.all_without_magnitude_defect.n")
    put("nTrapsCostDefined", cost["defined"], "idem .defined (deviation defined: not a sign flip, not undefined)")
    put("nTrapsCostSignFlip", cost["sign_flip"], "idem .sign_flip")
    put("costMedianDeviation", 100 * cost["R_quartiles"][1], "idem R_quartiles[1]", "{:.0f}")
    put("costMedianDeviationLo", 100 * cost["R_median_ci95"][0], "idem R_median_ci95", "{:.0f}")
    put("costMedianDeviationHi", 100 * cost["R_median_ci95"][1], "idem R_median_ci95", "{:.0f}")
    ci("costShareGtHalf", cost["share_R_gt_50"], *cost["share_R_gt_50_ci95"], "idem share_R_gt_50", fmt="{:.0f}")
    put("costShareGeTwo", 100 * cost["share_F_ge_2"], "idem share_F_ge_2", "{:.0f}")
    put("costCountGeTwo", cost["count_F_ge_2"], "idem count_F_ge_2")
    put("costCountGtTen", cost["bins"][">10x"], "idem bins['>10x']")
    lb = load("paper/cost.json")["templates"]["per_template"]["line_broadening"]["F_max"]
    put("lineBroadeningFactor", lb, "templates.per_template.line_broadening.F_max", "{:.0f}")
    # PREREG_PAPER_ANALYSES §23: universal failures and anchor instances (template functions, no model output)
    me = load("paper/mechanism_examples.json")
    # PREREG_PAPER_ANALYSES §24: external-teacher probes (PREREG_OPD §4.2); arm T was gated on them and not run
    tp = load("paper/teacher_probe.json")
    put("teacherGate", tp["gate"], "paper/teacher_probe.json gate (zero-hit mechanisms a teacher had to know)")
    put("teacherProbeZ", tp["teachers"]["qwen3-32b"]["zero_hit_mechanisms"], "idem teachers.qwen3-32b.zero_hit_mechanisms (14 zero-hit mechanisms of Qwen3-8B)")
    put("teacherKnownLarge", tp["teachers"]["qwen3-32b"]["n_zero_hit_known"], "idem teachers.qwen3-32b.n_zero_hit_known")
    put("teacherKnownMoe", tp["teachers"]["qwen3-30b-a3b-2507"]["n_zero_hit_known"], "idem teachers.qwen3-30b-a3b-2507.n_zero_hit_known")
    put("nUniversalFail", me["n_universal_failures"], "paper/mechanism_examples.json n_universal_failures (fail in every kept local configuration)")
    co2 = me["anchors"]["co2_forcing_log"][0]
    put("coTwoDoublingLinear", co2["shortcut"], "idem anchors.co2_forcing_log (dc = 280 ppm) shortcut, W m^-2", "{:.2f}")
    put("coTwoDoublingLog", co2["correct"], "idem correct", "{:.2f}")
    sp = me["anchors"]["sun_photometer_airmass"][0]
    put("sunPhotometerDeviation", 100 * sp["relative_deviation"], "idem anchors.sun_photometer_airmass (theta = 87 deg) relative deviation of optical depth", "{:.0f}")

    # ---- atlas ------------------------------------------------------------------------------------
    at = load("paper/atlas.json")
    put("atlasTemplateShare", 100 * float(at["A1_variance"]["template_share"]), "atlas.json A1_variance.template_share", "{:.0f}")
    put("atlasPermP", float(at["A1_variance"]["perm_p"]), "A1_variance.perm_p", "{:.4f}")
    for key, tag in [("category", "Category"), ("type", "Type"), ("cluster", "Cluster")]:
        v = at["A2_organisation"][key]
        put(f"atlasIcc{tag}", v["icc"], f"atlas.json A2_organisation.{key}.icc", "{:.2f}")
        put(f"atlasIcc{tag}P", v["perm_p"], f"idem perm_p", "{:.2f}")
        put(f"atlasGroups{tag}", v["groups"], "idem groups")

    # ---- SI: dynamic-sampling near-ablation and error magnitude (synthesis round 3) ---------------------
    r3 = load("paper/synthesis_round3.json")
    r1 = r3["R1_dynamic_sampling_near_ablation"]
    put("dynUnits", r1["units"], "synthesis_round3.json R1 units (templates learned by A)")
    for arm in ["A", "S", "P"]:
        if arm in r1["per_arm"]:
            put(f"dyn{arm}Learned", r1["per_arm"][arm]["learned"], f"R1 per_arm.{arm}.learned")
            put(f"dyn{arm}Mean", 100 * r1["per_arm"][arm]["mean"], f"R1 per_arm.{arm}.mean")
    for k, v in r1.items():
        if k.startswith("A-S") or k == "A_minus_S":
            if isinstance(v, dict) and "mean" in v:
                ci("dynAminusS", v["mean"], *v["ci95"], f"R1 {k}")
    r2 = r3["R2_error_magnitude"]
    for arm in ["base", "A", "P"]:
        if arm in r2:
            put(f"errMedian{arm}", r2[arm]["median"], f"R2_error_magnitude.{arm}.median (relative error of wrong outside answers)", "{:.2f}")

    # ---- knowing-doing, interventions ------------------------------------------------------------------
    x1 = load("x1/x1a.json")["deepseek-v4-flash.think-off"]
    put("judgedNoShortcut", 100 * x1["shortcut_among_judged0_wrong"]["mean"], "x1/x1a.json shortcut_among_judged0_wrong", "{:.0f}")
    put("judgedYesShortcut", 100 * x1["shortcut_among_judged1_wrong"]["mean"], "shortcut_among_judged1_wrong", "{:.0f}")
    x8 = load("x8/report.json")["qwen3-8b-off"]["E-C_outside"]
    ci("recallQwen", x8["mean"], *x8["ci95"], "x8/report.json qwen3-8b-off E-C_outside")
    hb = re.search(r"handbook-none \+([\d.]+)% \[\+([\d.]+)%, \+([\d.]+)%\]", (RESULTS / "e7" / "report.txt").read_text())
    put("handbookGain", float(hb.group(1)), "e7/report.txt handbook-none")
    put("handbookGainLo", float(hb.group(2)), "idem")
    put("handbookGainHi", float(hb.group(3)), "idem")
    g = re.search(r"outside acc\s+none (\d+)/(\d+)=(\d+)%\s+handbook (\d+)/(\d+)=(\d+)%\s+given (\d+)/(\d+)=(\d+)%", (RESULTS / "e7" / "report.txt").read_text())
    put("handbookNone", g.group(3), "e7/report.txt outside acc none")
    put("handbookWith", g.group(6), "idem handbook")
    put("handbookGiven", g.group(9), "idem given")

    d3 = load("d3/report.json")["deepseek-v4-flash.think-off"]["templates"]
    put("detectH", 100 * d3["hit_rate"]["mean"], "d3/report.json deepseek-v4-flash.think-off templates.hit_rate (H)", "{:.0f}")
    put("detectF", 100 * d3["false_alarm_rate"]["mean"], "idem false_alarm_rate (F)", "{:.0f}")
    ci("detectHminusF", d3["discrimination"]["mean"], *d3["discrimination"]["ci95"], "idem discrimination (H − F)")
    put("detectParse", 100 * d3["parse_rate"], "idem parse_rate", "{:.0f}")
    put("judgeRows", load("x7/x7_report_strict.json")["deepseek"]["rows"], "x7/x7_report_strict.json deepseek rows")

    # ---- domain RL (FINAL.md) ------------------------------------------------------------------------------
    tables = md_tables("FINAL.md")
    t1 = next(t for t in tables if t[0][0] == "probe" and "base" in t[0])
    rows = {r[0]: r for r in t1[1:]}
    def gain(name):
        r = rows[name]
        base = num(r[2]); runs = [num(x) for x in r[3:7]]
        return sum(runs) / 4 - base
    put("rlGainTestId", gain("test_id"), "FINAL.md table 1 test_id mean Δ")
    put("rlGainTestBook", gain("test_book (unseen textbooks)"), "FINAL.md test_book")
    put("rlGainCrossDomain", gain("cross-domain (131)"), "FINAL.md cross-domain")
    put("rlGainScaffoldStripped", gain("scaffolding, stripped"), "FINAL.md scaffolding stripped")
    t2 = next(t for t in tables if t[0][:3] == ["probe", "real", "random"])
    rnd = [num(r[2]) for r in t2[1:]]
    put("randomRewardMin", min(rnd), "FINAL.md table 2 random column (five probes shown)")
    put("randomRewardMax", max(rnd), "idem")
    put("nRandomProbesShown", len(rnd), "idem")
    nine = re.search(r"All nine probes fall between (−?-?[\d.]+) and \+?([\d.]+)", (RESULTS / "FINAL.md").read_text(encoding="utf-8"))
    put("randomRewardNineMin", num(nine.group(1)), "FINAL.md §2 prose, all nine probes")
    put("randomRewardNineMax", num(nine.group(2)), "idem")
    put("nProbes", len(rows), "FINAL.md table 1 rows")
    put("nProbeRuns", len(rows) * 4, "FINAL.md table 1: probes × four runs")
    put("nCrossDomain", int(re.search(r"cross-domain \((\d+)\)", " ".join(rows)).group(1)), "FINAL.md table 1 row label cross-domain (n)")

    # ---- held-out fold transfer (template study) ----------------------------------------------------------------
    for fam, tag, path in [("qwen3-8b", "Qwen", "report_qwen3-8b.json"), ("qwen35-9b", "QwenB", "report_qwen35-9b_s42-43.json")]:
        tr = load(f"x/{path}")["transfer"]
        for arm in ["A", "D", "F", "WA", "WD"]:
            t = tr[arm]
            put(f"heldStart{tag}{arm}", 100 * t["start"]["mean"], f"x/{path} transfer.{arm}.start")
            put(f"heldEnd{tag}{arm}", 100 * t["held_out"]["mean"], f"idem held_out")
            ci(f"heldGain{tag}{arm}", t["endpoint-start"]["mean"], *t["endpoint-start"]["ci95"], f"idem endpoint-start")
        gains = [100 * tr[a]["endpoint-start"]["mean"] for a in ["A", "D", "F", "WA", "WD"]]
        put(f"heldGain{tag}Min", min(gains), f"x/{path} transfer, min over arms", "{:.0f}")
        put(f"heldGain{tag}Max", max(gains), "idem max", "{:.0f}")
    asym = load("paper/rigor_eval_qwen3-8b.json")["asymmetry_held_out"]["A"]["asymmetry"]
    ci("asymHeldQwenA", asym["mean"], *asym["ci95"], "rigor_eval_qwen3-8b.json asymmetry_held_out.A (outside gain − inside gain)")

    # ---- textbook traps after RL -------------------------------------------------------------------------------
    for fam, tag in [("qwen3-8b", "Qwen"), ("qwen35-9b", "QwenB")]:
        rep = load(f"traps_rl/report_{fam}.json")
        put(f"trapsBase{tag}", 100 * rep["arms"]["base"]["all"]["trap_acc"], f"traps_rl/report_{fam}.json arms.base")
        a = rep["arms"]["A"]["all"]
        put(f"trapsA{tag}", 100 * a["trap_acc"], "arms.A.all.trap_acc")
        put(f"trapsA{tag}Lo", 100 * a["trap_acc_ci95"][0], "arms.A.all.trap_acc_ci95")
        put(f"trapsA{tag}Hi", 100 * a["trap_acc_ci95"][1], "idem")
        gso = a["gain_over_start"]
        ci(f"trapsGain{tag}", gso["mean"], *gso["ci95"], "arms.A.all.gain_over_start")
        put(f"trapsR{tag}", 100 * rep["arms"]["R"]["all"]["trap_acc"], "arms.R.all.trap_acc")
    p2s = load("x/report_qwen35-9b.json")["P2"]["F-A_H-F_held"]
    for key, path in [("QwenBSeed", "x/report_qwen35-9b.json"), ("QwenB", "x/report_qwen35-9b_s42-43.json")]:
        t = load(path)["transfer"]
        put(f"transferArmsMet{key}", sum(1 for arm in ["A", "D", "F", "WA", "WD"] if t[arm]["endpoint-start"]["mean"] >= 0.15 and t[arm]["endpoint-start"]["ci95"][0] > 0), f"{path} transfer: arms with held-out outside gain ≥ 15 points and lower bound > 0")
    ci("pTwoFminusAQwenBSeed", p2s["mean"], *p2s["ci95"], "x/report_qwen35-9b.json P2.F-A_H-F_held (pre-registered seed-42 analysis; pTwoFminusAQwenB pools seeds 42 and 43)")
    # Frontier and open models against the trained 9B in the same harness (PREREG_PAPER_ANALYSES §21, §22)
    f21 = load("paper/frontier21.json")["configurations"]
    ds = f21["frontier21-deepseek-v4-flash-off"]["textbook"]
    put("frontierN", ds["traps_covered"], "paper/frontier21.json textbook.traps_covered (all 46 regime traps, three repetitions)")
    put("frontierAQwenB", 100 * ds["vs"]["qwen35-9b-A"]["arm_trap"], "idem vs.qwen35-9b-A.arm_trap (arm A, four endpoints)")
    put("frontierDSOff", 100 * ds["trap"]["mean"], "frontier21-deepseek-v4-flash-off textbook.trap.mean")
    ci("frontierDiffDS", ds["vs"]["qwen35-9b-A"]["trap"]["mean"], *ds["vs"]["qwen35-9b-A"]["trap"]["ci95"], "idem vs.qwen35-9b-A.trap (A − DeepSeek, bootstrap over traps)")
    ci("frontierDSTrapMinusParent", ds["trap_minus_parent"]["mean"], *ds["trap_minus_parent"]["ci95"], "idem trap_minus_parent")
    put("frontierGPT55Off", 100 * f21["frontier21-gpt-5.5-off"]["textbook"]["trap"]["mean"], "frontier21-gpt-5.5-off textbook.trap.mean")
    put("frontierGPT54Off", 100 * f21["frontier21-gpt-5.4-off"]["textbook"]["trap"]["mean"], "frontier21-gpt-5.4-off textbook.trap.mean")
    put("frontierMiniMaxOff", 100 * f21["frontier21-minimax-m3-off"]["textbook"]["trap"]["mean"], "frontier21-minimax-m3-off textbook.trap.mean")
    gpt_tmp = [100 * f21[k]["textbook"]["trap_minus_parent"]["mean"] for k in ("frontier21-gpt-5.5-off", "frontier21-gpt-5.4-off")]
    put("frontierGPTTrapMinusParentMax", max(gpt_tmp), "largest trap − parent among the two GPT thinking-off configurations", "{:.1f}")
    on = [100 * v["textbook"]["trap"]["mean"] for k, v in f21.items() if v["thinking"] == "on"]
    put("frontierOnMin", min(on), "frontier21 thinking-on configurations, textbook.trap.mean, smallest", "{:.0f}")
    put("frontierOnMax", max(on), "idem, largest", "{:.0f}")
    gaps = [100 * v["templates"]["inside_minus_outside"]["mean"] for k, v in f21.items() if v["thinking"] == "off" and "deepseek" not in k]
    put("frontierTemplateGapMin", min(gaps), "frontier21 GPT-5.5, GPT-5.4, MiniMax-M3 thinking off, templates.inside_minus_outside.mean (final-set templates, 32 clusters), smallest", "{:.0f}")
    put("frontierTemplateGapMax", max(gaps), "idem, largest", "{:.0f}")
    ds_tmpl = f21["frontier21-deepseek-v4-flash-off"]["templates"]
    put("frontierDSHeldOutside", 100 * ds_tmpl["outside"]["mean"], "frontier21-deepseek-v4-flash-off templates.outside.mean (final-set templates)")
    put("heldOutsideQwenBA", 100 * ds_tmpl["vs_9b_heldout"]["arm_outside"]["mean"], "idem templates.vs_9b_heldout.arm_outside.mean (arm A, held-out fold; equals results/x transfer.A.held_out)")
    f22 = load("paper/frontier22.json")
    put("sharedN", f22["shared_traps"], "paper/frontier22.json shared_traps (regime traps in both the X and the D benchmark sets)")
    put("frontierAQwenBShared", 100 * f22["arms_on_shared"]["qwen35-9b-A"]["trap"], "idem arms_on_shared.qwen35-9b-A.trap")
    for key, tag in [("qwen3.5-27b.think-off", "QwenLargeOff"), ("gemma4-31b.think-off", "GemmaLargeOff")]:
        tb = f22["configurations"][key]["textbook"]
        put(f"shared{tag}", 100 * tb["trap"]["mean"], f"frontier22 {key} textbook.trap.mean (38 traps, three repetitions)")
        ci(f"sharedDiff{tag}", tb["vs"]["qwen35-9b-A"]["trap"]["mean"], *tb["vs"]["qwen35-9b-A"]["trap"]["ci95"], f"idem vs.qwen35-9b-A.trap (A − configuration)")

    g27 = load("paper/f25_gap.json")["families"]["qwen35-9b"]["qwen3.5-27b.think-off"]
    put("gapClosedLarge", 100 * g27["G"], "paper/f25_gap.json families.qwen35-9b.qwen3.5-27b.think-off.G (arm-A gain / gap of the 27B model over the 9B base, 46 traps)", "{:.0f}")
    e1 = load("paper/f23_e1.json")["arms"]["qwen3-8b:A"]["all"]
    put("indepRegimeN", e1["S23a_regime_gain"]["units"], "paper/f23_e1.json arms.qwen3-8b:A regime traps of the independent set")
    put("indepGeometryN", e1["geometry_gain"]["units"], "idem geometry traps")
    put("indepParentN", e1["parent_gain"]["units"], "idem parent problems")
    ci("indepRegimeGain", e1["S23a_regime_gain"]["mean"], *e1["S23a_regime_gain"]["ci95"], "idem S23a_regime_gain (arm A − base, fold 1 seed 42)")

    # ---- reachability (P1) ------------------------------------------------------------------------------------------
    p1 = load("x/report_qwen3-8b.json")["P1"]
    put("nZeroHit", len(p1["Z"]), "x/report_qwen3-8b.json P1.Z")
    zl = set().union(*(set(v) for v in p1["Z_at_or_above_50"]["A"].values()))
    put("nZeroHitLearnedA", len(zl), "P1.Z_at_or_above_50.A (any seed)")
    put("nZeroHitNotLearnedA", len(p1["Z"]) - len(zl), "idem")
    zany = set().union(*(set(v) for arm in p1["Z_at_or_above_50"].values() for v in arm.values()))
    put("nZeroHitLearnedAny", len(zany), "P1.Z_at_or_above_50 over arms A, D, F and both seeds")
    put("nZeroHitNotLearnedAny", len(p1["Z"]) - len(zany), "idem")
    p1b = load("x/report_qwen35-9b.json")["P1"]
    put("nZeroHitQwenB", len(p1b["Z"]), "x/report_qwen35-9b.json P1.Z (seed 42)")
    zlb = set().union(*(set(v) for arm in p1b["Z_at_or_above_50"].values() for v in arm.values()))
    zeb = set.intersection(*(set(v) for v in p1b["Z_at_or_above_50_every_seed"].values()))
    put("nZeroHitLearnedQwenB", len(zeb), "P1.Z_at_or_above_50_every_seed, mechanisms learned by every arm")
    assert zlb == zeb, "Qwen3.5-9B zero-hit mechanisms learned by any arm must equal those learned by every arm for the sentence to hold"
    put("nZeroHitPooled", len(p1["Z"]) + len(p1b["Z"]), "P1.Z of both families, summed")
    put("nZeroHitLearnedPooled", len(zany) + len(zlb), "P1.Z_at_or_above_50 of both families (any arm, any seed), summed")
    x8b = load("x8/report.json")["qwen3.5-9b-off"]["E-C_outside"]
    ci("recallQwenB", x8b["mean"], *x8b["ci95"], "x8/report.json qwen3.5-9b-off E-C_outside")
    rg = load("paper/rigor.json")["p1"]["qwen3-8b"]
    auc = rg["auc_pass64"]
    put("p1AUC", auc["auc"] if "auc" in auc else auc["mean"], "paper/rigor.json p1.qwen3-8b.auc_pass64", "{:.2f}")
    if "ci95" in auc:
        put("p1AUCLo", auc["ci95"][0], "idem ci95", "{:.2f}")
        put("p1AUCHi", auc["ci95"][1], "idem ci95", "{:.2f}")
    mant, exp = f"{rg['fisher_p_two_sided']:.1e}".split("e")
    put("p1FisherP", f"{mant} \\times 10^{{{int(exp)}}}", "paper/rigor.json p1.qwen3-8b.fisher_p_two_sided")
    pc = load("paper/pcurve.json")["families"]
    for fam, tag in [("qwen3-8b", "Qwen"), ("qwen35-9b", "QwenB")]:
        for i, r in enumerate(pc[fam]["bins"]):
            put(f"pcurve{tag}Bin{i}Learned", r["learned"], f"paper/pcurve.json {fam} bins[{i}]")
            put(f"pcurve{tag}Bin{i}N", r["n"], "idem")
    sw = load("paper/rigor_eval_qwen3-8b.json")["shortcut_share_wrong_only_0.02"]
    ci("shortcutHeldA", sw["A:held"]["mean"], *sw["A:held"]["ci95"], "rigor_eval_qwen3-8b.json shortcut_share_wrong_only_0.02 A:held")
    ci("shortcutHeldR", sw["R:held"]["mean"], *sw["R:held"]["ci95"], "idem R:held")
    put("shortcutBaseLevel", 100 * sw["base:held+train:level"]["mean"], "idem base level", "{:.0f}")
    s1 = load("paper/synthesis.json")["S1_error_anatomy"]
    for k in ["base", "A", "P", "Pshuf"]:
        put(f"shortcutLevel{k}", 100 * s1["levels"][k]["shortcut"]["mean"], f"synthesis.json S1 levels.{k}.shortcut")
    c = s1["contrasts_shortcut"]["A-P"]
    ci("shortcutAminusP", c["mean"], *c["ci95"], "S1 contrasts_shortcut A-P")

    # ---- DOSE ------------------------------------------------------------------------------------------------------
    d = load("dose/report_qwen3-8b.json")
    ci("doseH1", d["H1"]["value"], *d["H1"]["ci95"], "dose/report_qwen3-8b.json H1")
    put("doseHOneRounded", round(100 * d["H1"]["value"]), "idem H1.value, rounded to a whole point (abstract)")
    put("doseMechanisms", d["H1"]["mechanisms"], "dose/report_qwen3-8b.json H1.mechanisms")
    put("doseObservations", d["H1"]["observations"], "idem H1.observations")
    obs = d["observations"]
    put("doseQwenDosedLearned", sum(1 for o in obs if o["dose"] > 0 and o["rl"] >= 0.5), "dose/report_qwen3-8b.json observations: dose ≥ 1 and rl ≥ 0.5")
    rw = d["secondary"]["rl-warm"]["all_dosed"]
    ci("doseRLminusWarm", rw["value"], *rw["ci95"], "dose/report_qwen3-8b.json secondary.rl-warm.all_dosed (trained minus warm-up outside accuracy, dosed observations)")
    put("doseWarmOnlyLearned", sum(1 for o in obs if o["dose"] > 0 and o["warm"] >= 0.5), "dose/report_qwen3-8b.json observations: dose ≥ 1 and warm ≥ 0.5")
    put("doseQwenDosedN", sum(1 for o in obs if o["dose"] > 0), "idem dose ≥ 1")
    put("doseQwenZeroLearned", sum(1 for o in obs if o["dose"] == 0 and o["rl"] >= 0.5), "idem dose = 0 and rl ≥ 0.5")
    put("doseQwenZeroN", sum(1 for o in obs if o["dose"] == 0), "idem dose = 0")
    put("doseOneLearned", sum(1 for o in obs if o["dose"] == 1 and o["rl"] >= 0.5), "idem dose = 1 and rl ≥ 0.5")
    put("doseOneN", sum(1 for o in obs if o["dose"] == 1), "idem dose = 1")
    pooled = load("dose_ministral/family_compare.json")["p_bins"]["pooled_dosed"]["[0.02, 2.0)"]
    put("pooledHighPLearned", pooled["learned"], "family_compare.json p_bins.pooled_dosed['[0.02, 2.0)'].learned")
    put("pooledHighPN", pooled["n"], "idem n")
    rr = load("paper/rigor.json")["dose_randomisation"]
    put("doseRandP", rr["randomisation_p"], "paper/rigor.json dose_randomisation.randomisation_p", "{:.3f}")
    fc = load("dose_ministral/family_compare.json")
    for fam, tag in [("qwen3-8b", "Qwen"), ("ministral3-8b", "Min")]:
        for dose in ["0", "1", "2", "4"]:
            v = fc["dose_levels"][fam][dose]
            put(f"dose{tag}D{dose}Learned", v["learned"], f"family_compare.json dose_levels.{fam}.{dose}")
            put(f"dose{tag}D{dose}N", v["n"], "idem")
            put(f"dose{tag}D{dose}Warm", 100 * v["warm_mean"], "idem warm_mean")
            put(f"dose{tag}D{dose}RL", 100 * v["rl_mean"], "idem rl_mean")
    dosed = [fc["dose_levels"]["ministral3-8b"][d] for d in ("1", "2", "4")]
    put("doseMinDosedLearned", 100 * sum(v["learned"] for v in dosed) / sum(v["n"] for v in dosed), "family_compare dose_levels ministral3-8b, dose ≥ 1 learned / n", "{:.0f}")
    m = fc["families"]["ministral3-8b"]["H1"]
    ci("doseMinH1", m["value"], *m["ci95"], "family_compare.json families.ministral3-8b.H1")
    dt = load("dose_text/dose_text_qwen3-8b.json")["H1_text"]
    ci("doseTextH1", dt["value"], *dt["ci95"], "dose_text_qwen3-8b.json H1_text")

    # ---- OPD -------------------------------------------------------------------------------------------------------------
    o = load("opd/report_qwen3-8b_seeds42-43.json")["main_outcome"]
    ci("opdP", o["per_arm"]["main_set"]["P"]["mean"], *o["per_arm"]["main_set"]["P"]["ci95"], "opd/report_qwen3-8b_seeds42-43.json per_arm.main_set.P")
    put("opdPRounded", round(100 * o["per_arm"]["main_set"]["P"]["mean"]), "idem, rounded to a whole percent (abstract)")
    put("opdS", 100 * o["per_arm"]["main_set"]["S"]["mean"], "per_arm.main_set.S")
    e = load("opd/report_ext_qwen3-8b.json")["main_outcome"]
    put("opdPshuf", 100 * e["per_arm"]["main_set"]["Pshuf"]["mean"], "report_ext per_arm.main_set.Pshuf")
    ci("opdPshufAll", e["contrasts"]["all"]["Pshuf-A"]["mean"], *e["contrasts"]["all"]["Pshuf-A"]["ci95"], "report_ext contrasts.all.Pshuf-A")
    put("opdAllA", 100 * e["per_arm"]["all"]["A"]["mean"], "report_ext per_arm.all.A")
    put("opdAllP", 100 * e["per_arm"]["all"]["P"]["mean"], "report_ext per_arm.all.P")
    put("opdAllPshuf", 100 * e["per_arm"]["all"]["Pshuf"]["mean"], "report_ext per_arm.all.Pshuf")
    ci("opdPminusPshufAll", e["contrasts"]["all"]["P-Pshuf"]["mean"], *e["contrasts"]["all"]["P-Pshuf"]["ci95"], "report_ext contrasts.all.P-Pshuf (same pipeline, dynamic sampling off in both)")
    ci("opdPminusPshufMain", e["contrasts"]["main_set"]["P-Pshuf"]["mean"], *e["contrasts"]["main_set"]["P-Pshuf"]["ci95"], "report_ext contrasts.main_set.P-Pshuf")
    ci("opdPminusS", o["contrasts"]["main_set"]["P-S"]["mean"], *o["contrasts"]["main_set"]["P-S"]["ci95"], "opd/report_qwen3-8b_seeds42-43.json contrasts.main_set.P-S")
    put("opdMainSet", load("opd/report_qwen3-8b_seeds42-43.json")["main_set"]["size"], "opd main_set.size")
    s2n = load("paper/synthesis.json")["S2_reachability_source"]["units"]
    put("sTwoUnits", s2n, "synthesis.json S2_reachability_source.units")
    put("samplesPerMechanism", 20 * 64, "20 outside rows × 64 samples per row (Methods)")
    put("nRegimeTraps", load("traps_rl/report_qwen3-8b.json")["traps"], "traps_rl/report_qwen3-8b.json traps")
    s2 = load("paper/synthesis.json")["S2_reachability_source"]["per_outcome"]
    for k in ["A", "P"]:
        put(f"aucStudent{k}", s2[k]["auc_student"], f"synthesis.json S2 per_outcome.{k}.auc_student", "{:.2f}")
        put(f"aucTeacher{k}", s2[k]["auc_teacher"], f"idem auc_teacher", "{:.2f}")
        ci(f"aucDiff{k}", s2[k]["student_minus_teacher"], *s2[k]["ci95"], "idem student_minus_teacher", scale=1.0, fmt="{:.2f}")

    # ---- round 4 ----------------------------------------------------------------------------------------------------------
    r4 = load("paper/round4_conditional.json")["families"]
    total_learned = sum(iv["learned"]["n"] for f in r4.values() for iv in f["interventions"].values())
    total_swap = sum(iv["learned"]["class_counts"]["swap"] for f in r4.values() for iv in f["interventions"].values())
    put("r4LearnedCells", total_learned, "round4_conditional.json sum learned.n")
    drops = [v["per_mechanism"][m]["d_inside"] for fam in r4.values() for v in fam["interventions"].values() for m in v["learned"]["mechanisms"] if v["per_mechanism"][m]["d_inside"] is not None]
    put("r4MaxInsideDrop", -100 * min(drops), "round4_conditional.json largest fall of inside accuracy in a learned cell", "{:.0f}")
    for key, fam, arm in [("QwenA", "qwen3-8b", "A"), ("QwenBA", "qwen35-9b", "A"), ("QwenDose", "qwen3-8b", "DOSE"), ("MinDose", "ministral3-8b", "DOSE"), ("P", "qwen3-8b", "P")]:
        L = r4[fam]["interventions"][arm]["learned"]
        put(f"r4InsideStart{key}", 100 * L["inside_start_mean"], f"round4_conditional {fam} {arm} learned.inside_start_mean (base inside accuracy on the mechanisms learned)", "{:.0f}")
        put(f"r4InsideEnd{key}", 100 * L["inside_end_mean"], f"round4_conditional {fam} {arm} learned.inside_end_mean", "{:.0f}")
        pm = r4[fam]["interventions"][arm]["per_mechanism"]
        outs = [pm[m]["outside_start"] for m in L["mechanisms"] if pm[m].get("outside_start") is not None]
        put(f"r4OutsideStart{key}", 100 * sum(outs) / len(outs), f"round4_conditional {fam} {arm} per_mechanism outside_start, mean over learned mechanisms", "{:.0f}")
    put("r4SwapCells", total_swap, "sum learned.class_counts.swap")
    a8 = r4["qwen3-8b"]["interventions"]["A"]["learned"]
    ci("r4DinsideQwenA", a8["d_inside"]["mean"], *a8["d_inside"]["ci95"], "round4_conditional qwen3-8b A learned.d_inside")
    ci("r4DhfQwenA", a8["d_hf"]["mean"], *a8["d_hf"]["ci95"], "idem d_hf")
    a9 = r4["qwen35-9b"]["interventions"]["A"]["learned"]
    ci("r4DhfQwenBA", a9["d_hf"]["mean"], *a9["d_hf"]["ci95"], "qwen35-9b A learned.d_hf")
    rt = load("paper/round4_traps.json")["families"]["qwen3-8b"]
    put("r4DoseTraps", 100 * rt["groups"]["dose_rl"]["all"]["trap_acc"], "round4_traps.json qwen3-8b groups.dose_rl")
    vx = rt["vs_x_A"]
    ci("r4DoseMinusA", vx["dose_rl_minus_A"]["mean"], *vx["dose_rl_minus_A"]["ci95"], "round4_traps.json qwen3-8b vs_x_A.dose_rl_minus_A")
    ci("r4DoseMinusWA", vx["dose_rl_minus_WA"]["mean"], *vx["dose_rl_minus_WA"]["ci95"], "idem dose_rl_minus_WA")
    put("r4ATraps", 100 * vx["A_trap_acc"], "idem A_trap_acc")
    put("r4WATraps", 100 * vx["WA_trap_acc"], "idem WA_trap_acc")
    mt = load("paper/round4_traps.json")["families"]["ministral3-8b"]["groups"]
    for g, tag in [("base", "Base"), ("dose_warm", "Warm"), ("dose_rl", "RL")]:
        put(f"r4MinTraps{tag}", 100 * mt[g]["all"]["trap_acc"], f"round4_traps.json ministral3-8b groups.{g}")
    rf = load("paper/round4_families.json")
    for key, tag in [("ministral3-8b~qwen3-8b", "MinQwen"), ("qwen3-8b~qwen35-9b", "QwenQwenB"), ("ministral3-8b~qwen35-9b", "MinQwenB")]:
        v = rf["S16a"]["pairs"][key]
        ci(f"rho{tag}", v["rho"], *v["ci95"], f"round4_families.json S16a.pairs.{key}", scale=1.0, fmt="{:.2f}")
    put("zeroInAllFamilies", len(rf["S16a"]["zero_in_all"]), "S16a.zero_in_all")
    nest = rf["S16b"]["mechanisms"]
    put("nestBoth", len(nest["both"]), "S16b.mechanisms.both")
    put("nestBOnly", len(nest["b_only"]), "S16b.mechanisms.b_only")
    put("nestAOnly", len(nest["a_only"]), "S16b.mechanisms.a_only")
    put("nestBothPlusB", len(nest["both"]) + len(nest["b_only"]), "S16b both + b_only")

    # ---- warm-up vs reward form (training fold, Qwen3-8B) --------------------------------------------------------------
    w = load("x/report_qwen3-8b.json")["warm_x_reward"]
    ci("warmupGain", w["WA-A"]["mean"], *w["WA-A"]["ci95"], "x/report_qwen3-8b.json warm_x_reward.WA-A")
    ci("counterfactualGain", w["D-A"]["mean"], *w["D-A"]["ci95"], "warm_x_reward.D-A")
    ci("counterfactualGainWarm", w["WD-WA"]["mean"], *w["WD-WA"]["ci95"], "warm_x_reward.WD-WA")

    # ---- judgement share before and after answer-reward RL (mechanism means) ---------------------------------------------
    pm = r4["qwen3-8b"]["interventions"]["A"]["per_mechanism"]
    hs = [v["H_start"] for v in pm.values() if v.get("H_start") is not None]
    he = [v["H_end"] for v in pm.values() if v.get("H_end") is not None]
    put("hStartQwenA", 100 * sum(hs) / len(hs), "round4_conditional qwen3-8b A per_mechanism H_start (mechanism mean)", "{:.0f}")
    put("hEndQwenA", 100 * sum(he) / len(he), "idem H_end", "{:.0f}")

    # ---- scale and thinking (inside − outside for named configurations) ------------------------------------------------------
    for key, tag in [("gemma4-e4b.think-off", "GemmaSmall"), ("gemma4-31b.think-off", "GemmaLarge"), ("qwen3.5-27b.think-on", "QwenLargeThink"), ("gemma4-31b.think-on", "GemmaLargeThink")]:
        put(f"io{tag}", 100 * cons[key]["inside_minus_outside"]["mean"], f"contrasts.inside_minus_outside {key}", "{:.0f}")

    # ---- P2 judgement transfer (x report) ---------------------------------------------------------------------------------
    for fam, tag in [("qwen3-8b", "Qwen"), ("qwen35-9b_s42-43", "QwenB")]:
        v = load(f"x/report_{fam}.json")["P2"]["F-A_H-F_held"]
        if isinstance(v, dict) and "mean" in v:
            ci(f"p2FminusA{tag}", v["mean"], *v["ci95"], f"x/report_{fam}.json P2.F-A_H-F_held")

    # ---- §25-A E5 prolonged focused RL, §25-B luck recount, §23 E4 Qwen3.5-27B (2026-11-09) --------------------------------
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    rl = load("paper/f25_rl.json")
    assert rl["signal"]["steps"] == 200
    put("extSteps", 200, "explore/f25_chain.sh STEPS; paper/f25_rl.json signal.steps")
    put("extMechs", words[len(rl["targets"])], "paper/f25_rl.json targets")
    put("extLearned", words[rl["learned"]], "paper/f25_rl.json learned")
    put("extZeroPassStart", words[rl["zero_pass64"]["A120_pass@64"]], "paper/f25_rl.json zero_pass64.A120_pass@64")
    put("extZeroPassEnd", words[rl["zero_pass64"]["ext_pass@64"]], "paper/f25_rl.json zero_pass64.ext_pass@64")
    ob = rl["signal"]["outside_by_template"]
    put("extNoSignal", words[sum(1 for v in ob.values() if v["signal"] == 0)], "paper/f25_rl.json signal.outside_by_template (signal 0)")
    lk = load("paper/f25_luck.json")["families"]["qwen3-8b"]
    put("luckFunctionalRare", 100 * lk["rare"]["functional_share"], "paper/f25_luck.json families.qwen3-8b.rare.functional_share", "{:.0f}")
    mant, exp = f"{lk['reach']['p1_table_functional_only']['fisher_p']:.1e}".split("e")
    put("luckFisherP", f"{mant} \\times 10^{{{int(exp)}}}", "paper/f25_luck.json families.qwen3-8b.reach.p1_table_functional_only.fisher_p")
    e4 = load("paper/f23_e4.json")
    sp = e4["vs"]["qwen35-9b"]["spearman"]
    ci("rhoQwenBLarge", sp["rho"], *sp["ci95"], "paper/f23_e4.json vs.qwen35-9b.spearman", scale=1.0, fmt="{:.2f}")
    put("nZeroHitLarge", len(e4["zero_hit"]), "paper/f23_e4.json zero_hit")
    put("nZeroHitLargeShared", len(e4["vs"]["qwen35-9b"]["zero_hit_both"]), "paper/f23_e4.json vs.qwen35-9b.zero_hit_both")

    print("% generated by paper/fig/make_numbers.py from results/ — do not edit")
    for name, value, source in OUT:
        print(f"\\newcommand{{\\{name}}}{{{value}}} % {source}")


if __name__ == "__main__":
    main()
