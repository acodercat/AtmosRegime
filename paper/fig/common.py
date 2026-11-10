"""Shared style and loaders for the manuscript figures.

Every figure script reads committed files under ``results/`` only (never ``experiments/``),
so the figures can be regenerated from a clean checkout with ``uv sync --extra paper``.
Run from the repository root: ``uv run python paper/fig/fig1.py``.
"""

from __future__ import annotations

import csv
import os
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

# Nature Communications: single column 88 mm, double column 180 mm; 5–7 pt text in figures.
MM = 1 / 25.4
SINGLE = 88 * MM
DOUBLE = 180 * MM

# One colour per role, used identically in every figure. Hues are Paul Tol's colour-blind-safe
# qualitative schemes (bright and muted; Tol, technical note 3.2, 2021): two families of colour,
# blue for the trained student and warm for injected knowledge, green for the teacher, and greys
# for baselines and controls. Pairs that sit side by side were checked for protan/deutan separation.
TOL = {
    "blue": "#4477AA", "cyan": "#77B5D9", "green": "#117733", "yellow": "#DDAA33", "rose": "#CC6677",
    "purple": "#AA3377", "light_rose": "#EE99AA", "light_green": "#A3CFA3", "grey": "#BBBBBB",
}
C = {
    "base": "#808080",            # pre-training model
    "warm": "#BBBBBB",            # warm-up SFT
    "rl": TOL["blue"],            # answer-reward RL (A arm, main runs, Qwen3-8B)
    "rl2": TOL["cyan"],           # second family (Qwen3.5-9B)
    "random": "#D0D0D0",          # random-reward control
    "dose": TOL["rose"],          # injected worked examples
    "dose2": TOL["light_rose"],   # injected examples, third family (Ministral)
    "text": TOL["yellow"],        # injected text statement
    "distil": TOL["green"],       # privileged on-policy distillation (P)
    "distil_bad": TOL["light_green"],  # distillation controls (S, Pshuf, Pk)
    "frontier": "#222222",        # frontier API model
    "inside": "#4D4D4D",
    "outside": "#D55E00",
    "given": TOL["green"],
    "flag": TOL["purple"],
    "shortcut": "#D55E00",        # answers equal to the standard-formula value (orange, as in Fig. 1)
}
INK = "#333333"   # axes, ticks, error bars, annotations
RULE = "#999999"  # reference lines


def seq_cmap():
    """One-hue sequential map for accuracy heat maps (light to dark blue; grey for missing)."""
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap("Blues")
    cmap = LinearSegmentedColormap.from_list("blues_trunc", base(np.linspace(0.08, 1.0, 256)))
    cmap.set_bad("#DDDDDD")
    return cmap

FAMILY_LABEL = {
    "qwen3-8b": "Qwen3-8B",
    "qwen35-9b": "Qwen3.5-9B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "ministral3-8b": "Ministral 3 8B",
}

CONFIG_LABEL = {
    "qwen3-8b.think-off": "Qwen3-8B",
    "qwen3-8b.think-on": "Qwen3-8B (thinking)",
    "qwen3.5-9b.think-off": "Qwen3.5-9B",
    "qwen3.5-9b.think-on": "Qwen3.5-9B (thinking)",
    "qwen3.5-27b.think-off": "Qwen3.5-27B",
    "qwen3.5-27b.think-on": "Qwen3.5-27B (thinking)",
    "gemma4-e4b.think-off": "Gemma 4 E4B",
    "gemma4-e4b.think-on": "Gemma 4 E4B (thinking)",
    "gemma4-31b.think-off": "Gemma 4 31B",
    "gemma4-31b.think-on": "Gemma 4 31B (thinking)",
    "granite4.2-8b.think-off": "Granite 4.2 8B",
    "granite4.2-8b.think-on": "Granite 4.2 8B (thinking)",
    "granite4.2-30b.think-off": "Granite 4.2 30B",
    "granite4.2-30b.think-on": "Granite 4.2 30B (thinking)",
    "ministral3-8b.think-off": "Ministral 3 8B",
    "ministral3-14b.think-off": "Ministral 3 14B",
    "deepseek-v4-flash.think-off": "DeepSeek V4 Flash",
    "deepseek-v4-flash.think-on": "DeepSeek V4 Flash (thinking)",
}

CATEGORY_LABEL = {
    "air_quality": "Air quality",
    "atmospheric_aerosols": "Aerosols",
    "atmospheric_chemistry": "Chemistry",
    "atmospheric_dynamics": "Dynamics",
    "atmospheric_radiation": "Radiation",
    "atmospheric_thermodynamics": "Thermodynamics",
    "boundary_layer": "Boundary layer",
    "climate_dynamics": "Climate",
    "cloud_physics": "Cloud physics",
    "hydrology": "Hydrology",
    "non_atmospheric_environmental": "Other environmental",
    "observation_and_modeling": "Observation & modelling",
    "other": "Other",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Arimo", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": "#222222",
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": "#222222",
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "errorbar.capsize": 0,
            "axes.prop_cycle": matplotlib.cycler(color=[TOL["blue"], TOL["rose"], TOL["green"], TOL["yellow"], TOL["purple"], TOL["cyan"], "#808080"]),
            # Unhinted text metrics: Agg's hinted extents at 120 dpi run a few per cent wider than the
            # PDF backend's, which misplaces panel letters next to long tick labels.
            "text.hinting": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )


def load(rel: str):
    """Load a JSON file under results/."""
    with open(RESULTS / rel, encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(rel: str) -> list[dict[str, str]]:
    with open(RESULTS / rel, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def curve(rel: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (step, value) for one column of a training-curve CSV, skipping empty cells."""
    steps, values = [], []
    for row in load_csv(rel):
        v = row.get(column, "")
        if v in ("", "nan", "None"):
            continue
        try:
            values.append(float(v))
        except ValueError:
            continue
        steps.append(int(float(row["step"])))
    return np.asarray(steps), np.asarray(values)


_ROW = re.compile(r"^\|(.+)\|\s*$")


def md_table(rel: str, first_header_cell: str) -> list[list[str]]:
    """Parse the first Markdown table in ``results/<rel>`` whose header starts with
    ``first_header_cell``. Returns rows (header first) as lists of stripped cells with
    Markdown emphasis removed. Used for the two result files that exist only as Markdown
    (results/FINAL.md, results/seeds/summary.md)."""
    rows: list[list[str]] = []
    capturing = False
    for line in (RESULTS / rel).read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if not m:
            if capturing:
                break
            continue
        cells = [c.strip().replace("**", "") for c in m.group(1).split("|")]
        if not capturing:
            if cells and cells[0] == first_header_cell:
                capturing = True
                rows.append(cells)
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    if not rows:
        raise KeyError(f"no table starting with {first_header_cell!r} in {rel}")
    return rows


def md_tables(rel: str) -> list[list[list[str]]]:
    """All Markdown tables in ``results/<rel>``, each as rows of stripped cells (separator rows dropped)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in (RESULTS / rel).read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if not m:
            if current:
                tables.append(current)
                current = []
            continue
        cells = [c.strip().replace("**", "") for c in m.group(1).split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        current.append(cells)
    if current:
        tables.append(current)
    return tables


def d_contrasts() -> dict[str, dict]:
    """Diagnostic template contrasts for the 16 local configurations over the 69 templates whose
    wording matches the training study (the three templates reworded for training are dropped;
    PREREG_PAPER_ANALYSES §19, results/paper/d_reword.json, explore/d_reword_sensitivity.py).
    Returns configuration key -> {inside_minus_outside, given_minus_outside, flag_minus_outside}."""
    return {c["configuration"]: c["templates_69"]["contrasts"] for c in load("paper/d_reword.json")["configurations"]}


def num(s: str) -> float:
    """float() for Markdown cells that use the typographic minus or a leading plus."""
    return float(s.replace("−", "-").replace("+", "").strip())


def pct(x: float) -> float:
    return 100.0 * x


def ci_err(mean: float, ci: list[float]) -> np.ndarray:
    """Asymmetric error-bar array (2×1) in the same units as mean."""
    return np.array([[mean - ci[0]], [ci[1] - mean]])


def _visible_bbox(ax, renderer):
    """Tight bbox of an axes in display units, with each legend counted by its visible text and handles
    rather than by its padded (frameless) box, so that a legend above a panel does not push the
    panel letter further away than a tick label would."""
    from matplotlib.legend import Legend
    from matplotlib.transforms import Bbox
    extra, boxes = [], []
    for art in ax.get_default_bbox_extra_artists():
        if isinstance(art, Legend):
            boxes += [t.get_window_extent(renderer) for t in art.get_texts() if t.get_text()]
            boxes += [h.get_window_extent(renderer) for h in art.legend_handles if h is not None and h.get_visible()
                      and h.get_window_extent(renderer).width > 0]
        else:
            extra.append(art)
    return Bbox.union([ax.get_tightbbox(renderer, bbox_extra_artists=extra)] + boxes)


def legend_above(ax, *args, ncol: int = 1, **kw):
    """The one placement used in every figure for a legend that does not fit inside the frame:
    above the frame, left-aligned with it, the same padding everywhere."""
    opts = dict(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol, borderaxespad=0.4, borderpad=0.2,
                handletextpad=0.4, columnspacing=1.0)
    opts.update(kw)
    return ax.legend(*args, **opts)


def _space(gs, name):
    v = getattr(gs, name, None) if hasattr(gs, name) else getattr(gs, "_" + name)
    return v if v is not None else getattr(gs.figure.subplotpars, name)


def _set_space(gs, name, value):
    if hasattr(gs, "update"):  # top-level GridSpec
        gs.update(**{name: value})
    else:  # GridSpecFromSubplotSpec
        setattr(gs, "_" + name, value)


def space_panels(fig, hgap_pt: float = 18.0, vgap_pt: float = 22.0, skip=(), passes: int = 12) -> None:
    """Set the gridspec spacing so that the tightest white gap between horizontally neighbouring
    panels is ``hgap_pt`` and between rows ``vgap_pt``, measured on the panels' full extents (tick
    labels, axis labels and legends included). The same two numbers are used for every figure. Call
    after all artists are drawn and before any manual ``set_position`` and before ``label_panels``;
    ``skip`` lists axes that are not panels (a colourbar)."""
    skip = set(map(id, skip))
    axs = [ax for ax in fig.axes if id(ax) not in skip and ax.get_subplotspec() is not None]
    pt = 72.0 / fig.dpi
    w_in, h_in = fig.get_size_inches()
    for _ in range(passes):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        tb = {id(ax): ax.get_tightbbox(r) for ax in axs}
        changed = False
        by_gs: dict = {}
        for ax in axs:
            gs = ax.get_subplotspec().get_gridspec()
            by_gs.setdefault(id(gs), (gs, []))[1].append(ax)
        for gs, members in by_gs.values():
            if gs.ncols < 2:
                continue
            rows: dict = {}
            for ax in members:
                rows.setdefault(ax.get_subplotspec().rowspan.start, []).append(ax)
            gaps = []
            for row in rows.values():
                row.sort(key=lambda a: a.get_subplotspec().colspan.start)
                gaps += [(tb[id(b)].x0 - tb[id(a)].x1) * pt for a, b in zip(row, row[1:])]
            if gaps and abs(hgap_pt - min(gaps)) > 0.3:
                avg_w = float(np.mean([ax.get_position().width for ax in members])) * w_in * 72
                _set_space(gs, "wspace", _space(gs, "wspace") + (hgap_pt - min(gaps)) / avg_w)
                changed = True
        by_top: dict = {}
        for ax in axs:
            top = ax.get_subplotspec().get_topmost_subplotspec()
            by_top.setdefault(id(top.get_gridspec()), (top.get_gridspec(), []))[1].append((top.rowspan.start, ax))
        for gs, members in by_top.values():
            if gs.nrows < 2:
                continue
            rows = {}
            for ri, ax in members:
                rows.setdefault(ri, []).append(ax)
            keys = sorted(rows)
            gaps = [(min(tb[id(a)].y0 for a in rows[k1]) - max(tb[id(a)].y1 for a in rows[k2])) * pt for k1, k2 in zip(keys, keys[1:])]
            if abs(vgap_pt - min(gaps)) > 0.3:
                avg_h = float(np.mean([ax.get_position().height for _, ax in members])) * h_in * 72
                _set_space(gs, "hspace", _space(gs, "hspace") + (vgap_pt - min(gaps)) / avg_h)
                changed = True
        if not changed:
            break
        for ax in fig.axes:
            if ax.get_subplotspec() is not None:
                ax._set_position(ax.get_subplotspec().get_position(fig), which="both")


def align_columns(fig, columns) -> None:
    """Make panels stacked in one column start at the same x: each column's panels are narrowed from
    the left until their full extents (tick labels included) share the rightmost left edge, keeping
    their right edges fixed. Call after space_panels and before label_panels, so that the letters of
    one column sit on one vertical line and each letter stays at its own panel's edge."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for axes in columns:
        x0s = [inv.transform(_visible_bbox(ax, renderer))[0][0] for ax in axes]
        target = max(x0s)
        for ax, x0 in zip(axes, x0s):
            shift = target - x0
            if shift > 1e-4:
                p = ax.get_position()
                ax.set_position([p.x0 + shift, p.y0, p.width - shift, p.height])


def label_panels(fig, rows, pad_pt: float = 4.0) -> None:
    """Place bold panel letters so that every letter sits at the left edge of its panel's full
    extent (tick labels and axis label included) and all letters in a row share one baseline,
    just above the tallest element of that row. ``rows`` is a list of rows, each a list of
    (axes, letter). Call after all artists are drawn and before saving."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad = pad_pt / 72.0 / fig.get_size_inches()[1]
    # Letters in one grid column share one x: the leftmost extent of any panel in that column.
    col_x: dict[int, float] = {}
    for row in rows:
        for ax, _ in row:
            spec = ax.get_subplotspec()
            # Only axes of the same gridspec share a column; nested subgridspecs with different
            # width ratios would otherwise push a letter far left of its own panel.
            col = (id(spec.get_gridspec()), spec.colspan.start) if spec is not None else id(ax)
            x0 = inv.transform(_visible_bbox(ax, renderer))[0][0]
            col_x[col] = min(col_x.get(col, x0), x0)
    for row in rows:
        boxes = [(ax, letter, inv.transform(_visible_bbox(ax, renderer))) for ax, letter in row]
        top = max(b[1][1] for _, _, b in boxes)
        for ax, letter, b in boxes:
            spec = ax.get_subplotspec()
            x = col_x[(id(spec.get_gridspec()), spec.colspan.start)] if spec is not None else b[0][0]
            fig.text(x, top + pad, letter, fontsize=8, fontweight="bold", va="bottom", ha="left")
            if os.environ.get("PANEL_GAPS"):  # report how far each letter sits above its own panel (pt)
                print(f"GAP {letter} {(top - b[1][1]) * fig.get_size_inches()[1] * 72 + pad_pt:.1f}")


def panel_label(ax, letter: str, dx: float = -0.12, dy: float = 1.04) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    print(out)
    return out
