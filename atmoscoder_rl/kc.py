"""Offline inputs for PREREG_MAIN amendment T-1; no model calls or training.

The bundle contains template inputs only. The later trainer export must add the
frozen 1,140 standard rows and construct A/D reward metadata. pairs.json is an
audit ledger, not an evaluation input. All rules shared with later T tools live here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

from .build_templates import build
from .coverage_clusters import units
from .grading import verify
from .paths import REPO_ROOT
from .pilot import FLAG_INSTRUCTION
from .template_wording import text as reviewed_text
from .templates import ALL_TEMPLATES
from .templates.base import INSIDE_MAX_GAP, OUTSIDE_MIN_GAP, gap, num

PROTOCOL = "T-1:2026-10-08"
FOLD_SEED = "kc-folds:20261008"
PAIRS = 20
SEEDS = {"train": 20260930, "gate": 20261009, "final": 20261010}
VARIANTS = {"train": ("inside", "outside", "given"),
            "gate": ("given", "flag"), "final": ("inside", "outside", "given", "flag")}
EXCLUDED = ("evaporative_cooling_limit", "finite_line_source")
REWORDED = ("clausius_clapeyron_extrapolation", "drop_ventilation", "line_broadening")
INPUT_FILES = ("train_F1.json", "train_F2.json", "gate.json", "final.json")
RECORD_FILES = ("split.json", "manifest.json", "summary.txt")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split() -> dict:
    clusters = {name: sorted(t for t in members if t not in EXCLUDED)
                for name, members in sorted(units().items())}
    require(len(clusters) == 32 and all(clusters.values()), "expected 32 nonempty clusters")
    require(sum(map(len, clusters.values())) == 70, "expected 70 eligible templates")
    names = sorted(clusters)
    random.Random(FOLD_SEED).shuffle(names)
    folds = {}
    for fold, trained, held in (("F1", names[:16], names[16:]), ("F2", names[16:], names[:16])):
        folds[fold] = {
            "trained_clusters": trained, "held_out_clusters": held,
            "trained_templates": sorted(t for c in trained for t in clusters[c]),
            "held_out_templates": sorted(t for c in held for t in clusters[c]),
        }
    return {"protocol": PROTOCOL, "seed": FOLD_SEED, "clusters": clusters,
            "excluded_templates": list(EXCLUDED), "reworded_templates": list(REWORDED), "folds": folds}


def registry(partition: dict) -> dict:
    names = sorted(t for ts in partition["clusters"].values() for t in ts)
    return {name: replace(ALL_TEMPLATES[name], text=partial(reviewed_text, name)) for name in names}


def metadata(partition: dict, name: str) -> tuple[str, str]:
    cluster = next(c for c, ts in partition["clusters"].items() if name in ts)
    fold = "F1" if cluster in partition["folds"]["F1"]["trained_clusters"] else "F2"
    return cluster, fold


def row_from_pair(pair: dict, variant: str) -> dict:
    name, purpose = pair["template"], pair["purpose"]
    params = pair["inside" if variant == "inside" else "outside"]
    t = ALL_TEMPLATES[name]
    value, shortcut = t.correct(params), t.shortcut(params)
    wording_variant = "outside" if variant == "flag" else variant
    problem = reviewed_text(name, params, wording_variant)
    if variant == "flag":
        problem += "\n\n" + FLAG_INSTRUCTION
    return {"id": f"{pair['pair']}_{variant}", "pair": pair["pair"],
            "protocol": PROTOCOL, "purpose": purpose, "seed": SEEDS[purpose],
            "template": name, "cluster": pair["cluster"], "cluster_fold": pair["cluster_fold"],
            "variant": variant, "category": t.category, "mechanism": t.mechanism,
            "problem": problem, "params": params,
            "sub_answers": [{"sub": "1", "value": f"{value:.6g}", "unit": t.unit}],
            "shortcut_value": float(f"{shortcut:.6g}"), "shortcut_values": {"1": float(f"{shortcut:.6g}")},
            "gap": round(gap(value, shortcut), 4)}


def build_inputs() -> tuple[dict, dict]:
    partition = split()
    reg = registry(partition)
    payloads: dict = {"split.json": partition, "pairs.json": []}
    payloads.update({name: [] for name in INPUT_FILES})
    reserved: set[str] = set()
    redraws = {}
    for purpose, seed in SEEDS.items():
        counts: dict[str, int] = {}
        rows = build(PAIRS, seed, registry=reg, exclude_texts=reserved, redraw_counts=counts)
        redraws[purpose] = counts
        for index in range(0, len(rows), 3):
            inside, outside, _ = rows[index:index + 3]
            name = inside["template"]
            cluster, fold = metadata(partition, name)
            pair_id = f"kc:{purpose}:{inside['pair']}"
            record = {"pair": pair_id, "template": name, "cluster": cluster,
                      "cluster_fold": fold, "purpose": purpose, "seed": seed,
                      "inside": inside["params"], "outside": outside["params"]}
            payloads["pairs.json"].append(record)
            reserved.update((inside["problem"], outside["problem"]))
            filename = f"train_{fold}.json" if purpose == "train" else f"{purpose}.json"
            payloads[filename].extend(row_from_pair(record, v) for v in VARIANTS[purpose])
    return payloads, redraws


def validate_inputs(payloads: dict) -> dict:
    """Validate meaning, not just hashes: pairing, keys, wording, folds and leakage."""
    partition = split()
    require(payloads["split.json"] == partition, "split differs from the fixed protocol")
    names = set(registry(partition))
    pairs = {}
    counts: Counter = Counter()
    seen_text: set[str] = set()
    seen_params: set[tuple] = set()
    for record in payloads["pairs.json"]:
        name, purpose, pid = record["template"], record["purpose"], record["pair"]
        require(name in names and purpose in SEEDS, f"unknown template/purpose: {pid}")
        require(pid not in pairs, f"duplicate pair id: {pid}")
        cluster, fold = metadata(partition, name)
        require((record["cluster"], record["cluster_fold"], record["seed"]) ==
                (cluster, fold, SEEDS[purpose]), f"wrong pair metadata: {pid}")
        pin, pout = record["inside"], record["outside"]
        require(pin.keys() == pout.keys(), f"parameter keys differ: {pid}")
        require(all(math.isfinite(v) and float(f"{v:.3g}") == v for p in (pin, pout) for v in p.values()),
                f"nonfinite or unrenderable parameter: {pid}")
        changed = [k for k in pin if pin[k] != pout[k]]
        require(len(changed) == 1, f"pair must change one given: {pid}")
        signature = (name, tuple(sorted(pin.items())), tuple(sorted(pout.items())))
        require(signature not in seen_params, f"repeated parameter pair: {pid}")
        seen_params.add(signature)
        t = ALL_TEMPLATES[name]
        require(gap(t.correct(pin), t.shortcut(pin)) <= INSIDE_MAX_GAP and
                gap(t.correct(pout), t.shortcut(pout)) >= OUTSIDE_MIN_GAP, f"invalid physical gap: {pid}")
        a, b = reviewed_text(name, pin, "inside"), reviewed_text(name, pout, "outside")
        require(a not in seen_text and b not in seen_text and a != b, f"repeated canonical prompt: {pid}")
        seen_text.update((a, b))
        key = changed[0]
        old, new = num(pin[key]), num(pout[key])
        require(any(b == a[:i] + new + a[i + len(old):] for i in range(len(a)) if a.startswith(old, i)),
                f"texts differ beyond the single trigger: {pid}")
        given = reviewed_text(name, pout, "given")
        prefix = 0
        while prefix < min(len(b), len(given)) and b[prefix] == given[prefix]:
            prefix += 1
        added = len(given) - len(b)
        require(added > 0 and given[prefix + added:] == b[prefix:], f"given is not insertion-only: {pid}")
        for variant in ("inside", "outside", "given", "flag"):
            row = row_from_pair(record, variant)
            expected = row["sub_answers"]
            value = float(expected[0]["value"])
            for rule in ("benchmark", "strict"):
                def passes(v: float) -> bool:
                    return verify({"1": {"value": v, "unit": t.unit}}, expected, units=rule)[0]
                require(passes(value) and passes(round(value, 2)) and not passes(0) and not passes(-value),
                        f"key/sign/rounding grading failure: {pid}/{variant}/{rule}")
                require(passes(row["shortcut_value"]) == (variant == "inside"),
                        f"shortcut grading failure: {pid}/{variant}/{rule}")
        pairs[pid] = record
        counts[purpose, name] += 1
    require(counts == Counter({(p, n): PAIRS for p in SEEDS for n in names}), "pair coverage/count differs")
    seen_ids = set()
    row_counts: Counter = Counter()
    for filename in INPUT_FILES:
        for row in payloads[filename]:
            pid = row["pair"]
            require(pid in pairs, f"row without audit pair: {filename}/{pid}")
            record = pairs[pid]
            purpose, variant = record["purpose"], row["variant"]
            target = f"train_{record['cluster_fold']}.json" if purpose == "train" else f"{purpose}.json"
            require(filename == target, f"cross-fold/purpose leakage: {filename}/{pid}")
            require(variant in VARIANTS[purpose], f"forbidden variant: {filename}/{variant}")
            require(row == row_from_pair(record, variant), f"row differs from verified pair: {filename}/{pid}")
            require(row["id"] not in seen_ids, f"duplicate row id: {row['id']}")
            seen_ids.add(row["id"])
            row_counts[pid, variant] += 1
    require(row_counts == Counter({(pid, v): 1 for pid, r in pairs.items() for v in VARIANTS[r["purpose"]]}),
            "missing or repeated pair variants")
    return {"clusters": 32, "templates": len(names), "pairs": len(pairs), "rows": len(seen_ids),
            "canonical_prompts": len(seen_text), "cross_purpose_duplicate_pairs": 0,
            "cross_purpose_duplicate_canonical_prompts": 0, "fold_leaks": 0}


def source_hashes() -> dict[str, str]:
    paths = list((REPO_ROOT / "atmoscoder_rl/templates").glob("*.py"))
    paths += [REPO_ROOT / p for p in (
        "atmoscoder_rl/kc.py", "atmoscoder_rl/build_templates.py", "atmoscoder_rl/template_wording.py",
        "atmoscoder_rl/coverage_clusters.py", "atmoscoder_rl/pilot.py", "atmoscoder_rl/grading.py",
        "atmoscoder_rl/engine.py", "atmoscoder_rl/screening.py", "explore/kc_folds.py",
        "docs/PREREG_MAIN.md", "splits/splits.json")]
    return {str(p.relative_to(REPO_ROOT)): sha256(p.read_bytes()) for p in sorted(paths)}


def build_bundle() -> dict[str, bytes]:
    payloads, redraws = build_inputs()
    checks = validate_inputs(payloads)
    artifacts = {name: json_bytes(value) for name, value in payloads.items()}
    manifest = {"protocol": PROTOCOL, "scope": "template inputs only; trainer export and K data not included",
                "seeds": SEEDS, "pairs_per_template": PAIRS, "variants": VARIANTS,
                "deduplication": "train then gate then final; reserve both reviewed inside/outside texts; redraw whole pairs",
                "redrawn_pairs": redraws, "checks": checks, "sources": source_hashes(),
                "files": {name: {"sha256": sha256(data), "bytes": len(data),
                                 "records": len(payloads[name]) if isinstance(payloads[name], list) else None}
                          for name, data in artifacts.items()}}
    artifacts["manifest.json"] = json_bytes(manifest)
    lines = [f"AtmosCoder T offline inputs ({PROTOCOL})", "",
             "PASS: 70 templates, 32 clusters, 16 clusters per training fold.",
             "PASS: no cross-fold training leakage or repeated canonical prompts/parameter pairs.",
             "PASS: single-trigger wording, insertion-only given, flag wording, keys and shortcuts.",
             "", "Template inputs only: the 1,140 standard training rows are added by the later trainer export.",
             "pairs.json is an audit ledger; gate.json is the gate's only model input.",
             "Final model evaluation must wait until all training checkpoints are frozen.", ""]
    for fold, spec in payloads["split.json"]["folds"].items():
        lines += [f"{fold}: {len(spec['trained_templates'])} train templates; "
                  f"{len(spec['held_out_templates'])} held-out templates",
                  "  train clusters: " + ", ".join(spec["trained_clusters"])]
    lines += [""]
    for name in INPUT_FILES:
        lines.append(f"{name}: {len(payloads[name])} rows; sha256 {manifest['files'][name]['sha256']}")
    for purpose, counts in redraws.items():
        lines.append(f"{purpose}: {sum(counts.values())} rejected draws (within-batch or reserved prompt collisions)")
    lines += ["", "manifest.json sha256 " + sha256(artifacts["manifest.json"])]
    artifacts["summary.txt"] = ("\n".join(lines) + "\n").encode()
    return artifacts


def check_existing(directory: Path, artifacts: dict[str, bytes]) -> None:
    """Preflight without mutation: refuse to replace any differing artifact."""
    if directory.exists():
        require(directory.is_dir(), f"not a directory: {directory}")
        existing = {p.name for p in directory.iterdir()}
        require(existing == set(artifacts), f"refusing to change existing bundle file set: {directory}")
        for name, data in artifacts.items():
            require((directory / name).read_bytes() == data, f"refusing to overwrite changed artifact: {directory / name}")


def write_bundle(directory: Path, artifacts: dict[str, bytes]) -> None:
    """Publish a fully validated directory, or verify an identical existing bundle."""
    check_existing(directory, artifacts)
    if directory.exists():
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kc-", dir=directory.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        for name, data in artifacts.items():
            (staged / name).write_bytes(data)
        staged.rename(directory)


def check_bundle(directory: Path, record_dir: Path | None = None) -> dict:
    """Read-only check: hashes, semantics, and reproduction from the fixed RNG streams."""
    manifest = json.loads((directory / "manifest.json").read_bytes())
    require(manifest["sources"] == source_hashes(), "source/protocol hash drift; preserve bundle and review changes")
    expected = build_bundle()
    require({p.name for p in directory.iterdir()} == set(expected), "bundle file set differs")
    for name, data in expected.items():
        require((directory / name).read_bytes() == data, f"bundle does not reproduce: {name}")
    if record_dir is not None:
        require(record_dir.is_dir(), f"missing review record: {record_dir}")
        check_existing(record_dir, {name: expected[name] for name in RECORD_FILES})
    return manifest["checks"]
