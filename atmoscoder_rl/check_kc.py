"""Offline regression and corruption checks for T-1 input preparation.

    python -m atmoscoder_rl.check_kc

Generates all inputs, checks historical builder compatibility, injects leakage and
grading defects, and exercises reproducibility/overwrite refusal in a temporary directory.
No models, no network, no changes to frozen artifacts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

from .build_templates import build
from . import kc
from .templates import ALL_TEMPLATES


def must_reject(label: str, action, message: str) -> None:
    try:
        action()
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(f"{label}: unexpected rejection: {error}") from error
        print(f"PASS: rejects {label}")
    else:
        raise AssertionError(f"{label}: accepted corrupt input")


def main() -> int:
    # Digests captured before adding exclusions/shared wording. Existing experiments
    # must not change when the T generator gains a new draw policy.
    old = build(20, 20260930, set(ALL_TEMPLATES))
    digest = hashlib.sha256(json.dumps(old, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == "dcbd2295f235dfcebfb1674b63130105548a43bda7adc4a1a6510f8229f1243a"
    print("PASS: original 72-template builder output unchanged")

    # Reserve an actual draw and replay the SAME seed: a later draw must be used.
    first = build(1, 20260930, ["cunningham_slip"])
    reserved = {r["problem"] for r in first if r["variant"] != "given"}
    counts = {}
    next_rows = build(1, 20260930, ["cunningham_slip"], exclude_texts=reserved, redraw_counts=counts)
    assert not reserved.intersection(r["problem"] for r in next_rows)
    assert counts["cunningham_slip"] >= 1
    print("PASS: reserved prompts force deterministic whole-pair redraws")

    artifacts = kc.build_bundle()
    payloads = {name: json.loads(data) for name, data in artifacts.items()
                if name.endswith(".json") and name != "manifest.json"}
    assert len(payloads["gate.json"]) == 2800 and len(payloads["final.json"]) == 5600
    assert sum(len(payloads[f"train_{f}.json"]) for f in ("F1", "F2")) == 4200
    for f, other in (("F1", "F2"), ("F2", "F1")):
        trained = set(payloads["split.json"]["folds"][f]["trained_clusters"])
        held = set(payloads["split.json"]["folds"][f]["held_out_clusters"])
        assert len(trained) == len(held) == 16 and not trained & held
        assert trained == set(payloads["split.json"]["folds"][other]["held_out_clusters"])
    print("PASS: all inputs validate; complete 16/16 cross-over folds and variant coverage")

    def corrupt(label, mutate, message):
        damaged = deepcopy(payloads)
        mutate(damaged)
        must_reject(label, lambda: kc.validate_inputs(damaged), message)

    corrupt("training row moved to held-out fold",
            lambda p: p["train_F2.json"].append(p["train_F1.json"].pop()), "cross-fold/purpose leakage")
    corrupt("modified numerical key",
            lambda p: p["gate.json"][0]["sub_answers"][0].update(value="123456789"), "row differs")
    corrupt("modified flag wording",
            lambda p: p["gate.json"][1].update(problem="Use another method."), "row differs")
    corrupt("missing final variant", lambda p: p["final.json"].pop(), "missing or repeated pair variants")
    corrupt("duplicate row", lambda p: p["gate.json"].append(deepcopy(p["gate.json"][0])), "duplicate row id")

    def repeat_pair(p):
        original = p["pairs.json"][0]
        target = next(r for r in p["pairs.json"] if r["purpose"] == "gate" and r["template"] == original["template"])
        target.update(inside=deepcopy(original["inside"]), outside=deepcopy(original["outside"]))
    corrupt("train/gate parameter-pair reuse", repeat_pair, "repeated parameter pair")

    def second_trigger(p):
        record = p["pairs.json"][0]
        unchanged = next(k for k in record["inside"] if record["inside"][k] == record["outside"][k])
        record["outside"][unchanged] += 1
    corrupt("two changed givens", second_trigger, "pair must change one given")
    corrupt("altered cluster assignment",
            lambda p: p["pairs.json"][0].update(cluster="wrong"), "wrong pair metadata")
    corrupt("missing entire template", lambda p: p["pairs.json"].__setitem__(
        slice(None), [r for r in p["pairs.json"] if r["template"] != "cunningham_slip"]), "pair coverage/count differs")

    with tempfile.TemporaryDirectory(prefix="check-kc-") as temp:
        out, record = Path(temp) / "inputs", Path(temp) / "record"
        kc.write_bundle(out, artifacts)
        small = {name: artifacts[name] for name in kc.RECORD_FILES}
        kc.write_bundle(record, small)
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()}
        checks = kc.check_bundle(out, record)
        assert checks["rows"] == 12600
        kc.write_bundle(out, artifacts)
        assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()}
        print("PASS: fresh regeneration agrees byte-for-byte; check and identical rerun do not write")

        damaged = dict(artifacts)
        damaged["gate.json"] += b" "
        must_reject("overwriting a frozen bundle", lambda: kc.write_bundle(out, damaged), "refusing to overwrite")
        assert before == {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()}
        # Even a self-consistent edited file hash must fail deterministic regeneration.
        edited = json.loads(artifacts["gate.json"])
        edited[0]["problem"] += " tampered"
        changed = kc.json_bytes(edited)
        (out / "gate.json").write_bytes(changed)
        manifest = json.loads(artifacts["manifest.json"])
        manifest["files"]["gate.json"].update(sha256=kc.sha256(changed), bytes=len(changed))
        (out / "manifest.json").write_bytes(kc.json_bytes(manifest))
        must_reject("tampering with an updated file hash", lambda: kc.check_bundle(out), "does not reproduce")
        (out / "gate.json").write_bytes(artifacts["gate.json"])
        manifest = json.loads(artifacts["manifest.json"])
        manifest["sources"]["docs/PREREG_MAIN.md"] = "0" * 64
        (out / "manifest.json").write_bytes(kc.json_bytes(manifest))
        must_reject("source/protocol drift", lambda: kc.check_bundle(out), "source/protocol hash drift")
    print("All T input checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
