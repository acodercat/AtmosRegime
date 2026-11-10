"""Sample the base model on the training split and keep only rollouts that pass verification.

    uv run --extra eval python -m atmoscoder_rl.sample_raft --out raft_verified.json --samples 8
    uv run --extra eval python -m atmoscoder_rl.sample_raft --out probe.json   # admission probe

(On the training box, run it in the eval venv: README, "Model calls on the server".)
Samples the model served as `local` on :8077 with the `local-raft` settings in `models.toml`
(temperature 1.0 and the 3072-token response cap, i.e. the RL rollout settings); serve any
candidate model under `--served-model-name local`.

This is the supervision for the RAFT arm (rejection-sampling fine-tuning). It exists to
separate two things that the gold-solver SFT confounds: that arm differs from RL both in the
algorithm (imitation rather than policy gradient) *and* in where the supervision comes from
(pipeline-authored solvers rather than the model's own output). Fine-tuning on the model's own
verified rollouts removes the second difference, so if RAFT still fails to transfer, the
policy gradient is what matters; if it transfers, being on-distribution is.

It doubles as the **admission probe** for a candidate model. Dynamic sampling only takes a
gradient from a row whose eight rollouts disagree, so the pass-count histogram this prints is
what decides whether a base model can be trained at all: a model that solves nothing, or
solves everything, hands DAPO an empty batch. Run it before committing GPU days to a family.

Grading goes through `reward.grade`, the same function the RL reward uses, so "verified" means
exactly what it means during training. Sampling is at temperature 1.0 to match the rollout
temperature, not evaluation's 0.3. The n samples of a problem are n independent requests.
"""

from __future__ import annotations

import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .datasets import train_families
from .evaluation.client import ChatModel, ModelError, build_model
from .prompts import SYSTEM, user_prompt
from .reward import grade


def _rows() -> list[dict]:
    """The same 1140 (problem, answers) pairs the RL arm trains on."""
    out = []
    for parent, variants in sorted(train_families(), key=lambda family: family[0]["id"]):
        out.append(parent)
        out.extend(variants)
    return out


def _sample(model: ChatModel, problem: dict[str, Any], n: int) -> dict[str, Any]:
    prompt = user_prompt(problem["problem"])
    try:
        texts = [model.generate(prompt, SYSTEM).content for _ in range(n)]
    except ModelError as error:                       # the endpoint, not the policy
        return {"id": problem["id"], "problem": problem["problem"],
                "completions": [], "sampled": 0, "error": str(error)[:120]}
    kept = [t for t in texts if grade(t, problem["sub_answers"]).passed]
    return {"id": problem["id"], "problem": problem["problem"],
            "completions": kept, "passed": len(kept), "sampled": len(texts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8, help="samples per problem")
    parser.add_argument("--workers", type=int, default=192,
                        help="problems in flight; each draws its samples as separate requests")
    parser.add_argument("--model", default="local-raft", help="a models.toml section")
    parser.add_argument("--cap-per-problem", type=int, default=0,
                        help="keep at most this many verified completions per problem (0 = all); "
                             "caps how much an easy problem can dominate the dataset")
    args = parser.parse_args()

    model = build_model(args.model)
    problems = _rows()
    print(f"{len(problems)} training problems x {args.samples} samples at T={model.temperature}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda p: _sample(model, p, args.samples), problems))

    if args.cap_per_problem:
        for record in results:
            record["completions"] = record["completions"][:args.cap_per_problem]

    kept = sum(len(r["completions"]) for r in results)
    solved = sum(1 for r in results if r["completions"])
    errors = [r for r in results if r.get("error")]
    args.out.write_text(json.dumps(results, ensure_ascii=False))
    print(f"  problems with at least one verified rollout: {solved}/{len(results)} "
          f"({100 * solved / len(results):.1f}%)")
    print(f"  verified rollouts kept: {kept} of {sum(r['sampled'] for r in results)} sampled")
    print(f"  endpoint failures: {len(errors)}")
    graded = [r for r in results if r["sampled"]]
    mixed = [r for r in graded if 0 < r["passed"] < r["sampled"]]
    print(f"  rows that give DAPO a gradient (mixed outcomes): {len(mixed)}/{len(graded)} "
          f"({100 * len(mixed) / max(len(graded), 1):.1f}%)")
    hist = collections.Counter(r["passed"] for r in graded)
    print("  pass count out of {}: {}".format(
        args.samples, "  ".join(f"{k}:{hist.get(k, 0)}" for k in range(args.samples + 1))))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
