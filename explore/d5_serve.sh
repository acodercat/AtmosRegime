#!/usr/bin/env bash
# D-5 textbook replication for one configuration (docs/PREREG_MAIN.md revisions D-5, D-5a): serve the model with the vLLM
# arguments it had in D (results/d5/serving.json; only the port differs), run the D-5 input three times, stop the server.
#   GPUS=0,1,2,3 bash explore/d5_serve.sh qwen3.5-27b.think-off
# Resume-safe: the runner skips every completed id of a repetition, so re-running the same command finishes what is missing.
# The runner stops at the first infrastructure error (exit 1, set -e ends the script and the server): re-run the command.
# Weights: Gemma 4 31B on 114's /local, Granite 8B and Ministral on 047's, Granite 30B on both; the rest on /path/to/shared.
source /path/to/workdir/env.sh   # before set -u: the system profile it pulls in reads unset variables
set -euo pipefail
CONFIG=$1
ROOT=/path/to/workdir
REPO=$(cd "$(dirname "$0")/.." && pwd)
INPUT=$ROOT/experiments/_inputs/d5/textbook.json
SHA=cc3d59e2ac1b661ab14dd154897bf9b303e2e44d2648119dd11f08acaaa27906   # PREREG_MAIN D-5 §2
LOG=$ROOT/logs/d5; mkdir -p "$LOG"

[ "$(sha256sum "$INPUT" | cut -d' ' -f1)" = "$SHA" ] || { echo "input $INPUT missing or not the sealed D-5 file"; exit 1; }
read -r SECTION PORT NGPU WORKERS < <(python3 -c '
import json, sys
v = json.load(open(sys.argv[1]))[sys.argv[2]]
print(v["section"], v["port"], v["gpus"], v["workers"])' "$REPO/results/d5/serving.json" "$CONFIG")
mapfile -t ARGV < <(python3 -c '
import json, sys
print("\n".join(json.load(open(sys.argv[1]))[sys.argv[2]]["vllm_arguments"]))' "$REPO/results/d5/serving.json" "$CONFIG")
[ "$(tr ',' '\n' <<< "$GPUS" | grep -c .)" = "$NGPU" ] || { echo "$CONFIG needs $NGPU GPUs, got GPUS=$GPUS"; exit 1; }
[ -e "${ARGV[1]}" ] || { echo "weights ${ARGV[1]} not on this machine"; exit 1; }

CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  setsid "$ROOT/verl/.venv/bin/python" -m vllm.entrypoints.openai.api_server "${ARGV[@]}" > "$LOG/server-$CONFIG.log" 2>&1 &
SERVER=$!
trap 'kill -- -$SERVER 2>/dev/null || kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "server exited (log $LOG/server-$CONFIG.log)"; exit 1; }
  sleep 10
done
cd "$REPO"
# Thinking mode as configured (D's supervisor checked this before any formal call): one short request with the section's
# own request fields; thinking on must return reasoning, thinking off (and Ministral, which has no switch) must not return
# a reasoning trace beside an answer.
"$ROOT/eval-venv/bin/python" - "$SECTION" "$CONFIG" "$LOG/probe_audit/$CONFIG" <<'PY'
import json, sys, urllib.request
from atmoscoder_rl.evaluation.client import load_config
from atmoscoder_rl.paths import REPO_ROOT
section, config, audit = sys.argv[1:]
want = config.endswith(".think-on")
if want:
    # D's own short_initial probe (explore/check_families_v2_live.py): the same synthetic problem through the v2 episode,
    # system prompt included; the first attempt must pass and carry reasoning. A bare chat question is not enough: Gemma 4
    # E4B answers "What is 17 + 25?" without a trace although every D think-on record has one (PREREG_MAIN D-5c).
    from atmoscoder_rl.evaluation_v2.client import Journal, build_model
    from atmoscoder_rl.evaluation_v2.episode import evaluate
    problem = dict(id="synthetic-serving-only", sub_answers=[dict(sub="1", value="42", unit="")],
                   problem="A counter starts at 17 and increases by 25. Return the final count (dimensionless).")
    record = evaluate(problem, build_model(section, Journal(audit, dict(configuration=config, problem="synthetic"))), .05,
                      max_attempts=1)
    reasoning = (record["attempts"][0].get("reasoning") or "").strip()
    ok = record.get("passed") is True and not record.get("error") and bool(reasoning)
    print(f"thinking check {config}: D probe passed {record.get('passed')}, reasoning {len(reasoning)} chars, expected a trace")
    sys.exit(0 if ok else 1)
c = load_config(section, REPO_ROOT / "models_v2.toml")
body = {"model": c["model_id"], "messages": [{"role": "user", "content": "What is 17 + 25? Answer with the number."}],
        "temperature": 0.0, "max_tokens": 2048, **c.get("extra_body", {})}
req = urllib.request.Request(c["base_url"] + "/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
content = (msg.get("content") or "").strip()
# With thinking off, a reasoning parser that never sees a closing tag puts the whole answer in the reasoning field and
# leaves content empty (Granite 4.2 under deepseek_r1 answers "42" that way); the runner grades that text as the answer,
# as in D. Thinking off therefore fails only when the server split the output into a trace and an answer.
thought = bool(reasoning) and bool(content)
print(f"thinking check {config}: reasoning {len(reasoning)} chars, content {len(content)} chars, "
      f"expected no trace beside an answer")
sys.exit(1 if thought else 0)
PY
# D's own runner (server-tokenized repair budgets, models_v2.toml). It names its subdirectory families_context_v2, so the
# D-5 files land in $ROOT/experiments/families_d5/families_context_v2/, apart from D's. It stops after an infrastructure
# error; re-running this script resumes.
for run in 1 2 3; do
  "$ROOT/eval-venv/bin/python" -m atmoscoder_rl.evaluation_v2.runner --model "$SECTION" --input "$INPUT" \
    --out-name "$CONFIG.textbook" --run "$run" --workers "$WORKERS" --experiments "$ROOT/experiments/families_d5"
done
echo "$CONFIG done"
