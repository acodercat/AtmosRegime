"""Model evaluation: ask a model each problem, execute its `solve()`, grade it, store the run.

- `client.py`  — the chat client for one `models.toml` section (every model call goes here)
- `episode.py` — one problem: prompt, execute, self-repair, grade -> one stored record
- `store.py`   — the resumable run file
- `runner.py`  — the CLI over a problem set (`python -m atmoscoder_rl.evaluation.runner`)

Needs the `eval` extra: `uv sync --extra eval`. Adapted 2026-09-26 from AtmosCoder-Bench's
evaluation harness, keeping only its code-mode path; grading is `atmoscoder_rl.engine`,
byte-identical to that benchmark's engine.
"""
