"""Launch one D model group (both thinking modes, or off only) under PREREG_MAIN revision D-2a.

    python -m explore.d_launch --spec results/d_launch/<name>.json          # on the serving host

Same sequence as the earlier verified launches: isolated checkout of the frozen D base commit, official-file
verification of the pinned weights, synthetic probe (no benchmark calls), review of the probe journal, then the
formal run through explore.families_v2_supervise. Only the serving throughput settings differ (D-2a); the one
code change applied to the checkout is the supervisor's GPU-memory check across all listed GPUs.

Spec fields: name, host, model_path, upstream_repo, revision, served_name, port, gpus ("4,5,6,7"),
tensor_parallel, data_parallel, max_num_seqs, max_num_batched_tokens, workers, modes (["off","on"] or ["off"]),
reasoning_parser, extra_vllm_args (list), language_model_only (bool), wall_seconds, no_thinking_switch (bool, optional).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/path/to/workdir/experiments')
BASE_REPO = ROOT / 'd_gemma_ready_20261010/repo'
BASE = '83ae1c3'
PYTHON = '/path/to/workdir/eval-venv/bin/python'
CAPS = {'off': 8192, 'on': 28672}
OLD_MEMORY = """            return int(subprocess.check_output(['nvidia-smi', '--id=' + gpu,
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())"""
NEW_MEMORY = """            return sum(int(x) for x in subprocess.check_output(['nvidia-smi', '--id=' + gpu,
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).split())"""


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def write(path: Path, value) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=1) + '\n')
    tmp.replace(path)


def run(args, **kw):
    return subprocess.run(args, check=True, **kw)


def upstream_listing(repo: str, revision: str) -> dict:
    url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def wait_port(port: int, seconds: int = 600) -> None:
    """Data-parallel servers release their port some seconds after shutdown; wait until it can be bound."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            with socket.socket() as s:
                s.bind(('127.0.0.1', port))
            return
        except OSError:
            require(time.monotonic() < deadline, f'port {port} still in use after {seconds}s')
            time.sleep(10)


def formal_only(spec: dict) -> int:
    dest = ROOT / f"d_{spec['name']}"
    statepath = dest / 'launch_state.json'
    state = json.loads(statepath.read_text())
    repo = dest / 'repo'
    folder = repo / f"results/d_{spec['name']}"
    require((folder / 'ready_contract.json').exists() and (folder / 'live_review.json').exists(), 'probe not passed/reviewed')
    committed = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(folder)], cwd=repo).returncode == 0
    require(committed and state.get('formal_commit'), 'ready contract not committed')
    require(json.loads((folder / 'live_review.json').read_text())['status'] == 'passed', 'probe review did not pass')

    def save(**kw):
        state.update(kw, updated_unix=time.time())
        write(statepath, state)
    try:
        save(state='formal_D', resumed_formal_only=True)
        wait_port(spec['port'])
        run([PYTHON, '-m', 'explore.families_v2_supervise', '--contract', str(folder / 'ready_contract.json'),
             '--run-dir', str(dest / 'run')], cwd=repo)
        final = json.loads((dest / 'run/status.json').read_text())
        save(state='complete' if final['state'] == 'complete' else 'stopped_incomplete', final=final)
    except BaseException as exc:
        save(state='stopped_incomplete', error=type(exc).__name__ + ': ' + str(exc))
        raise
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--formal-only', action='store_true',
                    help='resume after a passed, committed probe: wait for the port, then run the formal stage only')
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.formal_only:
        return formal_only(spec)
    require(socket.gethostname() == spec['host'], 'wrong host')
    # Without --enforce-eager vLLM compiles kernels (Qwen3.5's GDN layers need nvcc): run after `source env.sh`.
    import shutil
    require(shutil.which('nvcc') is not None, 'nvcc not on PATH: source $ATMOS_ROOT/env.sh before launching')
    dest = ROOT / f"d_{spec['name']}"
    require(not dest.exists(), f'{dest} exists; inspect before relaunching')
    dest.mkdir()
    state = {'state': 'preparing', 'spec': spec, 'started_unix': time.time()}
    statepath = dest / 'launch_state.json'

    def save(**kw):
        state.update(kw, updated_unix=time.time())
        write(statepath, state)
    save()
    try:
        repo = dest / 'repo'
        run(['git', 'clone', '--no-hardlinks', '--no-checkout', str(BASE_REPO), str(repo)])
        run(['git', 'checkout', '--detach', BASE], cwd=repo)
        sup = repo / 'explore/families_v2_supervise.py'
        text = sup.read_text()
        require(text.count(OLD_MEMORY) == 1, 'supervisor memory check not found at base')
        sup.write_text(text.replace(OLD_MEMORY, NEW_MEMORY))
        sys.path.insert(0, str(repo))
        from atmoscoder_rl.serving_binding import weight_revision
        from atmoscoder_rl.evaluation_v2.audit import summarize, record_bytes
        from atmoscoder_rl.frozen import sha256

        model = Path(spec['model_path'])
        revision = weight_revision(model)
        upstream = upstream_listing(spec['upstream_repo'], spec['revision'])
        require(upstream['sha'] == spec['revision'], 'upstream revision differs')
        expected = [x for x in upstream['siblings'] if x['rfilename'].endswith(('.json', '.jinja', '.safetensors', '.txt'))]
        manifest = revision['weights_manifest']
        require({x['rfilename'] for x in expected} <= set(manifest), 'official files missing locally')
        for entry in expected:
            path = model / entry['rfilename']
            require(path.stat().st_size == entry['size'], 'official size differs: ' + entry['rfilename'])
            if entry.get('lfs'):
                require(manifest[entry['rfilename']] == entry['lfs']['sha256'], 'official LFS hash differs: ' + entry['rfilename'])
            else:
                blob = path.read_bytes()
                require(hashlib.sha1(f'blob {len(blob)}\0'.encode() + blob).hexdigest() == entry['blobId'],
                        'official git hash differs: ' + entry['rfilename'])
        folder = repo / f"results/d_{spec['name']}"
        folder.mkdir(parents=True)
        write(folder / 'weights.json', revision)
        write(folder / 'upstream.json', upstream)
        write(folder / 'launch_spec.json', spec)
        inputs = (BASE_REPO / 'experiments/_inputs/families_v4').resolve()
        local_input = repo / 'experiments/_inputs/families_v4'
        local_input.parent.mkdir(parents=True)
        local_input.symlink_to(inputs, target_is_directory=True)

        registry = repo / 'models_v2.toml'
        sections = []
        for mode in spec['modes']:
            switch = 'true' if mode == 'on' else 'false'
            # A model without a thinking toggle (Ministral 3) gets no chat_template_kwargs: vLLM's mistral tokenizer
            # mode rejects any request that carries them. Every sampling field stays the same.
            kwargs = '' if spec.get('no_thinking_switch') else f', chat_template_kwargs = {{ enable_thinking = {switch} }}'
            sections.append(f"""
[d-v2-{spec['name']}-{mode}]
model_id = "{spec['served_name']}"
base_url = "http://127.0.0.1:{spec['port']}/v1"
api_key = "EMPTY"
temperature = 0.3
max_tokens = {CAPS[mode]}
timeout = 1800.0
thinking_echo_field = "reasoning"
extra_body = {{ top_p = 1.0, top_k = -1, min_p = 0.0, repetition_penalty = 1.0, presence_penalty = 0.0, frequency_penalty = 0.0{kwargs} }}
""")
        registry.write_text(registry.read_text() + ''.join(sections))
        argv = ['--model', str(model), '--served-model-name', spec['served_name'], '--host', '127.0.0.1',
                '--port', str(spec['port']), '--tensor-parallel-size', str(spec['tensor_parallel']),
                '--data-parallel-size', str(spec['data_parallel']), '--dtype', 'bfloat16',
                '--max-model-len', '40960', '--gpu-memory-utilization', '0.90',
                '--max-num-seqs', str(spec['max_num_seqs']), '--max-num-batched-tokens', str(spec['max_num_batched_tokens'])]
        if spec.get('language_model_only'):
            argv.append('--language-model-only')
        if spec.get('reasoning_parser'):
            argv += ['--reasoning-parser', spec['reasoning_parser']]
        argv += ['--generation-config', 'vllm', '--seed', '0'] + list(spec.get('extra_vllm_args', []))
        configs = [f"{spec['config_prefix']}.think-{m}" for m in spec['modes']]
        bindings = {key: dict(revision, weights_path=str(model), model_section=f"d-v2-{spec['name']}-{m}",
                              vllm_arguments=argv, thinking_mode_verified=False, verification_scope='pending_live_synthetic')
                    for key, m in zip(configs, spec['modes'])}
        write(folder / 'probe_bindings.json', bindings)
        rel = f"results/d_{spec['name']}"
        contract = json.loads((BASE_REPO / 'results/d_gemma_e4b/probe_contract.json').read_text())
        contract.update(bindings=f'{rel}/probe_bindings.json', records=rel, configurations=configs, gpu=spec['gpus'],
                        experiments=str(dest / 'outputs'), wall_seconds=7200, workers=spec['workers'],
                        committed_files=['models_v2.toml', 'explore/families_v2_supervise.py', f'{rel}/probe_bindings.json',
                                         f'{rel}/weights.json', f'{rel}/upstream.json', f'{rel}/launch_spec.json'])
        write(folder / 'probe_contract.json', contract)
        run([PYTHON, '-m', 'explore.families_v2_prepare', '--record-dir', str(folder)], cwd=repo)

        def commit(message: str) -> str:
            run(['git', 'add', 'models_v2.toml', 'explore/families_v2_supervise.py', rel], cwd=repo)
            run(['git', '-c', 'user.name=Claude automation', '-c', 'user.email=claude@localhost', 'commit', '-q', '-m', message], cwd=repo)
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        save(state='synthetic_probe', probe_commit=commit(f"D-2a: freeze {spec['name']} synthetic probe"))
        run([PYTHON, '-m', 'explore.families_v2_supervise', '--contract', str(folder / 'probe_contract.json'),
             '--run-dir', str(dest / 'probe')], cwd=repo)
        status = json.loads((dest / 'probe/status.json').read_text())
        require(status['state'] == 'complete' and status['scope'] == 'synthetic_only_no_benchmark_calls'
                and status['gpu_memory_after_mib'] == 0, 'probe/cleanup incomplete: ' + str(status.get('error')))
        attempt = Path(status['attempt'])
        probe = json.loads((attempt / 'probe.json').read_text())
        audit = summarize(attempt / 'probe_audit')
        require(probe['status'] == 'passed' and audit['usage_complete'], 'probe request evidence incomplete')
        events = [json.loads(x.read_bytes()) for x in (attempt / 'probe_audit').glob('*.json')]
        episodes = {x['session']: x for x in events if x['event'] == 'episode_complete'}
        completions = {x['call_id']: x for x in events if x['event'] == 'request_complete'}
        for key in bindings:
            rows = [x for x in probe['records'] if x['configuration'] == key]
            require(len(rows) == 8, 'missing probe cases for ' + key)
            record = next(x['record'] for x in rows if x['case'] == 'short_initial')
            require(record['passed'] and bool(record['attempts'][0].get('reasoning')) == key.endswith('.think-on'),
                    'thinking/code probe differs for ' + key)
            require(record_bytes(episodes[record['audit_session']]['record']) == record_bytes(record), 'probe record differs from journal')
            for response in record['attempts']:
                raw = completions[response['call_id']]
                require(raw['completion']['text'] == response['response'] and raw['budget'] == response['budget'],
                        'probe response differs from journal')
        write(folder / 'live_review.json', dict(status='passed', probe_sha256=sha256((attempt / 'probe.json').read_bytes()),
                                                audit=audit, attempt=str(attempt), probe_commit=state['probe_commit'],
                                                formal_calls_before_review=0))
        write(folder / 'live_probe.json', probe)
        write(folder / 'probe_status.json', status)
        for binding in bindings.values():
            binding.update(thinking_mode_verified=True, verification_scope='live_synthetic_passed')
        write(folder / 'ready_bindings.json', bindings)
        contract.update(probe_only=False, bindings=f'{rel}/ready_bindings.json', wall_seconds=spec['wall_seconds'])
        contract['committed_files'] += [f'{rel}/ready_bindings.json', f'{rel}/live_review.json', f'{rel}/live_probe.json',
                                        f'{rel}/probe_status.json']
        write(folder / 'ready_contract.json', contract)
        save(state='formal_D', formal_commit=commit(f"D-2a: freeze live-reviewed {spec['name']} binding"))
        wait_port(spec['port'])
        run([PYTHON, '-m', 'explore.families_v2_supervise', '--contract', str(folder / 'ready_contract.json'),
             '--run-dir', str(dest / 'run')], cwd=repo)
        final = json.loads((dest / 'run/status.json').read_text())
        save(state='complete' if final['state'] == 'complete' else 'stopped_incomplete', final=final)
    except BaseException as exc:
        save(state='stopped_incomplete', error=type(exc).__name__ + ': ' + str(exc))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
