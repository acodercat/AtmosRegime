"""Verify a local vLLM process, its model files and requested thinking mode before D.

Create a content revision (streaming reads; no inference):
    python -m atmoscoder_rl.serving_binding --weights /absolute/model/path --out revision.json

Supported launch: python -m vllm.entrypoints.openai.api_server <vllm_arguments>.
Run on the serving host. Unsupported launchers/mode switches fail closed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .frozen import json_bytes, require, sha256, write_bundle


def weight_files(root: Path) -> dict[str, Path]:
    require(root.is_absolute() and root.is_dir(), 'weights_path must be an existing absolute directory')
    # Includes model, tokenizer, configuration and custom code; excludes download bookkeeping.
    files = {p.relative_to(root).as_posix(): p for p in root.rglob('*')
             if p.is_file() and not any(part.startswith('.') for part in p.relative_to(root).parts)}
    require('config.json' in files, 'weights lack config.json')
    require(any(n.endswith(('.safetensors', '.bin', '.pt', '.gguf')) for n in files), 'weights lack model tensors')
    return files


def weight_revision(root: Path) -> dict:
    digests = {}
    for name, path in sorted(weight_files(root).items()):
        with path.open('rb') as handle:
            digests[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'weights_manifest': digests, 'weights_revision': 'sha256:' + sha256(json_bytes(digests))}


def option(argv: list[str], flag: str) -> str:
    values = [argv[i+1] for i, arg in enumerate(argv[:-1]) if arg == flag]
    values += [arg.split('=', 1)[1] for arg in argv if arg.startswith(flag + '=')]
    require(len(values) == 1, f'expected exactly one {flag} in vllm_arguments')
    return values[0]


def validate_binding(key: str, binding: dict, config: dict) -> None:
    require(binding.get('thinking_mode_verified') is True, f'{key}: thinking mode review missing')
    require(config.get('temperature') == .3, f'{key}: expected temperature 0.3')
    root = Path(binding.get('weights_path') or '')
    files = weight_files(root)
    manifest = binding.get('weights_manifest')
    require(isinstance(manifest, dict) and set(manifest) == set(files), 'weight file inventory mismatch')
    require(all(isinstance(d, str) and len(d) == 64 and set(d) <= set('0123456789abcdef')
                for d in manifest.values()), 'invalid weight digests')
    require(binding.get('weights_revision') == 'sha256:' + sha256(json_bytes(manifest)), 'weight revision mismatch')
    argv = binding.get('vllm_arguments')
    require(isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv), 'vllm_arguments must be an argv list')
    require(not any(a.startswith('--api-key') for a in argv), 'use a server environment variable for API credentials')
    model_path = Path(option(argv, '--model'))
    require(model_path.is_absolute() and model_path.resolve() == root.resolve(),
            'vLLM --model must be the absolute weights_path')
    if any(a == '--tokenizer' or a.startswith('--tokenizer=') for a in argv):
        tokenizer = Path(option(argv, '--tokenizer'))
        require(tokenizer.is_absolute() and tokenizer.resolve() == root.resolve(),
                'external or relative tokenizer is not in the weight revision')
    if any(a == '--chat-template' or a.startswith('--chat-template=') for a in argv):
        template = Path(option(argv, '--chat-template'))
        require(template.is_absolute() and template.resolve() in {p.resolve() for p in files.values()},
                'external or relative chat template is not in the weight revision')
    require(not any(a.split('=', 1)[0] in ('--lora-modules', '--enable-lora') for a in argv),
            'D base-model binding does not support unbound LoRA adapters')
    require(option(argv, '--served-model-name') == config.get('model_id'), 'served model id mismatch')
    endpoint = urlparse(config.get('base_url', ''))
    require(endpoint.scheme == 'http' and endpoint.hostname == '127.0.0.1' and endpoint.path.rstrip('/') == '/v1'
            and not endpoint.username and not endpoint.password and not endpoint.query and not endpoint.fragment,
            'D runtime verification requires http://127.0.0.1:<port>/v1 on the serving host')
    require(int(option(argv, '--port')) == endpoint.port, 'vLLM port differs from endpoint')
    extra = config.get('extra_body', {})
    require(not set(extra) & {'model', 'temperature', 'messages'}, 'extra_body overrides a frozen request field')
    switch = extra.get('chat_template_kwargs', {}).get('enable_thinking')
    mode = key.rsplit('.think-', 1)[-1]
    require(mode in ('on', 'off'), 'unknown thinking mode')
    # Ministral is pre-registered off-only and has no thinking toggle.
    if key.startswith('ministral3-') and mode == 'off':
        require(switch is None or switch is False, 'off-only model requested with thinking enabled')
    else:
        require(type(switch) is bool and switch == (mode == 'on'),
                f'{key}: enable_thinking contradicts mode or is unsupported/missing')


def process_identity(pid: int, binding: dict) -> dict:
    proc = Path('/proc') / str(pid)
    argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    require(len(argv) >= 4 and argv[1:3] == ['-m', 'vllm.entrypoints.openai.api_server'],
            'unsupported server process; expected python -m vllm.entrypoints.openai.api_server')
    require(argv[3:] == binding['vllm_arguments'], 'live vLLM arguments differ from binding')
    # Bind the PID to the listening socket, not just to a plausible command line.
    sockets = set()
    for fd in proc.joinpath('fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith('socket:['):
            sockets.add(target[8:-1])
    port = int(option(binding['vllm_arguments'], '--port'))
    listening = False
    for name in ('tcp', 'tcp6'):
        for line in proc.joinpath('net', name).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == '0A' and int(fields[1].split(':')[1], 16) == port and fields[9] in sockets:
                listening = True
    require(listening, 'specified PID does not own the endpoint listener')
    start = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    return {'pid': pid, 'start_ticks': start,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def endpoint_model(config: dict, root: Path) -> dict:
    secret = os.environ.get(config['api_key_env']) if config.get('api_key_env') else config.get('api_key')
    require(bool(secret), 'endpoint API credential is missing')
    request = Request(config['base_url'].rstrip('/') + '/models', headers={'Authorization': 'Bearer ' + secret})
    with urlopen(request, timeout=15) as response:
        data = json.load(response)
    cards = [m for m in data['data'] if m['id'] == config['model_id']]
    # vLLM ModelCard.root is the base model path (OpenAI models/serving implementation).
    require(len(cards) == 1 and Path(cards[0].get('root', '')).resolve() == root.resolve(),
            'endpoint /models does not identify the bound weights')
    return {'id': cards[0]['id'], 'root': str(root.resolve())}


def runtime_evidence(pid: int, binding: dict, config: dict, *, hash_weights: bool = True) -> dict:
    identity = process_identity(pid, binding)
    root = Path(binding['weights_path'])
    before = {n: [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns] for n,p in weight_files(root).items()}
    if hash_weights:
        revision = weight_revision(root)
        require(revision['weights_manifest'] == binding['weights_manifest'], 'weight content differs from frozen revision')
    card = endpoint_model(config, root)
    after = {n: [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns] for n,p in weight_files(root).items()}
    require(before == after and identity == process_identity(pid, binding), 'server or weights changed during verification')
    return {'process': identity, 'model': card, 'weights_revision': binding['weights_revision'],
            'file_stats': after}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    write_bundle(args.out.parent, {args.out.name: json_bytes(weight_revision(args.weights))})
    print(f'Weight revision written to {args.out}; no model calls.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
