"""D v2 runner: exact resume identity, durable history, bounded work after an error."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path

from ..evaluation.client import load_config
from ..frozen import json_bytes, require, sha256
from ..prompts import SYSTEM
from ..paths import REPO_ROOT
from .client import Journal, PROTOCOL, build_model
from .episode import evaluate
from .audit import record_bytes
from .recovery import recover

EXP_ID = 'families_context_v2'


def atomic_write(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        handle.write(record_bytes(value)); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Store:
    def __init__(self, path, meta):
        self.path, self.meta = path, meta
        self.results, self.superseded = {}, []
        if path.exists():
            doc = json.loads(path.read_text())  # Corruption is fatal, never a fresh run.
            require(doc['identity'] == meta, 'resume identity mismatch')
            self.results = {r['id']: r for r in doc['results']}
            require(len(self.results) == len(doc['results']), 'duplicate stored ids')
            self.superseded = doc['superseded']

    @property
    def done_ids(self):
        return {i for i, r in self.results.items() if not r.get('error')}

    def add(self, record):
        previous = self.results.get(record['id'])
        if previous:
            require(bool(previous.get('error')), 'refusing to replace completed scientific observation')
            self.superseded.append(previous)
        self.results[record['id']] = record
        passed = sum(r['passed'] is True for r in self.results.values())
        errors = sum(bool(r.get('error')) for r in self.results.values())
        failed = len(self.results) - passed - errors
        atomic_write(self.path, dict(identity=self.meta, metrics=dict(self.meta, passed=passed,
                     failed=failed, errors=errors, total=len(self.results),
                     accuracy=passed/(passed+failed) if passed+failed else None,
                     usage_note='Use immutable request journal for all-session cost; superseded/crashed calls remain there.'),
                     results=list(self.results.values()), superseded=self.superseded))


def run_job(model_section, input_path, output_path, out_name, repetition, workers=8):
    require(workers > 0, 'workers must be positive')
    config = load_config(model_section, REPO_ROOT / "models_v2.toml")
    meta = dict(model=out_name, model_id=config['model_id'], exp_id=EXP_ID, run=repetition,
                protocol=PROTOCOL, input_sha256=sha256(input_path.read_bytes()),
                client={k:v for k,v in config.items() if k != 'api_key'},
                mode='code', tolerance=.05, units='benchmark', system=SYSTEM, max_attempts=5)
    items = json.loads(input_path.read_text())
    ids = {p['id'] for p in items}
    require(len(ids) == len(items), 'duplicate input ids')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(output_path, meta)
        require(set(store.results) <= ids, 'unexpected stored ids')
        journal_dir = output_path.parent / 'audit' / output_path.stem
        recover(store, journal_dir, meta, items)
        remaining = iter(p for p in items if p['id'] not in store.done_ids)
        def one(problem):
            journal = Journal(journal_dir, dict(job=output_path.name, problem=problem['id'],
                                                input_sha256=meta['input_sha256'],
                                                run_identity_sha256=sha256(json_bytes(meta))))
            model = build_model(model_section, journal)
            try:
                return evaluate(problem, model, .05)
            finally:
                model.client.close()
        stopped = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = set()
            def fill():
                while len(pending) < workers and not stopped:
                    problem = next(remaining, None)
                    if problem is None:
                        break
                    pending.add(pool.submit(one, problem))
            fill()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    record = future.result()
                    store.add(record)
                    stopped |= bool(record.get('error'))
                    print(json.dumps(dict(id=record['id'], saved=len(store.results),
                                          error=record.get('error'), attempts=record['num_attempts'])), flush=True)
                fill()
        return 1 if stopped else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out-name', required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--experiments', type=Path, required=True)
    args = ap.parse_args()
    require('/' not in args.out_name and args.run > 0, 'invalid output identity')
    output = args.experiments / EXP_ID / f'{args.out_name}.run{args.run}.json'
    return run_job(args.model, args.input, output, args.out_name, args.run, args.workers)


if __name__ == '__main__':
    raise SystemExit(main())
