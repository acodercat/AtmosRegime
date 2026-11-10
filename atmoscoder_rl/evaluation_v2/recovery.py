"""Restore durable completed observations, or fail closed before any new request."""
import json
from collections import defaultdict
from ..frozen import require,json_bytes,sha256
from .audit import record_bytes,verify_records
from .client import PROTOCOL


def recover(store,path,meta,items):
    files=list(path.glob('*.json'))
    if not files:
        require(not store.results and not store.superseded, 'result cache has no audit history')
        return
    expected_hash=sha256(json_bytes(meta))
    ids={p['id'] for p in items}
    sessions=defaultdict(list)
    for file in files:
        row=json.loads(file.read_text())
        identity=row['identity']
        require(row['protocol']==PROTOCOL and identity.get('job')==store.path.name and
                identity.get('input_sha256')==meta['input_sha256'] and identity.get('problem') in ids,
                'audit identity differs from current job/input')
        require(identity.get('run_identity_sha256')==expected_hash,
                'audit lacks matching full execution identity; legacy logs require explicit verified migration')
        sessions[row['session']].append(row)
    episodes=[]
    for rows in sessions.values():
        identities={record_bytes(r['identity']) for r in rows}
        require(len(identities)==1, 'session identity changed')
        terminal=[r for r in rows if r['event']=='episode_complete']
        require(len(terminal)<=1, 'duplicate episode terminal')
        if not terminal:
            require(not any(r['event'].startswith('request_') for r in rows),
                    'unfinished generation session: preserve/reconcile existing response before any retry')
            continue  # Only tokenization occurred; no model request was dispatched.
        row=terminal[0]
        require(row['record']['audit_session']==row['session'] and row['record']['id']==row['identity']['problem'],
                'episode result identity differs')
        episodes.append(row)
    verify_records(path,[r['record'] for r in episodes])
    verify_records(path,list(store.results.values())+store.superseded)
    complete=defaultdict(list)
    for row in episodes:
        if not row['record'].get('error'):
            complete[row['record']['id']].append(row)
    require(all(len(rows)==1 for rows in complete.values()),
            'multiple completed observations for one problem; refusing to choose one')
    for problem_id,rows in complete.items():
        current=store.results.get(problem_id)
        if current and not current.get('error'):
            require(record_bytes(current)==record_bytes(rows[0]['record']), 'completed observation changed')
    # All checks finish before modifying the cache. Repeated recovery is idempotent.
    seen={r['audit_session'] for r in list(store.results.values())+store.superseded}
    planned=[]
    done=set(store.done_ids)
    for row in sorted(episodes,key=lambda r:(r['time_ns'],r['session'])):
        record=row['record']
        if record['audit_session'] in seen:
            continue
        require(record['id'] not in done,
                'a later episode exists after a completed observation; refusing replacement')
        planned.append(record)
        seen.add(record['audit_session'])
        if not record.get('error'):
            done.add(record['id'])
    for record in planned:
        store.add(record)
