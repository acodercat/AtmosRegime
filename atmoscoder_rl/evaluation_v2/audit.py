"""All-session journal accounting; unknown requests never become zero-cost claims."""
import json
from pathlib import Path
from ..evaluation.client import USAGE_KEYS
from ..frozen import json_bytes, require, sha256
from .usage import inspect_usage


def summarize(path: Path):
    files, starts, terminals = {}, {}, {}
    usage = dict.fromkeys(USAGE_KEYS, 0)
    unknown=set()
    invalid=set()
    for file in sorted(path.glob('*.json')):
        blob = file.read_bytes()
        row = json.loads(blob)
        files[file.name] = sha256(blob)
        event = row['event']
        if event == 'request_start':
            require(row['call_id'] not in starts, 'duplicate request start')
            starts[row['call_id']] = row
        elif event in {'request_complete', 'request_error'}:
            require(row['call_id'] not in terminals, 'duplicate request terminal')
            terminals[row['call_id']] = row
            evidence=inspect_usage(row.get('provider_usage'))
            if not evidence.complete:
                invalid.add(row['call_id'])
            for key,value in evidence.counts.items():
                if value is None:
                    unknown.add(key)
                else:
                    usage[key]+=value
            if evidence.counts['total_tokens'] is None:
                usage['total_tokens']+=sum(evidence.counts[k] or 0
                                          for k in ('prompt_tokens','completion_tokens'))
    require(set(terminals) <= set(starts), 'terminal without durable request start')
    uncertain = sum(i not in terminals or terminals[i]['event'] == 'request_error'
                    or i in invalid or not terminals[i].get('usage_complete') for i in starts)
    if set(starts)-set(terminals):
        unknown.update(('prompt_tokens','completion_tokens','total_tokens'))
    return dict(sha256=sha256(json_bytes(files)), events=len(files), requests=len(starts),
                uncertain_requests=uncertain, usage_complete=uncertain == 0,
                known_reported_usage=usage,
                usage={k:None if k in unknown else v for k,v in usage.items()},
                malformed_usage_requests=len(invalid),
                reasoning_note='Reasoning token share is 0 when provider does not report it; it is not inferred.')


def record_bytes(value):
    # Match v1's record encoding for legitimate +/-inf and model-produced NaN.
    # These are Python JSON extensions, not silently converted to finite values.
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=True)+'\n').encode()


def verify_records(path: Path, records: list[dict]):
    episodes, completions = {}, {}
    for file in path.glob('*.json'):
        row=json.loads(file.read_text())
        if row['event']=='episode_complete':
            require(row['session'] not in episodes, 'duplicate terminal episode')
            episodes[row['session']]=row
        elif row['event']=='request_complete':
            completions[row['call_id']]=row
    for record in records:
        saved=episodes.get(record['audit_session'])
        require(saved is not None and saved['identity']['problem']==record['id'] and
                record_bytes(saved['record'])==record_bytes(record), 'result differs from episode journal')
        for attempt in record.get('attempts',[])+record.get('partial_attempts',[]):
            call=completions.get(attempt['call_id'])
            require(call is not None and call['session']==record['audit_session'] and
                    call['completion']['text']==attempt['response'] and call['budget']==attempt['budget'],
                    'response or budget differs from immutable request journal')
