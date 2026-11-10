"""Actual-serving synthetic boundaries. No benchmark inputs, answers, or accuracy selection."""
import json
from pathlib import Path

from atmoscoder_rl.evaluation.client import ChatModel
from atmoscoder_rl.evaluation.episode import REPAIR_PROMPT, _attempt
from atmoscoder_rl.evaluation_v2.client import build_model, Journal, ProtocolError, CONTEXT
from atmoscoder_rl.evaluation_v2.episode import evaluate
from atmoscoder_rl.frozen import json_bytes, require, write_bundle
from atmoscoder_rl.paths import REPO_ROOT
from atmoscoder_rl.prompts import SYSTEM


def probe(contract, destination):
    bindings=json.loads((REPO_ROOT/contract['bindings']).read_text())
    problem=dict(id='synthetic-serving-only', problem='A counter starts at 17 and increases by 25. Return the final count (dimensionless).',
                 sub_answers=[dict(sub='1',value='42',unit='')])
    records=[]
    for key in contract['configurations']:
        journal=Journal(destination.parent/'probe_audit',dict(configuration=key,problem='synthetic'))
        model=build_model(bindings[key]['model_section'],journal)
        try:
            record=evaluate(problem,model,.05,max_attempts=1)
            require(record.get('passed') is True and not record.get('error'),'short code probe failed: '+key)
            require(bool(record['attempts'][0].get('reasoning'))==key.endswith('.think-on'),
                    'reasoning toggle mismatch: '+key)
            records.append(dict(configuration=key,case='short_initial',record=record))
            # All following calls are synthetic repairs; no question/response is truncated.
            model.content_try=1
            for name,code in [('short_repair','def solve():\n return unknown_variable'),
                              ('long_reasoning_no_final_code','Let us think. '*3500),
                              ('long_code','def solve():\n'+(' # synthetic comment\n'*8000)+' return unknown_variable'),
                              ('unicode','def solve():\n'+(' # 温度 ΔT 🌊\n'*2200)+' return unknown_variable')]:
                def repair_prompt(candidate):
                    outcome=_attempt(candidate)
                    return REPAIR_PROMPT.format(problem=problem['problem'],code=outcome.code,
                                               error=outcome.error or 'synthetic undefined variable')
                prompt=repair_prompt(code)
                if name in {'long_code','unicode'}:
                    line=' # synthetic comment\n' if name=='long_code' else ' # 温度 ΔT 🌊\n'
                    repeats=100
                    for _ in range(6):
                        prompt=repair_prompt('def solve():\n'+line*repeats+' return unknown_variable')
                        size=model.tokenize(ChatModel._request(model,prompt,SYSTEM))['prompt_tokens']
                        if 34500<=size<=35500:
                            break
                        repeats=max(1,round(repeats*35000/size))
                    require(34500<=size<=35500,'synthetic long prompt construction failed')
                completion=model.generate(prompt,SYSTEM)
                records.append(dict(configuration=key,case=name,call=model.last_call,usage=completion.usage))
                require(model.last_call['budget']['prompt_tokens']+model.last_call['budget']['max_tokens']<=CONTEXT,
                        'budget sum overflow')
                if name=='long_code':
                    require(model.last_call['budget']['max_tokens'] < model.max_tokens,'long code did not exercise cap reduction')
                write_bundle(destination.parent,{destination.name+'.'+key+'.'+name:json_bytes(records[-1])})
            # Exact final-token boundary, using the SAME serving template, including system and thinking prefix.
            def count(n):
                return model.tokenize(ChatModel._request(model,'x '*n,SYSTEM))['prompt_tokens']
            n=CONTEXT-1-count(1)+1
            for _ in range(4):
                actual=count(n)
                if actual==CONTEXT-1:
                    break
                n+=CONTEXT-1-actual
            require(count(n)==CONTEXT-1,'cannot construct exact one-token boundary')
            completion=model.generate('x '*n,SYSTEM)
            require(model.last_call['budget']['max_tokens']==1,'one-token boundary not applied')
            records.append(dict(configuration=key,case='remaining_one',call=model.last_call,usage=completion.usage))
            before=len(list((destination.parent/'probe_audit').glob('*.json')))
            for name,prompt,content_try in [('prompt_overfull','x '*(n+2),1),
                                             ('initial_must_keep_cap','x '*15000,0)]:
                model.content_try=content_try
                # Off initial cap allows 15k; use 35k to exceed both caps.
                if name=='initial_must_keep_cap': prompt='x '*35000
                try:
                    model.generate(prompt,SYSTEM)
                except ProtocolError:
                    records.append(dict(configuration=key,case=name,rejected_before_generation=True))
                else:
                    raise ValueError('expected boundary rejection: '+name)
            require(len(list((destination.parent/'probe_audit').glob('*.json')))>before,'missing rejected-request evidence')
        finally:
            model.client.close()
    write_bundle(destination.parent,{destination.name:json_bytes(dict(status='passed',records=records))})
