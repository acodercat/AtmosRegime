"""Synthetic D checks: protocol schedule, equal cluster weighting, gate boundaries and incomplete runs."""
import json
from pathlib import Path
import tempfile
from .diagnosis import configs, schedule, template_summary, verdict, load_complete, textbook_summary
from .coverage_clusters import units, unit_of
from .frozen import json_bytes
from .templates import ALL_TEMPLATES


def main() -> int:
    assert len(configs())==16 and len(schedule())==64
    assert all(c['mode']=='off' for c in configs() if c['family']=='Ministral 3')
    rows=[];rep={}
    for name in ALL_TEMPLATES:
        for v in ('inside','outside','given','flag'):
            for i in range(20):
                rid=f'{name}:{v}:{i}'
                rows.append({'id':rid,'template':name,'variant':v})
                rep[rid]={'id':rid,'passed':v!='outside','num_attempts':1}
    result=template_summary(rows,rep,draws=200)
    assert result['usable'] and result['usable_only']['cluster_count']==32
    assert result['usable_only']['contrasts']['inside_minus_outside']['mean']==1
    # Make one whole cluster solve outside. Its contribution is 1/32 regardless of template count.
    chosen=max(units(),key=lambda c:len(units()[c]))
    for r in rows:
        if r['template'] in units()[chosen] and r['variant']=='outside': rep[r['id']]['passed']=True
    result=template_summary(rows,rep,draws=200)
    assert abs(result['usable_only']['contrasts']['inside_minus_outside']['mean']-31/32)<1e-12
    # One mechanism with inside 9/20 is excluded from usable-only but retained in all-template results.
    name=next(iter(ALL_TEMPLATES))
    for i in range(9,20): rep[f'{name}:inside:{i}']['passed']=False
    result=template_summary(rows,rep,draws=200)
    assert not result['templates'][name]['usable'] and len(result['templates'])==72
    # Use real trap metadata, synthetic verdicts only: no model calls or scientific results.
    from .datasets import index_by_id,load_core,load_traps,split_ids
    from .traps import sound,family
    train=set(split_ids('train'))
    traps=[t for t in load_traps() if t['parent_id'] not in train and sound(t)]
    core=index_by_id(load_core())
    items=traps+[core[p] for p in sorted({t['parent_id'] for t in traps})]
    trials=[{t['id']:{'passed':not ('parent_id' in t and family(t)=='regime:recognise')} for t in items} for _ in range(3)]
    text=textbook_summary(items,trials,draws=200)
    assert text['regime_minus_geometry']['mean']==1 and text['usable']
    assert verdict(text,result)=='deficit_established'
    # Selection on inside/given success can reverse the sign. D-1 must use every
    # cluster, even though 15 usable clusters are enough to pass the eligibility gate.
    usable_clusters=set(sorted(units())[:15])
    for row in rows:
        eligible=unit_of(row['template']) in usable_clusters
        successes={'inside':15 if eligible else 0,'outside':10 if eligible else 5,
                   'given':20 if eligible else 0,'flag':0}[row['variant']]
        rep[row['id']]['passed']=int(row['id'].rsplit(':',1)[1])<successes
    reversal=template_summary(rows,rep,draws=200)
    assert reversal['usable']
    assert reversal['usable_only']['contrasts']['inside_minus_outside']['mean']==.25
    assert reversal['all']['contrasts']['inside_minus_outside']['mean']<0
    assert verdict(text,reversal)=='threshold_not_met'
    text['regime_minus_geometry']['mean']=.149
    assert verdict(text,result)=='threshold_not_met'
    text['usable']=False
    assert verdict(text,result)=='not_eligible'
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'run.json';items=[{'id':'one'},{'id':'two'}]
        assert load_complete(p,items)[1]
        meta={'units':'benchmark','tolerance':.05,'mode':'code'}
        complete={'metrics':meta,'results':[{'id':i['id'],'passed':False,'num_attempts':1} for i in items]}
        p.write_bytes(json_bytes(complete));assert not load_complete(p,items)[1]  # real failures count
        complete['results'][0].update(error='API unavailable',num_attempts=0)
        p.write_bytes(json_bytes(complete));assert load_complete(p,items)[1]  # API failures do not
        complete['results']*=2
        p.write_bytes(json_bytes(complete));assert load_complete(p,items)[1]
    print('PASS: 64-job schedule, cluster weighting, usability/decision thresholds and missing/API-error checks')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
