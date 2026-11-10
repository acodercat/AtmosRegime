"""Post-hoc V1 answer audit. No model calls; no edits to frozen inputs or observations.
Run from repository root: PYTHONPATH=. .venv/bin/python results/d_execution/evidence/given_audit_v1/build.py
Manual labels below are AI code-review judgments, not independent expert approvals.
"""
import ast
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
from collections import Counter
from atmoscoder_rl.engine import extract_code,run_solver
from atmoscoder_rl.grading import verify
from atmoscoder_rl.prompts import user_prompt

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'experiments/d_context_incident_20261010/stopped_snapshot.tar.gz'
INPUT=ROOT/'experiments/_inputs/families_v4/templates.json'
NAMES=['deposition_brownian','drop_ventilation','semivolatile_partitioning_temperature','capillary_rise_gravity','limiting_reagent','cooper_jacob_early_time','bay_tidal_resonance','evaporative_cooling_limit','ice_above_flotation']
LABELS={}
def label(name,ids,category,note):
    for number in ids:
        key=f'tpl_20261008_{name}_{number:02d}_given'
        assert key not in LABELS,key
        LABELS[key]=(category,note)
label('deposition_brownian',[i for i in range(1,21) if i!=3],'background_relation_error','Cunningham factor uses diameter/radius convention or coefficients inconsistent with reference; drift-diffusion replacement is used. Given names C but omits its equation. Isolated C replacement tested below.')
label('limiting_reagent',[1,2,3,6,8,9,10,11,13,15,16,17,20],'water_to_air_volume_missing','Uses limiting reactant correctly but reports concentration per water volume; omits liquid-water volume per air volume.')
label('ice_above_flotation',[1,2,4,6,12,14,18,19,20],'density_conversion_missing','Computes above-flotation ice thickness but treats that ice volume as meltwater volume; missing rho_ice/rho_water conversion. Row 02 also returns a nonstandard dict.')
label('ice_above_flotation',[9,15],'replacement_ignored','Returns meltwater equivalent of full ice thickness without subtracting refilled basin.')
label('bay_tidal_resonance',[4,5,10,12,13,15,19],'depth_factor_missing','Standing-wave expression lacks division by bay depth H; numerical output is reference current times H.')
label('bay_tidal_resonance',[7],'continuity_derivation_error','Uses eta*omega/H without integrating changing water volume along bay; computed standing-wave coefficient is unused.')
label('bay_tidal_resonance',[16],'continuity_derivation_error','Uses eta*k*tan(kL), rather than eta*omega/(kH)*tan(kL).')
label('bay_tidal_resonance',[17],'continuity_derivation_error','Uses eta/sqrt(gH)*tan(kL), missing g; reference divided by g.')
label('bay_tidal_resonance',[18],'standing_wave_factor_missing','Computes mouth-normalized standing-wave coefficient but output uses progressive-wave eta*g*k/omega, omitting tan(kL).')
label('semivolatile_partitioning_temperature',[1,4,6,8,10,14,15,16,18,19],'gas_particle_fraction_reversed','Temperature correction is applied, but output equals gas-phase fraction instead of particle-phase fraction.')
label('semivolatile_partitioning_temperature',[2,3,11,12],'background_partition_relation_error','Temperature correction is applied; uses Cstar/(1+Cstar), dropping absorbing aerosol concentration from the fraction.')
label('semivolatile_partitioning_temperature',[5,17],'background_partition_relation_error','Temperature correction is applied; inserts Cstar/(R*T) as partition coefficient, inconsistent with stated mass-concentration partitioning.')
label('semivolatile_partitioning_temperature',[13],'background_partition_relation_error','Returns C_OA/Cstar as concentration; ignores C_total and fraction normalization.')
label('cooper_jacob_early_time',[2,8,10,13,16,19],'time_conversion_error','Time in minutes converted inconsistently with Q and T per day. Several rows additionally contain incorrect E1 series.')
label('cooper_jacob_early_time',[3,11,18],'special_function_series_error','E1 small-u expansion has wrong sign and/or coefficient recurrence.')
label('cooper_jacob_early_time',[6,20],'special_function_series_error','Uses large-u asymptotic expansion at small u, producing divergent/nonfinite output.')
label('cooper_jacob_early_time',[14],'zero_initialized_series','Series term initialized to zero and multiplied recursively, leaving only the logarithmic approximation.')
label('capillary_rise_gravity',[3,4,5,6,12,14,20],'incorrect_ode_solution','Substitutes simple exponential relaxation for the stated nonlinear h*dh/dt equation; some coefficients also have extra/missing powers of r.')
label('capillary_rise_gravity',[7],'early_time_shortcut','Output is Lucas-Washburn early-time square-root law despite supplied gravity equation.')
label('capillary_rise_gravity',[8],'implicit_equation_and_bracketing_error','Implicit equation has wrong time coefficient and decreasing-function bisection updates reversed; returns equilibrium endpoint.')
label('capillary_rise_gravity',[9],'implicit_equation_sign_error','Uses x-log(1-x) where integrated equation requires -x-log(1-x).')
label('capillary_rise_gravity',[10,17],'radius_power_error','Implicit-equation time coefficient uses r^3 instead of r^2.')
label('capillary_rise_gravity',[16],'implicit_equation_and_iteration_error','Wrong transformed time coefficient and fixed-point iteration generate negative height.')
label('capillary_rise_gravity',[19],'implicit_equation_coefficient_error','Wrong logarithm coefficient in integrated time equation; Newton iteration clips to zero.')
label('drop_ventilation',[1,2,3,4,7,10,11,12,14,15,16,20],'ventilation_factor_unused','Computes f but it has no path to the returned rate. Some rows also contain radius-unit or velocity-branch errors. Output matches unventilated shortcut.')
label('drop_ventilation',[5],'velocity_units_and_branch_error','Uses cm radius against micrometre thresholds and multiplies cm/s by 100 when converting to m/s.')
label('drop_ventilation',[6,8],'radius_divisor_missing','Uses supplied equation for r*dr/dt as dr/dt; missing division by r.')
label('drop_ventilation',[17,18,19],'velocity_branch_units_error','Compares centimetre radius against threshold 30 stated in micrometres, choosing wrong fall-speed regime.')
label('evaporative_cooling_limit',[1],'root_direction_and_quantity_error','Bisection updates reversed for decreasing residual; returns final absolute temperature instead of requested cooling.')
label('evaporative_cooling_limit',[2,7],'initial_humidity_ignored','Initial mixing ratio is saturation mixing ratio; input relative humidity is not used to initialize water vapour.')
label('evaporative_cooling_limit',[3],'pressure_units_in_dewpoint','Forms dewpoint logarithm with vapour pressure in Pa divided by 6.112 hPa; root bracket invalid.')
label('evaporative_cooling_limit',[4],'humidity_percent_error','Multiplies saturation vapour pressure by RH=50 rather than RH/100.')
label('evaporative_cooling_limit',[5,13,17],'root_direction_error','Bisection interval updates are reversed for the defined decreasing energy residual.')
label('evaporative_cooling_limit',[10],'temperature_units_and_humidity_error','Mixes Celsius unknown with Kelvin target in energy equation and initializes vapour at saturation without RH.')
label('evaporative_cooling_limit',[15],'temperature_units_and_ratio_error','Feeds Kelvin into Celsius vapour-pressure formula and uses inverse molecular/gas-constant ratio.')
label('evaporative_cooling_limit',[16],'pressure_units_and_root_direction_error','Mixes hPa vapour pressure with Pa total pressure; bisection direction also reversed.')

CONCERNS={
'deposition_brownian':'Given supplies replacement but only names Cunningham C: ancillary empirical formula is still recalled. This is a completeness limitation, not evidence that the stored drift-diffusion key is wrong.',
'drop_ventilation':'Given explicitly supplies f and base growth terms. Mixed units in piecewise speed require conversion; observed code errors do not establish a prompt/key contradiction. Existing wording caveat remains; do not alter frozen D.',
'semivolatile_partitioning_temperature':'Given supplies Cstar(T), but not particle fraction 1/(1+Cstar/C_OA). Low given cannot isolate inability to use temperature correction.',
'capillary_rise_gravity':'Given supplies nonlinear ODE, not its integrated solution; solving the ODE remains part of the task. This computational burden is not missing physical information.',
'limiting_reagent':'Given supplies limiting-reactant rule; water-volume to air-volume conversion remains necessary and is explicitly requested.',
'cooper_jacob_early_time':'Given names E1(u) but does not define its integral/series; standard-library-only answer requires knowledge/implementation of this special function.',
'bay_tidal_resonance':'Given supplies elevation shape and volume-conservation hint, not final current formula; derivation/integration remains necessary.',
'evaporative_cooling_limit':'Given omits saturation-vapour-pressure relation and requires nonlinear solution. This mechanism is already excluded from T by prior evidence; this audit does not rehabilitate it.',
'ice_above_flotation':'Given specifies above-flotation ice thickness; conversion from ice to water volume remains necessary. It is not a water-equivalent thickness.'}

def digest(blob):return hashlib.sha256(blob).hexdigest()
def same(a,b):return a==b or (isinstance(a,(int,float)) and isinstance(b,(int,float)) and ((math.isnan(a) and math.isnan(b)) or math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10)))
def printable(value):return str(value) if isinstance(value,float) and not math.isfinite(value) else value

def main():
    with tarfile.open(ARCHIVE) as tar:
        raw=tar.extractfile('formal/qwen3.5-9b.think-off.templates.run1.json').read()
        proof=json.load(tar.extractfile('formal/receipts/qwen3.5-9b.think-off.templates.run1.json'))
        seal=json.load(tar.extractfile('formal/verified/qwen3.5-9b.think-off.templates.run1.json'))
    archived=json.loads((ROOT/'results/d_context_incident/archive_manifest.json').read_bytes())
    assert digest(ARCHIVE.read_bytes())==archived['archive_sha256']
    doc=json.loads(raw);items=json.loads(INPUT.read_bytes());by_id={x['id']:x for x in items};records={x['id']:x for x in doc['results']}
    assert proof['input_sha256']==digest(INPUT.read_bytes())
    assert seal['result_sha256']==digest(raw)
    source_names=['atmoscoder_rl/engine.py','atmoscoder_rl/grading.py','atmoscoder_rl/evaluation/episode.py','atmoscoder_rl/prompts.py']+['atmoscoder_rl/templates/'+n+'.py' for n in NAMES]
    source_hashes={}
    for name in source_names:
        old=subprocess.check_output(['git','show',proof['commit']+':'+name],cwd=ROOT)
        assert old==(ROOT/name).read_bytes(),name
        source_hashes[name]=digest(old)
    summary={};rows=[];replays=[];seen=set()
    for name in NAMES:
        module=importlib.import_module('atmoscoder_rl.templates.'+name)
        selected=[x for x in items if x['template']==name];assert len(selected)==80
        for item in selected:
            assert records[item['id']]['attempts'][0]['prompt']==user_prompt(item['problem'])
            expected=float(item['sub_answers'][0]['value'])
            assert math.isclose(module.correct(item['params']),expected,rel_tol=1e-5),item['id']
        given=sorted([x for x in selected if x['variant']=='given'],key=lambda x:x['id']);assert len(given)==20
        gold_check=module.check()
        for item in given:
            r=records[item['id']];identifier=item['id'];seen.add(identifier)
            assert item['params']==by_id[identifier.removesuffix('_given')+'_outside']['params']
            assert not r.get('error') and not r.get('failed_reason')
            code=extract_code(r['attempts'][-1]['response'])
            ok,answer,error=run_solver(code);assert ok,(identifier,error)
            passed,details=verify(answer,item['sub_answers'],.05,units='benchmark')
            assert passed==r['passed'],identifier
            actual=details[0]['actual'];assert same(actual,r['details'][0]['actual']),identifier
            if r['passed']:category,note='passed','Passes frozen 5% benchmark criterion; this does not certify every intermediate step.'
            else:category,note=LABELS[identifier]
            hypothesis=None
            p=item['params'];key=module.correct(p)
            if category=='water_to_air_volume_missing':hypothesis=actual*p['lwc']/1e6
            if category=='density_conversion_missing':hypothesis=actual*.917
            if category=='depth_factor_missing':hypothesis=actual/p['h']
            if category=='gas_particle_fraction_reversed':hypothesis=p['ctot']*1000-actual
            if category=='ventilation_factor_unused':
                assert math.isclose(actual,module.shortcut(p),rel_tol=1e-5)
                hypothesis=actual*module._fvent(p['r'])
                tree=ast.parse(code)
                loads=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='f' and isinstance(n.ctx,ast.Load)]
                assert not loads,identifier
            if category=='radius_divisor_missing':hypothesis=actual/(p['r']*1e-6)
            if hypothesis is not None:assert math.isclose(hypothesis,key,rel_tol=.001),(identifier,hypothesis,key)
            if name=='deposition_brownian' and not passed:
                tree=ast.parse(code);replaced=0
                for node in ast.walk(tree):
                    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('C','C_factor'):
                        node.value=ast.Constant(module._cunningham(p['d']));replaced+=1
                assert replaced==1
                changed=ast.unparse(ast.fix_missing_locations(tree));ok,result,error=run_solver(changed)
                assert ok and verify(result,item['sub_answers'],.05,units='benchmark')[0],(identifier,error)
                hypothesis=result['1']['value']
            pair={v:records[identifier.removesuffix('_given')+'_'+v]['passed'] for v in ('inside','outside','given')}
            pair['flag']=records['flag:'+identifier.removesuffix('_given')+'_outside']['passed']
            rows.append(dict(id=identifier,template=name,passed=passed,category=category,actual=printable(actual),expected=item['sub_answers'][0]['value'],paired_passes=json.dumps(pair,sort_keys=True),note=note,response_sha256=digest(r['attempts'][-1]['response'].encode()),diagnostic_only_corrected_value=hypothesis))
            replays.append(dict(id=identifier,source_record_sha256=digest(json.dumps(r,sort_keys=True,allow_nan=True).encode()),same_verdict=True,same_numeric_output=True,diagnostic_only_corrected_value=hypothesis))
        subset=[r for r in rows if r['template']==name]
        summary[name]=dict(given_passes=sum(r['passed'] for r in subset),given_total=20,categories=dict(Counter(r['category'] for r in subset)),prompt_assessment=CONCERNS[name],existing_numerical_crosscheck=gold_check)
        print(name,'reviewed and replayed',flush=True)
    assert set(LABELS)=={r['id'] for r in rows if not r['passed']}
    assert len(rows)==180 and len(seen)==180
    with (OUT/'rows.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    (OUT/'replay.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in replays))
    result=dict(scope='post_hoc_V1_only_no_new_model_calls',reviewer='single Codex AI reviewer; code reading plus mechanical replay; not independent expert physical approval',source_commit=proof['commit'],input_sha256=digest(INPUT.read_bytes()),result_sha256=digest(raw),archive_sha256=digest(ARCHIVE.read_bytes()),source_hashes=source_hashes,covered_given_rows=180,paired_input_and_key_checks=720,reviewed_response_scope='Final gradable code for each given row; all four paired initial prompts and stored keys checked; intermediate failed repair code not classified.',passes=sum(r['passed'] for r in rows),failures=sum(not r['passed'] for r in rows),replay_matches=180,classification='Each wrong row has a specific code-level discrepancy; categories name principal observed error, not exclusive psychological causes. Some passed programs contain latent inaccuracies within tolerance.',limits=['Selected nine mechanisms from observed V1 low given accuracy; no prevalence inference over all mechanisms/models.','Numerical crosschecks reuse repository alternative algorithms and are not a fresh external physics review.','Diagnostic corrections are post-hoc arithmetic/code checks, not new model observations, new scores, or evidence of learning.','No frozen task, score, threshold, or running evaluation was changed.'],mechanisms=summary,artifacts={n:digest((OUT/n).read_bytes()) for n in ('rows.tsv','replay.jsonl','build.py')})
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',result['passes'],result['failures'])

if __name__=='__main__':main()
