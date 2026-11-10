"""Wait for pinned official download; freeze, probe, review, then run registered D.

No training or benchmark call before verified weights and synthetic probe pass.
Only creates its own checkout and launches scoped supervisors on 047 GPU 3.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path('/path/to/workdir/experiments/d_execution')
DEST=ROOT.parent/'d_qwen35_27b_v2'
REPO=DEST/'repo'
SOURCE=ROOT.parent/'d_gemma_ready_20261010/repo'
PYTHON='/path/to/workdir/eval-venv/bin/python'
REV='fc05daec18b0a78c049392ed2e771dde82bdf654'
BASE='83ae1c3'

def require(ok,why):
    if not ok:raise ValueError(why)

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=1)+'\n');temp.replace(path)

def run(args,**kwargs):return subprocess.run(args,check=True,**kwargs)

def main():
    require(socket.gethostname()=='host-B','wrong host')
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/'qwen35_27b_autostart.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=ROOT/'qwen35_27b_autostart.json'
    require(not statepath.exists(),'already attempted; inspect before restarting')
    state=dict(state='waiting_for_download',pid=os.getpid(),started_ns=time.time_ns())
    def save():state['updated_ns']=time.time_ns();write(statepath,state)
    save()
    try:
        deadline=time.monotonic()+86400
        while True:
            require(time.monotonic()<deadline,'download wait deadline')
            require(not (ROOT/'STOP_qwen35_27b').exists(),'operator STOP')
            download=json.loads((ROOT/'qwen35_27b_download.json').read_bytes())
            require(download['model']=='Qwen/Qwen3.5-27B' and download['revision']==REV,'wrong model/revision')
            require(download['status']!='failed','download failed: '+download.get('error',''))
            if download['status']=='weights_verified_pending_service_probe':break
            time.sleep(30)
        state['state']='preparing';save()
        require(not DEST.exists(),'isolated destination already exists')
        DEST.mkdir()
        run(['git','clone','--no-hardlinks','--no-checkout',str(SOURCE),str(REPO)])
        run(['git','checkout','--detach',BASE],cwd=REPO)
        sys.path.insert(0,str(REPO))
        from atmoscoder_rl.serving_binding import weight_revision
        from atmoscoder_rl.evaluation_v2.audit import summarize,record_bytes
        from atmoscoder_rl.frozen import sha256,json_bytes
        from explore.families_v2_prepare import source_names
        from atmoscoder_rl.evaluation_v2.client import PROTOCOL
        from atmoscoder_rl.evaluation_v2.runner import EXP_ID
        model=Path(download['destination']);revision=weight_revision(model)
        require(revision['weights_manifest']==download['verified'],'weights changed since official verification')
        upstream=json.loads((ROOT/'qwen35_27b_upstream.json').read_bytes())
        expected=[x for x in upstream['siblings'] if x['rfilename'].endswith(('.json','.jinja','.safetensors','.txt'))]
        require(upstream['sha']==REV and set(download['verified'])=={x['rfilename'] for x in expected},'official file coverage differs')
        for entry in expected:
            path=model/entry['rfilename'];require(path.stat().st_size==entry['size'],'official size differs')
            if entry.get('lfs'):require(download['verified'][entry['rfilename']]==entry['lfs']['sha256'],'official LFS hash differs')
            else:
                blob=path.read_bytes();require(hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()==entry['blobId'],'official Git hash differs')
        folder=REPO/'results/d_qwen35_27b';folder.mkdir()
        write(folder/'weights.json',revision);write(folder/'download.json',download)
        write(folder/'upstream.json',upstream)
        shutil_input=(SOURCE/'experiments/_inputs/families_v4').resolve()
        require(shutil_input.is_dir(),'frozen D inputs unavailable')
        local_input=REPO/'experiments/_inputs/families_v4';local_input.parent.mkdir(parents=True)
        local_input.symlink_to(shutil_input,target_is_directory=True)
        registry=REPO/'models_v2.toml'
        original=registry.read_text();sections=[]
        for mode,cap in [('off',8192),('on',28672)]:
            switch='true' if mode=='on' else 'false'
            sections.append(f'''\n[d-v2-qwen35-27b-{mode}]
model_id = "d-v2-qwen35-27b"
base_url = "http://127.0.0.1:8093/v1"
api_key = "EMPTY"
temperature = 0.3
max_tokens = {cap}
timeout = 1800.0
thinking_echo_field = "reasoning"
extra_body = {{ top_p = 1.0, top_k = -1, min_p = 0.0, repetition_penalty = 1.0, presence_penalty = 0.0, frequency_penalty = 0.0, chat_template_kwargs = {{ enable_thinking = {switch} }} }}
''')
        registry.write_text(original+''.join(sections))
        argv=['--model',str(model),'--served-model-name','d-v2-qwen35-27b','--host','127.0.0.1','--port','8093','--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len','40960','--gpu-memory-utilization','0.90','--max-num-seqs','8','--max-num-batched-tokens','4096','--enforce-eager','--language-model-only','--reasoning-parser','qwen3','--generation-config','vllm','--seed','0']
        bindings={f'qwen3.5-27b.think-{mode}':dict(revision,weights_path=str(model),model_section=f'd-v2-qwen35-27b-{mode}',vllm_arguments=argv,thinking_mode_verified=False,verification_scope='pending_live_synthetic') for mode in ('off','on')}
        write(folder/'probe_bindings.json',bindings)
        contract=json.loads((REPO/'results/d_gemma_e4b/probe_contract.json').read_bytes())
        contract.update(bindings='results/d_qwen35_27b/probe_bindings.json',records='results/d_qwen35_27b',configurations=list(bindings),gpu=3,experiments=str(DEST/'outputs'),wall_seconds=7200,committed_files=['models_v2.toml','explore/families_v2_supervise.py','results/d_qwen35_27b/probe_bindings.json','results/d_qwen35_27b/weights.json','results/d_qwen35_27b/download.json','results/d_qwen35_27b/upstream.json'])
        write(folder/'probe_contract.json',contract)
        run([PYTHON,'-m','explore.families_v2_prepare','--record-dir',str(folder)],cwd=REPO)
        def commit(message):
            run(['git','add','models_v2.toml','results/d_qwen35_27b'],cwd=REPO)
            run(['git','-c','user.name=Codex automation','-c','user.email=codex@localhost','commit','-m',message],cwd=REPO)
            return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        state.update(state='synthetic_probe',probe_commit=commit('Freeze verified Qwen3.5-27B D synthetic probe'));save()
        run([PYTHON,'-m','explore.families_v2_supervise','--contract',str(folder/'probe_contract.json'),'--run-dir',str(DEST/'probe')],cwd=REPO)
        status=json.loads((DEST/'probe/status.json').read_bytes())
        require(status['state']=='complete' and status['scope']=='synthetic_only_no_benchmark_calls' and status['gpu_memory_after_mib']==0,'probe/cleanup incomplete')
        attempt=Path(status['attempt']);probe=json.loads((attempt/'probe.json').read_bytes());audit=summarize(attempt/'probe_audit')
        require(probe['status']=='passed' and audit['usage_complete'] and audit['requests']==12,'probe request evidence incomplete')
        events=[json.loads(x.read_bytes()) for x in (attempt/'probe_audit').glob('*.json')]
        episodes={x['session']:x for x in events if x['event']=='episode_complete'}
        completions={x['call_id']:x for x in events if x['event']=='request_complete'}
        expected_cases={'short_initial','short_repair','long_reasoning_no_final_code','long_code','unicode','remaining_one','prompt_overfull','initial_must_keep_cap'}
        for key in bindings:
            rows=[x for x in probe['records'] if x['configuration']==key]
            require(len(rows)==8 and {x['case'] for x in rows}==expected_cases,'missing probe cases')
            record=next(x['record'] for x in rows if x['case']=='short_initial')
            require(record['passed'] and bool(record['attempts'][0].get('reasoning'))==key.endswith('.think-on'),'thinking/code probe differs')
            require(record_bytes(episodes[record['audit_session']]['record'])==record_bytes(record),'probe record differs from journal')
            for response in record['attempts']:
                raw=completions[response['call_id']]
                require(raw['completion']['text']==response['response'] and raw['session']==record['audit_session'] and raw['budget']==response['budget'],'probe response differs from journal')
        write(folder/'live_review.json',dict(status='passed',probe_sha256=sha256((attempt/'probe.json').read_bytes()),audit=audit,attempt=str(attempt),probe_commit=state['probe_commit'],formal_calls_before_review=0))
        write(folder/'live_probe.json',probe);write(folder/'probe_status.json',status)
        for binding in bindings.values():binding.update(thinking_mode_verified=True,verification_scope='live_synthetic_passed')
        write(folder/'ready_bindings.json',bindings)
        contract.update(probe_only=False,bindings='results/d_qwen35_27b/ready_bindings.json',wall_seconds=86400)
        contract['committed_files']+=['results/d_qwen35_27b/ready_bindings.json','results/d_qwen35_27b/live_review.json','results/d_qwen35_27b/live_probe.json','results/d_qwen35_27b/probe_status.json']
        write(folder/'ready_contract.json',contract)
        state.update(state='formal_D',formal_commit=commit('Freeze live-reviewed Qwen3.5-27B D binding'));save()
        run([PYTHON,'-m','explore.families_v2_supervise','--contract',str(folder/'ready_contract.json'),'--run-dir',str(DEST/'run')],cwd=REPO)
        state['state']='complete'
    except BaseException as exc:
        state.update(state='stopped_incomplete',error=type(exc).__name__+': '+str(exc))
        raise
    finally:save()

if __name__=='__main__':main()
