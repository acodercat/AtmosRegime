"""Move the never-started 27B formal evaluation to an isolated 114 launch, unchanged scientific recipe."""
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
ROOT=Path('/path/to/workdir/experiments')
OLD=ROOT/'d_qwen35_27b_v2'
DEST=ROOT/'d_qwen35_27b_114_20261011';REPO=DEST/'repo'
PYTHON='/path/to/workdir/eval-venv/bin/python'
assert socket.gethostname()=='host-D'
def run(argv,**kw):return subprocess.run(argv,check=True,**kw)
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
old=json.loads((OLD/'run/status.json').read_bytes())
assert old['state']=='stopped_incomplete' and old['error']=='OSError: [Errno 98] Address already in use'
assert not list((OLD/'outputs/families_context_v2').glob('*.json'))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=OLD/'repo',text=True).strip()=='0189de99599421e104d36054e87d6ffe9338557b'
assert int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0
with socket.socket() as s:s.bind(('127.0.0.1',8094))
assert not DEST.exists();DEST.mkdir()
run(['git','clone','--no-hardlinks','--no-checkout',str(OLD/'repo'),str(REPO)])
run(['git','checkout','--detach','0189de99599421e104d36054e87d6ffe9338557b'],cwd=REPO)
inputs=(OLD/'repo/experiments/_inputs/families_v4').resolve()
(REPO/'experiments/_inputs').mkdir(parents=True)
(REPO/'experiments/_inputs/families_v4').symlink_to(inputs,target_is_directory=True)
registry=REPO/'models_v2.toml';raw=registry.read_text();assert raw.count('127.0.0.1:8093/v1')==2
registry.write_text(raw.replace('127.0.0.1:8093/v1','127.0.0.1:8094/v1'))
folder=REPO/'results/d_qwen35_27b_114';folder.mkdir()
contract=json.loads((REPO/'results/d_qwen35_27b/ready_contract.json').read_bytes())
bindings=json.loads((REPO/contract['bindings']).read_bytes())
for binding in bindings.values():
 args=binding['vllm_arguments'];assert args[args.index('--port')+1]=='8093';args[args.index('--port')+1]='8094'
write(folder/'bindings.json',bindings)
contract.update(bindings='results/d_qwen35_27b_114/bindings.json',records='results/d_qwen35_27b_114',gpu=0,experiments=str(DEST/'outputs'))
contract['committed_files']+=['results/d_qwen35_27b_114/bindings.json','results/d_qwen35_27b_114/restart.json']
write(folder/'restart.json',{'reason':'previous formal launch failed before any benchmark request; isolated host/port relocation',
 'old_status':old,'previous_source':'0189de99599421e104d36054e87d6ffe9338557b',
 'changes':{'host':'host-B -> host-D','gpu':'3 -> 0','port':'8093 -> 8094','result_directory':str(DEST/'outputs')},
 'unchanged':['weights','question bytes','sampling','context/output caps','five content attempts','grading','24-hour per-launch watchdog'],
 'fresh_live_probe_before_formal':True})
write(folder/'contract.json',contract)
run([PYTHON,'-m','explore.families_v2_prepare','--record-dir',str(folder)],cwd=REPO)
run(['git','add','models_v2.toml','results/d_qwen35_27b_114'],cwd=REPO)
run(['git','-c','user.name=Codex automation','-c','user.email=codex@localhost','commit','-m','Freeze 27B host and port relocation before first formal request'],cwd=REPO)
run([PYTHON,'-m','explore.families_v2_supervise','--contract',str(folder/'contract.json'),'--check'],cwd=REPO)
with (DEST/'supervisor.log').open('x') as log:
 proc=subprocess.Popen([PYTHON,'-u','-m','explore.families_v2_supervise','--contract',str(folder/'contract.json'),'--run-dir',str(DEST/'run')],cwd=REPO,env=dict(os.environ,PYTHONPATH=str(REPO)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(DEST/'launch.json',{'pid':proc.pid,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'host':'host-D','gpu':0,'port':8094,'started_unix':time.time()})
print('STARTED',proc.pid,flush=True)
