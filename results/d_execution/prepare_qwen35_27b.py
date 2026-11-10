"""Download only the registered Qwen3.5-27B at a fixed official revision."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from huggingface_hub import hf_hub_download

root=Path('/path/to/workdir/experiments/d_execution')
lock=(root/'qwen35_27b_download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (root/'qwen35_27b_download.json').exists(), 'Download already attempted; inspect before retry.'
info=json.loads((root/'qwen35_27b_upstream.json').read_bytes())
revision='fc05daec18b0a78c049392ed2e771dde82bdf654'
assert info['sha']==revision and info['id']=='Qwen/Qwen3.5-27B'
destination=Path('/path/to/shared/Qwen3.5-27B')
files=[x for x in info['siblings'] if x['rfilename'].endswith(('.json','.jinja','.safetensors','.txt'))]
state={'model':info['id'],'revision':revision,'destination':str(destination),'status':'preflight','verified':{},'started_unix':time.time()}
def save():
    p=root/'qwen35_27b_download.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'qwen35_27b_download.json')
save()
try:
    quota=subprocess.check_output(['lfs','quota','-u','USER','/path/to/shared'],text=True,timeout=15)
    line=next(x.split() for x in quota.splitlines() if x.strip().startswith('/path/to/shared'))
    available=(int(line[3])-int(line[1].rstrip('*')))*1024
    total=sum(x['size'] for x in files)
    checkpoint=Path('/path/to/workdir/checkpoints/k_v14_58717b1')
    allocated=int(subprocess.check_output(['du','-s','-B1',str(checkpoint)],text=True).split()[0]) if checkpoint.exists() else 0
    reserve=max(28*2**30,120*2**30-allocated)
    assert available>=total+reserve, 'Insufficient quota after K checkpoint/merge reservation'
    state.update(status='downloading',quota_available_bytes=available,planned_model_bytes=total,reserved_bytes=reserve,existing_K_allocated_bytes=allocated);save()
    for item in files:
        name=item['rfilename']; assert '/' not in name
        path=Path(hf_hub_download(repo_id=info['id'],filename=name,revision=revision,local_dir=str(destination)))
        sha=hashlib.sha256();git=hashlib.sha1();git.update(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as handle:
            while chunk:=handle.read(8*1024*1024):sha.update(chunk);git.update(chunk)
        assert path.stat().st_size==item['size'],name
        if item.get('lfs'):assert sha.hexdigest()==item['lfs']['sha256'],name
        else:assert git.hexdigest()==item['blobId'],name
        state['verified'][name]=sha.hexdigest();save()
    state['status']='weights_verified_pending_service_probe'
except BaseException as exc:
    state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
finally:
    state['updated_unix']=time.time();save()
