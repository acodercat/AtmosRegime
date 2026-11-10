"""Offline binding regressions. Tiny fake weight files and a local metadata-only HTTP server."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
from threading import Thread
from unittest.mock import patch

from .frozen import json_bytes
from .serving_binding import weight_revision, validate_binding, runtime_evidence, process_identity


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('invalid serving binding accepted')


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/'config.json').write_text('{}')
        (root/'model.safetensors').write_bytes(b'synthetic tensors, never loaded')
        card={'id':'test-local','root':str(root)}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                assert self.path=='/v1/models'
                self.send_response(200);self.end_headers()
                self.wfile.write(json_bytes({'data':[card]}))
            def log_message(self,*args):
                pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            port=server.server_address[1]
            config={'model_id':'test-local','base_url':f'http://127.0.0.1:{port}/v1',
                    'temperature':.3,'api_key':'EMPTY',
                    'extra_body':{'chat_template_kwargs':{'enable_thinking':True}}}
            binding={'model_section':'test','weights_path':str(root),**weight_revision(root),
                     'thinking_mode_verified':True,'vllm_arguments':['--model',str(root),
                         '--served-model-name','test-local','--port',str(port)]}
            key='qwen3-8b.think-on'
            validate_binding(key,binding,config)
            bad=copy.deepcopy(config);bad['extra_body']['chat_template_kwargs']['enable_thinking']=False
            rejects(lambda:validate_binding(key,binding,bad))
            bad=copy.deepcopy(binding);bad['weights_path']=str(root/'missing')
            rejects(lambda:validate_binding(key,bad,config))
            bad=copy.deepcopy(binding);bad['vllm_arguments']=['--model','wrong']
            rejects(lambda:validate_binding(key,bad,config))
            bad=copy.deepcopy(binding);bad['weights_revision']='manual-unverifiable-label'
            rejects(lambda:validate_binding(key,bad,config))
            off=copy.deepcopy(config);off.pop('extra_body')
            validate_binding('ministral3-8b.think-off',binding,off)
            rejects(lambda:validate_binding(key,binding,off))
            pid=os.getpid()
            real_read=Path.read_bytes
            cmdline=Path('/proc')/str(pid)/'cmdline'
            argv=['python','-m','vllm.entrypoints.openai.api_server',*binding['vllm_arguments']]
            def fake_read(path):
                if path==cmdline:
                    return ('\0'.join(argv)+'\0').encode()
                return real_read(path)
            # Only the command-line bytes are faked. PID/start identity, socket ownership,
            # HTTP metadata and content digests are checked against real local resources.
            with patch.object(Path,'read_bytes',fake_read):
                evidence=runtime_evidence(pid,binding,config)
                assert evidence==runtime_evidence(pid,binding,config,hash_weights=False)
                wrong=copy.deepcopy(binding);wrong['vllm_arguments']+=['--dtype','float16']
                rejects(lambda:process_identity(pid,wrong))
                card['root']=str(root/'different')
                rejects(lambda:runtime_evidence(pid,binding,config))
                card['root']=str(root)
                (root/'model.safetensors').write_bytes(b'altered model contents')
                rejects(lambda:runtime_evidence(pid,binding,config))
        finally:
            server.shutdown();server.server_close();thread.join()
    print('PASS: mode mismatch, absent weights, argv/revision mismatch, live PID/socket/API binding and weight tampering')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
