"""Derive only orchestration from the proven smoke monitor; trainer stays intact."""
from pathlib import Path
OUT=Path(__file__).resolve().parent
s=(OUT.parent/'A0_3_watgpu/monitor_smoke.py').read_text(encoding='utf-8')
s=s.replace('import json,os,resource,subprocess,sys,time','import json,os,resource,subprocess,sys,time,hashlib,shutil')
s=s.replace("ROOT=Path(__file__).resolve().parents[1]", "BASE=Path(__file__).resolve().parent\nROOT=BASE/'ModelLakeFishing'\nSEED=int(sys.argv[1]);assert SEED in (0,1,2)")
s=s.replace("RUN=Path('/u801/x98liu/model_lake/runs/A0_20260912/smoke_s0_e1_retry1')", "RUN=Path(f'/u801/x98liu/model_lake/runs/A0_20260912/A0GD_full_s{SEED}_e25')")
s=s.replace("OUT=ROOT.parent/'observations'", "OUT=BASE/f'observations_s{SEED}'")
s=s.replace("'Fresh smoke output required'", "'Fresh formal output required'")
s=s.replace("'--seed','0','--epochs','1'", "'--seed',str(SEED),'--epochs','25'")
s=s.replace(",'--smoke-only'",'')
s=s.replace("gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)","try:\n            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)\n        except (OSError,subprocess.TimeoutExpired):\n            gpu=subprocess.CompletedProcess([],1,'','GPU sample unavailable')")
s=s.replace('exact PyTorch allocation peak in SMOKE_REPORT.json','exact PyTorch allocation peak in A0_RUN_RECORDS.json')
s=s.replace("print(json.dumps(record),flush=True)\nraise SystemExit(p.returncode)","""if (RUN/'stdout').is_dir():shutil.copy2(OUT/'train.log',RUN/'stdout/train.log')
print(json.dumps(record),flush=True)
if p.returncode==0:
    import torch
    report=json.loads((RUN/'MANIFEST.json').read_text())
    native=json.loads((RUN/'A0_RUN_RECORDS.json').read_text())
    expected=json.loads((ROOT/'docs/1M/A0_runs/A0_PROTOCOL.json').read_text())['required_config_audit']['resolved_config']
    actual=json.loads((RUN/'metadata/resolved_config.json').read_text())['resolved_config']
    assert actual==expected and len(actual)==40
    assert report['epochs']==25 and report['start_epoch']==0 and report['seed']==SEED and report['resumed_from'] is None
    assert report['mechanism_gate']['passed'] and native['status']=='complete' and native['timing_complete']
    assert len(native['history'])==25
    assert report['graph_sha256']=='acddeb93d926efcb636c83d452fe360736a0e6e4ac68cab3f5fef210e7cef5db'
    ckpt=RUN/'ckpt/last.pt'
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert ck['epoch']==24 and len(ck['history'])==25
    assert not ck.get('extra',{}).get('smoke_only')
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    assert sha(ckpt)==native['a0']['checkpoint_sha256']
    proof={'status':'PASS','seed':SEED,'epochs':25,'checkpoint_epoch':24,'checkpoint_sha256':sha(ckpt),'config_keys_equal':40,'graph_sha256':report['graph_sha256'],'loss_first':native['history'][0]['total'],'loss_last':native['history'][-1]['total'],'mechanism_gate':report['mechanism_gate'],'resource_report':record,'run':str(RUN)}
    (OUT/'A04_VALIDATION.json').write_text(json.dumps(proof,indent=2)+'\\n')
raise SystemExit(p.returncode)""")
compile(s,'monitor_train.py','exec')
(OUT/'monitor_train.py').write_text(s,encoding='utf-8',newline='\n')
print('Formal monitor prepared; only seed/epochs/mode/output and observation checks differ.')
