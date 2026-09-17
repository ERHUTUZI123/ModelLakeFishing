"""Local RTX 4060/i7 delivery checks, outside the measured A0.6 pipeline."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import gc, hashlib, json, os, subprocess, sys, time, traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
HOST = 'x98liu@watgpu.cs.uwaterloo.ca'
REMOTE = '/u801/x98liu/model_lake/data1m/a0_20260912'
LOCAL = Path('D:/research/model_lake/data/data1m/a0_20260912/exports')
OUT = BASE / 'local_verification'
OUT.mkdir(exist_ok=True)
state = {'status': 'running', 'pid': os.getpid(), 'seeds': {},
         'scope': 'Supplemental delivery verification; official metrics and timings remain from the remote evaluator.',
         'score_atol': 1e-5,
         'tolerance_scope': 'Cross-device dot-product check only; saved scores, ties, ranks and metric gates are unchanged.'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def update(phase, **kwargs):
    state.update(phase=phase, updated_at_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    tmp = OUT / 'STATUS.json.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    tmp.replace(OUT / 'STATUS.json')
    print(json.dumps({'phase': phase, **kwargs}), flush=True)


def fetch(remote, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and expected:
        assert target.stat().st_size == expected['bytes'] and sha(target) == expected['sha256'], target
        return
    part = target.with_name(target.name + '.part')
    update('downloading', file=str(target), expected_bytes=expected.get('bytes') if expected else None)
    started = time.monotonic()
    for attempt in range(3):
        with (OUT / 'transfer.log').open('ab') as log:
            process = subprocess.Popen(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                        HOST + ':' + remote, str(part)], stdout=log, stderr=log)
            while process.poll() is None:
                time.sleep(5)
                if part.exists():
                    size = part.stat().st_size
                    update('downloading', file=str(target), downloaded_bytes=size,
                           transfer_mib_per_second=size / 2**20 / max(time.monotonic()-started, 1))
        if process.returncode == 0:
            break
        if attempt == 2:
            raise RuntimeError(f'scp failed: {remote}; see transfer.log')
        time.sleep(5)
    if expected:
        assert part.stat().st_size == expected['bytes'] and sha(part) == expected['sha256'], target
    part.replace(target)


def verify_vectors(path, expected_shape):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    assert array.shape == expected_shape and array.dtype == np.float32
    error = 0.0
    for start in range(0, len(array), 50000):
        block = torch.from_numpy(array[start:start+50000].copy()).to('cuda')
        assert torch.isfinite(block).all().item(), path
        error = max(error, (torch.linalg.vector_norm(block, dim=1)-1).abs().max().item())
    assert error < 1e-5, (path, error)
    return {'shape': list(array.shape), 'finite': True, 'max_norm_error': error}


def verify_pool(path, model_path, dataset_path, normalize, tie):
    models = np.load(model_path, mmap_mode='r', allow_pickle=False)
    datasets = np.load(dataset_path, mmap_mode='r', allow_pickle=False)
    maximum = 0.0
    with np.load(path, allow_pickle=False) as raw:
        queries, ids, scores = raw['query'], raw['model'], raw['score']
        priors, fused, top10 = raw['prior'], raw['fused'], raw['top10']
        assert ids.shape == scores.shape == (len(queries), 1000)
        assert ids.min() >= 0 and ids.max() < len(models)
        assert queries.min() >= 0 and queries.max() < len(datasets)
        assert np.isfinite(priors).all() and np.isfinite(scores).all() and np.isfinite(fused).all()
        assert np.array_equal(fused, (scores+np.float32(1))*np.float32(.5)+priors)
        for i, query in enumerate(queries):
            assert len(np.unique(ids[i])) == 1000
            assert np.array_equal(top10[i], ids[i][np.lexsort((tie[ids[i]], -fused[i]))[:10]])
        for start in range(0, len(queries), 16):
            a = torch.from_numpy(np.asarray(models[ids[start:start+16]])).to('cuda')
            b = torch.from_numpy(np.asarray(datasets[queries[start:start+16]])).to('cuda')
            if normalize:
                a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            actual = (a * b[:, None, :]).sum(-1).cpu().numpy()
            error = float(np.max(np.abs(actual-scores[start:start+16])))
            maximum = max(maximum, error)
            assert error <= state['score_atol'], (path, start, error)
        if 'total_ns' in raw:
            assert (raw['total_ns'] > 0).all()
            assert np.array_equal(raw['total_ns'], raw['hnsw_ns']+raw['rerank_ns'])
    return {'status': 'PASS', 'queries': len(queries), 'pairs_checked': int(ids.size),
            'gpu_max_score_abs_error': maximum, 'fused_exact_match': True,
            'saved_top10_exact_match': True, 'unique_candidates_per_query': 1000}


def main():
    global np, torch
    import numpy as np
    import torch
    import psutil
    from scale1m.eval_y2 import _tie_ranks
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    update('initializing', gpu=torch.cuda.get_device_name(), cpu_threads=8,
           torch_version=torch.__version__, available_ram_bytes=psutil.virtual_memory().available,
           code_sha256=sha(Path(__file__)))
    evidence = read(BASE.parent / 'A0_5/STATUS.json')
    assert evidence['state'] == 'COMPLETED'
    snapshot = OUT / 'EXACT_MANIFEST.remote.json'
    fetch(REMOTE+'/metrics/A0_EVALUATION_MANIFEST.json', snapshot)
    manifest = read(snapshot)
    assert all(manifest['seeds'][str(s)].get('exact') for s in range(3))
    tie = _tie_ranks(3016439)
    for seed in range(3):
        run = f'A0GD_full_s{seed}_e25'
        files = evidence['seeds'][seed]['validation']['files']
        target = LOCAL / run
        for name in ['z_d_eval.npy', 'z_m_eval.npy']:
            fetch(REMOTE+'/exports/'+run+'/'+name, target/name, files[name])
        name = manifest['seeds'][str(seed)]['exact']
        fetch(REMOTE+'/metrics/'+name, OUT/name, manifest['artifacts'][name])
        update('gpu_exact_verification', current_seed=seed)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {n: pool.submit(sha, target/n) for n in ['z_d_eval.npy', 'z_m_eval.npy']}
            vectors = {n: verify_vectors(target/n, shape) for n, shape in
                       [('z_m_eval.npy', (3016439, 128)), ('z_d_eval.npy', (18729, 128))]}
            for n, future in futures.items():
                assert future.result() == files[n]['sha256']
        check = verify_pool(OUT/name, target/'z_m_eval.npy', target/'z_d_eval.npy', True, tie)
        state['seeds'][str(seed)] = {'vectors': vectors, 'exact': check}
        update('exact_verified', current_seed=seed, verification=check)
        gc.collect()
    deadline = time.monotonic()+4*3600
    while not (BASE/'delivery/metrics/A0_EVALUATION_MANIFEST.json').exists():
        remote_state = read(BASE/'STATUS.json')
        if remote_state.get('stage_status') == 'blocked':
            raise RuntimeError('Remote evaluation failed; completed local exact checks are retained.')
        if time.monotonic() > deadline:
            raise TimeoutError('HNSW delivery still pending after four hours')
        update('waiting_for_hnsw_delivery')
        time.sleep(45)
    delivery = BASE/'delivery/metrics'
    final = read(delivery/'A0_EVALUATION_MANIFEST.json')
    assert final['binding_sha256'] == manifest['binding_sha256']
    for seed in range(3):
        name = final['seeds'][str(seed)]['hnsw']
        assert sha(delivery/name) == final['artifacts'][name]['sha256']
        target = LOCAL/f'A0GD_full_s{seed}_e25'
        update('gpu_hnsw_verification', current_seed=seed)
        state['seeds'][str(seed)]['hnsw'] = verify_pool(
            delivery/name, target/'z_m_eval.npy', target/'z_d_eval.npy', False, tie)
    update('complete', status='PASS', gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
           cpu_peak_working_set_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        update('failed', status='FAIL', error=traceback.format_exc())
        raise
