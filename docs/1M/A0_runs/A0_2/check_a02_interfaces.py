"""Read-only configuration/CLI checks; never invoke training or evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def main():
    from scale1m.train_rung import build_config
    import torch
    audit = json.loads((OUT.parent / 'audit/A0_CONFIG_AUDIT.json').read_text(encoding='utf-8'))
    current = build_config(argparse.Namespace(**audit['build_config_args']))
    assert current == audit['resolved_config'], 'Effective GD configuration drifted'
    modules = ['scale1m.prepare_a0_graph', 'scale1m.train_rung', 'scale1m.export_rf',
               'stage3HNSW.build_prior_sidecar', 'scale1m.eval_y2', 'scale1m.recompute_a0',
               'scale1m.a0_source_counts']
    commands = []
    for module in modules:
        args = [sys.executable, '-B', '-m', module, '--help']
        start = time.perf_counter()
        proc = subprocess.run(args, cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
        log = OUT / (module.replace('.', '_') + '.help.txt')
        log.write_text(proc.stdout + proc.stderr, encoding='utf-8')
        commands.append({'argv': args, 'cwd': str(REPO), 'returncode': proc.returncode,
                         'seconds': time.perf_counter() - start, 'log': str(log),
                         'sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
        assert proc.returncode == 0, f'{module} CLI import failed; inspect {log}'
    record = {'stage': 'A0.2', 'status': 'PASS', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'config_key_count': len(current), 'config_equal_A01': True,
              'resolved_config': current, 'commands': commands,
              'runtime': {'python': sys.version, 'executable': sys.executable,
                          'platform': platform.platform(), 'processor': platform.processor(),
                          'logical_cpus': os.cpu_count(), 'torch': torch.__version__,
                          'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
                          'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              'no_training_or_evaluation_invoked': True}
    (OUT / 'A0_INTERFACE_CHECKS.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'config_keys_unchanged':len(current),'CLI_help_passed':len(commands)}))


if __name__ == '__main__':
    main()
