"""Record A0.2 focused fixture verification without running the A0.3 suite."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TEST_NAMES = [
    'test_prepare_a0_graph.py', 'test_graph_store.py', 'test_checkpoint.py',
    'test_a0_smoke.py', 'test_export_rf.py', 'test_y2.py', 'test_y4.py', 'test_run_metadata.py',
    'test_a0_evaluation.py', 'test_recompute_a0.py', 'test_a0_producer_bindings.py',
    'test_a0_source_counts.py', 'test_a0_recompute_checks.py',
]


def code_hashes():
    paths = sorted(set(REPO.glob('scale1m/*.py')) | set(REPO.glob('scale1m/tests/*.py'))
                   | set(REPO.glob('stage2TrainGraphSAGE/*.py')) | set(REPO.glob('stage3HNSW/*.py')))
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    label = parser.parse_args().label
    assert label.isalnum(), 'Use a simple filename label'
    record = OUT / f'{label}_focused_checks.json'
    xml = OUT / f'{label}_focused_tests.xml'
    log = OUT / f'{label}_focused_tests.log'
    assert not any(p.exists() for p in (record, xml, log)), 'Preserve previous test evidence'
    tests = [str(Path('scale1m/tests') / name) for name in TEST_NAMES]
    assert all((REPO / p).is_file() for p in tests), 'Expected focused fixture file missing'
    before = code_hashes()
    start = time.perf_counter()
    argv = [sys.executable, '-B', '-m', 'pytest', *tests, '-q', '--junitxml=' + str(xml)]
    proc = subprocess.run(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_bytes(proc.stdout)
    after = code_hashes()
    suites = ET.parse(xml).getroot().findall('testsuite')
    counts = {key: sum(int(s.get(key, '0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    success = proc.returncode == 0 and before == after and not any(counts[k] for k in ('failures','errors','skipped'))
    result = {'stage': 'A0.2', 'status': 'PASS' if success else 'FAIL',
              'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'argv': argv,
              'cwd': str(REPO), 'returncode': proc.returncode, 'seconds': time.perf_counter() - start,
              'counts': counts, 'code_unchanged_during_tests': before == after, 'code_hashes': after,
              'log': str(log), 'junitxml': str(xml),
              'scope': 'Focused small fixtures only; not A0.3 full suites or real one-epoch smoke'}
    record.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(proc.stdout.decode('utf-8', errors='replace'))
    print(json.dumps({k:result[k] for k in ('status','counts','seconds','code_unchanged_during_tests')}))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
