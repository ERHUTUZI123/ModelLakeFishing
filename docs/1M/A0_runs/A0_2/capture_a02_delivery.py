"""Archive the final A0.2 implementation and bind its reviewable delivery."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
DOCS = RUNS.parent
GRAPH = Path('D:/research/model_lake/data/data1m/a0_20260912/graph')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def rec(path):
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    manifest_path = OUT / 'A0_2_MANIFEST.json'
    assert not manifest_path.exists(), 'Preserve the prior delivery; create a documented revision instead'
    tests = read(OUT / 'final_focused_checks.json')
    assert tests['status'] == 'PASS' and tests['code_unchanged_during_tests']
    for name, digest in tests['code_hashes'].items():
        assert sha(REPO / name) == digest, 'Source changed after final focused tests: ' + name
    assert read(OUT / 'A0_INTERFACE_CHECKS.json')['status'] == 'PASS'
    assert read(OUT / 'A0_GRAPH_REVIEW.json')['status'] == 'PASS'
    repair = read(GRAPH / 'A0_FEATURE_REPAIR.json')
    for name, digest in repair['implementation_sha256'].items():
        assert sha(REPO / 'scale1m' / name) == digest, 'Graph-generation implementation changed after preparation'
    coverage = read(OUT / 'A0_METRIC_SOURCE_COVERAGE.json')
    assert coverage['registered_items'] == 618 and coverage['new_metric_values_computed'] == 0
    for name, digest in coverage['implementation_sha256'].items():
        assert sha(Path(name)) == digest, 'Metric-source coverage refers to stale implementation'
    entry = read(OUT / 'A0_2_ENTRY.json')
    for snapshot in entry['snapshots'].values():
        assert sha(Path(snapshot['snapshot'])) == snapshot['sha256'], 'A0.2 entry snapshot changed'
    old = read(OUT / 'frozen/A0_SOURCE_MANIFEST.json')
    preserved = ['A0.1.md', 'A0_runs/A0_PROTOCOL.json', 'A0_runs/A0_METRIC_INVENTORY.json']
    for name in preserved:
        assert sha(DOCS / name) == sha(OUT / 'frozen' / Path(name).name), 'Frozen A0.1 artifact changed: ' + name
    assert sha(Path(old['authority']['path'])) == old['authority']['sha256']
    old_plan = (OUT / 'frozen/A0.md').read_text(encoding='utf-8')
    new_plan = (DOCS / 'A0.md').read_text(encoding='utf-8')
    body = lambda s: s[s.index('## 1. 唯一目标'):s.index('## 10. 执行反馈区')]
    assert body(old_plan) == body(new_plan), 'A0 experimental plan body changed'

    baseline = read(RUNS / 'audit/A0_CODE_CAPTURE.json')['implementation_files']
    names = {r['path'] for r in baseline}
    names.update(str(p.relative_to(REPO)).replace('\\', '/') for pattern in
                 ('scale1m/*a0*.py', 'scale1m/tests/test_*a0*.py') for p in REPO.glob(pattern))
    names.update(tests['code_hashes'])
    sources = [dict(rec(REPO / name), relative_path=name) for name in sorted(names)]
    archive = OUT / 'frozen/implementation_snapshot_A02.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name in sorted(names):
            output.add(REPO / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as check:
        for source in sources:
            data = check.extractfile(source['relative_path']).read()
            assert hashlib.sha256(data).hexdigest() == source['sha256']
    changes = []
    with tarfile.open(RUNS / 'frozen/implementation_snapshot.tar.gz', 'r:gz') as original:
        original_names = set(original.getnames())
        patch = []
        for name in sorted(names):
            before = original.extractfile(name).read() if name in original_names else b''
            after = (REPO / name).read_bytes()
            if before == after:
                continue
            changes.append({'path': name, 'A01_sha256': hashlib.sha256(before).hexdigest() if name in original_names else None,
                            'A02_sha256': hashlib.sha256(after).hexdigest(),
                            'not_in_A01_archive': name not in original_names,
                            'comparison_scope': 'A01 captured implementation bytes; absence from that archive alone does not prove the file is new'})
            patch.extend(difflib.unified_diff(before.decode('utf-8-sig').splitlines(keepends=True),
                                             after.decode('utf-8-sig').splitlines(keepends=True),
                                             fromfile='A01/' + name, tofile='A02/' + name))
    (OUT / 'A01_to_A02_implementation.patch').write_text(''.join(patch), encoding='utf-8')
    (OUT / 'A0_plan_feedback.patch').write_text(''.join(difflib.unified_diff(
        old_plan.splitlines(keepends=True), new_plan.splitlines(keepends=True), fromfile='A01/A0.md',tofile='A02/A0.md')), encoding='utf-8')
    status = subprocess.run(['git','status','--short'],cwd=REPO,capture_output=True,check=True).stdout
    (OUT / 'git_status_final.txt').write_bytes(status)
    write(OUT / 'A0_CODE_CAPTURE.json', {'stage':'A0.2','sources':sources,'changed_since_A01':changes,
                                        'archive':rec(archive),'plan_body_sections_1_to_9_unchanged':True})
    for name in ('A0_FEATURE_REPAIR.json','meta.json'):
        shutil.copy2(GRAPH / name, OUT / ('final_graph_' + name))
    shutil.copy2(GRAPH.with_name('graph.verification.json'),OUT / 'final_graph_verification.json')
    graph_files = [rec(p) for p in sorted(GRAPH.iterdir()) if p.is_file()]
    current_docs = [DOCS / 'A0.md', DOCS / 'A0.2.md', RUNS / 'A0_COMMANDS.md', RUNS / 'A0_DISCREPANCIES.md']
    artifacts = [rec(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    artifacts.extend(rec(p) for p in current_docs)
    stage = {'schema':'a0.stage_manifest.v1','stage':'A0.2','status':'complete',
             'created_at_utc':datetime.now(timezone.utc).isoformat(),'authority':old['authority'],
             'previous_stage_manifest':rec(OUT / 'frozen/A0_SOURCE_MANIFEST.json'),
             'graph_digest':read(GRAPH.with_name('graph.verification.json'))['new_graph_digest'],
             'new_graph_files':graph_files,'implementation_archive':rec(archive),
             'artifacts':artifacts,'scope':'A0.2 only; actual prepared graph and fixture/CLI checks; no A0.3 smoke or formal experiments',
             'remaining_metric_evidence_gaps':'See A0_METRIC_SOURCE_COVERAGE.json; never replace missing raw events with historical counters',
             'self_hash_policy':'This manifest excludes itself and the later root manifest/delivery validation to avoid cycles'}
    write(manifest_path,stage)
    root = dict(old)
    root.update(stage='A0.2',status='complete',created_at_utc=stage['created_at_utc'],
                previous_stage_manifest=stage['previous_stage_manifest'],
                plan_with_feedback=rec(DOCS / 'A0.md'),delivery=rec(DOCS / 'A0.2.md'),
                A0_1_delivery=rec(DOCS / 'A0.1.md'),A0_2=rec(manifest_path),
                later_stages='A0.3–A0.7 pending. No real smoke, formal training, embeddings, priors, index, or retrieval result generated.')
    root['artifacts'] = [rec(Path(item['path'])) for item in old['artifacts']]
    write(RUNS / 'A0_SOURCE_MANIFEST.json',root)
    checked = 0
    for item in stage['artifacts'] + stage['new_graph_files'] + root['artifacts'] + [root['A0_2']]:
        assert sha(Path(item['path'])) == item['sha256'], 'Delivery hash differs: ' + item['path']
        checked += 1
    validation = {'stage':'A0.2','status':'PASS','checked_file_bindings':checked,
                  'plan_body_sections_1_to_9_unchanged':True,'frozen_A01_contracts_unchanged':True,
                  'all_final_test_source_hashes_match':True,'archive_verified_file_count':len(sources),
                  'A0_2_manifest':rec(manifest_path),'root_source_manifest':rec(RUNS / 'A0_SOURCE_MANIFEST.json'),
                  'finished_at_utc':datetime.now(timezone.utc).isoformat()}
    write(OUT / 'A0_DELIVERY_VALIDATION.json',validation)
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
