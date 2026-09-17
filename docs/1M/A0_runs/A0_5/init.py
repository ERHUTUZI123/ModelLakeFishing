from pathlib import Path
import json,hashlib,shutil
OUT=Path(__file__).resolve().parent;DOC=OUT.parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=json.loads((DOC/'A0_runs/A0_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
assert manifest['stage']=='A0.4' and manifest['status']=='complete'
assert sha(Path(manifest['A0_4']['path']))==manifest['A0_4']['sha256']
status=json.loads((DOC/'A0_runs/A0_4/STATUS.json').read_text(encoding='utf-8'))
for row in status['seeds']:
    assert row['local_hash_validation']=='PASS' and row['validation']['status']=='PASS'
    assert sha(Path(row['local_delivery'])/'ckpt/last.pt')==row['validation']['checkpoint_sha256']
frozen=OUT/'frozen';frozen.mkdir()
for rel in ['A0.md','A0.4.md','A0_runs/A0_SOURCE_MANIFEST.json','A0_runs/A0_PROTOCOL.json']:
    shutil.copy2(DOC/rel,frozen/Path(rel).name)
write={'status':'PASS','job_id':1540848,'previous_manifest_sha256':sha(DOC/'A0_runs/A0_SOURCE_MANIFEST.json'),'checkpoint_sha256':{str(r['seed']):r['validation']['checkpoint_sha256'] for r in status['seeds']},'ops_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
(OUT/'LOCAL_PREFLIGHT.json').write_text(json.dumps(write,indent=2)+'\n',encoding='utf-8')
(DOC/'A0.5.md').write_text('''# A0.5 执行回填

状态：**进行中**。watgpu 作业 **1540848** 已提交，按 seed 0→1→2 执行。

| 项目 | 当前反馈 |
|---|---|
| 前置条件 | 三个 25 epoch 末 checkpoint、本地哈希和远端源代码均通过核验 |
| 数据输入 | 新图、ladder 使用冻结输入；缺失的 task-nodes 已从本地上传并核验 SHA |
| 执行配置 | 每 seed：embed / last / chunk 50000；随后 split-specific prior，显式 task-nodes |
| 输出 | `/u801/x98liu/model_lake/data1m/a0_20260912/exports/A0GD_full_s{0,1,2}_e25` |
| 验收 | 128 维、有限值、L2、行序、分块一致性、gold 身份、prior 边及公式；逐 seed 校验 |
| 证据区 | [A0_5](A0_runs/A0_5/)；实际脚本、输入核验和日志保留 |
| 当前下一步 | 等待三个 seed 完成，回收验收记录并回填实测结果 |

反馈：仅使用 A0.4 的新 checkpoint 生成本轮向量和 prior。最终检索流程保持 X4G+D → HNSW top1000 → task prior → top10。
''',encoding='utf-8')
print('A0.4 local delivery verified; A0.5 progress recorded.')
