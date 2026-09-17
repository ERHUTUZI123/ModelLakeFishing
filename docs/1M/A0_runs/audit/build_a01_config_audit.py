"""A0.1: invoke pure build_config only; no main/train/model/graph execution."""
import sys
sys.dont_write_bytecode = True
import argparse
import ast
import hashlib
import json
import inspect
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from scale1m.train_rung import build_config

args = argparse.Namespace(fanout=True, sparse_M=True, contrast_n_neg=256,
    contrast_max_pos_per_dataset=None, chunked_infer=50000, skip_diagnostics=True,
    batch_size=None, lake_gamma=0.5, global_n_datasets=128, num_layers=None)
cfg = build_config(args)
SOURCE = REPO / 'docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

doc_expected = {
    'num_layers': (1, '4.7'), 'top_frac': (0.1, '4.5'), 'lr': (0.01, '4.7'),
    'lambda_rank': (1.0, '4.7'), 'lambda_contrast': (1.0, '4.7'),
    'lambda_mse': (0.0, '4.7'), 'lambda_uniform': (0.0, '4.7'),
    'scorer': ('dot', '1; 4.4'), 'similar_to_mode': ('topk_unweighted', '3.6'),
    'similar_to_k': (10, '3.6'), 'edge_aware': (True, '4.3'),
    'weighted_relations': ([], '4.3'), 'rank_loss': ('ranknet', '4.4'),
    'rank_temperature': (0.1, '4.7'), 'rank_min_gap': (0.02, '4.7'),
    'rank_gap_weighted': (False, '4.4 formula'), 'separate_heads': (False, '4.3; 4.7'),
    'lambda_dm_contrast': (0.0, '4.7'), 'early_stop': (False, '4.7'),
    'lambda_global': (1.0, '4.7'), 'global_mode': ('lake', '4.6'),
    'lake_alpha': (0.75, '4.7'), 'lake_n0': (1.0, '4.7'),
    'global_n_neg': (256, '4.7'), 'batch_size': (1024, '4.7'),
    'fanout': (True, '6 GD command'), 'sparse_M': (True, '6 GD command'),
    'contrast_n_neg': (256, '4.7; 6'), 'infer_chunk': (50000, '4.7; 6'),
    'skip_diagnostics': (True, '6 GD command'), 'lake_gamma': (0.5, '4.7; 6'),
    'global_n_datasets': (128, '4.7; 6'),
}
inactive = {
    'dm_temperature': 'inactive because lambda_dm_contrast=0',
    'dm_hard_neg_weight': 'inactive because lambda_dm_contrast=0',
    'dm_warmup': 'inactive because lambda_dm_contrast=0',
    'patience': 'inactive because early_stop=False',
    'global_known_low': 'pool-mode setting inactive because global_mode=lake',
    'global_hard_frac': 'pool-mode setting inactive because global_mode=lake; not active hard-negative mining',
}
rows = []
for name, value in sorted(cfg.items()):
    expected = doc_expected.get(name)
    rows.append({'key': name, 'resolved_value': value,
        'classification': 'authoritative_document_mapped' if expected else 'implementation_bound_not_published',
        'source_section': expected[1] if expected else None,
        'document_expected_value': expected[0] if expected else None,
        'matches_document': value == expected[0] if expected else None,
        'implementation_behavior': inactive.get(name, 'preserve original resolved value'),
        'status': 'verified_document_match' if expected and value == expected[0] else 'bound_current_implementation' if not expected else 'discrepancy'})

linked = []
for seed in (0,1,2):
    p = REPO / f'docs/1M/X4_runs/X4GD_full_s{seed}_e25/metadata/resolved_config.json'
    obj = json.loads(p.read_text(encoding='utf-8'))
    old = obj['resolved_config']
    diffs = [{'key': k, 'current': cfg.get(k), 'linked': old.get(k)} for k in sorted(set(cfg) | set(old))
             if cfg.get(k) != old.get(k) or (k in cfg) != (k in old)]
    linked.append({'seed': seed, 'path': str(p), 'sha256': sha(p), 'resolved_key_count': len(old),
        'role': 'English evidence 4.7 explicit link: implementation identity cross-check only; not an alternate historical fact/metric source',
        'exact_config_equal': cfg == old, 'differences': diffs,
        'command_args_excluding_paths': {k:v for k,v in obj.get('args',{}).items() if k not in ('graph','out','family_vocab')},
        'new_num_layers_cli_default': None,
        'note': 'Current parser additionally exposes num_layers=None; absent old CLI key has no effect on unchanged resolved num_layers=1.'})

def function_defaults(rel, fn):
    tree = ast.parse((REPO / rel).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn)
    out = {}
    pos = node.args.posonlyargs + node.args.args
    for arg, default in zip(pos[-len(node.args.defaults):], node.args.defaults):
        try: out[arg.arg] = ast.literal_eval(default)
        except (ValueError, TypeError): out[arg.arg] = ast.unparse(default)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if default is not None:
            try: out[arg.arg] = ast.literal_eval(default)
            except (ValueError, TypeError): out[arg.arg] = ast.unparse(default)
    return out

train_defaults = function_defaults('stage2TrainGraphSAGE/train.py','train')
contrast_defaults = function_defaults('stage2TrainGraphSAGE/losses.py','contrastive_loss_sampled')
global_defaults = function_defaults('stage2TrainGraphSAGE/losses.py','global_lake_loss')
import torch
adam_defaults = {k: v.default for k,v in inspect.signature(torch.optim.Adam).parameters.items()
                 if v.default is not inspect.Parameter.empty}

paths = ['scale1m/train_rung.py','scale/export_ours.py','stage2TrainGraphSAGE/ablation.py',
         'stage2TrainGraphSAGE/train.py','stage2TrainGraphSAGE/losses.py','stage2TrainGraphSAGE/top1_audit.py',
         'scale1m/eval_y2.py','scale1m/eval_y4.py','scale1m/baselines.py','scale/global_metrics.py','scale1m/export_rf.py']
payload = {
    'stage': 'A0.1', 'status': 'configuration_and_static_behavior_audited',
    'authority': {'path':str(SOURCE), 'sha256':sha(SOURCE)},
    'safe_execution': {'invoked':'scale1m.train_rung.build_config(args) only', 'sys_dont_write_bytecode': True,
        'training_called': False, 'model_instantiated': False, 'optimizer_instantiated': False, 'graph_loaded': False,
        'historical_metric_files_read': False},
    'resolved_key_count': len(cfg), 'expected_key_count_assumption': {'assumed':45,'observed':len(cfg),'resolution':'enumerate actual 40 keys; do not manufacture five keys'},
    'document_mapped_key_count': len(doc_expected), 'implementation_bound_key_count': len(cfg)-len(doc_expected),
    'build_config_args': vars(args), 'resolved_config': cfg, 'per_key_audit': rows,
    'all_document_mapped_values_match': all(r['matches_document'] for r in rows if r['matches_document'] is not None),
    'doc_linked_config_identity_checks': linked,
    'runtime_static_bindings': {
        'train_signature_defaults': train_defaults,
        'contrastive_loss_sampled_signature_defaults': contrast_defaults,
        'global_lake_loss_signature_defaults': global_defaults,
        'optimizer': {'constructor':'torch.optim.Adam(model+scorer parameters, lr=lr)', 'source':'stage2TrainGraphSAGE/train.py:97',
            'current_torch_version':torch.__version__, 'current_library_signature_defaults':adam_defaults,
            'classification':'Adam/lr authoritative 4.7; other defaults bound to current installed library, not historical claims; recheck actual execution host'},
        'initialization_seed': {'value':0,'source':'stage2TrainGraphSAGE/top1_audit.py INIT_SEED','classification':'authoritative 4.7'},
        'edge_dropout':{'ordinary':0.3,'lineage':0.05,'source':'train signature; ablation does not override','classification':'authoritative 4.7 and static implementation agreement'},
        'tie_break': {'source':'scale1m/baselines.py:20; scale1m/eval_y2.py:83', 'algorithm':'uint64(mappedID * 11400714819323198485 + 0xD1B54A32D192ED03), modulo 2^64; argsort-stable gives unique tie ranks; smaller wins',
            'rng_seed':None, 'classification':'implementation-bound fixed bijection; there is no RNG seed to guess'},
        'calibration_queries': {'source':'scale1m/eval_y2.py:710', 'selection':'exact_pool.query IDs from same eligible held-out set', 'record_ids_and_hash_required':True},
        'timing':{'sources':['scale1m/eval_y2.py:759','scale1m/eval_y4.py:453'], 'warmup_count':'min(50, n_queries)', 'timed_repetitions_per_query':1,
            'hnsw_query_threads':1,'quantiles':'numpy.percentile(...,50/95) default method; local current default linear',
            'timer':'time.perf_counter_ns; differences / 1e6 milliseconds','classification':'static original implementation binding, not unpublished historical measurement'},
        'hnsw_random_seed':{'value':None,'source':'scale1m/eval_y2.py init_index omits random_seed','classification':'library default; bind actual hnswlib version/default before A0.3; no index instantiated in audit'},
        'chunked_inference_check':{'source':'scale1m/export_rf.py:241-252','max_abs_delta_strictly_less_than':1e-5,'scope':'original verification subgraph of min(verify_nodes,N) models with full dataset side','classification':'implementation-bound numerical tolerance referenced but not numerically specified by evidence 4.8'},
        'active_epoch_loss_fields':['rank','contrast','global','total'],
        'epoch_loss_aggregation':'arithmetic mean of per-batch parts values (train.py:247)',
        'native_scorer_fields':['n_queries','gold@1','gold-gap@1','gold@10','gold-gap@10','top3@10','median_gold_rank','n_roots','root_gold@1','root_gold@10','root_top3@10','root_gold-gap@10'],
        'native_pool_additional_fields':['N','median_rank_over_N','vs_random','gold_in_first_stage@1000','median_gold_rank_if_retrieved'],
        'bounded_pool_rank_policy':'median_gold_rank/median_rank_over_N/vs_random explicitly None in _pool_metrics; conditional median only, no fabricated full-lake rank',
    },
    'code_bindings':[{'path':str(REPO/p),'sha256':sha(REPO/p)} for p in paths],
    'future_A0_2_cli':{'prepare_a0_graph':'not implemented at A0.1 audit','train_rung --smoke-only':'not implemented','eval_y2 --protocol a0':'not implemented'},
    'remaining_execution_bindings':['actual formal execution host software defaults','actual new graph digest after A0.2','actual split/calibration query ID hashes','new effective config after A0.2 adaptations, checked identical in training recipe'],
}
assert payload['all_document_mapped_values_match']
assert all(x['exact_config_equal'] for x in linked)
target = HERE / 'A0_CONFIG_AUDIT.json'
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'path':str(target),'sha256':sha(target),'resolved_keys':len(cfg),'document_keys':len(doc_expected),'implementation_keys':len(cfg)-len(doc_expected),'all_document_keys_match':True,'all_3_doc_linked_configs_equal':True}))
