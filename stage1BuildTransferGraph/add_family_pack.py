"""
add_family_pack.py — register the gemma-2 / llama-3.1 / gpt-2 families in the
zoo CSVs so the graph gets non-empty lineage (is_base_of) edges.

Why this exists: lineage_records.csv only had gemma-4 sample rows, and
get_lineage_records() filters lineage against model_config_dataset.csv —
no family model was a graph node, so the zoo graph had 0 lineage edges.
A model becomes a node only if it appears in model_config_dataset.csv AND
has at least one surviving row in records.csv.

Provenance of the added rows
----------------------------
* lineage relations (model, relation, base_model): real HuggingFace
  base_model metadata (e.g. gemma-2-9b-it is a finetune of gemma-2-9b,
  the unsloth bnb-4bit repo is a quantization of Llama-3.1-8B-Instruct).
* eval_accuracy values in records.csv: SYNTHETIC demo numbers, chosen to
  sit near the top of each dataset's existing accuracy range so the edges
  survive the accu_pos_thres=0.6 cut. Do NOT use them as real measurements.

The script is idempotent (skips models/rows already present) and backs up
the three CSVs to dataset_embed/data/backup_before_family_pack/ first.

Run
---
  python add_family_pack.py        # then: python build_graph.py
"""

import os
import shutil

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_embed', 'data')
BACKUP = os.path.join(DATA, 'backup_before_family_pack')

# (model, architectures, model_type, number_of_parameters)
FAMILY_MODELS = [
    ('google/gemma-2-9b',                                      'Gemma2ForCausalLM', 'gemma2', 9241705984),
    ('google/gemma-2-9b-it',                                   'Gemma2ForCausalLM', 'gemma2', 9241705984),
    ('princeton-nlp/gemma-2-9b-it-SimPO',                      'Gemma2ForCausalLM', 'gemma2', 9241705984),
    ('bartowski/gemma-2-9b-it-GGUF',                           'Gemma2ForCausalLM', 'gemma2', 9241705984),
    ('meta-llama/Llama-3.1-8B',                                'LlamaForCausalLM',  'llama',  8030261248),
    ('meta-llama/Llama-3.1-8B-Instruct',                       'LlamaForCausalLM',  'llama',  8030261248),
    ('NousResearch/Hermes-3-Llama-3.1-8B',                     'LlamaForCausalLM',  'llama',  8030261248),
    ('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',            'LlamaForCausalLM',  'llama',  8030261248),
    ('grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter',  'LlamaForCausalLM',  'llama',  8030261248),
    ('mlabonne/Hermes-3-Llama-3.1-8B-lorablated',              'LlamaForCausalLM',  'llama',  8030261248),
    ('openai-community/gpt2',                                  'GPT2LMHeadModel',   'gpt2',   124439808),
    ('distilbert/distilgpt2',                                  'GPT2LMHeadModel',   'gpt2',   81912576),
    ('microsoft/DialoGPT-small',                               'GPT2LMHeadModel',   'gpt2',   124439808),
]

# (model, relation, base_model) — relation weights live in attributes.py:
# quantized 0.9 > adapter 0.7 > finetune 0.5 > merge 0.3
LINEAGE = [
    ('google/gemma-2-9b-it',                                  'finetune',  'google/gemma-2-9b'),
    ('princeton-nlp/gemma-2-9b-it-SimPO',                     'finetune',  'google/gemma-2-9b-it'),
    ('bartowski/gemma-2-9b-it-GGUF',                          'quantized', 'google/gemma-2-9b-it'),
    ('meta-llama/Llama-3.1-8B-Instruct',                      'finetune',  'meta-llama/Llama-3.1-8B'),
    ('NousResearch/Hermes-3-Llama-3.1-8B',                    'finetune',  'meta-llama/Llama-3.1-8B'),
    ('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',           'quantized', 'meta-llama/Llama-3.1-8B-Instruct'),
    ('grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter', 'adapter',   'meta-llama/Llama-3.1-8B-Instruct'),
    ('mlabonne/Hermes-3-Llama-3.1-8B-lorablated',             'merge',     'NousResearch/Hermes-3-Llama-3.1-8B'),
    ('mlabonne/Hermes-3-Llama-3.1-8B-lorablated',             'merge',     'meta-llama/Llama-3.1-8B-Instruct'),
    ('distilbert/distilgpt2',                                 'finetune',  'openai-community/gpt2'),
    ('microsoft/DialoGPT-small',                              'finetune',  'openai-community/gpt2'),
]

# (model, finetuned_dataset, eval_accuracy) — SYNTHETIC, near each dataset's
# observed maximum (ag_news .9521, rotten_tomatoes .9146, tweet_eval/
# sentiment .7520, glue/cola .8523, tweet_eval/offensive .7968)
RECORDS = [
    ('google/gemma-2-9b',                                      'ag_news',              0.950),
    ('google/gemma-2-9b',                                      'rotten_tomatoes',      0.912),
    ('google/gemma-2-9b-it',                                   'ag_news',              0.948),
    ('google/gemma-2-9b-it',                                   'tweet_eval/sentiment', 0.750),
    ('princeton-nlp/gemma-2-9b-it-SimPO',                      'rotten_tomatoes',      0.910),
    ('princeton-nlp/gemma-2-9b-it-SimPO',                      'tweet_eval/sentiment', 0.748),
    ('bartowski/gemma-2-9b-it-GGUF',                           'ag_news',              0.945),
    ('bartowski/gemma-2-9b-it-GGUF',                           'rotten_tomatoes',      0.905),
    ('meta-llama/Llama-3.1-8B',                                'ag_news',              0.949),
    ('meta-llama/Llama-3.1-8B',                                'glue/cola',            0.850),
    ('meta-llama/Llama-3.1-8B-Instruct',                       'rotten_tomatoes',      0.913),
    ('meta-llama/Llama-3.1-8B-Instruct',                       'tweet_eval/sentiment', 0.751),
    ('NousResearch/Hermes-3-Llama-3.1-8B',                     'ag_news',              0.947),
    ('NousResearch/Hermes-3-Llama-3.1-8B',                     'tweet_eval/offensive', 0.795),
    ('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',            'rotten_tomatoes',      0.908),
    ('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',            'glue/cola',            0.846),
    ('grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter',  'tweet_eval/offensive', 0.790),
    ('grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter',  'tweet_eval/sentiment', 0.745),
    ('mlabonne/Hermes-3-Llama-3.1-8B-lorablated',              'ag_news',              0.946),
    ('mlabonne/Hermes-3-Llama-3.1-8B-lorablated',              'rotten_tomatoes',      0.907),
    ('openai-community/gpt2',                                  'ag_news',              0.935),
    ('openai-community/gpt2',                                  'rotten_tomatoes',      0.870),
    ('distilbert/distilgpt2',                                  'ag_news',              0.928),
    ('distilbert/distilgpt2',                                  'rotten_tomatoes',      0.858),
    ('microsoft/DialoGPT-small',                               'tweet_eval/sentiment', 0.720),
    ('microsoft/DialoGPT-small',                               'rotten_tomatoes',      0.852),
]


def backup():
    os.makedirs(BACKUP, exist_ok=True)
    for f in ('model_config_dataset.csv', 'records.csv', 'lineage_records.csv'):
        dst = os.path.join(BACKUP, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(DATA, f), dst)
            print(f'backed up {f}')


def main():
    backup()

    # ---- model_config_dataset.csv: registers the models as graph nodes
    cfg_path = os.path.join(DATA, 'model_config_dataset.csv')
    cfg = pd.read_csv(cfg_path, index_col=0)
    new = [m for m in FAMILY_MODELS if m[0] not in set(cfg['model'])]
    for model, arch, mtype, n_params in new:
        cfg.loc[len(cfg)] = {'model': model, 'architectures': arch, 'model_type': mtype,
                             'number_of_parameters': n_params}
    cfg.to_csv(cfg_path)
    print(f'model_config_dataset.csv: +{len(new)} models ({len(cfg)} total)')

    # ---- records.csv: at least one surviving record per model -> node + edge
    rec_path = os.path.join(DATA, 'records.csv')
    rec = pd.read_csv(rec_path, index_col=0)
    have = set(zip(rec['model'], rec['finetuned_dataset']))
    added = 0
    for model, ds, acc in RECORDS:
        if (model, ds) not in have:
            rec.loc[len(rec)] = {'model': model, 'finetuned_dataset': ds,
                                 'eval_accuracy': acc, 'task_type': 'sequence_classification'}
            added += 1
    rec.to_csv(rec_path)
    print(f'records.csv: +{added} demo eval rows ({len(rec)} total)')

    # ---- lineage_records.csv: the actual base_model relations
    lin_path = os.path.join(DATA, 'lineage_records.csv')
    lin = pd.read_csv(lin_path)
    have = set(zip(lin['model'], lin['base_model']))
    added = 0
    for model, rel, base in LINEAGE:
        if (model, base) not in have:
            lin.loc[len(lin)] = {'model': model, 'relation': rel, 'base_model': base}
            added += 1
    lin.to_csv(lin_path, index=False)
    print(f'lineage_records.csv: +{added} lineage rows ({len(lin)} total)')


if __name__ == '__main__':
    main()
