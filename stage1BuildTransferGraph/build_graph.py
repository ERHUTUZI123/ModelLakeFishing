"""
build_graph.py — stage-1 entry point: records.csv -> GraphAttributes -> HGraph.

This is the production chain (the counterpart of the original repo's
gnn_train() glue in train_with_GNN.py): it reads the TransferGraph zoo
resources, builds all node/edge attributes via GraphAttributesWithDomain-
Similarity (including the x_m^(0) package when --contain_model_feature is
on), assembles the HGraph, and saves the HeteroData to disk.

Data prerequisites
------------------
* TransferGraph checkout with resources/experiments/<task_type>/ —
  default location is ../../transfergraph (override: TRANSFERGRAPH_ROOT)
* dataset embeddings under embedded_dataset/domain_similarity/<ref_model>/
  (downloadable from huggingface.co/datasets/TransferGraph/...)

Run
---
  cd stage1BuildTransferGraph

  # smoke run: random model features, no HuggingFace fetching
  python build_graph.py

  # full run: builds x_m^(0) for every zoo model — first run fetches each
  # model's README + param count from HuggingFace (resumable caches under
  # dataset_embed/data/), encodes descriptions with sentence-transformers
  python build_graph.py --contain_model_feature True
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_embed.utils.task import TaskType
from dataset_embed.utils.embed_utils import DatasetEmbeddingMethod
from dataset_embed.utils.graph import HGraph
from attributes import GraphAttributesWithDomainSimilarity

logger = logging.getLogger(__name__)


def str2bool(v) -> bool:
    return str(v).lower() in ("yes", "true", "t", "1")


def parse_args():
    # defaults mirror the original repo's tools/2-transferability_estimation/run.py
    p = argparse.ArgumentParser(description="Build the stage-1 HGraph from zoo records")
    p.add_argument('--task_type', default=TaskType.SEQUENCE_CLASSIFICATION, type=TaskType)
    p.add_argument('--gnn_method', default='SAGEConv', type=str)
    p.add_argument('--test_dataset', default='glue/sst2', type=str,
                   help='dataset whose accuracy edges are held out of the graph')
    p.add_argument('--contain_dataset_feature', default='True', type=str2bool)
    p.add_argument('--contain_data_similarity', default='True', type=str2bool)
    p.add_argument('--contain_model_feature', default='False', type=str2bool,
                   help='True triggers the x_m^(0) build (HuggingFace fetching on first run)')
    p.add_argument('--dataset_reference_model', default='EleutherAI_gpt-neo-125m', type=str)
    p.add_argument('--dataset_embed_method', default=DatasetEmbeddingMethod.DOMAIN_SIMILARITY,
                   type=DatasetEmbeddingMethod)
    p.add_argument('--dataset_distance_method', default='euclidean', type=str)
    p.add_argument('--finetune_ratio', default=1.0, type=float)
    p.add_argument('--top_pos_K', default=0.5, type=float)
    p.add_argument('--top_neg_K', default=0.5, type=float)
    p.add_argument('--accu_pos_thres', default=0.6, type=float)
    p.add_argument('--accu_neg_thres', default=0.2, type=float)
    p.add_argument('--distance_thres', default=-1, type=float)
    p.add_argument('--peft_method', default=None, type=str, choices=[None, 'lora'])
    p.add_argument('--out', default='hgraph_zoo.pt', type=str,
                   help='where to save the assembled HeteroData')
    return p.parse_args()


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()

    ga = GraphAttributesWithDomainSimilarity(args)

    graph = HGraph(
        args.gnn_method,
        ga.max_dataset_idx,
        ga.model_idx,
        ga.unique_model_id,
        ga.model_features,
        ga.unique_dataset_id,
        ga.data_features,
        ga.edge_index_accu_model_to_dataset,
        ga.edge_attr_accu_model_to_dataset,
        ga.edge_index_dataset_to_dataset,
        ga.edge_attr_dataset_to_dataset,
        ga.edge_index_tran_model_to_dataset,
        ga.edge_attr_tran_model_to_dataset,
        ga.edge_index_model_to_model,
        ga.edge_attr_model_to_model,
        ga.negative_pairs,
        contain_data_similarity=args.contain_data_similarity,
        contain_dataset_feature=args.contain_dataset_feature,
        contain_model_feature=args.contain_model_feature,
        model_size_bucket_id=ga.model_size_bucket_id,
        model_family_id=ga.model_family_id,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    payload = {
        'data': graph.data,
        'unique_model_id': ga.unique_model_id,
        'unique_dataset_id': ga.unique_dataset_id,
        'args': vars(args) | {'task_type': args.task_type.value,
                              'dataset_embed_method': args.dataset_embed_method.value},
    }
    if ga.xm0 is not None:
        # everything ModelNodeEncoder needs at training time (frozen features
        # and index columns already live inside graph.data['model'])
        payload['xm0_meta'] = {
            'num_size_buckets': ga.xm0['num_size_buckets'],
            'num_families': ga.xm0['num_families'],
            'family_vocab': ga.xm0['family_vocab'],
            'name_dim': ga.xm0['name_dim'],
            'desc_dim': ga.xm0['desc_dim'],
        }
    torch.save(payload, out_path)

    logger.info(f"saved HeteroData + id tables to {out_path}")
    logger.info(f"models: {graph.data['model'].num_nodes}, "
                f"datasets: {graph.data['dataset'].num_nodes}")
    for et in graph.data.edge_types:
        logger.info(f"  {et}: {graph.data[et].edge_index.shape[1]} edges")


if __name__ == '__main__':
    main()
