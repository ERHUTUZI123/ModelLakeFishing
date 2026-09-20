from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "docs" / "scale" / "P3" / "exports" / "ml_sub_L1L3b"
LAKE = ROOT / "stage1BuildTransferGraph" / "artifacts" / "modellens_v2_lake"
DEFAULT_JSON = ROOT / "docs" / "scale" / "P5" / "artifacts" / "cross_task_demo_audit.json"
DEFAULT_MD = Path(__file__).resolve().parents[3] / "weeks" / "week9_scale" / "PUBLIC_DEMO_CROSS_TASK_AUDIT.md"


AUDIT_CASES = [
    ("Text classification", 1659, "MTEB MTOPDomainClassification (en)"),
    ("Text retrieval", 1055, "MTEB ArguAna"),
    ("Reranking", 1062, "MTEB AskUbuntuDupQuestions"),
    ("Cross-lingual STS", 1984, "MTEB STS17 (fr-en)"),
    ("Clustering", 1085, "MTEB BiorxivClusteringS2S"),
    ("Bitext mining", 2060, "MTEB Tatoeba (ast-eng)"),
    ("Question answering / reasoning", 2615, "bbh_reasoning_about_colored_objects"),
    ("Multimodal VQA", 2460, "TextVQA"),
    ("Machine translation", 2778, "news-test2008"),
]

EXCLUDED_CASES = [
    {"dataset_id": 976, "dataset": "MMLU-Pro(acc)", "reason": "official description is missing/NaN"},
    {"dataset_id": 191, "dataset": "CrossNER_AI", "reason": "official description is missing/NaN"},
]


PUBLIC_DEMO_RESULTS = {
    1659: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "dqubit/frida-f16", "unknown", 23.8737, "—", 8),
            (2, "sncffcns/llm-jp-3-13b-it-20241127-lora", "llm", 22.6480, "—", 0),
            (3, "zomba/fsg-net", "unknown", 19.2341, "—", 0),
            (4, "samerzaher80/aethermind-srl", "unknown", 18.6596, "—", 0),
            (5, "lonestriker/skywork-13b-spicyboros-3-1-5-0bpw-h6-exl2", "unknown", 18.5158, "—", 0),
            (6, "169pi/alpie-core", "unknown", 17.9667, "—", 76),
            (7, "botbotrobotics/cabrallama3-70b", "llama", 17.7916, "70.0B", 13),
            (8, "viktorzver/frida", "unknown", 17.5736, "—", 18),
            (9, "leocristt/hackathon-embedding-model", "unknown", 16.6140, "—", 4),
            (10, "maziyarpanahi/calme-3-2-instruct-78b", "calme", 16.2538, "—", 0),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 3,
        "known_off_task": 6,
        "strict_compatible_ranks": [1, 8, 9],
        "known_off_task_ranks": [2, 3, 4, 5, 7, 10],
        "notes": "Three candidates are recognisable embedding models; no returned model has an exact MTOP result.",
    },
    1055: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "sncffcns/llm-jp-3-13b-it-20241127-lora", "llm", 17.7400, "—", 0),
            (2, "lonestriker/skywork-13b-spicyboros-3-1-5-0bpw-h6-exl2", "unknown", 17.0858, "—", 0),
            (3, "luminainc/jina-embeddings-v3", "unknown", 15.4773, "—", 26),
            (4, "bullerwins/mistral-large-instruct-2407-exl2-5-5bpw", "mistral", 14.9178, "—", 0),
            (5, "shindc/marian-finetuned-kde4-en-to-fr", "marian", 12.7700, "—", 6),
            (6, "prashrex/santacoder-gguf", "code", 12.7561, "—", 7),
            (7, "dqubit/frida-f16", "unknown", 12.6357, "—", 8),
            (8, "jiiyy/funnel", "unknown", 12.5007, "—", 2),
            (9, "devcamilosepulveda/333-qwen3sp-clover-usergrid", "qwen", 12.4882, "—", 1),
            (10, "ilhamdprastyo/jina-embeddings-v3-tei", "unknown", 12.4583, "—", 0),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 2,
        "known_off_task": 6,
        "strict_compatible_ranks": [3, 10],
        "known_off_task_ranks": [1, 2, 4, 5, 6, 9],
        "notes": "Two Jina embedding variants are strict retrieval matches; neither has an exact ArguAna record in the corpus.",
    },
    1062: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "hatemestinbejaia/mminilml-bi-encoder-kd-v1-student-tripletloss-teacher-marginloss-adptativemargin007", "code", 29.2712, "—", 0),
            (2, "dqubit/frida-f16", "unknown", 24.6899, "—", 8),
            (3, "zomba/fsg-net", "unknown", 20.6885, "—", 0),
            (4, "leocristt/hackathon-embedding-model", "unknown", 19.4686, "—", 4),
            (5, "hatemestinbejaia/mminilml-bi-encoder-kd-v1-0student-tripletlossadptativemargin-1teacher-marginloss-m15", "code", 19.0427, "—", 0),
            (6, "devcamilosepulveda/333-qwen3sp-clover-usergrid", "qwen", 17.9345, "—", 1),
            (7, "devcamilosepulveda/6-llama3sp-bamboo", "llama", 17.6915, "6.0B", 0),
            (8, "hkurita/sup-simcse-bert-base-uncased-mean", "bert", 17.5283, "110M", 4),
            (9, "bartowski/calmerys-78b-orpo-v0-1-gguf", "unknown", 17.3766, "—", 0),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 5,
        "known_off_task": 4,
        "strict_compatible_ranks": [1, 2, 4, 5, 8],
        "known_off_task_ranks": [3, 6, 7, 9],
        "notes": "The pasted primary table contains nine rows. An inconsistent isolated DOM card labelled rank 10 was excluded.",
    },
    1984: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "lonestriker/skywork-13b-spicyboros-3-1-5-0bpw-h6-exl2", "unknown", 20.2133, "—", 0),
            (2, "sncffcns/llm-jp-3-13b-it-20241127-lora", "llm", 18.6517, "—", 0),
            (3, "bullerwins/mistral-large-instruct-2407-exl2-5-5bpw", "mistral", 17.4262, "—", 0),
            (4, "gladiator/microsoft-deberta-v3-large-ner-wnut-17", "microsoft", 14.4577, "—", 0),
            (5, "lonestriker/skywork-13b-airo-claude-pippa-puffin-6-0bpw-h6-exl2", "claude", 14.3063, "—", 0),
            (6, "prashrex/santacoder-gguf", "code", 14.1499, "—", 7),
            (7, "hkurita/sup-simcse-bert-base-uncased-mean", "bert", 14.1405, "110M", 4),
            (8, "bighuggyd/mistralai-mistral-large-instruct-2407-exl2-4-0bpw-h8", "mistral", 13.8881, "—", 0),
            (9, "zomba/fsg-net", "unknown", 13.8122, "—", 0),
            (10, "shubhamrathore081/bge-base-en-1-5-ft", "unknown", 13.5839, "—", 0),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 0,
        "known_off_task": 8,
        "strict_compatible_ranks": [],
        "known_off_task_ranks": [1, 2, 3, 4, 5, 6, 8, 9],
        "notes": (
            "Two sentence-embedding models are English-only and fail the fr-en language "
            "requirement; the other eight cross task or modality boundaries."
        ),
    },
    1085: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "sncffcns/llm-jp-3-13b-it-20241127-lora", "llm", 33.0409, "—", 0),
            (2, "lonestriker/skywork-13b-spicyboros-3-1-5-0bpw-h6-exl2", "unknown", 18.9201, "—", 0),
            (3, "shubhamrathore081/bge-base-en-1-5-ft", "unknown", 18.0196, "—", 0),
            (4, "devcamilosepulveda/333-qwen3sp-clover-usergrid", "qwen", 17.4674, "—", 1),
            (5, "hkurita/sup-simcse-bert-base-uncased-mean", "bert", 16.6287, "110M", 4),
            (6, "zomba/fsg-net", "unknown", 16.4704, "—", 0),
            (7, "samerzaher80/aethermind-srl", "unknown", 16.2671, "—", 0),
            (8, "dqubit/frida-f16", "unknown", 16.2071, "—", 8),
            (9, "bullerwins/mistral-large-instruct-2407-exl2-5-5bpw", "mistral", 15.6878, "—", 0),
            (10, "gladiator/microsoft-deberta-v3-large-ner-wnut-17", "microsoft", 15.6260, "—", 0),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 3,
        "known_off_task": 7,
        "strict_compatible_ranks": [3, 5, 8],
        "known_off_task_ranks": [1, 2, 4, 6, 7, 9, 10],
        "notes": "Three sentence-embedding models are strict clustering matches; the remaining seven cross task or modality boundaries.",
    },
    2060: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "prashrex/santacoder-gguf", "code", 26.5994, "—", 7),
            (2, "bullerwins/mistral-large-instruct-2407-exl2-5-5bpw", "mistral", 24.7908, "—", 0),
            (3, "sangyup/multilingual-e5-local-copy", "e5", 24.0183, "—", 32),
            (4, "lonestriker/skywork-13b-spicyboros-3-1-5-0bpw-h6-exl2", "unknown", 23.7332, "—", 0),
            (5, "yixuan-chia/multilingual-e5-large-instruct-gguf", "e5", 23.0811, "—", 6),
            (6, "yoeven/multilingual-e5-large-instruct-q3-k-s-gguf", "e5", 22.7710, "—", 0),
            (7, "yoeven/multilingual-e5-large-instruct-q5-k-m-gguf", "e5", 22.5407, "—", 0),
            (8, "bighuggyd/mistralai-mistral-large-instruct-2407-exl2-4-0bpw-h8", "mistral", 22.2402, "—", 0),
            (9, "onelevelstudio/ml-e5-0-6b-instruct", "e5", 22.1457, "—", 0),
        ],
        "exact_dataset_verified": 5,
        "exact_dataset_near_optimal": 5,
        "verified_values": {
            "sangyup/multilingual-e5-local-copy": 0.897638,
            "yixuan-chia/multilingual-e5-large-instruct-gguf": 0.897638,
            "yoeven/multilingual-e5-large-instruct-q3-k-s-gguf": 0.897638,
            "yoeven/multilingual-e5-large-instruct-q5-k-m-gguf": 0.897638,
            "onelevelstudio/ml-e5-0-6b-instruct": 0.897638,
        },
        "strict_compatible": 5,
        "known_off_task": 4,
        "strict_compatible_ranks": [3, 5, 6, 7, 9],
        "known_off_task_ranks": [1, 2, 4, 8],
        "notes": "This is the genuine ModelLens success case: five multilingual E5 variants have exact, observed-optimal Tatoeba results. The pasted primary table contains nine rows.",
    },
    2615: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "sncffcns/llm-jp-3-13b-it-20241127-lora", "llm", 20.4711, "—", 0),
            (2, "dqubit/frida-f16", "unknown", 20.4181, "—", 8),
            (3, "zai-org/glm-4-7", "glm", 20.3069, "—", 0),
            (4, "lonestriker/skywork-13b-spicyboros-3-1-5-0bpw-h6-exl2", "unknown", 19.7895, "—", 0),
            (5, "samerzaher80/aethermind-srl", "unknown", 16.9927, "—", 0),
            (6, "zomba/fsg-net", "unknown", 16.9235, "—", 0),
            (7, "devcamilosepulveda/333-qwen3sp-clover-usergrid", "qwen", 16.6432, "—", 1),
            (8, "bullerwins/mistral-large-instruct-2407-exl2-5-5bpw", "mistral", 16.3332, "—", 0),
            (9, "deepset/flan-t5-xl-squad2", "flan", 16.1514, "3.0B", 11),
            (10, "devcamilosepulveda/3-qwen3sp-datamanagement", "qwen", 16.1196, "—", 0),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 5,
        "known_off_task": 4,
        "strict_compatible_ranks": [1, 3, 4, 8, 9],
        "known_off_task_ranks": [5, 6, 7, 10],
        "notes": "Several generative models are task-capable, but none has an exact result for this BBH query; four candidates are clearly task or language mismatches.",
    },
    2460: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "pcuenq/mobileclip-b-lt", "clip", 7.8389, "—", 33),
            (2, "pali-3-w/-ocr", "pali", 7.2695, "—", 0),
            (3, "sophosympatheia/aurora-nights-70b-v1-0", "aurora", 7.0491, "—", 0),
            (4, "line-corporation/clip-japanese-base", "clip", 6.9944, "—", 21092),
            (5, "umg-clip-l/14", "clip", 6.9933, "300M", 0),
            (6, "suko/janus", "janus", 6.9701, "—", 0),
            (7, "bbbbchan/llava-scissor-baseline-7b", "llava", 6.9667, "7.0B", 11),
            (8, "line-corporation/clip-japanese-base-v2", "clip", 6.8335, "200M", 13741),
            (9, "blip-vit/l-129m", "blip", 6.7959, "—", 0),
            (10, "rutenl/mobileclip2-s4-openclip-onnx", "clip", 6.7695, "—", 0),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 3,
        "known_off_task": 6,
        "strict_compatible_ranks": [2, 6, 7],
        "known_off_task_ranks": [1, 3, 4, 5, 8, 10],
        "notes": "Pali, Janus, and LLaVA are strict VQA-capable matches. Retrieval-only CLIP encoders cannot directly answer TextVQA questions.",
    },
    2778: {
        "run_timestamp": "2026-07-28, user-supplied public-demo run",
        "rows": [
            (1, "sncffcns/llm-jp-3-13b-it-20241127-lora", "llm", 29.2083, "—", 0),
            (2, "lonestriker/skywork-13b-spicyboros-3-1-5-0bpw-h6-exl2", "unknown", 25.8197, "—", 0),
            (3, "bullerwins/mistral-large-instruct-2407-exl2-5-5bpw", "mistral", 22.3544, "—", 0),
            (4, "prashrex/santacoder-gguf", "code", 21.3889, "—", 7),
            (5, "lonestriker/skywork-13b-airo-claude-pippa-puffin-6-0bpw-h6-exl2", "claude", 19.2768, "—", 0),
            (6, "blemond/0910a", "unknown", 19.2444, "—", 0),
            (7, "bighuggyd/mistralai-mistral-large-instruct-2407-exl2-4-0bpw-h8", "mistral", 19.1247, "—", 0),
            (8, "wilfredomartel/bge-m3-es-legal-155k", "unknown", 18.7388, "—", 163),
            (9, "luminainc/jina-embeddings-v3", "unknown", 18.6452, "—", 26),
            (10, "169pi/alpie-core", "unknown", 18.5958, "—", 76),
        ],
        "exact_dataset_verified": 0,
        "exact_dataset_near_optimal": 0,
        "strict_compatible": 0,
        "known_off_task": 4,
        "strict_compatible_ranks": [],
        "known_off_task_ranks": [1, 4, 8, 9],
        "notes": "No returned candidate is a confirmed German-to-English translation model; none has an exact news-test2008 result.",
    },
}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def build_cases() -> tuple[list[dict], dict]:
    model_ids = pd.read_csv(EXPORT / "model_ids.csv").sort_values("mappedID")
    dataset_ids = pd.read_csv(EXPORT / "dataset_ids.csv").sort_values("mappedID")
    pool = pd.read_csv(LAKE / "ml_dataset_pool.csv")
    gold = np.load(EXPORT / "gold_cands.npz")

    model_names = model_ids["model"].tolist()
    dataset_nodes = dataset_ids["dataset"].tolist()
    z_model = np.load(EXPORT / "z_m_eval.npy").astype(np.float64)
    z_dataset = np.load(EXPORT / "z_d_eval.npy").astype(np.float64)
    z_model /= np.linalg.norm(z_model, axis=1, keepdims=True) + 1e-12
    z_dataset /= np.linalg.norm(z_dataset, axis=1, keepdims=True) + 1e-12

    cases = []
    for family, dataset_id, expected_name in AUDIT_CASES:
        key = str(dataset_id)
        if key not in gold.files:
            raise KeyError(f"dataset {dataset_id} is not a held-out gold query")
        node = dataset_nodes[dataset_id]
        pool_rows = pool[pool["dataset_node"].eq(node)]
        if len(pool_rows) != 1:
            raise ValueError(f"expected one pool row for {node!r}, found {len(pool_rows)}")
        item = pool_rows.iloc[0]
        if str(item["dataset"]) != expected_name:
            raise ValueError(
                f"selection drift for id={dataset_id}: {item['dataset']!r} != {expected_name!r}"
            )

        candidates = gold[key][0].astype(int)
        values = gold[key][1].astype(float)
        observed = {int(m): float(v) for m, v in zip(candidates, values)}
        scores = z_model @ z_dataset[dataset_id]
        top = np.argsort(-scores, kind="stable")[:10]
        top_rows = []
        for rank, model_id in enumerate(top, 1):
            value = observed.get(int(model_id))
            top_rows.append(
                {
                    "rank": rank,
                    "model": model_names[int(model_id)],
                    "observed_value": value,
                    "mips": float(scores[int(model_id)]),
                }
            )

        best = float(values.max())
        verified = sum(row["observed_value"] is not None for row in top_rows)
        near = sum(
            row["observed_value"] is not None
            and float(row["observed_value"]) >= best - 0.01
            for row in top_rows
        )
        cases.append(
            {
                "task_family": family,
                "dataset_id": dataset_id,
                "dataset": str(item["dataset"]),
                "dataset_node": node,
                "task_type": str(item["task"]),
                "metric": str(item["chosen_metric"]),
                "description": str(item["desc"]),
                "labeled_candidates": int(len(candidates)),
                "observed_optimum": best,
                "model_lake_verified@10": verified,
                "model_lake_near_optimal@10": near,
                "model_lake_top10": top_rows,
                "modellens_public_demo": PUBLIC_DEMO_RESULTS.get(dataset_id),
            }
        )

    protocol = {
        "selection_frozen_before_public_demo_collection": True,
        "selection_kind": "cross-task qualitative capability battery",
        "prevalence_source": "deployment_quality_metrics.json over all 517 held-out queries",
        "model_lake_path": "z_m_eval @ z_d_eval, leakage-free held-out MIPS",
        "description_source": "ModelLens corpus-v2 via ml_dataset_pool.csv desc field",
        "missing_observed_value": "unknown, displayed as em dash",
        "excluded_missing_description": EXCLUDED_CASES,
        "public_demo_capture_policy": (
            "The contiguous primary rank table is authoritative. Isolated, inconsistent "
            "DOM cards outside that table are excluded and disclosed in the case notes."
        ),
        "manual_compatibility_policy": (
            "Strict-compatible requires direct support for the requested task, modality, "
            "and language. Known off-task is used only for an explicit role, modality, or "
            "language mismatch. Remaining candidates are marked ambiguous rather than failed."
        ),
    }
    return cases, protocol


def aggregate_summary(cases: list[dict]) -> dict:
    demos = [case["modellens_public_demo"] for case in cases if case["modellens_public_demo"]]
    public_rows = sum(len(demo["rows"]) for demo in demos)
    strict = sum(demo["strict_compatible"] for demo in demos)
    off_task = sum(demo["known_off_task"] for demo in demos)
    return {
        "included_cases": len(cases),
        "excluded_missing_description_cases": len(EXCLUDED_CASES),
        "model_lake_rows": sum(len(case["model_lake_top10"]) for case in cases),
        "model_lake_exact_verified": sum(case["model_lake_verified@10"] for case in cases),
        "model_lake_exact_near_optimal": sum(
            case["model_lake_near_optimal@10"] for case in cases
        ),
        "public_demo_collected_cases": len(demos),
        "public_demo_captured_rows": public_rows,
        "public_demo_exact_verified": sum(demo["exact_dataset_verified"] for demo in demos),
        "public_demo_exact_near_optimal": sum(
            demo["exact_dataset_near_optimal"] for demo in demos
        ),
        "public_demo_strict_compatible": strict,
        "public_demo_known_off_task": off_task,
        "public_demo_unclassified_or_ambiguous": public_rows - strict - off_task,
        "public_demo_zero_exact_coverage_cases": sum(
            demo["exact_dataset_verified"] == 0 for demo in demos
        ),
    }


def render_markdown(cases: list[dict], protocol: dict) -> str:
    aggregate = aggregate_summary(cases)
    summary_rows = []
    for index, case in enumerate(cases, 1):
        summary_rows.append(
            [
                str(index),
                case["task_family"],
                f"`{case['dataset']}`",
                f"`{case['task_type']}`",
                f"`{case['metric']}`",
                str(case["labeled_candidates"]),
                f"{case['model_lake_verified@10']}/10",
                f"{case['model_lake_near_optimal@10']}/10",
                (
                    f"{len(case['modellens_public_demo']['rows'])}"
                    if case["modellens_public_demo"]
                    else "Pending"
                ),
                (
                    str(case["modellens_public_demo"]["exact_dataset_verified"])
                    if case["modellens_public_demo"]
                    else "Pending"
                ),
                (
                    str(case["modellens_public_demo"]["strict_compatible"])
                    if case["modellens_public_demo"]
                    else "Pending"
                ),
            ]
        )

    doc = [
        "# Cross-Task Public Demo Audit Run Sheet",
        "",
        "This run sheet freezes the query set before collection of any ModelLens public-demo output. It is a qualitative capability battery spanning nine recognisable task families. Aggregate prevalence remains defined by the 517-query deployment evaluation in `deployment_quality_metrics.json`.",
        "",
        "Model Lake results use leakage-free held-out MIPS. Dataset descriptions are copied verbatim from the ModelLens corpus-v2 `desc` field. An em dash in a Model Lake table indicates an unknown exact-dataset result and does not assert poor performance.",
        "",
        "## Audit Summary",
        "",
        _md_table(
            [
                "Case",
                "Task family",
                "Dataset",
                "Demo task type",
                "Metric",
                "Labeled candidates",
                "Lake verified",
                "Lake near-optimal",
                "Lens rows captured",
                "Lens exact verified",
                "Lens strict compatible",
            ],
            summary_rows,
        ),
        "",
        "## Cross-Case Result",
        "",
        _md_table(
            ["Measure", "Model Lake", "ModelLens public demo"],
            [
                [
                    "Exact-dataset verified recommendations",
                    f"**{aggregate['model_lake_exact_verified']}/{aggregate['model_lake_rows']}**",
                    f"{aggregate['public_demo_exact_verified']}/{aggregate['public_demo_captured_rows']}",
                ],
                [
                    "Exact-dataset near-optimal recommendations",
                    f"**{aggregate['model_lake_exact_near_optimal']}/{aggregate['model_lake_rows']}**",
                    f"{aggregate['public_demo_exact_near_optimal']}/{aggregate['public_demo_captured_rows']}",
                ],
                [
                    "Strict task/modality/language compatibility",
                    "Not manually relabeled",
                    f"{aggregate['public_demo_strict_compatible']}/{aggregate['public_demo_captured_rows']}",
                ],
                [
                    "Known off-task recommendations",
                    "Not manually relabeled",
                    f"{aggregate['public_demo_known_off_task']}/{aggregate['public_demo_captured_rows']}",
                ],
            ],
        ),
        "",
        f"ModelLens has zero exact-dataset coverage in **{aggregate['public_demo_zero_exact_coverage_cases']}/{aggregate['public_demo_collected_cases']}** collected cases. The exception is `MTEB Tatoeba (ast-eng)`, where five of the nine captured primary-table rows are multilingual E5 variants with exact, observed-optimal results. This exception is retained rather than suppressed.",
        "",
        "Two pre-registered cases were not run because the official ModelLens corpus description is missing/NaN. They are excluded from every numerator and denominator:",
        "",
        *[
            f"- `{item['dataset']}` (dataset ID `{item['dataset_id']}`): {item['reason']}."
            for item in EXCLUDED_CASES
        ],
        "",
        "## Collection Protocol",
        "",
        "For each case, paste the description and task type exactly as shown. Preserve the demo rank, model, family, score, size, popularity, and link fields without interpretation. Record the run timestamp and retain a screenshot when possible.",
        "",
        "Compatibility labels are conservative manual audit labels. `strict-compatible` requires direct support for the requested task, modality, and language; `known off-task` requires an explicit mismatch. Any unresolved candidate remains `ambiguous` and is not counted as off-task.",
        "",
    ]

    for index, case in enumerate(cases, 1):
        lake_rows = []
        for row in case["model_lake_top10"]:
            lake_rows.append(
                [
                    str(row["rank"]),
                    f"`{row['model']}`",
                    _fmt(row["observed_value"]),
                    f"{row['mips']:.4f}",
                ]
            )
        demo = case["modellens_public_demo"]
        if demo:
            verified_values = demo.get("verified_values", {})
            strict_ranks = set(demo["strict_compatible_ranks"])
            off_task_ranks = set(demo["known_off_task_ranks"])
            demo_rows = [
                [
                    str(rank),
                    f"`{model}`",
                    family,
                    f"{score:.4f}",
                    size,
                    str(popularity),
                    _fmt(verified_values.get(model)),
                    (
                        "strict-compatible"
                        if rank in strict_ranks
                        else "known off-task"
                        if rank in off_task_ranks
                        else "ambiguous"
                    ),
                    f"[link](https://huggingface.co/{model})",
                ]
                for rank, model, family, score, size, popularity in demo["rows"]
            ]
            timestamp = demo["run_timestamp"]
            captured = len(demo["rows"])
            ambiguous = captured - demo["strict_compatible"] - demo["known_off_task"]
            audit_lines = [
                f"- Exact-dataset verified coverage: `{demo['exact_dataset_verified']}/{captured}`",
                f"- Exact-dataset near-optimal count: `{demo['exact_dataset_near_optimal']}/{captured}`",
                f"- Strict task/modality/language compatibility: `{demo['strict_compatible']}/{captured}`",
                f"- Known off-task contamination: `{demo['known_off_task']}/{captured}`",
                f"- Unclassified or ambiguous: `{ambiguous}/{captured}`",
                f"- Notes: {demo['notes']}",
            ]
        else:
            demo_rows = [
                [
                    str(rank),
                    "TBD",
                    "TBD",
                    "TBD",
                    "TBD",
                    "TBD",
                    "TBD",
                    "TBD",
                    "TBD",
                ]
                for rank in range(1, 11)
            ]
            timestamp = "TBD"
            audit_lines = [
                "- Exact-dataset verified coverage: `TBD/10`",
                "- Exact-dataset near-optimal count: `TBD/10`",
                "- Strict task/modality/language compatibility: `TBD/10`",
                "- Known off-task contamination: `TBD/10`",
                "- Unclassified or ambiguous: `TBD/10`",
                "- Notes: `TBD`",
            ]
        doc.extend(
            [
                f"## Case {index}: {case['task_family']} — {case['dataset']}",
                "",
                f"- Internal held-out dataset ID: `{case['dataset_id']}`",
                f"- Task type: `{case['task_type']}`",
                f"- Metric: `{case['metric']}`",
                f"- Labeled candidates: `{case['labeled_candidates']}`",
                f"- Observed optimum: `{case['observed_optimum']:.4f}`",
                f"- Model Lake verified coverage: `{case['model_lake_verified@10']}/10`",
                f"- Model Lake near-optimal count: `{case['model_lake_near_optimal@10']}/10`",
                "",
                "### ModelLens Demo Input",
                "",
                "**Task type**",
                "",
                "```text",
                case["task_type"],
                "```",
                "",
                "**Metric, if requested by the UI**",
                "",
                "```text",
                case["metric"],
                "```",
                "",
                "**Description**",
                "",
                "```text",
                case["description"],
                "```",
                "",
                "### Model Lake Top 10",
                "",
                _md_table(
                    ["Rank", "Model", "Observed normalized value", "MIPS"], lake_rows
                ),
                "",
                "### ModelLens Public Demo Top 10",
                "",
                f"Run timestamp: `{timestamp}`",
                "",
                _md_table(
                    [
                        "Rank",
                        "Model",
                        "Family",
                        "Score",
                        "Size",
                        "Popularity",
                        "Exact-dataset value",
                        "Audit class",
                        "Link",
                    ],
                    demo_rows,
                ),
                "",
                "### Audit Result",
                "",
                *audit_lines,
                "",
            ]
        )
    return "\n".join(doc).rstrip() + "\n"


def _selftest() -> int:
    assert len(AUDIT_CASES) == 9
    assert len({dataset_id for _, dataset_id, _ in AUDIT_CASES}) == len(AUDIT_CASES)
    cases, protocol = build_cases()
    assert len(cases) == 9
    assert all(len(case["model_lake_top10"]) == 10 for case in cases)
    assert all(case["description"] and case["task_type"] for case in cases)
    assert len([case for case in cases if case["modellens_public_demo"]]) == 9
    for case in cases:
        demo = case["modellens_public_demo"]
        captured_ranks = {row[0] for row in demo["rows"]}
        strict_ranks = set(demo["strict_compatible_ranks"])
        off_task_ranks = set(demo["known_off_task_ranks"])
        assert strict_ranks.isdisjoint(off_task_ranks)
        assert strict_ranks | off_task_ranks <= captured_ranks
        assert len(strict_ranks) == demo["strict_compatible"]
        assert len(off_task_ranks) == demo["known_off_task"]
    aggregate = aggregate_summary(cases)
    assert aggregate["model_lake_exact_verified"] == 72
    assert aggregate["model_lake_exact_near_optimal"] == 33
    assert aggregate["public_demo_captured_rows"] == 88
    assert aggregate["public_demo_exact_verified"] == 5
    assert aggregate["public_demo_exact_near_optimal"] == 5
    assert aggregate["public_demo_strict_compatible"] == 26
    assert aggregate["public_demo_known_off_task"] == 49
    assert aggregate["public_demo_unclassified_or_ambiguous"] == 13
    assert aggregate["public_demo_zero_exact_coverage_cases"] == 8
    rendered = render_markdown(cases, protocol)
    assert rendered.count("### ModelLens Public Demo Top 10") == 9
    print("cross_task_demo_audit self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()

    cases, protocol = build_cases()
    report = {
        "protocol": protocol,
        "aggregate": aggregate_summary(cases),
        "excluded_cases": EXCLUDED_CASES,
        "cases": cases,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(cases, protocol), encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {args.out_md}")
    print(f"machine-readable audit -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
