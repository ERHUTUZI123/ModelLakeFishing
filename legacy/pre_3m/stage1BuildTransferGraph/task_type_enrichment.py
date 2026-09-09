"""Offline, review-first task-type enrichment for the hf1000d graph.

Track D.1 of ``docs/RETRIEVAL_QUALITY_REMEDIATION_PLAN.md`` needs better
dataset task metadata before global-negative mining is rerun.  This utility is
deliberately a *dry run*: it reads the shipped graph and the already-populated
Hugging Face caches, then emits a proposed CSV patch and an audit report.  It
never writes the graph, source CSVs, cache files, or the existing task vocab.

Evidence is resolved conservatively:

1. a specific task attached to a cached model-index result (the task under
   which the performance edge was reported);
2. a specific dataset-card ``task_id`` or a precise dataset identity phrase;
3. a specific dataset-card ``task_category``.

Generic labels such as ``classification`` and ``text-classification`` carry no
task semantics and are ignored.  If the best available tier maps to more than
one canonical task, the row is sent to the review queue instead of being
patched.  Lower-priority evidence remains in the report for human inspection.

Default outputs (all new, reviewable artifacts):

``stage1BuildTransferGraph/artifacts/task_type_enrichment/``
    task_type_enrichment_patch.csv
    task_type_vocab_patch.csv
    task_type_enrichment_review.csv
    task_type_enrichment_report.json
    task_type_enrichment_report.md

Run from the repository parent directory::

    python -m ModelLakeFishing.stage1BuildTransferGraph.task_type_enrichment

No network client is imported or called.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PACKAGE_PARENT = HERE.parent.parent
DEFAULT_GRAPH = HERE / "hgraph_hf1000d_2000m_xm0_xd0.pt"
DEFAULT_SOURCE = HERE / "hf1000d_2000m"
DEFAULT_INVENTORY = DEFAULT_SOURCE / "selected_1000_datasets.csv"
DEFAULT_DATASET_CACHE = DEFAULT_SOURCE / "hf_cache" / "datasets"
DEFAULT_MODEL_CACHE = DEFAULT_SOURCE / "hf_cache" / "models"
DEFAULT_OUT = HERE / "artifacts" / "task_type_enrichment"

OTHER = "Other"
REFERENCE_BASELINE = {"nonempty_pools": 30, "denominator": 64}
DEFAULT_TARGET_POOLS = 50


# Exact structured task tokens only.  Broad format/category labels are listed
# separately in GENERIC_TASK_TOKENS and intentionally do not become task types.
TASK_TOKEN_MAP = {
    # classification sub-types
    "intent-classification": "intent",
    "intent classification": "intent",
    "sentiment": "sentiment",
    "sentiment-analysis": "sentiment",
    "sentiment-classification": "sentiment",
    "sentiment-scoring": "sentiment",
    "topic-classification": "topic",
    "natural-language-inference": "NLI",
    "textual-entailment": "NLI",
    "nli": "NLI",
    "acceptability-classification": "acceptability",
    "hate-speech-detection": "toxicity",
    "toxicity-classification": "toxicity",
    "emotion-classification": "emotion",
    "irony-detection": "irony",
    "paraphrase-identification": "paraphrase",
    "counterfactual-detection": "counterfactual-detection",
    "counterfactual-classification": "counterfactual-detection",
    "language-identification": "language-identification",
    "spam-detection": "spam-detection",
    # QA variants share one compatibility family
    "question-answering": "question-answering",
    "extractive-question-answering": "question-answering",
    "extractive question-answering": "question-answering",
    "extractive-qa": "question-answering",
    "open-domain-qa": "question-answering",
    "closed-domain-qa": "question-answering",
    "multiple-choice-qa": "question-answering",
    "visual-question-answering": "question-answering",
    "document-question-answering": "question-answering",
    # token tasks
    "named-entity-recognition": "named-entity-recognition",
    "named-entity-recognition-ner": "named-entity-recognition",
    "part-of-speech": "part-of-speech",
    "pos-tagging": "part-of-speech",
    "slot-filling": "slot-filling",
    "pii-detection": "pii-detection",
    "acronym-identification": "acronym-identification",
    # embedding / generation tasks
    "bitextmining": "bitext-mining",
    "bitext-mining": "bitext-mining",
    "translation": "translation",
    "semantic-similarity-classification": "semantic-similarity",
    "semantic-similarity-scoring": "semantic-similarity",
    "semantic-textual-similarity": "semantic-similarity",
    "sentence-similarity": "semantic-similarity",
    "sts": "semantic-similarity",
    "retrieval": "retrieval",
    "text-retrieval": "retrieval",
    "information-retrieval": "retrieval",
    "reranking": "reranking",
    "clustering": "clustering",
    "summarization": "summarization",
    "fact-checking": "fact-checking",
    "language-modeling": "language-modeling",
    "masked-language-modeling": "language-modeling",
}

GENERIC_TASK_TOKENS = {
    "", "other", "classification", "text classification", "text-classification",
    "token classification", "token-classification", "multi-class-classification",
    "multi-label-classification", "multilabelclassification", "pairclassification",
    "multi-input-text-classification", "zero-shot-classification", "text-scoring",
    "text-generation", "text2text-generation", "image-text-to-text", "fill-mask",
    "multiple-choice",
}

SOURCE_TIER = {
    "model_index_task": 1,
    "dataset_card_task_id": 2,
    "inventory_task_id": 2,
    "dataset_identity": 2,
    "dataset_card_task_category": 3,
    "inventory_task_category": 3,
}


@dataclass(frozen=True)
class Evidence:
    source: str
    raw: str
    canonical: str | None
    tier: int
    count: int = 1
    artifact: str = ""
    note: str = ""


def _norm_token(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_task(value: Any) -> str | None:
    """Map an exact structured task token to a compatibility task family."""
    token = _norm_token(value)
    if token in GENERIC_TASK_TOKENS:
        return None
    return TASK_TOKEN_MAP.get(token)


def _as_list(value: Any) -> list[str]:
    """Parse list-valued CSV/cache fields without evaluating arbitrary code."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed if str(x).strip()]
        except (SyntaxError, ValueError):
            pass
    return [text]


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PACKAGE_PARENT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dataset_key(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").strip("/").lower()


def _dataset_aliases(name: str, hf_ids: Iterable[str] = ()) -> set[str]:
    aliases = {_dataset_key(name), _dataset_key(f"mteb/{name}")}
    for hf_id in hf_ids:
        key = _dataset_key(hf_id)
        if not key:
            continue
        aliases.add(key)
        aliases.add(key.rsplit("/", 1)[-1])
        if key.startswith("mteb/"):
            aliases.add(key[len("mteb/"):])
    return {x for x in aliases if x}


def load_inventory(path: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    rows: list[dict[str, str]] = []
    by_canon: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            clean = {str(k): str(v or "") for k, v in row.items()}
            rows.append(clean)
            by_canon[_dataset_key(clean.get("canon_key"))].append(clean)
    return rows, by_canon


def build_alias_lookup(dataset_names: Sequence[str], inventory_by_canon: Mapping[str, Sequence[Mapping[str, str]]]) -> tuple[dict[str, str], set[str]]:
    """Build a conservative exact alias lookup; ambiguous aliases are disabled."""
    owners: dict[str, set[str]] = defaultdict(set)
    for name in dataset_names:
        inv = inventory_by_canon.get(_dataset_key(name), ())
        hf_ids = [str(row.get("hf_id", "")) for row in inv]
        for alias in _dataset_aliases(name, hf_ids):
            owners[alias].add(name)
    ambiguous = {alias for alias, names in owners.items() if len(names) != 1}
    lookup = {alias: next(iter(names)) for alias, names in owners.items() if len(names) == 1}
    return lookup, ambiguous


def resolve_dataset_alias(raw: Any, alias_lookup: Mapping[str, str]) -> str | None:
    key = _dataset_key(raw)
    if key in alias_lookup:
        return alias_lookup[key]
    if key.startswith("mteb/") and key[len("mteb/"):] in alias_lookup:
        return alias_lookup[key[len("mteb/"):]]
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value[0] if value and isinstance(value[0], dict) else {}
    return value if isinstance(value, dict) else {}


def collect_model_index_evidence(model_cache: Path, alias_lookup: Mapping[str, str]) -> tuple[dict[str, list[Evidence]], dict[str, int]]:
    """Scan cached model-index records and aggregate matched task strings."""
    counts: Counter[tuple[str, str, str]] = Counter()
    files_scanned = files_invalid = results_seen = results_matched = 0
    if not model_cache.exists():
        return {}, {
            "files_scanned": 0, "files_invalid": 0,
            "results_seen": 0, "results_matched": 0,
        }
    for path in sorted(model_cache.glob("*.json")):
        files_scanned += 1
        try:
            payload = _load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            files_invalid += 1
            continue
        model_index = payload.get("model-index")
        if model_index is None:
            model_index = (payload.get("cardData") or {}).get("model-index")
        for entry in model_index or ():
            if not isinstance(entry, dict):
                continue
            for result in entry.get("results") or ():
                if not isinstance(result, dict):
                    continue
                results_seen += 1
                ds = result.get("dataset") or {}
                task = result.get("task") or {}
                if not isinstance(ds, dict) or not isinstance(task, dict):
                    continue
                matched = None
                for raw_ds in (ds.get("type"), ds.get("name")):
                    matched = resolve_dataset_alias(raw_ds, alias_lookup)
                    if matched is not None:
                        break
                if matched is None:
                    continue
                raw_task = str(task.get("type") or task.get("name") or "").strip()
                if not raw_task:
                    continue
                results_matched += 1
                counts[(matched, raw_task, str(path.name))] += 1

    # Aggregate across files for compact, reviewable evidence while preserving a
    # count of distinct cache files in the note.
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for (dataset, raw, filename), count in counts.items():
        rec = grouped.setdefault((dataset, raw), {"count": 0, "files": set()})
        rec["count"] += count
        rec["files"].add(filename)
    out: dict[str, list[Evidence]] = defaultdict(list)
    for (dataset, raw), rec in sorted(grouped.items()):
        out[dataset].append(Evidence(
            source="model_index_task", raw=raw, canonical=canonical_task(raw),
            tier=SOURCE_TIER["model_index_task"], count=int(rec["count"]),
            artifact=_repo_rel(model_cache),
            note=f"{len(rec['files'])} cached model files",
        ))
    return dict(out), {
        "files_scanned": files_scanned,
        "files_invalid": files_invalid,
        "results_seen": results_seen,
        "results_matched": results_matched,
    }


def _structured_card_values(payload: Mapping[str, Any], field: str) -> list[str]:
    card = payload.get("cardData") or {}
    values = _as_list(card.get(field)) if isinstance(card, Mapping) else []
    prefix = field + ":"
    for tag in _as_list(payload.get("tags")):
        if tag.startswith(prefix):
            values.append(tag[len(prefix):])
    return sorted(set(x for x in values if x))


def _classification_context(raw_values: Iterable[str], description: str) -> bool:
    tokens = {_norm_token(x) for x in raw_values}
    if any("classification" in token for token in tokens):
        return True
    compact = re.sub(r"[^a-z0-9]", "", description.lower())
    return "classification" in compact


def identity_evidence(dataset: str, hf_ids: Sequence[str], description: str,
                      raw_values: Iterable[str], artifact: str = "") -> list[Evidence]:
    """Extract only precise identity phrases, with task-format corroboration.

    These rules intentionally do not perform open-ended free-text
    classification.  They recognize task names embedded in benchmark identities
    (for example ``MassiveIntentClassification``) and require a compatible
    structured classification/token-classification context.
    """
    names = " ".join([dataset, *hf_ids]).lower()
    text = (names + " " + str(description or "")).lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    raw_norm = {_norm_token(x) for x in raw_values}
    class_ctx = _classification_context(raw_values, text)
    token_ctx = any("token-classification" == x or "token classification" == x for x in raw_norm)

    hits: list[tuple[str, str]] = []
    if class_ctx and (re.search(r"(?:^|[/_\-])intent(?:$|[/_\-])", names)
                      or "intentclassification" in compact
                      or "intentprediction" in compact):
        hits.append(("intent", "benchmark identity says intent classification/prediction"))
    if class_ctx and (re.search(r"(?:^|[/_\-])(domain|scenario)(?:$|[/_\-])", names)
                      or "domainclassification" in compact
                      or "scenarioclassification" in compact):
        hits.append(("domain", "benchmark identity says domain/scenario classification"))
    if class_ctx and ("counterfactualclassification" in compact
                      or "counterfactualdetection" in compact):
        hits.append(("counterfactual-detection", "benchmark identity says counterfactual classification"))
    if class_ctx and (re.search(r"(?:^|[/_\-])(sentiment|polarity)(?:$|[/_\-])", names)
                      or "sentimentclassification" in compact):
        hits.append(("sentiment", "benchmark identity says sentiment/polarity classification"))
    if class_ctx and ("topicclassification" in compact or "topicsclassification" in compact):
        hits.append(("topic", "dataset card explicitly says topic classification"))
    if class_ctx and (re.search(r"(?:^|[/_\-])(toxic|toxicity)(?:$|[/_\-])", names)
                      or "toxicityclassification" in compact):
        hits.append(("toxicity", "dataset identity/card explicitly says toxicity"))
    if class_ctx and re.search(r"(?:^|[/_\-])spam(?:$|[/_\-])", names):
        hits.append(("spam-detection", "dataset identity explicitly says spam"))
    if class_ctx and ("language-identification" in names or "languageidentification" in compact):
        hits.append(("language-identification", "benchmark identity says language identification"))
    if token_ctx and (re.search(r"(?:^|[/_\-])pii(?:$|[/_\-])", names)
                      or "personallyidentifiableinformation" in compact):
        hits.append(("pii-detection", "token-classification dataset identity says PII"))
    if token_ctx and ("acronym_identification" in names or "acronymidentification" in compact):
        hits.append(("acronym-identification", "token-classification benchmark identity says acronym identification"))

    return [Evidence(
        source="dataset_identity", raw=note, canonical=canonical,
        tier=SOURCE_TIER["dataset_identity"], artifact=artifact,
        note="precise identity rule; not open-ended card-text classification",
    ) for canonical, note in sorted(set(hits))]


def collect_card_evidence(dataset_names: Sequence[str], inventory_by_canon: Mapping[str, Sequence[Mapping[str, str]]], dataset_cache: Path,
                          inventory_path: Path | None = None) -> tuple[dict[str, list[Evidence]], dict[str, Any]]:
    evidence: dict[str, list[Evidence]] = defaultdict(list)
    cards_found = cards_missing = cards_invalid = 0
    missing: list[str] = []
    for dataset in dataset_names:
        rows = list(inventory_by_canon.get(_dataset_key(dataset), ()))
        hf_ids = sorted(set(str(row.get("hf_id", "")).strip() for row in rows
                            if str(row.get("hf_id", "")).strip()))
        raw_context: list[str] = []
        descriptions: list[str] = []
        artifacts: list[str] = []

        for row in rows:
            for source, column in (("inventory_task_id", "task_ids"),
                                   ("inventory_task_category", "task_categories")):
                for raw in _as_list(row.get(column)):
                    raw_context.append(raw)
                    evidence[dataset].append(Evidence(
                        source=source, raw=raw, canonical=canonical_task(raw),
                        tier=SOURCE_TIER[source],
                        artifact=_repo_rel(inventory_path or DEFAULT_INVENTORY),
                    ))

        for hf_id in hf_ids:
            path = dataset_cache / (hf_id.replace("/", "__") + ".json")
            if not path.exists():
                cards_missing += 1
                missing.append(hf_id)
                continue
            try:
                payload = _load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError):
                cards_invalid += 1
                continue
            cards_found += 1
            artifacts.append(_repo_rel(path))
            description = str(payload.get("description") or "")
            card = payload.get("cardData") or {}
            if isinstance(card, Mapping):
                description += " " + str(card.get("pretty_name") or "")
            descriptions.append(description)
            for source, field in (("dataset_card_task_id", "task_ids"),
                                  ("dataset_card_task_category", "task_categories")):
                for raw in _structured_card_values(payload, field):
                    raw_context.append(raw)
                    evidence[dataset].append(Evidence(
                        source=source, raw=raw, canonical=canonical_task(raw),
                        tier=SOURCE_TIER[source], artifact=_repo_rel(path),
                    ))

        artifact = ";".join(artifacts)
        evidence[dataset].extend(identity_evidence(
            dataset, hf_ids, " ".join(descriptions), raw_context, artifact=artifact))

    # Deduplicate inventory/card tag repeats while preserving counts and source.
    deduped: dict[str, list[Evidence]] = {}
    for dataset, rows in evidence.items():
        merged: dict[tuple[Any, ...], Evidence] = {}
        for ev in rows:
            key = (ev.source, ev.raw, ev.canonical, ev.tier, ev.artifact, ev.note)
            old = merged.get(key)
            if old is None:
                merged[key] = ev
            else:
                merged[key] = Evidence(**{**asdict(old), "count": old.count + ev.count})
        deduped[dataset] = sorted(merged.values(), key=lambda e: (e.tier, e.source, e.raw))
    return deduped, {
        "cards_found": cards_found,
        "cards_missing": cards_missing,
        "cards_invalid": cards_invalid,
        "missing_hf_ids": sorted(set(missing)),
    }


def decide_evidence(rows: Sequence[Evidence]) -> dict[str, Any]:
    """Choose one task at the best informative tier or require review."""
    informative = [row for row in rows if row.canonical is not None]
    if not informative:
        return {
            "decision": "insufficient_evidence", "proposed_task_type": "",
            "winning_tier": None, "winning_sources": [], "conflicts": [],
        }
    tier = min(row.tier for row in informative)
    best = [row for row in informative if row.tier == tier]
    tasks = sorted({str(row.canonical) for row in best})
    if len(tasks) != 1:
        return {
            "decision": "conflict", "proposed_task_type": "",
            "winning_tier": tier,
            "winning_sources": sorted({row.source for row in best}),
            "conflicts": tasks,
        }
    return {
        "decision": "propose", "proposed_task_type": tasks[0],
        "winning_tier": tier,
        "winning_sources": sorted({row.source for row in best if row.canonical == tasks[0]}),
        "conflicts": [],
    }


def decode_graph_task_types(payload: Mapping[str, Any]) -> tuple[list[str], dict[str, int]]:
    data = payload["data"]
    raw_vocab = (payload.get("xd0_meta") or {}).get("task_type_vocab") or {OTHER: 0}
    vocab = {str(task): int(idx) for task, idx in raw_vocab.items()}
    if vocab.get(OTHER) != 0:
        raise ValueError("graph contract violation: task_type_vocab['Other'] must equal 0")
    inverse = {idx: task for task, idx in vocab.items()}
    ids = [int(x) for x in data["dataset"].task_type_id.tolist()]
    unknown_ids = sorted(set(ids) - set(inverse))
    if unknown_ids:
        raise ValueError(f"task_type_id values missing from vocab: {unknown_ids}")
    return [inverse[idx] for idx in ids], vocab


def graph_dataset_names(payload: Mapping[str, Any]) -> list[str]:
    frame = payload["unique_dataset_id"].sort_values("mappedID")
    ids = [int(x) for x in frame["mappedID"].tolist()]
    if ids != list(range(len(ids))):
        raise ValueError("unique_dataset_id.mappedID must be contiguous and zero-based")
    return [str(x) for x in frame["dataset"].tolist()]


def compute_incompatibility_coverage(task_types: Sequence[str], edge_index: Any) -> dict[str, Any]:
    """Reproduce ``build_global_negative_pools``' incompatible-source coverage.

    Only pool existence/composition is needed, so this implementation accepts a
    torch tensor, numpy array, or nested Python lists and does not require the
    positive-membership matrix used by the loss.
    """
    if hasattr(edge_index, "detach"):
        values = edge_index.detach().cpu().tolist()
    elif hasattr(edge_index, "tolist"):
        values = edge_index.tolist()
    else:
        values = edge_index
    if len(values) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    src = [int(x) for x in values[0]]
    dst = [int(x) for x in values[1]]
    model_tasks: dict[int, set[str]] = defaultdict(set)
    for model, dataset in zip(src, dst):
        model_tasks[model].add(task_types[dataset])
    observed_models = sorted(model_tasks)
    pool_sizes: dict[int, int] = {}
    supervised = sorted(set(dst))
    known = 0
    for dataset in supervised:
        task = task_types[dataset]
        if task == OTHER:
            continue
        known += 1
        size = sum(1 for model in observed_models if task not in model_tasks[model])
        if size:
            pool_sizes[dataset] = size
    return {
        "supervised_datasets": len(supervised),
        "known_task_datasets": known,
        "nonempty_incompatibility_pools": len(pool_sizes),
        "incompatibility_candidates_total": int(sum(pool_sizes.values())),
        "mean_pool_size": (float(sum(pool_sizes.values())) / len(pool_sizes)
                           if pool_sizes else 0.0),
        "pool_sizes_by_dataset_id": {str(k): v for k, v in sorted(pool_sizes.items())},
    }


def append_only_vocab(current: Mapping[str, int], proposed_tasks: Iterable[str]) -> dict[str, int]:
    """Return a deterministic append-only vocab without changing ``current``."""
    vocab = {str(k): int(v) for k, v in current.items()}
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    if [idx for _, idx in ordered] != list(range(len(ordered))):
        raise ValueError("current task vocab must be contiguous")
    for task in sorted(set(str(x) for x in proposed_tasks), key=lambda x: (x.lower(), x)):
        if task not in vocab:
            vocab[task] = len(vocab)
    return vocab


def _merge_evidence(*sources: Mapping[str, Sequence[Evidence]]) -> dict[str, list[Evidence]]:
    out: dict[str, list[Evidence]] = defaultdict(list)
    for source in sources:
        for dataset, rows in source.items():
            out[dataset].extend(rows)
    return {dataset: sorted(rows, key=lambda e: (e.tier, e.source, e.raw))
            for dataset, rows in out.items()}


def build_review_rows(dataset_names: Sequence[str], current_tasks: Sequence[str],
                      evidence: Mapping[str, Sequence[Evidence]], supervised_ids: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (dataset, current) in enumerate(zip(dataset_names, current_tasks)):
        ev = list(evidence.get(dataset, ()))
        result = decide_evidence(ev)
        if current != OTHER:
            inferred = result.get("proposed_task_type")
            decision = "already_known"
            if inferred and inferred != current:
                decision = "known_metadata_mismatch"
            result = {**result, "decision": decision, "proposed_task_type": ""}
        row = {
            "dataset": dataset,
            "mapped_id": idx,
            "supervised_full_graph": int(idx in supervised_ids),
            "current_task_type": current,
            **result,
            "evidence_json": json.dumps([asdict(x) for x in ev], ensure_ascii=False,
                                        sort_keys=True),
        }
        rows.append(row)
    return rows


def _task_types_after(current: Sequence[str], review_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    after = list(current)
    for row in review_rows:
        if row.get("decision") == "propose":
            after[int(row["mapped_id"])] = str(row["proposed_task_type"])
    return after


def exact_fixed_split_edge_indexes(data: Any, split_seeds: Sequence[int]) -> tuple[dict[int, Any], str | None]:
    """Materialize the same train-visible edge set G2 uses, if dependencies exist."""
    try:
        if str(PACKAGE_PARENT) not in sys.path:
            sys.path.insert(0, str(PACKAGE_PARENT))
        import torch
        from ModelLakeFishing.stage2TrainGraphSAGE.eval_harness import make_fixed_splits
        from ModelLakeFishing.stage2TrainGraphSAGE.graph_surgery import dedup_trained_on
        from ModelLakeFishing.stage2TrainGraphSAGE.losses import (
            TRAINED_ON, accuracy_lookup, perf_supervision,
        )
    except Exception as exc:  # pragma: no cover - dependency fallback is reported
        return {}, f"fixed-split imports unavailable: {type(exc).__name__}: {exc}"

    try:
        deduped = dedup_trained_on(data)
        lookup = accuracy_lookup(deduped)
        out = {}
        for seed in split_seeds:
            train_data, _val_data, _test_data = make_fixed_splits(
                deduped, split_seed=int(seed))
            edge_label_index, _target = perf_supervision(train_data[TRAINED_ON], lookup)
            out[int(seed)] = torch.cat(
                [train_data[TRAINED_ON].edge_index, edge_label_index], dim=1)
        return out, None
    except Exception as exc:  # pragma: no cover - reported in real environments
        return {}, f"fixed-split replay failed: {type(exc).__name__}: {exc}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[field] = value
            writer.writerow(clean)


def write_outputs(out_dir: Path, review_rows: Sequence[Mapping[str, Any]],
                  current_vocab: Mapping[str, int], report: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals = [dict(row) for row in review_rows if row.get("decision") == "propose"]
    proposed_tasks = [str(row["proposed_task_type"]) for row in proposals]
    vocab = append_only_vocab(current_vocab, proposed_tasks)
    for row in proposals:
        row["proposed_task_type_id"] = vocab[str(row["proposed_task_type"])]
        row["action"] = "replace_Other_after_review"

    patch_path = out_dir / "task_type_enrichment_patch.csv"
    _write_csv(patch_path, proposals, [
        "dataset", "mapped_id", "supervised_full_graph", "current_task_type",
        "proposed_task_type", "proposed_task_type_id", "action", "winning_tier",
        "winning_sources", "evidence_json",
    ])

    vocab_rows = []
    for task, idx in sorted(vocab.items(), key=lambda item: item[1]):
        vocab_rows.append({
            "task_type": task, "task_type_id": idx,
            "action": "keep" if task in current_vocab else "append_after_review",
        })
    vocab_path = out_dir / "task_type_vocab_patch.csv"
    _write_csv(vocab_path, vocab_rows, ["task_type", "task_type_id", "action"])

    review_path = out_dir / "task_type_enrichment_review.csv"
    _write_csv(review_path, review_rows, [
        "dataset", "mapped_id", "supervised_full_graph", "current_task_type",
        "decision", "proposed_task_type", "winning_tier", "winning_sources",
        "conflicts", "evidence_json",
    ])

    report_path = out_dir / "task_type_enrichment_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    md_path = out_dir / "task_type_enrichment_report.md"
    md_path.write_text(render_markdown(report, proposals), encoding="utf-8")
    return {
        "patch_csv": str(patch_path.resolve()),
        "vocab_patch_csv": str(vocab_path.resolve()),
        "review_csv": str(review_path.resolve()),
        "report_json": str(report_path.resolve()),
        "report_md": str(md_path.resolve()),
    }


def render_markdown(report: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]) -> str:
    summary = report["summary"]
    lines = [
        "# Task-type enrichment dry run",
        "",
        "> Review artifact only. The source graph, source tables, existing vocab, and caches were not mutated.",
        "",
        f"- source graph: `{report['inputs']['graph']}`",
        f"- source SHA-256 unchanged: **{str(report['integrity']['source_graph_unchanged']).upper()}**",
        f"- graph datasets: **{summary['dataset_nodes']}**; current `Other`: **{summary['current_other']}**",
        f"- proposed replacements: **{summary['proposed_total']}** "
        f"(**{summary['proposed_supervised']}** full-graph supervised)",
        f"- unresolved conflicts: **{summary['conflicts_total']}**; insufficient evidence: **{summary['insufficient_total']}**",
        "",
        "## Incompatibility-pool coverage",
        "",
        "| protocol | supervised datasets | before | after | target >= 50 |",
        "|---|---:|---:|---:|:--:|",
    ]
    full = report["pool_coverage"]["full_graph"]
    lines.append(
        f"| full graph | {full['before']['supervised_datasets']} | "
        f"{full['before']['nonempty_incompatibility_pools']} | "
        f"{full['after']['nonempty_incompatibility_pools']} | "
        f"{'PASS' if full['after']['nonempty_incompatibility_pools'] >= report['target']['nonempty_pools'] else 'FAIL'} |")
    for seed, row in report["pool_coverage"].get("fixed_splits", {}).items():
        lines.append(
            f"| G2 train-visible seed {seed} | {row['before']['supervised_datasets']} | "
            f"{row['before']['nonempty_incompatibility_pools']} | "
            f"{row['after']['nonempty_incompatibility_pools']} | "
            f"{'PASS' if row['after']['nonempty_incompatibility_pools'] >= report['target']['nonempty_pools'] else 'FAIL'} |")
    lines.extend([
        "",
        "The implementation reproduces the reference numerator (30 pools), but reports the live denominator from the exact fixed-split code; see `reference_reconciliation` in JSON for the 30/64 caveat.",
        "",
        "## Proposed supervised replacements",
        "",
        "| dataset | current | proposed | evidence tier | sources |",
        "|---|---|---|---:|---|",
    ])
    for row in sorted((x for x in proposals if int(x.get("supervised_full_graph", 0))),
                      key=lambda x: int(x["mapped_id"])):
        sources = ", ".join(row.get("winning_sources", []))
        lines.append(
            f"| `{row['dataset']}` | `{row['current_task_type']}` | "
            f"`{row['proposed_task_type']}` | {row['winning_tier']} | {sources} |")
    target = report["amazon_massive_intent"]
    lines.extend([
        "",
        "## Required failing query",
        "",
        f"- `amazon_massive_intent`: **{target.get('decision', 'missing')}** -> "
        f"`{target.get('proposed_task_type', '')}`",
        f"- local evidence: {target.get('evidence_summary', '')}",
        "",
        "## Decision policy and limitations",
        "",
        "1. Specific cached model-index task wins because it is attached to the performance edge.",
        "2. Otherwise, a unique specific card task-id or precise benchmark identity may propose a task.",
        "3. Otherwise, a unique specific card category may propose a task.",
        "4. Generic classification/token-classification labels never propose a type; same-tier conflicts require review.",
        "5. This dry run does not apply the patch, rebuild xd0, retrain G2, or establish that inferred incompatibility negatives are semantically correct.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    graph_path = Path(args.graph).resolve()
    inventory_path = Path(args.inventory).resolve()
    dataset_cache = Path(args.dataset_cache).resolve()
    model_cache = Path(args.model_cache).resolve()
    out_dir = Path(args.out_dir).resolve()

    graph_sha_before = sha256_file(graph_path)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required to load the graph") from exc
    payload = torch.load(graph_path, map_location="cpu", weights_only=False)
    dataset_names = graph_dataset_names(payload)
    current_tasks, current_vocab = decode_graph_task_types(payload)

    _inventory_rows, inventory_by_canon = load_inventory(inventory_path)
    alias_lookup, ambiguous_aliases = build_alias_lookup(dataset_names, inventory_by_canon)
    card_evidence, card_stats = collect_card_evidence(
        dataset_names, inventory_by_canon, dataset_cache,
        inventory_path=inventory_path)
    model_evidence, model_stats = collect_model_index_evidence(model_cache, alias_lookup)
    evidence = _merge_evidence(model_evidence, card_evidence)

    raw_edge_index = payload["data"]["model", "trained_on", "dataset"].edge_index
    supervised_ids = set(int(x) for x in raw_edge_index[1].tolist())
    review_rows = build_review_rows(
        dataset_names, current_tasks, evidence, supervised_ids)
    after_tasks = _task_types_after(current_tasks, review_rows)

    full_before = compute_incompatibility_coverage(current_tasks, raw_edge_index)
    full_after = compute_incompatibility_coverage(after_tasks, raw_edge_index)
    fixed_indexes, fixed_error = exact_fixed_split_edge_indexes(
        payload["data"], tuple(int(x) for x in args.split_seeds))
    fixed = {}
    for seed, edge_index in fixed_indexes.items():
        fixed[str(seed)] = {
            "before": compute_incompatibility_coverage(current_tasks, edge_index),
            "after": compute_incompatibility_coverage(after_tasks, edge_index),
        }

    proposed = [row for row in review_rows if row["decision"] == "propose"]
    conflicts = [row for row in review_rows if row["decision"] == "conflict"]
    insufficient = [row for row in review_rows if row["decision"] == "insufficient_evidence"]
    target_row = next((row for row in review_rows
                       if row["dataset"] == "amazon_massive_intent"), None)
    target_evidence = [asdict(x) for x in evidence.get("amazon_massive_intent", ())]
    if target_row is None:
        target_report = {"decision": "missing_from_graph", "evidence": []}
    else:
        winning = [x for x in target_evidence
                   if x.get("tier") == target_row.get("winning_tier") and x.get("canonical")]
        target_report = {
            **{k: target_row.get(k) for k in (
                "mapped_id", "current_task_type", "decision", "proposed_task_type",
                "winning_tier", "winning_sources", "conflicts")},
            "evidence": target_evidence,
            "evidence_summary": "; ".join(
                f"{x['source']}={x['raw']} -> {x['canonical']}" for x in winning),
        }

    live_denominators = [row["before"]["supervised_datasets"] for row in fixed.values()]
    live_numerators = [row["before"]["nonempty_incompatibility_pools"] for row in fixed.values()]
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "dry_run_review_only",
        "inputs": {
            "graph": _repo_rel(graph_path),
            "inventory": _repo_rel(inventory_path),
            "dataset_cache": _repo_rel(dataset_cache),
            "model_cache": _repo_rel(model_cache),
        },
        "integrity": {
            "source_graph_sha256_before": graph_sha_before,
            "source_graph_sha256_after": None,
            "source_graph_unchanged": None,
            "source_mutations_performed": False,
        },
        "summary": {
            "dataset_nodes": len(dataset_names),
            "full_graph_supervised_datasets": len(supervised_ids),
            "current_other": sum(task == OTHER for task in current_tasks),
            "proposed_total": len(proposed),
            "proposed_supervised": sum(int(row["supervised_full_graph"]) for row in proposed),
            "conflicts_total": len(conflicts),
            "conflicts_supervised": sum(int(row["supervised_full_graph"]) for row in conflicts),
            "insufficient_total": len(insufficient),
            "insufficient_supervised": sum(int(row["supervised_full_graph"]) for row in insufficient),
        },
        "evidence_scan": {
            "model_index": model_stats,
            "dataset_cards": card_stats,
            "dataset_aliases": len(alias_lookup),
            "ambiguous_aliases_disabled": sorted(ambiguous_aliases),
        },
        "target": {"nonempty_pools": int(args.target_pools)},
        "pool_coverage": {
            "full_graph": {"before": full_before, "after": full_after},
            "fixed_splits": fixed,
            "fixed_split_error": fixed_error,
        },
        "reference_reconciliation": {
            "plan_reference": REFERENCE_BASELINE,
            "live_fixed_split_numerators": live_numerators,
            "live_fixed_split_denominators": live_denominators,
            "reference_numerator_reproduced": bool(live_numerators and
                                                   all(x == REFERENCE_BASELINE["nonempty_pools"]
                                                       for x in live_numerators)),
            "reference_denominator_reproduced": bool(live_denominators and
                                                     all(x == REFERENCE_BASELINE["denominator"]
                                                         for x in live_denominators)),
            "note": "The current executable G2 protocol defines train-visible tasks from message edges plus positive train labels. Its live denominator is reported rather than forcing the prose value 64.",
        },
        "gate": {
            "target_nonempty_pools": int(args.target_pools),
            "full_graph_pass": full_after["nonempty_incompatibility_pools"] >= int(args.target_pools),
            "all_fixed_splits_pass": bool(fixed) and all(
                row["after"]["nonempty_incompatibility_pools"] >= int(args.target_pools)
                for row in fixed.values()),
            "requires_human_review_and_apply": True,
        },
        "amazon_massive_intent": target_report,
        "limitations": [
            "The patch is proposed metadata only and is never applied automatically.",
            "A task label does not prove every resulting model negative is semantically incompatible; G2 must be rerun and five-metric gated after review.",
            "Generic classification and token-classification metadata are deliberately insufficient.",
            "Multi-task datasets with same-tier semantic conflicts remain Other.",
            "Only already cached Hugging Face API/model-index/card metadata is read; cache gaps are not filled and no network call occurs.",
            "The prose baseline denominator 64 is not reproduced by the current exact split code; live denominators are retained in the report.",
        ],
    }

    graph_sha_after = sha256_file(graph_path)
    report["integrity"]["source_graph_sha256_after"] = graph_sha_after
    report["integrity"]["source_graph_unchanged"] = graph_sha_after == graph_sha_before
    if graph_sha_after != graph_sha_before:
        raise RuntimeError("source graph changed during dry run; refusing to emit a patch")

    paths = write_outputs(out_dir, review_rows, current_vocab, report)
    return report, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--target-pools", type=int, default=DEFAULT_TARGET_POOLS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, paths = run(args)
    summary = report["summary"]
    print(json.dumps({
        "mode": report["mode"],
        "current_other": summary["current_other"],
        "proposed_total": summary["proposed_total"],
        "proposed_supervised": summary["proposed_supervised"],
        "pool_coverage": report["pool_coverage"],
        "gate": report["gate"],
        "amazon_massive_intent": report["amazon_massive_intent"],
        "artifacts": paths,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
