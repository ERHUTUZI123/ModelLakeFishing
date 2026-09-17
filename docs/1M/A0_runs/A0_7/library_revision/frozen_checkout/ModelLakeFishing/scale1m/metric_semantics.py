"""
metric_semantics.py -- RF: which way is "better", per metric.

WHY THIS EXISTS (D-61)
    Min-max normalisation aligns scale, not direction. Normalising `wer` or
    `loss` inside its group and taking the argmax hands gold to the WORST
    model, and nothing raises: the run completes, the number is plausible, and
    the benchmark is quietly wrong. So direction is decided before any
    normalisation, from an explicit table.

FOUR CLASSES (D-61)
    higher   use as-is
    lower    flip after in-group normalisation (1 - v)
    reward   RL-style returns: unbounded, environment-specific semantics.
             Kept as edges, excluded from gold (D-59).
    unknown  anything the table does not recognise. Kept as edges, excluded
             from gold. NEVER guessed.

WHY PATTERNS AND NOT A LIST
    The snapshot carries 2,776 distinct raw metric names over 2.17M records,
    and the head is one family repeated at many cut-offs (`ndcg_at_1` ...
    `ndcg_at_1000`) under several similarity functions (`cosine_ndcg@10`,
    `euclidean_spearman`). Enumerating them invites silent misses, so the
    table is keyed on the BASE name after a normalisation that strips the
    cut-off and the similarity prefix. Everything unmatched falls to `unknown`,
    which is the safe direction.

Run (from ModelLakeFishing/):
    python -m scale1m.metric_semantics --report   # class coverage on the snapshot
"""
import re

HIGHER = {
    # classification / generation quality
    "accuracy", "acc", "acc_norm", "accuracy_norm", "f1", "f1_macro", "f1_micro",
    "f1_weighted", "precision", "recall", "exact_match", "em", "matthews_correlation",
    "mcc", "kappa", "cohen_kappa", "balanced_accuracy", "top_k_accuracy",
    "auc", "roc_auc", "pr_auc", "average_precision", "ap", "map", "mrr", "ndcg",
    "dcg", "hit_rate", "hits", "success_rate", "pass", "pass_rate",
    # text generation
    "bleu", "sacrebleu", "chrf", "chrf++", "rouge", "rouge1", "rouge2", "rougel",
    "rougelsum", "meteor", "bertscore", "bleurt", "comet", "ter_score", "gleu",
    "sari", "cider", "spice",
    # similarity / clustering / retrieval side scores
    "pearson", "spearman", "kendall", "cosine_similarity", "similarity",
    "v_measure", "ari", "nmi", "silhouette", "purity",
    # MTEB convention: the task's headline number, always higher-is-better
    "main_score",
    # segmentation / detection
    "iou", "miou", "dice", "psnr", "ssim", "map50", "map75",
}

LOWER = {
    "wer", "cer", "per", "ter", "mer", "wil", "edit_distance", "levenshtein",
    "loss", "eval_loss", "train_loss", "validation_loss", "perplexity", "ppl",
    "nll", "cross_entropy", "mae", "mse", "rmse", "mape", "smape", "msle",
    "error", "error_rate", "err", "eer", "fid", "kid", "lpips", "brier",
    "word_error_rate", "character_error_rate", "char_error_rate",
}

REWARD = {"mean_reward", "reward", "episode_reward", "episodic_reward",
          "return", "mean_return", "avg_reward", "average_reward", "score_reward"}

# Similarity-function prefixes MTEB puts in front of an otherwise normal metric.
SIM_PREFIXES = ("cosine_", "cos_sim_", "euclidean_", "manhattan_", "dot_",
                "max_", "l2_", "ip_")
# Cut-off suffix: `_at_10`, `_at_1000`.
AT_K = re.compile(r"_at_\d+$")
# MTEB ranking-difficulty diagnostics; not a quality score in either direction.
DIAGNOSTIC_PREFIXES = ("nauc_", "naucs_")


def normalize_name(raw) -> str:
    """`cosine_ndcg@10` -> `cosine_ndcg_at_10`; `Pass@1` -> `pass_at_1`.

    Case, separators and the `@k` spelling only. No semantic merging: the
    grouping key keeps the full normalised name, so `ndcg_at_1` and
    `ndcg_at_10` stay different groups (1Mplan 3.2).
    """
    s = str(raw).strip().lower()
    s = s.replace("@", "_at_")
    s = re.sub(r"[\s\-./]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def base_name(norm: str) -> str:
    """Strip the cut-off and the similarity prefix to reach the family name."""
    b = AT_K.sub("", norm)
    for p in SIM_PREFIXES:
        if b.startswith(p) and len(b) > len(p):
            b = b[len(p):]
            break
    return b


def classify(raw):
    """-> (normalised name, base name, one of higher/lower/reward/unknown)."""
    norm = normalize_name(raw)
    if not norm:
        return norm, norm, "unknown"
    if norm.startswith(DIAGNOSTIC_PREFIXES):
        return norm, norm, "unknown"
    b = base_name(norm)
    if b in REWARD or norm in REWARD:
        return norm, b, "reward"
    if b in HIGHER:
        return norm, b, "higher"
    if b in LOWER:
        return norm, b, "lower"
    # `..._std`, `..._stderr` are dispersions of a score, not a score
    if b.endswith(("_std", "_stderr", "_var", "_stddev")):
        return norm, b, "unknown"
    return norm, b, "unknown"


def in_gold(direction: str) -> bool:
    """Only direction-known metrics may decide a gold model (D-61)."""
    return direction in ("higher", "lower")


def orient(value_norm: float, direction: str) -> float:
    """Apply direction AFTER in-group min-max normalisation."""
    return 1.0 - value_norm if direction == "lower" else value_norm


def _report(argv=None):
    """Class coverage over the frozen snapshot -- the number F2 must publish."""
    import collections
    import json
    import os
    from scale1m.verify_raw import iter_records

    from scale1m.paths import data_root
    c = os.path.join(data_root(), "data1m", "candidates_full")
    prov = json.load(open(os.path.join(c, "PROVENANCE.json"), encoding="utf-8"))
    by_class = collections.Counter()
    names_by_class = collections.defaultdict(collections.Counter)
    for _, r in iter_records(c, prov["shards"]):
        mi = (r.get("cardData") or {}).get("model-index")
        if not mi:
            continue
        for e in (mi if isinstance(mi, list) else [mi]):
            if not isinstance(e, dict):
                continue
            for m in (e.get("metrics") or []):
                if not isinstance(m, dict) or m.get("type") is None:
                    continue
                norm, _b, cls = classify(m["type"])
                by_class[cls] += 1
                names_by_class[cls][norm] += 1
    tot = sum(by_class.values())
    out = {"records": tot,
           "by_class": {k: {"n": v, "pct": round(100.0 * v / max(tot, 1), 2)}
                        for k, v in by_class.most_common()},
           "distinct_names_by_class": {k: len(v) for k, v in names_by_class.items()},
           "top_unknown": names_by_class["unknown"].most_common(30)}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_report() if "--report" in sys.argv else print(__doc__))
