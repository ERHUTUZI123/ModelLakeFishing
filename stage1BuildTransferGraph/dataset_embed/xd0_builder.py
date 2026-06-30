"""
xd0_builder -- build dataset node features x_d^(0), the dataset-side analogue of
xm0_builder (see dataset_embedding_redesign.md, Tier 0 / Steps 1-5).

The OLD dataset feature was a single gpt-neo-125m domain centroid (one mean vector,
labels discarded) -> dataset-similarity effective rank ~3.7, which pins z_d (and
through the graph z_m) to a low-dimensional "domain cone". xd0 makes the dataset node
SYMMETRIC to the model node:

  x_d^(0) = [ e_domain || e_label || e_card || e_stats ]      (frozen semantic views)
            (+) learnable[ task_type_id , n_class_bucket_id , arity_id ]   (discrete)

Row i of every output array corresponds to the dataset whose mappedID == i
(the same row-order contract get_model_names enforces for models).

Frozen views (this file computes them):
  e_domain : multi-statistic domain embedding -- [mean || std] of the per-example
             gpt-neo features (REUSES the existing *_feature.npy, no re-probing),
             each statistic L2-normalized. (Step 1, cheap tier: moments only; a
             stronger encoder / multi-probe is a later upgrade.)
  e_label  : sentence-encoding of the dataset's class-label set -- the task axis the
             old centroid threw away (Step 2).
  e_card   : sentence-encoding of the HF dataset card text (Step 3), cached; empty
             vector on fetch failure (offline-safe).
  e_stats  : small standardized vector of cloud-geometry statistics (Step 4).

Learnable indices (this file outputs only the DISCRETE columns + vocabs; the
embedding tables live in a future DatasetNodeEncoder, exactly as e_size/e_fam do
for models -- NOT materialized here, so they cannot be silently frozen):
  task_type_id      : sentiment/NLI/paraphrase/topic/emotion/... append-only vocab,
                      Other == 0  (Step 5).
  n_class_bucket_id : bucketed number of classes (fixed constants).
  arity_id          : single / pair / multi text input.
"""

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .utils.embedder import determine_file_name_embedded_dataset

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_ENCODER = "all-MiniLM-L6-v2"     # same family as the model-side e_desc
TASK_TYPE_OTHER = "Other"
TASK_TYPE_ID_OTHER = 0

# n_class buckets (fixed, like the size buckets). Full spec: regression is its own
# bucket, distinct from unknown.
#   0: unknown   1: binary   2: three-way   3: 4-6   4: 7-20   5: >20   6: regression
N_CLASS_BUCKETS = 7
N_CLASS_REGRESSION = 6

# arity vocab (full spec): id 0 == unknown (default), 1 single, 2 pair, 3 multi-field,
# 4 span/QA. Append-only; the model sizes its arity table from num_arities in xd0_meta.
ARITY_VOCAB = {"unknown": 0, "single": 1, "pair": 2, "multi": 3, "span_or_qa": 4}
NUM_ARITIES = len(ARITY_VOCAB)

# ── curated dataset knowledge (the ~24-31 datasets in play) ───────────────────
# Names are the dataset-node names used in records.csv / model_config_dataset.csv.
# task_type drives the learnable task_type_id; labels drive e_label + n_class_bucket;
# pair-tasks drive arity. Anything not listed falls back to model_config-observed
# labels (passed in) and a name/label heuristic -> Other, exactly like the model
# family fallback.
_CANON = {
    # name: (task_type, [labels], arity)
    "ag_news":            ("topic",        ["world", "sports", "business", "science technology"], "single"),
    "dbpedia_14":         ("topic",        ["company", "school", "artist", "athlete", "politics", "transportation", "building", "nature", "village", "animal", "plant", "album", "film", "written work"], "single"),
    "trec":               ("question-type",["abbreviation", "entity", "description", "human", "location", "numeric"], "single"),
    "common_language":    ("language-id",  ["language identification across many spoken languages"], "single"),
    "dair-ai/emotion":    ("emotion",      ["sadness", "joy", "love", "anger", "fear", "surprise"], "single"),
    "tweet_eval/emotion": ("emotion",      ["anger", "joy", "optimism", "sadness"], "single"),
    "tweet_eval/sentiment":("sentiment",   ["negative", "neutral", "positive"], "single"),
    "tweet_eval/hate":    ("toxicity",     ["non-hate", "hate"], "single"),
    "tweet_eval/offensive":("toxicity",    ["non-offensive", "offensive"], "single"),
    "tweet_eval/irony":   ("irony",        ["non-irony", "irony"], "single"),
    "tweet_eval/emoji":   ("emoji",        ["emoji prediction over 20 emoji classes"], "single"),
    "hate_speech":        ("toxicity",     ["hate speech", "offensive language", "neither"], "single"),
    "imdb":               ("sentiment",    ["negative", "positive"], "single"),
    "rotten_tomatoes":    ("sentiment",    ["negative", "positive"], "single"),
    "glue/sst2":          ("sentiment",    ["negative", "positive"], "single"),
    "glue/cola":          ("acceptability",["unacceptable", "acceptable"], "single"),
    "glue/qqp":           ("paraphrase",   ["not duplicate", "duplicate"], "pair"),
    "glue/mnli":          ("NLI",          ["entailment", "neutral", "contradiction"], "pair"),
    "glue/qnli":          ("NLI",          ["entailment", "not entailment"], "pair"),
    "glue/rte":           ("NLI",          ["entailment", "not entailment"], "pair"),
    "glue/wnli":          ("NLI",          ["not entailment", "entailment"], "pair"),
    "xnli_en":            ("NLI",          ["entailment", "neutral", "contradiction"], "pair"),
    "nli_tr/snli_tr":     ("NLI",          ["entailment", "neutral", "contradiction"], "pair"),
    "nli_tr/multinli_tr": ("NLI",          ["entailment", "neutral", "contradiction"], "pair"),
    "mteb_amazon_reviews_multi_en": ("sentiment", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], "single"),
    "amazon_reviews_multi/en": ("sentiment", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], "single"),
    "twitter-financial-news-sentiment": ("sentiment", ["bearish", "bullish", "neutral"], "single"),
    "bertweet-covid19-cased-tweet-data": ("topic", ["covid-19 related tweet topic"], "single"),
}

# HF dataset repo ids for the dataset-card text (e_card). Within-group subsets
# (glue/*, tweet_eval/*) share one card; e_label + task_type carry the distinction.
_CARD_REPO = {
    "ag_news": "fancyzhx/ag_news", "dbpedia_14": "fancyzhx/dbpedia_14",
    "trec": "CogComp/trec", "common_language": "common_language",
    "dair-ai/emotion": "dair-ai/emotion", "imdb": "stanfordnlp/imdb",
    "rotten_tomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "hate_speech": "ucberkeley-dlab/measuring-hate-speech",
    "xnli_en": "facebook/xnli", "nli_tr/snli_tr": "nli_tr", "nli_tr/multinli_tr": "nli_tr",
    "mteb_amazon_reviews_multi_en": "mteb/amazon_reviews_multi",
    "amazon_reviews_multi/en": "mteb/amazon_reviews_multi",
    "twitter-financial-news-sentiment": "zeroshot/twitter-financial-news-sentiment",
}
def _card_repo(name: str) -> str:
    if name in _CARD_REPO:
        return _CARD_REPO[name]
    if name.startswith("glue/"):
        return "nyu-mll/glue"
    if name.startswith("tweet_eval/"):
        return "cardiffnlp/tweet_eval"
    return name.split("/")[0]            # best-effort fallback


# ── row-order contract ────────────────────────────────────────────────────────
def get_dataset_names(unique_dataset_id: pd.DataFrame) -> List[str]:
    ids = unique_dataset_id["mappedID"]
    assert ids.min() == 0 and ids.max() == len(ids) - 1 and ids.nunique() == len(ids), (
        "mappedID must be a 0-based consecutive range (call before any id shift)"
    )
    return unique_dataset_id.sort_values("mappedID")["dataset"].tolist()


# ── label resolution (curated -> observed non-generic -> heuristic) ───────────
def _is_generic(labels) -> bool:
    if not labels:
        return True
    return all(str(l).upper().startswith("LABEL_") or str(l).strip() in ("", "0", "1") for l in labels)


def resolve_labels(name: str, observed: dict | None) -> list[str]:
    if name in _CANON:
        return _CANON[name][1]
    if observed and name in observed and not _is_generic(observed[name]):
        return list(observed[name])
    return []                            # unknown -> empty (n_class bucket 0)


def resolve_task_type(name: str, labels: list[str]) -> str:
    if name in _CANON:
        return _CANON[name][0]
    n = name.lower()
    text = (n + " " + " ".join(labels)).lower()
    for key, tt in (("nli", "NLI"), ("entail", "NLI"), ("paraphrase", "paraphrase"),
                    ("duplicate", "paraphrase"), ("sentiment", "sentiment"),
                    ("emotion", "emotion"), ("hate", "toxicity"), ("offens", "toxicity"),
                    ("topic", "topic"), ("news", "topic"), ("irony", "irony")):
        if key in text:
            return tt
    return TASK_TYPE_OTHER


_QA_HINT = ("squad", "qa", "question-answer", "reading", "span", "boolq", "trivia",
            "naturalquestions", "drop", "race", "multiple-choice", "multiple_choice")


def resolve_arity(name: str, task_type: str) -> str:
    if name in _CANON:
        a = _CANON[name][2]
        return a if a in ARITY_VOCAB else "single"
    low = name.lower()
    if any(h in low for h in _QA_HINT) or task_type in ("QA", "question-answering"):
        return "span_or_qa"
    return "pair" if task_type in ("NLI", "paraphrase") else "single"


def n_class_bucket(n: int) -> int:
    """Class-count bucket (regression is assigned separately in build_xd0)."""
    if n <= 0:
        return 0          # unknown
    if n == 2:
        return 1          # binary
    if n == 3:
        return 2          # three-way
    if n <= 6:
        return 3
    if n <= 20:
        return 4
    return 5              # >20


# ── append-only task_type vocab (mirrors load_or_update_family_vocab) ──────────
def load_or_update_task_type_vocab(task_types: List[str], vocab_path: Path | str | None) -> dict:
    vocab = {TASK_TYPE_OTHER: TASK_TYPE_ID_OTHER}
    if vocab_path is not None:
        vocab_path = Path(vocab_path)
        if vocab_path.exists():
            prior = pd.read_csv(vocab_path)
            for tt, tid in zip(prior["task_type"], prior["task_type_id"]):
                tt, tid = str(tt), int(tid)
                if tt in vocab:
                    assert vocab[tt] == tid, f"task_type vocab conflict for {tt!r}"
                else:
                    assert tid == len(vocab), f"task_type vocab not contiguous at {tt!r}"
                    vocab[tt] = tid
    for tt in sorted(set(task_types)):           # append-only, deterministic
        if tt not in vocab:
            vocab[tt] = len(vocab)
    if vocab_path is not None:
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"task_type": list(vocab.keys()),
                      "task_type_id": list(vocab.values())}).to_csv(vocab_path, index=False)
    return vocab


# ── e_domain (moments) + e_stats from the existing per-example .npy cloud ──────
def _load_cloud(emb_dir: str, name: str) -> np.ndarray | None:
    path = determine_file_name_embedded_dataset(emb_dir, name)
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
        return arr if arr.ndim == 2 and arr.shape[0] > 0 else None
    except Exception:
        return None


def _domain_views(cloud: np.ndarray, dim: int) -> np.ndarray:
    """Full-spec domain view: [mean || std || q10 || q50 || q90], each L2-normalized
    -> 5*dim. The mean is the dataset centre; std + quantiles describe spread and the
    shape of the example-embedding cloud. Zeros if no cloud."""
    if cloud is None:
        return np.zeros(5 * dim, dtype=np.float32)
    q10, q50, q90 = np.percentile(cloud, [10, 50, 90], axis=0)
    parts = [cloud.mean(0), cloud.std(0), q10, q50, q90]

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    return np.concatenate([unit(p) for p in parts]).astype(np.float32)


# 10-feature e_stats (full spec). Label/length-dependent dims come from the xd0 aux
# sidecar (per-example label + word length, aligned to the cloud rows); cloud-only
# dims fall back gracefully when aux is absent.
N_STATS = 10


def _linear_probe_acc(cloud: np.ndarray, y: np.ndarray) -> float:
    """Held-out logistic-probe accuracy of the frozen cloud features against the
    dataset's own labels -- a separability/difficulty signal."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception:
        return 0.0
    m = y >= 0
    X, yy = cloud[m], y[m].astype(int)
    if X.shape[0] < 20 or len(np.unique(yy)) < 2:
        return 0.0
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.3, random_state=0, stratify=yy)
        # standardize -> stabilizes lbfgs on the 768-d frozen cloud (avoids non-convergence)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit((Xtr - mu) / sd, ytr)
        return float(clf.score((Xte - mu) / sd, yte))
    except Exception:
        return 0.0


def _fisher_ratio(cloud: np.ndarray, y: np.ndarray) -> float:
    """trace(between-class scatter) / trace(within-class scatter) on the cloud."""
    m = y >= 0
    X, yy = cloud[m], y[m].astype(int)
    classes = np.unique(yy)
    if X.shape[0] < 4 or classes.size < 2:
        return 0.0
    mu = X.mean(0)
    sb = sw = 0.0
    for c in classes:
        Xc = X[yy == c]
        muc = Xc.mean(0)
        sb += Xc.shape[0] * float(((muc - mu) ** 2).sum())
        sw += float(((Xc - muc) ** 2).sum())
    return float(sb / sw) if sw > 1e-9 else 0.0


def _dataset_stats(cloud: np.ndarray, aux: dict | None, n_classes: int) -> np.ndarray:
    """10 structural/difficulty features (raw; z-scored across datasets later):
    [log(n_train), K, H(Y), r_max, mu_len, p90_len, vocab_richness, num_text_fields,
     linear_probe_acc, fisher_ratio]."""
    if aux is None:
        # cloud-only fallback: only size + class count are knowable
        n = float(cloud.shape[0]) if cloud is not None else 0.0
        return np.array([np.log10(n + 1.0), float(n_classes), 0, 0, 0, 0, 0,
                         0, 0, 0], dtype=np.float32)
    y = aux.get("label_ids")
    lens = aux.get("word_lens")
    n_train = float(aux.get("n_train", cloud.shape[0] if cloud is not None else 0))
    # class distribution
    if y is not None and aux.get("label_kind") == "class":
        yv = y[y >= 0].astype(int)
        if yv.size:
            _, cnt = np.unique(yv, return_counts=True)
            p = cnt / cnt.sum()
            H = float(-(p * np.log(p + 1e-12)).sum())
            r_max = float(p.max())
            K = float(len(cnt))
        else:
            H, r_max, K = 0.0, 0.0, float(n_classes)
    else:
        H, r_max, K = 0.0, 0.0, float(n_classes)
    mu_len = float(np.mean(lens)) if lens is not None and lens.size else 0.0
    p90_len = float(np.percentile(lens, 90)) if lens is not None and lens.size else 0.0
    vocab_richness = float(aux.get("vocab_richness", 0.0))
    nf = aux.get("n_fields")
    num_fields = float(np.median(nf)) if nf is not None and nf.size else 1.0
    probe = _linear_probe_acc(cloud, y) if (cloud is not None and y is not None
                                            and aux.get("label_kind") == "class") else 0.0
    fisher = _fisher_ratio(cloud, y) if (cloud is not None and y is not None
                                         and aux.get("label_kind") == "class") else 0.0
    return np.array([np.log10(n_train + 1.0), K, H, r_max, mu_len, p90_len,
                     vocab_richness, num_fields, probe, fisher], dtype=np.float32)


def _load_aux(emb_dir: str, name: str) -> dict | None:
    stem = determine_file_name_embedded_dataset(emb_dir, name)[:-len("_feature.npy")]
    p = stem + "_aux.npz"
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p, allow_pickle=True)
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    except Exception:
        return None


# ── sentence-encoding (e_label, e_card) ───────────────────────────────────────
def _encode(texts: List[str], encoder_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(encoder_name)
    return enc.encode([t if t and t.strip() else "[none]" for t in texts],
                      convert_to_numpy=True, normalize_embeddings=True,
                      show_progress_bar=False).astype(np.float32)


def _fetch_cards(names: List[str], cache_path: Path | str | None,
                 max_chars: int, request_delay: float = 0.3) -> List[str]:
    """HF dataset-card text per dataset (mappedID order), cached; '' on failure."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["dataset", "card"])
    else:
        cache_df = pd.DataFrame(columns=["dataset", "card"])
    have = dict(zip(cache_df["dataset"].astype(str), cache_df["card"].fillna("").astype(str)))
    todo = [n for n in names if n not in have]
    if todo:
        from huggingface_hub import DatasetCard
        new = []
        for i, name in enumerate(todo):
            text = ""
            try:
                text = (DatasetCard.load(_card_repo(name), repo_type="dataset").text or "")[:max_chars]
            except Exception:
                text = ""
            have[name] = text
            new.append({"dataset": name, "card": text})
            print(f"  [card {i+1}/{len(todo)}] {name} <- {_card_repo(name)} [{len(text)} chars]")
            time.sleep(request_delay)
        if cache_path is not None:
            pd.concat([cache_df, pd.DataFrame(new)], ignore_index=True).to_csv(cache_path, index=False)
    return [have.get(n, "") for n in names]


# ── full assembly ─────────────────────────────────────────────────────────────
def build_xd0(
    unique_dataset_id: pd.DataFrame,
    emb_dir: str,
    *,
    observed_labels: dict | None = None,
    card_cache_path: Path | str | None = None,
    task_type_vocab_path: Path | str | None = None,
    encoder_name: str = DEFAULT_ENCODER,
    include_card: bool = True,
    max_card_chars: int = 2000,
    domain_dim: int = 768,
) -> dict:
    names = get_dataset_names(unique_dataset_id)
    N = len(names)

    # discrete + labels (needed before stats so K / regression feed e_stats)
    labels_per = [resolve_labels(n, observed_labels) for n in names]
    task_types = [resolve_task_type(n, lbl) for n, lbl in zip(names, labels_per)]
    arities = [resolve_arity(n, tt) for n, tt in zip(names, task_types)]
    vocab = load_or_update_task_type_vocab(task_types, task_type_vocab_path)
    task_type_id = np.array([vocab.get(tt, TASK_TYPE_ID_OTHER) for tt in task_types], dtype=np.int64)
    arity_id = np.array([ARITY_VOCAB.get(a, 0) for a in arities], dtype=np.int64)

    # frozen view 1 (domain: mean||std||q10||q50||q90) + view 4 (10 structural stats),
    # the latter using the aux sidecar (per-example label/length) when present.
    e_domain = np.zeros((N, 5 * domain_dim), dtype=np.float32)
    stats_raw = np.zeros((N, N_STATS), dtype=np.float32)
    n_class_bucket_id = np.zeros(N, dtype=np.int64)
    for i, name in enumerate(names):
        cloud = _load_cloud(emb_dir, name)
        aux = _load_aux(emb_dir, name)
        e_domain[i] = _domain_views(cloud, domain_dim)
        k_labels = len(labels_per[i])
        stats_raw[i] = _dataset_stats(cloud, aux, k_labels)
        # n_class bucket: regression (from aux) overrides the class-count bucket
        if aux is not None and aux.get("label_kind") == "regression":
            n_class_bucket_id[i] = N_CLASS_REGRESSION
        else:
            n_class_bucket_id[i] = n_class_bucket(k_labels)
    # z-score the stats across datasets so the dims are comparable
    mu, sd = stats_raw.mean(0), stats_raw.std(0)
    e_stats = ((stats_raw - mu) / np.where(sd > 1e-9, sd, 1.0)).astype(np.float32)

    # frozen view 2: e_label (label-set string)  +  view 3: e_card (HF card)
    label_strings = [", ".join(lbl) if lbl else f"{tt} task" for lbl, tt in zip(labels_per, task_types)]
    e_label = _encode(label_strings, encoder_name)
    if include_card:
        cards = _fetch_cards(names, card_cache_path, max_card_chars)
        e_card = _encode(cards, encoder_name)
    else:
        e_card = np.zeros((N, e_label.shape[1]), dtype=np.float32)

    frozen = np.concatenate([e_domain, e_label, e_card, e_stats], axis=1).astype(np.float32)
    assert frozen.shape[0] == N and not np.isnan(frozen).any(), "xd0 frozen build failed"
    assert task_type_id.max() < len(vocab) and n_class_bucket_id.max() < N_CLASS_BUCKETS

    return {
        "frozen": frozen,
        "task_type_id": task_type_id,
        "n_class_bucket_id": n_class_bucket_id,
        "arity_id": arity_id,
        "task_type_vocab": vocab,
        "num_task_types": len(vocab),
        "n_class_buckets": N_CLASS_BUCKETS,
        "arity_vocab": dict(ARITY_VOCAB),
        "num_arities": NUM_ARITIES,
        "view_dims": {"e_domain": e_domain.shape[1], "e_label": e_label.shape[1],
                      "e_card": e_card.shape[1], "e_stats": e_stats.shape[1]},
        "encoder_name": encoder_name,
        "names": names,
        "labels": labels_per,
        "task_types": task_types,
        "arities": arities,
    }
