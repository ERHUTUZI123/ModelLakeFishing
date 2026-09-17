"""
learnable.py — Stage 2, Step 2: the learnable-embedding "cash-in".

The two nn.Embedding tables (size_bucket, family) and the frozen/learnable split
already LIVE in ModelNodeEncoder and are exercised by model.py:
  - ownership: they are submodules of HeteroGraphSAGE, so they are already
    inside model.parameters() — one optimizer, one param group, no special LR.
  - forward:  encoder looks them up, concats with frozen x, feeds conv1.
  - the gradient boundary (frozen x gets no grad, tables do) is asserted by
    model.py's smoke test.

This file adds the parts of Step 2 that are NOT yet code:

  A. save    — checkpoint that BINDS encoder weights to family_vocab
                        (+ the reproducibility metadata from Step 6). Loading
                        refuses a vocab whose length no longer matches the family
                        table — the guard against orphaned learnable rows.
  B. zero-shot — build a NEW model's encoder inputs with the SAME rules the
                        builder used: missing param_count -> unknown bucket (0),
                        family absent from vocab -> Other (0). No retraining, no
                        new rows until a family clears FAMILY_MIN_COUNT upstream.
  C. diagnostics — sparse-gradient health: which bucket/family rows actually
                        moved from init (rare rows that never moved learned
                        nothing — the FAMILY_MIN_COUNT rationale), and whether the
                        high-frequency unknown/Other rows are being trained.

Run the mechanism smoke test:  python -m ModelLakeFishing.stage2TrainGraphSAGE.learnable
"""

import os
import sys
import csv

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# _STAGE1 stays on sys.path: stage1's modules import `dataset_embed` as a
# top-level package internally, so xm0_builder cannot be imported without it.
_STAGE1 = os.path.join(_REPO_ROOT, "ModelLakeFishing", "stage1BuildTransferGraph")
for _p in (_REPO_ROOT, _STAGE1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ModelLakeFishing.stage2TrainGraphSAGE.model import HeteroGraphSAGE  # noqa: E402
# size-bucket rules + constants: the SINGLE source of truth, reused for both
# the checkpoint's repro record and the zero-shot input construction.
# Full package path (matches model.py) rather than the top-level
# `stage1BuildTransferGraph` name, which is not importable under `python -m`.
from ModelLakeFishing.stage1BuildTransferGraph.dataset_embed.xm0_builder import (  # noqa: E402
    param_count_to_size_bucket,
    FAMILY_ID_OTHER,
    FAMILY_OTHER,
    NUM_SIZE_BUCKETS,
    SIZE_LOG10_MIN,
    SIZE_LOG10_MAX,
    SIZE_BUCKET_WIDTH,
    HASH_BUCKETS,
)

# build_xm0 defaults — recorded so the frozen half can be recomputed identically
# for a new model three months from now (CLAUDE.md Step 6).
DEFAULT_ENAME_SEED = 42
DEFAULT_ENAME_TOKEN_DIM = 64
DEFAULT_EDESC_ENCODER = "all-MiniLM-L6-v2"


# ── A. — bind weights to family_vocab ───────────────────────────────

def _validate_vocab_binding(family_vocab: dict, num_rows: int) -> None:
    """
    The vocab is the only credential for embedding-row identity, so equal length
    is NOT enough — a vocab can match in size yet still be unusable (duplicate or
    missing ids, or Other not pinned to row 0). Verify the binding is a true
    bijection family -> {0..num_rows-1} with FAMILY_OTHER == FAMILY_ID_OTHER.
    Raises ValueError on any violation (used at both save and load time).
    """
    if len(family_vocab) != num_rows:
        raise ValueError(
            f"family_vocab has {len(family_vocab)} entries but the family table "
            f"has {num_rows} rows — vocab cannot name every learnable row."
        )
    ids = set(family_vocab.values())
    if ids != set(range(num_rows)):
        raise ValueError(
            "family_vocab ids are not a contiguous bijection onto "
            f"{{0..{num_rows - 1}}} (duplicate, missing, or out-of-range ids) — "
            "embedding rows would be misnamed."
        )
    if family_vocab.get(FAMILY_OTHER) != FAMILY_ID_OTHER:
        raise ValueError(
            f"family_vocab must map {FAMILY_OTHER!r} -> {FAMILY_ID_OTHER} "
            "(the Other/unknown row); the zero-shot degradation path depends on it."
        )


def _repro_metadata(extra: dict | None = None) -> dict:
    """Everything needed to recompute features for a new model later."""
    repro = {
        "size_bucket_constants": {
            "NUM_SIZE_BUCKETS": NUM_SIZE_BUCKETS,
            "SIZE_LOG10_MIN": SIZE_LOG10_MIN,
            "SIZE_LOG10_MAX": SIZE_LOG10_MAX,
            "SIZE_BUCKET_WIDTH": SIZE_BUCKET_WIDTH,
        },
        "e_name_seed": DEFAULT_ENAME_SEED,
        "e_name_token_dim": DEFAULT_ENAME_TOKEN_DIM,
        "e_name_hash_buckets": HASH_BUCKETS,
        "e_desc_encoder": DEFAULT_EDESC_ENCODER,
    }
    if extra:
        repro.update(extra)
    return repro


def _arch_of(model: HeteroGraphSAGE) -> dict:
    """Dims needed to rebuild an identical model before load_state_dict."""
    enc = model.model_encoder
    arch = {
        "metadata": model.graph_metadata,
        "frozen_dim": enc.frozen_dim,
        "num_size_buckets": enc.size_embedding.num_embeddings,
        # family table may be absent (D1 use_family=False); 0 marks "no table"
        "num_families": (enc.family_embedding.num_embeddings
                         if enc.family_embedding is not None else 0),
        "size_dim": enc.size_embedding.embedding_dim,
        "family_dim": (enc.family_embedding.embedding_dim
                       if enc.family_embedding is not None else 16),
        # D1 §5.3 feature-variant record (defaults = legacy on old checkpoints)
        "name_dim": enc.name_dim,
        "use_desc": enc.use_desc,
        "use_family": enc.use_family,
        "name_proj_dim": enc.name_proj_dim,
        "name_proj_seed": enc.name_proj_seed,
        "num_model_tasks": enc.num_model_tasks,
        "model_task_dim": (enc.task_embedding.embedding_dim
                           if enc.task_embedding is not None else 16),
        "hidden_channels": model.model_proj.out_features,
        "out_dim": (model.model_head.out_features
                    if getattr(model, "separate_heads", False) else model.head.out_features),
        "separate_heads": getattr(model, "separate_heads", False),
        # model-reconstruction metadata (NOT the vocab<->weights binding): records
        # the SAGE depth so a 1-layer checkpoint reloads with matching state_dict
        # keys. Backward-compatible: old checkpoints lack it -> default 2 on load.
        "num_layers": getattr(model, "num_layers", 2),
        # Phase 2 edge-aware flags (default False keeps old checkpoints valid)
        "edge_aware": getattr(model, "edge_aware", False),
        "weighted_relations": getattr(model, "weighted_relations", None),
    }
    # dataset side: xd0 DatasetNodeEncoder (when present) needs its own dims +
    # the frozen dataset width. When absent, record the plain dataset_proj input.
    if getattr(model, "use_dataset_encoder", False):
        de = model.dataset_encoder
        arch.update({
            "dataset_in_dim": de.frozen_dim,
            "num_task_types": de.task_type_embedding.num_embeddings,
            "n_class_buckets": de.n_class_embedding.num_embeddings,
            "num_arities": de.arity_embedding.num_embeddings,
            "task_dim": de.task_type_embedding.embedding_dim,
            "nclass_dim": de.n_class_embedding.embedding_dim,
            "arity_dim": de.arity_embedding.embedding_dim,
            # v3 Z1 (None on pre-Z checkpoints -> legacy passthrough)
            "dataset_frozen_proj_dim": getattr(de, "frozen_proj_dim", None),
        })
    else:
        arch["dataset_in_dim"] = model.dataset_proj.in_features
    return arch


def _validate_task_vocab(vocab: dict, num_rows: int, *, label: str) -> None:
    """Same bijection discipline as family: contiguous ids, Other pinned to 0."""
    if len(vocab) != num_rows or set(vocab.values()) != set(range(num_rows)):
        raise ValueError(
            f"{label} ({len(vocab)} entries) is not a contiguous bijection onto "
            f"{{0..{num_rows - 1}}} — embedding rows would be misnamed.")
    if vocab.get("Other") != 0:
        raise ValueError(f"{label} must map 'Other' -> 0 (zero-shot fallback row).")


def save_checkpoint(model, family_vocab: dict, out_path: str, *, extra_repro=None,
                    task_type_vocab: dict | None = None,
                    model_task_vocab: dict | None = None):
    """
    Save weights + family_vocab + repro metadata as ONE bound artifact, and also
    drop a sidecar family_vocab.csv next to it (the vocab is the only credential
    for embedding-row identity — never let it drift away from the weights).

    family_vocab : the SAME dict xm0 produced (family -> id). len(family_vocab)
                   must equal the family table's row count, or the rows are
                   already orphaned — we assert that here, at save time.
    task_type_vocab : OPTIONAL — the xd0 dataset task_type vocab. When the model
                   has a DatasetNodeEncoder it is the credential for the
                   task_type embedding rows; bound and validated like family_vocab.
    model_task_vocab : OPTIONAL — the D1 MODEL-side task vocab (e_task rows,
                   from d1_model_task_vocab.py). REQUIRED when the encoder has a
                   task table; bound and validated like family_vocab.
    """
    enc = model.model_encoder
    if enc.family_embedding is not None:
        _validate_vocab_binding(family_vocab, enc.family_embedding.num_embeddings)
    if enc.task_embedding is not None:
        if model_task_vocab is None:
            raise ValueError(
                "encoder has a model-task table but no model_task_vocab was given "
                "— the e_task rows would be orphaned (same rule as family_vocab)")
        _validate_task_vocab(model_task_vocab, enc.task_embedding.num_embeddings,
                             label="model_task_vocab")

    payload = {
        "state_dict": model.state_dict(),
        "family_vocab": family_vocab,
        "arch": _arch_of(model),
        "repro": _repro_metadata(extra_repro),
    }
    if model_task_vocab is not None and enc.task_embedding is not None:
        payload["model_task_vocab"] = model_task_vocab
    if getattr(model, "use_dataset_encoder", False):
        rows = model.dataset_encoder.task_type_embedding.num_embeddings
        if task_type_vocab is not None:
            assert len(task_type_vocab) == rows and task_type_vocab.get("Other") == 0, (
                f"task_type_vocab ({len(task_type_vocab)}) must have {rows} entries "
                "with Other->0 (orphan-row guard for the dataset task_type table)"
            )
            payload["task_type_vocab"] = task_type_vocab

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)

    # sidecar CSV (human-readable, and the format xm0's vocab loader expects).
    # Named after the checkpoint stem so multiple checkpoints in one directory do
    # not clobber each other's vocab (e.g. stage2.pt -> stage2.family_vocab.csv).
    abs_out = os.path.abspath(out_path)
    stem = os.path.splitext(os.path.basename(abs_out))[0]
    csv_path = os.path.join(os.path.dirname(abs_out), f"{stem}.family_vocab.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "family_id"])
        for fam, fid in sorted(family_vocab.items(), key=lambda kv: kv[1]):
            w.writerow([fam, fid])

    # sidecar for the dataset task_type vocab (when the model has an xd0 encoder)
    if task_type_vocab is not None and getattr(model, "use_dataset_encoder", False):
        tt_path = os.path.join(os.path.dirname(abs_out), f"{stem}.task_type_vocab.csv")
        with open(tt_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task_type", "task_type_id"])
            for tt, tid in sorted(task_type_vocab.items(), key=lambda kv: kv[1]):
                w.writerow([tt, tid])

    # sidecar for the D1 model-side task vocab (e_task row credential)
    if model_task_vocab is not None and enc.task_embedding is not None:
        mt_path = os.path.join(os.path.dirname(abs_out), f"{stem}.model_task_vocab.csv")
        with open(mt_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["task", "task_id"])
            for tt, tid in sorted(model_task_vocab.items(), key=lambda kv: kv[1]):
                w.writerow([tt, tid])
    return out_path


def load_checkpoint(path: str, *, map_location="cpu"):
    """
    Rebuild the model from the stored arch, load weights, and verify the
    weights<->vocab binding. Returns (model, family_vocab, repro).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    arch = ckpt["arch"]
    model = HeteroGraphSAGE(
        metadata=arch["metadata"],
        frozen_dim=arch["frozen_dim"],
        num_size_buckets=arch["num_size_buckets"],
        # 0 marks "no family table" (use_family=False); nn.Embedding still needs
        # a positive row count at build time — the table is skipped anyway.
        num_families=arch["num_families"] or 1,
        dataset_in_dim=arch["dataset_in_dim"],
        hidden_channels=arch["hidden_channels"],
        out_dim=arch["out_dim"],
        size_dim=arch["size_dim"],
        family_dim=arch["family_dim"],
        num_layers=arch.get("num_layers", 2),   # default keeps old checkpoints valid
        edge_aware=arch.get("edge_aware", False),
        weighted_relations=arch.get("weighted_relations", None),
        separate_heads=arch.get("separate_heads", False),
        # D1 §5.3 feature variants: defaults keep every pre-D1 checkpoint valid
        name_dim=arch.get("name_dim"),
        use_desc=arch.get("use_desc", True),
        use_family=arch.get("use_family", True),
        name_proj_dim=arch.get("name_proj_dim"),
        name_proj_seed=arch.get("name_proj_seed", 42),
        num_model_tasks=arch.get("num_model_tasks"),
        model_task_dim=arch.get("model_task_dim", 16),
        # dataset side (xd0): None for old checkpoints -> plain dataset_proj
        num_task_types=arch.get("num_task_types"),
        n_class_buckets=arch.get("n_class_buckets"),
        num_arities=arch.get("num_arities"),
        task_dim=arch.get("task_dim", 16),
        nclass_dim=arch.get("nclass_dim", 8),
        arity_dim=arch.get("arity_dim", 4),
        dataset_frozen_proj_dim=arch.get("dataset_frozen_proj_dim"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    vocab = ckpt["family_vocab"]
    if model.model_encoder.family_embedding is not None:
        _validate_vocab_binding(vocab, model.model_encoder.family_embedding.num_embeddings)
    if model.model_encoder.task_embedding is not None:
        _validate_task_vocab(ckpt["model_task_vocab"],
                             model.model_encoder.task_embedding.num_embeddings,
                             label="model_task_vocab")
    # surface the dataset task_type / model task vocabs (if bound) via repro,
    # keeping the 3-tuple return signature every caller already unpacks.
    repro = dict(ckpt["repro"])
    if "task_type_vocab" in ckpt:
        repro["task_type_vocab"] = ckpt["task_type_vocab"]
    if "model_task_vocab" in ckpt:
        repro["model_task_vocab"] = ckpt["model_task_vocab"]
    return model, vocab, repro


# ── B. — same rules as the builder, unseen family -> Other ─────────

def new_model_input(frozen_vec, param_count, family, family_vocab, *, device=None):
    """
    Map a NEW (possibly never-seen) model to encoder inputs, by the SAME rules
    xm0 used so a cold-start model is treated identically to a trained one:

      param_count missing/None  -> size bucket 0 (unknown)
      family not in family_vocab -> family id 0 (Other)   [designed degradation]

    Returns (x_frozen[1, frozen_dim], size_bucket_id[1], family_id[1]).
    """
    fv = torch.as_tensor(frozen_vec, dtype=torch.float32).reshape(1, -1)
    size_id = torch.tensor([param_count_to_size_bucket(param_count)], dtype=torch.long)
    fam_id = torch.tensor([family_vocab.get(family, FAMILY_ID_OTHER)], dtype=torch.long)
    if device is not None:
        fv, size_id, fam_id = fv.to(device), size_id.to(device), fam_id.to(device)
    return fv, size_id, fam_id


def new_model_task_id(task, model_task_vocab, *, device=None):
    """
    D1 e_task zero-shot rule, isomorphic to size/family degradation:
    task missing or absent from the vocab -> id 0 (Other). `task` should be the
    CANONICAL task (d1_model_task_vocab.canon_task of the HF pipeline_tag).
    """
    tid = torch.tensor([model_task_vocab.get(task, 0)], dtype=torch.long)
    return tid.to(device) if device is not None else tid


@torch.no_grad()
def encode_new_model(model, frozen_vec, param_count, family, family_vocab,
                     task=None, model_task_vocab=None):
    """
    Run a new model through the TRAINED ModelNodeEncoder (frozen || size || fam
    [|| task]). This is the x_m^(0) of a cold-start model; the full z_m
    additionally needs the GNN with the node joined into the graph (serving-time,
    Step 6). Returned here at encoder level so the zero-shot routing (unseen
    family/task -> Other) is testable without faking graph structure.
    """
    # follow the model's device so this works after model.to("cuda")
    device = model.model_encoder.size_embedding.weight.device
    fv, size_id, fam_id = new_model_input(
        frozen_vec, param_count, family, family_vocab, device=device
    )
    task_id = None
    if model.model_encoder.task_embedding is not None:
        assert model_task_vocab is not None, (
            "encoder has an e_task table: pass model_task_vocab (from the "
            "checkpoint's repro['model_task_vocab']) so the Other fallback applies")
        task_id = new_model_task_id(task, model_task_vocab, device=device)
    return model.model_encoder(fv, size_id, fam_id, task_id)


# ── C. — sparse-gradient health of the two tables ──────────────────────────

def _encoder_tables(encoder) -> dict:
    """The encoder's LEARNABLE tables that actually exist (D1 variants may drop
    family and add task)."""
    tables = {"size": encoder.size_embedding}
    if getattr(encoder, "family_embedding", None) is not None:
        tables["family"] = encoder.family_embedding
    if getattr(encoder, "task_embedding", None) is not None:
        tables["task"] = encoder.task_embedding
    return tables


def snapshot_weights(encoder) -> dict:
    """Clone every table's weights — call BEFORE training to compare against later."""
    return {name: emb.weight.detach().clone()
            for name, emb in _encoder_tables(encoder).items()}


def embedding_health(encoder, init_weights: dict | None = None, *, move_atol: float = 1e-6) -> dict:
    """
    Report, per table, what the sparse gradients actually touched.

    With init_weights (a snapshot_weights from before training):
      unmoved_rows  — rows whose weight barely changed from init. A RARE family
                      row that never moved learned nothing (FAMILY_MIN_COUNT
                      rationale: below threshold, folding into Other beats a noise
                      row). Expect many unmoved family rows on the small zoo —
                      only the families actually present among the models move.
      row0_move     — how much the unknown-bucket / Other-family row (id 0) moved;
                      as the highest-frequency row it should be among the most
                      updated ("the prior of missingness", a feature not a bug).
    """
    report = {}
    tables = _encoder_tables(encoder)
    for name, emb in tables.items():
        w = emb.weight.detach()
        norms = w.norm(dim=1)
        entry = {
            "rows": int(w.shape[0]),
            "row0_norm": float(norms[0]),
            "mean_norm": float(norms.mean()),
        }
        if init_weights is not None:
            move = (w - init_weights[name]).norm(dim=1)
            unmoved = (move <= move_atol).nonzero().flatten().tolist()
            entry.update(
                unmoved_rows=unmoved,
                n_unmoved=len(unmoved),
                n_moved=int(w.shape[0]) - len(unmoved),
                row0_move=float(move[0]),
                max_move=float(move.max()),
            )
        report[name] = entry
    return report