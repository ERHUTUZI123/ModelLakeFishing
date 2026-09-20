import os
import re

import numpy as np
import pandas as pd

NAME_DIM = 64
TEXT_DIM = 384
TEXT_LO = NAME_DIM
TEXT_HI = NAME_DIM + TEXT_DIM
_WORD = re.compile(r"[a-z0-9]+")


def fixed_tie_break(n):
    ids = np.arange(int(n), dtype=np.uint64)
    with np.errstate(over="ignore"):
        return ids * np.uint64(11400714819323198485) + np.uint64(0xD1B54A32D192ED03)


class Context:
    def __init__(self, run_dir, attrs_path, ladder_path, graph_dir, device="cpu"):
        self.run_dir = run_dir
        self.graph_dir = graph_dir
        self.device = device
        self.attrs = pd.read_parquet(attrs_path)
        ladder = pd.read_parquet(ladder_path, columns=["mappedID", "model"])
        self.n = len(ladder)
        expected = np.arange(self.n, dtype=np.int64)
        if not np.array_equal(ladder["mappedID"].to_numpy(), expected):
            raise AssertionError("ladder mappedID is not contiguous 0..N-1")
        if len(self.attrs) != self.n or not np.array_equal(
                self.attrs["mappedID"].to_numpy(), expected):
            raise AssertionError("baseline sidecar is not aligned to mappedID")
        self.model_name = ladder["model"].astype(str).to_numpy()
        udi = pd.read_parquet(os.path.join(run_dir, "dataset_ids.parquet")).sort_values("mappedID")
        self.dataset_name = udi["dataset"].astype(str).to_numpy()
        self.dataset_root = udi["root"].astype(str).to_numpy()
        self.tie_break = fixed_tie_break(self.n)
        self._x_m = None
        self._x_d = None

    @property
    def x_m(self):
        if self._x_m is None:
            self._x_m = np.load(os.path.join(self.graph_dir, "x_model.npy"), mmap_mode="r")
        return self._x_m

    @property
    def x_d(self):
        if self._x_d is None:
            self._x_d = np.load(os.path.join(self.graph_dir, "x_dataset.npy"), mmap_mode="r")
        return self._x_d

    def query_text(self, dataset_id):
        return self.dataset_name[int(dataset_id)].replace("\t", " ")


def popularity(ctx):
    values = ctx.attrs["downloads"].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


class BM25:

    name = "L_bm25"
    K1 = 1.5
    B = 0.75

    def __init__(self, ctx, max_df=0.25, min_df=2):
        from sklearn.feature_extraction.text import CountVectorizer

        docs = (pd.Series(ctx.model_name).str.replace("/", " ", regex=False) + " "
                + ctx.attrs["tags"].astype(str) + " "
                + ctx.attrs["pipeline_tag"].astype(str)).to_numpy()
        vec = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+",
                              max_df=max_df, min_df=min_df, dtype=np.float32)
        matrix = vec.fit_transform(docs).tocsc()
        self.vocab = vec.vocabulary_
        length = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
        avg_length = float(length.mean()) if length.size else 1.0
        self.norm = (self.K1 * (1 - self.B + self.B * length / avg_length)).astype(np.float32)
        n = matrix.shape[0]
        df = np.diff(matrix.indptr).astype(np.float64)
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.matrix = matrix
        self.n = n

    def __call__(self, ctx, dataset_id):
        scores = np.zeros(self.n, dtype=np.float32)
        for token in set(_WORD.findall(ctx.query_text(dataset_id).lower())):
            column = self.vocab.get(token)
            if column is None:
                continue
            lo, hi = self.matrix.indptr[column], self.matrix.indptr[column + 1]
            rows = self.matrix.indices[lo:hi]
            tf = self.matrix.data[lo:hi]
            scores[rows] += (self.idf[column] * tf * (self.K1 + 1.0)
                             / (tf + self.norm[rows]))
        return scores


class FrozenMiniLM:

    name = "S_frozen_minilm"

    def __init__(self, ctx):
        import torch

        self.device = ctx.device
        model_text = np.ascontiguousarray(ctx.x_m[:, TEXT_LO:TEXT_HI], dtype=np.float32)
        query_text = np.ascontiguousarray(ctx.x_d[:, TEXT_LO:TEXT_HI], dtype=np.float32)
        if model_text.shape != (ctx.n, TEXT_DIM) or query_text.shape[1] != TEXT_DIM:
            raise AssertionError("MiniLM slice is not exactly [64:448]")
        self.model = torch.from_numpy(model_text)
        self.model = self.model / self.model.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.query = torch.from_numpy(query_text)
        self.query = self.query / self.query.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if self.device != "cpu":
            self.model = self.model.to(self.device)
            self.query = self.query.to(self.device)

    def __call__(self, ctx, dataset_id):
        scores = self.model @ self.query[int(dataset_id)]
        return scores.detach().cpu().numpy().astype(np.float32)
