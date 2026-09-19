"""Live acquisition correctness without a Hub account or a large crawl."""
from __future__ import annotations

import json

import pytest
import requests

from scale1m import hf_crawl as C
from scale1m.reproduction import live_snapshot as L


def row(ident, **extra):
    return {"id": ident, "private": False, "createdAt": "2026-09-19T00:00:00Z", **extra}


def next_url(kind):
    return "https://huggingface.co/api/" + kind + "?cursor=page2"


def install_pages(monkeypatch, pages):
    calls = []

    def fetch(session, url, timeout, retries, stats, remaining):
        calls.append(url)
        kind = "datasets" if "/datasets" in url else "models"
        result = pages[kind].pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(C, "fetch", fetch)
    return calls


def small_lake():
    return {"models": [([row("org/a"), row("org/private", private=True)], next_url("models")),
                       ([row("org/a"), row("org/b")], "")],
            "datasets": [([row("org/dataset", description="Example")], "")]}


def test_complete_public_streams_dedupe_and_bind(monkeypatch, tmp_path):
    calls = install_pages(monkeypatch, small_lake())
    result = L.download(tmp_path, token="DO_NOT_STORE_THIS_TOKEN")
    assert result["ok"] and result["both_cursors_exhausted"]
    assert result["counts"] == {"models": 2, "datasets": 1}
    assert result["snapshot_is_atomic"] is False
    assert len(calls) == 3
    assert "sort=createdAt" in calls[0]
    assert "private" in calls[0]
    assert L.input_digest(result) == L.input_digest(L.verify(tmp_path))
    manifest_text = (tmp_path / "LIVE_SNAPSHOT.json").read_text()
    assert "DO_NOT_STORE_THIS_TOKEN" not in manifest_text
    for path in tmp_path.rglob("*.json"):
        assert "DO_NOT_STORE_THIS_TOKEN" not in path.read_text()
    # Existing compatible downstream readers can consume the new shards.
    from scale1m.verify_raw import iter_records
    prov = json.loads((tmp_path / "models/PROVENANCE.json").read_text())
    assert [r["id"] for _, r in iter_records(str(tmp_path / "models"), prov["shards"])] == ["org/a", "org/b"]
    assert prov["stats"]["private_skipped"] == 1
    assert prov["stats"]["duplicates_skipped"] == 1
    assert L.download(tmp_path) == L.verify(tmp_path)  # no network after completion
    assert len(calls) == 3


def test_interruption_resumes_last_committed_cursor(monkeypatch, tmp_path):
    pages = small_lake()
    pages["models"][1] = requests.ConnectionError("interrupted")
    install_pages(monkeypatch, pages)
    with pytest.raises(requests.ConnectionError):
        L.download(tmp_path)
    assert not L.verify(tmp_path)["ok"]
    calls = install_pages(monkeypatch, {"models": [([row("org/b")], "")],
                                       "datasets": [([row("org/dataset")], "")]})
    report = L.download(tmp_path)
    assert report["counts"]["models"] == 2
    assert calls[0] == next_url("models")


def test_zero_public_rows_advance_without_empty_shards(monkeypatch, tmp_path):
    last = "https://huggingface.co/api/models?cursor=page3"
    install_pages(monkeypatch, {"models": [([row("org/private", private=True)], next_url("models")),
                                           ([row("org/a")], last), ([], "")],
                               "datasets": [([row("org/dataset")], "")]})
    assert L.download(tmp_path)["ok"]
    state = json.loads((tmp_path / "models/CURSOR.json").read_text())
    assert state["pages"] == 3 and len(state["shards"]) == 1
    assert state["shards"][0]["n_records"] == 1


def test_duplicate_only_page_creates_no_empty_shard(monkeypatch, tmp_path):
    install_pages(monkeypatch, {"models": [([row("org/a")], next_url("models")),
                                           ([row("org/a")], "")],
                               "datasets": [([row("org/dataset")], "")]})
    report = L.download(tmp_path)
    assert report["counts"]["models"] == 1
    shards = json.loads((tmp_path / "models/SHARDS.json").read_text())
    assert len(shards) == 1


def test_completely_empty_stream_is_not_a_reproducible_lake(monkeypatch, tmp_path):
    install_pages(monkeypatch, {"models": [([], "")], "datasets": [([row("org/dataset")], "")]})
    with pytest.raises(ValueError, match="empty"):
        L.download(tmp_path)
    assert not (tmp_path / "LIVE_SNAPSHOT.json").exists()


def test_crash_after_shard_write_refetches_only_uncommitted_page(monkeypatch, tmp_path):
    install_pages(monkeypatch, small_lake())
    atomic = L._atomic

    def interrupt(path, payload):
        if path.name == "CURSOR.json" and path.parent.name == "models" and payload["pages"] == 1:
            raise OSError("crash between shard and cursor commits")
        return atomic(path, payload)

    monkeypatch.setattr(L, "_atomic", interrupt)
    with pytest.raises(OSError):
        L.download(tmp_path)
    assert (tmp_path / "models/hf_models_00000.jsonl.gz").exists()
    monkeypatch.setattr(L, "_atomic", atomic)
    calls = install_pages(monkeypatch, small_lake())
    report = L.download(tmp_path)
    assert report["counts"]["models"] == 2
    assert "sort=createdAt" in calls[0]


@pytest.mark.parametrize("bad_page,next_cursor", [
    ({"error": "API changed"}, ""),
    ([{"id": "org/no-private-field"}], ""),
    ([{"private": False}], ""),
    ([row(" ")], ""),
    ([], next_url("models")),
    ([row("org/a")], "https://malicious.invalid/api/models?cursor=leak"),
])
def test_incomplete_or_unknown_response_never_commits_success(monkeypatch, tmp_path, bad_page, next_cursor):
    install_pages(monkeypatch, {"models": [(bad_page, next_cursor)]})
    with pytest.raises(ValueError):
        L.download(tmp_path)
    assert not (tmp_path / "LIVE_SNAPSHOT.json").exists()
    assert not L.verify(tmp_path)["ok"]


def test_repeated_cursor_stops_instead_of_looping(monkeypatch, tmp_path):
    install_pages(monkeypatch, {"models": [([row("org/a")], next_url("models")),
                                           ([row("org/b")], next_url("models"))]})
    with pytest.raises(ValueError, match="cursor repeated"):
        L.download(tmp_path)


def test_corrupt_committed_input_is_not_silently_replaced(monkeypatch, tmp_path):
    install_pages(monkeypatch, small_lake())
    L.download(tmp_path)
    path = tmp_path / "models/hf_models_00000.jsonl.gz"
    path.write_bytes(b"broken")
    assert not L.verify(tmp_path)["ok"]
    with pytest.raises(ValueError, match="damaged"):
        L.download(tmp_path)


def test_unlisted_shard_and_missing_completion_are_rejected(monkeypatch, tmp_path):
    install_pages(monkeypatch, small_lake())
    L.download(tmp_path)
    (tmp_path / "datasets/hf_models_99999.jsonl.gz").write_bytes(b"unlisted")
    assert not L.verify(tmp_path)["ok"]
    (tmp_path / "LIVE_SNAPSHOT.json").unlink()
    assert not L.verify(tmp_path)["ok"]


class Response:
    def __init__(self, status=200, records=None, headers=None):
        self.status_code = status
        self.records = records if records is not None else [row("org/a")]
        self.headers = headers or {}

    def json(self):
        return self.records

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Session:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        pass


def test_retry_after_and_server_retries_use_existing_fetch(monkeypatch):
    session = Session([Response(429, headers={"Retry-After": "3"}), Response(503), Response()])
    monkeypatch.setattr(C, "make_session", lambda token: session)
    waits = []
    monkeypatch.setattr(C.time, "sleep", waits.append)
    monkeypatch.setattr(C.random, "uniform", lambda *args: 0)
    safe = L._Session("https://huggingface.co/api/models", "TOKEN")
    stats = C.new_stats()
    records, following = C.fetch(safe, "https://huggingface.co/api/models", 1, 3, stats, 25)
    assert records[0]["id"] == "org/a" and following == ""
    assert stats["retries"] == 2 and waits == [3, 4]
    assert all(call[1]["allow_redirects"] is False for call in session.calls)


@pytest.mark.parametrize("response", [Response(302, headers={"Location": "https://other.invalid"}),
    Response(headers={"Link": '<https://other.invalid/api/models>; rel="next"'}),
    Response(headers={"Link": "malformed next link"})])
def test_token_session_blocks_redirect_or_bad_link_before_following(monkeypatch, response):
    session = Session([response])
    monkeypatch.setattr(C, "make_session", lambda token: session)
    safe = L._Session("https://huggingface.co/api/models", "TOKEN")
    with pytest.raises(ValueError):
        safe.get("https://huggingface.co/api/models", 1)
    assert len(session.calls) == 1


def test_fetch_retry_exhaustion_does_not_become_empty_page(monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda _: None)
    session = Session([Response(503), Response(503)])
    with pytest.raises(requests.HTTPError):
        C.fetch(session, "https://huggingface.co/api/models", 1, 1, C.new_stats(), 25)


def test_live_native_cards_feed_hf_only_preprocessing_pipeline(monkeypatch, tmp_path):
    """Production CLI arguments consume raw acquisition, with history forbidden."""
    import importlib

    import pandas as pd

    from scale1m import build_ladder_rf, merge_supervision
    from scale1m.reproduction.pipeline import live_steps

    def no_history(*args, **kwargs):
        pytest.fail("Live preprocessing attempted to load author historical data")

    monkeypatch.setattr(merge_supervision, "load_source", no_history)
    monkeypatch.setattr(build_ladder_rf, "family_from_history", no_history)
    monkeypatch.setenv("MLF_DATA_DIR", str(tmp_path / "absent-author-data"))
    native = []
    for root in range(15):
        for model in range(5):
            native.append(row(f"org/model-{root}-{model}", config={"model_type": "bert"},
                cardData={"model-index": [{"name": "Evaluation", "results": [{
                    "task": {"type": "text-classification"},
                    "dataset": {"type": f"dataset-{root}/main", "name": f"Task {root}"},
                    "metrics": [{"type": "accuracy", "value": 0.5 + model / 10}],
                }]}]}))
    native.append(row("org/unlabelled-model"))
    third = "https://huggingface.co/api/models?cursor=page3"
    dataset_rows = [row(f"dataset-{root}/main", description=f"Description {root}",
                        cardData={"task_categories": ["text-classification"]}) for root in range(15)]
    install_pages(monkeypatch, {
        "models": [([row("org/private", private=True)], next_url("models")),
                   (native[:40], third), (native[40:], "")],
        "datasets": [(dataset_rows, "")],
    })
    source, work = tmp_path / "snapshot", tmp_path / "work"
    captured = L.download(source)
    assert captured["counts"] == {"models": 76, "datasets": 15}
    # Invoke each actual stage's argument parser with the exact live plan's
    # arguments. Checkpoint/test-specific fixture writers are not substituted.
    for step in live_steps(source, work, "cpu")[:5]:
        module = importlib.import_module(step.argv[2])
        assert module.main(list(step.argv[3:])) == 0, step.name
    edges = pd.read_parquet(work / "rf/canon/supervision_merged.parquet")
    nodes = pd.read_parquet(work / "rf/canon/dataset_nodes_merged.parquet")
    cards = pd.read_parquet(work / "dataset_cards.parquet")
    candidates = pd.read_parquet(work / "ladder/full_model_ids.parquet")
    assert len(edges) == 75 and len(nodes) == 15
    assert set(edges.source) == {"hf_model_index"}
    assert nodes.gold_eligible.all() and cards.card_source.eq("hf_card").all()
    assert cards.description.str.startswith("Description ").all()
    assert len(candidates) == 76 and candidates.in_snapshot.all()
    assert candidates.mappedID.tolist() == list(range(76))
    assert "org/private" not in set(candidates.model)
    assert not (tmp_path / "absent-author-data").exists()
    assert not (work / "features/family_vocab.csv").exists()
    assert L.verify(source)["ok"]
