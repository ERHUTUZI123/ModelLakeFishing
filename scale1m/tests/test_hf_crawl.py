import gzip
import json
import os

import pytest

from scale1m import hf_crawl as C
from scale1m.verify_raw import normalize


def test_trim_v1_keeps_exactly_the_declared_fields():
    rec = {
        "_id": "internal",
        "id": "org/model",
        "modelId": "org/model",
        "downloads": 7,
        "likes": 1,
        "pipeline_tag": "text-classification",
        "library_name": "transformers",
        "tags": ["pytorch", "region:us"],
        "createdAt": "2026-01-01T00:00:00.000Z",
        "siblings": [{"rfilename": "x"}] * 500,
        "safetensors": {"total": 123, "parameters": {"F32": 123}},
        "cardData": {"base_model": "org/base", "datasets": ["d"], "license": "mit",
                     "widget": [{"text": "hello"}]},
    }
    out = C.trim(rec, v2=False)
    assert set(out) == set(C.KEEP_V1) | {"safetensors", "cardData"}
    assert "siblings" not in out and "_id" not in out
    assert out["safetensors"] == {"total": 123}
    assert set(out["cardData"]) == {"base_model", "datasets"}


def test_trim_v2_adds_the_fields_v1_silently_dropped():
    rec = {
        "id": "org/model", "author": "org", "downloads": 7,
        "lastModified": "2026-02-02T00:00:00.000Z", "gated": False,
        "trendingScore": 3, "downloadsAllTime": 99,
        "config": {"architectures": ["BertModel"], "model_type": "bert",
                   "tokenizer_config": {"unk_token": "[UNK]"}},
        "baseModels": {"relation": "quantized",
                       "models": [{"_id": "x", "id": "org/base"}]},
        "gguf": {"total": 1},
        "cardData": {"language": ["en", "fr"], "license": "mit",
                     "base_model": "org/base", "widget": [{"text": "hi"}]},
    }
    out = C.trim(rec, v2=True)
    assert out["author"] == "org"
    assert out["cardData"]["language"] == ["en", "fr"]
    assert out["baseModels"] == {"relation": "quantized", "ids": ["org/base"]}
    assert out["config"] == {"architectures": ["BertModel"], "model_type": "bert"}
    assert out["gguf"] is True
    assert "widget" not in out["cardData"]
    v1 = C.trim(rec, v2=False)
    assert "author" not in v1 and "baseModels" not in v1 and "config" not in v1
    assert "language" not in v1["cardData"]


def test_trim_base_models_and_config_survive_junk():
    assert C.trim_base_models(None) is None
    assert C.trim_base_models({"models": []}) is None
    assert C.trim_base_models({"relation": "finetune", "models": [{"no_id": 1}]}) \
        == {"relation": "finetune", "ids": []}
    assert C.trim_config({"tokenizer_config": {"a": 1}}) is None


def test_trim_omits_absent_and_null_fields():
    out = C.trim({"id": "a/b", "downloads": 0, "safetensors": None, "cardData": None})
    assert out == {"id": "a/b", "downloads": 0}
    out2 = C.trim({"id": "a/b", "safetensors": {"parameters": {"F32": 5}}})
    assert "safetensors" not in out2


def test_trim_model_index_drops_verify_token_keeps_values():
    card = {"model-index": [{"name": "m", "results": [{
        "task": {"type": "text-classification", "name": "Text Classification"},
        "dataset": {"type": "glue", "name": "GLUE SST2", "config": "sst2", "split": "test"},
        "metrics": [{"type": "accuracy", "value": 0.93, "verified": False,
                     "verifyToken": "e" * 3000}],
    }]}]}
    out = C.trim({"id": "a/b", "cardData": card})
    mi = out["cardData"]["model-index"]
    assert mi == [{"task": "text-classification", "dataset": "glue",
                   "dataset_name": "GLUE SST2", "config": "sst2", "split": "test",
                   "metrics": [{"type": "accuracy", "value": 0.93}]}]
    assert "verifyToken" not in json.dumps(out)


def test_trim_model_index_survives_malformed_cards():
    assert C.trim_model_index({"results": []}) is None
    assert C.trim_model_index("not-a-model-index") is None
    assert C.trim_model_index([{"results": "nope"}, None]) is None


def test_parse_ratelimit():
    assert C.parse_ratelimit({"RateLimit": '"api";r=493;t=67'}) == (493, 67)
    assert C.parse_ratelimit({}) == (None, None)


def test_next_url_reads_the_link_header():
    class R:
        headers = {"Link": '<https://hf.co/api/models?cursor=abc>; rel="next"'}
    assert C.next_url(R()) == "https://hf.co/api/models?cursor=abc"

    class R2:
        headers = {"Link": '<https://hf.co/x>; rel="prev"'}
    assert C.next_url(R2()) == ""

    class R3:
        headers = {}
    assert C.next_url(R3()) == ""


def _write_plain(tmp, idx, ids):
    path = os.path.join(tmp, C.shard_stem(idx) + ".jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for i in ids:
            fh.write(json.dumps({"id": i}) + "\n")
    return path


def test_finalize_shard_freezes_and_hashes(tmp_path):
    tmp = str(tmp_path)
    _write_plain(tmp, 0, ["a/b", "c/d"])
    shards = []
    C.finalize_shard(tmp, 0, 2, shards)
    gz = os.path.join(tmp, "hf_models_00000.jsonl.gz")
    assert not os.path.exists(os.path.join(tmp, "hf_models_00000.jsonl"))
    assert not (os.stat(gz).st_mode & 0o222), "shard must be read-only"
    assert shards[0]["sha256"] == C.sha256_of(gz)
    with gzip.open(gz, "rt", encoding="utf-8") as fh:
        assert [json.loads(l)["id"] for l in fh] == ["a/b", "c/d"]


def test_replay_truncates_a_torn_partial_shard(tmp_path):
    tmp = str(tmp_path)
    _write_plain(tmp, 0, ["a/b", "c/d"])
    shards = []
    C.finalize_shard(tmp, 0, 2, shards)
    path = os.path.join(tmp, C.shard_stem(1) + ".jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "e/f"}) + "\n")
        fh.write('{"id": "g/h"')
    seen = C.replay_seen_ids(tmp, shards, 1, 1)
    assert seen == {"a/b", "c/d", "e/f"}
    with open(path, "r", encoding="utf-8") as fh:
        assert fh.read() == json.dumps({"id": "e/f"}) + "\n"


def test_write_json_atomic_overwrites_a_frozen_file(tmp_path):
    path = str(tmp_path / "PROVENANCE.json")
    C.write_json_atomic(path, {"a": 1})
    os.chmod(path, 0o444)
    C.write_json_atomic(path, {"a": 2})
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh) == {"a": 2}


def test_normalize_is_the_shared_id_rule():
    assert normalize("  Org/Model  ") == "org/model"
    assert normalize("ORG/MODEL") == normalize("org/model")


@pytest.mark.parametrize("idx,stem", [(0, "hf_models_00000"), (12, "hf_models_00012")])
def test_shard_stem_is_zero_padded_and_sorts(idx, stem):
    assert C.shard_stem(idx) == stem


def _single_stream_args(out, **over):
    import types
    base = dict(out=out, restart=False, v2_fields=True, sort="downloads",
                direction="-1", page_size=C.PAGE_LIMIT_MAX, limit=10_000_000,
                shard_size=50_000, token=None, timeout=5.0, max_retries=1,
                min_ratelimit_remaining=25, scale=1.0)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_default_sort_is_the_frozen_v1_ordering():
    args = C.build_parser().parse_args([])
    assert (args.sort, args.direction) == ("downloads", "-1")


def test_sort_flag_accepts_createdAt_and_rejects_junk():
    args = C.build_parser().parse_args(["--sort", "createdAt", "--direction", "1"])
    assert (args.sort, args.direction) == ("createdAt", "1")
    with pytest.raises(SystemExit):
        C.build_parser().parse_args(["--sort", "size"])


def test_full_enumeration_ends_on_cursor_exhaustion_not_on_limit(tmp_path, monkeypatch):
    pages = [([{"id": "a/one", "createdAt": "2026-08-18T00:00:00.000Z"},
               {"id": "b/two", "createdAt": "2026-08-17T00:00:00.000Z"}], "")]

    def fake_fetch(session, url, timeout, max_retries, stats, min_remaining):
        assert "sort=createdAt" in url and "direction=-1" in url
        return pages.pop(0)

    monkeypatch.setattr(C, "fetch", fake_fetch)
    out = str(tmp_path / "candidates_full")
    rc = C.crawl_single(_single_stream_args(out, sort="createdAt"))
    assert rc == 0
    with open(os.path.join(out, "PROVENANCE.json"), encoding="utf-8") as fh:
        prov = json.load(fh)
    assert prov["sort"] == "createdAt" and prov["direction"] == "-1"
    assert prov["exhausted_cursor"] is True
    assert prov["total_records"] == 2
    assert prov["last_created_at"] == "2026-08-17T00:00:00.000Z"


def test_resume_refuses_to_splice_two_orderings(tmp_path, capsys):
    out = str(tmp_path / "candidates_full")
    os.makedirs(out)
    C.write_json_atomic(os.path.join(out, "CURSOR.json"),
                        {"next_url": "https://example/next", "n_written": 5,
                         "n_in_shard": 5, "shard_index": 0,
                         "sort": "createdAt", "direction": "-1", "stats": C.new_stats()})
    rc = C.crawl_single(_single_stream_args(out, sort="downloads"))
    assert rc == 2
    assert "createdAt" in capsys.readouterr().err
