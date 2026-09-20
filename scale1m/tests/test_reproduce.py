import io
import json
import os
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("repro.archive", reason="Optional archived release tooling")

from repro.archive import (
    MultipartReader,
    MultipartWriter,
    _safe_relative,
    build_bundle,
    extract_bundle,
    sha256_file,
)
from repro.verify import PAPER_RESULTS_PATH, _equal, _pointer, summarize


def test_json_pointer_and_recursive_numeric_tolerance():
    payload = {"a/b": {"~x": [1.0, 2.0]}}
    assert _pointer(payload, "/a~1b/~0x/1") == 2.0
    assert _equal([1.0, {"x": 2.0}], [1.0 + 1e-8, {"x": 2.0}], 1e-7)
    assert not _equal([1.0], [1.1], 1e-7)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "a/../../b"])
def test_archive_member_path_rejects_escape(name):
    with pytest.raises(ValueError):
        _safe_relative(name)


def test_multipart_roundtrip_and_installed_hashes(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "source-data"
    out = tmp_path / "assets"
    repo.mkdir()
    (data / "input").mkdir(parents=True)
    (data / "input" / "a.bin").write_bytes(bytes(range(251)) * 40)
    (data / "input" / "b.txt").write_text("fixed\n", encoding="utf-8")
    spec = {
        "description": "test",
        "files": [{"root": "data", "source": "input", "target": "installed"}],
        "copies": [{"from": "installed/b.txt", "to": "alias/b.txt"}],
    }
    asset, manifest = build_bundle("tiny", spec, repo, data, out, part_bytes=997)
    assert len(asset["parts"]) > 1
    target = tmp_path / "target"
    extract_bundle(asset, manifest, out, target)
    assert (target / "installed" / "a.bin").read_bytes() == \
        (data / "input" / "a.bin").read_bytes()
    assert (target / "alias" / "b.txt").read_text(encoding="utf-8") == "fixed\n"


def test_registered_paper_result_is_final_system_only():
    payload = json.loads(PAPER_RESULTS_PATH.read_text(encoding="utf-8"))
    assert payload["system"]["candidate_models"] == 3_016_439
    assert payload["system"]["candidate_pool"] == 1_000
    assert payload["metrics"]["gold@10"]["mean"] == 0.3031382167829068
    assert len(payload["recommendation_content_sha256"]) == 3


def test_summary_does_not_promote_informational_mismatch():
    result = summarize([
        {"id": "strict", "ok": True},
        {"id": "historic-code", "ok": False, "severity": "informational"},
    ])
    assert result["failed"] == 1
    assert result["strict_failed"] == 0
    assert result["informational_mismatches"] == 1
