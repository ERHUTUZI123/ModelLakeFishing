import json
import shutil

import numpy as np
import pytest

from scale1m import a0_graph_validation as validation
from scale1m import prepare_a0_graph as preparation
from scale1m.checkpoint import sha256_of
from scale1m.reproduction import preparation as reproduction
from scale1m.tests.test_prepare_a0_graph import tiny_source


@pytest.fixture
def frozen_graph(tmp_path, monkeypatch):
    source, historical, bindings = [tmp_path / name for name in ("source", "reference", "bindings")]
    source_digest = tiny_source(source)
    prepared = preparation.prepare_graph(source, historical,
        expected_source_digest=source_digest, expected_shape=(7, 3))
    bindings.mkdir()
    for name in ("meta.json", preparation.REPAIR_FILE):
        shutil.copy2(historical / name, bindings / name)
    source_meta_sha = sha256_of(source / "meta.json")
    for module in (preparation, validation):
        monkeypatch.setattr(module, "SOURCE_GRAPH_DIGEST", source_digest)
        monkeypatch.setattr(module, "SOURCE_META_SHA256", source_meta_sha)
    monkeypatch.setattr(preparation, "SOURCE_SHAPE", (7, 3))
    monkeypatch.setattr(validation, "PREPARED_GRAPH_DIGEST", prepared["new_graph_digest"])
    monkeypatch.setattr(reproduction, "PREPARED_META_SHA256", sha256_of(bindings / "meta.json"))
    monkeypatch.setattr(reproduction, "FROZEN_REPAIR_SHA256", sha256_of(bindings / preparation.REPAIR_FILE))
    return source, historical, bindings


def test_restores_identical_graph_and_records_fresh_provenance(frozen_graph, tmp_path):
    source, historical, bindings = frozen_graph
    before = {path.name: sha256_of(path) for path in source.iterdir()}
    out = tmp_path / "new-place" / "graph"
    record = reproduction.reconstruct_graph(source, bindings, out)
    assert record["status"] == "PASS"
    assert record["graph_digest"] == validation.PREPARED_GRAPH_DIGEST
    assert validation.verify_a0_graph(out)["status"] == "PASS"
    assert {path.name: sha256_of(path) for path in source.iterdir()} == before
    assert {path.name: sha256_of(path) for path in out.iterdir()} == {
        path.name: sha256_of(path) for path in historical.iterdir()}
    fresh = json.loads((out.parent / "graph.reconstruction.json").read_text(encoding="utf-8"))
    assert fresh["implementation_sha256"] == sha256_of(reproduction.__file__)
    assert "historical reference provenance" in fresh["provenance_scope"]
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        reproduction.reconstruct_graph(source, bindings, out)


@pytest.mark.parametrize("name", ["meta.json", "A0_FEATURE_REPAIR.json"])
def test_rejects_changed_archived_binding(frozen_graph, tmp_path, name):
    source, _, bindings = frozen_graph
    with (bindings / name).open("a", encoding="utf-8") as handle:
        handle.write(" ")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="binding SHA256 mismatch"):
        reproduction.reconstruct_graph(source, bindings, out)
    assert not out.exists()


def test_rejects_changed_source_even_with_rehashed_local_manifest(frozen_graph, tmp_path):
    source, _, bindings = frozen_graph
    x = np.load(source / "x_dataset.npy")
    x[0, 64] += 1
    np.save(source / "x_dataset.npy", x)
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    meta["files"]["x_dataset.npy"] = sha256_of(source / "x_dataset.npy")
    (source / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Source encoder metadata differs"):
        reproduction.reconstruct_graph(source, bindings, out)
    assert not out.exists()


def test_rejects_output_inside_source(frozen_graph):
    source, _, bindings = frozen_graph
    with pytest.raises(ValueError, match="must be separate"):
        reproduction.reconstruct_graph(source, bindings, source / "out")
