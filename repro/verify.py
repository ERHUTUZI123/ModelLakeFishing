"""Verification for the final paper result and its frozen inputs."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .archive import sha256_file, tree_digest, verify_installed


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_RESULTS_PATH = REPO_ROOT / "docs" / "PAPER_RESULTS.json"
ASSETS_PATH = REPO_ROOT / "repro" / "assets.json"
VALID_PROFILES = ("final", "inputs", "full")


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pointer(payload: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer."""
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with '/': %s" % pointer)
    value = payload
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _equal(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isfinite(float(expected)) and math.isfinite(float(actual))):
            return actual == expected
        return math.isclose(float(actual), float(expected), rel_tol=0.0,
                            abs_tol=float(tolerance))
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _equal(a, e, tolerance) for a, e in zip(actual, expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _equal(actual[key], expected[key], tolerance) for key in expected)
    return actual == expected


def _all_true(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(_all_true(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_all_true(item) for item in value)
    return value is True


def _result(check_id: str, ok: bool, **details: Any) -> dict[str, Any]:
    return {"id": check_id, "ok": bool(ok), **details}


def verify_assets(profile: str, data_root: Path,
                  assets_path: Path = ASSETS_PATH) -> list[dict]:
    required = {
        "inputs": ("inputs-3m-v1",),
        "final": ("paper-3m-v1",),
        "full": ("inputs-3m-v1",),
    }[profile]
    assets = _json(assets_path)
    checks: list[dict] = []
    for bundle_id in required:
        bundle = assets.get("bundles", {}).get(bundle_id)
        if bundle is None:
            checks.append(_result(
                "bundle:%s:declared" % bundle_id, False,
                error="bundle absent from repro/assets.json"))
            continue
        manifest_path = REPO_ROOT / Path(*bundle["manifest"].split("/"))
        if not manifest_path.is_file():
            checks.append(_result(
                "bundle:%s:manifest" % bundle_id, False,
                path=os.fspath(manifest_path), error="missing manifest"))
            continue
        manifest = _json(manifest_path)
        actual_tree = tree_digest(manifest["files"])
        expected_tree = bundle["tree_sha256"]
        checks.append(_result(
            "bundle:%s:manifest-tree" % bundle_id,
            actual_tree == expected_tree == manifest["tree_sha256"],
            actual=actual_tree, expected=expected_tree))
        errors = verify_installed(manifest, data_root)
        checks.append(_result(
            "bundle:%s:installed-files" % bundle_id, not errors,
            errors=errors[:50], n_errors=len(errors),
            n_files=len(manifest["files"])))
    return checks


def verify_snapshot_shards(profile: str, data_root: Path) -> list[dict]:
    if profile not in ("inputs", "full"):
        return []
    checks: list[dict] = []
    for name in ("candidates_full", "datasets_full"):
        directory = data_root / "data1m" / name
        provenance_path = directory / "PROVENANCE.json"
        if not provenance_path.is_file():
            checks.append(_result(
                "snapshot:%s:provenance" % name, False,
                error="missing %s" % provenance_path))
            continue
        provenance = _json(provenance_path)
        failures = []
        for row in provenance["shards"]:
            path = directory / row["file"]
            if not path.is_file():
                failures.append("missing %s" % row["file"])
            elif path.stat().st_size != int(row["bytes"]):
                failures.append("size %s" % row["file"])
            elif sha256_file(path) != row["sha256"]:
                failures.append("sha256 %s" % row["file"])
        checks.append(_result(
            "snapshot:%s:shards" % name, not failures,
            n_shards=len(provenance["shards"]), errors=failures[:20]))
    return checks


def verify_final_result(profile: str, data_root: Path) -> list[dict]:
    report_path = (data_root / "data1m" / "reproduced" / "final"
                   / "FINAL_RESULTS.json")
    if not report_path.is_file():
        return [_result("paper-result:file", False,
                        path=os.fspath(report_path), error="missing report")]
    try:
        actual = _json(report_path)
        expected = _json(PAPER_RESULTS_PATH)
    except Exception as exc:
        return [_result("paper-result:json", False, error=repr(exc))]

    tolerance = 1e-12 if profile == "final" else 0.01
    checks = [
        _result("paper-result:file", True, path=os.fspath(report_path)),
        _result("paper-result:system",
                _equal(actual.get("system"), expected["system"], 0.0),
                actual=actual.get("system"), expected=expected["system"]),
        _result("paper-result:evaluation",
                _equal(actual.get("evaluation"), expected["evaluation"], 0.0),
                actual=actual.get("evaluation"), expected=expected["evaluation"]),
        _result("paper-result:integrity",
                _all_true(actual.get("integrity", {})),
                actual=actual.get("integrity"), expected="all leaves true"),
    ]

    per_seed = actual.get("per_seed", [])
    expected_queries = expected["evaluation"]["eligible_queries"]
    expected_ef = expected["fixed_search_parameters"]["ef_search"]
    checks.append(_result(
        "paper-result:three-seeds", len(per_seed) == 3,
        actual=len(per_seed), expected=3))
    if len(per_seed) == 3:
        checks.append(_result(
            "paper-result:query-counts",
            [row.get("metrics", {}).get("queries") for row in per_seed]
            == expected_queries,
            actual=[row.get("metrics", {}).get("queries") for row in per_seed],
            expected=expected_queries))
        checks.append(_result(
            "paper-result:search-parameters",
            [row.get("ef_search") for row in per_seed] == expected_ef,
            actual=[row.get("ef_search") for row in per_seed],
            expected=expected_ef))

    for name, expected_metric in expected["metrics"].items():
        actual_metric = actual.get("summary", {}).get(name)
        checks.append(_result(
            "paper-result:metric:%s" % name,
            _equal(actual_metric, expected_metric, tolerance),
            actual=actual_metric, expected=expected_metric,
            tolerance=tolerance))

    if profile == "final" and len(per_seed) == 3:
        expected_hashes = expected["recommendation_content_sha256"]
        actual_hashes = [row.get("recommendations_sha256") for row in per_seed]
        checks.append(_result(
            "paper-result:recommendations", actual_hashes == expected_hashes,
            actual=actual_hashes, expected=expected_hashes))
    return checks


def summarize(checks: Iterable[dict]) -> dict[str, int]:
    rows = list(checks)
    strict = [row for row in rows if row.get("severity", "error") == "error"]
    return {
        "checks": len(rows),
        "passed": sum(bool(row["ok"]) for row in rows),
        "failed": sum(not bool(row["ok"]) for row in rows),
        "strict_failed": sum(not bool(row["ok"]) for row in strict),
        "warnings": sum(not bool(row["ok"]) and row.get("severity") == "warning"
                        for row in rows),
        "informational_mismatches": sum(
            not bool(row["ok"]) and row.get("severity") == "informational"
            for row in rows),
    }


def verify(profile: str, data_root: Path, runs_root: Path,
           output: Path | None = None) -> dict[str, Any]:
    if profile not in VALID_PROFILES:
        raise ValueError("profile must be one of %s" % (VALID_PROFILES,))
    data_root, runs_root = data_root.resolve(), runs_root.resolve()
    checks = verify_assets(profile, data_root)
    checks.extend(verify_snapshot_shards(profile, data_root))
    if profile in ("final", "full"):
        checks.extend(verify_final_result(profile, data_root))
    report = {
        "schema_version": 1,
        "profile": profile,
        "verified_at_unix": time.time(),
        "repository": os.fspath(REPO_ROOT),
        "data_root": os.fspath(data_root),
        "runs_root": os.fspath(runs_root),
        "summary": summarize(checks),
        "checks": checks,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n")
        os.replace(temporary, output)
    return report
