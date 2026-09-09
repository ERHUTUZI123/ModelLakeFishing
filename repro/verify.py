"""Machine-readable verification for the frozen 3M evidence chain.

The verifier deliberately uses only the Python standard library.  The
``archive`` profile can therefore be run immediately after cloning, before any
large artifacts or scientific Python packages are installed.  ``replay`` and
``full`` resolve generated reports below ``MLF_DATA_DIR`` and training
manifests below ``MLF_RUNS_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .archive import sha256_file, tree_digest, verify_installed


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = REPO_ROOT / "repro" / "expected_results.json"
AS_RUN_PATH = REPO_ROOT / "repro" / "as_run_manifest.json"
ASSETS_PATH = REPO_ROOT / "repro" / "assets.json"
VALID_PROFILES = ("archive", "inputs", "replay", "full")


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pointer(payload: Any, pointer: str) -> Any:
    """Resolve RFC 6901 JSON pointers without third-party dependencies."""
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with '/': %s" % pointer)
    value = payload
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(token)]
        else:
            value = value[token]
    return value


def _all_true(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(_all_true(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_all_true(item) for item in value)
    return value is True


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


def _report_path(relative: str, profile: str, data_root: Path,
                 runs_root: Path) -> Path:
    path = Path(*relative.replace("\\", "/").split("/"))
    if profile == "archive":
        return REPO_ROOT / path
    parts = path.parts
    if parts and parts[0] == "data1m":
        return data_root / path
    if parts and parts[0] == "runs":
        return runs_root.joinpath(*parts[1:])
    return REPO_ROOT / path


def _result(check_id: str, ok: bool, **details: Any) -> dict[str, Any]:
    return {"id": check_id, "ok": bool(ok), **details}


def verify_expected(profile: str, data_root: Path, runs_root: Path,
                    expected_path: Path = EXPECTED_PATH) -> tuple[list[dict], dict]:
    spec = _json(expected_path)
    default_tolerance = float(spec.get("numeric_tolerance_default", 1e-12))
    checks: list[dict] = []
    loaded: dict[str, Any] = {}

    for report in spec["reports"]:
        if profile not in report.get("profiles", VALID_PROFILES):
            continue
        relative = report["paths"].get(profile)
        if not relative:
            checks.append(_result(report["id"] + ":path", False,
                                  error="no path declared for profile"))
            continue
        path = _report_path(relative, profile, data_root, runs_root)
        if not path.is_file():
            checks.append(_result(report["id"] + ":file", False,
                                  path=os.fspath(path), error="missing report"))
            continue
        try:
            payload = _json(path)
            loaded[report["id"]] = payload
        except Exception as exc:  # malformed evidence must be reported, not hidden
            checks.append(_result(report["id"] + ":json", False,
                                  path=os.fspath(path), error=repr(exc)))
            continue

        checks.append(_result(report["id"] + ":file", True,
                              path=os.fspath(path)))
        for index, assertion in enumerate(report.get("assertions", [])):
            if profile not in assertion.get("profiles", VALID_PROFILES):
                continue
            check_id = "%s:%03d:%s" % (report["id"], index,
                                         assertion["pointer"])
            try:
                actual = _pointer(payload, assertion["pointer"])
                if "equals" in assertion:
                    tolerance = float(assertion.get(
                        "tolerance_by_profile", {}).get(
                            profile, assertion.get("tolerance", default_tolerance)))
                    ok = _equal(actual, assertion["equals"], tolerance)
                    detail = {"actual": actual, "expected": assertion["equals"],
                              "tolerance": tolerance}
                elif "length" in assertion:
                    ok = len(actual) == int(assertion["length"])
                    detail = {"actual": len(actual),
                              "expected": int(assertion["length"])}
                elif assertion.get("all_true") is True:
                    ok = _all_true(actual)
                    detail = {"actual": actual, "expected": "all leaves true"}
                else:
                    raise ValueError("unsupported assertion %r" % assertion)
                if assertion.get("classification"):
                    detail["classification"] = assertion["classification"]
                checks.append(_result(check_id, ok, **detail))
            except Exception as exc:
                checks.append(_result(check_id, False, error=repr(exc)))

    checks.extend(_derived_checks(profile, loaded,
                                  spec.get("derived_checks", {}),
                                  default_tolerance))
    return checks, loaded


def _derived_checks(profile: str, reports: dict[str, Any], expected: dict,
                    tolerance: float) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, value: float | int) -> None:
        want = expected[name]
        checks.append(_result("derived:" + name, _equal(value, want, tolerance),
                              actual=value, expected=want, tolerance=tolerance))

    y2 = reports.get("y2_final_two_stage")
    if y2:
        seeds = y2["hnsw_summary"]["gold@10"]["per_seed"]
        add("final_mean_from_three_seeds", sum(seeds) / len(seeds))
        add("exact_pool_retention_percent",
            100.0 * y2["decision"]["exact_pool_retention_mean"])
        add("hnsw_retention_percent",
            100.0 * y2["decision"]["hnsw_pool_retention_mean"])
        full_seeds = y2["summary"]["G_full_task"]["gold@10"]["per_seed"]
        add("overall_retention_percent", 100.0 * sum(
            actual / full for actual, full in zip(seeds, full_seeds)) / len(seeds))
        coverage = [y2["per_seed"][str(seed)]["rows"]
                    ["G_exact1000_task"]["gold_in_first_stage@1000"]
                    for seed in range(3)]
        overlap = [y2["per_seed"][str(seed)]["rows"]
                   ["G_exact1000_task"]["full_fused_top10_in_dense_top1000"]
                   for seed in range(3)]
        add("dense_pool_gold_coverage_mean", sum(coverage) / len(coverage))
        add("full_fused_top10_overlap_mean", sum(overlap) / len(overlap))
        x6 = reports.get("x6_training_free_baselines")
        if x6:
            add("final_over_bm25",
                y2["hnsw_summary"]["gold@10"]["mean"] /
                x6["summary"]["L_bm25"]["gold@10"]["mean"])

    graph = reports.get("f5_graph")
    if graph:
        edges = graph["edges"]
        add("edge_total_formula", 2 * edges["trained_on"] +
            edges["similar_to"] + 2 * edges["is_base_of"])
        coverage = reports.get("f15_dataset_card_coverage")
        if coverage:
            counts = coverage.get("card_source_counts", coverage.get("card_source", {}))
            add("dataset_cards_exact_plus_parent",
                counts["hf_card"] + counts["hf_card_via_parent"])
    return checks


def verify_assets(profile: str, data_root: Path,
                  assets_path: Path = ASSETS_PATH) -> list[dict]:
    if profile == "archive":
        return []
    assets = _json(assets_path)
    required = {
        "inputs": ("inputs-3m-v1",),
        "replay": ("eval-3m-v1", "graph-3m-v1"),
        "full": ("inputs-3m-v1",),
    }[profile]
    checks: list[dict] = []
    for bundle_id in required:
        bundle = assets.get("bundles", {}).get(bundle_id)
        if bundle is None:
            checks.append(_result("bundle:%s:declared" % bundle_id, False,
                                  error="bundle absent from repro/assets.json"))
            continue
        manifest_path = REPO_ROOT / Path(*bundle["manifest"].split("/"))
        if not manifest_path.is_file():
            checks.append(_result("bundle:%s:manifest" % bundle_id, False,
                                  path=os.fspath(manifest_path), error="missing"))
            continue
        manifest = _json(manifest_path)
        got_tree = tree_digest(manifest["files"])
        expected_tree = bundle["tree_sha256"]
        checks.append(_result("bundle:%s:manifest-tree" % bundle_id,
                              got_tree == expected_tree == manifest["tree_sha256"],
                              actual=got_tree, expected=expected_tree))
        errors = verify_installed(manifest, data_root)
        checks.append(_result("bundle:%s:installed-files" % bundle_id,
                              not errors, errors=errors[:50],
                              n_errors=len(errors), n_files=len(manifest["files"])))
    return checks


def verify_snapshot_shards(profile: str, data_root: Path) -> list[dict]:
    if profile not in ("inputs", "full"):
        return []
    checks: list[dict] = []
    for name in ("candidates_full", "datasets_full"):
        directory = data_root / "data1m" / name
        provenance_path = directory / "PROVENANCE.json"
        if not provenance_path.is_file():
            checks.append(_result("snapshot:%s:provenance" % name, False,
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
        checks.append(_result("snapshot:%s:shards" % name, not failures,
                              n_shards=len(provenance["shards"]),
                              errors=failures[:20]))
    return checks


def verify_as_run_archive() -> list[dict]:
    manifest = _json(AS_RUN_PATH)
    checks: list[dict] = []
    patch_hashes = []
    for relative in manifest["patches"]:
        path = REPO_ROOT / Path(*relative.split("/"))
        if path.is_file():
            patch_hashes.append(sha256_file(path))
        else:
            patch_hashes.append(None)
    checks.append(_result("as-run:patches", bool(patch_hashes) and
                          all(value == manifest["patch_sha256"]
                              for value in patch_hashes),
                          actual=patch_hashes,
                          expected=manifest["patch_sha256"]))

    # A missing git executable or shallow clone is a warning: reports and
    # patches remain verifiable, but the historical parent cannot be inspected.
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", manifest["base_git_commit"] + "^{commit}"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        checks.append(_result("as-run:base-commit", proc.returncode == 0,
                              severity="warning", expected=manifest["base_git_commit"],
                              stderr=proc.stderr.strip()))
    except OSError as exc:
        checks.append(_result("as-run:base-commit", False, severity="warning",
                              error=repr(exc)))

    for relative, expected_hash in manifest["recorded_code_hashes"].items():
        path = REPO_ROOT / Path(*relative.split("/"))
        got = sha256_file(path) if path.is_file() else None
        eol_equivalent = False
        if path.is_file() and got != expected_hash:
            # Git may materialize a text blob with LF or CRLF.  Treat this
            # transport-only difference as equivalent while retaining the raw
            # byte hash in the report; semantic edits still fail both forms.
            raw = path.read_bytes()
            lf = raw.replace(b"\r\n", b"\n")
            normalized = hashlib.sha256(lf).hexdigest()
            normalized_expected = manifest.get(
                "recorded_code_lf_normalized_hashes", {}).get(relative)
            eol_equivalent = (normalized_expected is not None and
                              normalized == normalized_expected)
        checks.append(_result("as-run:code:" + relative,
                              got == expected_hash or eol_equivalent,
                              severity="informational",
                              actual=got, expected=expected_hash,
                              line_ending_equivalent=eol_equivalent,
                              note="current code may contain documented later extensions"))
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
    checks, _loaded = (verify_expected(profile, data_root, runs_root)
                       if profile != "inputs" else ([], {}))
    checks.extend(verify_assets(profile, data_root))
    checks.extend(verify_snapshot_shards(profile, data_root))
    if profile == "archive":
        checks.extend(verify_as_run_archive())
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
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8", newline="\n")
        os.replace(temporary, output)
    return report
