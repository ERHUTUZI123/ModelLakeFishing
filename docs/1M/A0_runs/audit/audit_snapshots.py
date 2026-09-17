"""A0.1 read-only, complete frozen hub-snapshot integrity audit.

Writes only --output (exclusive creation unless explicitly revising this owned
audit report). Does not import project code, crawl,
download, or alter source attributes/content. Expected counts/date come solely
from EVIDENCE_SOURCE_LIBRARY_en.md section 2.1. Provenance and SHARDS metadata
are checked as archived hash/count bindings, never as replacement fact sources.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(path):
    result = path.stat()
    return {
        "size_bytes": result.st_size,
        "mtime_ns": result.st_mtime_ns,
        "ctime_ns": result.st_ctime_ns,
        "inode": result.st_ino,
        "device": result.st_dev,
        "file_attributes": getattr(result, "st_file_attributes", None),
        "is_symlink": path.is_symlink(),
    }


def directory_stats(path):
    return {p.name: stats(p) for p in sorted(path.iterdir()) if p.is_file()}


def id_digest_update(digest, value):
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def metadata_records(blob):
    return blob if isinstance(blob, list) else blob.get("shards", [])


def audit_snapshot(kind, directory, archive, expected):
    result = {
        "kind": kind,
        "directory": str(directory.resolve()),
        "started_at_utc": utc_now(),
        "expected_from_authoritative_document_section_2_1": expected,
        "errors": [],
        "findings": [],
        "metadata": {},
        "shards": [],
    }
    errors = result["errors"]
    if not directory.is_dir():
        errors.append("snapshot_directory_missing")
        result["status"] = "FAIL"
        return result
    before = directory_stats(directory)
    result["directory_files_at_start"] = before
    paths = sorted(directory.glob("*.jsonl.gz"))
    if len(paths) != expected["shards"]:
        errors.append("shard_count_mismatch_authoritative_document")
    metadata_paths = {
        "local_provenance": directory / "PROVENANCE.json",
        "local_shards": directory / "SHARDS.json",
        "local_cursor": directory / "CURSOR.json",
        "archived_provenance": archive,
    }
    metadata_blobs = {}
    for label, path in metadata_paths.items():
        record = {"path": str(path.resolve())}
        result["metadata"][label] = record
        if not path.is_file():
            errors.append(f"metadata_missing:{label}")
            continue
        record["start_stat"] = stats(path)
        record["sha256_before"] = sha256(path)
        try:
            metadata_blobs[label] = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"metadata_parse_error:{label}:{type(exc).__name__}:{exc}")
            continue
        if "provenance" in label:
            blob = metadata_blobs[label]
            record["snapshot_date_utc"] = blob.get("snapshot_date_utc")
            record["declared_total_records"] = blob.get("total_records")
            record["declared_pages"] = blob.get("pages")
            record["declared_duplicates_skipped"] = blob.get("duplicates_skipped")
            record["declared_exhausted_cursor"] = blob.get("exhausted_cursor")
            record["date_matches_authoritative_document"] = (
                blob.get("snapshot_date_utc") == expected["snapshot_date_utc"]
            )
            if not record["date_matches_authoritative_document"]:
                errors.append(f"snapshot_date_mismatch:{label}")
            if blob.get("total_records") != expected["records"]:
                errors.append(f"declared_total_mismatch:{label}")
    bindings = {}
    for label, blob in metadata_blobs.items():
        if label == "local_cursor":
            continue
        rows = metadata_records(blob)
        lookup = {row["file"]: row for row in rows}
        bindings[label] = lookup
        if len(lookup) != len(rows):
            errors.append(f"duplicate_metadata_shard_names:{label}")
        if set(lookup) != {p.name for p in paths}:
            errors.append(f"metadata_shard_membership_mismatch:{label}")

    # Exact sets retain strings, not lossy ID hashes: every record is checked.
    seen_raw = set()
    seen_normalized = set()
    duplicated_normalized_ids = set()
    normalized_id_order_digest = hashlib.sha256()
    raw_id_order_digest = hashlib.sha256()
    totals = {
        "physical_lines": 0,
        "parsed_records": 0,
        "valid_repository_ids": 0,
        "blank_lines": 0,
        "malformed_json_lines": 0,
        "invalid_repository_ids": 0,
        "duplicate_raw_id_occurrences": 0,
        "duplicate_normalized_id_occurrences": 0,
    }
    for number, path in enumerate(paths, 1):
        shard = {"file": path.name, "start_stat": stats(path)}
        result["shards"].append(shard)
        shard["sha256_before_stream"] = sha256(path)
        counters = {key: 0 for key in totals}
        shard_errors = []
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
                for line in stream:
                    counters["physical_lines"] += 1
                    if not line.strip():
                        counters["blank_lines"] += 1
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        counters["malformed_json_lines"] += 1
                        continue
                    counters["parsed_records"] += 1
                    repository_id = row.get("id") if isinstance(row, dict) else None
                    if not isinstance(repository_id, str) or not repository_id.strip():
                        counters["invalid_repository_ids"] += 1
                        continue
                    normalized_id = repository_id.strip().lower()
                    counters["valid_repository_ids"] += 1
                    counters["duplicate_raw_id_occurrences"] += repository_id in seen_raw
                    counters["duplicate_normalized_id_occurrences"] += normalized_id in seen_normalized
                    if normalized_id in seen_normalized:
                        duplicated_normalized_ids.add(normalized_id)
                    seen_raw.add(repository_id)
                    seen_normalized.add(normalized_id)
                    id_digest_update(raw_id_order_digest, repository_id)
                    id_digest_update(normalized_id_order_digest, normalized_id)
        except Exception as exc:
            shard_errors.append(f"gzip_stream_error:{type(exc).__name__}:{exc}")
        shard.update(counters)
        for key, value in counters.items():
            totals[key] += value
        shard["sha256_after_stream"] = sha256(path)
        shard["end_stat"] = stats(path)
        shard["unchanged_during_stream"] = (
            shard["start_stat"] == shard["end_stat"]
            and shard["sha256_before_stream"] == shard["sha256_after_stream"]
        )
        shard["binding_checks"] = {}
        for label, lookup in bindings.items():
            binding = lookup.get(path.name, {})
            checks = {
                "sha256_matches": binding.get("sha256") == shard["sha256_before_stream"],
                "bytes_match": binding.get("bytes") == shard["start_stat"]["size_bytes"],
                "records_match": binding.get("n_records") == counters["parsed_records"],
            }
            shard["binding_checks"][label] = checks
            if not all(checks.values()):
                shard_errors.append(f"binding_mismatch:{label}")
        if not shard["unchanged_during_stream"]:
            shard_errors.append("source_mutated_during_stream")
        for key in ("blank_lines", "malformed_json_lines", "invalid_repository_ids"):
            if counters[key]:
                shard_errors.append(f"invalid_content:{key}")
        shard["errors"] = shard_errors
        errors.extend(f"{path.name}:{error}" for error in shard_errors)
        if number % 10 == 0 or number == len(paths):
            print(json.dumps({"snapshot": kind, "shards_complete": number,
                              "shards_total": len(paths), "records": totals["parsed_records"],
                              "errors": len(errors)}), flush=True)

    result["measured"] = {
        **totals,
        "shards": len(paths),
        "compressed_bytes": sum(shard["start_stat"]["size_bytes"] for shard in result["shards"]),
        "unique_raw_repository_ids": len(seen_raw),
        "unique_normalized_repository_ids": len(seen_normalized),
        "raw_id_order_sha256": raw_id_order_digest.hexdigest(),
        "normalized_id_order_sha256": normalized_id_order_digest.hexdigest(),
        "id_digest_serialization": "Sorted shard filenames, physical record order; each UTF-8 ID prefixed by uint64 little-endian byte length.",
        "normalization": "repository_id.strip().lower()",
    }
    result["document_count_checks"] = {
        "parsed_records_match": totals["parsed_records"] == expected["records"],
        "raw_unique_records_match": len(seen_raw) == expected["records"],
        "shards_match": len(paths) == expected["shards"],
    }
    if not all(result["document_count_checks"].values()):
        errors.append("measured_count_mismatch_authoritative_document")
    if totals["duplicate_raw_id_occurrences"]:
        errors.append("duplicate_raw_repository_ids")
    if duplicated_normalized_ids:
        result["findings"].append("normalized_repository_id_collision")
        result["normalization_collision_scope"] = (
            "Section 2.1 gives dataset repository count, not a normalized-unique dataset count; "
            "raw uniqueness and snapshot byte integrity are evaluated separately."
        )
        collisions = {value: [] for value in sorted(duplicated_normalized_ids)}
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    repository_id = row.get("id")
                    normalized_id = repository_id.strip().lower() if isinstance(repository_id, str) else ""
                    if normalized_id in collisions:
                        collisions[normalized_id].append({
                            "repository_id": repository_id,
                            "shard": path.name,
                            "physical_line_1based": line_number,
                            "createdAt": row.get("createdAt"),
                            "lastModified": row.get("lastModified"),
                        })
        result["normalization_collisions"] = [
            {"normalized_id": value, "normalized_id_sha256": hashlib.sha256(value.encode()).hexdigest(),
             "records": rows, "only_case_differs": len({r["repository_id"].lower() for r in rows}) == 1}
            for value, rows in collisions.items()
        ]
        if kind == "model":
            errors.append("duplicate_normalized_model_ids_conflict_with_section_2_2")

    for label, path in metadata_paths.items():
        record = result["metadata"][label]
        if "sha256_before" not in record:
            continue
        record["sha256_after"] = sha256(path)
        record["end_stat"] = stats(path)
        record["unchanged"] = (
            record["sha256_before"] == record["sha256_after"]
            and record["start_stat"] == record["end_stat"]
        )
        if not record["unchanged"]:
            errors.append(f"metadata_mutated:{label}")
    after = directory_stats(directory)
    result["directory_files_at_end"] = after
    result["directory_membership_and_stats_unchanged"] = before == after
    if before != after:
        errors.append("directory_membership_or_file_stats_changed")
    result["finished_at_utc"] = utc_now()
    result["status"] = "PASS" if not errors else "FAIL"
    return result


def audit_collision_impact(data_root, docs_root, snapshot):
    """Inspect exactly the final node/card table lookup keys used by the matcher."""
    if not snapshot.get("normalization_collisions"):
        return
    import pandas as pd

    paths = [
        data_root / "data1m/rf/canon/dataset_nodes_merged.parquet",
        data_root / "data1m/ladder_rf/full_dataset_ids.parquet",
        data_root / "data1m/datasets_full/dataset_cards_merged.parquet",
        data_root / "data1m/datasets_full/dataset_cards.parquet",
    ]
    matcher_path = docs_root.parents[1] / "scale1m/match_dataset_cards.py"
    impact = {"matcher_code": {"path": str(matcher_path.resolve()), "sha256": sha256(matcher_path)},
              "tables": [], "errors": [], "status": "UNRESOLVED"}
    snapshot["normalization_collision_impact"] = impact
    for path in paths:
        before = {"stat": stats(path), "sha256": sha256(path)}
        frame = pd.read_parquet(path)
        ds = frame["dataset"].fillna("").astype(str).str.strip().str.lower()
        wanted_ids = set(ds)
        wanted_ids |= {d.split("/")[0] for d in wanted_ids if "/" in d}
        wanted_bases = {d.split("/")[-1] for d in wanted_ids}
        checks = []
        for collision in snapshot["normalization_collisions"]:
            normalized_id = collision["normalized_id"]
            basename = normalized_id.split("/")[-1]
            dataset_matches = ds.eq(normalized_id)
            related_dataset_rows = ds.str.split("/").map(lambda parts: basename in parts)
            matched_id_count = 0
            if "matched_id" in frame:
                matched_id_count = int(frame["matched_id"].fillna("").astype(str).str.strip().str.lower().eq(normalized_id).sum())
            checks.append({
                "normalized_id": normalized_id,
                "direct_dataset_matches": int(dataset_matches.sum()),
                "dataset_rows_containing_basename_component": int(related_dataset_rows.sum()),
                "matched_id_matches": matched_id_count,
                "in_matcher_wanted_ids_including_parents": normalized_id in wanted_ids,
                "in_matcher_wanted_bases_including_parents": basename in wanted_bases,
                "enters_matcher_retained_index": normalized_id in wanted_ids or basename in wanted_bases,
            })
        after = {"stat": stats(path), "sha256": sha256(path)}
        record = {"path": str(path.resolve()), "rows": len(frame), "before": before,
                  "after": after, "unchanged": before == after, "collision_checks": checks}
        impact["tables"].append(record)
        if before != after:
            impact["errors"].append(f"impact_input_changed:{path.name}")
        if path.name != "dataset_cards.parquet" and len(frame) != 18729:
            impact["errors"].append(f"final_node_count_mismatch:{path.name}")
    all_absent = all(
        not check["enters_matcher_retained_index"] and check["matched_id_matches"] == 0
        and check["direct_dataset_matches"] == 0
        for table in impact["tables"] for check in table["collision_checks"]
    )
    impact["matcher_code"]["sha256_after"] = sha256(matcher_path)
    if impact["matcher_code"]["sha256"] != impact["matcher_code"]["sha256_after"]:
        impact["errors"].append("matcher_code_changed")
    impact["status"] = "NO_EFFECT_ON_FROZEN_NODES_OR_CARD_MATCHING" if all_absent and not impact["errors"] else "REVIEW_REQUIRED"
    impact["affected_final_rows"] = 0 if impact["status"].startswith("NO_EFFECT") else None
    impact["treatment"] = (
        "Preserve frozen shards, row maps, cards and features unchanged; record the raw snapshot case collision as a historical finding. "
        "No canonicalization or matching rule change is part of A0.1."
    )
    if impact["errors"]:
        snapshot["errors"].extend(impact["errors"])
        snapshot["status"] = "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revise-owned-report", action="store_true", help="Retain the previous owned report inside the corrected report; never changes source inputs.")
    parser.add_argument("--revision-reason", default="")
    args = parser.parse_args()
    previous = None
    if args.output.exists():
        if not args.revise_owned_report or not args.revision_reason:
            raise SystemExit("Refusing to overwrite an existing audit output; provide a new --output path or explicit owned-report revision reason.")
        previous = {"sha256": sha256(args.output), "report": json.loads(args.output.read_text(encoding="utf-8")),
                    "revision_reason": args.revision_reason}
        if previous["report"].get("schema") != "a0.snapshot_audit.v1" or previous["report"].get("stage") != "A0.1":
            raise SystemExit("Existing output is not an owned A0.1 snapshot audit; refusing replacement.")
    if not args.output.parent.is_dir():
        raise SystemExit("Output parent directory must already exist.")
    fact = args.docs_root / "EVIDENCE_SOURCE_LIBRARY_en.md"
    fact_before = {"stat": stats(fact), "sha256": sha256(fact)}
    started = time.perf_counter()
    report = {
        "schema": "a0.snapshot_audit.v1",
        "stage": "A0.1",
        "started_at_utc": utc_now(),
        "scope": "Complete read-only audit of frozen model and dataset hub JSONL-GZIP snapshots; no new crawl or source modification.",
        "authoritative_document": {"path": str(fact.resolve()), "section": "2.1", "before": fact_before},
        "binding_metadata_role": "Archived/local PROVENANCE and SHARDS provide byte-level bindings to validate; do not override authoritative document.",
        "snapshot_date_evidence_limit": "Snapshot date is verified against frozen metadata declarations. This local audit does not independently prove historical API state.",
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "execution": {"argv": sys.argv, "python": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "snapshots": [],
    }
    if previous is not None:
        report["superseded_owned_audit"] = previous
    configs = [
        ("model", "candidates_full", "F1_runs", 3003759, 61),
        ("dataset", "datasets_full", "F15_runs", 1008417, 11),
    ]
    for kind, folder, archive_folder, count, shard_count in configs:
        report["snapshots"].append(audit_snapshot(
            kind,
            args.data_root / "data1m" / folder,
            args.docs_root / archive_folder / "PROVENANCE.json",
            {"records": count, "shards": shard_count, "snapshot_date_utc": "2026-08-18"},
        ))
    for snapshot in report["snapshots"]:
        if snapshot["kind"] == "dataset":
            audit_collision_impact(args.data_root, args.docs_root, snapshot)
    fact_after = {"stat": stats(fact), "sha256": sha256(fact)}
    report["authoritative_document"]["after"] = fact_after
    report["authoritative_document"]["unchanged"] = fact_before == fact_after
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["finished_at_utc"] = utc_now()
    report["status"] = "PASS" if (
        fact_before == fact_after and all(s["status"] == "PASS" for s in report["snapshots"])
    ) else "FAIL"
    with args.output.open("w" if previous is not None else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "errors": {s["kind"]: s["errors"] for s in report["snapshots"]}}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
