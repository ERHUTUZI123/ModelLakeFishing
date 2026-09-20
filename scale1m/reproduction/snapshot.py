from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

MANIFEST = Path(__file__).with_name("a0_snapshot.json")
PROFILES = ("full", "train", "replay")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("protocol") != "a0":
        raise ValueError("Expected an A0 snapshot manifest, schema_version=1")
    seen = set()
    for row in payload["files"]:
        name = row["path"]
        if not isinstance(name, str):
            raise ValueError("Unsafe manifest path: " + str(name))
        pure = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name
                or pure.is_absolute() or any(p in ("..", ".") for p in name.split("/"))
                or "" in name.split("/") or name != pure.as_posix()):
            raise ValueError("Unsafe manifest path: " + str(name))
        if name.casefold() in seen:
            raise ValueError("Duplicate manifest path: " + name)
        seen.add(name.casefold())
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid SHA-256: " + name)
        if not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise ValueError("Invalid file size: " + name)
        if not row["profiles"] or not set(row["profiles"]) <= set(PROFILES):
            raise ValueError("Invalid file profiles: " + name)
    return payload


def selected_files(manifest: dict, profile: str) -> list[dict]:
    if profile not in PROFILES:
        raise ValueError("Unknown profile: " + profile)
    rows = [r for r in manifest["files"] if profile in r["profiles"]]
    if not rows:
        raise ValueError("Empty snapshot profile: " + profile)
    return rows


def input_digest(manifest: dict, profile: str) -> str:
    rows = [{key: row[key] for key in ("path", "bytes", "sha256")}
            for row in selected_files(manifest, profile)]
    canonical = json.dumps(sorted(rows, key=lambda row: row["path"]),
                           sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def destination(root: Path, name: str) -> Path:
    root = root.resolve()
    path = root.joinpath(*PurePosixPath(name).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Snapshot destination escapes data root: " + name)
    return path


def source_config(manifest: dict, repo_id=None, revision=None) -> tuple[str, str]:
    source = manifest.get("source", {})
    repo_id = repo_id or source.get("repo_id")
    revision = revision or source.get("revision")
    if not repo_id or not revision:
        raise ValueError(
            "The historical A0 archive has no published Hugging Face repository/commit "
            "configured. Supply --hf-repo OWNER/DATASET --hf-revision FULL_COMMIT_SHA "
            "for an archive containing the manifest's exact files. The live Hub API "
            "cannot recreate the 2026-08-18 hub-wide snapshot. Nothing was downloaded.")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
        raise ValueError("--hf-repo must be OWNER/DATASET")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("--hf-revision must be a full 40-character commit SHA, not main/tag/date")
    return repo_id, revision.lower()


def matches(path: Path, row: dict) -> bool:
    return path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"]


def verify(root: Path, manifest: dict, profile: str) -> dict:
    rows = selected_files(manifest, profile)
    errors = []
    for row in rows:
        path = destination(root, row["path"])
        if not matches(path, row):
            errors.append({"path": row["path"], "error": "missing" if not path.exists() else "size/hash mismatch"})
    return {"protocol": "a0", "profile": profile, "files": len(rows),
            "bytes": sum(r["bytes"] for r in rows), "errors": errors, "ok": not errors}


def download(root: Path, manifest: dict, profile: str, *, repo_id=None,
             revision=None, offline=False, fetch=None) -> dict:
    rows = selected_files(manifest, profile)
    if offline:
        result = verify(root, manifest, profile)
        if not result["ok"]:
            raise ValueError("Offline input verification failed: " + json.dumps(result["errors"]))
        return result
    repo_id, revision = source_config(manifest, repo_id, revision)
    if fetch is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ValueError("Install requirements/download.txt before downloading") from exc
        fetch = hf_hub_download
    for i, row in enumerate(rows, 1):
        target = destination(root, row["path"])
        if target.exists():
            if not matches(target, row):
                raise ValueError("Existing file has different bytes; preserve it and choose an empty data root: " + str(target))
            print(f"[{i}/{len(rows)}] verified {row['path']}", flush=True)
            continue
        print(f"[{i}/{len(rows)}] download {row['path']}", flush=True)
        cached = Path(fetch(repo_id=repo_id, repo_type="dataset", revision=revision,
                            filename=row["path"], cache_dir=str(root / ".hf_cache")))
        if not matches(cached, row):
            raise ValueError("Downloaded file differs from the frozen A0 manifest: " + row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download-part")
        shutil.copyfile(cached, temporary)
        if not matches(temporary, row):
            temporary.unlink()
            raise ValueError("Copy verification failed: " + row["path"])
        os.replace(temporary, target)
    result = verify(root, manifest, profile)
    if not result["ok"]:
        raise ValueError("Installed snapshot verification failed")
    receipt = {**result, "source": {"repo_id": repo_id, "revision": revision, "repo_type": "dataset"}}
    (root / ("A0_DOWNLOAD_" + profile + ".json")).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
