"""A0.1 provenance capture only; never writes experiment inputs or code."""
from __future__ import annotations
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "docs/1M/A0_runs"
FROZEN = OUT / "frozen"
AUDIT = OUT / "audit"
EXPECTED_SOURCE = "6c7d883734fb9a05ca99c4a8409e366955c86ba2e644bd5328c9ba696ff344cd"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main() -> None:
    if (AUDIT / "A0_CODE_CAPTURE.json").exists():
        raise RuntimeError("A0.1 baseline already captured; refusing to overwrite it. Executed original script is frozen/capture_a01_state.executed.py.")
    FROZEN.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    source = REPO / "docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md"
    if sha(source) != EXPECTED_SOURCE:
        raise RuntimeError("Authoritative source changed since A0 planning; stop and reconcile")
    frozen = []
    for origin, name in [(source, source.name), (REPO / "docs/1M/A0.md", "A0.before_A01.md")]:
        destination = FROZEN / name
        content = origin.read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite different frozen source: {destination}")
        destination.write_bytes(content)
        frozen.append({"source": str(origin), "snapshot": str(destination), "sha256": sha(destination),
                       "size_bytes": len(content)})
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    (FROZEN / "git_status_before.txt").write_bytes(status)
    (FROZEN / "git_head.txt").write_bytes(git("rev-parse", "HEAD"))
    all_paths = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
                           .decode("utf-8").split("\0")) - {""})
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    source_roots = {"scale1m", "scale", "stage1BuildTransferGraph", "stage2TrainGraphSAGE",
                    "stage3HNSW", "scripts", "repro"}
    suffixes = {".py", ".sh", ".sbatch", ".ps1", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
    entries = []
    selected = []
    for rel in all_paths:
        p = Path(rel)
        path = REPO / p
        if not path.is_file():
            continue
        relevant = ((p.parts[0] in source_roots and p.suffix.lower() in suffixes)
                    or (len(p.parts) == 1 and (p.suffix.lower() in suffixes or p.name.startswith("requirements"))))
        if relevant:
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Source changed during capture: {path}")
            entries.append({"path": rel, "sha256": digest, "bytes": after.st_size,
                            "mtime_ns": after.st_mtime_ns, "untracked_at_capture": rel in untracked})
            selected.append(path)
    archive = FROZEN / "implementation_snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in selected:
            tar.add(path, arcname=path.relative_to(REPO).as_posix(), recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        for entry in entries:
            stream = tar.extractfile(entry["path"])
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != entry["sha256"]:
                raise RuntimeError("Implementation archive verification failed")
    print(f"[capture] archived and verified {len(entries)} implementation files", flush=True)
    patches = []
    excludes = ["--", ".", ":(exclude)docs/1M/A0_runs/**", ":(exclude)docs/1M/A0.1.md"]
    for label, args in [("head_to_worktree", ["diff", "--binary", "HEAD"]),
                        ("head_to_index", ["diff", "--binary", "--cached"]),
                        ("index_to_worktree", ["diff", "--binary"])]:
        path = FROZEN / f"git_{label}.patch.gz"
        process = subprocess.Popen(["git", *args, *excludes], cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        count = 0
        with gzip.open(path, "wb") as handle:
            assert process.stdout is not None
            for block in iter(lambda: process.stdout.read(8 << 20), b""):
                handle.write(block)
                count += len(block)
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.wait() != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))
        patches.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                        "uncompressed_bytes": count, "git_args": args + excludes,
                        "stderr": stderr.decode("utf-8", errors="replace")})
        print(f"[capture] {label}: {count} diff bytes", flush=True)
    extra_untracked = []
    extra_tar = FROZEN / "preexisting_untracked.tar.gz"
    with tarfile.open(extra_tar, "w:gz") as tar:
        for rel in untracked:
            if not rel or rel.startswith("docs/1M/A0_runs/") or rel == "docs/1M/A0.1.md":
                continue
            path = REPO / rel
            if path.is_file():
                digest = sha(path)
                tar.add(path, arcname=rel, recursive=False)
                extra_untracked.append({"path": rel, "sha256": digest, "bytes": path.stat().st_size})
    manifest = {
        "stage": "A0.1", "started_at": started,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": str(REPO), "git_head": git("rev-parse", "HEAD").decode().strip(),
        "source_snapshots": frozen, "status_snapshot": str(FROZEN / "git_status_before.txt"),
        "implementation_files": entries, "implementation_archive": {"path": str(archive), "sha256": sha(archive)},
        "patches": patches, "untracked_archive": {"path": str(extra_tar), "sha256": sha(extra_tar),
                                                     "files": extra_untracked},
        "excluded_from_git_patch": ["docs/1M/A0_runs/**", "docs/1M/A0.1.md"],
        "authority": "EVIDENCE_SOURCE_LIBRARY_en.md only; code capture is implementation evidence",
        "experiment_started": False,
    }
    write_json(AUDIT / "A0_CODE_CAPTURE.json", manifest)
    print(json.dumps({"capture": "complete", "implementation_files": len(entries),
                      "patches": len(patches), "untracked_files_archived": len(extra_untracked)}), flush=True)

if __name__ == "__main__":
    main()
