"""Hash-checked multipart release bundles.

GitHub release assets have a per-file size ceiling, while the frozen HNSW and
embedding artifacts are larger.  Bundles are therefore ordinary streaming tar
archives split into independently hashed parts.  No proprietary extraction
tool is required; download, extraction, and verification use Python's standard
library plus the GitHub CLI for authenticated private-repository downloads.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


BLOCK = 8 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(BLOCK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(files: Iterable[dict]) -> str:
    """Digest a logical tree independently of archive ordering and metadata."""
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda row: row["path"].encode("utf-8")):
        line = "%s %d %s\n" % (item["sha256"], item["bytes"], item["path"])
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


class MultipartWriter(io.RawIOBase):
    """A write-only stream that rolls over at an exact byte limit."""

    def __init__(self, directory: Path, stem: str, part_bytes: int):
        if part_bytes <= 0:
            raise ValueError("part_bytes must be positive")
        self.directory = directory
        self.stem = stem
        self.part_bytes = int(part_bytes)
        self._handle: BinaryIO | None = None
        self._part_index = 0
        self._part_size = 0
        self._part_hash: hashlib._Hash | None = None
        self.parts: list[dict] = []
        directory.mkdir(parents=True, exist_ok=True)

    def writable(self) -> bool:
        return True

    def _open_part(self) -> None:
        self._part_index += 1
        name = "%s.part%03d" % (self.stem, self._part_index)
        self._handle = (self.directory / name).open("wb")
        self._part_size = 0
        self._part_hash = hashlib.sha256()

    def _finish_part(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        name = "%s.part%03d" % (self.stem, self._part_index)
        assert self._part_hash is not None
        self.parts.append({
            "name": name,
            "bytes": self._part_size,
            "sha256": self._part_hash.hexdigest(),
        })
        self._handle = None
        self._part_hash = None

    def write(self, payload: bytes | bytearray | memoryview) -> int:
        view = memoryview(payload)
        total = len(view)
        while view:
            if self._handle is None:
                self._open_part()
            room = self.part_bytes - self._part_size
            if room == 0:
                self._finish_part()
                continue
            chunk = view[:room]
            written = self._handle.write(chunk)
            if written <= 0:
                raise OSError("multipart archive write made no progress")
            assert self._part_hash is not None
            self._part_hash.update(chunk[:written])
            self._part_size += written
            view = view[written:]
        return total

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if not self.closed:
            self._finish_part()
        super().close()


class MultipartReader(io.RawIOBase):
    """Concatenate already verified part files into one streaming reader."""

    def __init__(self, paths: list[Path]):
        self.paths = paths
        self._index = -1
        self._handle: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def _next(self) -> bool:
        if self._handle is not None:
            self._handle.close()
        self._index += 1
        if self._index >= len(self.paths):
            self._handle = None
            return False
        self._handle = self.paths[self._index].open("rb")
        return True

    def readinto(self, target: bytearray | memoryview) -> int:
        view = memoryview(target)
        done = 0
        while done < len(view):
            if self._handle is None and not self._next():
                break
            assert self._handle is not None
            n = self._handle.readinto(view[done:])
            if n:
                done += n
            elif not self._next():
                break
        return done

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def _safe_relative(name: str) -> Path:
    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError("unsafe archive member: %r" % name)
    if ":" in posix.parts[0]:
        raise ValueError("drive-qualified archive member: %r" % name)
    return Path(*posix.parts)


def verify_parts(bundle: dict, asset_dir: Path) -> list[Path]:
    paths = []
    for part in bundle["parts"]:
        path = asset_dir / part["name"]
        if not path.is_file():
            raise FileNotFoundError("missing release part: %s" % path)
        if path.stat().st_size != int(part["bytes"]):
            raise ValueError("size mismatch for %s" % path)
        got = sha256_file(path)
        if got != part["sha256"]:
            raise ValueError("SHA-256 mismatch for %s: %s != %s" %
                             (path, got, part["sha256"]))
        paths.append(path)
    return paths


def verify_installed(manifest: dict, data_root: Path, include_copies: bool = True) -> list[str]:
    errors: list[str] = []
    rows = list(manifest["files"])
    if include_copies:
        by_path = {row["path"]: row for row in rows}
        for copy in manifest.get("copies", []):
            source = by_path[copy["from"]]
            rows.append({**source, "path": copy["to"]})
    for row in rows:
        path = data_root / Path(*PurePosixPath(row["path"]).parts)
        if not path.is_file():
            errors.append("missing %s" % path)
            continue
        if path.stat().st_size != int(row["bytes"]):
            errors.append("size %s" % path)
            continue
        if sha256_file(path) != row["sha256"]:
            errors.append("sha256 %s" % path)
    return errors


def _materialize_copies(manifest: dict, data_root: Path, force: bool) -> None:
    for item in manifest.get("copies", []):
        source = data_root / Path(*PurePosixPath(item["from"]).parts)
        target = data_root / Path(*PurePosixPath(item["to"]).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if (target.is_file() and target.stat().st_size == source.stat().st_size
                    and sha256_file(target) == sha256_file(source)):
                continue
            if not force:
                raise FileExistsError("refusing to replace nonmatching %s" % target)
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def extract_bundle(bundle: dict, manifest: dict, asset_dir: Path,
                   data_root: Path, force: bool = False) -> None:
    """Extract into a scoped staging directory, verify, then install files."""
    parts = verify_parts(bundle, asset_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mlf-%s-" % bundle["id"],
                                    dir=data_root))
    try:
        reader = MultipartReader(parts)
        try:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if member.name == ".mlf_bundle_manifest.json":
                        continue
                    rel = _safe_relative(member.name)
                    if member.isdir():
                        (staging / rel).mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise ValueError("links/devices are forbidden in release bundles: %s"
                                         % member.name)
                    target = staging / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("cannot read archive member %s" % member.name)
                    with target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=BLOCK)
        finally:
            reader.close()

        errors = verify_installed(manifest, staging, include_copies=False)
        if errors:
            raise ValueError("staged bundle failed verification:\n" + "\n".join(errors))

        for row in manifest["files"]:
            rel = Path(*PurePosixPath(row["path"]).parts)
            source, target = staging / rel, data_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if (target.is_file() and target.stat().st_size == int(row["bytes"])
                        and sha256_file(target) == row["sha256"]):
                    source.unlink()
                    continue
                if not force:
                    raise FileExistsError("refusing to replace nonmatching %s" % target)
                target.unlink()
            os.replace(source, target)
        _materialize_copies(manifest, data_root, force=force)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    errors = verify_installed(manifest, data_root)
    if errors:
        raise ValueError("installed bundle failed verification:\n" + "\n".join(errors))


def download_parts(repository: str, tag: str, bundle: dict, destination: Path,
                   gh: str = "gh") -> None:
    """Download each missing part through authenticated ``gh``."""
    destination.mkdir(parents=True, exist_ok=True)
    for part in bundle["parts"]:
        target = destination / part["name"]
        if (target.is_file() and target.stat().st_size == int(part["bytes"])
                and sha256_file(target) == part["sha256"]):
            print("[download] verified cache %s" % part["name"], flush=True)
            continue
        command = [gh, "release", "download", tag, "--repo", repository,
                   "--pattern", part["name"], "--dir", os.fspath(destination),
                   "--clobber"]
        print("[download] %s" % part["name"], flush=True)
        subprocess.run(command, check=True)
        if target.stat().st_size != int(part["bytes"]):
            raise ValueError("downloaded size mismatch for %s" % target)
        if sha256_file(target) != part["sha256"]:
            raise ValueError("downloaded SHA-256 mismatch for %s" % target)


@dataclass(frozen=True)
class SourceFile:
    source: Path
    target: str


def _expand_source_entries(spec: dict, repo_root: Path, data_root: Path) -> list[SourceFile]:
    roots = {"repo": repo_root, "data": data_root}
    out: list[SourceFile] = []
    for entry in spec["files"]:
        root = roots[entry["root"]]
        source = root / Path(*PurePosixPath(entry["source"]).parts)
        target = PurePosixPath(entry["target"])
        if source.is_dir():
            patterns = entry.get("include", ["**/*"])
            found: set[Path] = set()
            for pattern in patterns:
                found.update(p for p in source.glob(pattern) if p.is_file())
            for path in sorted(found):
                rel = PurePosixPath(*path.relative_to(source).parts)
                out.append(SourceFile(path, (target / rel).as_posix()))
        elif source.is_file():
            out.append(SourceFile(source, target.as_posix()))
        else:
            raise FileNotFoundError(source)
    targets = [item.target for item in out]
    if len(targets) != len(set(targets)):
        raise ValueError("bundle spec contains duplicate target paths")
    return sorted(out, key=lambda item: item.target.encode("utf-8"))


def build_bundle(bundle_id: str, spec: dict, repo_root: Path, data_root: Path,
                 output_dir: Path, part_bytes: int) -> tuple[dict, dict]:
    """Maintainer helper: build parts and return (asset row, file manifest)."""
    source_files = _expand_source_entries(spec, repo_root, data_root)
    rows = []
    for index, item in enumerate(source_files, 1):
        print("[hash] %s (%d/%d)" % (item.target, index, len(source_files)), flush=True)
        rows.append({"path": item.target, "bytes": item.source.stat().st_size,
                     "sha256": sha256_file(item.source)})
    manifest = {
        "schema_version": 1,
        "bundle": bundle_id,
        "files": rows,
        "copies": spec.get("copies", []),
        "tree_sha256": tree_digest(rows),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    stem = "%s.tar" % bundle_id
    writer = MultipartWriter(output_dir, stem, part_bytes)
    try:
        with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            info = tarfile.TarInfo(".mlf_bundle_manifest.json")
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o444
            archive.addfile(info, io.BytesIO(payload))
            for item in source_files:
                archive.add(os.fspath(item.source), arcname=item.target,
                            recursive=False, filter=_portable_tar_info)
    finally:
        writer.close()
    asset = {
        "id": bundle_id,
        "description": spec["description"],
        "installed_bytes": sum(row["bytes"] for row in rows),
        "tree_sha256": manifest["tree_sha256"],
        "manifest": "repro/manifests/%s.json" % bundle_id,
        "parts": writer.parts,
    }
    return asset, manifest


def _portable_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o444
    return info
