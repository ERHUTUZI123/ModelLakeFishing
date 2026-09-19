"""Collect the current public Hub metadata lake with durable page checkpoints.

This is an acquisition interval, not an atomic or historical Hub snapshot. Each
validated API page becomes an immutable gzip shard. The cursor is committed only
after its shard is durable; an interrupted, uncommitted page is safely refetched.
No weights, dataset examples, private repositories, or author-local data are used.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse

from scale1m import hf_crawl as models
from scale1m import hf_crawl_datasets as datasets

PROTOCOL = "live-hf"
SCHEMA_VERSION = 1
KINDS = ("models", "datasets")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError("Live metadata output must not be a symbolic link")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _record(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size, "sha256": models.sha256_of(str(path))}


def _spec(kind: str, page_size: int) -> dict:
    return {"endpoint": "https://huggingface.co/api/" + kind,
            "expand": list(models.EXPAND_V2 if kind == "models" else
                           (*datasets.EXPAND, "private")),
            "sort": "createdAt", "direction": "-1", "page_size": page_size,
            "public_only": True, "population_limit": None}


def _url(spec: dict) -> str:
    pairs = [("limit", spec["page_size"]), ("sort", spec["sort"]),
             ("direction", spec["direction"])]
    pairs += [("expand[]", field) for field in spec["expand"]]
    return spec["endpoint"] + "?" + urllib.parse.urlencode(pairs)


def _validate_url(url: str, endpoint: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    target = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
            or parsed.path != target.path or parsed.fragment):
        raise ValueError("Unexpected HF pagination destination; refusing request")


class _Session:
    """Prevent a pagination/redirect response from forwarding an HF token away."""

    def __init__(self, endpoint: str, token: str | None):
        self.endpoint = endpoint
        self.session = models.make_session(token)

    def get(self, url: str, timeout: float):
        _validate_url(url, self.endpoint)
        response = self.session.get(url, timeout=timeout, allow_redirects=False)
        if 300 <= response.status_code < 400:
            raise ValueError("Unexpected redirect from the HF listing API")
        if response.status_code == 200:
            # A malformed next Link must not silently turn a partial crawl into
            # a completed lake. The public API currently emits absolute links.
            link = response.headers.get("Link", "")
            if link:
                pieces = re.split(r",\s*(?=<)", link)
                for piece in pieces:
                    match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel="([^"]+)"\s*', piece)
                    if match is None:
                        raise ValueError("Unrecognized HF pagination Link header")
                    if match.group(2) == "next":
                        _validate_url(match.group(1), self.endpoint)
        return response

    def close(self):
        self.session.close()


def _validate_records(records) -> None:
    if not isinstance(records, list):
        raise ValueError("HF listing schema changed: expected a JSON array")
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                or not record["id"].strip() or not isinstance(record.get("private"), bool)):
            raise ValueError("HF listing schema changed: expected id and expanded private flag")


def _read_shards(directory: Path, state: dict) -> set[str]:
    seen: set[str] = set()
    rows = 0
    for i, shard in enumerate(state["shards"]):
        if shard["file"] != models.shard_stem(i) + ".jsonl.gz":
            raise ValueError("Noncontiguous/unsafe live shard manifest")
        path = directory / shard["file"]
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Live shard escapes its collection directory")
        if (not path.is_file() or path.stat().st_size != shard["bytes"]
                or models.sha256_of(str(path)) != shard["sha256"]):
            raise ValueError("Missing or modified live shard: " + str(path))
        shard_rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                ident = record.get("id")
                if not isinstance(ident, str) or not ident or ident in seen:
                    raise ValueError("Missing or duplicate ID in committed live shards")
                seen.add(ident)
                shard_rows += 1
        if shard_rows != shard["n_records"]:
            raise ValueError("Live shard row count mismatch")
        rows += shard_rows
    if rows != state["n_written"]:
        raise ValueError("Live cursor/shards row count mismatch")
    return seen


def _state(directory: Path, spec: dict) -> dict:
    checkpoint = directory / "CURSOR.json"
    if checkpoint.exists():
        state = _json(checkpoint)
        if (not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION or state.get("spec") != spec
                or state.get("protocol") != PROTOCOL):
            raise ValueError("Existing live collection uses a different crawl configuration")
        return state
    if directory.exists() and any(directory.iterdir()):
        # An initial CURSOR.json.tmp can be left by interruption before the
        # first checkpoint replacement. No data page has been committed yet.
        if set(p.name for p in directory.iterdir()) != {"CURSOR.json.tmp"}:
            raise ValueError("Nonempty collection has no live cursor: " + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "spec": spec,
             "started_at": models.utcnow(), "finished_at": None,
             "next_url": _url(spec), "n_written": 0, "pages": 0, "shards": [],
             "stats": models.new_stats(), "exhausted_cursor": False,
             "visited_urls": []}
    _atomic(checkpoint, state)
    return state


def _collect(directory: Path, kind: str, *, token, timeout, max_retries, page_size) -> dict:
    spec = _spec(kind, page_size)
    state = _state(directory, spec)
    seen = _read_shards(directory, state)
    # The last downloaded page may have reached disk immediately before an
    # interruption of the cursor replacement. It is not committed input yet.
    next_name = models.shard_stem(len(state["shards"])) + ".jsonl.gz"
    committed = {row["file"] for row in state["shards"]}
    for path in directory.glob("hf_models_*"):
        if path.name in committed:
            continue
        if (path.name not in (next_name, next_name + ".part") or path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())):
            raise ValueError("Unexpected uncommitted file in live collection: " + str(path))
        path.unlink()
    session = _Session(spec["endpoint"], token)
    if state["next_url"]:
        state["stats"]["authenticated"] = bool(token)
    trim = (lambda row: models.trim(row, v2=True)) if kind == "models" else datasets.trim
    try:
        while state["next_url"]:
            current = state["next_url"]
            _validate_url(current, spec["endpoint"])
            records, following = models.fetch(session, current, timeout, max_retries,
                                               state["stats"], 25)
            _validate_records(records)
            if following:
                _validate_url(following, spec["endpoint"])
                if following == current or following in state["visited_urls"]:
                    raise ValueError("HF pagination cursor repeated; collection is incomplete")
            if not records and following:
                raise ValueError("HF returned an empty page with a next cursor")
            page_rows = []
            for record in records:
                if record["private"]:
                    state["stats"]["private_skipped"] = state["stats"].get("private_skipped", 0) + 1
                    continue
                if record["id"] in seen:
                    state["stats"]["duplicates_skipped"] += 1
                    continue
                seen.add(record["id"])
                page_rows.append(trim(record))
            # One page is one transaction. If interrupted before CURSOR.json
            # changes, only this deterministic next shard may be overwritten.
            if page_rows:
                path = directory / (models.shard_stem(len(state["shards"])) + ".jsonl.gz")
                part = path.with_name(path.name + ".part")
                with part.open("wb") as output:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
                        for row in page_rows:
                            stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(part, path)
                state["shards"].append({"file": path.name, "n_records": len(page_rows),
                                        "bytes": path.stat().st_size,
                                        "sha256": models.sha256_of(str(path))})
            state["n_written"] += len(page_rows)
            state["pages"] += 1
            state["stats"]["pages"] = state["pages"]
            state["visited_urls"].append(current)
            state["next_url"] = following
            state["exhausted_cursor"] = not bool(following)
            state["finished_at"] = models.utcnow() if not following else None
            _atomic(directory / "CURSOR.json", state)
            if state["pages"] % 25 == 0 or not following:
                print(f"[{kind}] {state['n_written']:,} public records; {state['pages']:,} pages"
                      + ("; cursor exhausted" if not following else ""), flush=True)
    finally:
        session.close()
    provenance = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, **spec,
                  "artifact": "current public HF " + kind + " metadata",
                  "snapshot_window_utc": {"crawl_started_at": state["started_at"],
                                          "crawl_finished_at": state["finished_at"]},
                  "snapshot_is_atomic": False, "total_records": state["n_written"],
                  "pages": state["pages"], "exhausted_cursor": state["exhausted_cursor"],
                  "shards": state["shards"], "stats": state["stats"]}
    # Derived indexes are reproducible from the durable cursor after a crash.
    _atomic(directory / "SHARDS.json", state["shards"])
    _atomic(directory / "PROVENANCE.json", provenance)
    return provenance


def _inspect(root: Path) -> dict:
    errors, files, counts, windows = [], [], {}, []
    for kind in KINDS:
        directory = root / kind
        try:
            if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
                raise ValueError("Live collection directory escapes the snapshot root")
            state = _json(directory / "CURSOR.json")
            spec = _spec(kind, state["spec"]["page_size"])
            if (state.get("protocol") != PROTOCOL or state.get("schema_version") != SCHEMA_VERSION
                    or state["spec"] != spec):
                raise ValueError("Unexpected live crawl schema/configuration")
            if not state["exhausted_cursor"] or state["next_url"] or not state["finished_at"]:
                raise ValueError("Crawl is incomplete; the API cursor has not been exhausted")
            if state["n_written"] < 1:
                raise ValueError("Public collection is empty")
            _read_shards(directory, state)
            listed = {row["file"] for row in state["shards"]}
            actual = {path.name for path in directory.glob("hf_models_*")}
            if listed != actual:
                raise ValueError("Uncommitted or unlisted raw shards in collection")
            provenance = _json(directory / "PROVENANCE.json")
            if (_json(directory / "SHARDS.json") != state["shards"]
                    or provenance["shards"] != state["shards"]
                    or provenance["total_records"] != state["n_written"]
                    or provenance["exhausted_cursor"] is not True):
                raise ValueError("Live provenance/shard index differs from cursor")
            if (len(state["visited_urls"]) != state["pages"]
                    or len(set(state["visited_urls"])) != state["pages"]):
                raise ValueError("Invalid or repeated committed pagination history")
            for url in state["visited_urls"]:
                _validate_url(url, spec["endpoint"])
            files += [_record(directory / name, root) for name in
                      [*sorted(listed), "CURSOR.json", "SHARDS.json", "PROVENANCE.json"]]
            counts[kind] = state["n_written"]
            windows.append(provenance["snapshot_window_utc"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append({"collection": kind, "error": str(exc)})
    interval = ({"started_at": min(w["crawl_started_at"] for w in windows),
                 "finished_at": max(w["crawl_finished_at"] for w in windows)} if windows else None)
    return {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "ok": not errors,
            "counts": counts, "files": sorted(files, key=lambda row: row["path"]),
            "bytes": sum(row["bytes"] for row in files), "errors": errors,
            "acquisition_interval_utc": interval, "snapshot_is_atomic": False,
            "both_cursors_exhausted": not errors}


def input_digest(report: dict) -> str:
    """Stable input binding, excluding verification timestamps/local paths."""
    if not report.get("ok"):
        raise ValueError("Cannot bind incomplete or modified live inputs")
    payload = json.dumps(sorted(report["files"], key=lambda row: row["path"]),
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify(root: Path) -> dict:
    root = Path(root).resolve()
    report = _inspect(root)
    try:
        manifest = _json(root / "LIVE_SNAPSHOT.json")
        if (not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL or manifest.get("files") != report["files"]
                or manifest.get("counts") != report["counts"] or manifest.get("ok") is not True
                or manifest.get("acquisition_interval_utc") != report["acquisition_interval_utc"]):
            raise ValueError("Live input manifest differs from the collected files")
    except (OSError, ValueError) as exc:
        report["errors"].append({"collection": "manifest", "error": str(exc)})
    report["ok"] = not report["errors"]
    return report


def download(root: Path, *, token: str | None = None, timeout: float = 60,
             max_retries: int = 8, page_size: int = 1000) -> dict:
    """Download/resume both complete public listing streams into ``root``.

    A completed collection is immutable and verified without contacting HF.
    To collect newer metadata, use a new root (the CLI's ``--snapshot-id``).
    """
    root = Path(root).resolve()
    if not 1 <= page_size <= models.PAGE_LIMIT_MAX or max_retries < 0 or timeout <= 0:
        raise ValueError("Invalid live API page size, retries, or timeout")
    if (root / "LIVE_SNAPSHOT.json").exists():
        report = verify(root)
        if not report["ok"]:
            raise ValueError("Existing live snapshot is damaged: " + json.dumps(report["errors"]))
        return report
    root.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        directory = root / kind
        if directory.is_symlink() or not directory.resolve().is_relative_to(root):
            raise ValueError("Live collection directory escapes the snapshot root")
        _collect(directory, kind, token=token or os.environ.get("HF_TOKEN"),
                 timeout=timeout, max_retries=max_retries, page_size=page_size)
    report = _inspect(root)
    if not report["ok"]:
        raise ValueError("Live collection verification failed: " + json.dumps(report["errors"]))
    _atomic(root / "LIVE_SNAPSHOT.json", report)
    return report
