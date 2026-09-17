"""
pull_corpus.py -- P0 step 1: pull the ModelLens corpus + checkpoint at PINNED
revisions and freeze them into an immutable raw/ layer.

Implements PLAN scale/MODELLENS_HEADTOHEAD_SCALE_PLAN.md §1.4 rules 1-3:
  rule 1  lock the commit SHA, never `main`
  rule 2  raw/ layer written once, then set read-only
  rule 3  sha256 + line anchor recorded into PROVENANCE.json

The corpus lives on the COMPETITOR's HF repo and is outside our control: it can
be updated, re-schema'd, rate-limited or taken down at any time. Everything we
report downstream is only defensible if this snapshot is byte-identical and
self-verifying. Freeze first, build later.

Run (from ModelLakeFishing/):
    .\\.venv\\Scripts\\python.exe -m scale.pull_corpus
    .\\.venv\\Scripts\\python.exe -m scale.pull_corpus --only v2
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

from scale1m.paths import data_root as _portable_data_root

# --- PINNED REVISIONS ------------------------------------------------------
# Resolved 2026-07-22. NEVER replace these with "main": main drifts, SHAs do
# not. If a re-pull yields a different SHA the upstream repo changed and the
# comparison must be re-baselined, not silently continued.
REPOS = {
    "v2": dict(
        repo_id="luisrui/ModelLens-corpus-v2",
        repo_type="dataset",
        revision="57a692ccdb20d84d1c803544f722b3727450c0e8",
        subdir="modellens_v2",
        role="primary corpus (checkpoint was trained on v2)",
    ),
    "v1": dict(
        repo_id="luisrui/ModelLens-corpus-v1",
        repo_type="dataset",
        revision="df6cb242ed54c96ec09c8644916c07ac14bff4e7",
        subdir="modellens_v1",
        role="cleaner-corpus robustness control (not the main line)",
    ),
    "ckpt": dict(
        repo_id="luisrui/ModelLens",
        repo_type="model",
        revision="68fabcb36d03f96b620a5b5f0ae786ce6da3e74f",
        subdir="modellens_ckpt",
        role="competitor baseline weights (archive: may also be taken down)",
    ),
}

SKIP = {".gitattributes"}
TEXT_LINE_COUNT = {".csv", ".json", ".md"}


def data_root() -> str:
    """Return ``MLF_DATA_DIR`` or the repository-local ``data`` directory."""
    return _portable_data_root()


def sha256_and_lines(path: str, count_lines: bool):
    """Stream the file once: sha256 always, raw newline count when useful.

    NOTE the newline count is a *byte-level* anchor, not a record count --
    data.csv has free-text `dataset_desp` with embedded newlines inside quoted
    fields, so lines != rows. The true record count is established separately
    in audit_corpus.py via a real CSV parser. Both are recorded; conflating
    them is exactly the kind of silent miscount this project has been bitten by.
    """
    h = hashlib.sha256()
    nl = 0
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
            if count_lines:
                nl += chunk.count(b"\n")
    return h.hexdigest(), size, (nl if count_lines else None)


def set_readonly(path: str):
    """Rule 2: raw/ is written once then frozen. Best-effort, reported."""
    try:
        os.chmod(path, 0o444)
        return True
    except OSError:
        return False


def pull_one(key: str, spec: dict, root: str, force: bool) -> dict:
    api = HfApi()
    info = api.repo_info(spec["repo_id"], repo_type=spec["repo_type"],
                         revision=spec["revision"], files_metadata=True)

    # Guard: the pinned SHA must be what the hub actually served us.
    if info.sha != spec["revision"]:
        raise RuntimeError(
            f"{key}: pinned revision {spec['revision']} but hub returned "
            f"{info.sha} -- refusing to continue")

    raw_dir = os.path.join(root, spec["subdir"], "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = []
    for sib in info.siblings:
        name = sib.rfilename
        if name in SKIP:
            continue
        dest = os.path.join(raw_dir, name)
        if os.path.exists(dest) and not force:
            print(f"  [skip-exists] {name}")
        else:
            if os.path.exists(dest):
                os.chmod(dest, 0o644)  # unfreeze before overwrite
            print(f"  [get] {name} ({(sib.size or 0)/1e6:.1f} MB) ...", flush=True)
            hf_hub_download(
                repo_id=spec["repo_id"], filename=name,
                repo_type=spec["repo_type"], revision=spec["revision"],
                local_dir=raw_dir,
            )

        ext = os.path.splitext(name)[1].lower()
        digest, size, nlines = sha256_and_lines(dest, ext in TEXT_LINE_COUNT)
        ro = set_readonly(dest)
        files.append(dict(name=name, sha256=digest, size_bytes=size,
                          newline_count=nlines, hub_size=sib.size,
                          readonly=ro))
        print(f"      sha256={digest[:16]}... size={size} lines={nlines} ro={ro}")

    prov = dict(
        key=key,
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        role=spec["role"],
        downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        raw_dir=raw_dir,
        n_files=len(files),
        total_bytes=sum(f["size_bytes"] for f in files),
        files=files,
        note=("newline_count is a byte-level anchor, NOT a record count; "
              "see audit_corpus.py for parsed row/model/dataset counts"),
    )
    prov_path = os.path.join(root, spec["subdir"], "PROVENANCE.json")
    if os.path.exists(prov_path):
        os.chmod(prov_path, 0o644)
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
    print(f"  -> PROVENANCE.json written ({prov['total_bytes']/1e6:.1f} MB total)")
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(REPOS), default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is already present")
    args = ap.parse_args()

    root = data_root()
    os.makedirs(root, exist_ok=True)
    print(f"data root: {root}")

    keys = [args.only] if args.only else ["v2", "ckpt", "v1"]
    out = {}
    for k in keys:
        print(f"\n=== {k}: {REPOS[k]['repo_id']} @ {REPOS[k]['revision'][:12]} ===")
        out[k] = pull_one(k, REPOS[k], root, args.force)

    print("\n=== SUMMARY ===")
    for k, p in out.items():
        print(f"{k:5s} {p['n_files']} files  {p['total_bytes']/1e6:9.1f} MB  "
              f"-> {p['raw_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
