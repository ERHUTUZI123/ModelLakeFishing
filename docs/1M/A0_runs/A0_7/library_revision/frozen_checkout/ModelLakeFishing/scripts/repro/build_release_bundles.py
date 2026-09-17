"""Maintainer-only builder for the hash-bound GitHub release bundles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from repro.archive import build_bundle  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", nargs="+", help="bundle ids from bundle_sources.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--part-bytes", type=int, default=1_800_000_000)
    args = parser.parse_args(argv)

    source_spec = json.loads((REPO_ROOT / "repro" / "bundle_sources.json")
                             .read_text(encoding="utf-8"))
    asset_path = REPO_ROOT / "repro" / "assets.json"
    assets = json.loads(asset_path.read_text(encoding="utf-8"))
    output_dir = (args.out if args.out is not None else
                  REPO_ROOT / "release_assets" / assets["release_tag"])
    for bundle_id in args.bundles:
        if bundle_id not in source_spec["bundles"]:
            parser.error("unknown bundle %s" % bundle_id)
        asset, manifest = build_bundle(
            bundle_id, source_spec["bundles"][bundle_id],
            args.repo_root.resolve(), args.data_root.resolve(),
            output_dir.resolve(), args.part_bytes)
        write_json(REPO_ROOT / asset["manifest"], manifest)
        assets["bundles"][bundle_id] = asset
        print("[built] %s: %d files, %d parts, %.3f GiB" % (
            bundle_id, len(manifest["files"]), len(asset["parts"]),
            asset["installed_bytes"] / (1024 ** 3)), flush=True)
    assets["release_state"] = "ASSETS_BUILT_NOT_YET_CONFIRMED_UPLOADED"
    write_json(asset_path, assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
