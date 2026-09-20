import argparse
import hashlib
import json
import os
import sys

from scale.pull_corpus import REPOS, data_root


def sha256_and_lines(path: str, count_lines: bool):
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


def verify_one(subdir_root: str) -> tuple[int, int, list]:
    prov_path = os.path.join(subdir_root, "PROVENANCE.json")
    if not os.path.exists(prov_path):
        return 0, 1, [f"MISSING PROVENANCE.json at {subdir_root}"]

    with open(prov_path, encoding="utf-8") as fh:
        prov = json.load(fh)

    raw_dir = os.path.join(subdir_root, "raw")
    ok, bad, failures = 0, 0, []

    for rec in prov["files"]:
        path = os.path.join(raw_dir, rec["name"])
        if not os.path.exists(path):
            bad += 1
            failures.append(f"MISSING  {rec['name']}")
            continue
        digest, size, nlines = sha256_and_lines(
            path, rec["newline_count"] is not None)
        problems = []
        if digest != rec["sha256"]:
            problems.append(f"sha256 {digest[:16]}!={rec['sha256'][:16]}")
        if size != rec["size_bytes"]:
            problems.append(f"size {size}!={rec['size_bytes']}")
        if rec["newline_count"] is not None and nlines != rec["newline_count"]:
            problems.append(f"lines {nlines}!={rec['newline_count']}")
        if problems:
            bad += 1
            failures.append(f"MISMATCH {rec['name']}: {'; '.join(problems)}")
        else:
            ok += 1

    return ok, bad, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="verify an alternate copy (e.g. the backup replica)")
    ap.add_argument("--only", choices=sorted(REPOS), default=None)
    args = ap.parse_args()

    root = args.root or data_root()
    keys = [args.only] if args.only else ["v2", "ckpt", "v1"]

    print(f"verifying under: {root}")
    total_ok, total_bad, all_fail = 0, 0, []
    for k in keys:
        subdir_root = os.path.join(root, REPOS[k]["subdir"])
        if not os.path.isdir(subdir_root):
            print(f"  {k:5s} SKIP (not present at this root)")
            continue
        ok, bad, failures = verify_one(subdir_root)
        total_ok += ok
        total_bad += bad
        all_fail += [f"[{k}] {f}" for f in failures]
        status = "OK" if bad == 0 else "FAIL"
        print(f"  {k:5s} {status}  {ok} verified, {bad} bad")

    print()
    if total_bad:
        print(f"*** VERIFY FAILED: {total_bad} problem(s) ***")
        for f in all_fail:
            print("   ", f)
        print("\nThe frozen snapshot no longer matches PROVENANCE.json.")
        print("Do NOT proceed to build or report numbers. Restore from a "
              "backup replica and re-verify.")
        return 1

    print(f"*** VERIFY OK: {total_ok} files intact -- safe to proceed ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
