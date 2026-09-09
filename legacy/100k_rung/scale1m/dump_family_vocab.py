"""Write a rung's family_vocab.csv out of the graph that carries it.

CLAUDE.md Step 6: "the vocab is the only credential for embedding-row identity;
lose it and the trained family rows become orphans." T4 produces that file for
the 100K rung, but the 12K and 30K rungs are the frozen CORE graph itself --
their 341-row vocab lives inside `xm0_meta`, and no CSV was ever written. The
six T6 runs of 2026-08-11 therefore bound `family_vocab_sha256: null` on those
two rungs (docs/1M/T6more.md P2-1).

Recovering it after the fact is legitimate here, and the reason is worth being
precise about: the vocab is a deterministic function of the graph, and the
graph's sha256 is already bound into every checkpoint and independently
verified. Nothing is being reconstructed from memory. This would NOT be
legitimate for a metric.

    python -m scale1m.dump_family_vocab \
        --graph stage1BuildTransferGraph/hgraph_ml_v2_sub.pt \
        --out   $DATA_ROOT/data1m/feats/12k/family_vocab.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(_HERE), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Column order is fixed by T4's file: any consumer that reads it positionally
# has to see the same two columns in the same order on every rung.
HEADER = ("family", "family_id")


def vocab_rows(vocab: dict) -> list[tuple[str, int]]:
    """Sorted by id, which is the row order of the embedding table itself."""
    rows = sorted(((str(name), int(fid)) for name, fid in vocab.items()),
                  key=lambda r: r[1])
    ids = [fid for _, fid in rows]
    if ids != list(range(len(rows))):
        raise ValueError(
            "family ids are not a contiguous 0..N-1 range; the embedding table "
            "and this file would disagree about what row %d means"
            % next(i for i, fid in enumerate(ids) if i != fid))
    return rows


def write_vocab(vocab: dict, out_path: str) -> tuple[int, str]:
    rows = vocab_rows(vocab)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    # newline="" + \n keeps the file byte-identical on Windows and Linux, so the
    # sha256 recorded in a checkpoint binding means the same thing on both.
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)
    os.replace(tmp, out_path)
    with open(out_path, "rb") as fh:
        return len(rows), hashlib.sha256(fh.read()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, help="a rung graph (.pt)")
    p.add_argument("--out", required=True, help="family_vocab.csv to write")
    args = p.parse_args(argv)

    import torch
    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    xm0 = payload["xm0_meta"]
    vocab = xm0["family_vocab"]
    n, sha = write_vocab(vocab, args.out)

    if n != int(xm0["num_families"]):
        print("[fail] wrote %d rows but xm0_meta says num_families=%d"
              % (n, xm0["num_families"]), file=sys.stderr)
        return 1

    gsha = hashlib.sha256(open(args.graph, "rb").read()).hexdigest()
    print("graph        %s" % args.graph)
    print("graph_sha256 %s" % gsha)
    print("rows         %d (== num_families)" % n)
    print("out          %s" % os.path.abspath(args.out))
    print("sha256       %s" % sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
