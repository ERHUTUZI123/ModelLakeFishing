import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from scale.pull_corpus import data_root

LAKE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "stage1BuildTransferGraph", "artifacts", "modellens_v2_lake")
PY = sys.executable


def content_digest(path: str) -> tuple:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = df.reindex(sorted(df.columns), axis=1)
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(6)
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    h = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    return len(df), h


ARTIFACTS = ["ml_observations.parquet", "ml_dataset_pool.csv",
             "ml_model_intake.csv"]


def main():
    print("=== P1 restore drill ===\n")

    print("[1/2] verify_corpus ...")
    r = subprocess.run([PY, "-m", "scale.verify_corpus", "--only", "v2"],
                       capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
    if r.returncode != 0:
        print("*** DRILL FAILED: corpus verify ***")
        return 1

    print("\n[2/2] intake determinism (re-run from frozen raw/) ...")
    committed = {a: content_digest(os.path.join(LAKE, a)) for a in ARTIFACTS}

    tmp = tempfile.mkdtemp(prefix="ml_intake_drill_")
    try:
        env = dict(os.environ, MLF_INTAKE_OUT=tmp)
        code = (
            "import scale.modellens_intake as m, os;"
            f"m.OUT=r'{tmp}';"
            "import sys; sys.exit(m.main())"
        )
        rr = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if rr.returncode != 0:
            print(rr.stdout[-500:]); print(rr.stderr[-500:])
            print("*** DRILL FAILED: re-run intake errored ***")
            return 1

        ok = True
        for a in ARTIFACTS:
            c = committed[a]
            d = content_digest(os.path.join(tmp, a))
            same = (c == d)
            ok &= same
            print(f"  {a:32s} rows {d[0]:>8,}  {'MATCH' if same else 'DIFFER'}")
        if not ok:
            print("\n*** DRILL FAILED: intake not reproducible ***")
            return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n*** P1 RESTORE DRILL PASSED: corpus intact + intake reproducible ***")
    print("note: off-machine restore descoped by D-6 (single local D: copy).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
