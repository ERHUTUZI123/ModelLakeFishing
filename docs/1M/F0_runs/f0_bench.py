import json, os, sys, time
sys.path.insert(0, r"D:\research\model_lake\codes\ModelLakeFishing")
import psutil, torch
from scale1m.graph_store import load_sharded          # imports torch_geometric

PT  = r"D:\research\model_lake\data\data1m\graphs\hgraph_100k.pt"
DIR = r"D:\research\model_lake\data\data1m\graphs\hgraph_100k_sharded"
mode = sys.argv[1]
p = psutil.Process()
base_peak = p.memory_info().peak_wset / 1e6
t0 = time.time()
if mode == "pt":
    ck = torch.load(PT, weights_only=False)
else:
    ck = load_sharded(DIR, mmap=(mode != "copy"))
    if mode == "mmap_touch":
        x = ck["data"]["model"].x
        s = 0.0
        for i in range(0, x.shape[0], 10000):
            s += float(x[i:i+10000].sum())
dt = time.time() - t0
mi = p.memory_info()
print(json.dumps({"mode": mode, "load_seconds": round(dt, 2),
                  "peak_wset_mb": round(mi.peak_wset / 1e6, 1),
                  "peak_over_baseline_mb": round((mi.peak_wset - base_peak * 1e6 / 1e6) / 1e6 if False else (mi.peak_wset / 1e6 - base_peak), 1),
                  "baseline_peak_mb": round(base_peak, 1),
                  "private_mb": round(mi.private / 1e6, 1),
                  "x_shape": list(ck["data"]["model"].x.shape)}))
