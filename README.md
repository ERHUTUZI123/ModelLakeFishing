# ModelLakeFishing

ModelLakeFishing retrieves suitable models for a target dataset--task query from a lake of **3,016,439 candidate models**. The evaluated online path is:

```text
model--dataset evidence graph
    -> graph-trained 128-dimensional embeddings
    -> inner-product HNSW top 1,000
    -> split-safe task-prior reranking
    -> top 10 recommended models
```

Across three root-disjoint held-out splits, the final system obtains `gold@10` of **0.327913**, **0.338783**, and **0.242718**, with a mean of **0.303138**. `gold@10` is one when the empirically best eligible model for a query appears among the ten returned models. The registered values and final ranking hashes are in [`docs/PAPER_RESULTS.json`](docs/PAPER_RESULTS.json).

## Reproduce the final paper result

This is the recommended reproduction. It downloads the frozen artifacts needed by the evaluated online path, generates the top-10 recommendations for all three splits, recomputes the final metrics, and checks the results against the registered values. It runs on CPU and does not train a model.

You need:

- Git and the [GitHub CLI](https://cli.github.com/), authenticated to an account with access to this repository;
- CPython 3.11;
- a 64-bit Windows or Linux machine with at least 16 GiB RAM;
- approximately 24 GiB free disk during download and verified extraction.

Clone the frozen revision:

```bash
gh auth login
gh repo clone ERHUTUZI123/ModelLakeFishing
cd ModelLakeFishing
git checkout paper-results-v1
```

Create the environment on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements\final.txt
.venv\Scripts\python.exe -m scale1m.reproduce doctor --profile final
```

Or on Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/final.txt
.venv/bin/python -m scale1m.reproduce doctor --profile final
```

Download, run, and verify on Windows:

```powershell
.venv\Scripts\python.exe -m scale1m.reproduce download --profile final
.venv\Scripts\python.exe -m scale1m.reproduce run --profile final
.venv\Scripts\python.exe -m scale1m.reproduce verify --profile final
```

Or on Linux:

```bash
.venv/bin/python -m scale1m.reproduce download --profile final
.venv/bin/python -m scale1m.reproduce run --profile final
.venv/bin/python -m scale1m.reproduce verify --profile final
```

A successful run reports:

```text
gold@10 per seed: 0.3279132791 / 0.3387829246 / 0.2427184466
gold@10 mean:     0.3031382168
strict failures:  0
```

The generated files are:

- `data/data1m/reproduced/final/FINAL_RESULTS.json`: protocol, integrity checks, final metrics, and local timing;
- `data/data1m/reproduced/final/recommendations.seed_0.npy` through `seed_2.npy`: query row ID followed by the ten returned model row IDs;
- `data/reproduction_report.final.json`: comparison with the registered paper result.

The release bundle is hash-checked before installation. The evaluator also asserts the 3,016,439-row candidate universe, contiguous and cross-split row mappings, eligible-query counts of 1,476 / 1,101 / 1,545, unique HNSW candidates, and the absence of held-out query roots from every split-specific task prior. The final recommendation content is then checked exactly by SHA-256.

The paper's reference retrieval-stage timing is **0.694 ms p50 / 1.223 ms p95**, measured one query at a time with one HNSW thread after warm-up. It starts from a precomputed query embedding and excludes query encoding, index loading, and index construction. Timing on another machine is descriptive and is not an equality gate.

## Full rebuild from frozen source data

The frozen-result procedure above is the strict reproduction of the published rankings. A full rebuild additionally reconstructs the model lake and evidence graph, trains the three split-specific graph models for 25 epochs, exports held-out embeddings, builds the task priors and HNSW indexes, and finally executes the same single final evaluation.

Use Linux with a compatible NVIDIA driver, a 48 GiB-class CUDA GPU, at least 64 GiB RAM (96 GiB recommended), and at least 120 GiB free disk:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/full-cu130.txt
.venv/bin/python -m scale1m.reproduce doctor --profile full
.venv/bin/python -m scale1m.reproduce download --profile full
.venv/bin/python -m scale1m.reproduce verify --profile inputs
.venv/bin/python -m scale1m.reproduce run --profile full --device cuda
.venv/bin/python -m scale1m.reproduce verify --profile full
```

The input download is the fixed 2026-08-18 snapshot. The live Hugging Face API is not queried. Because GPU training and multithreaded index construction are not bitwise portable, the full-rebuild metric gate uses an absolute tolerance of 0.01; the frozen-result procedure above remains the exact ranking-level check.

Set `MLF_DATA_DIR` to place downloaded data and generated outputs outside the repository, and `MLF_RUNS_DIR` to relocate training runs. The command-line options `--data-root` and `--runs-root` take precedence.

## Preview the 3D model lake

[`lake3d/lake3d_offline.html`](lake3d/lake3d_offline.html) is self-contained. From the repository root, start a local server:

```bash
python -m http.server 8000 --directory lake3d
```

Then open `http://127.0.0.1:8000/lake3d_offline.html`. From another command line:

```powershell
# Windows PowerShell
Start-Process "http://127.0.0.1:8000/lake3d_offline.html"
```

```bash
# macOS
open http://127.0.0.1:8000/lake3d_offline.html

# Linux
xdg-open http://127.0.0.1:8000/lake3d_offline.html
```

It can also be opened directly:

```powershell
Start-Process (Resolve-Path ".\lake3d\lake3d_offline.html")
```

```bash
# macOS
open lake3d/lake3d_offline.html

# Linux
xdg-open lake3d/lake3d_offline.html
```

## Relevant code

| Path | Purpose |
|---|---|
| `scale1m/eval_final.py` | Final HNSW retrieval, split-safe reranking, metrics, and ranking artifacts |
| `scale1m/reproduce.py` | Stable command-line entry point |
| `repro/` | Hash-checked downloader, execution orchestration, and final-result verification |
| `stage1BuildTransferGraph/` | Feature encoders and graph-construction helpers |
| `stage2TrainGraphSAGE/` | Heterogeneous graph encoder, objectives, sampling, and inference |
| `stage3HNSW/` | Split-safe task-prior construction |

## Troubleshooting

- A release download returning 404 usually means the GitHub account lacks repository access or `gh auth login` has not been completed.
- If `hnswlib` has no wheel for the platform, install a local C++ compiler or use the supported CPython 3.11 environment.
- A completed run is skipped safely. Add `--force` to the `run` command only when the final report should be regenerated.
- Use `download --offline --profile final` to verify and extract parts already present under `data/.downloads/paper-results-v1` without network access.
