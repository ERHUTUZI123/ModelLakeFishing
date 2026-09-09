# ModelLakeFishing

ModelLakeFishing retrieves models for a target dataset--task query from a lake of **3,016,439 candidates**. The evaluated online path is:

```text
model--dataset evidence graph
    -> graph-trained 128-dimensional dense embeddings
    -> inner-product HNSW top 1,000
    -> split-safe task-evidence reranking
    -> top 10 recommended models
```

Across three root-disjoint held-out splits, this system obtains `gold@10 = 0.327913 / 0.338783 / 0.242718` (mean `0.303138`). Here, `gold@10` asks whether the empirically best model for an eligible dataset--task query is present in the ten returned models. The complete evidence registry is [`docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md`](docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md).

## Choose a reproduction track

| Track | What it verifies | Frozen download | Training | Recommended machine |
|---|---|---:|---|---|
| Archive audit | Committed reports, provenance, registered numbers, and derived identities | none | no | any CPU |
| Artifact replay | X5/X6/Y2/Y4 from frozen graph, embeddings, exact pools, and HNSW indexes | 16.880 GiB installed | no | 30+ GiB free disk; CUDA recommended for X6 |
| Full rebuild | Frozen snapshots through canonicalization, graph construction, three-seed training, export, and evaluation | 0.396 GiB input; 120+ GiB workspace | yes | Linux, 48 GiB-class CUDA GPU, 64 GiB RAM minimum (96 GiB recommended) |

The archive audit is the fastest way to check every fixed result. Artifact replay independently recomputes the final evaluation results without training. A full rebuild is a configuration-equivalent repeat of the complete experiment and is not expected to be bitwise identical across GPU stacks.

## 1. Obtain the frozen revision

This repository and its release assets are currently private. Your GitHub account must have repository access, and the GitHub CLI must be authenticated before it can download the release assets. Browser login alone does not authenticate the CLI.

```bash
gh auth login
gh repo clone ERHUTUZI123/ModelLakeFishing
cd ModelLakeFishing
git checkout 3m-evidence-v1
```

All commands below run from the repository root. Use the fixed tag, not a moving `main` branch.

## 2. Create the environment

The supported interpreter is CPython 3.11; the original X4 training jobs used Python 3.11.4.

PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements\replay.txt
.venv\Scripts\python.exe -m scale1m.reproduce doctor --profile replay
```

Bash (Linux or macOS):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/replay.txt
.venv/bin/python -m scale1m.reproduce doctor --profile replay
```

For archive-only verification, the standard library is sufficient. For a full Linux/CUDA rebuild, install `requirements/full-cu130.txt` instead and run `doctor --profile full`. Do not use the historical root `requirements.txt` for the fixed 3M reproduction.

## 3. Archive audit: verify all registered evidence

PowerShell:

```powershell
.venv\Scripts\python.exe -m scale1m.reproduce verify --profile archive
```

Bash:

```bash
.venv/bin/python -m scale1m.reproduce verify --profile archive
```

This performs no training and downloads no large artifacts. It validates the committed evidence reports against [`repro/expected_results.json`](repro/expected_results.json), checks recorded provenance and code hashes where available, evaluates derived numerical identities, writes `data/reproduction_report.archive.json`, and returns a nonzero exit code on a strict failure.

## 4. Artifact replay: recompute the final evaluations

PowerShell:

```powershell
.venv\Scripts\python.exe -m scale1m.reproduce download --profile replay
.venv\Scripts\python.exe -m scale1m.reproduce run --profile replay --device cuda
.venv\Scripts\python.exe -m scale1m.reproduce verify --profile replay
```

Bash:

```bash
.venv/bin/python -m scale1m.reproduce download --profile replay
.venv/bin/python -m scale1m.reproduce run --profile replay --device cuda
.venv/bin/python -m scale1m.reproduce verify --profile replay
```

`download` verifies every multipart release asset before extraction and every installed file against its manifest. `run` recomputes X5 eligibility and dense retrieval, X6 training-free baselines, Y2 exact/HNSW retrieval, and the Y4 candidate-pool sensitivity curve. Outputs go to `data/data1m/reproduced/`; completed stages are skipped safely on a resumed run. `verify` is the final acceptance gate and writes `data/reproduction_report.replay.json`.

X6 scans 3,016,439 candidates for each distinct query. A CUDA device is strongly recommended. The Y2/Y4 frozen-pool and HNSW stages can also be run on CPU with `--device cpu`. To inspect or restrict the execution graph:

```powershell
.venv\Scripts\python.exe -m scale1m.reproduce run --profile replay --dry-run
.venv\Scripts\python.exe -m scale1m.reproduce run --profile replay --only y2 y4 --device cpu
```

The installed replay bundles are:

| Bundle | Installed size | Contents |
|---|---:|---|
| `eval-3m-v1` | 11.409 GiB | three-seed evaluation embeddings, row maps, gold candidates, split-safe priors, exact pools, and HNSW indexes |
| `graph-3m-v1` | 5.471 GiB | frozen 3M evidence graph and baseline sidecars |

Set `MLF_DATA_DIR` to place downloads and outputs outside the repository, and `MLF_RUNS_DIR` to relocate training runs. Command-line `--data-root` and `--runs-root` take precedence.

## 5. Full rebuild

Use a Linux x86-64 machine with a compatible NVIDIA driver. The canonical package set is PyTorch `2.12.0+cu130`, PyG `2.7.0`, and `pyg_lib=true`; the original jobs ran on an NVIDIA L40S and observed approximately 37.95 GiB peak GPU allocation.

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

The orchestrator installs the frozen 2026-08-18 Hub snapshots and five historical graphs, then runs canonicalization, evidence merge, dataset-card matching, ladder and feature construction, graph construction, split audit, three 25-epoch training/export jobs, split-safe prior construction, and the same X5/X6/Y2/Y4 evaluations. Re-crawling the live Hugging Face API creates a new dataset snapshot and therefore is not part of reproducing the fixed results.

## Expected high-level results

| Evidence | Registered result |
|---|---:|
| Model snapshot | 3,003,759 repositories; 61 shards; 3,004 pages |
| Dataset snapshot | 1,008,417 repositories; 11 shards; 1,009 pages |
| Candidate closure | 3,016,439 models, including 12,680 historical-only models |
| Evidence graph | 18,729 dataset--task nodes; 247,803 supervision edges; 2,588,316 typed directed edges |
| Eligible queries (seeds 0/1/2) | 1,476 / 1,101 / 1,545 |
| Graph-trained dense retriever, mean gold@10 | 0.142667 |
| BM25 / frozen MiniLM / popularity, mean gold@10 | 0.101630 / 0.016693 / 0 |
| Task prior alone, mean gold@10 | 0.294307 |
| Exact full-pool fusion, mean gold@10 | 0.321603 |
| Exact top-1,000 fusion, mean gold@10 | 0.303364 |
| HNSW top-1,000 fusion, mean gold@10 | 0.303138 |
| HNSW recall@1,000 (seeds 0/1/2) | 0.992264 / 0.994062 / 0.993192 |
| Y4 K=1k/2k/5k/10k, mean gold@10 | 0.303138 / 0.310078 / 0.315449 / 0.317404 |

The deployed/evaluated system remains at `K=1000`; larger Y4 pools are sensitivity measurements, not intermediate system versions.

## What counts as reproduced

- **Archive audit:** registered JSON values and declared file hashes are exact.
- **Artifact replay:** frozen inputs and discrete retrieval outputs are hash-bound; numerical checks use explicit tolerances from the registry.
- **Full rebuild:** canonical data, row maps, and graph contracts are exact where registered. GPU training is configuration-equivalent, because GPU arithmetic and sampling are not bitwise portable. The original X4 runs are tied to base commit `7d3457a...` plus an archived dirty patch; see [`repro/as_run_manifest.json`](repro/as_run_manifest.json).
- **Latency:** wall-clock results are descriptive, machine-specific measurements and are never cross-machine equality gates. The archived `0.694 / 1.223 ms` p50/p95 starts from a precomputed query embedding and excludes query encoding, index loading, and index construction.

The verifier also checks snapshot and graph provenance, contiguous row mappings, cross-seed mapping equality, root-disjoint splits, eligible-query counts, absence of same-root prior leakage, frozen index bindings, HNSW recall, and all fixed metric assertions.

## Preview the 3D model-lake visualization

[`lake3d/lake3d_offline.html`](lake3d/lake3d_offline.html) contains the visualization, Three.js runtime, and plotted data in one offline file. The most reliable preview is a local HTTP server.

Start it from the repository root:

```bash
python -m http.server 8000 --directory lake3d
```

Then open `http://127.0.0.1:8000/lake3d_offline.html`. From another terminal, the command is:

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

Because the page is self-contained, it can also be opened directly:

```powershell
Start-Process (Resolve-Path ".\lake3d\lake3d_offline.html")
```

```bash
# macOS
open lake3d/lake3d_offline.html

# Linux
xdg-open lake3d/lake3d_offline.html
```

## Code map

| Path | Responsibility |
|---|---|
| `scale1m/` | Frozen-snapshot processing, 3M graph construction, training/export, baselines, and final evaluation |
| `repro/` | Release manifests, expected-result registry, downloader, orchestrator, and verifier |
| `stage1BuildTransferGraph/` | Feature encoders and graph-contract helpers used by the 3M path |
| `stage2TrainGraphSAGE/` | Heterogeneous encoder, objectives, sampling, inference, and split logic |
| `stage3HNSW/` | Split-safe task-prior construction |
| `scale/` | Shared exact metrics and HNSW utilities |
| `scripts/repro/` | Maintainer tooling for building release bundles |
| `legacy/` | Historical code excluded from the supported execution surface |

## Troubleshooting

- A release download returning 404 usually means the GitHub account lacks private-repository access or `gh auth login` has not been completed.
- `doctor --profile full` must report CUDA, sufficient memory/disk, and `pyg_lib=true` before full training.
- `hnswlib` may compile locally when a wheel is unavailable; install a platform compiler or use the canonical Python 3.11 environment.
- A replay latency different from the archived number is expected. Metric or integrity-gate differences are not.
- Use `download --offline --profile ...` to verify and extract parts already present under `data/.downloads/3m-evidence-v1` without network access.

For detailed provenance and interpretation, see the [English evidence library](docs/1M/EVIDENCE_SOURCE_LIBRARY_en.md), [evidence manifest](docs/1M/EVIDENCE_SOURCE_MANIFEST.md), and [as-run manifest](repro/as_run_manifest.json).
