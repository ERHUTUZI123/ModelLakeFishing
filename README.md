# ModelLakeFishing

ModelLakeFishing finds models for a **dataset–task query**: a dataset and the task you want to perform on it. It learns from model metadata and past evaluation results, uses HNSW to find 1,000 candidates, and reranks them to return ten models.

Follow this guide to download current Hugging Face data and run the full pipeline, from data preparation to training and evaluation. You do not need an existing `data/` folder or a trained ModelLakeFishing checkpoint.

**This reruns our method on newly collected data.** The paper used an older snapshot and additional historical evaluation records. A new download will contain different models and evidence, so its results may differ from the paper's results.

## What you download

The default `live` profile downloads metadata for all publicly listed models and datasets on Hugging Face at the time of collection. It does not filter models by popularity or impose a model-count limit. Private repositories are excluded.

The download includes model names, families, sizes, links between base models and their variants, dataset descriptions, and evaluation results reported in model cards. These reported results, stored in the cards' `model-index` field, provide the training and evaluation evidence for this run.

You do **not** download every model's weights or every dataset's examples. The pipeline does not run the candidate models. It downloads a small MiniLM text encoder separately to build features.

Each download is saved with its collection dates and file checksums. Later steps use these saved files. Since Hugging Face can change while a download is running, the collection covers a time period rather than one exact instant. See the [HF Hub API documentation](https://huggingface.co/docs/hub/api) for the data source.

## Hardware and setup

Use Linux x86-64 with Python 3.11 and:

| Resource | Requirement |
|---|---|
| GPU | One **48 GB NVIDIA GPU**, such as an L40S or RTX A6000 |
| NVIDIA driver | **580 or later**, for CUDA 13 |
| System RAM | At least **64 GiB**; **128 GiB** recommended |
| CPU | At least **8 cores** |
| Free disk space | Start with **120 GiB**; larger collections need more |

The original 3M-model training runs used about 38 GiB of GPU memory, measured by PyTorch on an L40S. A larger collection may need more memory. The pipeline uses one GPU; it does not combine the memory of several smaller GPUs.

Check your GPU and driver with `nvidia-smi`. The packages in [`requirements/full-cu130.txt`](requirements/full-cu130.txt) include the CUDA runtime, so you do not need to install a separate CUDA toolkit. If `hnswlib` needs to compile during installation, install a C++ compiler and the development headers for your Python version. On Ubuntu, `build-essential` provides the compiler tools.

For compatibility details, see [NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [PyTorch](https://pytorch.org/get-started/previous-versions/), and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). On a machine with several GPUs, use `export CUDA_VISIBLE_DEVICES=0` to select the first GPU.

## Run the pipeline

Open a Linux terminal and install the project:

```bash
git clone https://github.com/ERHUTUZI123/xiaoyang_graph_task.git ModelLakeFishing
cd ModelLakeFishing
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/full-cu130.txt
```

Then check your setup, download the data, check the files, and run the experiment:

```bash
python -m scale1m.reproduce doctor
python -m scale1m.reproduce download
python -m scale1m.reproduce verify
python -m scale1m.reproduce run --device cuda
```

- `doctor` checks the required packages, CUDA, GPU memory, and the PyG sampler. Fix any reported errors before continuing.
- `download` collects the public HF metadata and saves it under `data/live/<capture-id>/`. It remembers this collection for the next commands.
- `verify` checks that the download finished and that the saved files are intact.
- `run` prepares the data, trains the model, builds the search indexes, and evaluates retrieval.

All four commands use `live` by default; you do not need to add `--profile live`.

You can download without an HF account. If you have an access token, set `HF_TOKEN` to use your account's API allowance. The token is not saved in the downloaded data. The downloader retries failed requests and waits when it hits a rate limit. Download time and size depend on the current Hub contents and [HF rate limits](https://huggingface.co/docs/hub/rate-limits).

### What happens during a run

1. Clean the metadata and evaluation records, remove duplicates, and normalize scores so different metrics can be used together.
2. Match dataset descriptions and build the candidate list, model features, and evidence graph.
3. Train for **25 epochs on each of three splits**, using split seeds 0, 1, and 2 and initialization seed 0.
4. Export model and query embeddings, then build a task-based reranking prior for each split.
5. Score the full lake to create an exact reference, build HNSW indexes, and measure retrieval quality and speed. HNSW uses `M=32` and `efConstruction=200`; its search setting is chosen by comparison with exact retrieval.

The run uses the model counts and evaluation evidence in your download. If there is too little usable evidence for the required splits, it stops before building features.

The evaluation is **transductive**: the graph includes the test dataset–task nodes and their metadata during training, but hides their held-out performance edges in both directions. Seven dataset feature columns derived from performance scores are set to zero. The reranking prior also excludes evidence from the test queries' dataset roots. These measures keep held-out scores out of training and reranking.

## Find and read your results

Print the summary after a run finishes:

```bash
python -m scale1m.reproduce compare
```

For a live run, `compare` shows your measured results; it does not check them against a fixed paper score.

- **gold@10** is the fraction of eligible test queries for which the returned ten models include the best model in the held-out evaluation records.
- **Retention** compares HNSW's gold@10 with the gold@10 from scoring and reranking the entire lake. Both are measured on your downloaded data.
- **p50 latency** is the median query time. **p95 latency** is the time within which 95% of queries finish.

Latency covers HNSW search and reranking using precomputed query embeddings and one CPU thread. It excludes encoding, training, index building or loading, and downloads. Summary values average the three splits; retention is the average of the three per-split ratios.

The commands create this layout automatically inside the project:

```text
data/
  LIVE_CURRENT.json         # which collection the commands currently use
  live/<capture-id>/
    models/                 # downloaded model metadata and evaluation records
    datasets/               # downloaded dataset metadata
    LIVE_SNAPSHOT.json      # collection dates and file checksums
  .hf_home/                 # downloaded MiniLM encoder
  work/live/<capture-id>/
    rf/                     # cleaned metadata and evaluation records
    ladder/                 # model and dataset–task lists
    features/               # features and model-family labels
    graph/                  # training graph
    runs/                   # three training runs and checkpoints
    exports/                # embeddings, test labels, and reranking priors
    evaluation/
      LIVE_RESULTS.json     # retrieval quality and latency
      LIVE_MANIFEST.json    # files used and produced by evaluation
    LIVE_UNIVERSE.json      # model, dataset, and split counts
    LIVE_REPRODUCTION.json  # commands, file checksums, and run progress
    LIVE_SUMMARY.json       # summary printed by compare
```

## Resume a run or download newer data

If the download is interrupted, run the same command again:

```bash
python -m scale1m.reproduce download
```

To continue an interrupted experiment:

```bash
python -m scale1m.reproduce run --device cuda --resume
```

Resume checks that the data, code, settings, and completed outputs have not changed. It skips completed steps and continues training from the last checkpoint. If an interrupted preprocessing or export step left partial files that cannot be reused, choose a new output folder with `--workspace /path/to/new-run`.

A completed download is reused. To collect newer HF data and start a separate experiment:

```bash
python -m scale1m.reproduce download --new-snapshot
python -m scale1m.reproduce verify
python -m scale1m.reproduce run --device cuda
```

These options let you choose names and locations:

| Option | Use |
|---|---|
| `--snapshot-id NAME` | Choose a saved collection, or name a new one when downloading |
| `--data-root /path/to/data` | Store downloads and results elsewhere; use the same option on each command |
| `--workspace /path/to/new-run` | Choose a separate output folder for an experiment |

You can also set `MLF_DATA_DIR` instead of passing `--data-root`. Keep the experiment's output folder separate from its downloaded inputs.

To see all the steps before running anything:

```bash
python -m scale1m.reproduce plan
```

## Paper results and fixed settings

The paper's A0 experiment used a **2026-08-18 snapshot plus five historical evidence graphs**. It contained 3,016,439 candidate models, 18,729 dataset–task nodes, and 247,803 performance edges. Across three splits, mean gold@10 was **0.2968**, mean retention was **93.47%**, and mean per-split p50/p95 latency was **0.747/1.102 ms**. The recorded values are in [`expected.json`](scale1m/reproduction/expected.json).

The live workflow uses current HF model-card evaluation records. It does not download those five historical graphs, so it does not recreate the paper's exact data.

The older `full`, `train`, and `replay` profiles require the original files listed in [`a0_snapshot.json`](scale1m/reproduction/a0_snapshot.json). An HF download location for that archive is not configured. Use the default `live` workflow for the instructions above.

Feature encoding uses `sentence-transformers/all-MiniLM-L6-v2`, fixed at commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Each collection gets its own model-family vocabulary.

## Where the code lives

| Path | What it does |
|---|---|
| `scale1m/reproduce.py` | Runs the commands in this guide |
| `scale1m/reproduction/live_snapshot.py` | Downloads HF metadata and checks saved files |
| `scale1m/reproduction/pipeline.py` | Runs the pipeline steps in order |
| `scale1m/reproduction/live_stages.py` | Checks the prepared data and graph |
| `scale1m/evaluate_live.py` | Measures exact and HNSW retrieval quality and speed |
| `scale1m/` | Prepares data, builds features and graphs, trains, and exports embeddings |
| `stage2TrainGraphSAGE/` | Implements the graph encoder and training |
| `stage3HNSW/build_prior_sidecar.py` | Builds the task-based reranking prior |

To view the model-lake visualization, run:

```bash
python -m http.server 8000 --directory lake3d
```

Then open [http://127.0.0.1:8000/lake3d_offline.html](http://127.0.0.1:8000/lake3d_offline.html).
