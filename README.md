# GLoRAG-Ex

**Counterfactual Knowledge Graph-Based Local and Global Explanations for RAG**

GloRAG-Ex explains *why* a graph-based Retrieval-Augmented Generation (RAG) pipeline answers a question correctly or incorrectly by searching for **counterfactual edits** to the retrieved knowledge subgraph. Given a question, it finds the minimum-cost set of node/edge deletions and additions that *flip* the RAG system's answer (correct → incorrect, or incorrect → correct), and aggregates these local, per-instance counterfactuals into **global explanations** describing which structural and semantic features drive the RAG system's behavior across an entire benchmark.

The repository contains the full experimental pipeline used in the paper: counterfactual search, an optional **Pivotal-Star Probe (PSP)** search accelerator, global explanation aggregation, and evaluation against several competitor explainability methods (Shapley-value attribution, KG-SMILE, RAG-Ex text-span removal, etc.).

## Table of Contents

- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
  - [1. Build embedding indices](#1-build-embedding-indices)
  - [2. Run RAG / LLM-only baselines](#2-run-rag--llm-only-baselines)
  - [3. Generate counterfactual explanations](#3-generate-counterfactual-explanations)
  - [4. Aggregate global explanations](#4-aggregate-global-explanations)
  - [5. Evaluate correctness, robustness & ablations](#5-evaluate-correctness-robustness--ablations)
- [Competitor / baseline methods](#competitor--baseline-methods)
- [Configuration notes](#configuration-notes)

## How it works

1. **Retrieve** a subgraph of a LightRAG-backed knowledge graph for a given question (`hybrid`, `local`, `global`, or `naive` retrieval modes).
2. **Search** for a minimum-cost sequence of graph edit operations (`delete_node`, `delete_edge`, `add_node`, `add_edge`) that flips the RAG system's answer, using a cost-ordered best-first search over the space of subgraph edits (`src/counterfactuals/generate.py`). Three edit-cost/flip directions are supported:
   - `ft` — breaking edits that flip a correct answer to incorrect (True → False)
   - `ff` / `tf` — corrective edits that flip an incorrect answer to correct
3. **Accelerate** the search with the **Pivotal-Star Probe (PSP)**, which prioritizes candidate deletions likely to be pivotal, optionally at various `--psp-k` values.
4. **Aggregate** many local (per-question) counterfactuals into global explanations across a benchmark, at the feature, element, cost, and operation-type level (`src/global_explanations/`).
5. **Evaluate** counterfactual quality via correctness against ground-truth supporting facts, noise robustness, and sufficiency, and compare against several competitor explanation methods.

## Repository structure

```
GloRAG-Ex/
├── code/
│   ├── src/
│   │   ├── counterfactuals/       # Core counterfactual search (generate.py, PSP, ablations, perturbations)
│   │   ├── global_explanations/   # Aggregation of local explanations into global ones
│   │   ├── embeddings/            # HNSW index building & querying for nodes/edges
│   │   ├── correctness/           # Precision vs. ground-truth supporting facts
│   │   ├── quality_metrics/       # Noise resistance / sufficiency metrics
│   │   ├── cfe_evaluation/        # Counterfactual explanation evaluation
│   │   ├── medical/               # Medical-domain dataset variant (vector DB, retrieval, eval)
│   │   ├── llm/                   # vLLM / sentence-transformers model wrappers
│   │   ├── statistics/, robustness_plots/, shapley_results/, sampled_questions/
│   │   ├── query.py, retrieve.py, parser.py, base.py, llm_judge.py, dataset_setup.py
│   │   └── explanation_result.py
│   ├── benchmark/                 # RAG-only / LLM-only baseline runners + evaluation
│   ├── datasets/                  # synthetic, hotpotqa, musique, 2wiki QA data
│   ├── KGs/lightrag/              # Pre-built LightRAG knowledge graphs per dataset
│   ├── kg_smile/                  # KG-SMILE baseline implementation
│   ├── generate_cfe.sh            # Example: generate counterfactuals (musique)
│   ├── run_cf.sh                  # Full pipeline: baselines → deletions → PSP → additions
│   ├── run_ff_psp.sh              # Corrective (F→T) counterfactuals + PSP, all datasets
│   ├── run_ablation.sh            # Component ablation study (cache, index, PSP-k, LLM, retrieval mode)
│   ├── run_correctness.sh         # Correctness evaluation across all methods/datasets
│   ├── run_robustness.sh          # Noise-resistance evaluation
│   └── visualize_results.py
├── competitors/                   # Baseline explainability methods for comparison
│   ├── Shapley/                   # Shapley-value attribution (graph & text)
│   ├── RAGEX-RAGE-SHAPLEY/        # RAGE + RAG-Ex + Shapley combined baselines
│   ├── RAGE/                      # RAGE baseline
│   ├── kg_smile/                  # KG-SMILE baseline
│   ├── KGRAG-Ex/                  # KGRAG-Ex baseline
│   ├── LLMX/                      # LLM-only explanation baseline
│   ├── PoolNoiseSelector/         # Noise-injection utility for robustness experiments
│   ├── Musique/, 2wiki_dataset/   # Dataset-specific competitor resources
└── requirements.txt
```

## Installation

Requires Python 3.11+ and a CUDA-capable GPU (the pipeline uses [vLLM](https://github.com/vllm-project/vllm) for local LLM inference and `faiss`/`hnswlib` for indices).

```bash
git clone https://github.com/george-balanos/GloRAG-Ex.git
cd GloRAG-Ex
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> All scripts under `code/` automatically prefer a `.venv` created at the repo root (`../.venv/bin/python` relative to `code/`) if present, then fall back to `uv run`, then plain `python`.

By default the generation LLM is `mistralai/Mistral-Small-3.2-24B-Instruct-2506`, the judge LLM is `Qwen/Qwen2.5-32B-Instruct`, and the embedding model is `sentence-transformers/all-MiniLM-L6-v2` (see `code/src/llm/utils.py`). These can be swapped via `--gen-llm` in the ablation entrypoint or by editing `VLLM_MODEL` / `JUDGE_MODEL`.

## Datasets

Four QA benchmarks are supported out of the box, each backed by a pre-built LightRAG knowledge graph under `code/KGs/lightrag/<dataset>/`:

| Dataset     | Description                                   |
|-------------|------------------------------------------------|
| `synthetic` | Synthetic multi-hop QA benchmark              |
| `hotpotqa`  | HotpotQA multi-hop QA                         |
| `musique`   | MuSiQue multi-hop QA                          |
| `2wiki`     | 2WikiMultihopQA                               |

A medical-domain variant is also available under `code/src/medical/`.

## Usage

All commands below are run from the `code/` directory.

### 1. Build embedding indices

HNSW node/edge embedding indices are built automatically by the run scripts if missing, or can be built explicitly:

```bash
python -m src.embeddings.build_index --dataset synthetic
```

### 2. Run RAG / LLM-only baselines

```bash
# RAG-only baseline
python benchmark/run.py --dataset synthetic --rag-mode hybrid --top-k 2

# LLM-only baseline (no retrieval)
python benchmark/run.py --dataset synthetic --rag-mode bypass --top-k 0

# Build the comparison file used as input for counterfactual generation
python -m benchmark.evaluation --dataset synthetic --rag-mode hybrid --top-k 2 --rag-only
```

### 3. Generate counterfactual explanations

```bash
# Breaking edits, T -> F (deletions only)
python -m src.counterfactuals.generate \
    --dataset musique --mode ft --rag-mode hybrid --top-k 2 \
    --ops delete_node,delete_edge --max-cost 20 --max-llm-calls 200 --adm 2

# Corrective edits, F -> F (additions + deletions)
python -m src.counterfactuals.generate \
    --dataset musique --mode ff --rag-mode hybrid --top-k 2 \
    --ops add_node,add_edge,delete_node,delete_edge --max-cost 20 --max-llm-calls 200 --adm 2
```

Or run the full end-to-end pipeline (baselines → deletions-only → deletions+PSP → additions) for `synthetic` and `hotpotqa`:

```bash
./run_cf.sh
```

For corrective (F→T) counterfactuals with PSP enabled across all four datasets:

```bash
./run_ff_psp.sh
```

Key `generate.py` flags:

| Flag | Description |
|---|---|
| `--mode {ff,ft,tf}` | Flip direction: `ff`/`tf` corrective, `ft` breaking |
| `--ops` | Comma-separated subset of `delete_node,delete_edge,add_node,add_edge` |
| `--max-cost` / `--max-llm-calls` | Search budget |
| `--psp` / `--psp-k` | Enable the Pivotal-Star Probe and set top-K pivots (T→F only, requires `delete_node`) |
| `--add-heuristic {none,tier,blend}` | Ordering heuristic for addition operations |
| `--adm {1,2,3}` | Add-mode variant |

### 4. Aggregate global explanations

Global (benchmark-level) explanations are built by aggregating local counterfactuals — see `code/src/global_explanations/` (feature-level, element-level, cost, and operation-type aggregation) and `code/src/global_explanations/generate_global_explanations.ipynb`.

### 5. Evaluate correctness, robustness & ablations

```bash
# Precision of counterfactual/attribution edits vs. ground-truth supporting facts,
# across GloRAG-Ex, Shapley, KG-SMILE, RAG-Ex, and Shapley-Text methods
./run_correctness.sh

# Noise-resistance / robustness evaluation across all datasets
./run_robustness.sh

# Component ablation (cache, embedding index, PSP-k, generation LLM, retrieval mode)
./run_ablation.sh
```

## Competitor / baseline methods

The `competitors/` directory contains independent implementations of alternative explanation methods used for comparison in evaluation:

- **Shapley** — Shapley-value attribution over graph elements and text spans
- **KG-SMILE** (`kg_smile/`, `competitors/kg_smile/`) — surrogate-model-based KG explanation
- **RAG-Ex** — text-span removal-based explanations (sentence/paragraph granularity)
- **RAGE / RAGEX-RAGE-SHAPLEY** — additional attribution baselines
- **KGRAG-Ex**, **LLMX** — further baseline explanation methods
- **PoolNoiseSelector** — utility for injecting controlled noise for robustness experiments

## Configuration notes

- Scripts assume two GPUs are available: the generation LLM runs on `CUDA_VISIBLE_DEVICES=0` and the judge LLM on `CUDA_VISIBLE_DEVICES=1` (`code/src/llm/utils.py`); the ablation script also references a `SHAP_DEVICE` variable for competitor runs.
- Large generated artifacts (`*.json`, `*.csv`, `*.graphml`, `*.png`, `*.npy`, `*.bin`) are git-ignored — pre-built KGs under `code/KGs/lightrag/` are the exception, tracked so the pipeline can run out of the box.
- No license file is currently included in this repository.
