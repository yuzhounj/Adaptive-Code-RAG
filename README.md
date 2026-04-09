# Adaptive-Code-RAG

A closed-loop optimization RAG system that uses LLM evaluation feedback + REINFORCE policy gradient to end-to-end train a CodeBERT retriever. The training goal is to maximize the LLM-judged relevance score of retrieved snippets, going beyond simple semantic similarity.

**Training** (CodeSearchNet) uses LLM relevance scores as the reward signal — no code generation during training. **Final validation** (`scripts/compare_eval.py`) runs code generation on HumanEval to measure pass rate.

## System Overview

### Training Loop (CodeSearchNet, no code generation)

```
CodeSearchNet query (docstring)
        │
        ▼
CodeBERT Encoder (trainable)
        │  query_emb [768]
        ▼
FAISS Index (periodic rebuild)
        │  top-k candidate indices (no gradient)
        ▼
Differentiable Re-scoring
        │  scores = query_emb @ corpus_embs.T
        │  log_probs = log_softmax(scores)  ← gradient attached
        ▼
Retrieved Code Snippets
        │
        ▼
SnippetRelevanceJudge (LLM-as-judge, frozen)
        │  relevance_score ∈ [0, 1] per snippet
        ▼
REINFORCE Loss (per-snippet)
  loss = -sum_i(log_probs[i] * (relevance_i - EMA_baseline)) - entropy_bonus
        │
        ▼
AdamW update → CodeBERT encoder only
```

### Final Validation (HumanEval, via compare_eval.py)

```
HumanEval problem
        │
        ▼
CodeBERT Encoder (frozen checkpoint)  →  HumanEval FAISS index
        │  retrieved snippets
        ▼
Prompt Builder  →  LLM Generator (Qwen2.5-Coder, frozen)
        │  generated code
        ▼
Test Executor  →  pass@k
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Ollama Setup

```bash
# Install Ollama (if not already installed)
brew install ollama

# Pull the model (one-time download, ~4.5GB)
ollama pull qwen2.5-coder:7b

# Start the service (keep running in the background during training)
ollama serve
```

### Step 1: Build Corpus

Build the **CSN training corpus** (used during training only). The HumanEval eval corpus is built on-the-fly at each eval checkpoint — no separate build step needed.

```bash
python scripts/build_corpus.py --config configs/default.yaml
```

This will:
1. Load CodeSearchNet from HuggingFace (cached to `data/cache/`)
2. Encode each snippet (docstring + code) with CodeBERT
3. Save FAISS index and metadata to `data/corpus/`


### Step 2: Train

```bash
python scripts/train.py --config configs/default.yaml
```

With overrides:
```bash
python scripts/train.py rl.learning_rate=5e-6 rl.batch_size=4
```

Resume from checkpoint:
```bash
python scripts/train.py --resume outputs/checkpoints/step_500.pt
```

### Step 3: Evaluate

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint outputs/checkpoints/step_5000.pt

# Evaluate pretrained baseline (no RL training)
python scripts/evaluate.py

# Evaluate no-retrieval baseline
python scripts/evaluate.py --baseline
```

### Step 4: Ablation Experiments

Run systematic ablation experiments across four dimensions: `top_k`, `entropy_coeff`, `index_refresh_interval`, and `relevance_model`.

```bash
bash scripts/run_ablation.sh            # run all groups (11 runs total)
bash scripts/run_ablation.sh top_k      # only top_k group
bash scripts/run_ablation.sh entropy    # only entropy_coeff group
bash scripts/run_ablation.sh refresh    # only index_refresh_interval group
bash scripts/run_ablation.sh relevance  # only relevance_model group
```

Each run saves its checkpoint to `outputs/ablation/<name>/` and TensorBoard logs to `outputs/logs/ablation/<name>/`. View all curves side by side:

```bash
tensorboard --logdir outputs/logs/ablation
```

| Group | Variants | Default |
|---|---|---|
| `top_k` | 2, 5, 7 | **3** |
| `entropy_coeff` | 0.0, 0.001, 0.1 | **0.01** |
| `index_refresh_interval` | 50, 200, 99999 | **500** |
| `relevance_model` | qwen2.5-coder:3b, :7b | **:14b** |

### Step 5: Generate Samples

```bash
# Generate for first 3 HumanEval problems
python scripts/generate_samples.py --n 4

# Generate for specific task
python scripts/generate_samples.py --task-id "HumanEval/42" --n 4
```

## Project Structure

```
Adaptive-Code-RAG/
├── configs/
│   └── default.yaml          # All hyperparameters
├── src/
│   ├── config.py              # Dataclass configs + YAML loader
│   ├── data/
│   │   ├── schema.py          # HumanEvalProblem, CodeSnippet, RetrievedContext
│   │   ├── humaneval_loader.py
│   │   ├── codesearchnet_loader.py  # CodeSearchNet → HumanEvalProblem adapter
│   │   └── corpus_builder.py  # HumanEval corpus builder + metadata I/O
│   ├── retriever/
│   │   ├── encoder.py         # CodeBERT encoder (gradient-aware)
│   │   ├── faiss_index.py     # FAISS IndexFlatIP wrapper
│   │   └── retriever.py       # DifferentiableRetriever
│   ├── generator/
│   │   ├── prompt_builder.py  # RAG prompt construction
│   │   └── llm_client.py      # Async LLM client via litellm
│   ├── reward/
│   │   ├── executor.py        # Subprocess-based test execution
│   │   ├── llm_judge.py       # LLM-as-judge scoring
│   │   └── reward_fn.py       # Unified reward interface
│   ├── rl/
│   │   ├── policy.py          # REINFORCE + EMA baseline
│   │   └── trainer.py         # RLTrainer main loop
│   └── utils/
│       ├── metrics.py         # pass@k (unbiased), MRR, NDCG
│       ├── checkpoint.py      # Save/load model checkpoints
│       └── logging_utils.py   # TensorBoard + W&B logging
├── scripts/
│   ├── build_corpus.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate_samples.py
└── tests/
    ├── test_data.py
    ├── test_retriever.py
    ├── test_reward.py
    └── test_rl.py
```

## Key Design Choices

### Differentiable Retrieval

FAISS does not support gradients, so we use a two-stage approach:
1. FAISS retrieves top-k candidates (fast, no gradient)
2. Re-score candidates differentiably: `scores = query_emb @ corpus_embs_k.T`
3. Gradients flow through `query_emb` back to CodeBERT encoder

The FAISS index is rebuilt periodically (every `index_refresh_interval` steps) to stay aligned with the updated encoder.

### REINFORCE with Baseline

```python
advantages[i] = snippet_reward[i] - baseline
loss = -sum_i(log_probs[i] * advantages[i]) - entropy_coeff * entropy
```

- Baseline: exponential moving average (EMA) of rewards, decay=0.99
- Entropy bonus: prevents the retriever from collapsing to always retrieving the same snippets
- Multiple samples (n=4) per problem reduce variance

### Per-Snippet Reward

Training uses CodeSearchNet which has no test cases, so the training reward is purely LLM relevance:

```
snippet_reward[i] = relevance_score[i]    # during training (CSN)
```

Each snippet gets an independent gradient signal via REINFORCE, allowing the retriever to learn which snippets are most relevant rather than treating all k equally.

During **HumanEval evaluation**, execution reward (pass rate) is used separately to measure code generation quality — it is not mixed into the training signal.

### pass@k Metric

Uses the unbiased estimator from Chen et al. 2021:
```
pass@k = 1 - C(n-c, k) / C(n, k)
```
where n=total samples, c=correct samples, k=k.

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retriever.top_k` | 5 | Number of retrieved snippets |
| `rl.index_refresh_interval` | 100 | Steps between CSN FAISS index rebuilds |
| `generator.n_samples` | 4 | Completions per problem for reward estimation |
| `rl.learning_rate` | 1e-5 | AdamW learning rate |
| `rl.batch_size` | 8 | Problems per gradient step |
| `rl.max_steps` | 5000 | Total training steps |
| `rl.entropy_coeff` | 0.01 | Entropy regularization weight |
| `data.codesearchnet_max_samples` | 10000 | Max CodeSearchNet samples to load for training |

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Data schema creation and validation
- FAISS index build, search, save/load
- Reward executor (correct/wrong/empty solutions)
- REINFORCE policy loss computation and backpropagation
- EMA baseline variance reduction
