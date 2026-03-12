# Adaptive-Code-RAG

A closed-loop optimization RAG system that uses LLM evaluation feedback + REINFORCE policy gradient to end-to-end train a CodeBERT retriever. The goal is to make retrieval serve "code generation quality" rather than semantic similarity alone.

## System Overview

```
Query (HumanEval problem)
        │
        ▼
CodeBERT Encoder (trainable)
        │  query_emb [768]
        ▼
FAISS Index (periodic rebuild)
        │  top-k candidate indices
        ▼
Differentiable Re-scoring
        │  scores = query_emb @ corpus_embs.T
        │  log_probs = log_softmax(scores)  ← gradient attached
        ▼
Retrieved Code Snippets
        │
        ▼
Prompt Builder
        │
        ▼
LLM Generator (GPT-4o-mini, frozen)
        │  n=4 code completions
        ▼
Reward Function
  - execution: run test cases, binary 0/1
  - llm_judge: GPT-4o-mini quality score
  - hybrid: weighted combination
        │
        ▼
REINFORCE Loss
  loss = -log_prob * (reward - baseline) - entropy_bonus
        │
        ▼
AdamW update → CodeBERT encoder
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_key_here
```

### Step 1: Build Corpus

Build FAISS index from HumanEval canonical solutions (~164 snippets):

```bash
python scripts/build_corpus.py --config configs/default.yaml
```

This will:
1. Load HumanEval problems
2. Encode canonical solutions with CodeBERT
3. Save FAISS index to `data/corpus/`

### Step 2: Train

```bash
python scripts/train.py --config configs/default.yaml
```

With overrides:
```bash
python scripts/train.py rl.learning_rate=5e-6 rl.batch_size=4 reward.mode=hybrid
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

### Step 4: Generate Samples

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
│   │   └── corpus_builder.py  # HumanEval corpus builder + metadata I/O
│   ├── retriever/
│   │   ├── encoder.py         # CodeBERT encoder (gradient-aware)
│   │   ├── faiss_index.py     # FAISS IndexFlatIP wrapper
│   │   └── retriever.py       # DifferentiableRetriever
│   ├── generator/
│   │   ├── prompt_builder.py  # RAG prompt construction
│   │   └── llm_client.py      # Async OpenAI client
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
advantage = reward - baseline
loss = -log_prob.sum() * advantage - entropy_coeff * entropy
```

- Baseline: exponential moving average (EMA) of rewards, decay=0.99
- Entropy bonus: prevents the retriever from collapsing to always retrieving the same snippets
- Multiple samples (n=4) per problem reduce variance

### Reward Modes

| Mode | Description |
|------|-------------|
| `execution` | Binary pass/fail from running test cases (default) |
| `llm_judge` | Continuous 0-1 score from GPT-4o-mini judge |
| `hybrid` | `0.7 * execution + 0.3 * llm_judge` |

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
| `retriever.index_refresh_interval` | 50 | Steps between FAISS rebuilds |
| `generator.n_samples` | 4 | Completions per problem for reward estimation |
| `rl.learning_rate` | 1e-5 | AdamW learning rate |
| `rl.batch_size` | 8 | Problems per gradient step |
| `rl.max_steps` | 5000 | Total training steps |
| `rl.entropy_coeff` | 0.01 | Entropy regularization weight |
| `reward.mode` | execution | Reward computation method |

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
