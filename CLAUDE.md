# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description: Adaptive-Code-RAG
A closed-loop optimization RAG system that uses LLM evaluation feedback + REINFORCE policy gradient to end-to-end train a CodeBERT retriever. The goal is to make retrieval serve "code generation quality" rather than semantic similarity alone.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start local Ollama service (must run before training)
ollama pull qwen2.5-coder:7b   # one-time download, ~4.5GB
ollama serve                    # keep running in the background

# Build corpus (must run before training)
python scripts/build_corpus.py --config configs/default.yaml

# Train
python scripts/train.py
python scripts/train.py rl.learning_rate=5e-6 reward.execution_weight=0.8   # with overrides
python scripts/train.py --resume outputs/checkpoints/step_500.pt

# Evaluate
python scripts/evaluate.py --checkpoint outputs/checkpoints/step_5000.pt
python scripts/evaluate.py --baseline   # no-retrieval baseline

# Inference demo
python scripts/generate_samples.py --task-id "HumanEval/42" --n 4

# Tests
pytest tests/ -v
pytest tests/test_rl.py -v   # single test file
pytest tests/test_reward.py::test_executor_correct_solution -v   # single test
```

## Architecture

This is a closed-loop RL system where only the **CodeBERT query encoder** is trained — the LLM generator and reward function are frozen. The training signal flows backward from test execution results through REINFORCE to the encoder.

### Gradient Path (the critical design)

FAISS does not support gradients, so retrieval uses a two-stage trick:
1. `encoder.encode_query()` encodes the problem prompt **with gradient** → `query_emb [768]`
2. FAISS searches using `query_emb.detach()` → returns top-k candidate **indices** (no grad)
3. Corpus embeddings for those indices are fetched and re-scored: `scores = query_emb @ corpus_embs_k.T` — **gradient flows here**
4. `log_probs = log_softmax(scores)` is stored in `RetrievedContext.log_probs` and used for REINFORCE loss

The corpus embeddings stored in `FaissIndex.corpus_embeddings` (numpy) are periodically rebuilt via `retriever.refresh_index()` every `index_refresh_interval` steps.

### Data Flow

```
HumanEvalProblem → DifferentiableRetriever.retrieve() → RetrievedContext (with log_probs)
    → build_prompt() → LLMClient.generate() → [code strings]
    → RewardFunction.compute_snippet_rewards(problem, generated_codes, snippets)
        → batch_execute() → execution_reward (mean pass rate)
        → SnippetRelevanceJudge.score_batch() → [relevance_score per snippet]
        → snippet_reward[i] = 0.7 * execution_reward + 0.3 * relevance_score[i]
    → REINFORCEPolicy.compute_loss(log_probs, snippet_rewards) → loss tensor
        → loss = -sum_i(log_probs[i] * (snippet_reward[i] - baseline)) - entropy_bonus
    → loss.backward() → optimizer.step() on CodeBERT only
```

### Key Modules

- **`src/config.py`**: `TrainingConfig` is a nested dataclass (RetrieverConfig, GeneratorConfig, RewardConfig, RLConfig, DataConfig). Loaded via `load_config(path, overrides={})` using `dacite`. All scripts accept positional `key=value` override args.

- **`src/retriever/retriever.py`**: `DifferentiableRetriever` — the architectural core. `retrieve()` returns `RetrievedContext` with gradient-attached `log_probs`. `compute_log_prob()` returns the scalar used in REINFORCE (`log_probs.sum()`).

- **`src/rl/policy.py`**: `REINFORCEPolicy` with `RunningMeanBaseline` (EMA, decay=0.99). Per-snippet loss = `-sum_i(log_probs[i] * (snippet_reward[i] - baseline)) - entropy_coeff * entropy`. Each snippet gets an independent gradient signal. Entropy bonus prevents retrieval collapse.

- **`src/reward/executor.py`**: Runs generated code via `subprocess.run()` with a timeout. The test script is assembled as `function_code + "\n\n" + problem.test + f"\ncheck({entry_point})"`. Returns 1.0/0.0 binary.

- **`src/rl/trainer.py`**: `RLTrainer.train_step()` processes a full batch, accumulates losses per problem, then calls `backward()` once on the mean. Index refresh, eval, and checkpointing are triggered by step count modulo config intervals.

### Config Overrides

CLI overrides use dot-notation positional args (no `--` prefix):
```bash
python scripts/train.py rl.batch_size=4 reward.execution_weight=0.8
```

### Data Layout

- `data/corpus/corpus_meta.json` — CodeSnippet metadata (code, docstrings, indices)
- `data/corpus/faiss.index` — FAISS IndexFlatIP binary
- `data/corpus/faiss.index.embeddings.npy` — numpy corpus embeddings (loaded into `FaissIndex.corpus_embeddings` for gradient re-scoring)
- `outputs/checkpoints/step_N.pt` — encoder model + optimizer state dicts + step number
