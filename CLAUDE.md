# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description: Adaptive-Code-RAG
A closed-loop optimization RAG system that uses LLM evaluation feedback + REINFORCE policy gradient to end-to-end train a CodeBERT retriever. The training goal is to maximize the LLM-judged relevance score of retrieved snippets, going beyond simple semantic similarity.

**Training** (CodeSearchNet) uses LLM relevance scores as the reward signal — no code generation during training. Periodic evaluation every `eval_interval` steps uses a held-out CSN split and measures `avg_snippet_relevance`. **Final validation** (`scripts/compare_eval.py`) runs code generation on HumanEval to measure pass rate.

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
python scripts/train.py rl.learning_rate=5e-6 data.codesearchnet_max_samples=5000   # with overrides
python scripts/train.py --resume outputs/checkpoints/step_500.pt

# Evaluate (single config)
python scripts/evaluate.py --checkpoint outputs/checkpoints/step_5000.pt
python scripts/evaluate.py --baseline   # no-retrieval baseline

# Compare all three configs and export bar chart + TensorBoard
python scripts/compare_eval.py                                                      # baseline + pretrained
python scripts/compare_eval.py --checkpoint outputs/checkpoints/step_5000.pt       # all three columns
tensorboard --logdir outputs/logs/eval_comparison                                   # view results

# Ablation experiments (top_k / entropy_coeff / index_refresh_interval / relevance_model)
bash scripts/run_ablation.sh                                        # run all groups
bash scripts/run_ablation.sh top_k                                  # single group
tensorboard --logdir outputs/logs/ablation                          # view all curves

# Code search web demo (Pretrained vs RL-Trained side-by-side)
python scripts/search_server.py --checkpoint outputs/checkpoints/step_3000.pt
# visit http://127.0.0.1:8000

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

The corpus embeddings stored in `FaissIndex.corpus_embeddings` (numpy) are periodically rebuilt via `retriever.refresh_index()` every `rl.index_refresh_interval` steps.

### Dual-Index Design (Training vs Eval)

**Training** uses the CSN FAISS index (built by `build_corpus.py`, 10k snippets). Each train step: CSN query → retrieve CSN snippets → LLM relevance reward → REINFORCE.

**Eval** (every `eval_interval` steps) temporarily builds a fresh **HumanEval index** from canonical solutions using the current encoder weights, runs retrieval + generation + execution on the 164 HumanEval problems, then restores the CSN index.

### Training Data Flow

```
CodeSearchNet entry → DifferentiableRetriever.retrieve() [CSN index] → RetrievedContext (with log_probs)
    → RewardFunction.compute_snippet_rewards(problem, snippets)
        → SnippetRelevanceJudge.score_batch() → [relevance_score per snippet]
    → REINFORCEPolicy.compute_loss(log_probs, snippet_rewards) → loss tensor
        → loss = -sum_i(log_probs[i] * (snippet_reward[i] - baseline)) - entropy_bonus
    → loss.backward() → optimizer.step() on CodeBERT only
```

### Eval Data Flow

```
HumanEval problem → build HumanEval index (current encoder) → retrieve top-k
    → build_prompt() → LLMClient.generate() → [code strings]
    → RewardFunction.compute() → batch_execute() → pass@k
    → RewardFunction.compute_snippet_rewards() → avg_snippet_relevance
```

### Key Modules

- **`src/config.py`**: `TrainingConfig` is a nested dataclass (RetrieverConfig, GeneratorConfig, RewardConfig, RLConfig, DataConfig). `DataConfig` includes `codesearchnet_max_samples`. `RewardConfig` no longer has `execution_weight`/`relevance_weight` — training reward is always pure relevance (CSN has no test cases). Loaded via `load_config(path, overrides={})` using `dacite`. All scripts accept positional `key=value` override args.

- **`src/data/codesearchnet_loader.py`**: `load_codesearchnet(language, max_samples, cache_dir)` — loads CodeSearchNet from HuggingFace, filters empty entries, shuffles with seed=42, and returns `List[HumanEvalProblem]` with `test=""`. Reuses the existing `HumanEvalProblem` schema.

- **`src/retriever/retriever.py`**: `DifferentiableRetriever` — the architectural core. `retrieve()` returns `RetrievedContext` with gradient-attached `log_probs` used directly in REINFORCE loss.

- **`src/rl/policy.py`**: `REINFORCEPolicy` with `RunningMeanBaseline` (EMA, decay=0.99). Per-snippet loss = `-sum_i(log_probs[i] * (snippet_reward[i] - baseline)) - entropy_coeff * entropy`. Each snippet gets an independent gradient signal. Entropy bonus prevents retrieval collapse. Supports multiple advantage methods: `"ema_baseline"` (original EMA), `"grpo_softmax"` (group-relative softmax normalization), and optional global penalty via `global_penalty_coeff` to lower probabilities of all retrieved snippets when their average relevance is below a running quality reference.

- **`src/reward/executor.py`**: Runs generated code via `subprocess.run()` with a timeout. The test script is assembled as `function_code + "\n\n" + problem.test + f"\ncheck({entry_point})"`. Returns 1.0/0.0 binary.

- **`src/rl/trainer.py`**: `RLTrainer.train_step()` processes a full batch, accumulates losses per problem, then calls `backward()` once on the mean. `evaluate()` swaps in a fresh HumanEval index, runs eval, then restores the CSN index. Index refresh, eval, and checkpointing are triggered by step count modulo config intervals.

### Config Overrides

CLI overrides use dot-notation positional args (no `--` prefix):
```bash
python scripts/train.py rl.batch_size=4 data.codesearchnet_max_samples=5000
```

### Global Penalty Configuration

To enable global penalty that lowers probabilities of all retrieved snippets when their average relevance is below a running quality reference:
```bash
python scripts/train.py rl.advantage_method=ema_baseline rl.global_penalty_coeff=0.2 rl.global_penalty_threshold=0.05
```

Or with GRPO:
```bash
python scripts/train.py rl.advantage_method=grpo_softmax rl.global_penalty_coeff=0.1
```

Set `global_penalty_coeff=0.0` to disable (default). The penalty term is: `penalty = coeff × max(0, reference - avg_reward - threshold) × sum(log_probs)` where `reference` is the EMA baseline (same decay rate as baseline).

### Data Layout

- `data/corpus/corpus_meta.json` — CodeSnippet metadata (code, docstrings, indices)
- `data/corpus/faiss.index` — FAISS IndexFlatIP binary
- `data/corpus/faiss.index.embeddings.npy` — numpy corpus embeddings (loaded into `FaissIndex.corpus_embeddings` for gradient re-scoring)
- `data/cache/` — HuggingFace dataset cache (CodeSearchNet downloads here)
- `outputs/checkpoints/step_N.pt` — encoder model + optimizer state dicts + step number

### Training Data Strategy

- **Training corpus & queries**: CodeSearchNet (~412k Python pairs, default `max_samples=10000`). No test cases → reward is purely LLM relevance (`compute_snippet_rewards` calls only `SnippetRelevanceJudge`; execution path is not invoked during training).
- **Eval corpus**: HumanEval canonical solutions (164 items). Rebuilt at each eval point using the latest encoder.
- **Eval metrics**: `pass@1`, `pass@{n_samples}` via execution, plus `avg_snippet_relevance` from LLM judge.
