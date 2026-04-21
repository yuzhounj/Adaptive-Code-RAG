import os
import random
import shutil
import torch
from typing import List, Optional
from tqdm import tqdm

from src.config import TrainingConfig
from src.data.schema import HumanEvalProblem
from src.retriever.retriever import DifferentiableRetriever
from src.retriever.encoder import CodeBERTEncoder
from src.generator.llm_client import LLMClient
from src.reward.reward_fn import RewardFunction
from src.rl.policy import REINFORCEPolicy
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging_utils import TrainingLogger


class RLTrainer:
    """Full closed-loop RL training loop for CodeBERT retriever."""

    def __init__(
        self,
        config: TrainingConfig,
        retriever: DifferentiableRetriever,
        encoder: CodeBERTEncoder,
        llm_client: LLMClient,
        reward_fn: RewardFunction,
        device: torch.device,
    ):
        self.config = config
        self.retriever = retriever
        self.encoder = encoder
        self.llm_client = llm_client
        self.reward_fn = reward_fn
        self.device = device

        self.policy = REINFORCEPolicy(
            baseline_decay=config.rl.baseline_decay,
            entropy_coeff=config.rl.entropy_coeff,
            advantage_method=config.rl.advantage_method,
            grpo_temperature=config.rl.grpo_temperature,
            global_penalty_coeff=config.rl.global_penalty_coeff,
            global_penalty_threshold=config.rl.global_penalty_threshold,
        )

        self.optimizer = torch.optim.AdamW(
            encoder.model.parameters(),
            lr=config.rl.learning_rate,
        )

        self.logger = TrainingLogger(config)
        self.global_step = 0

    def train_step(self, batch: List[HumanEvalProblem]) -> dict:
        """Run one training step on a batch of problems."""
        # 1. Retrieve with gradient (fast, serial)
        contexts = [self.retriever.retrieve(problem) for problem in batch]

        # 2. Compute per-snippet relevance rewards (no LLM generation — CSN has no test cases)
        all_snippet_rewards = []
        for problem, context in zip(batch, contexts):
            snippet_rewards = self.reward_fn.compute_snippet_rewards(
                problem, snippets=context.snippets
            )
            all_snippet_rewards.append(snippet_rewards)

        flat = [r for rewards in all_snippet_rewards for r in rewards]
        raw_mean_reward = sum(flat) / len(flat) if flat else 0.0

        batch_losses = []
        batch_rewards = []
        batch_advantages = []
        batch_entropies = []
        batch_pg_losses = []
        batch_baselines = []

        for context, snippet_rewards in zip(contexts, all_snippet_rewards):
            loss_output = self.policy.compute_loss(
                log_probs=context.log_probs,
                snippet_rewards=snippet_rewards,
            )
            batch_losses.append(loss_output.loss)
            batch_rewards.append(loss_output.raw_reward)
            batch_advantages.append(loss_output.advantage)
            batch_entropies.append(loss_output.entropy)
            batch_pg_losses.append(loss_output.pg_loss)
            batch_baselines.append(loss_output.baseline_val)

        # 3. Aggregate and backprop
        total_loss = torch.stack(batch_losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.encoder.model.parameters(),
            self.config.rl.max_grad_norm,
        ).item()
        self.optimizer.step()

        n = len(batch_rewards)
        mean_reward = sum(batch_rewards) / n
        reward_std = (sum((r - mean_reward) ** 2 for r in batch_rewards) / n) ** 0.5

        return {
            "loss": total_loss.item(),
            "pg_loss": sum(batch_pg_losses) / n,
            "entropy": sum(batch_entropies) / n,
            "mean_reward": raw_mean_reward,
            "reward_std": reward_std,
            "mean_advantage": sum(batch_advantages) / n,
            "baseline": sum(batch_baselines) / n,
            "grad_norm": grad_norm,
            "snippet_rewards": flat,
        }

    def train(
        self,
        train_problems: List[HumanEvalProblem],
        eval_problems: Optional[List[HumanEvalProblem]] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """Main training loop."""
        os.makedirs("outputs", exist_ok=True)
        log_mode = "w" if self.global_step == 0 else "a"
        self._metrics_log = open("outputs/metrics.log", log_mode)
        fail_dir = "outputs/fail_cases"
        if os.path.exists(fail_dir):
            shutil.rmtree(fail_dir)
        os.makedirs(fail_dir)

        if resume_from:
            self.global_step = load_checkpoint(
                resume_from, self.encoder, self.optimizer
            )
            print(f"Resumed from step {self.global_step}")

        rl_cfg = self.config.rl

        pbar = tqdm(total=rl_cfg.max_steps, initial=self.global_step, desc="Training")

        # Always rebuild index on startup to ensure corpus embeddings match current encoder
        self.retriever.refresh_index(step=self.global_step)

        # Initial evaluation at step 0 (baseline before training)
        if eval_problems and self.global_step == 0:
            eval_metrics = self.evaluate(eval_problems)
            self.logger.log(eval_metrics, step=0, prefix="eval")
            print(f"\nStep 0 eval (baseline): {eval_metrics}")

        while self.global_step < rl_cfg.max_steps:
            # Sample batch
            batch = random.sample(train_problems, min(rl_cfg.batch_size, len(train_problems)))

            # Train step
            metrics = self.train_step(batch)
            self.global_step += 1
            pbar.update(1)

            # Logging
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rew": f"{metrics['mean_reward']:.3f}",
            })
            if self.global_step % rl_cfg.log_interval == 0:
                self.logger.log(metrics, step=self.global_step)
            if self.global_step % rl_cfg.metrics_log_interval == 0:
                self._metrics_log.write(
                    f"[step {self.global_step}] "
                    f"loss={metrics['loss']:.4f}  pg={metrics['pg_loss']:.4f}  ent={metrics['entropy']:.4f} | "
                    f"rew={metrics['mean_reward']:.4f}±{metrics['reward_std']:.4f}  "
                    f"adv={metrics['mean_advantage']:.4f}  base={metrics['baseline']:.4f} | "
                    f"gnorm={metrics['grad_norm']:.4f} | "
                    f"snippet_rewards=[{', '.join(f'{r:.3f}' for r in metrics['snippet_rewards'])}]\n"
                )
                self._metrics_log.flush()

            # Evaluation
            if eval_problems and self.global_step % rl_cfg.eval_interval == 0:
                eval_metrics = self.evaluate(eval_problems)
                self.logger.log(eval_metrics, step=self.global_step, prefix="eval")
                print(f"\nStep {self.global_step} eval: {eval_metrics}")

            # Refresh FAISS index
            if self.global_step % rl_cfg.index_refresh_interval == 0:
                self.retriever.refresh_index(step=self.global_step)

            # Checkpoint
            if self.global_step % rl_cfg.checkpoint_interval == 0:
                ckpt_path = f"{self.config.output.checkpoint_dir}/step_{self.global_step}.pt"
                save_checkpoint(ckpt_path, self.encoder, self.optimizer, self.global_step)

        pbar.close()
        self.logger.close()
        self._metrics_log.close()

    def evaluate(self, eval_problems: List[HumanEvalProblem]) -> dict:
        """Evaluate retrieval quality on held-out CSN problems using the current CSN index.

        No index swap, no code generation. Metrics: avg_snippet_relevance (position-weighted).
        """
        eval_logs_root = os.path.join("outputs", "eval_logs")
        if self.global_step == 0 and os.path.exists(eval_logs_root):
            shutil.rmtree(eval_logs_root)
        log_dir = os.path.join(eval_logs_root, f"step_{self.global_step}")
        os.makedirs(log_dir, exist_ok=True)

        self.encoder.eval()
        try:
            with torch.no_grad():
                contexts = [self.retriever.retrieve(p) for p in tqdm(eval_problems, desc="Eval-retrieve", leave=False)]

            # Flatten all (problem, snippet) pairs and score in one event loop
            # to avoid 100 serial asyncio.run() calls.
            judge = self.reward_fn._get_judge()
            all_pairs = [(p, s) for p, ctx in zip(eval_problems, contexts) for s in ctx.snippets]
            flat_scores = judge.score_pairs_batch(all_pairs)

            # Rebuild per-problem score lists
            idx = 0
            problem_scores: list = []
            for p, ctx in zip(eval_problems, contexts):
                n = len(ctx.snippets)
                problem_scores.append(flat_scores[idx:idx + n])
                idx += n

            metric_top_k = self.config.rl.eval_metric_top_k
            all_rel_scores = []
            for case_idx, (p, ctx, scores) in enumerate(zip(eval_problems, contexts, problem_scores)):
                if ctx.snippets:
                    eval_scores = scores[:metric_top_k] if metric_top_k is not None else scores
                    weights = [1.0 / (i + 1) for i in range(len(eval_scores))]
                    total_w = sum(weights)
                    weighted_relevance = sum(w * s for w, s in zip(weights, eval_scores)) / total_w
                    all_rel_scores.append(weighted_relevance)

                    if case_idx < 10:
                        sim_scores = ctx.scores.tolist() if hasattr(ctx.scores, "tolist") else list(ctx.scores)
                        task_id_safe = p.task_id.replace("/", "_")
                        fname = os.path.join(log_dir, f"case_{case_idx:02d}_{task_id_safe}.txt")
                        lines = [f"# Case: {p.task_id}\n", "## Prompt\n", p.prompt, "\n## Retrieved Snippets\n"]
                        for i, snippet in enumerate(ctx.snippets):
                            sim = sim_scores[i] if i < len(sim_scores) else float("nan")
                            llm_score = scores[i] if i < len(scores) else float("nan")
                            lines.append(f"\n### Snippet {i + 1}  |  LLM Score: {llm_score:.4f}  |  Similarity: {sim:.4f}\n")
                            lines.append(f"Docstring: {snippet.docstring}\n")
                            lines.append(f"Code:\n{snippet.code}\n")
                        lines.append(f"\n## Weighted Relevance: {weighted_relevance:.4f}\n")
                        with open(fname, "w", encoding="utf-8") as f:
                            f.writelines(lines)

            avg_relevance = sum(all_rel_scores) / len(all_rel_scores) if all_rel_scores else 0.0
            return {"avg_snippet_relevance": avg_relevance}
        finally:
            self.encoder.train()
