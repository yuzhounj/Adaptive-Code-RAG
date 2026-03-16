import os
import random
import shutil
import time
import torch
from typing import List, Optional
from tqdm import tqdm

from src.config import TrainingConfig
from src.data.schema import CodeSnippet, HumanEvalProblem
from src.retriever.retriever import DifferentiableRetriever
from src.retriever.encoder import CodeBERTEncoder
from src.generator.llm_client import LLMClient
from src.generator.prompt_builder import build_prompt
from src.reward.reward_fn import RewardFunction
from src.rl.policy import REINFORCEPolicy
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging_utils import TrainingLogger
from src.utils.metrics import compute_pass_at_k


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
        )

        self.optimizer = torch.optim.AdamW(
            encoder.model.parameters(),
            lr=config.rl.learning_rate,
        )

        self.logger = TrainingLogger(config)
        self.global_step = 0

    def train_step(self, batch: List[HumanEvalProblem]) -> dict:
        """Run one training step on a batch of problems."""
        t0 = time.perf_counter()

        # 1. Retrieve with gradient (fast, serial)
        contexts = [self.retriever.retrieve(problem) for problem in batch]
        t1 = time.perf_counter()

        # 2. Compute per-snippet relevance rewards (no LLM generation — CSN has no test cases)
        batch_losses = []
        batch_rewards = []
        batch_advantages = []

        for problem, context in zip(batch, contexts):
            snippet_rewards = self.reward_fn.compute_snippet_rewards(
                problem, snippets=context.snippets
            )
            loss_output = self.policy.compute_loss(
                log_probs=context.log_probs,
                snippet_rewards=snippet_rewards,
            )
            batch_losses.append(loss_output.loss)
            batch_rewards.append(loss_output.raw_reward)
            batch_advantages.append(loss_output.advantage)
        t2 = time.perf_counter()

        # 3. Aggregate and backprop
        total_loss = torch.stack(batch_losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.encoder.model.parameters(),
            self.config.rl.max_grad_norm,
        )
        self.optimizer.step()
        t3 = time.perf_counter()

        return {
            "loss": total_loss.item(),
            "mean_reward": sum(batch_rewards) / len(batch_rewards),
            "mean_advantage": sum(batch_advantages) / len(batch_advantages),
            "time/retrieve": t1 - t0,
            "time/reward": t2 - t1,
            "time/backward": t3 - t2,
            "time/total": t3 - t0,
        }

    def train(
        self,
        train_problems: List[HumanEvalProblem],
        eval_problems: Optional[List[HumanEvalProblem]] = None,
        humaneval_snippets: Optional[List[CodeSnippet]] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """Main training loop."""
        os.makedirs("outputs", exist_ok=True)
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

        # Initial evaluation at step 0 (baseline before training)
        if eval_problems and humaneval_snippets and self.global_step == 0:
            eval_metrics = self.evaluate(eval_problems, humaneval_snippets)
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
                "ret": f"{metrics['time/retrieve']:.1f}s",
                "rwd": f"{metrics['time/reward']:.1f}s",
            })
            if self.global_step % rl_cfg.log_interval == 0:
                self.logger.log(metrics, step=self.global_step)

            # Refresh FAISS index
            if self.global_step % rl_cfg.index_refresh_interval == 0:
                self.retriever.refresh_index(step=self.global_step)

            # Evaluation
            if eval_problems and humaneval_snippets and self.global_step % rl_cfg.eval_interval == 0:
                eval_metrics = self.evaluate(eval_problems, humaneval_snippets)
                self.logger.log(eval_metrics, step=self.global_step, prefix="eval")
                print(f"\nStep {self.global_step} eval: {eval_metrics}")

            # Checkpoint
            if self.global_step % rl_cfg.checkpoint_interval == 0:
                ckpt_path = f"{self.config.output.checkpoint_dir}/step_{self.global_step}.pt"
                save_checkpoint(ckpt_path, self.encoder, self.optimizer, self.global_step)

        pbar.close()
        self.logger.close()

    def evaluate(self, problems: List[HumanEvalProblem], humaneval_snippets: List[CodeSnippet]) -> dict:
        """Evaluate on HumanEval using a fresh HumanEval index built with the current encoder.

        Metrics: pass@1, pass@{n_samples}, avg_snippet_relevance.
        The CSN index is temporarily swapped out and restored after eval.
        """
        self.encoder.eval()

        # Save CSN index and replace with a fresh HumanEval index
        saved_faiss = self.retriever.faiss_index
        saved_snippets = self.retriever.corpus_snippets
        from src.retriever.faiss_index import FaissIndex
        eval_faiss = FaissIndex(embedding_dim=self.config.retriever.embedding_dim)
        self.retriever.faiss_index = eval_faiss
        self.retriever.build_index(humaneval_snippets)

        try:
            with torch.no_grad():
                contexts = [self.retriever.retrieve(p) for p in tqdm(problems, desc="Eval-retrieve", leave=False)]

            prompts = [build_prompt(p, ctx.snippets) for p, ctx in zip(problems, contexts)]
            all_generated_codes = self.llm_client.generate_batch(
                prompts, n=self.config.generator.n_samples, temperature=0.0
            )
            all_rewards = [
                self.reward_fn.compute(p, codes)
                for p, codes in zip(problems, all_generated_codes)
            ]

            pass_at_1 = compute_pass_at_k(all_rewards, k=1)
            pass_at_k = compute_pass_at_k(all_rewards, k=self.config.generator.n_samples)

            all_rel_scores = []
            for p, ctx in tqdm(zip(problems, contexts), total=len(problems), desc="Eval-judge", leave=False):
                if ctx.snippets:
                    scores = self.reward_fn.compute_snippet_rewards(p, ctx.snippets)
                    all_rel_scores.append(sum(scores) / len(scores))
            avg_relevance = sum(all_rel_scores) / len(all_rel_scores) if all_rel_scores else 0.0

            return {
                "pass@1": pass_at_1,
                f"pass@{self.config.generator.n_samples}": pass_at_k,
                "avg_snippet_relevance": avg_relevance,
            }
        finally:
            # Always restore CSN index and training mode
            self.retriever.faiss_index = saved_faiss
            self.retriever.corpus_snippets = saved_snippets
            self.encoder.train()
