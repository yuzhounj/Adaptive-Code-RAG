import random
import torch
from typing import List, Optional
from tqdm import tqdm

from src.config import TrainingConfig
from src.data.schema import HumanEvalProblem
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
        batch_losses = []
        batch_rewards = []
        batch_advantages = []

        for problem in batch:
            # 1. Retrieve with gradient
            context = self.retriever.retrieve(problem)
            log_prob = self.retriever.compute_log_prob(context)  # scalar with grad

            # 2. Build prompt and generate
            prompt = build_prompt(problem, context.snippets)
            generated_codes = self.llm_client.generate(
                prompt, n=self.config.generator.n_samples
            )

            # 3. Compute rewards (no gradient)
            rewards = self.reward_fn.compute(problem, generated_codes)

            # 4. REINFORCE loss
            loss_output = self.policy.compute_loss(
                log_probs=context.log_probs,
                rewards=rewards,
            )

            batch_losses.append(loss_output.loss)
            batch_rewards.append(loss_output.raw_reward)
            batch_advantages.append(loss_output.advantage)

        # 5. Aggregate and backprop
        total_loss = torch.stack(batch_losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.encoder.model.parameters(),
            self.config.rl.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "mean_reward": sum(batch_rewards) / len(batch_rewards),
            "mean_advantage": sum(batch_advantages) / len(batch_advantages),
        }

    def train(
        self,
        train_problems: List[HumanEvalProblem],
        eval_problems: Optional[List[HumanEvalProblem]] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """Main training loop."""
        if resume_from:
            self.global_step = load_checkpoint(
                resume_from, self.encoder, self.optimizer
            )
            print(f"Resumed from step {self.global_step}")

        rl_cfg = self.config.rl

        pbar = tqdm(total=rl_cfg.max_steps, initial=self.global_step, desc="Training")

        while self.global_step < rl_cfg.max_steps:
            # Sample batch
            batch = random.sample(train_problems, min(rl_cfg.batch_size, len(train_problems)))

            # Train step
            metrics = self.train_step(batch)
            self.global_step += 1
            pbar.update(1)

            # Logging
            if self.global_step % rl_cfg.log_interval == 0:
                self.logger.log(metrics, step=self.global_step)
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.3f}",
                })

            # Refresh FAISS index
            if self.global_step % rl_cfg.index_refresh_interval == 0:
                self.retriever.refresh_index()

            # Evaluation
            if eval_problems and self.global_step % rl_cfg.eval_interval == 0:
                eval_metrics = self.evaluate(eval_problems)
                self.logger.log(eval_metrics, step=self.global_step, prefix="eval")
                print(f"\nStep {self.global_step} eval: {eval_metrics}")

            # Checkpoint
            if self.global_step % rl_cfg.checkpoint_interval == 0:
                ckpt_path = f"{self.config.output.checkpoint_dir}/step_{self.global_step}.pt"
                save_checkpoint(ckpt_path, self.encoder, self.optimizer, self.global_step)

        pbar.close()
        self.logger.close()

    def evaluate(self, problems: List[HumanEvalProblem]) -> dict:
        """Evaluate pass@k on eval set."""
        all_rewards = []

        for problem in tqdm(problems, desc="Evaluating", leave=False):
            context = self.retriever.retrieve(problem)
            prompt = build_prompt(problem, context.snippets)
            generated_codes = self.llm_client.generate(
                prompt, n=self.config.generator.n_samples
            )
            rewards = self.reward_fn.compute(problem, generated_codes)
            all_rewards.append(rewards)

        pass_at_1 = compute_pass_at_k(all_rewards, k=1)
        pass_at_k = compute_pass_at_k(all_rewards, k=self.config.generator.n_samples)

        return {
            "pass@1": pass_at_1,
            f"pass@{self.config.generator.n_samples}": pass_at_k,
        }
