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
from src.rl.policy import PPOPolicy
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logging_utils import TrainingLogger
from transformers import get_linear_schedule_with_warmup


class RLTrainer:
    """Full closed-loop RL training loop upgraded with PPO and Gold Snippet Injection."""

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

        self.policy = PPOPolicy(
            initial_entropy_coeff=config.rl.entropy_coeff,
            min_entropy_coeff=0.001,
            clip_eps=0.2
        )

        self.optimizer = torch.optim.AdamW(
            encoder.model.parameters(),
            lr=config.rl.learning_rate,
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.rl.warmup_steps,
            num_training_steps=config.rl.max_steps
        )

        self.logger = TrainingLogger(config)
        self.global_step = 0
        self.ppo_epochs = 3

    def train_step(self, batch: List[HumanEvalProblem]) -> dict:
        # --- 1. Rollout Phase (No Gradient) ---
        self.encoder.eval()
        with torch.no_grad():
            rollout_contexts = [self.retriever.retrieve(problem) for problem in batch]

        # 【终极修复 1】：打破冷启动僵局，强行注入 Ground Truth 黄金样本
        for i, problem in enumerate(batch):
            snippets = rollout_contexts[i].snippets

            # 在全量知识库中精准捞出这道题对应的真实代码片段
            gold_snippet = next((s for s in self.retriever.corpus_snippets if s.snippet_id == problem.task_id), None)

            if gold_snippet:
                # 如果 FAISS 检索能力太弱，前 k 个里居然没有正确答案
                if not any(s.snippet_id == gold_snippet.snippet_id for s in snippets):
                    if len(snippets) > 0:
                        # 把最没用的最后一个凑数片段踢掉，替换成真正的黄金片段
                        snippets[-1] = gold_snippet
                    else:
                        snippets.append(gold_snippet)

        all_snippet_rewards = []
        old_log_probs_list = []

        for problem, context in zip(batch, rollout_contexts):
            # 获取 LLM Judge 的打分（现在它必定能看到黄金片段，并打出 0.9+ 的极高分）
            snippet_rewards = self.reward_fn.compute_snippet_rewards(
                problem, snippets=context.snippets
            )
            all_snippet_rewards.append(snippet_rewards)

            # 由于我们强行替换了片段，必须重新无梯度计算一次正确的 old_log_probs
            with torch.no_grad():
                old_log_probs, _ = self.retriever.rescore(problem, context.snippets)
            old_log_probs_list.append(old_log_probs.detach())

        flat = [r for rewards in all_snippet_rewards for r in rewards]
        raw_mean_reward = sum(flat) / len(flat) if flat else 0.0

        # --- 2. Optimization Phase (PPO Epochs) ---
        self.encoder.eval()  # 全程防 Dropout 干扰
        final_metrics = {}
        last_grad_norm = 0.0  # 【终极修复 2】：缓存上一次的梯度范数，杜绝 0.0000 误报

        for epoch in range(self.ppo_epochs):
            batch_losses, batch_rewards, batch_advantages = [], [], []
            batch_entropies, batch_pg_losses, batch_ratios = [], [], []
            batch_kl = []

            for i, problem in enumerate(batch):
                snippets = rollout_contexts[i].snippets
                snippet_rewards = all_snippet_rewards[i]
                old_log_probs = old_log_probs_list[i]

                new_log_probs, _ = self.retriever.rescore(problem, snippets)

                loss_output = self.policy.compute_loss(
                    log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    snippet_rewards=snippet_rewards,
                )

                batch_losses.append(loss_output.loss)
                batch_rewards.append(loss_output.raw_reward)
                batch_advantages.append(loss_output.advantage)
                batch_entropies.append(loss_output.entropy)
                batch_pg_losses.append(loss_output.pg_loss)
                batch_ratios.append(loss_output.ratio)

                with torch.no_grad():
                    kl = (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum().item()
                    batch_kl.append(kl)

            total_loss = torch.stack(batch_losses).mean()
            n = len(batch_rewards)

            # 无论是否触发早停，先把基础 metrics 装好，使用 last_grad_norm 兜底
            final_metrics = {
                "loss": total_loss.item(),
                "pg_loss": sum(batch_pg_losses) / n if n else 0,
                "entropy": sum(batch_entropies) / n if n else 0,
                "mean_reward": raw_mean_reward,
                "mean_advantage": sum(batch_advantages) / n if n else 0,
                "ppo_ratio": sum(batch_ratios) / n if n else 0,
                "grad_norm": last_grad_norm,
                "snippet_rewards": flat,
            }

            # --- KL Early Stopping Shield ---
            mean_kl = sum(batch_kl) / len(batch_kl) if batch_kl else 0.0
            if mean_kl > 0.05 and epoch > 0:
                tqdm.write(f"  [Early Stop] KL={mean_kl:.4f} > 0.05 at epoch {epoch}")
                break

            self.optimizer.zero_grad()
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.encoder.model.parameters(),
                self.config.rl.max_grad_norm,
            ).item()

            # 更新缓存，并覆盖写入 dict
            last_grad_norm = grad_norm
            final_metrics["grad_norm"] = grad_norm

            self.optimizer.step()

        self.scheduler.step()

        # --- 3. Dynamic Entropy Decay ---
        self.policy.update_entropy_coeff(self.global_step, self.config.rl.max_steps)
        final_metrics["ent_coeff"] = self.policy.entropy_coeff

        return final_metrics

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

        self.retriever.refresh_index(step=self.global_step)

        if eval_problems and self.global_step == 0:
            eval_metrics = self.evaluate(eval_problems)
            self.logger.log(eval_metrics, step=0, prefix="eval")
            print(f"\nStep 0 eval (baseline): {eval_metrics}")

        while self.global_step < rl_cfg.max_steps:
            batch = random.sample(train_problems, min(rl_cfg.batch_size, len(train_problems)))

            metrics = self.train_step(batch)
            self.global_step += 1
            pbar.update(1)

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "rew": f"{metrics['mean_reward']:.3f}",
            })

            if self.global_step % rl_cfg.log_interval == 0:
                self.logger.log(metrics, step=self.global_step)

            if self.global_step % rl_cfg.metrics_log_interval == 0:
                self._metrics_log.write(
                    f"[step {self.global_step}] "
                    f"loss={metrics['loss']:.4f}  pg={metrics['pg_loss']:.4f}  ent={metrics['entropy']:.4f} (coef={metrics['ent_coeff']:.4f}) | "
                    f"rew={metrics['mean_reward']:.4f}  ratio={metrics['ppo_ratio']:.4f} | "
                    f"gnorm={metrics['grad_norm']:.4f} | "
                    f"snippet_rewards=[{', '.join(f'{r:.3f}' for r in metrics['snippet_rewards'])}]\n"
                )
                self._metrics_log.flush()

            if eval_problems and self.global_step % rl_cfg.eval_interval == 0:
                eval_metrics = self.evaluate(eval_problems)
                self.logger.log(eval_metrics, step=self.global_step, prefix="eval")
                print(f"\nStep {self.global_step} eval: {eval_metrics}")

            if self.global_step % rl_cfg.index_refresh_interval == 0:
                self.retriever.refresh_index(step=self.global_step)

            if self.global_step % rl_cfg.checkpoint_interval == 0:
                ckpt_path = f"{self.config.output.checkpoint_dir}/step_{self.global_step}.pt"
                save_checkpoint(ckpt_path, self.encoder, self.optimizer, self.global_step)

        pbar.close()
        self.logger.close()
        self._metrics_log.close()

    def evaluate(self, eval_problems: List[HumanEvalProblem]) -> dict:
        eval_logs_root = os.path.join("outputs", "eval_logs")
        if self.global_step == 0 and os.path.exists(eval_logs_root):
            shutil.rmtree(eval_logs_root)
        log_dir = os.path.join(eval_logs_root, f"step_{self.global_step}")
        os.makedirs(log_dir, exist_ok=True)

        self.encoder.eval()
        try:
            with torch.no_grad():
                contexts = [self.retriever.retrieve(p) for p in tqdm(eval_problems, desc="Eval-retrieve", leave=False)]

            judge = self.reward_fn._get_judge()
            all_pairs = [(p, s) for p, ctx in zip(eval_problems, contexts) for s in ctx.snippets]
            flat_scores = judge.score_pairs_batch(all_pairs)

            idx = 0
            problem_scores: list = []
            for p, ctx in zip(eval_problems, contexts):
                n = len(ctx.snippets)
                problem_scores.append(flat_scores[idx:idx + n])
                idx += n

            all_rel_scores = []
            for case_idx, (p, ctx, scores) in enumerate(zip(eval_problems, contexts, problem_scores)):
                if ctx.snippets:
                    weights = [1.0 / (i + 1) for i in range(len(scores))]
                    total_w = sum(weights)
                    weighted_relevance = sum(w * s for w, s in zip(weights, scores)) / total_w
                    all_rel_scores.append(weighted_relevance)

                    if case_idx < 10:
                        sim_scores = ctx.scores.tolist() if hasattr(ctx.scores, "tolist") else list(ctx.scores)
                        task_id_safe = p.task_id.replace("/", "_")
                        fname = os.path.join(log_dir, f"case_{case_idx:02d}_{task_id_safe}.txt")
                        lines = [f"# Case: {p.task_id}\n", "## Prompt\n", p.prompt, "\n## Retrieved Snippets\n"]
                        for i, snippet in enumerate(ctx.snippets):
                            sim = sim_scores[i] if i < len(sim_scores) else float("nan")
                            llm_score = scores[i] if i < len(scores) else float("nan")
                            lines.append(
                                f"\n### Snippet {i + 1}  |  LLM Score: {llm_score:.4f}  |  Similarity: {sim:.4f}\n")
                            lines.append(f"Docstring: {snippet.docstring}\n")
                            lines.append(f"Code:\n{snippet.code}\n")
                        lines.append(f"\n## Weighted Relevance: {weighted_relevance:.4f}\n")
                        with open(fname, "w", encoding="utf-8") as f:
                            f.writelines(lines)

            avg_relevance = sum(all_rel_scores) / len(all_rel_scores) if all_rel_scores else 0.0
            return {"avg_snippet_relevance": avg_relevance}
        finally:
            self.encoder.train()