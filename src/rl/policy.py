import torch
from typing import List
from dataclasses import dataclass


@dataclass
class PolicyLossOutput:
    loss: torch.Tensor
    advantage: float
    entropy: float
    raw_reward: float
    pg_loss: float
    baseline_val: float
    ratio: float  # [Optimization 2] Track PPO probability ratio


class PPOPolicy:
    """
    [Optimization 2 & 3] Proximal Policy Optimization with Dynamic Entropy Decay.
    Replaces standard REINFORCE for highly stable RL fine-tuning.
    """

    def __init__(
            self,
            initial_entropy_coeff: float = 0.01,
            min_entropy_coeff: float = 0.001,
            clip_eps: float = 0.2
    ):
        self.initial_entropy_coeff = initial_entropy_coeff
        self.entropy_coeff = initial_entropy_coeff
        self.min_entropy_coeff = min_entropy_coeff
        self.clip_eps = clip_eps

    def update_entropy_coeff(self, current_step: int, total_steps: int):
        """[Optimization 3] Linearly decay entropy to shift from Exploration to Exploitation."""
        # Decide decay span (e.g., decay over the first 80% of training)
        decay_span = total_steps * 0.8
        decay_ratio = min(1.0, current_step / decay_span) if decay_span > 0 else 1.0
        self.entropy_coeff = self.initial_entropy_coeff - decay_ratio * (
                    self.initial_entropy_coeff - self.min_entropy_coeff)

    def compute_loss(
            self,
            log_probs: torch.Tensor,  # [k] current log probs (with grad)
            old_log_probs: torch.Tensor,  # [k] old log probs (detached)
            snippet_rewards: List[float],  # [k] per-snippet rewards
    ) -> PolicyLossOutput:
        mean_reward = sum(snippet_rewards) / len(snippet_rewards) if snippet_rewards else 0.0
        baseline_val = mean_reward

        raw_advantages = torch.tensor(
            [r - baseline_val for r in snippet_rewards],
            dtype=torch.float32,
            device=log_probs.device,
        )

        # Normalize advantage
        adv_std = raw_advantages.std()
        advantages = raw_advantages / (adv_std + 1e-8) if adv_std > 1e-4 else raw_advantages

        # [Optimization 2] PPO Clipped Surrogate Objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        pg_loss = -torch.min(surr1, surr2).sum()

        # Entropy bonus
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()

        # Total Loss
        loss = pg_loss - self.entropy_coeff * entropy

        return PolicyLossOutput(
            loss=loss,
            advantage=advantages.mean().item(),
            entropy=entropy.item(),
            raw_reward=mean_reward,
            pg_loss=pg_loss.item(),
            baseline_val=baseline_val,
            ratio=ratio.mean().item(),
        )