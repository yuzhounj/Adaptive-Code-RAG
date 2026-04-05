import torch
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PolicyLossOutput:
    loss: torch.Tensor
    advantage: float
    entropy: float
    raw_reward: float
    pg_loss: float
    baseline_val: float


class REINFORCEPolicy:
    """
    REINFORCE policy gradient for retriever training.

    Uses a Query-level local baseline.
    Advantage = snippet_reward - mean(all_snippet_rewards_for_this_query)
    Loss = -log_prob * advantage - entropy_coeff * entropy
    """

    def __init__(self, entropy_coeff: float = 0.01):
        self.entropy_coeff = entropy_coeff

    def compute_loss(
            self,
            log_probs: torch.Tensor,  # [k] log probs of retrieved snippets
            snippet_rewards: List[float],  # [k] per-snippet relevance rewards
    ) -> PolicyLossOutput:
        """
        Compute per-snippet REINFORCE loss with Query-level baseline.
        """
        # 计算当前 Query 检索出的 K 个 Snippet 的平均分，作为局部 Baseline
        mean_reward = sum(snippet_rewards) / len(snippet_rewards) if snippet_rewards else 0.0
        baseline_val = mean_reward

        raw_advantages = torch.tensor(
            [r - baseline_val for r in snippet_rewards],
            dtype=torch.float32,
            device=log_probs.device,
        )

        # Normalize advantage scale for stable learning rate across batches.
        # Skip normalization when all rewards are equal (std ≈ 0) to avoid
        # dividing by near-zero and exploding the loss.
        adv_std = raw_advantages.std()
        advantages = raw_advantages / (adv_std + 1e-8) if adv_std > 1e-4 else raw_advantages

        # Per-snippet policy gradient loss
        pg_loss = -(log_probs * advantages).sum()

        # Entropy bonus to prevent retrieval collapse
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()

        loss = pg_loss - self.entropy_coeff * entropy

        return PolicyLossOutput(
            loss=loss,
            advantage=advantages.mean().item(),
            entropy=entropy.item(),
            raw_reward=mean_reward,
            pg_loss=pg_loss.item(),
            baseline_val=baseline_val,
        )
