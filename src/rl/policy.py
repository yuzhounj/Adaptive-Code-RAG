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


class GRPOPolicy:
    """
    Group Relative Policy Optimization (GRPO) policy.
    No baseline tracking needed. Normalizes rewards within the retrieved group.
    """

    def __init__(self, entropy_coeff: float = 0.01):
        self.entropy_coeff = entropy_coeff

    def compute_loss(
            self,
            log_probs: torch.Tensor,  # [G] sampled log probs
            snippet_rewards: List[float],  # [G] raw rewards for the sampled snippets
    ) -> PolicyLossOutput:

        device = log_probs.device
        rewards_tensor = torch.tensor(snippet_rewards, dtype=torch.float32, device=device)

        mean_reward = rewards_tensor.mean()

        # === GRPO 组内归一化 ===
        if len(snippet_rewards) > 1:
            std_reward = rewards_tensor.std()
            # std 加上极小值防止除以 0；如果所有奖励完全一样(std < 1e-4)，优势设为0
            if std_reward > 1e-4:
                advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
            else:
                advantages = torch.zeros_like(rewards_tensor)
        else:
            advantages = torch.zeros_like(rewards_tensor)

        # 策略梯度损失： L = - mean(log_prob * advantage)
        # 注意用 detach 截断 advantage 的梯度（以防万一）
        pg_loss = -(log_probs * advantages.detach()).mean()

        # 熵正则化 bonus，鼓励分布的多样性
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()

        # 最终损失
        loss = pg_loss - self.entropy_coeff * entropy

        return PolicyLossOutput(
            loss=loss,
            advantage=advantages.mean().item(),  # 归一化后理论上接近 0
            entropy=entropy.item(),
            raw_reward=mean_reward.item(),
            pg_loss=pg_loss.item()
        )