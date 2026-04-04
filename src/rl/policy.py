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
    def __init__(self, entropy_coeff: float = 0.01, baseline_decay: float = 0.99):
        self.entropy_coeff = entropy_coeff
        self.global_baseline = 0.0
        self.baseline_decay = baseline_decay

    def compute_loss(self, log_probs: torch.Tensor, snippet_rewards: List[float]) -> PolicyLossOutput:
        device = log_probs.device
        rewards_tensor = torch.tensor(snippet_rewards, dtype=torch.float32, device=device)
        mean_reward = rewards_tensor.mean()

        # 更新全局基线
        if self.global_baseline == 0.0:
            self.global_baseline = mean_reward.item()
        else:
            self.global_baseline = self.global_baseline * self.baseline_decay + mean_reward.item() * (1 - self.baseline_decay)

        if len(snippet_rewards) > 1:
            std_reward = rewards_tensor.std()
            if std_reward > 1e-4:
                advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
            else:
                # 组内全等时，用全局绝对基线衡量好坏！
                advantages = rewards_tensor - self.global_baseline
        else:
            advantages = rewards_tensor - self.global_baseline

        pg_loss = -(log_probs * advantages.detach()).mean()
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()
        loss = pg_loss - self.entropy_coeff * entropy

        return PolicyLossOutput(
            loss=loss,
            advantage=advantages.abs().mean().item(), # 记录绝对优势值均值，方便观察
            entropy=entropy.item(),
            raw_reward=mean_reward.item(),
            pg_loss=pg_loss.item()
        )