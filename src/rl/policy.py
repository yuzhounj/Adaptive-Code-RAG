import torch
import torch.nn.functional as F
import math
from typing import List, Optional
from dataclasses import dataclass


class RunningMeanBaseline:
    """Exponential moving average baseline to reduce REINFORCE variance."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._value: float = 0.0
        self._initialized: bool = False

    def update(self, reward: float) -> None:
        if not self._initialized:
            self._value = reward
            self._initialized = True
        else:
            self._value = self.decay * self._value + (1 - self.decay) * reward

    def get(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0
        self._initialized = False


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

    Loss = -log_prob * advantage - entropy_coeff * entropy
    where advantage = reward - baseline (ema_baseline) or group-relative (grpo_softmax)
    """

    def __init__(
        self,
        baseline_decay: float = 0.99,
        entropy_coeff: float = 0.01,
        advantage_method: str = "ema_baseline",
        grpo_temperature: float = 1.0,
    ):
        self.baseline = RunningMeanBaseline(decay=baseline_decay)
        self.entropy_coeff = entropy_coeff
        self.advantage_method = advantage_method
        self.grpo_temperature = grpo_temperature

        if advantage_method not in ["ema_baseline", "grpo_softmax"]:
            raise ValueError(f"Unknown advantage_method: {advantage_method}")

    def _compute_grpo_advantages(
        self,
        snippet_rewards: List[float],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute GRPO advantages using softmax normalization within group.

        Args:
            snippet_rewards: per-snippet relevance scores [k]
            device: torch device for tensor creation

        Returns:
            advantages tensor of shape [k]
        """
        if not snippet_rewards:
            return torch.zeros(0, device=device, dtype=torch.float32)

        k = len(snippet_rewards)

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(snippet_rewards, dtype=torch.float32, device=device)

        # Apply temperature scaling: logits = rewards / temperature
        logits = rewards_tensor / self.grpo_temperature

        # Softmax to get probability distribution
        probs = F.softmax(logits, dim=0)

        # Advantages: probability - uniform probability (1/k)
        advantages = probs - 1.0 / k

        return advantages

    def compute_loss(
        self,
        log_probs: torch.Tensor,       # [k] log probs of retrieved snippets
        snippet_rewards: List[float],  # [k] per-snippet relevance rewards
    ) -> PolicyLossOutput:
        """
        Compute per-snippet REINFORCE loss.

        Each snippet gets its own advantage = snippet_reward - baseline.
        Loss = -sum_i(log_probs[i] * advantages[i]) - entropy_coeff * entropy

        Args:
            log_probs: gradient-attached log probs from retriever [k]
            snippet_rewards: per-snippet LLM relevance scores [k]

        Returns:
            PolicyLossOutput with loss tensor, mean advantage, and entropy
        """
        device = log_probs.device
        mean_reward = sum(snippet_rewards) / len(snippet_rewards) if snippet_rewards else 0.0

        if self.advantage_method == "ema_baseline":
            # Original EMA baseline method
            baseline_val = self.baseline.get()
            self.baseline.update(mean_reward)

            raw_advantages = torch.tensor(
                [r - baseline_val for r in snippet_rewards],
                dtype=torch.float32,
                device=device,
            )

            # Normalize advantage scale for stable learning rate across batches.
            # Skip normalization when all rewards are equal (std ≈ 0) to avoid
            # dividing by near-zero and exploding the loss.
            adv_std = raw_advantages.std()
            advantages = raw_advantages / (adv_std + 1e-8) if adv_std > 1e-4 else raw_advantages

        elif self.advantage_method == "grpo_softmax":
            # GRPO with softmax normalization within group
            advantages = self._compute_grpo_advantages(snippet_rewards, device)
            baseline_val = mean_reward  # For logging, use mean reward as reference

            # Optional: normalize GRPO advantages for consistent scale
            # This helps maintain similar gradient magnitudes across different groups
            adv_std = advantages.std()
            if adv_std > 1e-4:
                advantages = advantages / (adv_std + 1e-8)
        else:
            raise ValueError(f"Unknown advantage_method: {self.advantage_method}")

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
