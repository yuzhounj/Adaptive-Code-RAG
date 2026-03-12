import torch
import torch.nn.functional as F
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


class REINFORCEPolicy:
    """
    REINFORCE policy gradient for retriever training.

    Loss = -log_prob * advantage - entropy_coeff * entropy
    where advantage = reward - baseline
    """

    def __init__(self, baseline_decay: float = 0.99, entropy_coeff: float = 0.01):
        self.baseline = RunningMeanBaseline(decay=baseline_decay)
        self.entropy_coeff = entropy_coeff

    def compute_loss(
        self,
        log_probs: torch.Tensor,   # [k] log probs of retrieved snippets
        rewards: List[float],       # rewards from n_samples generations
    ) -> PolicyLossOutput:
        """
        Compute REINFORCE loss.

        Args:
            log_probs: gradient-attached log probs from retriever [k]
            rewards: list of rewards from n code generations

        Returns:
            PolicyLossOutput with loss tensor, advantage, and entropy
        """
        # Average reward across n_samples
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Update baseline and compute advantage
        advantage = mean_reward - self.baseline.get()
        self.baseline.update(mean_reward)

        # Scalar log prob: sum of log probs of retrieved set
        log_prob_scalar = log_probs.sum()

        # Policy gradient loss
        pg_loss = -log_prob_scalar * advantage

        # Entropy bonus to prevent retrieval collapse
        # Entropy of softmax distribution over retrieved snippets
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum()

        loss = pg_loss - self.entropy_coeff * entropy

        return PolicyLossOutput(
            loss=loss,
            advantage=advantage,
            entropy=entropy.item(),
            raw_reward=mean_reward,
        )
