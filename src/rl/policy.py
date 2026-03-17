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
    pg_loss: float
    baseline_val: float


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
        mean_reward = sum(snippet_rewards) / len(snippet_rewards) if snippet_rewards else 0.0

        # Compute per-snippet advantages: clip negative advantages to zero.
        # This prevents all-zero-reward batches from producing destructive
        # negative gradients while preserving the absolute quality signal.
        baseline_val = self.baseline.get()
        self.baseline.update(mean_reward)

        advantages = torch.tensor(
            [max(r - baseline_val, 0.0) for r in snippet_rewards],
            dtype=torch.float32,
            device=log_probs.device,
        )

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
