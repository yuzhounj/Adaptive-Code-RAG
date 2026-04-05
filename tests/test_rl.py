import pytest
import torch
from src.rl.policy import REINFORCEPolicy, PolicyLossOutput


def test_reinforce_policy_loss():
    policy = REINFORCEPolicy(entropy_coeff=0.01)

    # Create fake log_probs with gradient
    log_probs = torch.tensor([-0.5, -1.2, -0.8, -1.0], requires_grad=True)
    rewards = [1.0, 0.0, 1.0, 1.0]  # 3/4 pass rate

    output = policy.compute_loss(log_probs, rewards)

    assert isinstance(output, PolicyLossOutput)
    assert isinstance(output.loss, torch.Tensor)
    assert output.loss.requires_grad
    assert isinstance(output.advantage, float)
    assert isinstance(output.entropy, float)
    # The baseline should be exactly the mean of the rewards in the new Query-level logic
    assert output.raw_reward == pytest.approx(0.75)
    assert output.baseline_val == pytest.approx(0.75)


def test_reinforce_loss_backward():
    policy = REINFORCEPolicy(entropy_coeff=0.01)

    log_probs = torch.tensor([-0.5, -1.2], requires_grad=True)
    rewards = [1.0, 0.0]  # Different rewards to ensure non-zero advantages

    output = policy.compute_loss(log_probs, rewards)
    output.loss.backward()

    assert log_probs.grad is not None
    assert not torch.all(log_probs.grad == 0)