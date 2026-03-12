import pytest
import torch
from src.rl.policy import REINFORCEPolicy, RunningMeanBaseline, PolicyLossOutput


def test_running_mean_baseline():
    baseline = RunningMeanBaseline(decay=0.99)
    assert baseline.get() == 0.0

    baseline.update(1.0)
    assert baseline.get() == 1.0

    baseline.update(0.0)
    # After second update: 0.99 * 1.0 + 0.01 * 0.0 = 0.99
    assert abs(baseline.get() - 0.99) < 1e-6


def test_reinforce_policy_loss():
    policy = REINFORCEPolicy(baseline_decay=0.99, entropy_coeff=0.01)

    # Create fake log_probs with gradient
    log_probs = torch.tensor([-0.5, -1.2, -0.8], requires_grad=True)
    rewards = [1.0, 0.0, 1.0, 1.0]  # 3/4 pass rate

    output = policy.compute_loss(log_probs, rewards)

    assert isinstance(output, PolicyLossOutput)
    assert isinstance(output.loss, torch.Tensor)
    assert output.loss.requires_grad
    assert isinstance(output.advantage, float)
    assert isinstance(output.entropy, float)
    assert output.raw_reward == pytest.approx(0.75)


def test_reinforce_loss_backward():
    policy = REINFORCEPolicy()
    log_probs = torch.tensor([-0.5, -1.2, -0.8], requires_grad=True)
    rewards = [1.0, 1.0]

    output = policy.compute_loss(log_probs, rewards)
    output.loss.backward()

    assert log_probs.grad is not None
    assert not torch.all(log_probs.grad == 0)


def test_baseline_variance_reduction():
    """Verify baseline reduces effective advantage magnitude over time."""
    policy = REINFORCEPolicy(baseline_decay=0.9)

    # Simulate repeated reward of 0.5
    for _ in range(50):
        policy.baseline.update(0.5)

    # Advantage should now be close to 0 for reward=0.5
    advantage = 0.5 - policy.baseline.get()
    assert abs(advantage) < 0.05
