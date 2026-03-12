import torch
from pathlib import Path
from typing import Optional


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra: Optional[dict] = None,
) -> None:
    """Save model + optimizer state."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load checkpoint. Returns the saved step number."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    step = state.get("step", 0)
    print(f"Loaded checkpoint from {path} (step {step})")
    return step
