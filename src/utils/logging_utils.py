import os
from typing import Optional
from src.config import TrainingConfig


class TrainingLogger:
    """Unified logger supporting W&B and TensorBoard."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._writer = None
        self._wandb = None

        if config.logging.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(config.logging.log_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=config.logging.log_dir)
                print(f"TensorBoard logging to {config.logging.log_dir}")
            except ImportError:
                print("TensorBoard not available, skipping")

        if config.logging.use_wandb:
            try:
                import wandb
                wandb.init(project=config.logging.project_name)
                self._wandb = wandb
                print(f"W&B logging to project {config.logging.project_name}")
            except ImportError:
                print("W&B not available, skipping")

    def log(self, metrics: dict, step: int, prefix: str = "train") -> None:
        """Log metrics to all configured backends."""
        prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if self._writer:
            for key, value in prefixed.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, global_step=step)

        if self._wandb:
            self._wandb.log({**prefixed, "step": step})

    def close(self) -> None:
        if self._writer:
            self._writer.close()
        if self._wandb:
            self._wandb.finish()
