from dataclasses import dataclass, field
from typing import Optional
import yaml
import dacite
from pathlib import Path


@dataclass
class RetrieverConfig:
    model_name: str = "microsoft/codebert-base"
    top_k: int = 5
    max_seq_len: int = 512
    embedding_dim: int = 768
    index_refresh_interval: int = 50


@dataclass
class GeneratorConfig:
    model: str = "openai/gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    n_samples: int = 4
    temperature: float = 0.8
    max_tokens: int = 512
    timeout: int = 30


@dataclass
class RewardConfig:
    mode: str = "execution"
    execution_timeout: int = 10
    llm_judge_model: str = "openai/gpt-4o-mini"
    llm_judge_api_key_env: str = "OPENAI_API_KEY"
    hybrid_execution_weight: float = 0.7


@dataclass
class RLConfig:
    learning_rate: float = 1e-5
    batch_size: int = 8
    max_steps: int = 5000
    baseline_decay: float = 0.99
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 200
    checkpoint_interval: int = 500
    log_interval: int = 10
    index_refresh_interval: int = 50


@dataclass
class DataConfig:
    train_split: float = 0.8
    cache_dir: str = "data/cache"
    corpus_dir: str = "data/corpus"
    humaneval_dir: str = "data/humaneval"


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_dir: str = "outputs/logs"
    project_name: str = "adaptive-code-rag"


@dataclass
class OutputConfig:
    checkpoint_dir: str = "outputs/checkpoints"


@dataclass
class TrainingConfig:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str = "configs/default.yaml", overrides: Optional[dict] = None) -> TrainingConfig:
    """Load configuration from YAML file with optional dot-notation overrides."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            d = data
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    return dacite.from_dict(
        data_class=TrainingConfig,
        data=data,
        config=dacite.Config(strict=False)
    )
