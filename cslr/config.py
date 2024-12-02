from dataclasses import dataclass
import json


@dataclass(frozen=True)
class PoseViTConfig:
    in_channels: int = 130
    out_channels: int = 1024
    max_length: int = 64
    n_layers: int = 8
    n_heads: int = 4
    pool: str = 'cls_token'


@dataclass(frozen=True)
class ProjectionConfig:
    in_channels: int = 1024
    hidden_channels: int = 1024
    out_channels: int = 128
    normalize_output: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    root: str
    out_dir: str
    vocab_size: int
    batch_size: int = 8
    n_workers: int = 1
    debug: bool = False
    max_lr: float = 1e-4
    n_epochs: int = 100
    n_warmup_epochs: int = 20
    backbone = PoseViTConfig()
    projection = ProjectionConfig()


def load_config(filepath: str) -> ExperimentConfig:
    with open(filepath, 'r') as file:
        config_data = json.load(file)
    return ExperimentConfig(**config_data)