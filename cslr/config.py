from pydantic import BaseModel
import json


class PoseViTConfig(BaseModel):
    in_channels: int = 130
    out_channels: int = 1024
    max_length: int = 64
    n_layers: int = 8
    n_heads: int = 4
    pool: str = 'cls_token'


class ProjectionConfig(BaseModel):
    in_channels: int = 1024
    hidden_channels: int = 1024
    out_channels: int = 128
    normalize_output: bool = True


class ClassificationHeadConfig(BaseModel):
    in_channels: int = 1024
    hidden_channels: tuple[int] = (728,)
    out_channels: int = 500
    use_batch_norm: bool = True
    n_epochs: int = 150
    lr: float = 1e-3


class KNNConfig(BaseModel):
    n_neighbors: int = 15
    metric: str = 'cosine'


class ExperimentConfig(BaseModel):
    root: str
    out_dir: str
    vocab_size: int
    batch_size: int = 8
    n_workers: int = 1
    debug: bool = False
    max_lr: float = 1e-4
    n_epochs: int = 100
    n_warmup_epochs: int = 20
    backbone: PoseViTConfig = PoseViTConfig()
    projection: ProjectionConfig = ProjectionConfig()
    classification: ClassificationHeadConfig = ClassificationHeadConfig()
    knn: KNNConfig = KNNConfig()


def load_config(filepath: str) -> ExperimentConfig:
    with open(filepath, 'r') as file:
        config_data = json.load(file)
    return ExperimentConfig.model_validate(config_data)
