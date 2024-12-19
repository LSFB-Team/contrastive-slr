import os

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


def save_metrics(log_dir: str, metrics: dict[str, any]):
    torch.save(metrics, f"{log_dir}/metrics.pth")


def log_metrics_to_dir(log_dir: str, metrics: dict[str, any], prefix: str = ""):
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)
    for name, value in metrics.items():
        if isinstance(value, Tensor) and value.numel() == 1:
            logger.add_scalar(f"{prefix}{name}", value)
    logger.flush()
    logger.close()

