import matplotlib.pyplot as plt
import seaborn as sns

from torch import Tensor
import lightning as pl


class TrainerBase(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_metrics(
        self,
        metrics: dict[str, any],
        batch_size: int,
        on_step: bool = False,
        on_epoch: bool = True,
    ):
        for name, value in metrics.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                self.log(
                    name,
                    value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )
