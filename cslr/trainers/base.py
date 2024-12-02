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
        logger = self.logger.experiment
        for name, value in metrics.items():
            if isinstance(value, Tensor) and value.numel() > 1:
                fig = None
                if len(value.shape) == 1:
                    log_fn = logger.add_scalars
                    log_value = {str(i): v for i, v in enumerate(value)}
                elif len(value.shape) == 2:
                    log_fn = logger.add_figure
                    fig = plt.figure(figsize=(8, 8))
                    sns.heatmap(value.detach().cpu().numpy(), annot=True, fmt=".ég")
                    plt.title(f"{name} - Epoch {self.current_epoch}")
                    plt.tight_layout()
                    log_value = fig
                else:
                    raise ValueError(
                        f"Cannot log tensor of shape {value.shape}. "
                        "Only 1D and 2D tensors are supported."
                    )
                try:
                    if on_step:
                        log_fn(f"{name}/step", log_value, self.global_step)
                    if on_epoch:
                        log_fn(f"{name}/epoch", log_value, self.current_epoch)
                finally:
                    if fig is not None:
                        plt.close()
            else:
                self.log(
                    name,
                    value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    batch_size=batch_size,
                )

