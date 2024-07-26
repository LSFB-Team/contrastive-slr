import torch
from torch import nn, optim
import lightning as pl
from pytorch_metric_learning.losses import SupConLoss

from scheduler.linear_warmup import LinearSchedulerWithWarmup
from utils import get_input_mask


class ContrastiveModule(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        projector: nn.Module,
        inpput_processor: callable = None,
        generate_masks: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.criterion = SupConLoss()
        self.input_processor = inpput_processor
        self.mask = generate_masks

    def training_step(self, batch, batch_idx):
        x, labels = batch

        if self.input_processor is not None:
            x = self.input_processor(x)

        # Compute masks on the fly. Should be done in the dataloader
        if self.mask:
            masks = get_input_mask(x, self.device)
            embeddings = self.backbone(x, masks)

        else:
            embeddings = self.backbone(x)

        projections = self.projector(embeddings)
        loss = self.criterion(projections, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        if self.input_processor is not None:
            x = self.input_processor(x)

        # Compute masks on the fly. Should be done in the dataloader
        if self.mask:
            masks = get_input_mask(x, self.device)
            embeddings = self.backbone(x, masks)

        else:
            embeddings = self.backbone(x)

        projections = self.projector(embeddings)
        loss = self.criterion(projections, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearSchedulerWithWarmup(
            optimizer, n_warmup_steps=20, n_drop_steps=80, max_lr=1e-4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
