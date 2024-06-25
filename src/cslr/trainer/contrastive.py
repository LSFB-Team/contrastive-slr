import torch
from torch import nn, optim
import lightning as pl
from pytorch_metric_learning.losses import SupConLoss

from cslr.scheduler.linear_warmup import LinearSchedulerWithWarmup


class ContrastiveModule(pl.LightningModule):
    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.criterion = SupConLoss()

    def training_step(self, batch, batch_idx):
        features, labels = batch
        x = torch.cat([features['pose'], features['left_hand'], features['right_hand']], dim=-1).float()
        embeddings = self.backbone(x)
        projections = self.projector(embeddings)
        loss = self.criterion(projections, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        x = torch.cat([features['pose'], features['left_hand'], features['right_hand']], dim=-1).float()
        embeddings = self.backbone(x)
        projections = self.projector(embeddings)
        loss = self.criterion(projections, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = LinearSchedulerWithWarmup(optimizer, n_warmup_steps=20, n_drop_steps=80, max_lr=1e-3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
