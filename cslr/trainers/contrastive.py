import lightning as pl
from pytorch_metric_learning.losses import SupConLoss
import torch
from torch import nn, optim

from cslr.schedulers.linear_warmup import LinearSchedulerWithWarmup


class ContrastiveModule(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        projection_head: nn.Module,
        max_lr: float = 1e-4,
        n_epochs: int = 100,
        n_warmup_epochs: int = 20,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = projection_head
        self.criterion = SupConLoss()

        self.max_lr = max_lr
        self.n_epochs = n_epochs
        self.n_warmup_epochs = n_warmup_epochs

        self.test_results = {
            'embeddings': [],
            'labels': [],
            'ids': [],
        }

    def training_step(self, batch, batch_idx):
        _, (features, masks), labels = batch
        embeddings = self.backbone(features, masks)
        projections = self.projector(embeddings)
        contrastive_loss = self.criterion(projections, labels)
        self.log("train_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True)
        return contrastive_loss

    def validation_step(self, batch, batch_idx):
        _, (features, masks), labels = batch
        embeddings = self.backbone(features, masks)
        projections = self.projector(embeddings)
        contrastive_loss = self.criterion(projections, labels)
        self.log("val_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        ids, (features, masks), labels = batch
        embeddings = self.backbone(features, masks)
        self.test_results['embeddings'].append(embeddings.detach().half().cpu())
        self.test_results['labels'].append(labels.detach().cpu())
        self.test_results['ids'].append(ids)

    def on_test_epoch_end(self):
        all_embeddings = torch.cat(self.test_results['embeddings'])
        all_labels = torch.cat(self.test_results['labels'])
        all_ids = sum(self.test_results['ids'], start=tuple())
        self.test_results = {'embeddings': all_embeddings, 'labels': all_labels, 'ids': all_ids}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearSchedulerWithWarmup(
            optimizer,
            n_warmup_steps=self.n_warmup_epochs,
            n_drop_steps=self.n_epochs - self.n_warmup_epochs,
            max_lr=self.max_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
