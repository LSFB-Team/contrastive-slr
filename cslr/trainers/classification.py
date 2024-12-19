from torch import nn, optim

from cslr.trainers.base import TrainerBase
from cslr.losses import FocalLoss
from cslr.metrics import ClassificationMetrics


class ClassificationModule(TrainerBase):
    def __init__(
        self,
            backbone: nn.Module,
            classification_head: nn.Module,
            n_classes: int,
            lr: float = 1e-3,
    ):
        super().__init__()
        self.backbone = backbone
        self.cls_head = classification_head
        self.criterion = FocalLoss(gamma=2.0)

        self.lr = lr
        self.save_hyperparameters('lr')

        self.train_metrics = ClassificationMetrics('train/', n_classes)
        self.val_metrics = ClassificationMetrics('val/', n_classes)

    def training_step(self, batch, batch_idx):
        instance_id, (features, masks), labels = batch
        embeddings = self.backbone(features, masks)
        logits = self.cls_head(embeddings.detach())
        cls_loss = self.criterion(logits, labels)
        self.log("train_focal_loss", cls_loss, on_step=True, on_epoch=True)

        probs = logits.softmax(dim=-1)
        metrics = self.train_metrics(probs, labels)
        self.log_metrics(metrics, labels.size(0), on_step=False, on_epoch=True)

        return cls_loss

    def validation_step(self, batch, batch_idx):
        instance_id, (features, masks), labels = batch
        embeddings = self.backbone(features, masks)
        logits = self.cls_head(embeddings.detach())
        cls_loss = self.criterion(logits, labels)
        self.log("val_focal_loss", cls_loss, on_step=True, on_epoch=True)

        probs = logits.softmax(dim=-1)
        metrics = self.val_metrics(probs, labels)
        self.log_metrics(metrics, labels.size(0), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
