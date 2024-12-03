from torch import nn, optim

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Recall, Precision

from cslr.trainers.base import TrainerBase


class ClassificationMetrics(MetricCollection):
    def __init__(self, n_classes: int, **kwargs):
        acc_args = dict(task="multiclass", num_classes=n_classes, ignore_index=-1)
        metrics = {
            "accuracy": Accuracy(**acc_args),
            "top_3": Accuracy(top_k=3, **acc_args),
            "top_5": Accuracy(top_k=5, **acc_args),
            "top-10": Accuracy(top_k=10, **acc_args),
            "macro_accuracy": Accuracy(average="macro", **acc_args),
            "recall": Recall(average=None, **acc_args),
            "precision": Precision(average=None, **acc_args),
        }
        super().__init__(metrics, **kwargs)


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
        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr
        self.save_hyperparameters('lr')

        self.train_metrics = ClassificationMetrics(n_classes)
        self.val_metrics = ClassificationMetrics(n_classes)

    def training_step(self, batch, batch_idx):
        instance_id, (features, masks), labels = batch
        embeddings = self.backbone(features, masks)
        logits = self.cls_head(embeddings.detach())
        cls_loss = self.criterion(logits, labels)
        self.log("cls_train_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=True)

        probs = logits.softmax(dim=-1)
        metrics = self.train_metrics(probs, labels)
        self.log_metrics(metrics, labels.size(0), on_step=False, on_epoch=True)

        return cls_loss

    def validation_step(self, batch, batch_idx):
        instance_id, (features, masks), labels = batch
        embeddings = self.backbone(features, masks)
        logits = self.cls_head(embeddings.detach())
        cls_loss = self.criterion(logits, labels)
        self.log("cls_val_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=True)

        probs = logits.softmax(dim=-1)
        metrics = self.val_metrics(probs, labels)
        self.log_metrics(metrics, labels.size(0), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
