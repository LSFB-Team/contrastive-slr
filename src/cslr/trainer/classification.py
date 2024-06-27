import torch
from torch import nn, optim
import lightning as pl
from torchmetrics.classification import Accuracy
from utils import get_input_mask


class ClassificationModule(pl.LightningModule):
    def __init__(
        self, backbone: nn.Module, classification_head: nn.Module, n_classes: int
    ):
        super().__init__()
        self.backbone = backbone
        self.head = classification_head
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.train_top5 = Accuracy("multiclass", num_classes=n_classes, top_k=5)
        self.train_top10 = Accuracy("multiclass", num_classes=n_classes, top_k=10)

        self.val_acc = Accuracy("multiclass", num_classes=n_classes)
        self.val_top5 = Accuracy("multiclass", num_classes=n_classes, top_k=5)
        self.val_top10 = Accuracy("multiclass", num_classes=n_classes, top_k=10)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        x = torch.cat(
            [features["pose"], features["left_hand"], features["right_hand"]], dim=-1
        ).float()

        masks = get_input_mask(x, self.device)

        embeddings = self.backbone(x, masks)
        logits = self.head(embeddings.detach())
        loss = self.criterion(logits, labels)
        self.log("cls_train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = logits.argmax(dim=-1)
        self.train_acc(preds, labels)
        self.train_top5(logits, labels)
        self.train_top10(logits, labels)
        self.log("cls_train_acc", self.train_acc, on_epoch=True)
        self.log("cls_train_top5", self.train_top5, on_epoch=True)
        self.log("cls_train_top10", self.train_top10, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        x = torch.cat(
            [features["pose"], features["left_hand"], features["right_hand"]], dim=-1
        ).float()

        masks = get_input_mask(x, self.device)

        embeddings = self.backbone(x, masks)
        logits = self.head(embeddings.detach())
        loss = self.criterion(logits, labels)
        self.log("cls_val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = logits.argmax(dim=-1)
        self.val_acc(preds, labels)
        self.val_top5(logits, labels)
        self.val_top10(logits, labels)
        self.log("cls_val_acc", self.val_acc, on_epoch=True)
        self.log("cls_val_top5", self.val_top5, on_epoch=True)
        self.log("cls_val_top10", self.val_top10, on_epoch=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3)
