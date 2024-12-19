from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Recall, Precision, AUROC, ROC, ConfusionMatrix


class ClassificationMetrics(MetricCollection):
    def __init__(self, prefix: str, n_classes: int, **kwargs):
        acc_args = dict(task="multiclass", num_classes=n_classes, ignore_index=-1)
        metrics = {
            "accuracy": Accuracy(**acc_args),
            "top-3": Accuracy(top_k=3, **acc_args),
            "top-5": Accuracy(top_k=5, **acc_args),
            "top-10": Accuracy(top_k=10, **acc_args),
            "macro_accuracy": Accuracy(average="macro", **acc_args),
        }
        super().__init__(metrics, prefix=prefix, **kwargs)


class ClassificationFullMetrics(MetricCollection):
    def __init__(self, prefix: str, n_classes: int, **kwargs):
        acc_args = dict(task="multiclass", num_classes=n_classes, ignore_index=-1)
        metrics = {
            "accuracy": Accuracy(**acc_args),
            "top-3": Accuracy(top_k=3, **acc_args),
            "top-5": Accuracy(top_k=5, **acc_args),
            "top-10": Accuracy(top_k=10, **acc_args),
            "macro_accuracy": Accuracy(average="macro", **acc_args),
            "recall": Recall(average=None, **acc_args),
            "precision": Precision(average=None, **acc_args),
            "roc": ROC(**acc_args),
            "auc-roc": AUROC(**acc_args),
            "conf-mat": ConfusionMatrix(**acc_args),
        }
        super().__init__(metrics, prefix=prefix, **kwargs)

