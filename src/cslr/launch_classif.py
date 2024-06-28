import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from data.lsfb import load_datasets, load_dataloaders
from model.backbone.pose_vit import PoseViT
from model.head.projection import ProjectionHead
from model.head.classification import ClassificationHead
from trainer.classification import ClassificationModule
from trainer.contrastive import ContrastiveModule
from utils import load_dataset_path
from sampler import MultinomialBalancedSampler

DATA_PATH = load_dataset_path(
    "/run/media/ppoitier/ppoitier/datasets/sign-languages/lsfb/isol"
)


def load_sampler(datasets):
    train_label_indices = [
        datasets["train"].targets[inst_id] for inst_id in datasets["train"].instances
    ]
    test_label_indices = [
        datasets["test"].targets[inst_id] for inst_id in datasets["test"].instances
    ]

    samplers = dict()

    samplers["train"] = MultinomialBalancedSampler(
        train_label_indices, num_samples=50000
    )
    samplers["test"] = MultinomialBalancedSampler(test_label_indices, num_samples=50000)

    return samplers


def main():
    datasets = load_datasets(DATA_PATH)
    samplers = load_sampler(datasets)
    dataloaders = load_dataloaders(datasets, batch_size=128, sampler=samplers["train"])

    backbone = PoseViT(
        in_channels=150, out_channels=1024, sequence_length=48, pool="clf_token"
    )
    projector = ProjectionHead(in_channels=1024, out_channels=128, hidden_channels=512)

    backbone = ContrastiveModule.load_from_checkpoint(
        "./checkpoints/epoch=149-step=14250.ckpt",
        backbone=backbone,
        projector=projector,
    ).backbone

    classif = ClassificationHead(in_channels=1024, out_channels=400)

    module = ClassificationModule(backbone, classif, 400)

    logger = TensorBoardLogger("./logs/contrastive", name="cslr_classif")
    trainer = pl.Trainer(max_epochs=10, logger=logger, default_root_dir="./checkpoints")
    trainer.fit(
        module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


if __name__ == "__main__":
    main()
