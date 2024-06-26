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
    label_indices = [
        datasets["train"].targets[inst_id] for inst_id in datasets["train"].instances
    ]
    sampler = MultinomialBalancedSampler(label_indices, num_samples=50000)
    return sampler


def main():
    datasets = load_datasets(DATA_PATH)
    sampler = load_sampler(datasets)
    dataloaders = load_dataloaders(datasets, batch_size=128, sampler=sampler)

    backbone = ContrastiveModule.load_from_checkpoint(
        "./checkpoints/epoch=99-step=15200.ckpt"
    )

    classif = ClassificationHead(in_channels=1024, out_channels=400)

    module = ClassificationModule(backbone, classif)

    logger = TensorBoardLogger("./logs/contrastive", name="cslr_classif")
    trainer = pl.Trainer(max_epochs=10, logger=logger, default_root_dir="./checkpoints")
    trainer.fit(
        module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


if __name__ == "__main__":
    main()
