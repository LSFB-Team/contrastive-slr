import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from cslr.data.lsfb import load_datasets, load_dataloaders
from cslr.model.backbone.pose_vit import PoseViT
from cslr.model.head.projection import ProjectionHead
from cslr.trainer.contrastive import ContrastiveModule
from cslr.utils import load_dataset_path
from cslr.sampler import MultinomialBalancedSampler

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
    dataloaders = load_dataloaders(datasets, batch_size=320, sampler=sampler)

    backbone = PoseViT(
        in_channels=150, out_channels=1024, sequence_length=48, pool="clf_token"
    )
    projector = ProjectionHead(in_channels=1024, out_channels=128, hidden_channels=512)

    module = ContrastiveModule(backbone, projector)

    logger = TensorBoardLogger("./logs/contrastive", name="cslr_subcon")
    trainer = pl.Trainer(max_epochs=100, logger=logger)
    trainer.fit(
        module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


if __name__ == "__main__":
    main()
