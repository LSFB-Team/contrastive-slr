import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from data.mnist import load_datasets, load_dataloaders
from model.backbone.mnist_cnn import MNIST_CNN
from model.head.projection import ProjectionHead
from trainer.contrastive import ContrastiveModule
from utils import load_dataset_path
from sampler import MultinomialBalancedSampler

DATA_PATH = "./mnist"


def main():
    datasets = load_datasets(DATA_PATH)
    dataloaders = load_dataloaders(datasets, batch_size=512)

    backbone = MNIST_CNN()
    projector = ProjectionHead(in_channels=64, out_channels=32, hidden_channels=64)

    module = ContrastiveModule(backbone, projector)

    logger = TensorBoardLogger("./logs/contrastive", name="mnist_subcon")
    trainer = pl.Trainer(max_epochs=20, logger=logger, default_root_dir="./checkpoints")
    trainer.fit(
        module,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


if __name__ == "__main__":
    main()
