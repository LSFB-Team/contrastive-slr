import pickle

import click
import torch

torch.set_float32_matmul_precision("high")

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from cslr.config import load_config, ExperimentConfig
from cslr.data.slr import load_datasets, load_dataloaders
from cslr.modules.backbones import PoseViT
from cslr.modules.heads import ProjectionHead, ClassificationHead
from cslr.trainers import ContrastiveModule, ClassificationModule


def train_contrastive_model(dataloaders, config: ExperimentConfig):
    backbone_config = config.backbone
    backbone = PoseViT(
        in_channels=backbone_config.in_channels,
        out_channels=backbone_config.out_channels,
        max_length=backbone_config.max_length,
        n_heads=backbone_config.n_heads,
        n_layers=backbone_config.n_layers,
        pool=backbone_config.pool,
    )

    projection_config = config.projection
    projection_head = ProjectionHead(
        in_channels=projection_config.in_channels,
        out_channels=projection_config.out_channels,
        hidden_channels=projection_config.hidden_channels,
        normalize_output=projection_config.normalize_output,
    )

    module = ContrastiveModule(
        backbone=backbone,
        projection_head=projection_head,
        max_lr=config.max_lr,
        n_epochs=config.n_epochs,
        n_warmup_epochs=config.n_warmup_epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_contrastive_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        dirpath=f"{config.out_dir}/checkpoints",
    )
    trainer = pl.Trainer(
        fast_dev_run=config.debug,
        max_epochs=config.n_epochs,
        logger=TensorBoardLogger(save_dir=f"{config.out_dir}/logs"),
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )

    print(
        "Training finished. Loading best checkpoint and computing all final embeddings..."
    )
    trainer.test(
        module,
        dataloaders=dataloaders["testing"],
        ckpt_path=None if config.debug else checkpoint_callback.best_model_path,
        verbose=True,
    )
    print(f"Computed {len(module.test_results['embeddings'])} embeddings.")
    if not config.debug:
        print("Saving embeddings...")
        with open(f"{config.out_dir}/embeddings.pkl", "wb") as file:
            pickle.dump(module.test_results, file)
        print("Embeddings saved.")


def train_classification_head(dataloaders, config: ExperimentConfig):
    backbone_config = config.backbone
    backbone = PoseViT(
        in_channels=backbone_config.in_channels,
        out_channels=backbone_config.out_channels,
        max_length=backbone_config.max_length,
        n_heads=backbone_config.n_heads,
        n_layers=backbone_config.n_layers,
        pool=backbone_config.pool,
    )

    cls_head_config = config.classification
    cls_head = ClassificationHead(
        in_channels=cls_head_config.in_channels,
        out_channels=cls_head_config.out_channels,
    )

    module = ClassificationModule(
        backbone, cls_head, n_classes=config.vocab_size, lr=cls_head_config.lr
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_focal_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        dirpath=f"{config.out_dir}/cls_head/checkpoints",
    )
    trainer = pl.Trainer(
        fast_dev_run=config.debug,
        max_epochs=cls_head_config.n_epochs,
        logger=TensorBoardLogger(save_dir=f"{config.out_dir}/cls_head/logs"),
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )


@click.command()
@click.option("--config-path", required=True, type=click.Path(exists=True))
def main(config_path: str):
    config = load_config(config_path)

    print("Loading data...")
    root, vocab_size = config.root, config.vocab_size
    datasets = load_datasets(
        urls={
            "training": f"{root}/shards/{vocab_size}/" + "shard_{000000..000003}.tar",
            "validation": f"{root}/shards/{vocab_size}/shard_000004.tar",
            "testing": f"{root}/shards/{vocab_size}/" + "shard_{000000..000004}.tar",
        }
    )
    dataloaders = load_dataloaders(datasets, config.batch_size, config.n_workers)

    print("Training backbone using contrastive learning...")
    train_contrastive_model(dataloaders, config)
    print("Contrastive model training finished.")

    print("Training classification head...")
    train_classification_head(dataloaders, config)
    print("Classification head training finished.")


if __name__ == "__main__":
    main()
