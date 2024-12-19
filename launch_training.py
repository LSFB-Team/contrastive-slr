import click
import time

import torch
torch.set_float32_matmul_precision("medium")

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from cslr.config import load_config, ExperimentConfig
from cslr.data.slr import load_datasets, load_dataloaders
from cslr.modules.backbones import PoseViT
from cslr.modules.heads import ProjectionHead, ClassificationHead
from cslr.trainers import ContrastiveModule, ClassificationModule
from cslr.evaluators.knn import evaluate_knn_classifier


def train_contrastive_model(exp_id: str, dataloaders, config: ExperimentConfig):
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
        dirpath=f"{config.out_dir}/checkpoints/{exp_id}",
    )
    trainer = pl.Trainer(
        fast_dev_run=config.debug,
        max_epochs=config.n_epochs,
        logger=TensorBoardLogger(save_dir=f"{config.out_dir}/logs/{exp_id}"),
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
        torch.save(module.test_results, f"{config.out_dir}/embeddings_{exp_id}.pth")
        print("Embeddings saved.")


def train_classification_head(exp_id: str, dataloaders, config: ExperimentConfig):
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
        hidden_channels=cls_head_config.hidden_channels,
        out_channels=cls_head_config.out_channels,
        use_batch_norm=cls_head_config.use_batch_norm,
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
        dirpath=f"{config.out_dir}/cls_head/checkpoints/{exp_id}",
    )
    trainer = pl.Trainer(
        fast_dev_run=config.debug,
        max_epochs=cls_head_config.n_epochs,
        logger=TensorBoardLogger(save_dir=f"{config.out_dir}/cls_head/logs/{exp_id}"),
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
    exp_id = str(int(time.time()))
    config = load_config(config_path)

    print("Loading data...", flush=True)
    root, vocab_size = config.root, config.vocab_size
    datasets = load_datasets(
        urls={
            "training": f"{root}/shards/{vocab_size}/" + "shard_{000000..000002}.tar",
            "validation": f"{root}/shards/{vocab_size}/" + "shard_{000003..000004}.tar",
            "testing": f"{root}/shards/{vocab_size}/" + "shard_{000000..000004}.tar",
        }
    )
    print("Loading dataloaders...")
    dataloaders = load_dataloaders(datasets, config.batch_size, config.n_workers)

    print("Training backbone using contrastive learning...")
    train_contrastive_model(exp_id, dataloaders, config)
    print("Contrastive model training finished.")

    print("Training classification head...")
    train_classification_head(exp_id, dataloaders, config)
    print("Classification head training finished.")

    if not config.debug:
        print("Evaluate KNN classifier...")
        evaluate_knn_classifier(
            instance_ids={x: datasets[x].instance_ids for x in ['training', 'validation']},
            embeddings_path=f"{config.out_dir}/embeddings_{exp_id}.pth",
            log_dir=f"{config.out_dir}/knn/{exp_id}",
            knn_config=config.knn,
        )
    print("Done.")


if __name__ == "__main__":
    main()
