import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from cslr.data.lsfb import load_datasets, load_dataloaders
from cslr.model.backbone.pose_vit import PoseViT
from cslr.model.head.projection import ProjectionHead
from cslr.trainer.contrastive import ContrastiveModule


def main():
    datasets = load_datasets("/run/media/ppoitier/ppoitier/datasets/sign-languages/lsfb/isol")
    dataloaders = load_dataloaders(datasets, batch_size=320)

    backbone = PoseViT(in_channels=150, out_channels=1024, sequence_length=48, pool='clf_token')
    projector = ProjectionHead(in_channels=1024, out_channels=128, hidden_channels=512)

    module = ContrastiveModule(backbone, projector)

    logger = TensorBoardLogger('../../logs/contrastive', name="cslr_subcon")
    trainer = pl.Trainer(max_epochs=100, logger=logger)
    trainer.fit(module, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['test'])


if __name__ == '__main__':
    main()
