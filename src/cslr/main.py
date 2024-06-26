from data.lsfb import load_datasets, load_dataloaders
from model.backbone.pose_vit import PoseViT
from model.head.projection import ProjectionHead
from model.head.classification import ClassificationHead
from trainer.classification import ClassificationModule
from trainer.contrastive import ContrastiveModule
from utils import load_dataset_path
from sampler import MultinomialBalancedSampler


backbone = PoseViT(
    in_channels=150, out_channels=1024, sequence_length=48, pool="clf_token"
)
projector = ProjectionHead(in_channels=1024, out_channels=128, hidden_channels=512)

backbone = ContrastiveModule.load_from_checkpoint(
    "./checkpoints/epoch=99-step=15200.ckpt",
    backbone=backbone,
    projector=projector,
)
