import torch
from torch import Tensor
from torch.utils.data import DataLoader, default_collate

from lsfb_dataset.datasets import LSFBIsolLandmarks, LSFBIsolConfig
from sign_language_tools.common.transforms import *
from sign_language_tools.pose.transform import *


def load_datasets(root: str, max_seq_len: int = 64, n_labels: int = 400):
    transform = TransformTuple(Compose([
        Concatenate(['pose', 'left_hand', 'right_hand']),
        Padding(min_length=1),
        RandomRotation3D(mode='horizontal'),
        RandomTranslation(),
        RandomRotation2D(),
        RandomResample(min_length=10, max_length=64),
        TemporalRandomCrop(size=48),
        Padding(min_length=48, mode='repeat'),
        Clip(),
        Split({'pose': 33, 'left_hand': 21, 'right_hand': 21}),
        ApplyToAll(Flatten())
    ]))

    return {
        x: LSFBIsolLandmarks(LSFBIsolConfig(
            root=root,
            split=x,
            sequence_max_length=max_seq_len,
            n_labels=n_labels,
            transform=transform,
        ))
        for x in ['train', 'test']
    }


def _merge_feature_dicts(features_dicts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    merged_dict = dict()
    for key in features_dicts[0].keys():
        merged_dict[key] = torch.stack([d[key] for d in features_dicts]).flatten(0, 1).contiguous()
    return merged_dict


def _collate_fn(batch):
    features, targets = default_collate(batch)
    features = _merge_feature_dicts(features)
    targets = torch.stack([targets, targets]).reshape(-1)
    return features, targets


def load_dataloaders(datasets):
    return {
        x: DataLoader(datasets[x], batch_size=32, collate_fn=_collate_fn, shuffle=(x == 'train'), num_workers=7)
        for x in ['train', 'test']
    }
