import torch
import webdataset as wds
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

from sign_language_tools.common.transforms import Compose, Randomize, TransformTuple
from sign_language_tools.pose.transform import *


def _process_sample(sample: dict) -> dict:
    return {
        "features": {
            "upper_pose": sample["pose.upper_pose.npy"],
            "left_hand": sample["pose.left_hand.npy"],
            "right_hand": sample["pose.right_hand.npy"],
            # "lips": sample["pose.lips.npy"],
        },
        "label": int(sample["label.idx"]),
    }


class SLRWebDataset(Dataset):
    def __init__(self, url: str, transform=None, show_progress: bool = False):
        super().__init__()
        self.transform = transform
        self.instance_ids: list[str] = []
        self.samples: list[dict] = []
        web_dataset = wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.map(_process_sample),
        )
        for sample in tqdm(web_dataset, disable=not show_progress, unit=" samples"):
            self.samples.append(sample)
            self.instance_ids.append(sample['__key__'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sample_id = sample["__key__"]
        features = sample["features"]
        target = sample["label"]
        if self.transform is not None:
            sample = self.transform(features)
        return sample_id, sample, target


def default_transforms(mode: str = "training"):
    normalization_transforms = Compose([
        DropCoordinates("z"),
        NormalizeEdgeLengths(unitary_edge=(11, 12)),
        CenterOnLandmarks((11, 12)),
    ])

    if mode == "validation" or mode == "testing":
        return Compose(
            [
                Concatenate(["upper_pose", "left_hand", "right_hand"]),
                normalization_transforms,
                TemporalCrop(size=64, location='start'),
                Clip(),
                Flatten(),
                Padding(min_length=64, mode="constant", return_mask=True),
            ]
        )

    return TransformTuple(
        Compose(
            [
                Concatenate(["upper_pose", "left_hand", "right_hand"]),
                normalization_transforms,
                TemporalRandomCrop(size=64),
                Randomize(GaussianNoise(0.002), probability=0.6),
                Randomize(HorizontalFlip(), probability=0.3),
                Randomize(RandomRotation2D(angle_range=(-0.3, 0.3)), probability=0.6),
                Randomize(
                    RandomTranslation(dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2)),
                    probability=0.6,
                ),
                Randomize(RandomScale(min_scale=0.5, max_scale=1.5), probability=0.2),
                Clip(),
                Flatten(),
                Padding(min_length=64, mode="constant", return_mask=True),
            ]
        )
    )


def load_datasets(
    urls: dict[str, str],
    transforms: dict[str, str] | None = None,
):
    if transforms is None:
        transforms = {x: default_transforms(mode=x) for x in ["training", "validation", "testing"]}
    datasets = {}
    for mode in ["training", "validation", "testing"]:
        print(f"-- loading {mode} dataset...", flush=True)
        datasets[mode] = SLRWebDataset(urls[mode], transforms[mode], show_progress=True)
    return datasets


def load_dataloaders(
    datasets: dict[str, SLRWebDataset],
    batch_size: int,
    n_workers: int,
):
    def _training_collate_fn(batch):
        batch = default_collate(batch)
        sample_ids, ((features_1, mask_1), (features_2, mask_2)), targets = batch
        return (
            sample_ids,
            (
                torch.concat((features_1, features_2), dim=0).float(),
                torch.concat((mask_1, mask_2), dim=0).byte(),
            ),
            torch.concat((targets, targets), dim=0).long(),
        )

    def _validation_collate_fn(batch):
        batch = default_collate(batch)
        sample_ids, (features, masks), targets = batch
        return sample_ids, (features.float(), masks.byte()), targets.long()

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=(x == "training"),
            num_workers=n_workers,
            collate_fn=_training_collate_fn if x == "training" else _validation_collate_fn,
        )
        for x in ["training", "validation", "testing"]
    }

    return dataloaders
