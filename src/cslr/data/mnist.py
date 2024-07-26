from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_datasets(root: str):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return {
        x: datasets.MNIST(
            root=root, train=x == "train", transform=transform, download=True
        )
        for x in ["train", "test"]
    }


def load_dataloaders(datasets, batch_size: int, sampler=None):
    return {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train" and sampler is None),
            sampler=sampler if x == "train" else None,
        )
        for x in ["train", "test"]
    }
