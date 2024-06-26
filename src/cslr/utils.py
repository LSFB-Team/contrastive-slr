import os
import torch


def load_dataset_path(default="./isol"):
    if os.path.exists("./DATA_PATH"):
        with open("./DATA_PATH", "r") as f:
            DATA_PATH = f.read().strip()
        return DATA_PATH
    return default


def get_input_mask(data):
    b, _, _ = data.shape

    # Summing on axes 2 as if all the values are 0 it means that the pose is not present
    mask = torch.sum(data, axis=2)

    # Adding element for the classification token
    ones = torch.ones(b, 1)
    mask = torch.cat([ones, mask], dim=1)
    mask = mask.bool()

    # Reverting the mask as True means "masked" in pytorch
    return ~mask
