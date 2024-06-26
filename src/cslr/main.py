import itertools
from data.lsfb import load_dataloaders, load_datasets
from sign_language_tools.visualization.video import VideoPlayer
from sign_language_tools.pose.mediapipe.edges import UPPER_POSE_EDGES, HAND_EDGES
import time

DATA_ROOT = "/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/isol"

datasets = load_datasets(DATA_ROOT)


for elem in itertools.islice(datasets["train"], 5):
    print(elem[0][0]["pose"].shape)

    # First video
