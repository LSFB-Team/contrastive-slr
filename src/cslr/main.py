import itertools
from data.lsfb import load_dataloaders, load_datasets
from sign_language_tools.visualization.video import VideoPlayer
from sign_language_tools.pose.mediapipe.edges import UPPER_POSE_EDGES, HAND_EDGES
import time

DATA_ROOT = "/home/jeromefink/Documents/unamur/signLanguage/Data/lsfb_v2/isol"

datasets = load_datasets(DATA_ROOT)


for elem in itertools.islice(datasets["train"], 5):
    print(elem[1])

    # First video
    player = VideoPlayer(DATA_ROOT, screenshot_dir="./screenshot", fps=50)

    player.attach_pose("Pose", elem[0][0]["pose"], connections=UPPER_POSE_EDGES)
    player.attach_pose("Left Hand", elem[0][0]["left_hand"], connections=HAND_EDGES)
    player.attach_pose("Right Hand", elem[0][0]["right_hand"], connections=HAND_EDGES)

    player.play()

    # Second video

    player = VideoPlayer(DATA_ROOT, screenshot_dir="./screenshot", fps=50)

    player.attach_pose("Pose", elem[0][1]["pose"], connections=UPPER_POSE_EDGES)
    player.attach_pose("Left Hand", elem[0][1]["left_hand"], connections=HAND_EDGES)
    player.attach_pose("Right Hand", elem[0][1]["right_hand"], connections=HAND_EDGES)

    player.play()

    time.sleep(3)
