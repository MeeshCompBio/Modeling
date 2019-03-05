import tensorflow as tf
import numpy as np
import retro

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque

import random
import warnings

warnings.filterwarnings('ignore')

env = retro.make(game="SpaveInvaders-Atari2600")
print("The size of our frame is: ", env.action_spave.n)
print("The action size is : ", env.action_space.n)

possible_actions = np.array(np.identity(env.action_space.n,
                                        dtaype=int).tolist()
                            )


def preprocess_frame(frame):
    # Greyscale frame, color is not important
    gray = rgb2gray(frame)
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]\n",
    cropped_frame = gray[8:-12, 4:-12]
    # Normalize Pixel Values\n",
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame  # 110x84x1 frame"

stack_size = 4

