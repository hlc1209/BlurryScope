from os import error
import torch
import numpy as np
import scipy.io
from torch import tensor
import torch.nn as nn
from skimage import io, transform, color, filters
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import tifffile as tiff

import operator
from functools import reduce
from functools import partial

from multiprocessing.pool import ThreadPool
import time
import itertools

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torchvision.transforms import v2
import torch
import numpy as np

from logger_config import get_logger
logger = get_logger(__name__)

import functools
def debug_or_info_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import logger_config
        
        if logger_config.IS_DEBUG or logger_config.IS_INFO:
            return func(*args, **kwargs)
    return wrapper

def execute_once(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
    def reset():
        wrapper.has_run = False
    wrapper.has_run = False
    wrapper.reset = reset
    return wrapper

@debug_or_info_only
@execute_once
def analyze_image(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    shape = image.shape
    
    dtype = image.dtype
    
    min_val = np.min(image)
    max_val = np.max(image)
    
    if np.issubdtype(dtype, np.floating):
        if min_val >= 0 and max_val <= 1:
            data_range = "(0, 1)"
        else:
            data_range = f"arbitrary range ({min_val}, {max_val})"
    elif np.issubdtype(dtype, np.integer):
        if min_val >= 0 and max_val <= 255:
            data_range = "0-255"
        else:
            data_range = f"arbitrary range ({min_val}, {max_val})"
    else:
        data_range = f"arbitrary range ({min_val}, {max_val})"
    
    logger.info(f"Image analysis:")
    logger.info(f"Shape: {shape}")
    logger.info(f"Data type: {dtype}")
    logger.info(f"Data range: {data_range}")

train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomChoice([
        v2.RandomRotation(0),
        v2.RandomRotation(90),
        v2.RandomRotation(180),
        v2.RandomRotation(270)
    ]),
])

class TIFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        logger.warning(f"Loading images from {root_dir}")

        for first_level_dir in os.listdir(root_dir):
            first_level_path = os.path.join(root_dir, first_level_dir)
            if os.path.isdir(first_level_path):
                for label in os.listdir(first_level_path):
                    label_path = os.path.join(first_level_path, label)
                    if os.path.isdir(label_path):
                        for file in os.listdir(label_path):
                            if file.endswith('.tif'):
                                img_path = os.path.join(label_path, file)
                                image = tiff.imread(img_path)
                                image = np.transpose(image, (1, 2, 0, 3))
                                image = image.reshape((512, 512, 15)).transpose(2, 0, 1)
                                self.images.append(torch.tensor(image)/255.0)
                                if int(label) == 0 or int(label) == 1:
                                    self.labels.append(int(0))
                                elif int(label) == 2 or int(label) == 3:
                                    self.labels.append(int(1))
                                else:
                                    raise ValueError(f"Invalid label: {label}")
        
        analyze_image(self.images[0].numpy())
        logger.warning(f"Found {len(self.images)} images in {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        logger.debug(f"image shape: {image.shape}, label: {label}")

        if self.transform:
            image = self.transform(image)

        logger.debug(f"image shape after transform: {image.shape}, label: {label}")

        logger.debug(f"Loading image at index: {idx}, Label: {label}")
        
        return image, label
