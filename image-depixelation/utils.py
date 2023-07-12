from typing import Union, Sequence

import torch
import torchvision
from PIL import Image
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from datasets import RandomImagePixelationDataset
import matplotlib.pyplot as plt



def create_datasets(image_path, train_ratio, width_range, height_range, size_range):
    # List all subdirectories in image_path
    all_dirs = [os.path.join(image_path, d) for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]

    # Split directories into train and test sets
    train_dirs, test_dirs = train_test_split(all_dirs, train_size=train_ratio, random_state=42)

    # Create datasets from the lists of directories
    train_dataset = RandomImagePixelationDataset(train_dirs, width_range, height_range, size_range)
    test_dataset = RandomImagePixelationDataset(test_dirs, width_range, height_range, size_range)

    return train_dataset, test_dataset


def random_augmented_image(image: Image, image_size: Union[int, Sequence[int]], seed: int) -> torch.Tensor:
    random.seed(seed)

    image = torchvision.transforms.Resize(image_size)(image)
    potential_transformations = [
        torchvision.transforms.RandomRotation(45),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter()
    ]
    image = potential_transformations.pop(random.randrange(len(potential_transformations)))(image)
    image = potential_transformations.pop(random.randrange(len(potential_transformations)))(image)
    tensor = torchvision.transforms.ToTensor()(image)
    return torch.nn.Dropout()(tensor)




def stack_with_padding(batch_as_list: list):
    max_values = []
    for values in zip(*[image[0].shape for image in batch_as_list]):
        max_values.append(max(values))
    shape = (len(batch_as_list),) + tuple(max_values)

    stacked_pixelated_images = []
    stacked_known_arrays = []
    target_arrays = []
    image_files = []

    for item in batch_as_list:
        pixelated_image, known_array, target_array, image_file = item

        padded_pixelated_image = np.zeros(shape[1:], dtype=pixelated_image.dtype)
        padded_pixelated_image[:, :pixelated_image.shape[1], :pixelated_image.shape[2]] = pixelated_image
        stacked_pixelated_images.append(padded_pixelated_image)

        padded_known_array = np.zeros(shape[1:], dtype=known_array.dtype)
        padded_known_array[:, :known_array.shape[1], :known_array.shape[2]] = known_array
        stacked_known_arrays.append(padded_known_array)

        target_arrays.append(torch.tensor(target_array))
        image_files.append(image_file)

    stacked_pixelated_images = torch.tensor(np.array(stacked_pixelated_images))
    stacked_known_arrays = torch.tensor(np.array(stacked_known_arrays))


    return stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files