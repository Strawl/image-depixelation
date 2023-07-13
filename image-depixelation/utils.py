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
import uuid



def create_datasets(image_path, train_ratio, width_range, height_range, size_range):
    # List all subdirectories in image_path
    all_dirs = [os.path.join(image_path, d) for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]

    # Split directories into train and test sets
    train_dirs, test_dirs = train_test_split(all_dirs, train_size=train_ratio, random_state=42)

    # Create datasets from the lists of directories
    train_dataset = RandomImagePixelationDataset(train_dirs, width_range, height_range, size_range)
    test_dataset = RandomImagePixelationDataset(test_dirs, width_range, height_range, size_range)

    return train_dataset, test_dataset


def save(model: torch.nn.Module, models_dir: str):
    # Create the directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Generate a unique identifier for the model
    model_id = uuid.uuid4()

    # Save the model with the UUID filename
    model_path = os.path.join(models_dir, f"{model_id}.pth")
    torch.save(model.state_dict(), model_path)
    return model_path

def load(model: torch.nn.Module, models_dir: str, model_id: str):
    # Construct the model path from the directory and the UUID
    model_path = os.path.join(models_dir, f"{model_id}.pth")

    # Load the model state_dict from the file
    model.load_state_dict(torch.load(model_path))
    return model


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

        padded_known_array = np.ones(shape[1:], dtype=known_array.dtype)
        padded_known_array[:, :known_array.shape[1], :known_array.shape[2]] = known_array
        stacked_known_arrays.append(padded_known_array)

        target_arrays.append(torch.tensor(target_array))
        image_files.append(image_file)

    stacked_pixelated_images_numpy = np.array(stacked_pixelated_images)
    stacked_known_arrays_numpy = np.array(stacked_known_arrays)
    stacked_pixelated_images = torch.tensor(
        np.concatenate((stacked_pixelated_images_numpy, stacked_known_arrays_numpy), axis=1)
    )
    stacked_known_arrays = torch.tensor(np.array(stacked_known_arrays))

    return stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files


def plot_losses(train_losses: list, eval_losses: list):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.title('Training and Evaluation Losses')
    plt.show()
