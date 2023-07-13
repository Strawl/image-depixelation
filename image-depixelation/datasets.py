from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
from torchvision import transforms
from PIL import Image
import config
import dill
import pickle



def _to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")
    
    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    
    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255
    
    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def _prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim < 3 or image.shape[-3] != 1:
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (..., 1, H, W)")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")
    
    # The (height, width) slices to extract the area that should be pixelated. Since we
    # need this multiple times, specify the slices explicitly instead of using [:] notation
    area = (..., slice(y, y + height), slice(x, x + width))
    
    # This returns already a copy, so we are independent of "image"
    pixelated_image = _pixelate(image, x, y, width, height, size)
    
    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False
    
    # Create a copy to avoid that "target_array" and "image" point to the same array
    target_array = image[area].copy()
    
    return pixelated_image, known_array, target_array


def _pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    # Need a copy since we overwrite data directly
    image = image.copy()
    curr_x = x
    
    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            image[block] = image[block].mean()
            curr_y += size
        curr_x += size
    
    return image


def _get_absolute_path(path):
    expanded_path = os.path.expanduser(path)
    absolute_path = os.path.abspath(expanded_path)
    return absolute_path


class RandomImagePixelationDataset(Dataset):
    def __init__(
            self,
            image_dirs: list, 
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int],
            dtype: Optional[type] = None
    ):

        self.image_paths = []

        for image_dir in image_dirs:
            image_dir = os.path.abspath(image_dir)
            filelist = glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
            self.image_paths.extend([_get_absolute_path(f) for f in filelist if f.endswith(".jpg") and os.path.isfile(f)])

        self.image_paths.sort()

        if width_range[0] < 2 or height_range[0] < 2 or size_range[0] < 2:
            raise ValueError("The minimum sizes cannot be smaller than 2")
        if width_range[0] > width_range[1] or height_range[0] > height_range[1] or size_range[0] > size_range[1]:
            raise ValueError("The minimum sizes cannot exceed the maximum sizes")
        
        self.resize_transforms = transforms.Compose([
                transforms.Resize(size=config.IMG_SIZE),
                transforms.CenterCrop(size=(config.IMG_SIZE, config.IMG_SIZE)),
        ])

        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = self.resize_transforms(image)
        numpy_array = np.array(image, dtype=self.dtype)
        image_width, image_height = image.size
        grayscale_image = _to_grayscale(numpy_array)
        generator = np.random.default_rng(seed=index)
        width = generator.integers(low=self.width_range[0], high=self.width_range[1] + 1)
        height = generator.integers(low=self.height_range[0], high=self.height_range[1] + 1)
        width = min(width, image_width)
        height = min(height, image_height)
        x = generator.integers(low=0, high=image_width - width + 1)
        y = generator.integers(low=0, high=image_height - height + 1)
        size = generator.integers(low=self.size_range[0], high=self.size_range[1] + 1)
        pixelated_image, known_array, target_array = _prepare_image(grayscale_image, x, y, width, height, size)
        pixelated_image = (pixelated_image / 255).astype(np.float32)
        target_array = (target_array / 255).astype(np.float32)

        #full_image = np.concatenate((pixelated_image, known_array), axis=0)
        return pixelated_image, known_array, target_array, self.image_paths[index]


    def __len__(self):
        return len(self.image_paths)


import numpy as np

class ImageDepixelationTestDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as file:
            data_dict = pickle.load(file)
        
        self.pixelated_images = data_dict['pixelated_images']
        self.known_arrays = data_dict['known_arrays']

    def __getitem__(self, index):
        pixelated_image = self.pixelated_images[index]
        pixelated_image = (pixelated_image / 255).astype(np.float32)
        known_array = self.known_arrays[index]

        full_image = np.concatenate((pixelated_image, known_array), axis=0)

        return full_image, known_array

    def __len__(self):
        return len(self.pixelated_images)



