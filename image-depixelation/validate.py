import glob
import hashlib
import os

import PIL
import numpy as np
from PIL import Image

MAX_FILE_SIZE = 250_000
MIN_H_DIM = 100
MIN_W_DIM = 100


def _log(log_file, input_dir, filename, error_code):
    # Save relative path to log, starting after the (common) input directory
    print(f"{filename[len(input_dir) + 1:]},{error_code}", file=log_file)


def validate_images(input_dir: str, log_file: str, formatter: str = "07d"):
    if not os.path.isdir(input_dir):
        raise ValueError(f"'{input_dir}' is not an existing directory")
    
    # Create directories if they do not exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Get sorted list of file names (exclude directories)
    input_dir = os.path.abspath(input_dir)
    filenames = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
    filenames = [f for f in filenames if os.path.isfile(f)]
    filenames.sort()
    
    # Prepare counter for valid files
    n_valid = 0
    # Prepare set to store image hashes in (use set for fast lookup)
    hashes = set()
    
    # Create clean logfile
    with open(log_file, "w", encoding="utf-8") as f:
        # Loop over file names and sequentially check for validity
        # If a rule is violated, log it and skip to next file
        for filename in filenames:
            # Check for rule 1
            if not (filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".jpeg")
                    or filename.endswith(".JPEG")):
                _log(f, input_dir, filename, 1)
                continue
            
            # Check for rule 2
            if os.path.getsize(filename) > MAX_FILE_SIZE:
                _log(f, input_dir, filename, 2)
                continue
            
            # Check for rule 3
            try:
                image = Image.open(filename)
            except (FileNotFoundError, PIL.UnidentifiedImageError, ValueError, TypeError):
                _log(f, input_dir, filename, 3)
                continue
            
            # Check for rule 4
            np_image = np.array(image)
            # Check: number of dimensions, height and width, number of channels, mode
            if (len(np_image.shape) != 2 and len(np_image.shape) != 3) or \
                    (np_image.shape[0] < MIN_H_DIM or np_image.shape[1] < MIN_W_DIM) or \
                    (len(np_image.shape) == 3 and (np_image.shape[2] != 3 or image.mode != "RGB")):
                _log(f, input_dir, filename, 4)
                continue
            
            # Image itself is no longer needed (image data is already copied in np_image)
            image.close()
            
            # Check for rule 5
            if (np.std(np_image, axis=(0, 1)) <= 0).all():
                _log(f, input_dir, filename, 5)
                continue
            
            # Check for rule 6
            # Get hash value from image data
            hashing_function = hashlib.sha256()
            hashing_function.update(np_image)
            np_image_hash = hashing_function.digest()
            # Check if image data existed in other file too
            if np_image_hash in hashes:
                _log(f, input_dir, filename, 6)
                continue
            
            # Add image data hash to set of hashes
            hashes.add(np_image_hash)
            
            n_valid += 1
    
    # Return number of valid files
    return n_valid, len(filenames)
