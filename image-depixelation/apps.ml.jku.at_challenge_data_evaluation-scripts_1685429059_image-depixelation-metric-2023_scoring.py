import argparse
import bz2
import gzip
import lzma
import os
import shutil
import zipfile

import numpy as np


# Copied from submission_serialization.py because scoring.py must be self-contained on the challenge server
def deserialize(path_or_filehandle):
    def read(f):
        submission = []
        # Read the first byte, which represents the number of bytes that each of the following prediction array lengths
        # requires (see serialization process above)
        max_len_bytes = int.from_bytes(f.read(1), byteorder="big", signed=False)
        
        while True:
            # Read the length of the following prediction array
            length_bytes = f.read(max_len_bytes)
            # If there is nothing left (EOF), we processed the entire file. If the serialization was correct and without
            # any errors, the not-EOF means that there is guaranteed to be a length entry and then the corresponding
            # prediction array afterward, i.e., it's either length+array or EOF, and nothing in between. If, for some
            # reason, this is not the case (maybe corrupt data), then either the int.from_bytes or np.frombuffer will
            # fail and, in turn, the entire deserialization.
            if not length_bytes:
                return submission
            # We serialized the length with an offset by 1, so add it again to get the true length
            length = int.from_bytes(length_bytes, byteorder="big", signed=False) + 1
            # Read the prediction array, i.e., the file content from the current byte position c until c+length. This
            # only works because we know that every element is of data type np.uint8 = 1 byte, so the entire array must
            # be length bytes.
            prediction = np.frombuffer(f.read(length), dtype=np.uint8)
            submission.append(prediction)
    
    # If it is a path to file, wrap the read function into a with-open context manager
    if isinstance(path_or_filehandle, (str, bytes, os.PathLike)) and os.path.isfile(path_or_filehandle):
        with open(path_or_filehandle, "rb") as fh:
            return read(fh)
    # Otherwise, assume it is already a filehandle, in which case the user is responsible for proper file closing
    return read(path_or_filehandle)


def load_data(file: str):
    # Try to load the data without any file extension assumptions regarding compression. This means we simply have to
    # try multiple open calls of the different compression modules in sequential order. While this is a bruteforce
    # approach, it allows us to potentially open any file regardless of its (even incorrect or missing) extension
    
    # "mode" cannot be "rb", so set it manually to "r" (still need the parameter or the function invocation fails)
    def zip_open(file_, _):
        with zipfile.ZipFile(file_, "r") as myzip:
            return myzip.open(myzip.namelist()[0])
    
    # Try compressed versions first, and only last, try the uncompressed open function
    open_fns = [(bz2.open, True), (zip_open, True), (lzma.open, True), (gzip.open, True), (open, False)]
    for open_fn, is_compressed in open_fns:
        try:
            with open_fn(file, "rb") as f:
                return deserialize(f), is_compressed
        except (zipfile.BadZipFile, OSError, lzma.LZMAError, gzip.BadGzipFile):
            pass  # Allowed errors, just try the next module
    raise ValueError("passed file is not a (compressed) serialized submission file")


def rmse(predictions: list, targets: list):
    def rmse_(i: int, prediction_array: np.ndarray, target_array: np.ndarray):
        if prediction_array.shape != target_array.shape:
            raise IndexError(f"prediction[{i}] shape is {prediction_array.shape} but should be {target_array.shape}")
        prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
        return np.sqrt(np.mean((prediction_array - target_array) ** 2))
    
    # Compute RMSE for each sample and then return the overall mean
    rmses = [rmse_(i, prediction, target) for i, prediction, target in zip(range(len(targets)), predictions, targets)]
    return np.mean(rmses)


def scoring_file(prediction_file: str, target_file: str):
    """
    Computes the mean RMSE over all samples within two lists of NumPy arrays that are stored in the serialized files
    ``prediction_file`` (the submitted predictions) and ``target_file`` (the ground truth). Both of these files can
    optionally be compressed. The following compressions are supported:
    
    - zip compression (https://docs.python.org/3/library/zipfile.html, including the requirement of the zlib module:
      https://docs.python.org/3/library/zlib.html)
    - gzip compression (https://docs.python.org/3/library/gzip.html, also requires the zlib module)
    - bzip2 compression (https://docs.python.org/3/library/bz2.html)
    - lzma compression (https://docs.python.org/3/library/lzma.html)
    
    If the ``prediction_file`` is an uncompressed serialized file, it will be automatically compressed using the bzip2
    compression algorithm, and the original file is then overwritten by this compressed file to save disk space.
    
    :param prediction_file: Path to the serialized file containing a list of NumPy arrays of data type uint8 which
        represent the submitted predictions.
    :param target_file: Path to the serialized file containing a list of NumPy arrays of data type uint8 which represent
        the ground truth.
    :return: Mean RMSE over all sample pairs in both lists.
    """
    predictions, is_prediction_file_compressed = load_data(prediction_file)
    if not is_prediction_file_compressed:
        temp = prediction_file + "_temp"
        with open(prediction_file, "rb") as f_in:
            with bz2.open(temp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(prediction_file)
        os.rename(temp, prediction_file)
    
    if not isinstance(predictions, list):
        raise TypeError(f"expected a list of NumPy arrays as serialized file but got {type(predictions)} instead")
    if not all(isinstance(prediction, np.ndarray) and np.uint8 == prediction.dtype for prediction in predictions):
        raise TypeError("list of predictions contains elements which are not NumPy arrays of dtype uint8")
    
    targets, _ = load_data(target_file)
    if len(targets) != len(predictions):
        raise IndexError(f"list length of submitted predictions is {len(predictions)} but should be {len(targets)}")
    
    return rmse(predictions, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, help="Path to submission file.")
    parser.add_argument("--target", type=str, default=None, help="Path to target file.")
    args = parser.parse_args()
    
    rmse_loss = scoring_file(prediction_file=args.submission, target_file=args.target)
    print(rmse_loss)
