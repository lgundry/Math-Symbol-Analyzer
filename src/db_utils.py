import os
import numpy as np
from PIL import Image
import pickle

def loadFromDir(path):
    """
    Generates a database from a directory of images.
    Each column in the database corresponds to a flattened image.
    The last row of each column contains the directory index as a label.
    """
    # Count the number of files and directories in the given path
    num_files = 0
    num_dirs = 0
    for root, dirs, files in os.walk(path):
        num_files += len(files)
        num_dirs += len(dirs)

    # Initialize the database and labels
    db = np.zeros((2026, num_files))
    labels = [""] * num_dirs

    directory_index = 0
    file_index = 0
    for directory in os.listdir(path):
        directory_path = os.path.join(path, directory)
        if os.path.isdir(directory_path):  # Skip non-directory files
            labels[directory_index] = directory
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                image_array = np.asarray(Image.open(file_path)).flatten()
                db[:-1, file_index] = image_array
                db[-1, file_index] = directory_index
                file_index += 1
            directory_index += 1

    return db, labels

def save_np_array(np_array, filename="database.npy"):
    """Saves a numpy array to a file."""
    np.save(filename, np_array)

def load_np_array(filename):
    """Loads a numpy array from a file."""
    return np.load(filename)

def save_array(array, filename):
    """Saves a Python object (e.g., list) to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(array, file)

def load_array(filename):
    """Loads a Python object from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)
