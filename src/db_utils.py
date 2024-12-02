import os
import numpy as np
from PIL import Image
import pickle

def loadFromDir(path):
    """
    Generates a database from a directory of images.
    Each row in the database corresponds to a flattened image.
    The last column of each row contains the directory index as a label.
    """
    # Count the number of files and directories in the given path
    num_files = sum(len(files) for _, _, files in os.walk(path))
    num_dirs = sum(len(dirs) for _, dirs, _ in os.walk(path))

    # Initialize the database and labels
    db = np.zeros((num_files, 2026))  # Each row: 2025 pixels + 1 label
    labels = [""] * num_dirs

    directory_index = 0
    file_index = 0
    for directory in os.listdir(path):
        directory_path = os.path.join(path, directory)
        if os.path.isdir(directory_path):
            labels[directory_index] = directory
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                image_array = np.asarray(Image.open(file_path)).flatten()
                db[file_index, :-1] = image_array  # Image data (2025 elements)
                db[file_index, -1] = directory_index  # Label (last element)
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
