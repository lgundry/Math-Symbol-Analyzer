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

    # Use a smaller data type (float32) to reduce memory usage
    # Optionally, use memory-mapped arrays if the dataset is too large
    db = np.memmap('db.npy', dtype=np.float32, mode='w+', shape=(num_files, 2026))  # Each row: 2025 pixels + 1 label
    labels = [""] * num_dirs

    print("Loading images into db")

    directory_index = 0
    file_index = 0
    for directory in os.listdir(path):
        directory_path = os.path.join(path, directory)
        if os.path.isdir(directory_path):
            labels[directory_index] = directory
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                
                # Open the image and resize it to 45x45 pixels
                image = Image.open(file_path).convert('L')  # Convert to grayscale
                image = image.resize((45, 45))  # Resize to 45x45
                
                # Flatten the image to a 2025-element array
                image_array = np.asarray(image).flatten()
                
                # Ensure the image array has the correct size (2025 pixels)
                assert image_array.shape[0] == 2025, f"Image size mismatch: {image_array.shape[0]} != 2025"
                
                db[file_index, :-1] = image_array  # Image data (2025 elements)
                db[file_index, -1] = directory_index  # Label (last element)
                file_index += 1
            directory_index += 1

    return db, labels


def save_np_array(np_array, filename="database.npy"):
    """Saves a numpy array to a file."""
    print("saving db")
    if isinstance(np_array, np.memmap):
        np_array.flush()  # Ensure data is written if it's a memory-mapped array
    else:
        np.save(filename, np_array)

def load_np_array(filename):
    print("Loading db from file")
    """Loads a numpy array from a file."""
    return np.load(filename)

def save_array(array, filename):
    """Saves a Python object (e.g., list) to a file using pickle."""
    print("Saving standard array")
    with open(filename, 'wb') as file:
        pickle.dump(array, file)

def load_array(filename):
    print("Loading standard array")
    """Loads a Python object from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)
