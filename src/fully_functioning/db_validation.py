import os
import numpy as np
from PIL import Image

def validateDB(db, labels, path):
    """
    Validates the database against the original images in the directory.
    Ensures each image in the database matches the original file,
    and the labels correspond to the correct directory.
    """
    file_index = 0
    for directory in os.listdir(path):
        directory_path = os.path.join(path, directory)
        if os.path.isdir(directory_path):  # Skip non-directory files
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                
                # Load and flatten the image
                image_array = np.asarray(Image.open(file_path)).flatten()
                
                # Check if the image data matches the database
                assert np.array_equal(db[file_index, :-1], image_array), \
                    f"Data mismatch for file: {file_path}"
                
                # Check if the label matches the directory
                assert labels[int(db[file_index, -1])] == directory, \
                    f"Label mismatch for file: {file_path}"
                
                file_index += 1

    print("Validation successful! Database matches original images and labels.")

