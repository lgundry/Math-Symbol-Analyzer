import os
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    db, labels = loadFromDir("data/extracted_images")

# generate a 2 dimensional numpy array (matrix). Each column is an individual image, where each row is a particular feature. The final row of each column contains an identifier and should not be used in the network
def loadFromDir(path):
    
    #Count the number of files and directories in given path
    num_files = 0
    num_dirs = 0
    for root, dirs, files in os.walk(path):
        num_files += len(files)
        num_dirs += len(dirs)

    db = np.zeros((2026, num_files))
    labels = [""] * num_dirs

    directory_index = 0
    file_index = 0
    for directory in os.listdir(path):
        labels[directory_index] = directory
        directory_path = os.path.join(path, directory)
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            image_array = np.asarray(Image.open(file_path)).flatten()
            db[:-1, file_index] = image_array
            db[-1, file_index] = directory_index
            file_index += 1
        directory_index += 1

    return db, labels
        

def load_from_file(path):
    return np.load(path)

main()