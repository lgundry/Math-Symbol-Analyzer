import os
import numpy as np
import pickle
import json
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    db, labels = loadFromDir("data/extracted_images")

    save_np_array(db)
    save_array(labels, "labels.plk")

    # for testing purposes:
    save_readable(db, labels)

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

def save_array(array, filename):
    with open(filename, 'wb') as file:
        pickle.dump(array, file)

def save_np_array(np_array):
    np.save("database", np_array)

def load_array(filename):
    with open(filename, 'rb') as file:
        array = pickle.load(file)
    return array

def save_readable(np_array, array):
    np.savetxt("database_readable.txt", np_array)
    with open("labels_readable.txt", 'w') as file:
        json.dump(array, file)

main()