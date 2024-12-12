import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

# Define augmentation pipeline
seq = iaa.Sequential([
    iaa.Crop(percent=(0, 0.1)),  # random crops
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # Gaussian blur
    iaa.LinearContrast((0.75, 1.5)),  # Contrast adjustment
    iaa.AdditiveGaussianNoise(scale=(0.0, 0.05*255)),  # Add noise
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)

# Root directories
root_directory = "data/extracted_images"
output_directory = "data/augmented_images"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Traverse input directory
for subdir, _, files in os.walk(root_directory):
    relative_path = os.path.relpath(subdir, root_directory)
    output_subdir = os.path.join(output_directory, relative_path)
    os.makedirs(output_subdir, exist_ok=True)  # Create subdirectory in output

    for file in files:
        input_path = os.path.join(subdir, file)
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Read image and preprocess
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Skipping {input_path}, could not read image.")
                    continue
                
                image_resized = cv2.resize(image, (45, 45))  # Resize to 45x45

                # Save original image
                original_output_path = os.path.join(output_subdir, f"original_{file}")
                cv2.imwrite(original_output_path, image_resized)

                # Create one augmented version
                image_augmented = seq(image=image_resized)
                augmented_output_path = os.path.join(output_subdir, f"augmented_{file}")
                cv2.imwrite(augmented_output_path, image_augmented)

            except Exception as e:
                print(f"Error processing {input_path}: {e}")
