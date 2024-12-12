import os
import numpy as np
from PIL import Image

def crop_center(image, target_width, target_height):
    
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2
    return image.crop((left, top, right, bottom))

def process_and_save_image(image_path, target_width=45, target_height=45):
    
    image = Image.open(image_path).convert('L')
    cropped_image = crop_center(image, target_width, target_height)
    
    cropped_image.save(image_path)

def process_directory(input_dir, target_width=45, target_height=45):
    
    for label_name in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label_name)

        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)

                if file_name.endswith(".jpg"):

                    process_and_save_image(file_path, target_width, target_height)

                    print(f"Processed and overwritten: {file_path}")

def main():
    input_dir = 'data/augmented_images'

    process_directory(input_dir)

    print("Processing complete. All images have been overwritten.")

if __name__ == "__main__":
    main()
