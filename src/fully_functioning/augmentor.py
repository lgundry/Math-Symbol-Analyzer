import os
import random
from PIL import Image, ImageEnhance

def augment_image(image: Image.Image):
    """ Apply multiple augmentations to a single image """
    augmentations = [
        rotate_image,
        flip_image,
        scale_image,
        translate_image,
        adjust_brightness
    ]
    
    # Randomly apply some augmentations
    augmented_image = image
    for aug in random.sample(augmentations, random.randint(2, len(augmentations))):  # Apply 2 to all augmentations
        augmented_image = aug(augmented_image)
    
    return augmented_image

def rotate_image(image: Image.Image):
    """ Rotate image by a random degree between -30 and 30 """
    angle = random.randint(-30, 30)
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

def flip_image(image: Image.Image):
    """ Randomly flip the image horizontally or vertically """
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def scale_image(image: Image.Image):
    """ Randomly scale the image by a factor between 0.8 and 1.2 """
    scale_factor = random.uniform(0.8, 1.2)
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, resample=Image.BICUBIC)

def translate_image(image: Image.Image):
    """ Randomly shift (translate) the image """
    max_shift = 5  # max translation in pixels
    x_shift = random.randint(-max_shift, max_shift)
    y_shift = random.randint(-max_shift, max_shift)
    return image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))

def adjust_brightness(image: Image.Image):
    """ Randomly adjust the brightness of the image """
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.7, 1.3)  # Random brightness factor between 0.7 and 1.3
    return enhancer.enhance(factor)

def save_augmented_images(image: Image.Image, output_subdir: str, base_filename: str, augment_count: int):
    """ Save the augmented images """
    for i in range(augment_count):
        augmented_image = augment_image(image)
        augmented_image.save(os.path.join(output_subdir, f"{base_filename}_aug_{i+1}.jpg"))

def augment_dataset(input_dir: str, output_dir: str, augment_count: int = 5):
    """ Augment all images in the input directory and save them to the corresponding output subdirectories """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Walk through each subdirectory in the input directory
    for subdir_name in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir_name)
        
        # Skip non-directories
        if not os.path.isdir(subdir_path):
            continue
        
        # Create a corresponding subdirectory in the output directory
        output_subdir = os.path.join(output_dir, subdir_name)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        # Process each image file in the subdirectory
        image_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        for image_file in image_files:
            image_path = os.path.join(subdir_path, image_file)
            image = Image.open(image_path)
            base_filename = os.path.splitext(image_file)[0]  # Remove extension for naming
            save_augmented_images(image, output_subdir, base_filename, augment_count)
            print(f"Augmented {image_file} from subdirectory {subdir_name} and saved {augment_count} augmented images.")

if __name__ == "__main__":
    input_dir = 'data/extracted_images'  # Replace with the path to your original dataset
    output_dir = 'data/augmented_images'  # Replace with the path where augmented images should be saved
    augment_count = 3  # Number of augmented versions to generate per image
    
    augment_dataset(input_dir, output_dir, augment_count)
