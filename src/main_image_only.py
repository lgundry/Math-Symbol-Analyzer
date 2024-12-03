import numpy as np
from PIL import Image
from db_utils import loadFromDir, save_np_array, load_np_array, save_array, load_array
from db_validation import validateDB
from image_only import EncoderDecoderNetwork
from tests import main as testDB

def main():
    # Toggle between generating or loading the database and labels
    regenerate_db = False
    validate_DB = False
    test_network = False

    if regenerate_db:
        db, labels = loadFromDir("data/extracted_images")
        save_np_array(db, "database.npy")
        save_array(labels, "labels.pkl")
    else:
        db = load_np_array("database.npy")
        labels = load_array("labels.pkl")

    if validate_DB:
        validateDB(db, labels, "data/extracted_images")

    if test_network:
        testDB()
    else:
        db_dims = db.shape
        db_row_count = db_dims[0]
        db_column_count = db_dims[1]

        # The index of the 1 in the row/column of one_hot_labels is also the index in labels for the appropriate description
        label_count = len(labels)
        one_hot_labels = np.eye(label_count, dtype=int)

        input_size = db.shape[1] - 1
        output_size = input_size
        softmax_size = label_count
        hidden_size = 512
        encoded_size = 128

        shuffled_db = np.random.permutation(db)

        train = shuffled_db[:int(0.9 * db_row_count), :]
        valid = shuffled_db[int(0.9 * db_row_count):int(0.95 * db_row_count), :]
        test = shuffled_db[int(0.95 * db_row_count):, :]

        network = EncoderDecoderNetwork(input_size, hidden_size, encoded_size, output_size)
        image_output = network.forward(db[0, :-1])

        # Test the backward pass and training
        epochs = db_column_count                                                                     
        learning_rate = .001
        cycles = 1000

        for cycle in range(cycles):
            print("cycle: " +  str(cycle))
            for index in range(epochs):

                # Get the label index for the current image
                label_index = int(train[index, -1])  # Get the label (last column of the row)

                # Load the target image (corresponding to the label)
                target_image_path = "definitions/" + labels[label_index] + ".jpg"
                target_image = Image.open(target_image_path).convert("L")
                target_image_as_array = np.asarray(target_image).flatten()

                # Extract the current image (flattened) for training
                input_image = train[index, :-1]  # Get the flattened image data (excluding the label)

                # Train the network on this single image
                image_loss, output_image = network.train(input_image, target_image_as_array, learning_rate)
                
                if (index % 100 == 0):
                    print("epoch " + str(index))
                    print(f'Loss: {image_loss}')

                    if ((cycle > 1 and index > 1) and (index * cycle) % 10000 == 0):
                        print(labels[label_index])
                        display_image_normalized(output_image, 45, 45)

from PIL import Image
import numpy as np

def display_image_normalized(image_output, width, height):
    """
    Normalizes and converts a flattened NumPy array into a displayable image.
    
    Args:
        image_output (np.ndarray): Flattened 1D array of the image output.
        width (int): Width of the image.
        height (int): Height of the image.
    """
    # Reshape the output into the original dimensions
    reshaped_image = image_output.reshape((height, width))
    
    # Normalize the image to fit in the range [0, 255]
    min_val = reshaped_image.min()
    max_val = reshaped_image.max()
    if max_val > min_val:
        normalized_image = ((reshaped_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized_image = np.clip(reshaped_image, 0, 255).astype(np.uint8)
    
    # Convert to a PIL Image
    pil_image = Image.fromarray(normalized_image, mode='L')
    
    # Save the image in case `show()` doesn't work
    pil_image.save("output_image.png")
    print("Image saved as 'output_image.png'.")
    
    # Display the image
    pil_image.show()  # This uses the default image viewer on your system

    # Pause the script until the user closes the viewer
    input("Press Enter after viewing the image...")

if __name__ == "__main__":
    main()
