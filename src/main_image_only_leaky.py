import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from db_utils import loadFromDir, save_np_array, load_np_array, save_array, load_array
from db_validation import validateDB
from image_only_leaky import EncoderDecoderNetwork  # Ensure this imports your updated network
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

        label_count = len(labels)

        input_size = db.shape[1] - 1
        output_size = input_size
        hidden_size = 128
        encoded_size = 32

        shuffled_db = np.random.permutation(db)

        train = shuffled_db[:int(0.9 * db_row_count), :]
        valid = shuffled_db[int(0.9 * db_row_count):int(0.95 * db_row_count), :]
        test = shuffled_db[int(0.95 * db_row_count):, :]

        # Initialize network with Leaky ReLU
        network = EncoderDecoderNetwork(input_size, hidden_size, encoded_size, output_size)
        network.forward(db[0, :-1])

        epochs = db_column_count                                                                     
        learning_rate = .001
        cycles = 70

        # List to track the loss over time for graphing
        loss_history = []  # Initialize loss history to track over multiple cycles
        cumulative_loss = 0  # Variable to calculate the running average
        val_loss_history = []  # Track validation loss

        for cycle in range(cycles):

            for index in range(epochs):

                label_index = int(train[index, -1])  # Get the label (last column of the row)
                target_image_path = "definitions/" + labels[label_index] + ".jpg"
                target_image = Image.open(target_image_path).convert("L")
                target_image_as_array = np.asarray(target_image).flatten()

                input_image = train[index, :-1]  # Get the flattened image data (excluding the label)

                # Train the network on this single image
                image_loss, output_image = network.train(input_image, target_image_as_array, learning_rate)

                cumulative_loss += image_loss  # Add to cumulative loss for average calculation
                avg_loss = cumulative_loss / ((cycle * epochs) + (index + 1))
                loss_history.append(avg_loss)  # Track average loss over time

                if index % 100 == 0:
                    print(f"Epoch {index}, Cycle {cycle}")
                    print(f'Loss: {image_loss}')
                    print(f'Average Loss: {avg_loss}')

                if ((cycle > 1 and index > 1) and (index * cycle) % 10002 == 0):
                    print(labels[label_index])
                    display_image_normalized(input_image, 45, 45, "image_input.jpg")
                    display_image_normalized(output_image, 45, 45, "image_output.jpg")

                    # Plot the loss history and save/display it
                    plt.plot(loss_history)
                    plt.title(f"Training Loss Over Time - Cycle {cycle + 1}")
                    plt.xlabel("Iterations")
                    plt.ylabel("Average Loss")
                    plt.savefig("loss_graph.png")
                    plt.close()  # Close the figure to avoid overlap in next cycle plot

            # Validation loss calculation
            val_loss = 0
            for validation_cycle in range(valid.shape[0]):
                val_label_index = int(valid[validation_cycle, -1])
                target_image_path = "definitions/" + labels[val_label_index] + ".jpg"
                target_image = Image.open(target_image_path).convert("L")
                target_image_as_array = np.asarray(target_image).flatten()
                input_image = valid[validation_cycle, :-1]  # Get the flattened image data (excluding the label)

                output_image = network.forward(input_image)
                denormalized_output = np.clip(output_image * 255, 0, 255).astype(np.uint8)
                val_image_loss = compute_loss(denormalized_output, target_image_as_array)
                val_loss += val_image_loss

            avg_val_loss = val_loss / valid.shape[0]
            val_loss_history.append(avg_val_loss)

            print(f"Cycle {cycle + 1}: Training Loss = {avg_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

            # Plot the loss history and save/display it after each cycle
            plt.plot(loss_history, label='Training Loss')
            plt.plot(val_loss_history, label='Validation Loss')
            plt.title(f"Training and Validation Loss Over Time - Cycle {cycle + 1}")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"loss_graph_cycle_{cycle + 1}.png")
            plt.close()  # Close the figure to avoid overlap in next cycle plot

        # Final graph display after training
        plt.plot(loss_history)
        plt.title("Training Average Loss Over Time")
        plt.xlabel("Iterations")
        plt.ylabel("Average Loss")
        plt.savefig("final_loss_graph.png")
        plt.close()

def display_image_normalized(image_output, width, height, filename):
    reshaped_image = image_output.reshape((height, width))
    denormalized_image = np.clip(reshaped_image * 255, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(denormalized_image, mode='L')
    pil_image.save(filename)
    print(f"Image saved as '{filename}'.")

def compute_loss(output, target):
    return np.mean((output - target) ** 2)

if __name__ == "__main__":
    main()
