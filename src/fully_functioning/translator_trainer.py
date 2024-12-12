import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from db_utils import loadFromDir, save_np_array, load_np_array, save_array, load_array
from db_validation import validateDB
from image_only_leaky import EncoderDecoderNetwork


def main():
    # Toggle between generating or loading the database and labels
    regenerate_db = False
    validate_DB = False

    if regenerate_db:
        db, labels = loadFromDir("data/augmented_images")
        save_np_array(db, "database.npy")
        save_array(labels, "labels.pkl")
    else:
        db = load_np_array("database.npy")
        labels = load_array("labels.pkl")

    if validate_DB:
        validateDB(db, labels, "data/augmented_images")
        
    db_dims = db.shape
    db_row_count = db_dims[0]
    db_column_count = db_dims[1]

    label_count = len(labels)

    input_size = db.shape[1] - 1
    output_size = input_size
    hidden_size = 256
    encoded_size = 64

    shuffled_db = np.random.permutation(db)

    train = shuffled_db[: int(0.9 * db_row_count), :]
    valid = shuffled_db[int(0.9 * db_row_count) : int(0.95 * db_row_count), :]
    test = shuffled_db[int(0.95 * db_row_count) :, :]

    # Initialize the network
    network = EncoderDecoderNetwork(input_size, hidden_size, encoded_size, output_size)
    output_image = network.forward(db[0, :-1])

    # Variables for different training loop parameters
    cycles = train.shape[0]
    learning_rate = 0.0001
    epochs = 20
    # List to tracking
    loss_history = [] 
    cumulative_loss = 0
    val_loss_history = []

    for epoch in range(epochs):

        for index in range(cycles):

            # Grab the image label and use it to open the target image
            label_index = int(train[index, -1])
            target_image_path = "definitions/" + labels[label_index] + ".jpg"
            target_image = Image.open(target_image_path).convert("L")
            target_image_as_array = np.asarray(target_image).flatten()

            input_image = train[index, :-1]

            # Train the network on the input image
            image_loss, output_image = network.train(
                input_image, target_image_as_array, learning_rate
            )

            cumulative_loss += image_loss
            avg_loss = cumulative_loss / ((epoch * cycles) + (index + 1))
            loss_history.append(avg_loss)

            if index % 100 == 0:
                print(f"Cycle {index}, Epoch {epoch}")
                print(f"Loss: {image_loss}")
                print(f"Average Loss: {avg_loss}")

        # Validation loop - if loss is high compared to training, the network is not generalizing properly
        val_loss = 0
        for validation_cycle in range(valid.shape[0]):
            val_label_index = int(valid[validation_cycle, -1])
            target_image_path = "definitions/" + labels[val_label_index] + ".jpg"
            target_image = Image.open(target_image_path).convert("L")
            target_image_as_array = np.asarray(target_image).flatten()
            input_image = valid[validation_cycle, :-1]  # Get the flattened image data (excluding the label)
            target_image_normalized = target_image_as_array / 255

            output_image = network.forward(input_image)
            val_image_loss = compute_loss(output_image, target_image_normalized)
            val_loss += val_image_loss

        avg_val_loss = val_loss / valid.shape[0]
        val_loss_history.append(avg_val_loss)
        display_image_normalized(output_image, 45, 45, "image_output.jpg")
        display_image_normalized(input_image / 255, 45, 45, "image_input.jpg")

        print(
            f"Epoch {epoch + 1}: Training Loss = {avg_loss:.4f}, Validation Loss = {avg_val_loss:.4f}"
        )

        # Plot the loss history and save it
        plt.plot(loss_history, label="Training Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.title(f"Training and Validation Loss Over Time - Cycle {epoch + 1}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_graph_cycle_{epoch + 1}.png")
        plt.close()

    # Final graph display after training
    plt.plot(loss_history)
    plt.title("Training Average Loss Over Time")
    plt.xlabel("Iterations")
    plt.ylabel("Average Loss")
    plt.savefig("final_loss_graph.png")
    plt.close()
    
    network.save("encoder_decoder2.npz")


def display_image_normalized(image_output, width, height, filename):
    reshaped_image = image_output.reshape((height, width))
    denormalized_image = np.clip(reshaped_image * 255, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(denormalized_image, mode="L")
    pil_image.save(filename)
    print(f"Image saved as '{filename}'.")


def compute_loss(output, target):
    return np.mean((output - target) ** 2)


if __name__ == "__main__":
    main()
