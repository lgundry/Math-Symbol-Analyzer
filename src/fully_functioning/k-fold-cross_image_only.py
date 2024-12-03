import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from db_utils import loadFromDir, save_np_array, load_np_array, save_array, load_array
from db_validation import validateDB
from image_only_leaky import EncoderDecoderNetwork
from tests import main as testDB
from sklearn.model_selection import KFold
import os

def main():
    regenerate_db = False
    validate_DB = False
    test_network = False
    k_folds = 5  # Set the number of folds for cross-validation

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
        encoded_size = 64

        kfold = KFold(n_splits=k_folds, shuffle=True)
        loss_history = []
        val_loss_history = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(db)):

            # Split into training and validation sets
            train = db[train_idx]
            valid = db[val_idx]

            # Initialize the network only for the first fold
            if fold == 0:
                network = EncoderDecoderNetwork(input_size, hidden_size, encoded_size, output_size)
            else:
                # Reload the network with the previous fold's weights
                print(f"Loading weights for fold {fold} from 'fold_{fold - 1}.npz'")
                network = load_weights(f"fold_{fold - 1}.npz", input_size, hidden_size, encoded_size, output_size)

            if network is None:
                raise ValueError(f"Network initialization failed for fold {fold + 1}. Make sure the previous fold's weights are saved properly.")

            epochs = train.shape[0]
            learning_rate = .001

            cumulative_loss = 0
            # Train on the entire training dataset for a single epoch
            for index in range(epochs):
                label_index = int(train[index, -1])  # Get the label (last column of the row)
                target_image_path = "definitions/" + labels[label_index] + ".jpg"
                target_image = Image.open(target_image_path).convert("L")

                target_image_as_array = np.asarray(target_image).flatten()

                input_image = train[index, :-1]  # Get the flattened image data (excluding the label)

                image_loss, output_image = network.train(input_image, target_image_as_array, learning_rate)

                cumulative_loss += image_loss
                avg_loss = cumulative_loss / (index + 1)
                loss_history.append(avg_loss)

                if index % 100 == 0:
                    print(f"Fold {fold + 1}, Epoch {index}")
                    print(f'Loss: {image_loss}')
                    print(f'Average Loss: {avg_loss}')

            # Validation loss calculation
            val_loss = 0
            for validation_cycle in range(valid.shape[0]):
                val_label_index = int(valid[validation_cycle, -1])
                target_image_path = "definitions/" + labels[val_label_index] + ".jpg"
                target_image = Image.open(target_image_path).convert("L")
                target_image_as_array = np.asarray(target_image).flatten()
                input_image = valid[validation_cycle, :-1]  # Get the flattened image data (excluding the label)

                output_image = network.forward(input_image)

                # Normalize target for loss calculation
                target_image_as_array = target_image_as_array / 255.0

                val_image_loss = compute_loss(output_image, target_image_as_array)
                val_loss += val_image_loss

                if validation_cycle % (valid.shape[0]/2) == 0:
                    newpath = f"generated_images/images_{fold}"
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    display_image_normalized(input_image, 45, 45, f"generated_images/images_{fold}/input_{labels[val_label_index]}.png")
                    display_image_normalized(output_image, 45, 45, f"generated_images/images_{fold}/output_{labels[val_label_index]}.png")
                    display_image_normalized(target_image_as_array, 45, 45, f"generated_images/images_{fold}/target_{labels[val_label_index]}.png")

            avg_val_loss = val_loss / valid.shape[0]
            val_loss_history.append(avg_val_loss)

            print(f"Fold {fold + 1}: Training Loss = {avg_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

            # Save the weights after each fold
            print(f"Saving weights for fold {fold + 1} to 'fold_{fold}.npz'")
            save_weights(network, f"fold_{fold}.npz")

            # Plot the loss history and save/display it after each fold
            plt.plot(loss_history, label='Training Loss')
            plt.plot(val_loss_history, label='Validation Loss')
            plt.title(f"Training and Validation Loss Over Time - Fold {fold + 1}")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"loss_graph_fold_{fold + 1}.png")
            plt.close()

        # Final graph display after all folds
        plt.plot(loss_history)
        plt.title("Training Average Loss Over Time (Across All Folds)")
        plt.xlabel("Iterations")
        plt.ylabel("Average Loss")
        plt.savefig("final_loss_graph.png")
        plt.close()

def load_weights(filepath, input_size, hidden_size, encoded_size, output_size):
    """
    Load the weights from a file and return a network initialized with those weights.
    """
    try:
        data = np.load(filepath)
        network = EncoderDecoderNetwork(input_size, hidden_size, encoded_size, output_size)
        # Set the weights and biases from the loaded data
        network.weights_input_hidden = data['weights_input_hidden']
        network.weights_hidden_encoded = data['weights_hidden_encoded']
        network.weights_encoded_hidden = data['weights_encoded_hidden']
        network.weights_hidden_output = data['weights_hidden_output']
        network.bias_first_hidden = data['bias_first_hidden']
        network.bias_encoded = data['bias_encoded']
        network.bias_second_hidden = data['bias_second_hidden']
        network.bias_output = data['bias_output']
        print(f"Weights loaded from {filepath}")
        return network
    except Exception as e:
        print(f"Error loading weights from {filepath}: {e}")
        return None


def save_weights(network, filepath):
    """
    Save the weights and biases of the network to a .npz file.
    """
    np.savez(filepath, 
             weights_input_hidden=network.weights_input_hidden,
             weights_hidden_encoded=network.weights_hidden_encoded,
             weights_encoded_hidden=network.weights_encoded_hidden,
             weights_hidden_output=network.weights_hidden_output,
             bias_first_hidden=network.bias_first_hidden,
             bias_encoded=network.bias_encoded,
             bias_second_hidden=network.bias_second_hidden,
             bias_output=network.bias_output)
    print(f"Weights saved to {filepath}")

def display_image_normalized(image_output, width, height, filename):
    reshaped_image = image_output.reshape((height, width))
    denormalized_image = np.clip(reshaped_image * 255, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(denormalized_image, mode='L')
    pil_image.save(filename)
    print(f"Image saved as '{filename}'.")

def compute_loss(output, target):
    return np.mean(np.square(target - output))

if __name__ == "__main__":
    main()
