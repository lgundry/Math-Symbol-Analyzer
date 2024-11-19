from db_utils import loadFromDir, save_np_array, load_np_array, save_array, load_array
from db_validation import validateDB
from neural_network import EncoderDecoderNetwork

def main():
    # Toggle between generating or loading the database and labels
    regenerate_db = False
    validate_DB = False
    test_network = True

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
        # Initialize the network with the appropriate sizes
        input_size = db.shape[0] - 1  # Exclude the last row for the label
        bottleneck_size = 128  # Size of the encoded representation
        hidden_sizes = [512, 256]  # Hidden layer sizes
        num_labels = len(set(labels))  # Number of unique labels

        network = EncoderDecoderNetwork(input_size, bottleneck_size, hidden_sizes, num_labels)

        # Test forward pass with random data (or use actual data from db)
        test_input = db[:-1, 0]  # Use the first image's features as input
        reconstructed, predicted_label = network.forward(test_input)

        # Print the outputs of the network
        print("Reconstructed (input) data:", reconstructed)
        print("Predicted label:", predicted_label)

        # Perform a backward pass (fake label for testing)
        target_label = 5  # Choose a random target label for testing
        network.backward(test_input, target_label, reconstructed, predicted_label)

        print("Backward pass completed.")

if __name__ == "__main__":
    main()
