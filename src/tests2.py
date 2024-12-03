import numpy as np
from repo.src.fully_functioning.neural_network_sigmoid import EncoderDecoderNetwork

# Define the dimensions of the network
input_size = 45 * 45  # For example, 45x45 image flattened
hidden_size = 100
encoded_size = 50
output_size = 45 * 45  # Reconstructed image size
softmax_size = 10  # Assuming 10 classes for label prediction

# Instantiate the network
network = EncoderDecoderNetwork(input_size, hidden_size, encoded_size, output_size, softmax_size)

# Create a dummy input image (45x45 pixels) and a dummy label (for classification)
dummy_input = np.random.randint(0, 1, (1, input_size))  # Random binary input image
dummy_target_image = np.random.randint(0, 2, (1, output_size))  # Random binary target image
dummy_target_label = np.random.randint(0, 1, (1, softmax_size))  # One-hot encoded label (for classification)

# Test forward pass
image_output, softmax_output = network.forward(dummy_input)
# print("Forward Pass Output (Image):")
# print(image_output)
# print("Forward Pass Output (Softmax):")
# print(softmax_output)

# Test training for 5 epochs with a learning rate of 0.01
network.train(dummy_input, dummy_target_image, dummy_target_label, epochs=10000, learning_rate=0.01)

