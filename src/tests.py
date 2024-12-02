import numpy as np
from neural_network_sigmoid import EncoderDecoderNetwork 

def main():
    # Test setup
    input_size = 2025   # Example input size (for flattened 45x45px image)
    hidden_size = 45    # Number of neurons in the hidden layer
    encoded_size = 7    # Encoded (latent) layer size
    output_size = 2025  # Image output size (reconstruction)
    softmax_size = 81   # Number of classes for softmax (for label prediction)

    # Create random input data (e.g., 5 samples, each 2025 elements)
    X = np.random.rand(1, input_size)

    # Create random target images (e.g., 5 samples, each 2025 elements, same size as X)
    target_image = np.random.rand(1, output_size)

    # Create random target labels (e.g., 5 samples, each one-hot encoded, 10 classes)
    target_label = np.zeros((1, softmax_size))
    random_class = np.random.randint(0, softmax_size)
    target_label[0, random_class] = 1

    # Instantiate the network
    network = EncoderDecoderNetwork(input_size, hidden_size, encoded_size, output_size, softmax_size)

    # Test the forward pass
    image_output, softmax_output = network.forward(X)

    # Check the shapes of the outputs
    print("Shape of image_output:", image_output.shape)
    print("Shape of softmax_output:", softmax_output.shape)

    # Test the backward pass and training
    epochs = 100000
    learning_rate = .0001

    # Train the model
    network.train(X, target_image, target_label, epochs, learning_rate)

    # After training, check if the image and label losses are decreasing
    # For example, you can print the last few epochs' loss values
