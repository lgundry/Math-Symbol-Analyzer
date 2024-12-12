import numpy as np

class EncoderDecoderNetwork:
    def __init__(self, input_size, hidden_size, encoded_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size
        self.output_size = output_size

        # Initialize weights - uses Xavier Initialization
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / (self.input_size + self.hidden_size))
        self.weights_hidden_encoded = np.random.randn(self.hidden_size, self.encoded_size) * np.sqrt(2 / (self.hidden_size + self.encoded_size))
        self.weights_encoded_hidden = np.random.randn(self.encoded_size, self.hidden_size) * np.sqrt(2 / (self.encoded_size + self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / (self.hidden_size + self.output_size))

        # Initialize biases to zero
        self.bias_first_hidden = np.zeros((1, self.hidden_size))
        self.bias_encoded = np.zeros((1, self.encoded_size))
        self.bias_second_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, input):
        # Input to First Hidden Layer
        self.first_hidden_layer_input = np.dot(input, self.weights_input_hidden) + self.bias_first_hidden
        self.first_hidden_layer_output = self.leaky_relu(self.first_hidden_layer_input)

        # First Hidden Layer to Encoded Layer
        self.encoded_layer_input = np.dot(self.first_hidden_layer_output, self.weights_hidden_encoded) + self.bias_encoded
        self.encoded_layer_output = self.leaky_relu(self.encoded_layer_input)

        # Encoded Layer to Second Hidden Layer
        self.second_hidden_layer_input = np.dot(self.encoded_layer_output, self.weights_encoded_hidden) + self.bias_second_hidden
        self.second_hidden_layer_output = self.leaky_relu(self.second_hidden_layer_input)
        
        # Second Hidden Layer to Image Output (Reconstruction) with Sigmoid Activation
        self.output_layer_input = np.dot(self.second_hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.image_output = self.sigmoid(self.output_layer_input)
        
        return self.image_output

    def backward(self, input, target_image, learning_rate, clip_value=1.0):
        # Calculate the error
        image_output_error = target_image - self.image_output
        image_output_delta = image_output_error * self.sigmoid_derivative(self.image_output)
        
        second_hidden_error = image_output_delta.dot(self.weights_hidden_output.T)
        second_hidden_delta = second_hidden_error * self.leaky_relu_derivative(self.second_hidden_layer_output)

        encoded_error = second_hidden_delta.dot(self.weights_encoded_hidden.T)
        encoded_delta = encoded_error * self.leaky_relu_derivative(self.encoded_layer_output)

        first_hidden_error = encoded_delta.dot(self.weights_hidden_encoded.T)
        first_hidden_delta = first_hidden_error * self.leaky_relu_derivative(self.first_hidden_layer_output)

        # Clip gradients before updating weights
        image_output_delta = np.clip(image_output_delta, -clip_value, clip_value)
        first_hidden_delta = np.clip(first_hidden_delta, -clip_value, clip_value)
        second_hidden_delta = np.clip(second_hidden_delta, -clip_value, clip_value)
        encoded_delta = np.clip(encoded_delta, -clip_value, clip_value)

        # Update weights and biases with clipped gradients
        self.weights_hidden_output += self.second_hidden_layer_output.T.dot(image_output_delta) * learning_rate
        self.bias_output += np.sum(image_output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_encoded_hidden += self.encoded_layer_output.T.dot(second_hidden_delta) * learning_rate
        self.bias_second_hidden += np.sum(second_hidden_delta, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_encoded += self.first_hidden_layer_output.T.dot(encoded_delta) * learning_rate
        self.bias_encoded += np.sum(encoded_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += input.T.dot(first_hidden_delta) * learning_rate
        self.bias_first_hidden += np.sum(first_hidden_delta, axis=0, keepdims=True) * learning_rate


    def train(self, input, target_image, learning_rate):
        # Normalize input and target image to [0, 1]
        input = input / 255.0  # Normalize input
        target_image = target_image / 255.0  # Normalize target image

        input = input.reshape(1, -1)
        target_image = target_image.reshape(1, -1)

        # Perform a forward pass
        image_output = self.forward(input)
            
        # Calculate loss for the image reconstruction (Mean Squared Error)
        image_loss = np.mean(np.square(target_image - image_output))  # Compare image outputs to true images
            
        # Perform a backward pass to update the weights and biases
        self.backward(input, target_image, learning_rate)

        return image_loss, image_output

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  # Clamp x values to avoid overflow
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def l2_regularization(self, lambda_reg):
        l2_loss = lambda_reg * np.sum(np.square(self.weights_input_hidden))
        return l2_loss
    
    def clip_gradients(self, max_norm=5.0):
        self.weights_input_hidden = np.clip(self.weights_input_hidden, -max_norm, max_norm)
        self.weights_hidden_output = np.clip(self.weights_hidden_output, -max_norm, max_norm)
        
    def save(self, file_path):
        """
        Saves the current state of the network (weights, biases, and architecture).
        """
        np.savez(
            file_path,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            encoded_size=self.encoded_size,
            output_size=self.output_size,
            weights_input_hidden=self.weights_input_hidden,
            weights_hidden_encoded=self.weights_hidden_encoded,
            weights_encoded_hidden=self.weights_encoded_hidden,
            weights_hidden_output=self.weights_hidden_output,
            bias_first_hidden=self.bias_first_hidden,
            bias_encoded=self.bias_encoded,
            bias_second_hidden=self.bias_second_hidden,
            bias_output=self.bias_output,
        )
        print(f"Model saved to {file_path}")

    @staticmethod
    def load(file_path):
        """
        Loads the network state from a file and returns a new network instance.
        """
        data = np.load(file_path)
        network = EncoderDecoderNetwork(
            input_size=int(data['input_size']),
            hidden_size=int(data['hidden_size']),
            encoded_size=int(data['encoded_size']),
            output_size=int(data['output_size']),
        )
        network.weights_input_hidden = data['weights_input_hidden']
        network.weights_hidden_encoded = data['weights_hidden_encoded']
        network.weights_encoded_hidden = data['weights_encoded_hidden']
        network.weights_hidden_output = data['weights_hidden_output']
        network.bias_first_hidden = data['bias_first_hidden']
        network.bias_encoded = data['bias_encoded']
        network.bias_second_hidden = data['bias_second_hidden']
        network.bias_output = data['bias_output']
        print(f"Model loaded from {file_path}")
        return network


