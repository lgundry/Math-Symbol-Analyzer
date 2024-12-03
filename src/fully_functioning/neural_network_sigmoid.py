import numpy as np
import matplotlib.pyplot as plt

class EncoderDecoderNetwork:
    def __init__(self, input_size, hidden_size, encoded_size, output_size, softmax_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size
        self.output_size = output_size
        self.softmax_size = softmax_size

        # Initialize weights - uses Xavier Initialization
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / (self.input_size + self.hidden_size))
        self.weights_hidden_encoded = np.random.randn(self.hidden_size, self.encoded_size) * np.sqrt(2 / (self.hidden_size + self.encoded_size))
        self.weights_encoded_hidden = np.random.randn(self.encoded_size, self.hidden_size) * np.sqrt(2 / (self.encoded_size + self.hidden_size))
        self.weights_encoded_softmax = np.random.randn(self.encoded_size, self.softmax_size) * np.sqrt(2 / (self.encoded_size + self.softmax_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / (self.hidden_size + self.output_size))

        # Initialize biases to zero
        self.bias_first_hidden = np.zeros((1, self.hidden_size))
        self.bias_encoded = np.zeros((1, self.encoded_size))
        self.bias_second_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        self.bias_softmax = np.zeros((1, self.softmax_size))

    def forward(self, input):

        # Input to First Hidden Layer
        self.first_hidden_layer_input = np.dot(input, self.weights_input_hidden) + self.bias_first_hidden
        self.first_hidden_layer_output = self.sigmoid(self.first_hidden_layer_input)

        # First Hidden Layer to Encoded Layer
        self.encoded_layer_input = np.dot(self.first_hidden_layer_output, self.weights_hidden_encoded) + self.bias_encoded
        self.encoded_layer_output = self.sigmoid(self.encoded_layer_input)

        # Encoded Layer to Second Hidden Layer
        self.second_hidden_layer_input = np.dot(self.encoded_layer_output, self.weights_encoded_hidden) + self.bias_second_hidden
        self.second_hidden_layer_output = self.sigmoid(self.second_hidden_layer_input)

        # Encoded layer to Softmax Output (Label Prediction)
        self.softmax_layer_input = np.dot(self.encoded_layer_output, self.weights_encoded_softmax) + self.bias_softmax
        self.softmax_output = self.softmax(self.softmax_layer_input)
        
        # Second Hidden Layer to Image Output (Reconstruction)
        self.output_layer_input = np.dot(self.second_hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.image_output = self.sigmoid(self.output_layer_input)
        
        return self.image_output, self.softmax_output

    def backward(self, input, target_image, target_label, learning_rate, clip_value=1.0):
        # Calculate the error
        image_output_error = target_image - self.image_output
        image_output_delta = image_output_error * self.sigmoid_derivative(self.image_output)

        softmax_output_delta = (self.softmax_output - target_label) * self.softmax_output * (1 - self.softmax_output)
        
        second_hidden_error = image_output_delta.dot(self.weights_hidden_output.T)
        second_hidden_delta = second_hidden_error * self.sigmoid_derivative(self.second_hidden_layer_output)

        encoded_error = second_hidden_delta.dot(self.weights_encoded_hidden.T) + softmax_output_delta.dot(self.weights_encoded_softmax.T)
        encoded_delta = encoded_error * self.sigmoid_derivative(self.encoded_layer_output)

        first_hidden_error = encoded_delta.dot(self.weights_hidden_encoded.T)
        first_hidden_delta = first_hidden_error * self.sigmoid_derivative(self.first_hidden_layer_output)

        # Gradient clipping: Clip the gradients to the specified value
        image_output_delta = np.clip(image_output_delta, -clip_value, clip_value)
        softmax_output_delta = np.clip(softmax_output_delta, -clip_value, clip_value)
        first_hidden_delta = np.clip(first_hidden_delta, -clip_value, clip_value)
        second_hidden_delta = np.clip(second_hidden_delta, -clip_value, clip_value)
        encoded_delta = np.clip(encoded_delta, -clip_value, clip_value)

        # Debugging: Print bias gradients before updating biases
        print(f"Bias gradients before update:")
        print(f"Bias output gradient: {np.sum(image_output_delta, axis=0)}")
        print(f"Bias softmax gradient: {np.sum(softmax_output_delta, axis=0)}")
        print(f"Bias second hidden gradient: {np.sum(second_hidden_delta, axis=0)}")
        print(f"Bias encoded gradient: {np.sum(encoded_delta, axis=0)}")
        print(f"Bias first hidden gradient: {np.sum(first_hidden_delta, axis=0)}")

        # Update weights and biases with clipped gradients
        self.weights_hidden_output += self.second_hidden_layer_output.T.dot(image_output_delta) * learning_rate
        self.bias_output += np.sum(image_output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_encoded_softmax += self.encoded_layer_output.T.dot(softmax_output_delta) * learning_rate
        self.bias_softmax += np.sum(softmax_output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_encoded_hidden += self.encoded_layer_output.T.dot(second_hidden_delta) * learning_rate
        self.bias_second_hidden += np.sum(second_hidden_delta, axis=0, keepdims=True) * learning_rate

        self.weights_hidden_encoded += self.first_hidden_layer_output.T.dot(encoded_delta) * learning_rate
        self.bias_encoded += np.sum(encoded_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += input.T.dot(first_hidden_delta) * learning_rate
        self.bias_first_hidden += np.sum(first_hidden_delta, axis=0, keepdims=True) * learning_rate


    def train(self, input, target_image, target_label, learning_rate):
        input = input.reshape(1, -1)
        target_image = target_image.reshape(1, -1)
        target_label = target_label.reshape(1, -1)

        # Perform a forward pass (for all samples at once)
        image_output, softmax_output = self.forward(input)
            
        # Calculate loss for the image reconstruction (Mean Squared Error)
        image_loss = np.mean(np.square(target_image - image_output))  # Compare image outputs to true images
            
        # Calculate loss for the label prediction (Mean Squared Error)
        label_loss = self.cross_entropy_loss(target_label, softmax_output)
            
        # Combine both losses (you can adjust the weights of each task here if needed)
        total_loss = 0.5 * image_loss + 0.5 * label_loss
            
        # Perform a backward pass to update the weights and biases
        self.backward(input, target_image, target_label, learning_rate)
        
        print(f'Total Loss: {total_loss}, Image Loss: {image_loss}, Label Loss: {label_loss}')
                

    def sigmoid(self, x):
        # Clip the values more aggressively to avoid overflow
        x = np.clip(x, -1000, 1000)  # Limits the values to avoid large numbers in exp
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        # Clip the values before applying softmax
        x = np.clip(x, -1000, 1000)  # Clip values to avoid overflow
        e_x = np.exp(x - np.max(x))  # Subtract max to prevent large exponents
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    def l2_regularization(self, lambda_reg):
        l2_loss = lambda_reg * np.sum(np.square(self.weights_input_hidden))  # Example for one weight matrix
        return l2_loss
    
    def cross_entropy_loss(self, y_true, y_pred):
        # Clip the values of predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def clip_gradients(self, max_norm=5.0):
        self.weights_input_hidden = np.clip(self.weights_input_hidden, -max_norm, max_norm)
        self.weights_hidden_output = np.clip(self.weights_hidden_output, -max_norm, max_norm)
        # Repeat for all weight matrices



