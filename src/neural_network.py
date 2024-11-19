import numpy as np
import matplotlib.pyplot as plt

class EncoderDecoderNetwork:
    def __init__(self, input_size, hidden_size, encoded_size, output_size, softmax_size):
        # input first_hidden encoded second_hidden output
        # 2025    45, 7, 45,   2026
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size
        self.output_size = output_size
        self.softmax_size = softmax_size

        # initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_encoded = np.random.randn(self.hidden_size, self.encoded_size)
        self.weights_encoded_hidden = np.random.randn(self.encoded_size, self.hidden_size)
        self.weights_encoded_softmax = np.random.randn(self.encoded_size, self.softmax_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        #initialize biases
        self.bias_first_hidden = np.zeros((1, self.hidden_size))
        self.bias_encoded = np.zeros((1, self.encoded_size))
        self.bias_second_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        self.bias_softmax = np.zeros((1, self.softmax_size))

    def forward(self, X):
        # Input to First Hidden Layer
        self.first_hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_first_hidden
        self.first_hidden_layer_output = self.sigmoid(self.first_hidden_layer_input)

        # First Hidden Layer to encoded Layer
        self.encoded_layer_input = np.dot(self.first_hidden_layer_output, self.weights_hidden_encoded) + self.bias_encoded
        self.encoded_layer_output = self.sigmoid(self.encoded_layer_input)

        # Encoded Layer to Second Hidden Layer
        self.second_hidden_layer_input = np.dot(self.encoded_layer_output, self.weights_encoded_hidden) + self.bias_second_hidden
        self.second_hidden_layer_output = self.sigmoid(self.second_hidden_layer_input)
        
        # Second Hidden Layer to Image Output (Reconstruction)
        self.output_layer_input = np.dot(self.second_hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.image_output = self.sigmoid(self.output_layer_input)

        # Encoded layer to Softmax Output (Label Prediction)
        self.softmax_layer_input = np.dot(self.encoded_layer_output, self.weights_encoded_softmax) + self.bias_softmax
        self.softmax_output = self.softmax(self.softmax_layer_input)
        
        return self.image_output, self.softmax_output

    def backward(self, X, y, learning_rate):
        # Calculate the error
        image_output_error = y - self.image_output
        image_output_delta = image_output_error * self.sigmoid_derivative(self.image_output)

        softmax_output_error = self.softmax_output - y
        softmax_output_delta = softmax_output_error
        
        first_hidden_error = image_output_delta.dot(self.weights_hidden_output.T)
        first_hidden_delta = first_hidden_error * self.sigmoid_derivative(self.first_hidden_layer_output)

        second_hidden_error = first_hidden_delta.dot(self.weights_encoded_hidden.T)
        second_hidden_delta = second_hidden_error * self.sigmoid_derivative(self.second_hidden_layer_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.second_hidden_layer_output.T.dot(image_output_delta) * learning_rate
        self.bias_output += np.sum(image_output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_encoded_softmax += self.encoded_layer_output.T.dot(softmax_output_delta) * learning_rate
        self.bias_softmax += np.sum(softmax_output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_encoded_hidden += self.first_hidden_layer_output.T.dot(second_hidden_delta) * learning_rate
        self.bias_second_hidden += np.sum(second_hidden_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += X.T.dot(first_hidden_delta) * learning_rate
        self.bias_first_hidden += np.sum(first_hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y_image, y_label, epochs, learning_rate):
        for epoch in range(epochs):
            # Perform a forward pass (for all samples at once)
            image_output, softmax_output = self.forward(X)
            
            # Calculate loss for the image reconstruction (Mean Squared Error)
            image_loss = np.mean(np.square(y_image - image_output))  # Compare image outputs to true images
            
            # Calculate loss for the label prediction (Mean Squared Error)
            label_loss = np.mean(np.square(y_label - softmax_output))  # Compare label predictions to true labels
            
            # Combine both losses (you can adjust the weights of each task here if needed)
            total_loss = image_loss + label_loss
            
            # Perform a backward pass to update the weights and biases
            self.backward(X, y_image, y_label, learning_rate)
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Total Loss: {total_loss}, Image Loss: {image_loss}, Label Loss: {label_loss}')


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
