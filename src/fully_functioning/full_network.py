import numpy as np

class EncoderDecoderNetwork:
    def __init__(self, network_path):
        self.load(network_path)

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
        
        # Encoded layer to Softmax Hidden Layer
        self.encoded_s_hidden_input = np.dot(self.encoded_layer_output, self.weights_encoded_s_hidden) + self.bias_s_hidden
        self.encoded_s_hidden_output = self.leaky_relu(self.encoded_s_hidden_input)
        
        # Second Hidden Layer to Image Output (Reconstruction) with Sigmoid Activation
        self.output_layer_input = np.dot(self.second_hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.image_output = self.leaky_relu(self.output_layer_input)
        
        # Softmax Hidden Layer to Softmax Output
        self.hidden_softmax_input = np.dot(self.encoded_s_hidden_output, self.weights_s_hidden_softmax) + self.bias_softmax
        self.softmax_output = self.softmax(self.hidden_softmax_input)
        
        return self.image_output, self.softmax_output

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  # Clamp x values to avoid overflow
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    def softmax(self, x):
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x / np.sum(x, axis=1, keepdims=True)
    
    def softmax_cross_entropy_loss_gradient(softmax_output, true_labels):
        return softmax_output - true_labels
    
    def clip_gradients(self, max_norm=5.0):
        self.weights_input_hidden = np.clip(self.weights_input_hidden, -max_norm, max_norm)
        self.weights_hidden_output = np.clip(self.weights_hidden_output, -max_norm, max_norm)
        
    def cross_entropy_loss(self, y_true, y_pred):
        # Clip the values of predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        
    def save(self, file_path):
        
        np.savez(
            file_path,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            encoded_size=self.encoded_size,
            output_size=self.output_size,
            softmax_hidden_size=self.softmax_hidden_size,
            softmax_size=self.softmax_size,
            weights_input_hidden=self.weights_input_hidden,
            weights_hidden_encoded=self.weights_hidden_encoded,
            weights_encoded_hidden=self.weights_encoded_hidden,
            weights_hidden_output=self.weights_hidden_output,
            weights_encoded_s_hidden=self.weights_encoded_s_hidden,
            weights_s_hidden_softmax=self.weights_s_hidden_softmax,
            bias_first_hidden=self.bias_first_hidden,
            bias_encoded=self.bias_encoded,
            bias_second_hidden=self.bias_second_hidden,
            bias_output=self.bias_output,
            bias_s_hidden=self.bias_s_hidden,
            bias_softmax=self.bias_softmax,
        )
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        
        data = np.load(file_path)
        
        #Initialize sizes
        self.input_size=int(data['input_size'])
        self.hidden_size=int(data['hidden_size'])
        self.encoded_size=int(data['encoded_size'])
        self.output_size=int(data['output_size'])
        self.softmax_hidden_size=int(data['softmax_hidden_size'])
        self.softmax_size=int(data['softmax_size'])
        
        # Initialize weights and biases
        self.weights_input_hidden=data['weights_input_hidden']
        self.weights_hidden_encoded=data['weights_hidden_encoded']
        self.weights_encoded_hidden=data['weights_encoded_hidden']
        self.weights_hidden_output=data['weights_hidden_output']
        self.weights_encoded_s_hidden=data['weights_encoded_s_hidden']
        self.weights_s_hidden_softmax=data['weights_s_hidden_softmax']
        self.bias_first_hidden=data['bias_first_hidden']
        self.bias_encoded=data['bias_encoded']
        self.bias_second_hidden=data['bias_second_hidden']
        self.bias_output=data['bias_output']
        self.bias_s_hidden=data['bias_s_hidden']
        self.bias_softmax=data['bias_softmax']
