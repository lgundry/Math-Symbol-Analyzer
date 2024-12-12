import numpy as np

class EncoderDecoderNetwork:
    def __init__(self, input_size, hidden_size, encoded_size, output_size, softmax_hidden_size, softmax_size):
        # Translator sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoded_size = encoded_size
        self.output_size = output_size
        
        # Categorizer sizes
        self.softmax_hidden_size = softmax_hidden_size
        self.softmax_size = softmax_size

        # Initialize translator weights (overwritten by load) - uses Xavier Initialization
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2 / (self.input_size + self.hidden_size))
        self.weights_hidden_encoded = np.random.randn(self.hidden_size, self.encoded_size) * np.sqrt(2 / (self.hidden_size + self.encoded_size))
        self.weights_encoded_hidden = np.random.randn(self.encoded_size, self.hidden_size) * np.sqrt(2 / (self.encoded_size + self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2 / (self.hidden_size + self.output_size))
        
        # Initialize categorizer weights
        self.weights_encoded_s_hidden = np.random.randn(self.encoded_size, self.softmax_hidden_size) * np.sqrt(2 / (self.encoded_size + self.softmax_hidden_size))
        self.weights_s_hidden_softmax = np.random.randn(self.softmax_hidden_size, self.softmax_size) * np.sqrt(2 / (self.softmax_hidden_size + self.softmax_size))

        # Initialize biases to zero for translator (overwritten by load)
        self.bias_first_hidden = np.zeros((1, self.hidden_size))
        self.bias_encoded = np.zeros((1, self.encoded_size))
        self.bias_second_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
        # Initialize biases to zero for categorizer
        self.bias_s_hidden = np.zeros((1, self.softmax_hidden_size))
        self.bias_softmax = np.zeros((1, self.softmax_size))

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
        
        return self.image_output, self.softmax_output, self.encoded_layer_output

    def backward(self, input_from_encoded, target_label, learning_rate, clip_value=1.0):
       # Calculate the error
        softmax_output_error = self.softmax_output - target_label  # Cross-entropy loss gradient
        softmax_output_delta = softmax_output_error  # No activation function derivative for softmax cross-entropy combination

        softmax_hidden_error = softmax_output_delta.dot(self.weights_s_hidden_softmax.T)
        softmax_hidden_delta = softmax_hidden_error * self.leaky_relu_derivative(self.encoded_s_hidden_output)

        # Clip gradients
        softmax_output_delta = np.clip(softmax_output_delta, -clip_value, clip_value)
        softmax_hidden_delta = np.clip(softmax_hidden_delta, -clip_value, clip_value)

        # Update weights and biases for the softmax layers
        self.weights_s_hidden_softmax -= self.encoded_s_hidden_output.T.dot(softmax_output_delta) * learning_rate
        self.bias_softmax -= np.sum(softmax_output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_encoded_s_hidden -= input_from_encoded.T.dot(softmax_hidden_delta) * learning_rate
        self.bias_s_hidden -= np.sum(softmax_hidden_delta, axis=0, keepdims=True) * learning_rate


    def train(self, input, target_label, learning_rate):
        # Normalize input and target image to [0, 1]
        input = input / 255.0  # Normalize input
        
        # Perform a forward pass
        image_output, label_output, categorizer_input = self.forward(input)
            
        # Calculate loss for the image reconstruction (Mean Squared Error)
        label_loss = self.cross_entropy_loss(target_label, label_output)
            
        # Perform a backward pass to update the weights and biases
        self.backward(categorizer_input, target_label, learning_rate)

        return label_loss, image_output, label_output

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

    def l2_regularization(self, lambda_reg):
        l2_loss = lambda_reg * np.sum(np.square(self.weights_input_hidden))
        return l2_loss
    
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
        self.input_size=int(data['input_size'])
        self.hidden_size=int(data['hidden_size'])
        self.encoded_size=int(data['encoded_size'])
        self.output_size=int(data['output_size'])
        
        self.weights_input_hidden = data['weights_input_hidden']
        self.weights_hidden_encoded = data['weights_hidden_encoded']
        self.weights_encoded_hidden = data['weights_encoded_hidden']
        self.weights_hidden_output = data['weights_hidden_output']
        self.bias_first_hidden = data['bias_first_hidden']
        self.bias_encoded = data['bias_encoded']
        self.bias_second_hidden = data['bias_second_hidden']
        self.bias_output = data['bias_output']
        print(f"Model loaded from {file_path}")
