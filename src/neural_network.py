import numpy as np
import matplotlib.pyplot as plt

class EncoderDecoderNetwork:
    def __init__(self, input_size, bottleneck_size, hidden_sizes, num_labels):
        # Initialize weights for encoder layers
        self.encoder_weights = []
        layer_sizes = [input_size] + hidden_sizes + [bottleneck_size]
        for i in range(len(layer_sizes) - 1):
            self.encoder_weights.append(self.init_weights(layer_sizes[i], layer_sizes[i + 1]))
        
        # Initialize weights for decoder layers
        self.decoder_weights = []
        decoder_layer_sizes = [bottleneck_size] + hidden_sizes[::-1] + [input_size]
        for i in range(len(decoder_layer_sizes) - 1):
            self.decoder_weights.append(self.init_weights(decoder_layer_sizes[i], decoder_layer_sizes[i + 1]))
        
        # Initialize weights for label prediction layer
        self.label_weights = self.init_weights(bottleneck_size, num_labels)

    
    def init_weights(self, input_size, output_size):
        # Xavier initialization for weights
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (output_size, input_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)
    
    def forward(self, x):
        # Encoder: Compress the input
        bottleneck = self.sigmoid(self.encoder_weights @ x)
        
        # Decoder: Reconstruct the continuous features
        reconstructed_features = self.sigmoid(self.decoder_weights @ bottleneck)
        
        # Predict the label
        label_logits = self.label_weights @ bottleneck
        predicted_label = self.softmax(label_logits)
        
        return reconstructed_features, predicted_label
    
    def compute_loss(self, x, y, label, alpha=1.0, beta=1.0):
        # Forward pass
        reconstructed_features, predicted_label = self.forward(x)
        
        # MSE Loss for features
        mse_loss = np.mean((reconstructed_features - x) ** 2)
        
        # CrossEntropy Loss for labels
        true_label_one_hot = np.zeros(predicted_label.shape)
        true_label_one_hot[label] = 1
        cross_entropy_loss = -np.sum(true_label_one_hot * np.log(predicted_label))
        
        # Total loss
        total_loss = alpha * mse_loss + beta * cross_entropy_loss
        return total_loss

    def train(self, data, labels, learning_rate=0.01, epochs=100):
        # Store metrics for graphing
        reconstruction_losses = []
        classification_losses = []
        total_losses = []

        for epoch in range(epochs):
            total_loss = 0
            reconstruction_loss = 0
            classification_loss = 0

            for x, label in zip(data, labels):
                reconstructed, predicted_label = self.forward(x)
                target = np.zeros(len(self.label_weights))
                target[label] = 1
                
                # Compute reconstruction loss (e.g., Mean Squared Error)
                reconstruction_loss += np.mean((reconstructed - x) ** 2)
                
                # Compute classification loss (e.g., Cross-Entropy)
                classification_loss += -np.sum(target * np.log(predicted_label + 1e-10))
                
                # Backpropagate and update weights
                self.backward(x, label, reconstructed, predicted_label, learning_rate)

            # Average the losses
            total_loss = reconstruction_loss + classification_loss
            reconstruction_losses.append(reconstruction_loss / len(data))
            classification_losses.append(classification_loss / len(data))
            total_losses.append(total_loss / len(data))

            print(f"Epoch {epoch + 1}/{epochs} - Reconstruction Loss: {reconstruction_losses[-1]:.4f}, "
                  f"Classification Loss: {classification_losses[-1]:.4f}, "
                  f"Total Loss: {total_losses[-1]:.4f}")

        # Plot the metrics
        self.plot_metrics(reconstruction_losses, classification_losses, total_losses)

    def plot_metrics(self, reconstruction_losses, classification_losses, total_losses, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(reconstruction_losses, label='Reconstruction Loss', color='blue')
        plt.plot(classification_losses, label='Classification Loss', color='orange')
        plt.plot(total_losses, label='Total Loss', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Time')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()