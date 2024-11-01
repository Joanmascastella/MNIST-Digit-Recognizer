import torch.nn as nn
from helpful_functions import initialize_weights

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoded_size):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, encoded_size)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()  # to scale output to [0, 1]
        )

        # Apply weights initialization
        self.apply(initialize_weights)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

