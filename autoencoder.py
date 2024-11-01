import torch.nn as nn
import torch.nn.init as init

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
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
