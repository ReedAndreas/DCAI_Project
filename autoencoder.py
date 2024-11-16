import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # Halves dimensions
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # Halves dimensions
            nn.Conv3d(32, 1, kernel_size=3, stride=2, padding=1),  # Halves dimensions
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                1, 32, kernel_size=4, stride=2, padding=1
            ),  # Doubles dimensions
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, 16, kernel_size=4, stride=2, padding=1
            ),  # Doubles dimensions
            nn.ReLU(),
            nn.ConvTranspose3d(
                16, 1, kernel_size=4, stride=2, padding=1
            ),  # Doubles dimensions
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Ensure output size matches input size
        if decoded.size() != x.size():
            decoded = F.interpolate(
                decoded, size=x.size()[2:], mode="trilinear", align_corners=False
            )
        return decoded
