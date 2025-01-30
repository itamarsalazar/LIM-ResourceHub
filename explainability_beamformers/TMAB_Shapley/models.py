import torch
import torch.nn as nn

class SimpleBeamformer(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(SimpleBeamformer, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)

        # Decoder 
        d1 = self.dec1(x2)
        d2 = self.dec2(d1) # + x1)
        return d2


# Modelo sin skip connections ni batch normalization
class SimpleBeamformerV3(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(SimpleBeamformerV3, self).__init__()
        
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Decoder path
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        d1 = self.dec1(x2)
        d2 = self.dec2(d1)
        return d2
