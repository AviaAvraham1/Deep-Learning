import torch.nn as nn

""" These are implementations to use in the different training variations."""
# TODO: delete unnecessary comments

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),  # 32x32 → 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 16x16 → 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),  # 8x8 → 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),  # 8x8 → 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),  # 4x4 → 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)  # ✅ Fully connected layer (latent_dim = 128)
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim, 128 * 4 * 4)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, output_padding=0),  # 4x4 → 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),  # 8x8 → 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # 16x16 → 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Final layer, keeping it 32x32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128, 4, 4)  # Ensure correct reshape
        return self.network(x)

class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 512),  
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, num_classes)  # ✅ Final output layer
        )

    def forward(self, x):
        return self.network(x)
