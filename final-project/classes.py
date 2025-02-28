import torch.nn as nn

""" These are implementations to use in the different training variations."""
# TODO: delete unnecessary comments

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 2*3, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            nn.ReLU(),
            nn.Conv2d(2*3, 2*3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*3, 2*3, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            nn.ReLU(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*3, latent_dim)
        )

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        c_hid = 3
        act_fn = nn.GELU
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.network = nn.Sequential(
            nn.ConvTranspose2d(2*3, 2*3, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            nn.ReLU(),
            nn.Conv2d(2*3, 2*3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*3, 3, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        return self.network(x)


class Classifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)
