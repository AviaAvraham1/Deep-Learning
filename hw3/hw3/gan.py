import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np
from .autoencoder import EncoderCNN, DecoderCNN

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        in_channels = in_size[0]
        self.cnn = EncoderCNN(in_channels, out_channels=128)
        self.num_features = self._calc_num_cnn_features(in_size)
        self.fc = nn.Linear(self.num_features, 1)
        # ========================

    def _calc_num_cnn_features(self, in_shape):
        with torch.no_grad():
            x = torch.zeros(1, *in_shape)
            out_shape = self.cnn(x).shape
        return int(np.prod(out_shape))

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        features = self.cnn(x)  # Extract features using the CNN
        features_flat = features.view(features.size(0), -1)  # Flatten the features
        y = self.fc(features_flat)  # Map to scalar output
        #y.squeeze(-1)  # Squeeze to remove extra dimension if present
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim


        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.featuremap_size = featuremap_size

        # CORRECTED: Output size is 128 * featuremap_size^2
        self.linear = nn.Linear(z_dim, 128 * featuremap_size ** 2)
        
        # Use DecoderCNN with adjusted in_channels=256
        #self.decoder = DecoderCNN(in_channels=256, out_channels=out_channels) # something doesn't work with the Decoder modules, putting here a fixed one...
        self.decoder = self._build_custom_decoder(out_channels)
        # ========================

    def _build_custom_decoder(self, out_channels):
        return nn.Sequential(
            # Layer 1: 4x4 → 8x8
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 2: 8x8 → 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: 16x16 → 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 4: 32x32 → 64x64
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn(n, self.z_dim, device=device)
        if with_grad:
            return self(z)
        else:
            with torch.no_grad():
                return self(z)

        # if we want a cooler version:
        # from contextlib import nullcontext
        # with torch.no_grad() if not with_grad else nullcontext():
        #     samples = self.forward(z) 

        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        # Project latent space z into feature map
        h_flat = self.linear(z)
        # CORRECTED: Reshape to (128, 4, 4) instead of (256, 4, 4)
        h = h_flat.view(-1, 128, self.featuremap_size, self.featuremap_size)
        x = self.decoder(h)

        # Decode feature map into an image
        #x = self.decoder(h)  # Shape: (N, C, H, W)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    # Generate noisy labels for real data
    target_real = data_label + (torch.rand_like(y_data) - 0.5) * label_noise
    # Generate noisy labels for generated data
    target_gen = (1 - data_label) + (torch.rand_like(y_generated) - 0.5) * label_noise

    # Compute BCE loss for real data
    loss_data = F.binary_cross_entropy_with_logits(y_data, target_real)
    # Compute BCE loss for generated data
    loss_generated = F.binary_cross_entropy_with_logits(y_generated, target_gen)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    target = torch.full_like(y_generated, data_label)
    
    # Loss calculation
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(y_generated, target)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_model.train()
    gen_model.eval()
    
    # Forward real data
    y_data = dsc_model(x_data)
    
    # Generate fake data
    z = torch.randn(x_data.shape[0], gen_model.z_dim, device=x_data.device)
    x_gen = gen_model(z).detach()  # Detach to avoid backprop into generator
    y_gen = dsc_model(x_gen)
    
    # Compute discriminator loss and update
    print(dsc_loss_fn)
    dsc_loss = dsc_loss_fn(y_data, y_gen)
    dsc_optimizer.zero_grad()
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    dsc_model.eval()
    gen_model.train()
    
    # Generate fake data with gradients
    x_gen = gen_model.sample(x_data.shape[0], with_grad=True)
    y_gen = dsc_model(x_gen)
    
    # Compute generator loss and update
    gen_loss = gen_loss_fn(y_gen)
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) % 5 != 0:
        return saved
    # ========================
    torch.save(gen_model, checkpoint_file)
    print(f"*** Saved checkpoint {checkpoint_file} ")
    saved = True
    return saved
