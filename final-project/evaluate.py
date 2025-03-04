import logging
import torch
from classes import Encoder, Classifier, Decoder
from utils2 import load_model

def evaluate_classifier(encoder, classifier, train_loader, val_loader, test_loader, args):
    encoder.eval()
    classifier.eval()

    def compute_accuracy(dataloader):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(args.device), labels.to(args.device)
                latent = encoder(images)
                outputs = classifier(latent)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        return correct / total

    train_acc = compute_accuracy(train_loader)
    val_acc = compute_accuracy(val_loader)
    test_acc = compute_accuracy(test_loader)

    logging.info(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    
    return train_acc, val_acc, test_acc


def evaluate_autoencoder(encoder, decoder, train_loader, val_loader, test_loader, device):
    """
    Computes MAE for autoencoder reconstruction on Train, Validation, and Test sets.
    """
    encoder.eval()
    decoder.eval()
    loss_fn = torch.nn.L1Loss()

    def compute_mae(dataloader):
        total_mae = 0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                latent = encoder(images)
                reconstructed = decoder(latent)
                mae = loss_fn(reconstructed, images)
                total_mae += mae.item()
        return total_mae / len(dataloader)

    train_mae = compute_mae(train_loader)
    val_mae = compute_mae(val_loader)
    test_mae = compute_mae(test_loader)

    logging.info(f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")

    return train_mae, val_mae, test_mae

import torch
import matplotlib.pyplot as plt

def plot_reconstructions_util(encoder, decoder, test_loader, device, filename="reconstruction_results.png"):
    encoder.eval()
    decoder.eval()

    # select 5 random images from the test set
    images, _ = next(iter(test_loader))
    images = images[:5].to(device)

    with torch.no_grad():
        latent = encoder(images)
        reconstructed = decoder(latent)

    # convert tensors to NumPy format
    images = images.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5  # Unnormalize
    reconstructed = reconstructed.cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5  # Unnormalize

    # plot original images and their reconstructions
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(images[i])  # original image
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i])  # reconstructed image
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")

    plt.savefig(filename)
    plt.close()

    print(f"Reconstruction results saved to {filename}")


def plot_reconstruction(args, test_loader):
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    dataset_class = "MNIST" if args.mnist else "CIFAR10"
    encoder = load_model(encoder, f"models/encoder_frozen.pt", args.device)
    decoder = load_model(decoder, f"models/trained_decoder.pt", args.device)
    filename = f"reconstruction_results_{dataset_class.lower()}.png"
    plot_reconstructions_util(encoder, decoder, test_loader, args.device, filename=filename)