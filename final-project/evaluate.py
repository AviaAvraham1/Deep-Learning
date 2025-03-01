import logging
import torch

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


