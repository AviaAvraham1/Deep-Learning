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
    
    return train_acc, val_acc, test_acc  # ✅ Return all accuracy values
