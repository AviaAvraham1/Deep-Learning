from utils2 import save_model
import torch
import logging
from tqdm import tqdm
from evaluate import evaluate_autoencoder, evaluate_classifier


SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def train_autoencoder(encoder, decoder, train_loader, val_loader, test_loader, args, encoder_filename, decoder_filename, log_filename):
    logging.basicConfig(filename="logs/"+log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Autoencoder Training")

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    loss_fn = torch.nn.MSELoss() 

    encoder.train()
    decoder.train()

    best_val_loss = float('inf')  # track best validation loss

    for epoch in range(args.epochs):
        epoch_train_loss = 0
        for images, _ in tqdm(train_loader):
            images = images.to(args.device)
            latent = encoder(images)

            latent += 0.05 * torch.randn_like(latent)  # small Gaussian noise added

            # forward pass
            reconstructed = decoder(latent)
            loss = loss_fn(reconstructed, images)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # validation Step
        encoder.eval()
        decoder.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(args.device)
                latent = encoder(images)

                latent += 0.05 * torch.randn_like(latent)  # small Gaussian noise added

                reconstructed = decoder(latent)
                loss = loss_fn(reconstructed, images)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(encoder, "models/"+encoder_filename)
            save_model(decoder, "models/"+decoder_filename)
            logging.info(f"New best model saved (Val Loss: {best_val_loss:.4f})")

        # return to training mode
        encoder.train()
        decoder.train()

    logging.info(f"Best Validation Loss: {best_val_loss:.4f}")

    evaluate_autoencoder(encoder, decoder, train_loader, val_loader, test_loader, args.device)

    logging.info("Autoencoder training complete")



def train_classifier_on_frozen_encoder(encoder, classifier, train_loader, val_loader, test_loader, args, classifier_filename, log_filename):

    logging.basicConfig(filename="logs/"+log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Classifier Training on Frozen Encoder")

    # freeze the encoder so it's not updated
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    classifier.train()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                latent = encoder(images)  # get frozen encoder features

            # forward pass
            preds = classifier(latent)
            loss = loss_fn(preds, labels)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            correct_train += (preds.argmax(dim=1) == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train

        # validation step
        classifier.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                latent = encoder(images)
                preds = classifier(latent)
                correct_val += (preds.argmax(dim=1) == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(classifier, "models/"+classifier_filename)
            logging.info(f"New best classifier saved! (Val Acc: {best_val_acc:.4f})")

        classifier.train()  # return classifier to training mode

    logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")

    evaluate_classifier(encoder, classifier, train_loader, val_loader, test_loader, args)

    logging.info("Classifier training complete")


def train_joint_encoder_classifier(encoder, classifier, train_loader, val_loader, test_loader, args):

    logging.basicConfig(filename="logs/joint_training.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Joint Training of Encoder & Classifier")

    device = args.device
    encoder.to(device)
    classifier.to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()

        total_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)

            # forward pass (includes encoder then classifier)
            latent = encoder(images)
            preds = classifier(latent)
            loss = loss_fn(preds, labels)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_train += (preds.argmax(dim=1) == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # validation
        encoder.eval()
        classifier.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                latent = encoder(images)
                preds = classifier(latent)
                correct_val += (preds.argmax(dim=1) == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(encoder, "models/encoder_joint.pth")
            save_model(classifier, "models/classifier_joint.pth")
            logging.info(f"New best model saved! (Val Acc: {best_val_acc:.4f})")

    evaluate_classifier(encoder, classifier, train_loader, val_loader, test_loader, args)

    logging.info("Joint Training Complete")


def train_contrastive_encoder():
    # save_model(encoder, "encoder_contrastive.pth")
    pass