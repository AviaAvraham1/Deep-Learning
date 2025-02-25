from utils2 import save_model
import torch
import logging

# TODO: implement the functions below. save the trained models, including the decoder

def train_autoencoder(encoder, decoder, train_loader, val_loader, test_loader, args, encoder_filename, decoder_filename, log_filename):

    logging.basicConfig(filename="logs/"+log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Autoencoder Training")

    # setup the optimizer with the encoder and decoder parameters
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    loss_fn = torch.nn.L1Loss()  # MAE loss (since we are requestd to report results in MAE)
    # loss_fn = torch.nn.MSELoss()

    # set the models to training mode
    encoder.train()
    decoder.train()

    best_val_loss = float('inf')  # track best validation loss

    for epoch in range(args.epochs):
        epoch_train_loss = 0
        for images, _ in train_loader:
            images = images.to(args.device)
            latent = encoder(images)
            reconstructed = decoder(latent)
            loss = loss_fn(reconstructed, images)

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
                reconstructed = decoder(latent)
                loss = loss_fn(reconstructed, images)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(encoder, encoder_filename)
            save_model(decoder, decoder_filename)
            logging.info(f"New best model saved (Val Loss: {best_val_loss:.4f})")

        # return to training mode
        encoder.train()
        decoder.train()

    logging.info(f"Best Validation Loss: {best_val_loss:.4f}")

    # final test evaluation
    encoder.eval()
    decoder.eval()
    total_test_loss = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(args.device)
            latent = encoder(images)
            reconstructed = decoder(latent)
            loss = loss_fn(reconstructed, images)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    logging.info(f"Test reconstruction MAE: {avg_test_loss:.4f}")
    logging.info("Autoencoder training complete")


def train_classifier_on_frozen_encoder(encoder, classifier, train_loader, val_loader, test_loader, args, classifier_filename, log_filename):

    logging.basicConfig(filename="logs/"+log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Classifier Training on Frozen Encoder")

    # freeze the encoder so it's not updated
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # set up optimizer and loss function
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    classifier.train()  # set classifier to training mode

    best_val_acc = 0.0  # track best validation accuracy

    for epoch in range(args.epochs):
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                latent = encoder(images)  # get frozen encoder features

            preds = classifier(latent)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            correct_train += (preds.argmax(dim=1) == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train

        # validation Step
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
            save_model(classifier, classifier_filename)
            logging.info(f"New best classifier saved! (Val Acc: {best_val_acc:.4f})")

        classifier.train()  # return classifier to training mode

    logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # final test evaluation
    classifier.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            latent = encoder(images)
            preds = classifier(latent)
            correct_test += (preds.argmax(dim=1) == labels).sum().item()
            total_test += labels.size(0)

    test_acc = correct_test / total_test
    logging.info(f"Test accuracy: {test_acc:.4f}")

    logging.info("Classifier training complete")


def train_joint_encoder_classifier():
    # save_model(encoder, "encoder_joint.pth")
    # save_model(classifier, "classifier_joint.pth")
    pass

def train_contrastive_encoder():
    # save_model(encoder, "encoder_contrastive.pth")
    pass