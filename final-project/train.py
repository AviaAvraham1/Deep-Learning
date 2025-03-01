from utils2 import save_model
import torch
import logging
from tqdm import tqdm
from evaluate import evaluate_autoencoder, evaluate_classifier
from classes import ProjectionHead
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F



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

        for images, labels in tqdm(train_loader):
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
            save_model(encoder, "models/encoder_joint.pt")
            save_model(classifier, "models/classifier_joint.pt")
            logging.info(f"New best model saved! (Val Acc: {best_val_acc:.4f})")

    evaluate_classifier(encoder, classifier, train_loader, val_loader, test_loader, args)

    logging.info("Joint Training Complete")


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Computes the SimCLR NT-Xent loss for a batch.
    Args:
        z_i: First view embeddings (batch, latent_dim)
        z_j: Second view embeddings (batch, latent_dim)
        temperature: Temperature scaling factor (default: 0.5)
    """
    batch_size = z_i.shape[0]

    # normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # compute similarity matrix (dot product between all samples)
    similarity_matrix = torch.cat([z_i, z_j], dim=0)  # 2N x latent_dim
    similarity_matrix = torch.mm(similarity_matrix, similarity_matrix.T)  # (2N x 2N)

    # extract positive samples (diagonal shift)
    sim_ij = torch.diag(similarity_matrix, batch_size)  # Positive pairs
    sim_ji = torch.diag(similarity_matrix, -batch_size)  # Positive pairs
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    # compute denominator for all negatives
    mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
    mask.fill_diagonal_(False)  # Remove self-similarity
    negatives = similarity_matrix[mask].view(2 * batch_size, -1)

    # compute NT-Xent loss
    numerator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(negatives / temperature), dim=1)

    loss = -torch.log(numerator / denominator).mean()
    return loss

def get_contrastive_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def train_contrastive_encoder(encoder, train_loader, val_loader, test_loader, args):
    """
    Trains an encoder using SimCLR contrastive learning
    """
    logging.basicConfig(filename="logs/contrastive_training.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Contrastive Learning with SimCLR")

    device = args.device
    encoder.to(device)
    
    projection_head = ProjectionHead(in_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=args.lr)

    contrastive_transform = get_contrastive_transforms()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        encoder.train()
        projection_head.train()
        total_train_loss = 0

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):

            # convert tensor to PIL image and apply data augmentation for contrastive learning
            img_1 = [contrastive_transform(to_pil_image(img)) for img in images]
            img_2 = [contrastive_transform(to_pil_image(img)) for img in images]

            # convert back to tensors
            img_1 = torch.stack(img_1)
            img_2 = torch.stack(img_2)

            img_1, img_2 = img_1.to(device), img_2.to(device)

            # forward pass through encoder + projection head
            z_i = projection_head(encoder(img_1))
            z_j = projection_head(encoder(img_2))

            # compute contrastive loss
            loss = nt_xent_loss(z_i, z_j, temperature=0.5)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # validation step
        encoder.eval()
        projection_head.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, _ in val_loader:
                img_1 = [contrastive_transform(to_pil_image(img)) for img in images]
                img_2 = [contrastive_transform(to_pil_image(img)) for img in images]

                img_1 = torch.stack(img_1)
                img_2 = torch.stack(img_2)

                img_1, img_2 = img_1.to(device), img_2.to(device)

                z_i = projection_head(encoder(img_1))
                z_j = projection_head(encoder(img_2))

                val_loss = nt_xent_loss(z_i, z_j, temperature=0.5)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(encoder, "models/encoder_contrastive.pt")
            logging.info(f"New best encoder saved (Val Loss: {best_val_loss:.4f})")

    logging.info("Contrastive Learning Complete")
