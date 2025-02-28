from utils2 import save_model
import torch
import logging
from tqdm import tqdm

# TODO: implement the functions below. save the trained models, including the decoder

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def train_autoencoder(encoder, decoder, train_loader, val_loader, test_loader, args, encoder_filename, decoder_filename, log_filename):
    logging.basicConfig(filename="logs/"+log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Autoencoder Training")

    # Create schedulers and optimizer
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Using both MSE and L1 loss for better reconstruction
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    encoder.train()
    decoder.train()

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10  # Early stopping patience

    for epoch in range(args.epochs):
        epoch_train_loss = 0
        for images, _ in tqdm(train_loader):
            images = images.to(args.device)
            
            # Add slight noise to input images for denoising autoencoder behavior
            if epoch > 5:  # Start adding noise after a few epochs of normal training
                noise_factor = 0.1
                noisy_images = images + noise_factor * torch.randn_like(images)
                noisy_images = torch.clamp(noisy_images, -1, 1)
                latent = encoder(noisy_images)
            else:
                latent = encoder(images)
                
            reconstructed = decoder(latent)
            
            # Combined loss
            loss = 0.8 * mse_loss(reconstructed, images) + 0.2 * l1_loss(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation Step
        encoder.eval()
        decoder.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(args.device)
                latent = encoder(images)
                reconstructed = decoder(latent)
                loss = 0.8 * mse_loss(reconstructed, images) + 0.2 * l1_loss(reconstructed, images)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(encoder, "models/"+encoder_filename)
            save_model(decoder, "models/"+decoder_filename)
            logging.info(f"New best model saved (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Return to training mode
        encoder.train()
        decoder.train()

    logging.info(f"Best Validation Loss: {best_val_loss:.4f}")

    # Final test evaluation
    encoder.eval()
    decoder.eval()
    total_test_loss = 0
    total_mae_loss = 0  # Also track MAE for reporting

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(args.device)
            latent = encoder(images)
            reconstructed = decoder(latent)
            loss = mse_loss(reconstructed, images)
            mae = l1_loss(reconstructed, images)
            total_test_loss += loss.item()
            total_mae_loss += mae.item()

    avg_test_loss = total_test_loss / len(test_loader)
    avg_mae_loss = total_mae_loss / len(test_loader)
    logging.info(f"Test reconstruction MSE: {avg_test_loss:.4f}")
    logging.info(f"Test reconstruction MAE: {avg_mae_loss:.4f}")
    logging.info("Autoencoder training complete")


def train_classifier_on_frozen_encoder(encoder, classifier, train_loader, val_loader, test_loader, args, classifier_filename, log_filename):
    logging.basicConfig(filename="logs/"+log_filename, level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting Classifier Training on Frozen Encoder")

    # Freeze the encoder so it's not updated
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Set up optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Loss function with label smoothing for better generalization
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    classifier.train()  # Set classifier to training mode

    best_val_acc = 0.0  # Track best validation accuracy
    patience_counter = 0
    max_patience = 15  # Early stopping patience

    # Create augmented versions of the training dataset for better generalization
    latent_features = []
    labels_list = []
    
    # Extract and store latent features for faster training
    encoder.eval()
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Extracting latent features"):
            images = images.to(args.device)
            latent = encoder(images)
            latent_features.append(latent.cpu())
            labels_list.append(labels)
    
    latent_features = torch.cat(latent_features)
    labels_list = torch.cat(labels_list)
    
    for epoch in range(args.epochs):
        epoch_train_loss = 0
        correct_train = 0
        total_train = 0

        # Create a shuffled index array for this epoch
        indices = torch.randperm(latent_features.size(0))
        
        # Train in batches
        batch_size = train_loader.batch_size
        num_batches = len(indices) // batch_size
        
        for batch_idx in range(num_batches):
            # Get batch indices
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # Get batch data
            batch_features = latent_features[batch_indices].to(args.device)
            batch_labels = labels_list[batch_indices].to(args.device)
            
            # Apply feature noise as data augmentation (only after a few epochs)
            if epoch > 5:
                noise_factor = 0.05
                batch_features = batch_features + noise_factor * torch.randn_like(batch_features)
            
            preds = classifier(batch_features)
            loss = loss_fn(preds, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_train_loss += loss.item()
            correct_train += (preds.argmax(dim=1) == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        avg_train_loss = epoch_train_loss / num_batches
        train_acc = correct_train / total_train

        # Validation Step
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
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)

        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(classifier, "models/"+classifier_filename)
            logging.info(f"New best classifier saved! (Val Acc: {best_val_acc:.4f})")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        classifier.train()  # Return classifier to training mode

    logging.info(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Load the best model for final evaluation
    classifier = torch.load("models/"+classifier_filename)
    
    # Final test evaluation
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