import torch
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
from utils2 import load_data
from train import train_autoencoder, train_classifier_on_frozen_encoder, train_joint_encoder_classifier, train_contrastive_encoder
from classes import Encoder, Decoder, Classifier

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for optimizer')
    parser.add_argument('--contrastive', action='store_true', default=False, help='Use contrastive learning during training') # added argument for convenience

    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)

    train_loader, val_loader, test_loader = load_data(args)

    encoder = Encoder(latent_dim=args.latent_dim).to(args.device)
    classifier = Classifier(latent_dim=args.latent_dim, num_classes=NUM_CLASSES).to(args.device)
    
    if args.self_supervised:
        if args.contrastive:
            print("Training Encoder with Contrastive Learning...")
            train_contrastive_encoder(encoder, train_loader, args)
            print("Training Classifier on Contrastive Encoder...")
            train_classifier_on_frozen_encoder(encoder, classifier, train_loader, val_loader, args)
        else:
            decoder = Decoder(latent_dim=args.latent_dim).to(args.device)
            print("Training Self-Supervised Autoencoder...")
            train_autoencoder(encoder, decoder, train_loader, val_loader, test_loader, args, 
                                encoder_filename="encoder_frozen.pt",
                                decoder_filename="trained_decoder.pt",
                                log_filename="frozen_autoencoder.log")
            print("Training Classifier on Frozen Encoder...")
            train_classifier_on_frozen_encoder(encoder, classifier, train_loader, val_loader, test_loader, args, 
                                            classifier_filename="frozen_encoder_classifier.pt", 
                                            log_filename="frozen_encoder_classifier.log")
        
        
    else:
        print("Training Encoder & Classifier Jointly...")
        train_joint_encoder_classifier(encoder, classifier, train_loader, val_loader, args, contrastive=args.contrastive)

