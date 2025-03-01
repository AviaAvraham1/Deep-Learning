import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pickle



def load_data(args):
    """ Loads dataset and returns train/val/test DataLoaders. """
    
    # convert MNIST to 3-channel RGB if needed (by copying the same value to all channels) and to match CIFAR images size
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3) if args.mnist else transforms.Lambda(lambda x: x),  # convert grayscale to RGB
        transforms.Resize((32, 32)),  # resize MNIST to match CIFAR-10 dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # transforms.Normalize(mean=[0.5], std=[0.5]) if args.mnist else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset_class = datasets.MNIST if args.mnist else datasets.CIFAR10
    dataset = dataset_class(root=args.data_path, train=True, download=True, transform=transform)
    test_dataset = dataset_class(root=args.data_path, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    model.eval()
    return model
