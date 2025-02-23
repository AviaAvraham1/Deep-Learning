import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_data(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) if args.mnist else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset_class = datasets.MNIST if args.mnist else datasets.CIFAR10
    
    dataset = dataset_class(root=args.data_path, train=True, download=False, transform=transform)
    test_dataset = dataset_class(root=args.data_path, train=False, download=False, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# TODO: delete unnecessary comments
def save_model(model, filename):
    """ Saves a PyTorch model to a file. """
    torch.save(model.state_dict(), filename)
    print(f"✅ Model saved to {filename}")

def load_model(model, filename, device):
    """ Loads a PyTorch model from a file. """
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {filename}")
    return model
