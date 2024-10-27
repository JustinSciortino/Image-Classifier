import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_cifar10_data(train_size=500, test_size=100):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    def select_subset(dataset, num_per_class):
        indices = []
        class_count = [0] * 10
        for i, (_, label) in enumerate(dataset):
            if class_count[label] < num_per_class:
                indices.append(i)
                class_count[label] += 1
        return Subset(dataset, indices)

    train_data = select_subset(train_data, train_size)
    test_data = select_subset(test_data, test_size)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, test_loader
