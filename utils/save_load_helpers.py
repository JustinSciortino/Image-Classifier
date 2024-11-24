import numpy as np
import os
import torch

#* Used to save numpy arrays to a .npz file
def save_data(train_features, train_labels, test_features, test_labels, 
              train_features_pca, test_features_pca):
    """Save NumPy arrays to a .npz file."""
    TRAINED_MODEL_DIR = "trained_models"
    model_path = os.path.join(TRAINED_MODEL_DIR, "cifar10_data.npz")
    np.savez(model_path, 
             train_features=train_features, 
             train_labels=train_labels,
             test_features=test_features, 
             test_labels=test_labels,
             train_features_pca=train_features_pca, 
             test_features_pca=test_features_pca)
    
#* Used to load numpy arrays from a .npz file
def load_data(filename):
    TRAINED_MODEL_DIR = "trained_models"
    model_path = os.path.join(TRAINED_MODEL_DIR, filename)
    data = np.load(model_path)
    return (data['train_features'], data['train_labels'], 
            data['test_features'], data['test_labels'], 
            data['train_features_pca'], data['test_features_pca'])

#* Used to save tensors to a .pt file
def save_tensors(train_features_tensors, train_labels_tensors, test_features_tensors, test_labels_tensors):
    TRAINED_MODEL_DIR = "trained_models"
    model_path = os.path.join(TRAINED_MODEL_DIR, "cifar10_tensors.pt")
    data = {
        'train_features_tensors': train_features_tensors,
        'train_labels_tensors': train_labels_tensors,
        'test_features_tensors': test_features_tensors,
        'test_labels_tensors': test_labels_tensors,
    }
    torch.save(data, model_path)
    print(f"Tensors saved to {model_path}")

#* Used to load tensors from a .pt file
def load_tensors():
    TRAINED_MODEL_DIR = "trained_models"
    model_path = os.path.join(TRAINED_MODEL_DIR, "cifar10_tensors.pt")

    if os.path.exists(model_path):
        data = torch.load(model_path)
        print(f"Tensors loaded from {model_path}")
        return (
            data['train_features_tensors'],
            data['train_labels_tensors'],
            data['test_features_tensors'],
            data['test_labels_tensors'],
        )
    else:
        print(f"File {model_path} does not exist.")
        return None, None, None, None
