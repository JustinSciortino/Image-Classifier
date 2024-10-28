import numpy as np
import os

def save_data(train_features, train_labels, test_features, test_labels, 
              train_features_pca, test_features_pca, filename):
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
    
def load_data(filename):
    TRAINED_MODEL_DIR = "trained_models"
    model_path = os.path.join(TRAINED_MODEL_DIR, filename)
    data = np.load(model_path)
    return (data['train_features'], data['train_labels'], 
            data['test_features'], data['test_labels'], 
            data['train_features_pca'], data['test_features_pca'])