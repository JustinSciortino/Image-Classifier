from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.decomposition import PCA
import torch
import numpy as np

def extract_resnet18_features(data_loader):
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  
    resnet18.eval()

    features, labels = [], []
    with torch.no_grad():
        for images, lbls in data_loader:
            outputs = resnet18(images)
            outputs = outputs.view(outputs.size(0), -1)
            features.append(outputs.numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def apply_pca(features, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features