from torchvision import models
from torchvision.models import ResNet18_Weights
import torch
import numpy as np
import logging
logging.basicConfig(filename='ddataset.log', level=logging.INFO)

def extract_resnet18_features(data_loader):
    #* Load pre trained ResNet18 model
    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    #* Retrieve all layers besides the last one and reconstruct a new model with the layers, 
    #* Remove the last layer of the ResNet18 model
    resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])  

    #* Set the model in evaluation mode
    resnet18.eval()

    features, labels = [], []
    with torch.no_grad(): #* Gradients not needed so disabled

        #* Iterate through the data loader images and labels
        for images, labels_ in data_loader:

            #* Pass the images through the model and extract the features
            outputs = resnet18(images)
            logging.info(f"Output shape: {outputs.shape}")

            #* Flatten the output tensor to a 2D array 
            outputs = outputs.view(outputs.size(0), -1)

            #* Convert the tensor (containing the feature vectors) to a numpy array and append to the features list
            features.append(outputs.numpy())

            #* Conver the labels tensor to a numpy array and append to the labels list
            labels.append(labels_.numpy())

    return np.concatenate(features), np.concatenate(labels)