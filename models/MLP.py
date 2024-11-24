TRAINED_MODEL_DIR = "trained_models"
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import warnings
import psutil
import time


if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)
#* Ingore warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
"""
Useful sources for information:
https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning
https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP:
    def __init__(self, train_tensors, train_labels, test_tensors, test_labels, num_features=512, hidden_size=512, output_size=10, learning_rate=0.001, momentum=0.9, batch_size=128, epochs=30, layers=3):

        #* Hyperparameters
        self.num_features = num_features
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs

        #* Metrics used to be saved/loaded
        self.training_time = 0
        self.GPU_mememory_allocated = 0
        self.GPU_max_memory_allocated = 0
        self.CPU_memory_usage = 0

        #* MLP architecture with different number of layers
        if layers == 3:
          self.model = nn.Sequential( 
              nn.Linear(num_features, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, output_size)
          ).to(device)

        elif layers == 2:
          self.model = nn.Sequential( 
              nn.Linear(num_features, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, output_size)
          ).to(device)

        elif layers == 4:
          self.model = nn.Sequential( 
              nn.Linear(num_features, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, output_size)
          ).to(device)

        elif layers == 6:
          self.model = nn.Sequential( 
              nn.Linear(num_features, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, output_size)
          ).to(device)


        #* Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        self.train_tensors = train_tensors.to(device)
        self.train_labels = train_labels.to(device)
        self.test_tensors = test_tensors.to(device)
        self.test_labels = test_labels.to(device)

        #* Create DataLoaders from the tensors
        train_dataset = TensorDataset(train_tensors, train_labels)
        test_dataset = TensorDataset(test_tensors, test_labels)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    #* Train model for one epoch
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    #* Evaluate the model on the test set and return the metrics
    def evaluate(self):
        start_time = time.time()
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        #* Metrics
        evaluation_time = time.time() - start_time
        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return total_loss / len(self.test_loader), accuracy, conf_matrix, precision, recall, f1, evaluation_time
    
    #* Train the model for the specified number of epochs in the constructor
    def train(self):
        start_time = time.time()
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
        self.training_time = time.time() - start_time
    
    #* Save the model to a file
    def save_model(self, filename="mlp_model.pth"):
        filepath = os.path.join(TRAINED_MODEL_DIR, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "training_time": self.training_time, 
            "GPU_mememory_allocated": self.GPU_mememory_allocated,
            "GPU_max_memory_allocated": self.GPU_max_memory_allocated,
            "CPU_memory_usage": self.CPU_memory_usage
        }, filepath)
        print(f"Model and metadata saved to {filepath}")

    #* Load the model from a file
    def load_model(self, filename="mlp_model.pth"):
        filepath = os.path.join(TRAINED_MODEL_DIR, filename)
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_time = checkpoint.get("training_time", 0)
        self.GPU_mememory_allocated = checkpoint.get("GPU_mememory_allocated", 0)
        self.GPU_max_memory_allocated = checkpoint.get("GPU_max_memory_allocated", 0)
        self.CPU_memory_usage = checkpoint.get("CPU_memory_usage", 0)
        self.model.eval()
        print(f"Model loaded from {filepath}, Training Time: {self.training_time:.2f} seconds")

    #* Get the memory usage of the model for both CPU and GPU
    def get_memory_usage(self):
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            print(f"GPU Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
            self.GPU_mememory_allocated = torch.cuda.memory_allocated() / 1e6
            self.GPU_max_memory_allocated = torch.cuda.max_memory_allocated() / 1e6
        else:
            process = psutil.Process(os.getpid())
            print(f"CPU Memory Usage: {process.memory_info().rss / 1e6:.2f} MB")
            self.CPU_memory_usage = process.memory_info().rss / 1e6

    #* Get the memory usage of the model for both CPU and GPU and the training time
    def get_mememory_training_data(self):
        return self.GPU_mememory_allocated, self.GPU_max_memory_allocated, self.CPU_memory_usage, self.training_time