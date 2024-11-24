import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time
import psutil

class VGG11(nn.Module):
    def __init__(self, layers = 8,num_classes=10, kernel_size=3):
        super(VGG11, self).__init__()
        self.kernel_size = kernel_size

        self.training_time = 0
        self.GPU_mememory_allocated = 0
        self.GPU_max_memory_allocated = 0
        self.CPU_memory_usage = 0

        #* CNN architecture for different experimental configurations
        if layers == 8:
            self.features = nn.Sequential(

                nn.Conv2d(3, 64, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
               
                nn.Conv2d(256, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
             
                nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            
                nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
              
                nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        elif layers == 6:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  

                nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), 
                
                nn.Conv2d(128, 256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), 
             
                nn.Conv2d(256, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
              
                nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), 
            )
        elif layers == 10:
            self.features = nn.Sequential(
              
                nn.Conv2d(3, 64, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  

                nn.Conv2d(64, 128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), 

                nn.Conv2d(128, 256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  

                nn.Conv2d(256, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), 

                nn.Conv2d(512, 512, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  
            )
        
        #* Used to calculate the flattened size after the convolutional layers, dynamically based on the architecture
        input_shape = (3, 32, 32)  
        _input = torch.zeros(1, *input_shape)  #* Input tensor 
        flattened_size = self.calculate_flattened_size(_input)

        #* Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    #* Dynamically calculate the flattened size based on the architecture
    def calculate_flattened_size(self, input_tensor):
        with torch.no_grad():
            output = self.features(input_tensor)
        return output.view(output.size(0), -1).size(1)

    def forward(self, x):
        #* Pass input through feature extractor (conv layers)
        x = self.features(x)
        #* Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        #* Pass through the classifier 
        x = self.classifier(x)
        return x

    #* Train the model
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, momentum=0.9, device='cpu'):

        self.to(device)

        #* Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        start_time = time.time()

        #* Set the model to training mode
        self.train()

        #* Train for the specified number of epochs
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            total_batches = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                #* Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                #* Backward pass and optimization
                loss.backward()
                optimizer.step()

                #* Accumulate loss
                epoch_loss += loss.item()
                total_batches += 1
            
            #* Print statistics every epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / total_batches:.4f}')

        self.training_time = time.time()-start_time

    #* Evaluate the model
    def evaluate_model(self, test_loader, device='cpu'):
        self.to(device)
        self.eval()

        correct = 0
        total = 0
        y_true = []
        y_pred = []
        start_time = time.time()

        #* No gradient calculations
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                #* Forward pass
                outputs = self(inputs)
                
                #* Predictions
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                #* Store true and predicted labels for the metrics
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total

        #* Metrics
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

        conf_matrix = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        eval_time = time.time()-start_time

        return accuracy, conf_matrix, precision, recall, f1, eval_time

    #* Save the model
    def save_model(self, path):
        TRAINED_MODEL_DIR = "trained_models"
        os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
        filepath = os.path.join(TRAINED_MODEL_DIR, path)
        torch.save({
            'model_state_dict': self.state_dict(),
            'training_time': self.training_time,
            'GPU_memory_allocated': self.GPU_mememory_allocated,
            'GPU_max_memory_allocated': self.GPU_max_memory_allocated,
            'CPU_memory_usage': self.CPU_memory_usage,
        }, filepath)

    #* Load the model
    def load_model(self, path, device='cpu'):
        TRAINED_MODEL_DIR = "trained_models"
        os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
        filepath = os.path.join(TRAINED_MODEL_DIR, path)
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_time = checkpoint.get('training_time', 0)
        self.GPU_mememory_allocated = checkpoint.get('GPU_memory_allocated', 0)
        self.GPU_max_memory_allocated = checkpoint.get('GPU_max_memory_allocated', 0)
        self.CPU_memory_usage = checkpoint.get('CPU_memory_usage', 0)
        self.to(device)

    #* Get memory usage of the model CPU or GPU
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

    #* Get memory usage of the model CPU or GPU and training time
    def get_mememory_training_data(self):
        return self.GPU_mememory_allocated, self.GPU_max_memory_allocated, self.CPU_memory_usage, self.training_time