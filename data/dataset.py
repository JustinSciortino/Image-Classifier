import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
#import logging
#logging.basicConfig(filename='dataset.log', level=logging.ERROR)

def load_cifar10_data(train_size=500, test_size=100):
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    def select_first_n_images(dataset, num_per_class):
        indices = [] #store the indices (positions) of the images in the dataset that we want to include in our subset
        class_counts = [0] * 10 #list contain counters for the number of images per class class_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # One counter for each class, all starting at 0
        for i, (_, label) in enumerate(dataset): #i is the index of the image in the dataset, label is the class of the image, (_, label) unpacks the image data and its class label where label is an interger from 0-9
            if class_counts[label] < num_per_class: #condition checks if we’ve already selected the required number of images for the current class (num_per_class).
                indices.append(i)
                class_counts[label] += 1 #Increments the counter in class_counts for this specific class, tracking that we’ve selected one more image for this class.
                #logging.info(f"Class {label}: {class_counts[label]}/{num_per_class} images selected")
            # Check if we have all required images
            if all(count == num_per_class for count in class_counts):
                break
        # Verify final counts
        #for label, count in enumerate(class_counts):
            #if count != num_per_class:
                #logging.error(f"Class {label} has {count} images instead of {num_per_class}")
                
        # Log total images selected
        #total_selected = sum(class_counts)
        #expected_total = num_per_class * 10
        #logging.info(f"Total images selected: {total_selected}/{expected_total}")

        return Subset(dataset, indices) #Subset object containing only the images with indices in indices
                                        #Does not contain tensors
                                        #Subset object itself contains references to the original data in the form of transformed image tensors along with the associated labels, not the raw image data


    train_data = select_first_n_images(train_data, train_size)
    #logging.info(f"Training set size: {len(train_data)}")
    test_data = select_first_n_images(test_data, test_size)
    #logging.info(f"Test set size: {len(test_data)}")
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_loader, test_loader
