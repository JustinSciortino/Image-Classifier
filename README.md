# Image-Classifier

---

## **Table of Contents**
1. [Installation Instructions](#installation-instructions)
2. [How to Perform Data Pre-processing](#how-to-perform-data-pre-processing)
3. [How to Train or Retrain the Models](#how-to-train-or-retrain-the-models)
4. [How to Evaluate the Models](#how-to-evaluate-the-models)
5. [Folder Structure and File Descriptions](#folder-structure-and-file-descriptions)

---

## **Installation Instructions**
Follow these steps to set up the project:

1. Clone the repository
```git clone https://github.com/JustinSciortino/Image-Classifier.git```

2. Navigate into the project directory
```cd Image-Classifier```
3. Install dependencies
```pipenv install```

4. Download the CNN trained models from the google drive and put them in the ```Image-Classifier/trained_models``` folder
https://drive.google.com/drive/folders/1mPnkFKZpHD2w0E7ymSbIeDvMzoFhw1j6?usp=sharing

5. Activate the virtual environment by entering the following command from the Image-Classifier directory:
   ```pipenv shell```

## **How to Perform Data Pre-processing**

After activating the virtual environment, to perform data pre-processing again, enter the following in the terminal: 

```python main.py --generate_data```

## **How to Train or Retrain the Models**

All models and configurations are already trained and are saved in ```Image-Classifier/trained_models```. However to retrain some or all of the models, enter the following command:

```python main.py --retrain <model name>``` or ```python main.py --retrain all``` to retrain all the models

Note that for the model names listed below, once the command is entered to retrain them, the trained models will be deleted and newly trained models will be added back. The only exception is for the MLP and CNN models which the original saved models will not be deleted, only any other model that was trained afterwards (file name will end with _retrained). To retrain those models, enter the following command, ```python main.py --retrain mlp``` or ```python main.py --retrain cnn```.  

Model names that could be retrained: custom_naive_bayes_pca, custom_naive_bayes, sklearn_naive_bayes_pca, sklearn_naive_bayes, custom_decision_tree_pca, custom_decision_tree_d50, custom_decision_tree_d15, custom_decision_tree_d10, custom_decision_tree_d5, sklearn_decision_tree_pca, sklearn_decision_tree_d50, sklearn_decision_tree_d10

## **How to Evaluate the Models**

To evaluate the models, use the following command: ```python main.py```

## **Folder Structure and File Descriptions**
- ```Image-Classifier/data```
    - ```dataset.py```: Contains all logic to load the CIFAR-10 dataset 
    - Directory where the ```cifar-10-batches-py``` and the ```cifar-10-python.tar.gz``` will be imported into
- ```Image-Classifier/models```
    - ```Image-Classifier/models/notebooks```
        - ```CNN.ipynb```: The jupyter notebook that was used to train the CNN model configurations on Google Colab using a GPU
        - ```MLP.ipynb```: The jupyter notebook that was used to train the MLP model configurations on Google Colab using a GPU
    - ```MLP.py```: MLP class and model, contains save and load methods to save and load then model, contains methods to train, evaluate and get metrics
    - ```VGG11.py```: CNN class and model, contains save and load methods to save and load then model, contains methods to train, calculate the flattened size, forward pass, evaluate and get metrics
    - ```CustomDecisionTree.py```: Decision Tree class and model, contains methods to build the tree, fit the tree, find the best split, calculate the gini impurity, predictions, and save/load the model
    - ```DecisionTreeWrapper.py```: Simply a wrapper class of the Scikit-learn's implementation of the Decision Tree Classifier. The wrapper class has added methods to save and load the model
    - ```CustomNaiveBayes.py```: Naive Bayes class and model, contains methods to fit the model, calculate the likelihood, make predictions (which calculates the posterior probabilities too), and save/load the model
    - ```GuassianNBWrapper.py```: Simply a wrapper class of the Scikit-learn's implementation of the Gaussian Naive Bayes model. The wrapper class has added methods to save and load the model
- ```Image-Classifier/trained_models```
    - Contains all the trained models for the four models and their configurations
    - Contains ```cifar_data.npz``` and ```cifar10_tensors.pt``` which are files of saved processed numpy and tensors. It avoids having to pre-process the data every single time the program is ran, and we can simply load the data from those files. 
    - Note that the CNN models in the Google Drive should be downloaded and put into this directory
- ```Image-Classifier/utils```
    - ```feature_extraction.py```: Contains a function that when called extracts all the features from the data loader using ResNet-18. The function also removes the last layer of ResNet-18
    - ```help_msg.py```: Contains a function that when called prints a help message to the user. Used when the user does not enter the right command to either pre-process or retrain the models again (list of instructions). 
    - ```reduce_feature_vector_size.py```: Contains a function to apply PCA, reduce the feature vector from 512 to 50 and return the vector as a numpy array
    - ```save_load_helpers.py```: Contains two save and two load functions used to save/load the pre-processed numpy arrays (cifar_data.npz) and tensors (cifar10_tensors.pt) from files. 
- ```Image-Classifier/main.py```
    - Contains all related functionality to retrain, pre-process and evaluate all the saved models. Also contains a function to return the evaluation metrics for the Naive Bayes and Decision Tree models. 
- ```Image-Classifier/Pipfile```
    - Contains project dependencies
- ```Image-Classifier/COMP 472 Project Report```
    - Contains the project report (PDF)
