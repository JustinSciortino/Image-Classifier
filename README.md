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

All models and configurations are already trained are in saved in ```Image-Classifier/trained_models```. However to retrain some or all of the models, enter the following command:

```python main.py --retrain <model name>``` or ```python main.py --retrain all``` to retrain all the models

Model names: custom_naive_bayes_pca, custom_naive_bayes, sklearn_naive_bayes_pca, sklearn_naive_bayes, custom_decision_tree_pca, custom_decision_tree_d50, custom_decision_tree_d15, custom_decision_tree_d10, custom_decision_tree_d5, sklearn_decision_tree_pca, sklearn_decision_tree_d50, sklearn_decision_tree_d10

## **How to Evaluate the Models**

To evaluate the models, use the following command: ```python main.py```

## **Folder Structure and File Descriptions**
