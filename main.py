from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from data import load_cifar10_data
from utils import extract_resnet18_features, apply_pca, help_msg, save_data, load_data
from models import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
import os
import torch
import argparse
import numpy as np

TRAINED_MODEL_DIR = "trained_models"
if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def save_sklearn_model(model, filename):
    TRAINED_MODEL_DIR = "trained_models"
    model_path = os.path.join(TRAINED_MODEL_DIR, filename)
    
    # Check if attributes exist before attempting to save them
    data_to_save = {
        "theta_": model.theta_,             # Class means
        "class_prior_": model.class_prior_, # Class priors
        "classes_": model.classes_,         # Class labels
        "epsilon_": model.epsilon_,          # Variance smoothing parameter
        "var_":model.var_
    }
    # Only include 'sigma_' if it exists
    if hasattr(model, "sigma_"):
        data_to_save["sigma_"] = model.sigma_

    np.savez(model_path, **data_to_save)
    

def load_sklearn_model(filename):
    model = GaussianNB()
    model_path = os.path.join(TRAINED_MODEL_DIR, filename)
    if not model_path.endswith('.npz'):
        model_path += '.npz'
    checkpoint = np.load(model_path)
    
    model.theta_ = checkpoint["theta_"]
    model.class_prior_ = checkpoint["class_prior_"]
    model.classes_ = checkpoint["classes_"]
    model.epsilon_ = checkpoint["epsilon_"]
    model.var_ = checkpoint["var_"]
    
    # Load 'sigma_' if it exists in the file
    if "sigma_" in checkpoint:
        model.sigma_ = checkpoint["sigma_"]

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", help="Force retrain the specified model", nargs="?", const="all")
    parser.add_argument("--generate_data", help="Generate/ get the training and testing data", nargs="?", const="all")
    args = parser.parse_args()
    trained_models = ['custom_naive_bayes', 'sklearn_naive_bayes']

    if args.retrain is not None:
        if args.retrain not in trained_models and args.retrain != "all":
            help_msg()
            exit()

    # Load and preprocess data
    print("Loading CIFAR-10 data...")
    train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca, train_loader, test_loader = [None] * 8

    if args.generate_data or not os.path.exists("cifar10_tensors.pt"):
        train_loader, test_loader = load_cifar10_data() #Tensors
        file_path = os.path.join(TRAINED_MODEL_DIR, "cifar10_data.npz")
        if not os.path.exists(file_path):
            previous_data_file = 'cifar10_data.npz'
            print("Generating CIFAR-10 data...")
            if os.path.exists(previous_data_file):
                print(f"Deleting previously saved data file: {previous_data_file}")
                os.remove(previous_data_file)

            if os.path.exists("cifar10_tensors.pt"):
                print("Deleting previously saved tensors file: cifar10_tensors.pt")
                os.remove("cifar10_tensors.pt")

            # Extract features from the training and test sets
            #* feautres = X, labels = y
            train_features, train_labels = extract_resnet18_features(train_loader) #Numpy arrays
            test_features, test_labels = extract_resnet18_features(test_loader) #Numpy arrays
            # Apply PCA for dimensionality reduction
            train_features_pca = apply_pca(train_features)
            test_features_pca = apply_pca(test_features)

            save_data(train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca, 'cifar10_data.npz')
        else:
            train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca = load_data(filename="cifar10_data.npz")
            #print(f"Printing train features: {train_features}")
            #print(f"Printing train labels: {train_labels}")
        #save_tensors(train_loader, test_loader, 'cifar10_tensors.pt')
    #else:
        #train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca = load_data(filename="cifar10_data.npz")
        #train_loader, test_loader = load_tensors('cifar10_tensors.pt')

    print("Evaluating Custom Gaussian Naive Bayes Model...")
    gnb_custom = GaussianNaiveBayes()
    if args.retrain == "custom_naive_bayes" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "custom_naive_bayes.npz"))):
        print("Retraining Custom Gaussian Naive Bayes Model...")
        gnb_custom.fit(train_features_pca, train_labels)
        gnb_custom.save("custom_naive_bayes.npz")  # Save model
    else:
        print("Loading saved Custom Gaussian Naive Bayes Model...")
        gnb_custom.load("custom_naive_bayes.npz")  # Load model


    gnb_custom_preds = gnb_custom.predict(test_features_pca)
    custom_metrics = evaluate_model(test_labels, gnb_custom_preds)
    print("Custom Gaussian Naive Bayes Evaluation Metrics:", custom_metrics)

    print("\nEvaluating Sklearn Gaussian Naive Bayes Model...")
    if args.retrain == "sklearn_naive_bayes" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "sklearn_naive_bayes.npz"))):
        print("Retraining Sklearn Gaussian Naive Bayes Model...")
        gnb_sklearn = GaussianNB()
        gnb_sklearn.fit(train_features_pca, train_labels)
        save_sklearn_model(gnb_sklearn, "sklearn_naive_bayes.npz")
    else:
        print("Loading saved Sklearn Gaussian Naive Bayes Model...")
        gnb_sklearn = load_sklearn_model("sklearn_naive_bayes.npz")

    sklearn_preds = gnb_sklearn.predict(test_features_pca)
    sklearn_metrics = evaluate_model(test_labels, sklearn_preds)
    print("Sklearn Gaussian Naive Bayes Evaluation Metrics:", sklearn_metrics)
