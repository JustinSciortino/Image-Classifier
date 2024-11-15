from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from data import load_cifar10_data
from utils import extract_resnet18_features, apply_pca, help_msg, save_data, load_data, save_data_resnet
from models import GaussianNaiveBayes, GaussianNBWrapper, DecisionTreeWrapper, CustomDecisionTree
from sklearn.naive_bayes import GaussianNB
import os
import torch
import argparse
import numpy as np
import logging
logging.basicConfig(filename='ddataset.log', level=logging.INFO)

TRAINED_MODEL_DIR = "trained_models"
if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return (
        f"Accuracy: {accuracy}\n"
        f"Confusion Matrix:\n{conf_matrix}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1 Score: {f1}"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", help="Force retrain the specified model", nargs="?", const="all")
    parser.add_argument("--generate_data", help="Generate/ get the training and testing data", nargs="?", const="all")
    args = parser.parse_args()
    trained_models = ['custom_naive_bayes', 'sklearn_naive_bayes', 
                      'sklearn_decision_tree', 'custom_decision_tree', "custom_naive_bayes_resnet",
                      'sklearn_naive_bayes_resnet', 'sklearn_decision_tree_resnet', 'custom_decision_tree_resnet',
                      ]

    if args.retrain is not None:
        if args.retrain not in trained_models and args.retrain != "all":
            help_msg()
            exit()

    # Load and preprocess data
    print("Loading CIFAR-10 data...")
    file_path = os.path.join(TRAINED_MODEL_DIR, "cifar10_data.npz")
    train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca, train_loader, test_loader = [None] * 8
    

    if args.generate_data:
        train_loader, test_loader = load_cifar10_data() #Tensors

        if os.path.exists(file_path):
            previous_data_file = 'cifar10_data.npz'
            print(f"Deleting previously saved data file: {previous_data_file}")
            os.remove(previous_data_file)

            # Extract features from the training and test sets
        #* feautres = X, labels = y
        train_features, train_labels = extract_resnet18_features(train_loader) #Numpy arrays
        test_features, test_labels = extract_resnet18_features(test_loader) #Numpy arrays
        # Apply PCA for dimensionality reduction
        logging.info(f"train_features shape: {train_features.shape}")
        logging.info(f"test_features shape: {test_features.shape}")
        train_features_pca = apply_pca(train_features)
        test_features_pca = apply_pca(test_features)
        logging.info(f"train_features_pca shape: {train_features_pca.shape}")
        logging.info(f"test_features_pca shape: {test_features_pca.shape}")

        save_data(train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca, 'cifar10_data.npz')

    else:

        if not os.path.exists(file_path):
            # Extract features from the training and test sets
            #* feautres = X, labels = y
            train_features, train_labels = extract_resnet18_features(train_loader) #Numpy arrays
            test_features, test_labels = extract_resnet18_features(test_loader) #Numpy arrays
            # Apply PCA for dimensionality reduction
            logging.info(f"train_features shape: {train_features.shape}")
            logging.info(f"test_features shape: {test_features.shape}")
            train_features_pca = apply_pca(train_features)
            test_features_pca = apply_pca(test_features)
            logging.info(f"train_features_pca shape: {train_features_pca.shape}")
            logging.info(f"test_features_pca shape: {test_features_pca.shape}")

            save_data(train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca, 'cifar10_data.npz')
            
        else:
            train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca = load_data(filename="cifar10_data.npz")
            logging.info(f"train_features shape: {train_features.shape}")
            logging.info(f"test_features shape: {test_features.shape}")
            logging.info(f"train_features_pca shape: {train_features_pca.shape}")
            logging.info(f"test_features_pca shape: {test_features_pca.shape}")


    print("Evaluating Custom Gaussian Naive Bayes Model...")
    gnb_custom = GaussianNaiveBayes()
    if args.retrain == "custom_naive_bayes" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "custom_naive_bayes.npz"))):
        print("Retraining Custom Gaussian Naive Bayes Model...")
        gnb_custom.fit(train_features_pca, train_labels)
        gnb_custom.save("custom_naive_bayes.npz")  
    else:
        print("Loading saved Custom Gaussian Naive Bayes Model...")
        gnb_custom.load("custom_naive_bayes.npz")  # Load model

    gnb_custom_preds = gnb_custom.predict(test_features_pca)
    custom_metrics = evaluate_model(test_labels, gnb_custom_preds)
    print("Custom Gaussian Naive Bayes Evaluation Metrics:", custom_metrics)

    print("Evaluating Custom Gaussian Naive Bayes Model...Resnet")
    gnb_custom_resnet = GaussianNaiveBayes()
    if args.retrain == "custom_naive_bayes_resnet" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "custom_naive_bayes_resnet.npz"))):
        print("Retraining Custom Gaussian Naive Bayes Model...Resnet")
        gnb_custom_resnet.fit(train_features, train_labels)
        gnb_custom_resnet.save("custom_naive_bayes_resnet.npz")  
    else:
        print("Loading saved Custom Gaussian Naive Bayes Model...")
        gnb_custom_resnet.load("custom_naive_bayes_resnet.npz")  # Load model

    gnb_custom_preds_resnet = gnb_custom_resnet.predict(test_features)
    custom_metrics_resnet = evaluate_model(test_labels, gnb_custom_preds_resnet)
    print("Custom Gaussian Naive Bayes Evaluation Metrics Resnet:", custom_metrics_resnet)

    print("\nEvaluating Sklearn Gaussian Naive Bayes Model...")
    if args.retrain == "sklearn_naive_bayes" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "sklearn_naive_bayes.npz"))):
        print("Retraining Sklearn Gaussian Naive Bayes Model...")
        gnb_sklearn = GaussianNBWrapper()
        gnb_sklearn.fit(train_features_pca, train_labels)
        gnb_sklearn.save_model("sklearn_naive_bayes.npz")
    else:
        print("Loading saved Sklearn Gaussian Naive Bayes Model...")
        gnb_sklearn = GaussianNBWrapper()
        gnb_sklearn.load_model("sklearn_naive_bayes.npz")

    sklearn_preds = gnb_sklearn.predict(test_features_pca)
    sklearn_metrics = evaluate_model(test_labels, sklearn_preds)
    print("Sklearn Gaussian Naive Bayes Evaluation Metrics:", sklearn_metrics)

    print("\nEvaluating Sklearn Gaussian Naive Bayes Model...Resnet")
    if args.retrain == "sklearn_naive_bayes_resnet" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "sklearn_naive_bayes_resnet.npz"))):
        print("Retraining Sklearn Gaussian Naive Bayes Model...Resnet")
        gnb_sklearn_resnet = GaussianNBWrapper()
        gnb_sklearn_resnet.fit(train_features, train_labels)
        gnb_sklearn_resnet.save_model("sklearn_naive_bayes_resnet.npz")
    else:
        print("Loading saved Sklearn Gaussian Naive Bayes Model...")
        gnb_sklearn_resnet = GaussianNBWrapper()
        gnb_sklearn_resnet.load_model("sklearn_naive_bayes_resnet.npz")

    sklearn_preds_resnet = gnb_sklearn_resnet.predict(test_features)
    sklearn_metrics_resnet = evaluate_model(test_labels, sklearn_preds_resnet)
    print("Sklearn Gaussian Naive Bayes Evaluation Metrics Resnet:", sklearn_metrics_resnet)

    print("\nEvaluating Custom Decision Tree Model...")
    if args.retrain == "custom_decision_tree" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "custom_decision_tree.npz")):
        print("Retraining Custom Decision Tree Model...")
        dt_custom = CustomDecisionTree(50)
        dt_custom.fit(train_features_pca, train_labels)
        dt_custom.save_model("custom_decision_tree.npz")
    else:
        print("Loading saved Custom Decision Tree Model...")
        dt_custom = CustomDecisionTree()
        dt_custom.load_model("custom_decision_tree.npz")

    custom_dt_preds = dt_custom.predict(test_features_pca)
    custom_dt_metrics = evaluate_model(test_labels, custom_dt_preds)
    print("Custom Decision Tree Evaluation Metrics:", custom_dt_metrics)

    print("\nEvaluating Custom Decision Tree Model...Resnet")
    if args.retrain == "custom_decision_tree_resnet" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "custom_decision_tree_resnet.npz")):
        print("Retraining Custom Decision Tree Model...Resnet")
        dt_custom_resnet = CustomDecisionTree(50)
        dt_custom_resnet.fit(train_features, train_labels)
        dt_custom_resnet.save_model("custom_decision_tree_resnet.npz")
    else:
        print("Loading saved Custom Decision Tree Model...Resnet")
        dt_custom_resnet = CustomDecisionTree()
        dt_custom_resnet.load_model("custom_decision_tree_resnet.npz")

    custom_dt_preds_resnet = dt_custom_resnet.predict(test_features)
    custom_dt_metrics_resnet = evaluate_model(test_labels, custom_dt_preds_resnet)
    print("Custom Decision Tree Evaluation Metrics Resnet:", custom_dt_metrics_resnet)

    print("\nEvaluating Sklearn Decision Tree Model...")
    if args.retrain == "sklearn_decision_tree" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "sklearn_decision_tree.npz")):
        print("Retraining Sklearn Decision Tree Model...")
        dt_sklearn = DecisionTreeWrapper(50)
        dt_sklearn.fit(train_features_pca, train_labels)
        dt_sklearn.save_model("sklearn_decision_tree.npz")
    else:
        print("Loading saved Sklearn Decision Tree Model...")
        dt_sklearn = DecisionTreeWrapper()
        dt_sklearn.load_model("sklearn_decision_tree.npz")

    sklearn_dt_preds = dt_sklearn.predict(test_features_pca)
    sklearn_dt_metrics = evaluate_model(test_labels, sklearn_dt_preds)
    print("Sklearn Decision Tree Evaluation Metrics:", sklearn_dt_metrics)

    print("\nEvaluating Sklearn Decision Tree Model...Resnet")
    if args.retrain == "sklearn_decision_tree_resnet" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "sklearn_decision_tree_resnet.npz")):
        print("Retraining Sklearn Decision Tree Model...Resnet")
        dt_sklearn_resnet = DecisionTreeWrapper(50)
        dt_sklearn_resnet.fit(train_features, train_labels)
        dt_sklearn_resnet.save_model("sklearn_decision_tree_resnet.npz")
    else:
        print("Loading saved Sklearn Decision Tree Model...Resnet")
        dt_sklearn_resnet = DecisionTreeWrapper()
        dt_sklearn_resnet.load_model("sklearn_decision_tree_resnet.npz")

    sklearn_dt_preds_resnet = dt_sklearn_resnet.predict(test_features)
    sklearn_dt_metrics_resnet = evaluate_model(test_labels, sklearn_dt_preds_resnet)
    print("Sklearn Decision Tree Evaluation Metrics Resnet:", sklearn_dt_metrics_resnet)
