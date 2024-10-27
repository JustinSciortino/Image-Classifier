from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from data.dataset import load_cifar10_data
from data.data_utils import extract_resnet18_features
from data.data_utils import apply_pca
from models.custom_naive_bayes import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
import os
import torch
import argparse

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


    
def print_help_msg_retrain_models():
    msg = """
    Usage: python main.py [options]
    Options:
        --retain <model_name> Force retrain the specified model
        --retrain all Force retrain all models
        --retrain Retrain all models
        """
    print(msg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", help="Force retrain the specified model", nargs="?", const="all")
    args = parser.parse_args()
    trained_models = ['custom_naive_bayes', 'sklearn_naive_bayes']

    if args.retrain is not None:
        if args.retrain not in trained_models and args.retrain != "all":
            print_help_msg_retrain_models()
            exit()

    # Load and preprocess data
    print("Loading CIFAR-10 data...")
    train_loader, test_loader = load_cifar10_data()

    # Extract features from the training and test sets
    train_features, train_labels = extract_resnet18_features(train_loader)
    test_features, test_labels = extract_resnet18_features(test_loader)
    # Apply PCA for dimensionality reduction
    train_features_pca = apply_pca(train_features)
    test_features_pca = apply_pca(test_features)

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

    """gnb_sklearn = load_model("sklearn_naive_bayes.pth", GaussianNB)
    if args.retrain == "sklearn_naive_bayes" or (args.retrain == "all" and gnb_sklearn is None):
        print("Retraining Scikit-learn Gaussian Naive Bayes Model...")
        gnb_sklearn = GaussianNB()
        gnb_sklearn.fit(train_features_pca, train_labels)
        save_model(gnb_sklearn, "sklearn_naive_bayes.pth")
    elif gnb_sklearn is None:
        print("No saved model found. Training Scikit-learn Gaussian Naive Bayes Model...")
        gnb_sklearn = GaussianNB()
        gnb_sklearn.fit(train_features_pca, train_labels)
        save_model(gnb_sklearn, "sklearn_naive_bayes.pth")
    
    gnb_sklearn_preds = gnb_sklearn.predict(test_features_pca)
    sklearn_metrics = evaluate_model(test_labels, gnb_sklearn_preds)
    print("Scikit-learn Gaussian Naive Bayes Evaluation Metrics:", sklearn_metrics)"""


"""
def save_model(model, model_name):
    torch.save(model.state_dict(), os.path.join(TRAINED_MODEL_DIR, model_name))

def load_model(model_name, model_class):
    model_path = os.path.join(TRAINED_MODEL_DIR, model_name)
    model = model_class()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        return None
"""