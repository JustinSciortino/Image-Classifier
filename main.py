from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from data import load_cifar10_data, load_cifar10_data_CNN
from utils import extract_resnet18_features, apply_pca, help_msg, save_data, load_data, save_tensors, load_tensors
from models import GaussianNaiveBayes, GaussianNBWrapper, DecisionTreeWrapper, CustomDecisionTree, MLP, VGG11
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
    return (
        f"\nAccuracy: {accuracy}\n"
        f"Confusion Matrix:\n{conf_matrix}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1 Score: {f1}\n"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", help="Force retrain the specified model", nargs="?", const="all")
    parser.add_argument("--generate_data", help="Generate/ get the training and testing data", nargs="?", const="all")
    args = parser.parse_args()
    trained_models = ["custom_naive_bayes_pca", "custom_naive_bayes", "sklearn_naive_bayes_pca",
                       "sklearn_naive_bayes", "custom_decision_tree_pca","custom_decision_tree_d50",
                         "custom_decision_tree_d15", "custom_decision_tree_d10", "custom_decision_tree_d5",
                        "sklearn_decision_tree_pca", "sklearn_decision_tree_d50", "sklearn_decision_tree_d10", "mlp", "cnn" ]

    if args.retrain is not None:
        if args.retrain not in trained_models and args.retrain != "all":
            help_msg()
            exit()

    # Load and preprocess data
    file_path = os.path.join(TRAINED_MODEL_DIR, "cifar10_data.npz")
    train_loader, test_loader = [None] * 2
    train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca, train_loader, test_loader = [None] * 8
    train_features_tensors, train_labels_tensors, test_features_tensors, test_labels_tensors = [None] * 4
    

    if args.generate_data:
        print("Generating CIFAR-10 data...")
        train_loader, test_loader = load_cifar10_data() #Tensors, raw images

        if os.path.exists(file_path):
            previous_data_file = os.path.join(TRAINED_MODEL_DIR, "cifar10_data.npz")
            print(f"Deleting previously saved data file: {previous_data_file}")
            os.remove(previous_data_file)
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "cifar10_tensors.pt")):
            previous_data_file_tensors = os.path.join(TRAINED_MODEL_DIR, "cifar10_tensors.pt")
            print(f"Deleting previously saved data file: {previous_data_file_tensors}")
            os.remove(previous_data_file_tensors)

            # Extract features from the training and test sets
        #* feautres = X, labels = y
        print("Extracting features from the training and test sets...")
        train_features, train_labels, train_features_tensors, train_labels_tensors = extract_resnet18_features(train_loader) #Numpy arrays and tensors now
        test_features, test_labels, test_features_tensors, test_labels_tensors = extract_resnet18_features(test_loader) #Numpy arrays and tensors now
        train_features_pca = apply_pca(train_features)
        test_features_pca = apply_pca(test_features)

        save_data(train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca)
        save_tensors(train_features_tensors, train_labels_tensors, test_features_tensors, test_labels_tensors)

    else:
        print("Loading CIFAR-10 data...")
        if not os.path.exists(file_path):
            print("Generating CIFAR-10 npz file...")
            train_loader, test_loader = load_cifar10_data() #Tensors
            # Extract features from the training and test sets
            #* feautres = X, labels = y
            train_features, train_labels, train_features_tensors, train_labels_tensors = extract_resnet18_features(train_loader) #Numpy arrays
            test_features, test_labels, test_features_tensors, test_labels_tensors = extract_resnet18_features(test_loader) #Numpy arrays
            # Apply PCA for dimensionality reduction
            train_features_pca = apply_pca(train_features)
            test_features_pca = apply_pca(test_features)

            save_data(train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca)
        
        else:
            print("Loading CIFAR-10 npz file...")
            train_loader, test_loader = load_cifar10_data()
            train_features, train_labels, test_features, test_labels, train_features_pca, test_features_pca = load_data(filename="cifar10_data.npz")
        
        if not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "cifar10_tensors.pt")):
            print("Generating CIFAR-10 tensors file...")
            train_loader, test_loader = load_cifar10_data() #Tensors
            _, _, train_features_tensors, train_labels_tensors = extract_resnet18_features(train_loader) 
            _, _, test_features_tensors, test_labels_tensors = extract_resnet18_features(test_loader) 
            save_tensors(train_features_tensors, train_labels_tensors, test_features_tensors, test_labels_tensors)
            
        else:
            print("Loading CIFAR-10 tensors file...")
            train_loader, test_loader = load_cifar10_data()
            train_features_tensors, train_labels_tensors, test_features_tensors, test_labels_tensors = load_tensors()


    print("\nEvaluating Custom Naive Bayes Model Implementation with PCA reduction (50 features)")
    gnb_custom_pca = GaussianNaiveBayes()

    if args.retrain == "custom_naive_bayes_pca" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Naive_Bayes_PCA.npz"))):

        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Naive_Bayes_PCA.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Custom_Naive_Bayes_PCA.npz"))
            print("Removed previously saved model: trained_models/Custom_Naive_Bayes_PCA.npz")

        print("Training Custom Gaussian Naive Bayes Model with PCA reduction (50 features)")
        gnb_custom_pca.fit(train_features_pca, train_labels)
        gnb_custom_pca.save("Custom_Naive_Bayes_PCA.npz")  

    else:
        print("Loading saved Custom Gaussian Naive Bayes Model with PCA reduction (50 features)")
        gnb_custom_pca.load("Custom_Naive_Bayes_PCA.npz")  

    gnb_custom_predictions_pca = gnb_custom_pca.predict(test_features_pca)
    gnb_custom_pca_metrics = evaluate_model(test_labels, gnb_custom_predictions_pca)
    print("Custom Gaussian Naive Bayes Evaluation Metrics with PCA reduction (50 features):", gnb_custom_pca_metrics)

    print("Evaluating Custom Gaussian Naive Bayes Model Implementation without PCA reduction (512 features)")
    gnb_custom = GaussianNaiveBayes()

    if args.retrain == "custom_naive_bayes" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Naive_Bayes.npz"))):

        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Naive_Bayes.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Custom_Naive_Bayes.npz"))
            print("Removed previously saved model: trained_models/Custom_Naive_Bayes.npz")

        print("Training Custom Gaussian Naive Bayes Model (512 features, No PCA Reduction)")
        gnb_custom.fit(train_features, train_labels)
        gnb_custom.save("Custom_Naive_Bayes.npz")  

    else:
        print("Loading saved Custom Gaussian Naive Bayes Model without PCA reduction (512 features)")
        gnb_custom.load("Custom_Naive_Bayes.npz")  # Load model

    gnb_custom_predictions = gnb_custom.predict(test_features)
    gnb_custom_metrics = evaluate_model(test_labels, gnb_custom_predictions)
    print("Custom Gaussian Naive Bayes Evaluation Metrics without PCA reduction (512 features):", gnb_custom_metrics)

    print("\nEvaluating Sklearn Gaussian Naive Bayes Model with PCA reduction (50 features)")
    if args.retrain == "sklearn_naive_bayes_pca" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Naive_Bayes_PCA.npz"))):

        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Naive_Bayes_PCA.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Naive_Bayes_PCA.npz"))
            print("Removed previously saved model: trained_models/Sklearn_Naive_Bayes_PCA.npz")

        print("Training Sklearn Gaussian Naive Bayes Model with PCA reduction (50 features)")
        gnb_sklearn_pca = GaussianNBWrapper()
        gnb_sklearn_pca.fit(train_features_pca, train_labels)
        gnb_sklearn_pca.save_model("Sklearn_Naive_Bayes_PCA.npz")

    else:
        print("Loading saved Sklearn Gaussian Naive Bayes Model with PCA reduction (50 features)")
        gnb_sklearn_pca = GaussianNBWrapper()
        gnb_sklearn_pca.load_model("Sklearn_Naive_Bayes_PCA.npz")

    gnb_sklearn_pca_predictions = gnb_sklearn_pca.predict(test_features_pca)
    gnb_sklearn_pca_metrics = evaluate_model(test_labels, gnb_sklearn_pca_predictions)
    print("Sklearn Gaussian Naive Bayes Evaluation Metrics with PCA reduction (50 features):", gnb_sklearn_pca_metrics)

    print("\nEvaluating Sklearn Gaussian Naive Bayes Model without PCA reduction (512 features)")
    if args.retrain == "sklearn_naive_bayes" or (args.retrain == "all" and not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Naive_Bayes.npz"))):

        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Naive_Bayes.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Naive_Bayes.npz"))
            print("Removed previously saved model: trained_models/Sklearn_Naive_Bayes.npz")

        print("Training Sklearn Gaussian Naive Bayes Model without PCA reduction (512 features)")
        gnb_sklearn = GaussianNBWrapper()
        gnb_sklearn.fit(train_features, train_labels)
        gnb_sklearn.save_model("Sklearn_Naive_Bayes.npz")

    else:
        print("Loading saved Sklearn Gaussian Naive Bayes Model...")
        gnb_sklearn = GaussianNBWrapper()
        gnb_sklearn.load_model("Sklearn_Naive_Bayes.npz")

    gnb_sklearn_predictions = gnb_sklearn.predict(test_features)
    gnb_sklearn_metrics = evaluate_model(test_labels, gnb_sklearn_predictions)
    print("Sklearn Gaussian Naive Bayes Evaluation Metrics without PCA reduction (512 features):", gnb_sklearn_metrics)

    print('\n-------------------------------------------------------------\n')

    print("\nEvaluating Custom Decision Tree Model with PCA reduction (50 features) and a maximum depth of 50")
    if args.retrain == "custom_decision_tree_pca" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_PCA.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_PCA.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_PCA.npz"))
            print("Removed previously saved model: trained_models/Custom_Decision_Tree_PCA.npz")

        print("Training Custom Decision Tree Model with PCA reduction (50 features) and a maximum depth of 50")
        dt_custom_pca = CustomDecisionTree(50)
        dt_custom_pca.fit(train_features_pca, train_labels)
        dt_custom_pca.save_model("Custom_Decision_Tree_PCA.npz")

    else:
        print("Loading saved Custom Decision Tree Model with PCA reduction (50 features) and a maximum depth of 50")
        dt_custom_pca = CustomDecisionTree()
        dt_custom_pca.load_model("Custom_Decision_Tree_PCA.npz")

    dt_custom_pca_predictions = dt_custom_pca.predict(test_features_pca)
    dt_custom_pca_metrics = evaluate_model(test_labels, dt_custom_pca_predictions)
    print(f"Actual tree Depth: {dt_custom_pca.get_tree_depth()}")
    print("Custom Decision Tree Evaluation Metrics with PCA reduction and a maximum depth of 50:", dt_custom_pca_metrics)    

    print("\nEvaluating Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 50")
    if args.retrain == "custom_decision_tree_d50" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D50.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D50.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D50.npz"))
            print("Removed previously saved model: trained_models/Custom_Decision_Tree_D50.npz")

        print("Training Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 50")
        dt_custom_d50 = CustomDecisionTree(50)
        dt_custom_d50.fit(train_features, train_labels)
        dt_custom_d50.save_model("Custom_Decision_Tree_D50.npz")

    else:
        print("Loading saved Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 50")
        dt_custom_d50 = CustomDecisionTree()
        dt_custom_d50.load_model("Custom_Decision_Tree_D50.npz")

    dt_custom_d50_predictions = dt_custom_d50.predict(test_features)
    dt_custom_d50_metrics = evaluate_model(test_labels, dt_custom_d50_predictions)
    print(f"Actual tree Depth: {dt_custom_d50.get_tree_depth()}")
    print("Custom Decision Tree Evaluation Metrics without PCA reduction (512 features) and a maximum depth of 50:", dt_custom_d50_metrics)

    print("\nEvaluating Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 15")
    if args.retrain == "custom_decision_tree_d15" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D15.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D15.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D15.npz"))
            print("Removed previously saved model: trained_models/Custom_Decision_Tree_D15.npz")

        print("Training Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 15")
        dt_custom_d15 = CustomDecisionTree(15)
        dt_custom_d15.fit(train_features, train_labels)
        dt_custom_d15.save_model("Custom_Decision_Tree_D15.npz")

    else:
        print("Loading saved Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 15")
        dt_custom_d15 = CustomDecisionTree()
        dt_custom_d15.load_model("Custom_Decision_Tree_D15.npz")

    dt_custom_d15_predictions = dt_custom_d15.predict(test_features)
    dt_custom_d15_metrics = evaluate_model(test_labels, dt_custom_d15_predictions)
    print(f"Actual tree Depth: {dt_custom_d15.get_tree_depth()}")
    print("Custom Decision Tree Evaluation Metrics without PCA reduction (512 features) and a maximum depth of 15:", dt_custom_d15_metrics)

    print("\nEvaluating Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 10")
    if args.retrain == "custom_decision_tree_d10" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D10.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D10.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D10.npz"))
            print("Removed previously saved model: trained_models/Custom_Decision_Tree_D10.npz")

        print("Training Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 10")
        dt_custom_d10 = CustomDecisionTree(10)
        dt_custom_d10.fit(train_features, train_labels)
        dt_custom_d10.save_model("Custom_Decision_Tree_D10.npz")

    else:
        print("Loading saved Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 10")
        dt_custom_d10 = CustomDecisionTree()
        dt_custom_d10.load_model("Custom_Decision_Tree_D10.npz")

    dt_custom_d10_predictions = dt_custom_d10.predict(test_features)
    dt_custom_d10_metrics = evaluate_model(test_labels, dt_custom_d10_predictions)
    print(f"Actual tree Depth: {dt_custom_d10.get_tree_depth()}")
    print("Custom Decision Tree Evaluation Metrics without PCA reduction (512 features) and a maximum depth of 10:", dt_custom_d10_metrics)

    print("\nEvaluating Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 5")
    if args.retrain == "custom_decision_tree_d5" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D5.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D5.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Custom_Decision_Tree_D5.npz"))
            print("Removed previously saved model: trained_models/Custom_Decision_Tree_D5.npz")

        print("Training Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 5")
        dt_custom_d5 = CustomDecisionTree(5)
        dt_custom_d5.fit(train_features, train_labels)
        dt_custom_d5.save_model("Custom_Decision_Tree_D5.npz")

    else:
        print("Loading saved Custom Decision Tree Model without PCA reduction (512 features) and a maximum depth of 5")
        dt_custom_d5 = CustomDecisionTree()
        dt_custom_d5.load_model("Custom_Decision_Tree_D5.npz")

    dt_custom_d5_predictions = dt_custom_d5.predict(test_features)
    dt_custom_d5_metrics = evaluate_model(test_labels, dt_custom_d5_predictions)
    print(f"Actual tree Depth: {dt_custom_d5.get_tree_depth()}")
    print("Custom Decision Tree Evaluation Metrics without PCA reduction (512 features) and a maximum depth of 5:", dt_custom_d5_metrics)

    print("\nEvaluating Sklearn Decision Tree Model with PCA reduction (50 features) and a maximum depth of 50")
    if args.retrain == "sklearn_decision_tree_pca" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_PCA.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_PCA.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_PCA.npz"))
            print("Removed previously saved model: trained_models/Sklearn_Decision_Tree_PCA.npz")

        print("Training Sklearn Decision Tree Model with PCA reduction (50 features) and a maximum depth of 50")
        dt_sklearn_pca = DecisionTreeWrapper(50)
        dt_sklearn_pca.fit(train_features_pca, train_labels)
        dt_sklearn_pca.save_model("Sklearn_Decision_Tree_PCA.npz")
        

    else:
        print("Loading saved Sklearn Decision Tree Model with PCA reduction (50 features) and a maximum depth of 50")
        dt_sklearn_pca = DecisionTreeWrapper()
        dt_sklearn_pca.load_model("Sklearn_Decision_Tree_PCA.npz")

    dt_sklearn_pca_predictions = dt_sklearn_pca.predict(test_features_pca)
    dt_sklearn_pca_metrics = evaluate_model(test_labels, dt_sklearn_pca_predictions)
    print(f"Actual tree Depth: {dt_sklearn_pca.get_tree_depth()}")
    print("Sklearn Decision Tree Evaluation Metrics with PCA reduction (50 features) and a maximum depth of 50:", dt_sklearn_pca_metrics)
    

    print("\nEvaluating Sklearn Decision Tree Model without PCA reduction (512 features) and a maximum depth of 50")
    if args.retrain == "sklearn_decision_tree_d50" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_D50.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_D50.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_D50.npz"))
            print("Removed previously saved model: trained_models/Sklearn_Decision_Tree_D50.npz")

        print("Training Sklearn Decision Tree Model without PCA reduction (512 features) and a maximum depth of 50")
        dt_sklearn = DecisionTreeWrapper(50)
        dt_sklearn.fit(train_features, train_labels)
        dt_sklearn.save_model("Sklearn_Decision_Tree_D50.npz")

    else:
        print("Loading saved Sklearn Decision Tree Model without PCA reduction (512 features) and a maximum depth of 50")
        dt_sklearn = DecisionTreeWrapper()
        dt_sklearn.load_model("Sklearn_Decision_Tree_D50.npz")

    dt_sklearn_predictions = dt_sklearn.predict(test_features)
    dt_sklearn_metrics = evaluate_model(test_labels, dt_sklearn_predictions)
    print(f"Actual tree Depth: {dt_sklearn.get_tree_depth()}")
    print("Sklearn Decision Tree Evaluation Metrics without PCA reduction (512 features) and a maximum depth of 50:", dt_sklearn_metrics)


    print("\nEvaluating Sklearn Decision Tree Model without PCA reduction (512 features) and a maximum depth of 10")
    if args.retrain == "sklearn_decision_tree_d10" or args.retrain == "all" or not os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_D10.npz")):
        
        if os.path.exists(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_D10.npz")):
            os.remove(os.path.join(TRAINED_MODEL_DIR, "Sklearn_Decision_Tree_D10.npz"))
            print("Removed previously saved model: trained_models/Sklearn_Decision_Tree_D10.npz")

        print("Training Sklearn Decision Tree Model without PCA reduction (512 features) and a maximum depth of 10")
        dt_sklearn_d10 = DecisionTreeWrapper(10)
        dt_sklearn_d10.fit(train_features, train_labels)
        dt_sklearn_d10.save_model("Sklearn_Decision_Tree_D10.npz")

    else:
        print("Loading saved Sklearn Decision Tree Model without PCA reduction (512 features) and a maximum depth of 10")
        dt_sklearn_d10 = DecisionTreeWrapper()
        dt_sklearn_d10.load_model("Sklearn_Decision_Tree_D10.npz")

    dt_sklearn_d10_predictions = dt_sklearn_d10.predict(test_features)
    dt_sklearn_d10_metrics = evaluate_model(test_labels, dt_sklearn_d10_predictions)
    print(f"Actual tree Depth: {dt_sklearn_d10.get_tree_depth()}")
    print("Sklearn Decision Tree Evaluation Metrics without PCA reduction (512 features) and a maximum depth of 10:", dt_sklearn_d10_metrics)

    print('\n-------------------------------------------------------------\n')

    cnn_mlp_msg ="""
    Note that the MLP and CNN models were trained on Google Colab in order to utilize GPUs using jupyter notebooks
    that are located in models/notebooks. While the main.py is configured to train the CNN or MLP models, it should not be used unless you will be training them on a GPU. 
    To train the models in Google colab, open the notebooks in Google Colab and create a folder name 'trained_models' in the root directory of the project, and upload 
    the cifar10_data.npz file and cifar10_tensors.pt file to the 'trained_models' folder. The trained MLP and CNN models will be saved in the 'trained_models' folder.
    Do not forget to change the runtime to GPU in the notebook settings (Runtime > Change runtime type > Select GPU).

    In addition, the MLP models do not train with PCA reduction as it was redundant and the results without PCA reduction were
    better to evaluated the model. 

    """
    print(cnn_mlp_msg)
    print("\nEvaluating MLP Model without PCA reduction (512 features)")
    batch_size = 128
    learning_rate = 0.001
    epoch = 30
    hidden_sizes = [256, 512, 1024]
    layers = [3,2,4,6]

    for layer in layers:
        for hidden_size in hidden_sizes:

            print(f"\nEvaluating MLP Model with batch_size={batch_size}, learning_rate={learning_rate}, epochs={epoch}, hidden_size={hidden_size}, number of layers={layer}, 512 features")
            file_path_mlp = os.path.join(f"MLP_HiddenSize{hidden_size}_{layer}Layers.pth")

            if os.path.exists(os.path.join(TRAINED_MODEL_DIR, file_path_mlp)) and (args.retrain == "mlp" or args.retrain == "all"):
                print("Retraining MLP Model")
                mlp = MLP(
                    train_tensors=train_features_tensors,
                    train_labels=train_labels_tensors,
                    test_tensors=test_features_tensors,
                    test_labels=test_labels_tensors,
                    num_features=512,
                    hidden_size=hidden_size,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epoch,
                    layers=layer
                )
                mlp.train()
                mlp.get_memory_usage()

                file_mlp_name, file_mlp_extension = os.path.splitext(file_path_mlp)
                file_path_mlp = file_mlp_name + "_retrained" + file_mlp_extension
                if os.path.exists(os.path.join(TRAINED_MODEL_DIR, file_path_mlp)):
                    previous_mlp_retrained_file = os.path.join(TRAINED_MODEL_DIR, file_path_mlp)
                    print(f"Removing previous retrained model: {previous_mlp_retrained_file}")
                    os.remove(previous_mlp_retrained_file)
                mlp.save_model(filename=file_path_mlp)

                _, accuracy, conf_matrix, precision, recall, f1, evaluate_time = mlp.evaluate()
                GPU_mememory_allocated, GPU_max_memory_allocated, CPU_memory_usage, training_time = mlp.get_mememory_training_data()
                print(f"Accuracy: {accuracy}")
                print(f"Confusion Matrix:\n{conf_matrix}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print(f"Evaluation Time: {evaluate_time} seconds")
                print(f"Training Time: {training_time} seconds")
                print(f"GPU Memory Allocated: {GPU_mememory_allocated} MB")
                print(f"GPU Max Memory Allocated: {GPU_max_memory_allocated} MB")
                print(f"CPU Memory Usage: {CPU_memory_usage} MB")
                if layer == 3:
                    print(f"Three-layer MLP model")
                elif layer == 2:
                    print(f"Two-layer MLP model")
                elif layer == 4:
                    print(f"Four-layer MLP model")
                else:
                    print(f"Six-layer MLP model")
            
            elif os.path.exists(os.path.join(TRAINED_MODEL_DIR, file_path_mlp)):
                print("Loading saved MLP Model")
                mlp = MLP(
                    train_tensors=train_features_tensors,
                    train_labels=train_labels_tensors,
                    test_tensors=test_features_tensors,
                    test_labels=test_labels_tensors,
                    num_features=512,
                    hidden_size=hidden_size,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epoch,
                    layers=layer
                )

                file_mlp_name, file_mlp_extension = os.path.splitext(file_path_mlp)
                file_path_mlp_retrained = file_mlp_name + "_retrained" + file_mlp_extension

                if os.path.exists(os.path.join(TRAINED_MODEL_DIR, file_path_mlp_retrained)):
                    file_path_mlp = file_path_mlp_retrained
                mlp.load_model(filename=file_path_mlp)

                _, accuracy, conf_matrix, precision, recall, f1, evaluate_time = mlp.evaluate()
                GPU_mememory_allocated, GPU_max_memory_allocated, CPU_memory_usage, training_time = mlp.get_mememory_training_data()
                print(f"Accuracy: {accuracy}")
                print(f"Confusion Matrix:\n{conf_matrix}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print(f"Evaluation Time: {evaluate_time} seconds")
                print(f"Training Time: {training_time} seconds")
                print(f"GPU Memory Allocated: {GPU_mememory_allocated} MB")
                print(f"GPU Max Memory Allocated: {GPU_max_memory_allocated} MB")
                print(f"CPU Memory Usage: {CPU_memory_usage} MB")
                if layer == 3:
                    print(f"Three-layer MLP model")
                elif layer == 2:
                    print(f"Two-layer MLP model")
                elif layer == 4:
                    print(f"Four-layer MLP model")
                else:
                    print(f"Six-layer MLP model")
                
            else:
                print(f"trained_models/{file_path_mlp} does not exist. Please use Google Colab to train the model or use the following command to retrain the MLP models: python main.py --retrain mlp.")

    
    print("\nEvaluating VGG11 Model")
    cnn_learning_rate = 0.01
    cnn_epochs = 20
    cnn_kernel_size = [3,5,7]
    cnn_layers = [8,10,6]
    train_loader_, test_loader_ = load_cifar10_data_CNN()

    for kernel_size in cnn_kernel_size:
        for layer in cnn_layers:
            print(f"\nEvaluating VGG11 Model with learning_rate={cnn_learning_rate}, epochs={cnn_epochs}, kernel_size={kernel_size}, number of conv layers={layer}")
            file_path_vgg11 = os.path.join(f"VGG11_Epoch{cnn_epochs}_Kernel{kernel_size}_{layer}ConvLayers.pth")

            if os.path.exists(os.path.join(TRAINED_MODEL_DIR, file_path_vgg11)) and (args.retrain == "cnn" or args.retrain == "all"):
                print("Retraining VGG11 Model")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = VGG11(layers=layer, num_classes=10, kernel_size=kernel_size)
                model.to(device)
                model.train_model(train_loader, num_epochs=cnn_epochs, learning_rate=cnn_learning_rate, momentum=0.9, device=device)
                model.get_memory_usage()

                file_cnn_name, file_cnn_extension = os.path.splitext(file_path_vgg11)
                file_path_vgg11 = file_cnn_name + "_retrained" + file_cnn_extension
                if os.path.exists(os.path.join(TRAINED_MODEL_DIR, file_path_vgg11)):
                    previous_vgg11_retrained_file = os.path.join(TRAINED_MODEL_DIR, file_path_vgg11)
                    print(f"Removing previous retrained model: {previous_vgg11_retrained_file}")
                    os.remove(previous_vgg11_retrained_file)
                model.save_model(file_path_vgg11)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                accuracy, conf_matrix, precision, recall, f1, eval_time = model.evaluate_model(test_loader_, device)
                GPU_mememory_allocated, GPU_max_memory_allocated, CPU_memory_usage, training_time = model.get_mememory_training_data()
                print(f"Accuracy: {accuracy}%")
                print(f"Confusion Matrix:\n{conf_matrix}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print(f"Evaluation Time: {eval_time} seconds")
                print(f"Training Time: {training_time} seconds")
                print(f"GPU Memory Allocated: {GPU_mememory_allocated} MB")
                print(f"GPU Max Memory Allocated: {GPU_max_memory_allocated} MB")
                print(f"CPU Memory Usage: {CPU_memory_usage} MB")

            elif os.path.exists(os.path.join(TRAINED_MODEL_DIR, file_path_vgg11)):
                print("Loading saved VGG11 Model")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = VGG11(layers=layer, num_classes=10, kernel_size=kernel_size)
                model.to(device)

                file_cnn_name, file_cnn_extension = os.path.splitext(file_path_vgg11)
                file_cnn_path_retrained = file_cnn_name + "_retrained" + file_cnn_extension
                if os.path.exists(os.path.join(TRAINED_MODEL_DIR,  file_cnn_path_retrained)):
                    file_path_vgg11 = file_cnn_path_retrained
                model.load_model(path=file_path_vgg11, device=device)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                accuracy, conf_matrix, precision, recall, f1, eval_time = model.evaluate_model(test_loader_, device)
                GPU_mememory_allocated, GPU_max_memory_allocated, CPU_memory_usage, training_time = model.get_mememory_training_data()
                print(f"Accuracy: {accuracy}%")
                print(f"Confusion Matrix:\n{conf_matrix}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print(f"Evaluation Time: {eval_time} seconds")
                print(f"Training Time: {training_time} seconds")
                print(f"GPU Memory Allocated: {GPU_mememory_allocated} MB")
                print(f"GPU Max Memory Allocated: {GPU_max_memory_allocated} MB")
                print(f"CPU Memory Usage: {CPU_memory_usage} MB")

            else:
                print(f"trained_models/{file_path_vgg11} does not exist. Please use Google Colab to train the model or use the following command to retrain the CNN models: python main.py --retrain cnn.")
    

