import numpy as np
import os

class CustomDecisionTree():
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y, 0)

    def build_tree(self, X, y, current_depth): #* X = n samples and n features where each row is a sample/data point (2D array), y = n samples/labels
        """
        ie of root/tree created
        root = {
            'feature': 0,
            'threshold': 5,
            'left': {
                'feature': 1,
                'threshold': 2,
                'left': 0,
                'right': 1
            },
            'right': 1
        }
        """
        #* If we've reached the max depth, create a leaf node
        if current_depth >= self.max_depth:
            return self._create_leaf_node(y)

        #* Create a leaf node if all the samples in y have the same label > stop splitting no longer needed
        if len(np.unique(y)) == 1:
            return self._create_leaf_node(y)

        best_feature, best_threshold = self.find_best_split(X, y)

        #* Split the data into left and right based on the best feature and threshold
        #* Boolean arrays for each sample ie [True, False, True, True, False, False] where each boolean represents a sample in the dataset
        left_idx = X[:, best_feature] < best_threshold 
        right_idx = X[:, best_feature] >= best_threshold

        #* Recursively build the tree
        #* X[left_idx] gets all samples where left_idx is True
        #* X[right_idx] gets all samples where right_idx is True
        left_subtree = self.build_tree(X[left_idx], y[left_idx], current_depth + 1) 
        right_subtree = self.build_tree(X[right_idx], y[right_idx], current_depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold, 'left': left_subtree, 'right': right_subtree}

    def find_best_split(self, X, y): #* Determine the best threshold and feature to split on
        best_gini = 1.0 #* Worst possible gini impurity
        best_feature = None
        best_threshold = None

        #* Iterate over each feature and threshold to calculate the gini impurity to find the bets gini impurity and split on the that feature and threshold
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self.calculate_gini(X, y, feature, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold #* Int and float values

    def calculate_gini(self, X, y, feature, threshold):
        """
        G = 1 - Summation from i = 1 to n (number of unique classes) of (p_i)^2 where p_i is the proportion of samples that belong to class i
        Gini impurity of a split = N_left (total number of samples in the left subset)/N (total number of samples before the split) * G_left (gini impurity for the left subset) + N_right/N * G_right
        """

        #*Split the data into left and right subsets based on the feature and threshold
        left_idx = X[:, feature] < threshold
        right_idx = X[:, feature] >= threshold

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 1.0 #* Return gini impurity of 1 if either subset is empty

        #* counts the number of occurences of each class label in the left and right subsets
        left_labels, left_counts = np.unique(y[left_idx], return_counts=True)
        right_labels, right_counts = np.unique(y[right_idx], return_counts=True)

        #* Total number of samples
        total_samples = len(y)

        #* Proportion of samples in the left and right subsets
        left_proportion = len(y[left_idx]) / total_samples
        right_proportion = len(y[right_idx]) / total_samples

        #* Calculate the gini impurity of the left and right subsets
        left_gini = 1 - sum([(count / len(y[left_idx])) ** 2 for count in left_counts])
        right_gini = 1 - sum([(count / len(y[right_idx])) ** 2 for count in right_counts])

        #* Return the weighted gini impurity of the split
        return left_proportion * left_gini + right_proportion * right_gini

    def _create_leaf_node(self, y):
        return np.bincount(y).argmax() #* Return the most common class label in y

    def predict(self, X):
        #* Iterate over each sample in X and make a prediction for each sample
        return np.array([self.predict_single(x, self.root) for x in X])

    def predict_single(self, x, node): #* Recursively traverse the tree to make a prediction for a single sample, x is a 1D array of feature values for a single sample/data point, node is the current node in the decision tree
        if isinstance(node, dict): #* Checks if it is an internal node, node is a split node if it is a dictionary and will contain the feature, threshold, left subtree and right subtree keys
            #* Compare the feature value in the sample to the threshold in the node
            if x[node['feature']] < node['threshold']:
                return self.predict_single(x, node['left'])
            else:
                return self.predict_single(x, node['right'])
        else:
            return node #* Node is not an internal node, return the class label/ predicted class label
    
    def save_model(self, filename):
        TRAINED_MODEL_DIR = "trained_models"
        os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
        
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        model_path = os.path.join(TRAINED_MODEL_DIR, filename)
        
        data_to_save = {
            "tree": self.root,
            "max_depth": self.max_depth
        }
        np.savez(model_path, **data_to_save)
    
    def load_model(self, filename):
        TRAINED_MODEL_DIR = "trained_models"
    
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        model_path = os.path.join(TRAINED_MODEL_DIR, filename)
        
        # Load the saved data
        loaded_data = np.load(model_path, allow_pickle=True)
        
        # Update the current instance with loaded parameters
        self.max_depth = loaded_data['max_depth'].item()
        self.root = loaded_data['tree'].item()
