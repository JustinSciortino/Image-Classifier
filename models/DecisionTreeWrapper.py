from sklearn.tree import DecisionTreeClassifier
import os
import numpy as np

#* Just a wrapper class for the DecisionTreeClassifier Scikit-learn model
class DecisionTreeWrapper:
    def __init__(self, max_depth=None) -> None:
        self.model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, filename):
        TRAINED_MODEL_DIR = "trained_models"
        os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)  
        
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        model_path = os.path.join(TRAINED_MODEL_DIR, filename)
        
        #* Save all necessary attributes of the Scikit-learn DecisionTree
        data_to_save = {
            "tree_": self.model.tree_,
            "n_features_in_": self.model.n_features_in_,
            "n_classes_": self.model.n_classes_,
            "n_outputs_": self.model.n_outputs_,
            "classes_": self.model.classes_,
            "max_depth": self.model.max_depth,
            "criterion": self.model.criterion
        }
        
        np.savez(model_path, **data_to_save)
    
    #* Load the model from the file and set the attributes
    def load_model(self, filename):
        TRAINED_MODEL_DIR = "trained_models"

        if not filename.endswith('.npz'):
            filename += '.npz'
        
        model_path = os.path.join(TRAINED_MODEL_DIR, filename)
        
        #* Load the model from the file and set the attributes
        checkpoint = np.load(model_path, allow_pickle=True)
        
        #* Initialize new DecisionTreeClassifier and set attributes
        self.model = DecisionTreeClassifier(criterion='gini')
        self.model.tree_ = checkpoint['tree_'].item()
        self.model.n_features_in_ = checkpoint['n_features_in_'].item()
        self.model.n_classes_ = checkpoint['n_classes_'].item()
        self.model.n_outputs_ = checkpoint['n_outputs_'].item()
        self.model.classes_ = checkpoint['classes_']
        self.model.max_depth = checkpoint['max_depth'].item()

    #* Get the depth of the trained decision tree
    def get_tree_depth(self):
        return self.model.get_depth()