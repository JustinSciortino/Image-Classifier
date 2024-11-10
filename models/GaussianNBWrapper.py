from sklearn.naive_bayes import GaussianNB
import os
import numpy as np

class GaussianNBWrapper:
    def __init__(self):
        self.model = GaussianNB()
    
    # Train the model
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, filename):
        TRAINED_MODEL_DIR = "trained_models"
        os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(TRAINED_MODEL_DIR, filename)

        if not filename.endswith('.npz'):
            filename += '.npz'
            
        # Check if attributes exist before attempting to save them
        data_to_save = {
            "theta_": self.model.theta_,             # Class means
            "class_prior_": self.model.class_prior_, # Class priors
            "classes_": self.model.classes_,         # Class labels
            "epsilon_": self.model.epsilon_,          # Variance smoothing parameter
            "var_":self.model.var_
        }
        # Only include 'sigma_' if it exists
        if hasattr(self.model, "sigma_"):
            data_to_save["sigma_"] = self.model.sigma_

        np.savez(model_path, **data_to_save)
    
    def load_model(self, filename):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, filename)
        checkpoint = np.load(model_path)
        
        self.model.theta_ = checkpoint["theta_"]
        self.model.class_prior_ = checkpoint["class_prior_"]
        self.model.classes_ = checkpoint["classes_"]
        self.model.epsilon_ = checkpoint["epsilon_"]
        self.model.var_ = checkpoint["var_"]
        
        # Load 'sigma_' if it exists in the file
        if "sigma_" in checkpoint:
            self.model.sigma_ = checkpoint["sigma_"]