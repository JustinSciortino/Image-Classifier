from sklearn.naive_bayes import GaussianNB
import os
import numpy as np

#* Just a wrapper class for the GaussianNB Scikit-learn model
class GaussianNBWrapper:
    def __init__(self):
        self.model = GaussianNB()
    
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
            
        #* Save all necessary attributes of the GaussianNB model
        data_to_save = {
            "theta_": self.model.theta_,             
            "class_prior_": self.model.class_prior_, 
            "classes_": self.model.classes_,         
            "epsilon_": self.model.epsilon_,          
            "var_":self.model.var_
        }
        #* Only include 'sigma_' if it exists
        if hasattr(self.model, "sigma_"):
            data_to_save["sigma_"] = self.model.sigma_

        np.savez(model_path, **data_to_save)
    
    #* Load the model from the file and set the attributes
    def load_model(self, filename):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, filename)
        checkpoint = np.load(model_path)
        
        self.model.theta_ = checkpoint["theta_"]
        self.model.class_prior_ = checkpoint["class_prior_"]
        self.model.classes_ = checkpoint["classes_"]
        self.model.epsilon_ = checkpoint["epsilon_"]
        self.model.var_ = checkpoint["var_"]
        
        #* Load 'sigma_' if it exists in the file
        if "sigma_" in checkpoint:
            self.model.sigma_ = checkpoint["sigma_"]