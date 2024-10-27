import numpy as np
import os

class GaussianNaiveBayes():
    def __init__(self):
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X, y):
        for class_label in np.unique(y):
            X_class = X[y == class_label]
            self.means[class_label] = X_class.mean(axis=0)
            self.variances[class_label] = X_class.var(axis=0)
            self.priors[class_label] = X_class.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for class_label in self.means:
                prior = np.log(self.priors[class_label])
                class_conditional = -0.5 * np.sum(np.log(2 * np.pi * self.variances[class_label]))
                class_conditional -= 0.5 * np.sum(((x - self.means[class_label]) ** 2) / (self.variances[class_label] + 1e-9))
                posterior = prior + class_conditional
                posteriors.append(posterior)
            predictions.append(np.argmax(posteriors))
        return np.array(predictions)
    
    def save(self, model_name):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        np.savez(model_path, 
                 means=self.means,
                 variances=self.variances,
                 priors=self.priors)

    def load(self, model_name):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        if not model_path.endswith('.npz'):
            model_path += '.npz'
        checkpoint = np.load(model_path, allow_pickle=True)
        self.means = checkpoint['means'].item()
        self.variances = checkpoint['variances'].item()
        self.priors = checkpoint['priors'].item()
    
    """def save(self, model_name):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        torch.save({
            'means': self.means,
            'variances': self.variances,
            'priors': self.priors
        }, model_path)

    def load(self, model_name):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        checkpoint = torch.load(model_path)
        self.means = checkpoint['means']
        self.variances = checkpoint['variances']
        self.priors = checkpoint['priors']"""
