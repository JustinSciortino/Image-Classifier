import numpy as np
import os

class GaussianNaiveBayes():

    """
    Key Concepts:
    P(y|X) what we want: probability of class y given features X
    P(X|y) likelihood: probability of features given class
    P(y) prior: probability of class occurring
    1. P(y|X) ∝ P(X|y) * P(y)  [Bayes' Theorem]
    2. P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)  [Naive Independence Assumption]
    3. P(xᵢ|y) follows Gaussian distribution for each feature i and class y
    """

    def __init__(self):

        #* Means of each feature for each class
        #* Format: {class_label: [mean1, mean2, ...], ...}
        self.class_means = {} 

        #* Variances of each feature for each class
        #* Format: {class_0: [var_feature1, var_feature2, ...], ...}
        self.class_variances = {}

        #* Prior probavility of each class
        #* Format: {class_label: number, ...}
        self.class_priors = {}

        #* Unique class labels, e.g., [0,1,2,3,4,5,6,7,8,9] for CIFAR-10
        self.classes = None  

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        """
        Calculate the means, variances and priors for each class

        X is a 2D numpy array with shape (n_samples, n_features)
        - n_samples is the number of training examples (rows in X)
        - n_features is the number of features per example (columns in X)
        
        y is a 1D numpy array with shape (n_samples,)
        - Contains the class label for each training example

        """

        #* np.unique gets all unique class labels from y so [0,1,2,3,4,5,6,7,8,9] for the cifar-10 dataset
        self.classes = np.unique(y)
        
        # Get number of samples in training set
        # X.shape returns a tuple (n_rows, n_columns)
        # X.shape[0] gets first element of tuple = number of rows = n_samples
        n_samples = X.shape[0]
        
        # Calculate parameters for each class separately
        for c in self.classes:  # For each class label (0 through 9 for CIFAR-10)
            
            # X[y == c] creates boolean mask where y equals current class
            # Then gets all rows from X where mask is True
            # Result: X_samples_current_c contains only the samples belonging to current class
            X_samples_current_c = X[y == c]
            
            # Calculate mean of each feature for current class
            # np.mean(X_c, axis=0) computes mean along axis 0 (down columns)
            # Result: array of means, one for each feature
            self.class_means[c] = np.mean(X_samples_current_c, axis=0)
            
            # Calculate variance of each feature for current class
            # np.var(X_c, axis=0) computes variance along axis 0 (down columns)
            # Add small number (1e-9) to prevent division by zero in later calculations
            self.class_variances[c] = np.var(X_samples_current_c, axis=0) + 1e-9
            
            # Calculate prior probability of current class
            # X_samples_current_c.shape[0] gets number of samples in current class
            # Divided by total samples = probability of this class occurring
            self.class_priors[c] = X_samples_current_c.shape[0] / n_samples

    def calculate_likelihood(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """
        Source for the formulas and addapted code: https://github.com/oniani/ai/blob/main/model/ml/gaussian_naive_bayes.py#L43

        Calculates log likelihood of a single sample x belonging to a class
        with given mean and variance parameters
        
        x: single sample feature vector (n_features,)
        mean: mean vector for a class (n_features,)
        var: variance vector for a class (n_features,)
        """
        
        # Implementation of Gaussian probability density function in log space
        # Log likelihood = -0.5 * sum(log(2π*σ²) + (x-μ)²/σ²)
        # Where: μ = mean, σ² = variance
        return -0.5 * np.sum(
            # First term: log(2πσ²)
            np.log(2 * np.pi * var) +
            
            # Second term: (x-μ)²/σ²
            # (x - mean)**2 computes squared difference for each feature
            # Divided by variance normalizes the difference
            ((x - mean) ** 2) / var
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: 2D array of samples to predict, shape (n_samples, n_features)
        Returns: 1D array of predicted class labels, shape (n_samples,)
        """
        predictions = []  # Will store predicted class for each sample
        
        # For each sample in X
        for x in X:  # x is a single sample feature vector
            posteriors = []  # Will store posterior probability for each class
            
            # Calculate posterior probability for each class
            for c in self.classes:
                # Start with log of prior probability
                # np.log converts to log space to prevent numerical underflow
                posterior = np.log(self.class_priors[c])
                
                # Add log likelihood (using helper function)
                posterior += self.calculate_likelihood(
                    x,                    # Current sample
                    self.class_means[c],  # Mean vector for current class
                    self.class_variances[c]    # Variance vector for current class
                )
                
                # Append posterior probability for this class
                posteriors.append(posterior)
            
            # np.argmax gets index of highest posterior probability
            # self.classes[index] converts index to actual class label
            # predictions.append adds this prediction to our list
            predictions.append(self.classes[np.argmax(posteriors)])
        
        # Convert predictions list to numpy array for consistency and return 
        return np.array(predictions)
    
    def save(self, model_name):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        np.savez(model_path,
                 means=self.class_means,
                 variances=self.class_variances,
                 priors=self.class_priors,
                 classes=self.classes)

    def load(self, model_name):
        TRAINED_MODEL_DIR = "trained_models"
        model_path = os.path.join(TRAINED_MODEL_DIR, model_name)
        if not model_path.endswith('.npz'):
            model_path += '.npz'

        checkpoint = np.load(model_path, allow_pickle=True)
        self.class_means = checkpoint['means'].item()
        self.class_variances = checkpoint['variances'].item()
        self.class_priors = checkpoint['priors'].item()
        self.classes = checkpoint['classes'].item()