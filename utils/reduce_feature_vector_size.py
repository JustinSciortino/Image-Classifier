from sklearn.decomposition import PCA

#* Apply PCA to reduce the dimensionality of the features
def apply_pca(features, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    #* Return the reduced features as numpy array
    return reduced_features