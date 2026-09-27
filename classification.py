"""This module contains functions to fit classification models for HW 2 in AE 498 Computational Systems Engineering.

"""
import numpy as np

def fit_knn(X, y, n_neighbors):
    """Function to fit a KNN classifier to a given dataset.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor.
    y : array_like
        Output/response data (n elements), where each element is an observation.
    n_neighbors : int
        Number of k neighbors to use for each prediction.

    Returns
    -------
    y_predicted : array_like
        Predicted responses for input data (n elements).
    error : float
        Error rate for the input data, calculated as 1/n_obs * number of incorrect predictions.

    Notes
    -----
    The test functions assume the k nearest neighbors of observation x include x.

    """

    # YOUR CODE HERE
    #  Suman's code
    X=np.asarray(X)
    y=np.asarray(y)
    y_pred=[]
    for i in range(len(X)):
        distances = np.sqrt(np.sum((X - X[i])**2, axis=1)) # Euclidean distance
        nearest_indices = np.argsort(distances)[:n_neighbors] # Ckecking k-Nearest-Neighbours
        nearest_labels = y[nearest_indices] # checking the class of those neighbours
        labels, counts = np.unique(nearest_labels, return_counts=True) # majority voting
        predicted_class = labels[np.argmax(counts)]
        y_pred.append(predicted_class)

    y_pred=np.asarray(y_pred)
    error=np.mean(y_pred!=y)
    return error,y_pred.tolist() # I did this because I ran pytest inside the HW2 directory and it gave errors pointing to this

