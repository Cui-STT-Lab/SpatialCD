import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def compute_normalized_gamma(lda, feat_image_train):
    """
    Compute the normalized gamma matrix for the given LDA model and training feature image.

    Parameters:
    lda (object): The LDA model object.
    feat_image_train (pd.DataFrame): The training feature image DataFrame.

    Returns:
    np.ndarray: The normalized gamma matrix.
    """
    # Compute the unnormalized gamma
    gamma = lda._unnormalized_transform(feat_image_train.values)
    
    # Normalize the gamma matrix
    gamma /= gamma.sum(axis=1)[:, np.newaxis]
    
    return gamma

def compute_num_rare(lda, feat_image_train, perc_rare_thresh=0.05):
    """
    Analyze the number of cell types present at a frequency lower than the specified threshold.

    Parameters:
    ldas (list): List of LDA model objects.
    feat_image_train (pd.DataFrame): The training feature image DataFrame.
    perc_rare_thresh (float): The threshold for rare cell types.

    Returns:
    list: Number of cell types present at a frequency lower than the specified threshold for each model.
    """
    theta = compute_normalized_gamma(lda, feat_image_train)
    column_means = np.mean(theta, axis=0)
    
    # Number of cell-types present at fewer than `perc_rare_thresh` on average across pixels
    numrare = np.sum(column_means < perc_rare_thresh) 
    
    return numrare

def compute_rmse(true_theta, gamma):
    """
    Compute the Root Mean Square Error (RMSE) between the true theta matrix and the gamma matrix.

    Parameters:
    true_theta (np.ndarray): The true theta matrix.
    gamma (np.ndarray): The gamma matrix.

    Returns:
    float: The RMSE value.
    """
    # Ensure the matrices have the same shape
    assert true_theta.shape == gamma.shape, "Matrices must have the same shape"
    
    # Compute the RMSE
    rmse = np.sqrt(np.mean((true_theta - gamma) ** 2))
    
    return rmse
