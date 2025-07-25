import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def load_single_sample(path_to_data, sample_id='Sample_1', corpus_file='corpus.csv', pos_file='pos.csv'):
    """
    Load spot coordinate data for a specific sample from corpus and position CSV files.

    Parameters:
    - path_to_data (str): Path to the directory containing corpus.csv and pos.csv.
    - sample_id (str): Identifier for the sample (default: 'Sample_1').
    - corpus_file (str): Filename of the corpus CSV (default: 'corpus.csv').
    - pos_file (str): Filename of the position CSV (default: 'pos.csv').

    Returns:
    - cell_coords_df (pd.DataFrame): DataFrame with 'x' and 'y' coordinates, indexed by cell IDs.
    - ds (dict): Dictionary with the sample ID as key and the coordinate DataFrame as value.
    """
    corpus = pd.read_csv(os.path.join(path_to_data, corpus_file), index_col=0)
    pos = pd.read_csv(os.path.join(path_to_data, pos_file), index_col=0)

    feat = corpus.copy()
    feat.index = map(lambda x: (sample_id, x), corpus.index)

    cell_idx = feat.index.map(lambda x: x[1])
    cell_coords = pos.loc[cell_idx][['x', 'y']].values

    cell_coords_df = pd.DataFrame(cell_coords, index=cell_idx, columns=['x', 'y'])
    ds = {sample_id: cell_coords_df}

    return feat, ds




# def compute_normalized_gamma(lda, feat_image_train):
#     """
#     Compute the normalized gamma matrix for the given LDA model and training feature image.

#     Parameters:
#     lda (object): The LDA model object.
#     feat_image_train (pd.DataFrame): The training feature image DataFrame.

#     Returns:
#     np.ndarray: The normalized gamma matrix.
#     """
#     # Compute the unnormalized gamma
#     gamma = lda._unnormalized_transform(feat_image_train.values)
    
#     # Normalize the gamma matrix
#     gamma /= gamma.sum(axis=1)[:, np.newaxis]
    
#     return gamma

# def compute_num_rare(lda, feat_image_train, perc_rare_thresh=0.05):
#     """
#     Analyze the number of cell types present at a frequency lower than the specified threshold.

#     Parameters:
#     ldas (list): List of LDA model objects.
#     feat_image_train (pd.DataFrame): The training feature image DataFrame.
#     perc_rare_thresh (float): The threshold for rare cell types.

#     Returns:
#     list: Number of cell types present at a frequency lower than the specified threshold for each model.
#     """
#     theta = compute_normalized_gamma(lda, feat_image_train)
#     column_means = np.mean(theta, axis=0)
    
#     # Number of cell-types present at fewer than `perc_rare_thresh` on average across pixels
#     numrare = np.sum(column_means < perc_rare_thresh) 
    
#     return numrare

# def compute_rmse(true_theta, gamma):
#     """
#     Compute the Root Mean Square Error (RMSE) between the true theta matrix and the gamma matrix.

#     Parameters:
#     true_theta (np.ndarray): The true theta matrix.
#     gamma (np.ndarray): The gamma matrix.

#     Returns:
#     float: The RMSE value.
#     """
#     # Ensure the matrices have the same shape
#     assert true_theta.shape == gamma.shape, "Matrices must have the same shape"
    
#     # Compute the RMSE
#     rmse = np.sqrt(np.mean((true_theta - gamma) ** 2))
    
#     return rmse


# def plot_gamma_gt_correlation_styled(
#     gamma_matrix,
#     gt_theta,
#     colLabs="LDA Topics",
#     rowLabs="Ground Truth Topics",
#     title="Correlation Between Ground Truth Theta and Gamma",
#     annotation=False,
#     figsize=(8, 6),
#     save_path=None
# ):
#     """
#     Compute and plot a styled heatmap of the correlation between gamma (estimated topic proportions)
#     and gt_theta (ground truth topic proportions).

#     Parameters:
#         gamma_matrix (np.ndarray or pd.DataFrame): shape (n_samples, n_topics)
#         gt_theta (pd.DataFrame): shape (n_samples, n_topics)
#         colLabs (str): Label for x-axis
#         rowLabs (str): Label for y-axis
#         title (str): Title of the heatmap
#         annotation (bool): Whether to annotate heatmap cells with correlation values
#         figsize (tuple): Size of the plot
#         save_path (str): Path to save the PNG image (optional)
#     """
    
#     # Ensure gamma is a DataFrame with same index
#     if isinstance(gamma_matrix, np.ndarray):
#         gamma = pd.DataFrame(gamma_matrix, index=gt_theta.index, 
#                              columns=[f"gamma_{i+1}" for i in range(gamma_matrix.shape[1])])
#     else:
#         gamma = gamma_matrix.loc[gt_theta.index]

#     # Compute correlation matrix (k x k)
#     cor_matrix = pd.DataFrame(
#         np.corrcoef(gamma.T, gt_theta.T)[:gamma.shape[1], gamma.shape[1]:],
#         index=gamma.columns, columns=gt_theta.columns
#     )

#     # Transpose so rows = GT topics, columns = estimated topics
#     cor_matrix = cor_matrix.T

#     # Create custom diverging colormap (blue → white → red)
#     colors = ["blue", "white", "red"]
#     correlation_palette = LinearSegmentedColormap.from_list("custom_corr", colors, N=209)

#     # Plot heatmap
#     plt.figure(figsize=figsize)
#     ax = sns.heatmap(
#         cor_matrix,
#         cmap=correlation_palette,
#         vmin=-1,
#         vmax=1,
#         center=0,
#         annot=annotation,
#         fmt=".2f",
#         linewidths=0.5,
#         linecolor="black",
#         cbar_kws={
#             "label": "Correlation",
#             "orientation": "vertical",
#             "shrink": 0.9
#         }
#     )

#     # Axis labels and title
#     if colLabs:
#         ax.set_xlabel(colLabs, fontsize=13)
#     if rowLabs:
#         ax.set_ylabel(rowLabs, fontsize=13)
#     if title:
#         ax.set_title(title, fontsize=15)

#     # Axis text styling
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=12, color="black")
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=12, color="black")

#     # Add panel styling (similar to ggplot2)
#     ax.figure.set_facecolor("white")
#     ax.set_facecolor("white")
#     for _, spine in ax.spines.items():
#         spine.set_visible(True)
#         spine.set_linewidth(2)
#         spine.set_color("black")

#     plt.tight_layout()

#     # Save figure
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")

#     plt.show()


# def plot_reordered_correlation(
#     gamma_matrix,
#     gt_theta,
#     colLabs="Estimated Topics",
#     rowLabs="Ground Truth Topics",
#     title="Correlation (Reordered by Best-Matching Topics)",
#     annotation=False,
#     figsize=(8, 6),
#     save_path=None
# ):
#     """
#     Plot a correlation heatmap where columns are reordered by each row's maximum correlation,
#     and rows are reversed (inverted order).
#     """
#     # Step 1: Ensure gamma is a DataFrame with matching index
#     if isinstance(gamma_matrix, np.ndarray):
#         gamma = pd.DataFrame(gamma_matrix, index=gt_theta.index, 
#                              columns=[f"X{i+1}" for i in range(gamma_matrix.shape[1])])
#     else:
#         gamma = gamma_matrix.loc[gt_theta.index]
#         gamma.columns = [f"X{i+1}" for i in range(gamma.shape[1])]

#     # gt_theta.columns = [f"X{i+1}" for i in range(gt_theta.shape[1])]

#     # Step 2: Compute correlation matrix (ground truth rows × estimated columns)
#     cor_matrix = pd.DataFrame(
#         np.corrcoef(gt_theta.T, gamma.T)[:gt_theta.shape[1], gt_theta.shape[1]:],
#         index=gt_theta.columns,
#         columns=gamma.columns
#     )

#     # Step 3: Reorder columns by row-wise max correlation
#     best_matches = cor_matrix.idxmax(axis=1).tolist()
#     unique_best_matches = []
#     [unique_best_matches.append(x) for x in best_matches if x not in unique_best_matches]

#     reordered_cor_matrix = cor_matrix[unique_best_matches]

#     # Step 4: Reverse the row order
#     reordered_cor_matrix = reordered_cor_matrix.iloc[::-1]

#     # Step 5: Plot
#     colors = ["blue", "white", "red"]
#     palette = LinearSegmentedColormap.from_list("custom_corr", colors, N=209)

#     plt.figure(figsize=figsize)
#     ax = sns.heatmap(
#         reordered_cor_matrix,
#         cmap=palette,
#         vmin=-1,
#         vmax=1,
#         center=0,
#         annot=annotation,
#         fmt=".2f",
#         linewidths=0.5,
#         linecolor="black",
#         cbar_kws={
#             "label": "Correlation",
#             "orientation": "vertical",
#             "shrink": 0.9
#         }
#     )

#     # Axis labels and styling
#     ax.set_xlabel(colLabs, fontsize=13)
#     ax.set_ylabel(rowLabs, fontsize=13)
#     ax.set_title(title, fontsize=15)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=12, color="black")
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=12, color="black")

#     ax.figure.set_facecolor("white")
#     ax.set_facecolor("white")
#     for _, spine in ax.spines.items():
#         spine.set_visible(True)
#         spine.set_linewidth(2)
#         spine.set_color("black")

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")

#     plt.show()

#     return reordered_cor_matrix

