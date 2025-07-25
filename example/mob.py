import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
#from dataset import Dataset
import pickle
import os
#from preprocessing import *
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from spatial_lda.knnfeaturization import knn
from spatial_lda.featurization import neighborhood_to_cluster
from spatial_lda.featurization import make_merged_difference_matrices
from spatial_lda.featurization import make_nearest_neighbor_graph
from spatial_lda.featurization import make_minimum_spaning_tree_mask
from spatial_lda.featurization import make_difference_matrix
from spatial_lda.model import order_topics_consistently
import spatial_lda
from spatial_lda.online_lda import LatentDirichletAllocation
import spatial_lda.admm as admm
from spatial_lda.knnfeaturization import knn
from spatial_lda.knnfeaturization import compute_rescaled_distances
from spatial_lda.function import compute_num_rare
import time
import spatial_lda.remodel
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
PATH_TO_MODELS = '/mnt/home/vophuong/Documents/SpatialDC-January/outputs/mob/may4/'
PATH_TO_FOLDER = '/mnt/home/vophuong/Documents/SpatialDC-January/outputs/mob/'  # Replace with your desired path

os.makedirs(PATH_TO_MODELS, exist_ok=True)


# %%
sample_name = 'MOB'
corpus = pd.read_csv('/mnt/home/vophuong/Documents/SpatialDC-HPCC/data/mob/mobCorpusRep8.csv', index_col=0)
feat = corpus.copy()
feat.index = map(lambda x: (sample_name, x), corpus.index)

pos = pd.read_csv('/mnt/home/vophuong/Documents/SpatialDC-HPCC/data/mob/mobPosRep8.csv', index_col=0)
cell_idx = feat.index.map(lambda x: x[1])
# data = pos[pos.index.isin(cell_idx)]
# ds={sample_name:data}
# df = ds[sample_name]
cell_coords = pos.loc[cell_idx][['x', 'y']].values # extract from pos matrix the coordinates of all the spots of current sample (filtered by cell_idx from feature matrix).
cell_coords_df = pd.DataFrame(cell_coords, index=cell_idx, columns=['x', 'y'])
df = cell_coords_df
ds={sample_name:cell_coords_df}

coords = ['x', 'y']

# %%
n_neighbors = 1
sample_names = ['MOB']
from spatial_lda.knnfeaturization import make_diff_matrices
diff_matrix, distance_matrix = make_diff_matrices(sample_names, ds, n_neighbors)


# %%
N_nu_LIST = [1/0.1]
option = "diff_penalty"
# N_TOPICS_LIST = range(2,21,1)
N_TOPICS_LIST = [12]
# n_runs = 4 

verbosity=0
max_primal_dual_iter=400 
max_dirichlet_iter=20
max_dirichlet_ls_iter=10
rho=1e-1
mu=2.0
primal_tol=1e-3
threshold=None
# admm_rho=1.0
# primal_dual_mu=2
admm_rho=0.1,
primal_dual_mu=1e+5
max_admm_iter=15
# max_lda_iter=100


# diff_penalty=1
n_runs = 5 
N_PARALLEL_PROCESSES = 8
max_lda_iter=15

# %% [markdown]
# ### Training

# %%
spatialdc1_models = {}
perplexities = []
num_rares = []

for n_topics in N_TOPICS_LIST:
    for nu in N_nu_LIST:
        diff_penalty = nu
        inv_penalty = 1 / nu  # For naming and saving
        model_key = (n_topics, diff_penalty)

        # Define paths
        path_to_train_model = os.path.join(PATH_TO_MODELS, f'spatialdc_penalty={inv_penalty}_topics={n_topics}_knn={n_neighbors}_option={option}.pkl')
        path_to_gamma = os.path.join(PATH_TO_MODELS, f'gamma_spatialdc_penalty={inv_penalty}_topics={n_topics}_knn={n_neighbors}_option={option}.csv')
        path_to_ppxt = os.path.join(PATH_TO_MODELS, f'ppxt_spatialdc_penalty={inv_penalty}_topics={n_topics}_knn={n_neighbors}_option={option}.csv')

        logging.info(f'Running n_topics={n_topics}, penalty ={inv_penalty}, option={option}')

        if not os.path.exists(path_to_train_model):
            logging.info(f'Training new model for n_topics={n_topics}, diff_penalty={inv_penalty}')

            spatialdc_model = spatial_lda.remodel.train(
                sample_features=feat,
                difference_matrices=diff_matrix,
                difference_penalty=diff_penalty,
                distance_matrices=distance_matrix,
                option=option,
                n_topics=n_topics,
                n_iters=n_runs,
                admm_rho=admm_rho,
                primal_dual_mu=primal_dual_mu, 
                primal_tol=primal_tol,
                max_dirichlet_iter=max_dirichlet_iter, 
                max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                max_primal_dual_iter=max_primal_dual_iter, 
                max_admm_iter=max_admm_iter,
                max_lda_iter=max_lda_iter,
                n_parallel_processes=N_PARALLEL_PROCESSES,
                threshold=threshold
                )  

            spatialdc1_models[model_key] = spatialdc_model

            with open(path_to_train_model, 'wb') as f:
                pickle.dump(spatialdc_model, f)
        else:
            logging.info(f'Loading existing model from {path_to_train_model}')
            with open(path_to_train_model, 'rb') as f:
                spatialdc1_models[model_key] = pickle.load(f)

        # Get model
        model = spatialdc1_models[model_key]

        # Extract the beta matrix (topic-word distributions)
        if hasattr(model, 'components_'):
            beta_matrix = model.components_
        else:
            raise AttributeError("The loaded model does not have 'components_' attribute.")

        # Normalize the beta matrix
        # beta_matrix = beta_matrix / beta_matrix.sum(axis=1, keepdims=True)

        # Convert to DataFrame
        beta_df = pd.DataFrame(beta_matrix)

        # Save to CSV
        output_path = '/mnt/home/vophuong/Documents/SpatialDC-January/outputs/mob/may4/beta_spatialdc_penalty=0.1_topics=12_knn=4_option=diff_penalty_may4.csv'
        beta_df.to_csv(output_path, index=False)

        print(f"Beta matrix saved to: {output_path}")

        # Save gamma
        gamma = model.topic_weights
        gamma_df = pd.DataFrame(gamma)
        gamma_df.to_csv(path_to_gamma, index=True)

        # Compute metrics
        perplexity = model.perplexity(feat)
        num_rare = compute_num_rare(model, feat, 0.05)

        perplexities.append(perplexity)
        num_rares.append(num_rare)


# %% Save Results
if len(perplexities) != len(N_TOPICS_LIST) * len(N_nu_LIST):
    raise ValueError("Mismatch in stored perplexities and topic/penalty combinations.")

results_df = pd.DataFrame({
    'Topic': np.repeat(N_TOPICS_LIST, len(N_nu_LIST)),
    'Penalty': np.tile([1 / nu for nu in N_nu_LIST], len(N_TOPICS_LIST)),
    'Perplexity': perplexities,
    'NumRare': num_rares
})
results_df.to_csv(path_to_ppxt, index=True)
logging.info(f"Results saved to {path_to_ppxt}")
