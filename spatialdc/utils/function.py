import os
import pickle
import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from spatialdc.evaluation.evaluation import compute_num_rare

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_single_sample(path_to_data, sample_id='single_sample', corpus_file='corpus.csv', pos_file='pos.csv'):
    """
    Load spot coordinate data for a specific sample from corpus and position CSV files.

    Parameters:
    - path_to_data (str): Path to the directory containing corpus.csv and pos.csv.
    - sample_id (str): Identifier for the sample (default: 'single_sample').
    - corpus_file (str): Filename of the corpus CSV (default: 'corpus.csv').
    - pos_file (str): Filename of the position CSV (default: 'pos.csv').

    Returns:
    - cell_coords_df (pd.DataFrame): DataFrame with 'x' and 'y' coordinates, indexed by cell IDs.
    - ds (dict): Dictionary with the sample ID as key and the coordinate DataFrame as value.
    """
    corpus = pd.read_csv(os.path.join(path_to_data, corpus_file), index_col=0)
    pos = pd.read_csv(os.path.join(path_to_data, pos_file), index_col=0)
    print("Loading and formatting data for a single sample...")
    feat = corpus.copy()
    feat.index = map(lambda x: (sample_id, x), corpus.index)

    cell_idx = feat.index.map(lambda x: x[1])
    cell_coords = pos.loc[cell_idx][['x', 'y']].values

    cell_coords_df = pd.DataFrame(cell_coords, index=cell_idx, columns=['x', 'y'])
    ds = {sample_id: cell_coords_df}

    return feat, ds

def load_multi_sample(path_to_data, sample_names, sample_size, corpus_file='corpus.csv', pos_file='pos.csv'):
    corpus = pd.read_csv(os.path.join(path_to_data, corpus_file), index_col=0)
    pos = pd.read_csv(os.path.join(path_to_data, pos_file), index_col=0)
    print("Loading and formatting data for multiple samples...")
    
    new_index = []
    for i, sample_id in enumerate(sample_names):
        start = i * sample_size
        end = start + sample_size
        sample_block_index = list(map(lambda x: (sample_id, x), corpus.index[start:end]))
        new_index.extend(sample_block_index)

    feat = corpus.copy()
    feat.index = new_index
    
    ds = {        sample_names[i]: pos.iloc[i * sample_size:(i + 1) * sample_size].copy()        for i in range(len(sample_names))    }
    return feat, ds

def save_spatialdc_results(spatialdc_model, n_topics, n_neighbors, corpus, PATH_TO_MODELS):
    
    perplexities = []
    num_rares = []

    path_to_model = os.path.join(PATH_TO_MODELS, f'spatialdc_topics={n_topics}_knn={n_neighbors}.pkl')
    path_to_gamma = os.path.join(PATH_TO_MODELS, f'gamma_spatialdc_topics={n_topics}_knn={n_neighbors}.csv')
    path_to_beta = os.path.join(PATH_TO_MODELS, f'beta_spatialdc_topics={n_topics}_knn={n_neighbors}.csv')
    path_to_ppxt = os.path.join(PATH_TO_MODELS, f'ppxt_spatialdc_topics={n_topics}_knn={n_neighbors}.csv')

    # Save the model
    with open(path_to_model, 'wb') as f:
        pickle.dump(spatialdc_model, f)

    # Extract the beta matrix (topic-word distributions)
    if hasattr(spatialdc_model, 'components_'):
        beta_matrix = spatialdc_model.components_
    else:
        raise AttributeError("The loaded model does not have 'components_' attribute.")

    # Normalize the beta matrix
    # beta_matrix = beta_matrix / beta_matrix.sum(axis=1, keepdims=True)

    # Convert to DataFrame
    beta_df = pd.DataFrame(beta_matrix)

    # Save to CSV
    beta_df.to_csv(path_to_beta, index=False)

    # Save gamma
    gamma = spatialdc_model.topic_weights
    gamma_df = pd.DataFrame(gamma)
    gamma_df.to_csv(path_to_gamma, index=True)

    # Compute metrics
    perplexity = spatialdc_model.perplexity(corpus)
    num_rare = compute_num_rare(spatialdc_model, corpus, 0.05)

    perplexities.append(perplexity)
    num_rares.append(num_rare)

    results_df = pd.DataFrame({
        'n_topics': n_topics,
        'Perplexity': perplexities,
        'NumRare': num_rares
    })
    results_df.to_csv(path_to_ppxt, index=True)
    logging.info(f"Results saved to {path_to_ppxt}")