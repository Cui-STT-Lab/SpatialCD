import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def compute_rescaled_distances(feature_matrix, ds, train_difference_matrix, sample_name, coords):
    # Extract cleaned cell indices
    cleaned_cell_idx = feature_matrix.index.map(lambda x: x[1])

    # Get coordinates of spots
    cell_coords = ds[sample_name].loc[cleaned_cell_idx][coords].values
    # Rescale coordinates
    min_coords = np.min(cell_coords, axis=0)
    max_coords = np.max(cell_coords, axis=0)
    rescaled_coords = (cell_coords - min_coords) / (max_coords - min_coords)
    
    # Initialize lists to store source nodes and destination nodes
    src_nodes = []
    dst_nodes = []

    # Iterate over each row in the DataFrame
    for _, row in train_difference_matrix.iterrows():
        for j, value in enumerate(row):
            if value == -1:
                src_nodes.append(j)
            elif value == 1:
                dst_nodes.append(j)

    # Compute the differences in coordinates for each pair of points
    coord_difference = rescaled_coords[src_nodes] - rescaled_coords[dst_nodes]
    
    # Compute Euclidean distances from rescaled coordinates
    rescaled_edge_lengths = np.sqrt(np.sum(coord_difference**2.0, axis=1))

    return rescaled_edge_lengths

def compute_rescaled_distances_mpoa(feature_matrix, ds, train_difference_matrix, sample_name, coords):

    # Get coordinates of spots
    cell_coords = ds[sample_name].loc[coords].values
    # Rescale coordinates
    min_coords = np.min(cell_coords, axis=0)
    max_coords = np.max(cell_coords, axis=0)
    rescaled_coords = (cell_coords - min_coords) / (max_coords - min_coords)
    
    # Initialize lists to store source nodes and destination nodes
    src_nodes = []
    dst_nodes = []

    # Iterate over each row in the DataFrame
    for _, row in train_difference_matrix.iterrows():
        for j, value in enumerate(row):
            if value == -1:
                src_nodes.append(j)
            elif value == 1:
                dst_nodes.append(j)

    # Compute the differences in coordinates for each pair of points
    coord_difference = rescaled_coords[src_nodes] - rescaled_coords[dst_nodes]
    
    # Compute Euclidean distances from rescaled coordinates
    rescaled_edge_lengths = np.sqrt(np.sum(coord_difference**2.0, axis=1))

    return rescaled_edge_lengths

def compute_distances(dfs, train_difference_matrix, sample_name, coords):
    # Get coordinates of spots, NOTE: require dfs to be a dictionary, each sample includes cells in feature matrix
    cell_coords = dfs[sample_name][coords].values
    # Rescale coordinates
    min_coords = np.min(cell_coords, axis=0)
    max_coords = np.max(cell_coords, axis=0)
    rescaled_coords = (cell_coords - min_coords) / (max_coords - min_coords)
    
    # Initialize lists to store source nodes and destination nodes
    src_nodes = []
    dst_nodes = []

    # Iterate over each row in the DataFrame
    for _, row in train_difference_matrix.iterrows():
        for j, value in enumerate(row):
            if value == -1:
                src_nodes.append(j)
            elif value == 1:
                dst_nodes.append(j)

    # Compute the differences in coordinates for each pair of points
    coord_difference = rescaled_coords[src_nodes] - rescaled_coords[dst_nodes]
    
    # Compute Euclidean distances from rescaled coordinates
    rescaled_edge_lengths = np.sqrt(np.sum(coord_difference**2.0, axis=1))

    return rescaled_edge_lengths

def knn(df, n_neighbors, name):
    # Initialize the NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # Include the point itself in neighbors
    nn.fit(df)
    
    # Find the nearest neighbors for all points
    distances, indices = nn.kneighbors(df)
    
    # Initialize the target matrix
    num_points = len(df)
    KNN_matrix = np.zeros((num_points * n_neighbors, num_points), dtype=int)
    
    # Fill the target matrix
    row = 0
    for i, neighbors in enumerate(indices):
        for j in range(1, n_neighbors + 1):  # Skip the first neighbor (itself)
            KNN_matrix[row, i] = -1  # Mark the current point
            KNN_matrix[row, neighbors[j]] = 1  # Mark the neighbor
            row += 1
    
    # Convert graph matrix to DataFrame
    KNN_matrix_df = pd.DataFrame(KNN_matrix.astype(int))

    KNN_matrix_df.index = map(lambda x: (name, x), KNN_matrix_df.index)

    return KNN_matrix_df

def knn_graph_single_sample(ds, n_neighbors = 3, sample_name = 'single_sample'):
    rescaled_edge_lengths_array = []
    knn_graph_matrix = {}

    knn_mtx = knn(ds[sample_name], n_neighbors, sample_name)
    
    # Get coordinates of spots
    cell_coords = ds[sample_name].values
    
    # Rescale coordinates
    min_coords = np.min(cell_coords, axis=0)
    max_coords = np.max(cell_coords, axis=0)
    rescaled_coords = (cell_coords - min_coords) / (max_coords - min_coords)
    
    # Initialize lists to store source nodes and destination nodes
    src_nodes = []
    dst_nodes = []
    
    # Iterate over each row in the DataFrame
    for _, row in knn_mtx.iterrows():
        for j, value in enumerate(row):
            if value == -1:
                src_nodes.append(j)
            elif value == 1:
                dst_nodes.append(j)
    
    # Compute the differences in coordinates for each pair of points
    coord_difference = rescaled_coords[src_nodes] - rescaled_coords[dst_nodes]
    
    # Compute Euclidean distances from rescaled coordinates
    rescaled_edge_lengths = np.sqrt(np.sum(coord_difference**2.0, axis=1))
    
    # Save KNN_matrix and rescaled_edge_lengths
    KNN_matrix = csr_matrix(knn_mtx.astype(float))
    knn_graph_matrix[sample_name] = KNN_matrix
    rescaled_edge_lengths_array.append(rescaled_edge_lengths)
    
    return knn_graph_matrix

def knn_graph_multi_sample(ds, n_neighbors = 3, sample_names=None):
    rescaled_edge_lengths_array = []
    knn_graph_matrix = {}
    for sample_name in sample_names:
        knn_mtx = knn(ds[sample_name], n_neighbors, sample_name)
        
        # Get coordinates of spots
        cell_coords = ds[sample_name].values
        
        # Rescale coordinates
        min_coords = np.min(cell_coords, axis=0)
        max_coords = np.max(cell_coords, axis=0)
        rescaled_coords = (cell_coords - min_coords) / (max_coords - min_coords)
        
        # Initialize lists to store source nodes and destination nodes
        src_nodes = []
        dst_nodes = []
        
        # Iterate over each row in the DataFrame
        for _, row in knn_mtx.iterrows():
            for j, value in enumerate(row):
                if value == -1:
                    src_nodes.append(j)
                elif value == 1:
                    dst_nodes.append(j)
        
        # Compute the differences in coordinates for each pair of points
        coord_difference = rescaled_coords[src_nodes] - rescaled_coords[dst_nodes]
        
        # Compute Euclidean distances from rescaled coordinates
        rescaled_edge_lengths = np.sqrt(np.sum(coord_difference**2.0, axis=1))
        
        # Save KNN_matrix and rescaled_edge_lengths
        KNN_matrix = csr_matrix(knn_mtx.astype(float))
        knn_graph_matrix[sample_name] = KNN_matrix
        rescaled_edge_lengths_array.append(rescaled_edge_lengths)
        
    return knn_graph_matrix