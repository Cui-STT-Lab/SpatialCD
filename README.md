# SpatialDC: Reference-free spatially informed cell-type deconvolution for spatial transcriptomics

<img width="1589" height="1013" alt="SpatialDC_jul25" src="https://github.com/user-attachments/assets/88f62515-10cb-43ea-92b4-be16020b3763" />

spatialDC is a spatial transcriptomics deconvolution method that leverages graph-regularized topic models to accurately recover cell-type transcriptional profiles and their proportions at each spatial location. The method incorporates spatial neighborhood information through k-nearest neighbor graphs to improve deconvolution accuracy while maintaining computational efficiency.

## Installation

You can install the development version of spatialDC from GitHub with:

```bash
git clone https://github.com/username/spatialDC.git
cd spatialDC
pip install -e .
```

## Dependencies

```python
import pandas as pd
import os
import logging
import numpy as np
```

## Run spatialDC with Mouse Olfactory Bulb Data

### Data Loading and Preprocessing

The spatialDC workflow starts by loading spatial transcriptomics data and constructing spatial neighborhood graphs.

```python
import pandas as pd
import os
import logging
from spatialdc.utils.function import load_single_sample
from spatialdc.spatial.graph_construction import knn_graph_single_sample
from spatialdc.lda.model import train
from spatialdc.evaluation.evaluation import compute_num_rare

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
PATH_TO_DATA = '/mnt/home/vophuong/spatialDC/data/'
PATH_TO_MODELS = '/mnt/home/vophuong/spatialDC/example/output/'
os.makedirs(PATH_TO_MODELS, exist_ok=True)

# Load spatial transcriptomics data
corpus, pos = load_single_sample(PATH_TO_DATA)
sample_names = ['Sample_1']

# Construct k-nearest neighbor graph
n_neighbors = 4
knn_graph_matrix = knn_graph_single_sample(pos, n_neighbors, sample_name='Sample_1')
```

### Model Training

Train the spatialDC model with graph regularization:

```python
# Set number of topics (cell types)
n_topics = 12

# Train spatialDC model
spatialdc_model = train(
    corpus=corpus,
    graph_matrices=knn_graph_matrix,
    nu_penalty=10,  # Graph regularization strength
    n_topics=n_topics
)
```

### Results Extraction and Evaluation

Extract deconvolution results and compute evaluation metrics:

```python
# Define output paths
path_to_model = os.path.join(PATH_TO_MODELS, f'spatialdc_topics={n_topics}_knn={n_neighbors}.pkl')
path_to_gamma = os.path.join(PATH_TO_MODELS, f'gamma_spatialdc_topics={n_topics}_knn={n_neighbors}.csv')
path_to_beta = os.path.join(PATH_TO_MODELS, f'beta_spatialdc_topics={n_topics}_knn={n_neighbors}.csv')
path_to_ppxt = os.path.join(PATH_TO_MODELS, f'ppxt_spatialdc_topics={n_topics}_knn={n_neighbors}.csv')

# Extract beta matrix (topic-gene distributions)
if hasattr(spatialdc_model, 'components_'):
    beta_matrix = spatialdc_model.components_
else:
    raise AttributeError("The loaded model does not have 'components_' attribute.")

# Save beta matrix (gene expression profiles for each cell type)
beta_df = pd.DataFrame(beta_matrix)
beta_df.to_csv(path_to_beta, index=False)

# Save gamma matrix (cell type proportions for each spot)
gamma = spatialdc_model.topic_weights
gamma_df = pd.DataFrame(gamma)
gamma_df.to_csv(path_to_gamma, index=True)

# Compute evaluation metrics
perplexity = spatialdc_model.perplexity(corpus)
num_rare = compute_num_rare(spatialdc_model, corpus, 0.05)

# Save results
results_df = pd.DataFrame({
    'n_topics': [n_topics],
    'Perplexity': [perplexity],
    'NumRare': [num_rare]
})
results_df.to_csv(path_to_ppxt, index=True)
logging.info(f"Results saved to {path_to_ppxt}")

print(f"Model trained with {n_topics} topics")
print(f"Perplexity: {perplexity:.4f}")
print(f"Number of rare topics: {num_rare}")
```

<img width="3000" height="3000" alt="spatialplt_mob" src="https://github.com/user-attachments/assets/4f2e52ce-ee82-42d9-85be-b9f6511baec4" />

<img width="3000" height="3000" alt="heatmap_mob" src="https://github.com/user-attachments/assets/bf28374a-c120-4272-a32d-0e9f912888f6" />

### Model Selection Across Different Numbers of Topics

For optimal results, test different numbers of topics:

```python
# Test multiple values of K
topic_range = range(5, 20)
results = []

for k in topic_range:
    # Train model
    model = train(
        corpus=corpus,
        graph_matrices=knn_graph_matrix,
        nu_penalty=10,
        n_topics=k
    )
    
    # Evaluate
    perplexity = model.perplexity(corpus)
    num_rare = compute_num_rare(model, corpus, 0.05)
    
    results.append({
        'n_topics': k,
        'perplexity': perplexity,
        'num_rare': num_rare
    })
    
    print(f"K={k}: Perplexity={perplexity:.4f}, NumRare={num_rare}")

# Find optimal number of topics
results_df = pd.DataFrame(results)
optimal_k = results_df.loc[results_df['perplexity'].idxmin(), 'n_topics']
print(f"Optimal number of topics: {optimal_k}")
```

## Key Features

- **Graph-regularized deconvolution**: Incorporates spatial neighborhood information through k-nearest neighbor graphs
- **Reference-free approach**: No need for external single-cell reference data
- **Scalable implementation**: Efficient topic modeling framework
- **Comprehensive evaluation**: Built-in metrics for model assessment
- **Flexible parameters**: Adjustable graph regularization strength and neighborhood size

## Output Files

The spatialDC pipeline generates several output files:

- **Beta matrix** (`beta_*.csv`): Gene expression profiles for each cell type
- **Gamma matrix** (`gamma_*.csv`): Cell type proportions for each spatial location
- **Evaluation metrics** (`ppxt_*.csv`): Model performance metrics including perplexity and number of rare topics

## Parameters

- `n_topics`: Number of cell types to deconvolve
- `nu_penalty`: Graph regularization strength (higher values = more spatial smoothing)
- `n_neighbors`: Number of nearest neighbors for graph construction
- `corpus`: Gene expression count matrix (spots × genes)
- `graph_matrices`: Spatial neighborhood graph

## Citation

If you use spatialDC in your research, please cite:

```
[Your Citation Here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub or contact the development team.
