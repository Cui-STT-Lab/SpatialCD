# SpatialCD: Reference-free spatially informed cell-type deconvolution for spatial transcriptomics

<img width="1589" height="1013" alt="SpatialDC_jul25" src="https://github.com/user-attachments/assets/88f62515-10cb-43ea-92b4-be16020b3763" />

SpatialCD is a spatial transcriptomics deconvolution method that leverages graph-regularized topic models to accurately recover cell-type transcriptional profiles and their proportions at each spatial location. The method incorporates spatial neighborhood information through k-nearest neighbor graphs to improve deconvolution accuracy while maintaining computational efficiency.

## Installation

You can install the development version of SpatialCD from GitHub with:

```bash
git clone https://github.com/username/spatialCD.git
cd spatialCD
pip install -e .
```

## Dependencies

```python
import pandas as pd
import os
import logging
import numpy as np
```

## Run spatialCD with Mouse Olfactory Bulb Data

### Data Loading and Preprocessing

The spatialCD workflow starts by loading spatial transcriptomics data and constructing spatial neighborhood graphs.

```python
from spatialcd.lda.model import train
from spatialcd.spatial.graph_construction import *
from spatialcd.utils.function import *  

PATH_TO_DATA = '/Users/phuong/Library/CloudStorage/OneDrive-MichiganStateUniversity/Projects/spatialCD/data/'
sample_name = 'MOB'
corpus, pos = load_single_sample(PATH_TO_DATA, sample_id=sample_name, corpus_file='mob_corpus.csv', pos_file='mob_pos.csv')

n_neighbors = 4
knn_graph_matrix = knn_graph_single_sample(pos, n_neighbors, sample_name)
```

### Model Training

Train the spatialCD model with graph regularization:

```python
# Set number of topics (cell types)
n_topics = 12

# Train spatialCD model
spatialcd_model = train(
                corpus=corpus,
                graph_matrices= knn_graph_matrix,
                nu_penalty= 10,
                n_topics=n_topics
                )
```

### Results Extraction and Evaluation

SpatialCD extract deconvolution results and compute evaluation metrics and save to the defined path of output:

```python
save_results(spatialcd_model, n_topics, n_neighbors, corpus, PATH_TO_MODELS)
```

<img width="3000" height="3000" alt="spatialplt_mob" src="https://github.com/user-attachments/assets/4f2e52ce-ee82-42d9-85be-b9f6511baec4" />

<img width="3000" height="3000" alt="heatmap_mob" src="https://github.com/user-attachments/assets/bf28374a-c120-4272-a32d-0e9f912888f6" />

### Model Selection Across Different Numbers of Topics

For optimal results, test different numbers of topics:


## Key Features

- **Graph-regularized deconvolution**: Incorporates spatial neighborhood information through k-nearest neighbor graphs
- **Reference-free approach**: No need for external single-cell reference data
- **Scalable implementation**: Efficient topic modeling framework
- **Comprehensive evaluation**: Built-in metrics for model assessment
- **Flexible parameters**: Adjustable graph regularization strength and neighborhood size

## Output Files

The spatialCD pipeline generates several output files:

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

If you use spatialCD in your research, please cite:

```
[Your Citation Here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub or contact the development team.
