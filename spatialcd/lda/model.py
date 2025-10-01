from collections import OrderedDict
import itertools
import logging
from multiprocessing import Pool
import time
import numpy as np
import pandas as pd
from scipy.special import digamma
# from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

import spatialcd.lda.admm as admm
from spatialcd.lda.online_lda import LatentDirichletAllocation


def _update_alpha(counts, diff_matrix, diff_penalty, sample_id, verbosity=0, max_iter=15,
               max_primal_dual_iter=400, max_dirichlet_iter=20, max_dirichlet_ls_iter=10,
               rho=1e-1, mu=2.0, primal_tol=1e-3, threshold=None):
    if verbosity >= 1:
        logging.info(f'>>> Infering topic weights for sample {sample_id}')
    weight = 1. / diff_penalty
    cs = digamma(counts) - digamma(np.sum(counts, axis=1, keepdims=True))
    s = weight * np.ones(diff_matrix.shape[0])
    result = admm.admm(cs, diff_matrix, s, rho, verbosity=verbosity, mu=mu, primal_tol=primal_tol,
                       max_dirichlet_iter=max_dirichlet_iter, max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                       max_primal_dual_iter=max_primal_dual_iter, max_iter=max_iter,
                       threshold=threshold)
    if verbosity >= 1:
        logging.info(f'>>> Done inferring topic weights for sample {sample_id}')
    return result

def _wrap_update_alphas(inputs):
    return _update_alpha(**inputs)


def _update_alphas(sample_features, difference_matrices, difference_penalty, gamma,
                n_parallel_processes, verbosity, primal_dual_mu=2, admm_rho=0.1,
                max_primal_dual_iter=400, max_dirichlet_iter=20, max_dirichlet_ls_iter=10,
                max_iter=15, primal_tol=1e-3, threshold=None):
    sample_idxs = sample_features.index.map(lambda x: x[0])
    new_xis = np.zeros_like(gamma)
    if n_parallel_processes > 1:
        with Pool(n_parallel_processes) as pool:
            unique_idxs = np.unique(sample_idxs)
            sample_masks = [
                sample_idxs == sample_idx for sample_idx in unique_idxs]
            sample_counts = [gamma[sample_mask, :]
                             for sample_mask in sample_masks]
            sample_diff_matrices = [difference_matrices[sample_idx]
                                    for sample_idx in unique_idxs]
            diff_penalties = [difference_penalty for _ in unique_idxs]
            tasks = OrderedDict((('counts', sample_counts),
                                 ('diff_matrix', sample_diff_matrices),
                                 ('diff_penalty', diff_penalties),
                                 ('sample_id', unique_idxs),
                                 ('max_iter', itertools.repeat(max_iter)),
                                 ('max_primal_dual_iter', itertools.repeat(max_primal_dual_iter)),
                                 ('max_dirichlet_iter', itertools.repeat(max_dirichlet_iter)),
                                 ('max_dirichlet_ls_iter', itertools.repeat(max_dirichlet_ls_iter)),
                                 # Logging causes multiprocessing to get stuck
                                 # (https://pythonspeed.com/articles/python-multiprocessing/)
                                 ('verbosity', itertools.repeat(0)),                               
                                 ('rho', itertools.repeat(admm_rho)),
                                 ('mu', itertools.repeat(primal_dual_mu)),
                                 ('primal_tol', itertools.repeat(primal_tol)),
                                 ('threshold', itertools.repeat(threshold))))
            # convert into a list of keyword dictionaries
            kw_tasks = [{k: v for k, v in zip(tasks.keys(), values)}
                        for values in list(zip(*tasks.values()))]
            results = list(tqdm(pool.imap(_wrap_update_alphas, kw_tasks),
                                total=len(unique_idxs),
                                position=1,
                                desc='Update alphas'))
            new_xis = np.concatenate(results)
    else:
        for sample_idx in np.unique(sample_idxs):
            sample_mask = sample_idxs == sample_idx
            sample_counts = gamma[sample_mask, :]
            sample_diff_matrix = difference_matrices[sample_idx]
            new_xis[sample_mask] = _update_alpha(sample_counts,
                                              sample_diff_matrix,
                                              difference_penalty,
                                              sample_idx,
                                              max_primal_dual_iter=max_primal_dual_iter,
                                              max_dirichlet_iter=max_dirichlet_iter,
                                              max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                                              max_iter=max_iter,
                                              verbosity=verbosity,
                                              rho=admm_rho,
                                              mu=primal_dual_mu,
                                              primal_tol=primal_tol,
                                              threshold=threshold)
    return new_xis


def train(corpus, graph_matrices, n_topics, nu_penalty=1,
          max_primal_dual_iter=400, max_dirichlet_iter=20, max_dirichlet_ls_iter=10,
          max_lda_iter=100, max_admm_iter=15, n_iters=3, n_parallel_processes=8, verbosity=0,
          primal_dual_mu=2, admm_rho=1.0, primal_tol=1e-3, threshold=None):
    
    xis = None
    for i in range(n_iters):
        lda = LatentDirichletAllocation(n_components=n_topics, 
                                        random_state=42, learning_method='batch',
                                        n_jobs=n_parallel_processes, max_iter=max_lda_iter,
                                        doc_topic_prior=xis
                                        )
        lda.fit(corpus.values)
        gamma = lda._unnormalized_transform(corpus.values)
        xis = _update_alphas(sample_features=corpus,
                          difference_matrices=graph_matrices,
                          difference_penalty=nu_penalty,
                          gamma=gamma,
                          n_parallel_processes=n_parallel_processes,
                          max_iter=max_admm_iter,
                          max_primal_dual_iter=max_primal_dual_iter,
                          max_dirichlet_iter=max_dirichlet_iter,
                          max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                          verbosity=verbosity,
                          primal_dual_mu=primal_dual_mu,
                          admm_rho=admm_rho,
                          primal_tol=primal_tol,
                          threshold=threshold)
        
    lda.topic_weights = pd.DataFrame(lda.fit_transform(corpus.values),
                                     index=corpus.index)
    return lda

def infer(components, sample_features, difference_matrices, difference_penalty=1,
          max_primal_dual_iter=400, max_dirichlet_iter=20, max_dirichlet_ls_iter=10,
          max_admm_iter=15, n_parallel_processes=1):
    
    logging.info('>>> Starting inference')
    n_topics = components.shape[0]
    complete_lda = LatentDirichletAllocation(n_components=n_topics,
                                             random_state=42,
                                             n_jobs=n_parallel_processes,
                                             max_iter=2,
                                             doc_topic_prior=None,
                                             verbose=1, evaluate_every=1)
    complete_lda.set_components(components)
    gamma = complete_lda._unnormalized_transform(sample_features.values)
    xis = _update_alphas(sample_features,
                      difference_matrices,
                      difference_penalty,
                      max_iter=max_admm_iter,
                      max_primal_dual_iter=max_primal_dual_iter,
                      max_dirichlet_iter=max_dirichlet_iter,
                      max_dirichlet_ls_iter=max_dirichlet_ls_iter,
                      gamma=gamma,
                      n_parallel_processes=n_parallel_processes,
                      verbosity=1)
    complete_lda.doc_topic_prior_ = xis
    topic_weights = pd.DataFrame(complete_lda.transform(sample_features.values),
                                 index=sample_features.index)
    complete_lda.topic_weights = topic_weights
    return complete_lda


