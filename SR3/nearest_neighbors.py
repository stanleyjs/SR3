from collections.abc import Iterable
from functools import partial
from numbers import Number
import multiprocessing
# scipy stack packages
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse, coo_matrix, csr_matrix, kron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import svds as SVDS
from scipy.spatial.distance import pdist, squareform
import torch
from graphtools.base import Data as gt_Data

from .math import optimization, linalg, solvers
from . import utils

def exact_nearest_neighbors_graph(X, k=5, is_distance=True):
    if not is_distance:
        X = squareform(pdist(X))
    X_partitioned = np.argpartition(X, k + 1)
    neighbors = X_partitioned[:, :k + 1]
    edges = np.hstack(np.sort([[i] * k, ele[ele != i]], axis=0)
                      for i, ele in enumerate(neighbors)).T
    np.unique(edges, axis=0)
    return neighbors


def nmslib_graph(X, k=5, method='hnsw', space=None, n_jobs=1):
    n_jobs = int(n_jobs)
    if n_jobs <= 0:
        n_jobs = multiprocessing.cpu_count() + 1 + n_jobs
    import nmslib
    if issparse(X):
        X_dtype = nmslib.DataType.SPARSE_VECTOR
        if space is None:
            space = 'l2_sparse'
    else:
        X_dtype = nmslib.DataType.DENSE_VECTOR
        if space is None:
            space = 'l2'
    index = nmslib.init(method=method, space=space,
                        data_type=X_dtype)
    index.addDataPointBatch(X)
    index.createIndex({'post': 2}, print_progress=True)
    neighbors = index.knnQueryBatch(X, k=k + 1, num_threads=n_jobs)
    edges = np.hstack([np.sort([[ele[0][0]] * k, ele[0][1:]], axis=0)
                       for ele in neighbors]).T
    edges = np.unique(edges, axis=0)
    return edges


def approximate_nn_incidence(X, k=5, n_pca=10, rank_threshold=None, **kwargs):
    op = gt_Data(X, n_pca, rank_threshold)
    Xnu = op.data_nu
    try:
        import nmslib
        nn_func = nmslib_graph
    except:
        nn_func = exact_nearest_neighbors_graph
    edges = nn_func(Xnu, k, **kwargs)
    nedges = edges.shape[0]
    edge_jdx = edges.T.flatten()
    edge_vals = np.ones(nedges)
    edge_vals = np.hstack([edge_vals, -1 * edge_vals])
    edge_idx = np.hstack([np.arange(nedges), np.arange(nedges)])
    incidence_matrix = coo_matrix((edge_vals, (edge_idx, edge_jdx)))
    return incidence_matrix, edges


def tensor_incidence(X, phi = None, k=5, as_sparse=True, n_pca=10, rank_threshold=None, **kwargs):
    X = linalg.check_tensor(X)
    k = utils.match_and_pad_like(k, X.shape)
    n_pca = utils.match_and_pad_like(n_pca, X.shape)
    rank_threshold = utils.match_and_pad_like(rank_threshold, np.ones(X.ndim))
    L = csr_matrix((np.prod(X.shape), np.prod(X.shape)))
    phis = []
    Ads = []
    for mode in range(X.dim()):
        if k[mode] == 0:
            continue
        else:
            Y = linalg.tenmat(X, mode)
            if phi is None:
                phi_bak, _ = approximate_nn_incidence(Y, k=k[mode],
                                                  n_pca=n_pca[mode],
                                                  rank_threshold=rank_threshold[mode],
                                                  **kwargs)
            else:
                phi_bak = phi[mode]
            phi_c = phi_bak.tocsc()
            # phi = coo_matrix_to_torch(phi)
            left_n = np.prod(np.array(X.shape)[
                             np.flatnonzero(np.arange(X.dim()) > mode)])
            left_eye = speye(int(left_n))
            right_n = np.prod(np.array(X.shape)[
                              np.flatnonzero(np.arange(X.dim()) < mode)])
            right_eye = speye(int(right_n))
            Ad = kron(left_eye, kron(phi_c, right_eye))
            L = L + Ad.T.dot(Ad)
            phis.append(phi_bak)
            Ads.append(Ad.tocsc())
    return L.tocsc(), phis, Ads