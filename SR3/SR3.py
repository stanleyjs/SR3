import numpy as np
from numbers import Number
import multiprocessing
import torch
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from functools import partial
from scipy.sparse import issparse, coo_matrix, csr_matrix, kron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import svds as SVDS
from scipy.spatial.distance import pdist, squareform
from .math import optimization, tensors
from . import utils
global __use_pytorch
__use_pytorch = None


def __set_pytorch_backend__(val):
    global __use_pytorch
    if __use_pytorch is None or val is True:
        try:
            import torch
            __use_pytorch = True
        except:
            __use_pytorch = False
            print("error")
    else:
        __use_pytorch = False


__set_pytorch_backend__(None)

svd_func = TruncatedSVD


## GRAPH BUILDING ##

def match_and_pad_like(x, Y, criteria_func=lambda x, Y: x >= Y, fit_func=lambda z: z // 2):
    # match x to Y by expanding it and fitting it to criteria
    dims = Y.ndim
    Y = np.array(Y.shape)
    if isinstance(x, Number):
        x = np.ones(dims) * x
    if isinstance(x, (list, np.ndarray)):
        if len(x) != dims:
            z = -1e15 * np.ones(dims)
            for x, i in zip(x, np.arange(dims)):
                z[i] = x
            z = np.where(z < 0, 0, z)
        else:
            z = x
        z = np.rint(z).astype(int)
        z = np.where(criteria_func(z, Y), fit_func(Y), z)
    return z


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
    edges = np.hstack(np.sort([[ele[0][0]] * k, ele[0][1:]], axis=0)
                      for ele in neighbors).T
    edges = np.unique(edges, axis=0)
    return edges


def approximate_nn_incidence(X, k=5, factor=True, factor_d=10, **kwargs):
    if factor:
        Xnu = svd_func(factor_d).fit_transform(X)
    else:
        Xnu = X
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


def tensor_incidence(X, k=5, as_sparse=True, **kwargs):
    X = tensors.check_tensor(X)
    k = match_and_pad_like(k, X)
    L = csr_matrix((np.prod(X.shape), np.prod(X.shape)))
    phis = []
    Ads = []
    for mode in range(X.dim()):
        if k[mode] == 0:
            continue
        else:
            Y = tensors.tenmat(X, mode)
            phi_bak, _ = approximate_nn_incidence(Y, k=k[mode], **kwargs)
            phi = phi_bak.tocsr()
            #phi = coo_matrix_to_torch(phi)
            left_n = np.prod(np.array(X.shape)[
                             np.flatnonzero(np.arange(X.dim()) > mode)])
            left_eye = speye(int(left_n))
            right_n = np.prod(np.array(X.shape)[
                              np.flatnonzero(np.arange(X.dim()) < mode)])
            right_eye = speye(int(right_n))
            Ad = kron(left_eye, kron(phi, right_eye))
            L = L + Ad.T.dot(Ad)
            phis.append(phi_bak)
            Ads.append(Ad.tocoo())
    return L.tocoo(), phis, Ads


def objective_prox(x, U, V, A, nu, gammas, shrinkage_function):
    fidelity_penalty = 1 / 2 * (x - u).pow(2).sum()
    shrinkage_penalty = 0
    proximal_penalty = 0
    for mode, Ad in enumerate(A):
        distances = tensors.vecnorm(V[mode], dim=1)
        shrunken_distances = shrinkage_function.exact(distances)
        shrinkage_penalty += gammas[mode] * shrunken_distances

        u_reconstructed = torch.mm(Ad, U)
        v = torch.reshape(V[mode], -1, 1)
        proximal_penalty += (1 / (2 * nu)) * (v - u_reconstructed).pow(2).sum()

    return fidelity_penalty + shrinkage_penalty + proximal_penalty

class SR3(BaseEstimator):
    def __new__(solver):
        pass

    def __init__(self, k=None, graph_function=None):
        pass

    def fit(self, X):
        # construct a knn graph on X
        pass
