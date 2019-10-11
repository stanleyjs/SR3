import numpy as np
import multiprocessing
import torch
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from functools import partial
from scipy.sparse import issparse, coo_matrix
from scipy.sparse.linalg import svds as SVDS
from scipy.spatial.distance import pdist, squareform

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

## TENSOR UTILITIES ##


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))


def is_sparse_tensor(X):
    return 'sparse' in X.layout


def coo_matrix_to_torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def tenmat(X, rdim):
    # return the mode-rdim matricization of input,
    # i.e. the rows of the output are the result of flattening along
    # mode rdim.
    ndims = X.dim()
    Xshape = X.shape
    cdims = np.flatnonzero(np.arange(ndims) != rdim)
    Y = X.permute(
        rdim, *cdims).reshape(Xshape[rdim], np.prod(np.array(Xshape)[cdims]))
    return Y


def check_tensor(X, sigma=0.3):
    if not torch.is_tensor(X):
        try:
            if issparse(X):
                Y = coo_matrix_to_torch(X)
            else:
                Y = torch.from_numpy(X)
                # if np.count_nonzero(X) / np.prod(X.shape) <= 0.05:
                #Y = Y.to_sparse()
        except:
            raise ValueError("Your data type was uncastable to a tensor.")
    return Y
## GRAPH BUILDING ##


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
    X = check_tensor(X)
    L = torch.FloatTensor(np.prod(X.shape),np.prod(X.shape))
    phis = []
    Ads = []
    for mode in range(X.dim()):
        Y = tenmat(X, mode)
        phi, _ = approximate_nn_incidence(Y, **kwargs)
        phi = coo_matrix_to_torch(phi)

        left_n = np.prod(np.array(X.shape)[
                         np.flatnonzero(np.arange(X.dim()) > mode)])
        left_eye = torch.eye(int(left_n))
        right_n = np.prod(np.array(X.shape)[
                          np.flatnonzero(np.arange(X.dim()) < mode)])
        right_eye = torch.eye(int(right_n))
        if as_sparse:
            Ad = kronecker(left_eye, kronecker(phi.to_dense(), right_eye))
            L = torch.add(L, torch.matmul(Ad.T,Ad))
            Ad = Ad.to_sparse()
        else:
            phi = phi.to_dense()
            Ad = kronecker(left_eye, kronecker(phi, right_eye))
            L = torch.add(L, torch.matmul(Ad.T,Ad)) 
        phis.append(phi)
        Ads.append(Ad)
    if as_sparse:
        L = L.to_sparse()

    return L, phis, Ads


class SR3(BaseEstimator):
    def __new__(solver):
        pass

    def __init__(self, k=None, graph_function=None):
        if graph_function is None:
            self.graph_function = partial(approximate_nn_graph, k=k)
        else:
            self.graph_function = graph_function

    def fit(self, X):
        # construct a knn graph on X
        pass


class SR3_torch(SR3):

    def __init__(self):
        try:
            import torch
        except:
            raise ImportError("PyTorch is required "
                              "for solving using the SR3_torch class")
