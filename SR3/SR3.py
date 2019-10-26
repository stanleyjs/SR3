# python standard library
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

svd_func = TruncatedSVD


## GRAPH BUILDING ##

def match_and_pad_like(x, Y, criteria_func=lambda x, Y: x >= Y, fit_func=lambda z: z // 2):
    # match x to Y by expanding it and fitting it to criteria
    dims = len(Y)
    Y = np.array(Y)
    # x is either an iterable or an element.  It can contain a mixture of types.  We want to make a list of length len(Y).
    if not isinstance(x, Iterable):
        x = [x] * dims  # repeat whatever x is.

    # now we check to see if x is incomplete. If it is incomplete, pad it with zeros.
    if isinstance(x, (list, np.ndarray)):
        z = x + [0] * (dims - len(x))
    # finally round everything
    q = np.where([isinstance(ele, str)for ele in z], -1e6, z).astype(float)
    q = np.rint(q).astype(int)
    q = np.where(criteria_func(q, Y), fit_func(Y), q)
    for i in range(dims):
        if q[i] < 0:
            z[i] = z[i]
        else:
            z[i] = q[i]
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
    k = match_and_pad_like(k, X.shape)
    n_pca = match_and_pad_like(n_pca, X.shape)
    rank_threshold = match_and_pad_like(rank_threshold, np.ones(X.ndim))
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


class SR3(BaseEstimator):

    def __init__(self, solver, k=None, nu=1e-3,
                 shrinkage_function=optimization.Snowflake,
                 epsilon=1e-8,
                 npca=10, rank_threshold=None):
        if k is None:
            k = 5
        self.k = k
        if isinstance(nu, Number) and 1 >= nu > 0:
            self.nu = nu
        else:
            raise ValueError("Nu must be a number in the interval (0,1].")
        self.epsilon = epsilon
        self.shrinkage_function = optimization.ShrinkageFunction(
            shrinkage_function, epsilon=self.epsilon)
        self.solver = solver

        self.numel = None
        self.X = None
        self.L = None
        self.phi = None
        self.A = None
        self.nd = self.ndim = None
        self.I = None

    @property
    def x(self):
        return self.X.reshape(-1)

    def fit(self, X, phi = None, mask=False):
        # construct a knn graph on X
        # TODO: make this take nans in X and convert them to
        # the missing data case.
        X = linalg.check_tensor(X)
        self.X = X
        self.L, self.phi, self.A = tensor_incidence(self.X, phi=phi, k=self.k)
        self.nd = self.ndim = self.X.ndim
        self.numel = self.L.shape[0]
        self.I = speye(self.numel).tocsc()
        self.solver = self.solver.build(A=self.nu * self.I + self.L)
        return self

    def _proxObjective(self, u, V, gamma, A=None,
                        x=None, shrinkage_function=None):
    #Warning: This function is really broken!
        if shrinkage_function is None:
            shrinkage_function = self.shrinkage_function
        A = self.A if A is None else A
        x = self.x if x is None else x

        fidelity_penalty = 1 / 2 * (x - u).pow(2).sum()
        shrinkage_penalty = 0
        proximal_penalty = 0

        for mode, Ad in enumerate(A):
            distances = linalg.vecnorm(V[mode], dim=1)
            shrunken_distances = shrinkage_function.exact(distances)
            shrinkage_penalty += gamma[mode] * shrunken_distances

            u_reconstructed = torch.Tensor(Ad.dot(u))
            v = torch.reshape(V[mode], [-1, 1])
            proximal_penalty += (1 / (2 * self.nu)) * \
                (v - u_reconstructed).pow(2).sum()

        return fidelity_penalty + shrinkage_penalty + proximal_penalty

    def _getU(self, sumV, u_prev):
        # u_prev is a vector
        U = self.nu * self.I@self.x + sumV
        U = self.solver.solve(U)
        U = torch.tensor(U)
        return U

    def _getV(self, U, gamma):
        v = []
        gamma = self.nu * gamma
        sumV = np.zeros(self.numel)
        A = self.A
        for d in range(self.nd):
            uij = A[d].dot(U)
            v.append(uij)
            stride = (self.phi[d].shape[0])
            otherdim = len(uij) // stride
            uij = torch.reshape(torch.tensor(uij), [stride,-1])
            dists = linalg.vecnorm(uij,dim=1)
            temp = self.shrinkage_function.prox(dists, gamma[d]) / dists
            v[d] = (np.array(v[d])* np.matlib.repmat(temp, 1, otherdim))
            sumV = sumV + A[d].T.dot(v[d].squeeze())
        return v, sumV

    def getScale(self, gamma, iter=100,halt=True):
        vk = self._getV(self.x, gamma)
        uk = self._getU(vk[1], self.x)
        #Fk = self._proxObjective(uk,
                                  #vk[0],
                                  #gamma)
       #loss = [Fk]
        for k in range(iter):
            vk = self._getV(uk, gamma)
            uk = self._getU(vk[1], uk)
            #Fk = self._proxObjective(uk,
                                      #vk[0],
                                      #gamma)
            #loss.append(Fk)
            #if loss[-1] == loss[-2]:
                #break
        return uk, vk#, loss

    def transform(self, gamma):
        # Compute the transform of self.X at gamma.
        pass
