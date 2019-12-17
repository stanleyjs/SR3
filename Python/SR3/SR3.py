# python standard library
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
from . import utils, nearest_neighbors

svd_func = TruncatedSVD


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
        self.identity = None

    @property
    def x(self):
        return torch.tensor(np.array(self.X).reshape(-1, order='F'))

    def fit(self, X, phi=None, mask=False):
        # construct a knn graph on X
        # TODO: make this take nans in X and convert them to
        # the missing data case.
        X = linalg.check_tensor(X)
        self.X = X
        self.L, self.phi, self.A = nearest_neighbors.tensor_incidence(
            self.X, phi=phi, k=self.k)
        self.nd = self.ndim = self.X.ndim
        self.numel = self.L.shape[0]
        self.identity = speye(self.numel).tocsc()
        self.solver = self.solver.build(A=self.nu * self.identity + self.L)
        return self

    def _proxObjective(self, u, v, gamma, dists, A=None,
                       x=None, shrinkage_function=None):
        # Warning: This function is really broken!
        if shrinkage_function is None:
            shrinkage_function = self.shrinkage_function
        A = self.A if A is None else A
        x = self.x if x is None else x

        fidelity_penalty = 1 / 2 * (x - u).pow(2).sum()
        shrinkage_penalty = 0
        proximal_penalty = 0

        for mode, Ad in enumerate(A):
            shrunken_distances = shrinkage_function.exact_total(dists[mode])
            shrinkage_penalty += gamma[mode] * shrunken_distances

            u_reconstructed = Ad.dot(u)
            proximal_penalty += (1 / (2 * self.nu)) * \
                np.linalg.norm((v[mode] - u_reconstructed))**2

        return fidelity_penalty + shrinkage_penalty + proximal_penalty

    def _getU(self, sumV, u_prev):
        # u_prev is a vector
        U = self.nu * self.identity@self.x + sumV
        U = self.solver.solve(U)
        U = torch.tensor(U)
        return U

    def _getV(self, U, gamma):
        v = []
        gamma = gamma
        sumV = np.zeros(self.numel)
        A = self.A
        dists = []
        for d in range(self.nd):
            uij = A[d].dot(U)
            v.append(uij)
            stride = (self.phi[d].shape[0])
            otherdim = len(uij) // stride
            uij = np.reshape((uij), [stride, otherdim], order='F')
            uij = torch.tensor(uij)
            dists.append(linalg.vecnorm(uij, dim=1).squeeze())
            temp = self.shrinkage_function.prox(dists[d], gamma[d]) / dists[d]
            v[d] = (np.array(v[d]) * np.matlib.repmat(temp, 1, otherdim))
            sumV = sumV + A[d].T.dot(v[d].squeeze())
        return v, sumV, dists

    def getScale(self, gamma, iter=10, halt=True):
        vk = self._getV(self.x, gamma)
        uk = self._getU(vk[1], self.x)
        Fk = self._proxObjective(uk,
                                 vk[0],
                                 gamma,
                                 vk[2])
        loss = [Fk]
        for k in range(iter):
            vk = self._getV(uk, gamma)
            uk = self._getU(vk[1], uk)
            Fk = self._proxObjective(uk,
                                     vk[0],
                                     gamma,
                                     vk[2])
            loss.append(Fk)
            if loss[-1] == loss[-2]:
                break
        return uk, vk, loss

    def transform(self, gamma):
        # Compute the transform of self.X at gamma.
        pass
