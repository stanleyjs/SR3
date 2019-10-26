import numpy as np
import torch as torch
from scipy.sparse import issparse, coo_matrix

from sklearn.decomposition import TruncatedSVD


def torch_kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(
        A.size(0) * B.size(0), A.size(1) * B.size(1))


def is_sparse_tensor(X):
    return 'sparse' in X.layout

def around(X, n_digits):
    return torch.round(X * 10**n_digits) / (10**n_digits)

def coo_matrix_to_torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def vecnorm(Y, p=2, dim='rows'):
    if dim == 'rows':
        dim = 1
    elif dim == 'cols':
        dim = 0
    if torch.is_tensor(Y):
        Y = torch.pow(Y, p)
        if Y.is_sparse:
            nrm = torch.sparse.sum(Y, dim=dim)
        else:
            nrm = torch.sum(Y, dim=dim)
        return nrm.pow(1 / p)
    else:
        if issparse(Y):
            Y = sparse_power(Y, p=p)
            nrm = np.asarray(Y.sum(dim))
        else:
            Y = np.power(Y, p)
            nrm = np.sum(Y, axis=dim)
        return np.power(nrm, 1 / p)


def sparse_power(Y, Y0=None, p=2):
    if Y0 is None:
        Y0 = Y
    if p == 1:
        return Y
    else:
        return sparse_power(Y.multiply(Y0), Y0=Y0, p=p - 1)


def tenmat(X, rdim, as_np=True, force_sparse=False):
    # return the mode-rdim matricization of input,
    # i.e. the rows of the output are the result of flattening along
    # mode rdims
    ndims = X.dim()
    Xshape = np.array(X.shape)
    cdims = np.flatnonzero(np.arange(ndims) != rdim)
    if X.is_sparse:
        indices = np.array(X.coalesce().indices())
        ridx = indices[rdim, :]
        csize = Xshape[cdims]
        cidx = np.ravel_multi_index(indices[cdims, :], csize)
        if as_np:
            Y = coo_matrix((X.values(), (ridx, cidx)))
        elif not as_np:
            new_inds = np.vstack([ridx, cidx])
            Y = torch.sparse.FloatTensor(new_inds, X.values(),
                                         torch.size([Xshape[rdim],
                                                     np.prod(csize)]))
    else:
        Y = X.permute(
            rdim, *cdims).reshape(Xshape[rdim],
                                  np.prod(np.array(Xshape)[cdims]))
        if as_np:
            Y = np.array(Y)
            if force_sparse:
                Y = coo_matrix(Y)
        else:
            if force_sparse:
                Y = Y.to_sparse()
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
    else:
        Y = X
    return Y
