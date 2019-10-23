import numpy as np
import torch as torch
from scipy.sparse import issparse


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


def vecnorm(Y, p, dim='rows'):
    if dim == 'rows':
        dim = 1
    elif dim == 'cols':
        dim = 0
    Y = torch.pow(Y, p)
    if Y.issparse():
        nrm = torch.sparse.sum(Y, dim=dim)
    else:
        nrm = torch.sum(Y, dim=dim)
    return nrm.pow(1 / p)


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
