import SR3
import numpy as np
import torch
import nose2

def random_sparse(dims, tensor=True, prct=0.05, seed=42):
    generator = np.random.RandomState(seed)
    numel = np.prod(dims)
    keep = int(np.floor(numel * prct))
    linear_inds = generator.choice(np.arange(numel),
                                   keep, replace=False)
    output = np.zeros(numel)
    output[linear_inds] = generator.randn(keep)
    output = output.reshape(dims)
    subs = output.nonzero()
    if tensor:
        output = torch.tensor(output)
        output = output.to_sparse()
    return output, linear_inds, subs
