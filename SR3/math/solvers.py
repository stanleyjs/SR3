import ctypes
import glob
import os
import numpy as np
from numpy.ctypeslib import ndpointer, as_array
from scipy.sparse import issparse
import scipy.sparse.linalg as spla
from pygsp.graphs import Graph
from pygsp.filters import Filter
from SR3 import utils


def full_lib_path(lib, path):
    os.chdir(path)
    exts = ['so', 'dll', 'dylib']
    candidates = [file for file in glob.glob(lib + '*')
                  if file.lower().split('.')[-1] in exts]
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return path + candidates[0]
    else:
        return [path + candidate for candidate in candidates]


def vec_from_ctypes(ndpoint, numel, T=ctypes.c_double):
    return as_array(ctypes.cast(ndpoint, ctypes.POINTER(T * numel)).contents)


class Solver(object):
    def __init__(self, A):
        self._A = None
        self._A = self.A(A)
        if self._A is None:
            self.N = None
        else:
            self.N = self._A.shape[0]
        self.built = False

    def A(self, A=None):
        if A is None:
            return self._A
        else:
            if issparse(A):
                A = A.sorted_indices()
            else:
                raise ValueError("Input matrix must be sparse type.")

            if (self._A is None or not utils.matrix_is_equivalent(A, self._A)) and self.__check_SDDM(A):
                self._A = A
                self.N = self._A.shape[0]
                self.built = False
                return A
            else:
                return A

    def __check_SDDM(self, A):
        A = A.sorted_indices()
        rowsum = np.around(A.sum(0), 15)[0]
        D = A.diagonal()
        A.setdiag(0)
        # check that A is an SDDM matrix formed by a Laplacian L + alpha*eye
        if any(rowsum <= 0):
            raise ValueError(
                "Input matrix is singular or has negative row-sums.")
        if not np.equal.reduce(rowsum):
            raise ValueError(
                "Input matrix does not differ from a Laplacian by a constant diagonal.")
        if any(D <= 0):
            raise ValueError("A is not connected or has negative diagonal")
        if any(A.data > 0):
            raise ValueError("A has non-negative off diagonal elements")
        if not (np.allclose(A.T.nonzero(), A.nonzero())):
            raise ValueError("A does not have a symmetric non zero pattern")
        A.setdiag(D)
        return True

    def build(self, A=None, **kwargs):
        if A is None and self._A is None:
            raise ValueError(
                "Building a solver requires an objective matrix. Please pass an A matrix.")
        else:
            self._A = self.A(A)

class PCGSolver(Solver):
    def __init__(self, A=None, preconditioner = 'spilu',**kwargs):
        super().__init__(A)
        self.scipy_params = kwargs
        if self._A is not None:
            self.build(preconditioner = preconditioner, 
                        **self.scipy_params)

    def A(self, A=None):
        self._A = super().A(A)
        return self._A

    def build(self, A=None, preconditioner = 'spilu',**kwargs):
        super().build(A,**kwargs)
        A = self._A
        if preconditioner == 'spilu':
            M = spla.spilu(A)
            M = spla.LinearOperator(A.shape,M.solve)

            self.__solve = lambda b: spla.bicg(A,b,**kwargs)
        else:
            self.__solve = lambda b: spla.bicg(A,b,**kwargs)
        self.built = True
        return self

    def solve(self, b, A=None):
        if A is not None:
            self._A = self.A(A)
        if not self.built:
            self.build()

        return self.__solve(b)[0]
class PygspSolver(Solver):
    def __init__(self, A=None, **kwargs):
        super().__init__(A)
        self.pygsp_params = kwargs
        if self._A is not None:
            self.build(**self.pygsp_params)

    def A(self, A=None):
        self._A = super().A(A)
        return self._A

    def build(self, A=None, **kwargs):
        super().build(A, **kwargs)
        M = self.A(A).copy()
        offset = np.array(M.sum(0))[0][0]
        print(offset)
        M.setdiag(0)
        M = abs(M)
        M.eliminate_zeros()
        g = Graph(M)
        g.estimate_lmax()
        f = Filter(g, lambda x: 1 / (1 + offset*x))
        #self._G = g
        self.__solve = f.filter
        self.built = True
        return self

    def solve(self, b, order=30, A=None):
        if A is not None:
            self._A = self.A(A)
        if not self.built:
            self.build()
        return self.__solve(b, order=order)

class JuliaSolver(Solver):
    def __init__(self, path_to_library, A=None):
        try:
            self.__lib = self._connect(path_to_library)
        except:
            raise Exception("Unable to connect to Julia library.")
        super().__init__(A)
        self.__A_jl_ptr = None
        if A is not None:
            self.build()

    def A(self, A=None):
        self._A = super().A(A)
        if self._A is None:
            self.__A_jl_ptr = None
        else:
            if self.__jl_connected:
                self.__A_jl_ptr = self._julia_sparse_pointer(self._A)
        return self._A

    @property
    def A_jl_ptr(self):
        return self.__A_jl_ptr

    def _connect(self, path_to_library):
        libpath = str.encode(full_lib_path('laplacians', path_to_library))
        # not sure what to do here yet, i think loading the wrong lib can be bad.
        if isinstance(libpath, list):
            raise Exception("Ambiguous laplacians libraries found")
        lib = ctypes.CDLL(libpath, ctypes.RTLD_GLOBAL)
        lib.jl_init_with_image__threading.argtypes = (
            ctypes.c_char_p, ctypes.c_char_p)
        lib.jl_init_with_image__threading(None, libpath)
        # very important to turn the garbage collector off!
        lib.jl_gc_enable(0)

        self.__lib = lib

        # build the required functions
        f = lib.pointers_to_sparse
        f.restype = ctypes.POINTER(ctypes.c_void_p)
        f.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                      ctypes.c_int]
        self.__julia_sparse_ptr = f

        f = lib.build_SDDMsolver
        f.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        f.restype = ctypes.POINTER(ctypes.c_void_p)
        self.__init_solver = f

        f = lib.SDDMsolve
        f.argtypes = [ctypes.POINTER(ctypes.c_void_p), ndpointer(
            ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_int]
        f.restype = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        self.__solve = f
        self.__jl_connected = True
        return self

    def _julia_sparse_pointer(self, M):
        # this returns a ctypes.POINTER(ctypes.c_void_p)
        if self.__jl_connected:
            assert(issparse(M))
            dat = M.data.astype(ctypes.c_double)
            col_idx = (M.nonzero()[1] + 1).astype(ctypes.c_int)
            row_idx = (M.nonzero()[0] + 1).astype(ctypes.c_int)
            numel = M.nnz
            return self.__julia_sparse_ptr(dat,
                                           row_idx, col_idx,
                                           ctypes.c_int(numel))
        else:
            raise Exception("Julia solver not connected to library")

    def build(self, A=None):
        if A is not None:
            A = self.A(A)
        if self.__jl_connected:
            self.__A_jl_ptr = self._julia_sparse_pointer(self._A)
            self.__built_solver = self.__init_solver(self.A_jl_ptr)
            self.__solve.restype = ndpointer(
                ctypes.c_double, shape=(self.N,), flags="C_CONTIGUOUS")
            self.built = True
            return self
        else:
            raise Exception("Julia Solver not connected to a library.")

    def solve(self, b, A=None):
        if self.__jl_connected:
            if A is not None:
                self._A = self.A(A)
            if not self.built:
                self.build()
            return self.__solve(self.__built_solver, b.astype(ctypes.c_double), ctypes.c_int(len(b)))
        else:
            raise Exception("Julia solver not connected to library")
