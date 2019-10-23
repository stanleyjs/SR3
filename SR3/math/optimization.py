import numpy as np
import torch
from . import linalg


class ShrinkageFunction(object):
    def __new__(cls, desired_base, epsilon=1e-8):
        x = type(desired_base.__name__ + 'Shrinkage',
                 (ShrinkageFunction, desired_base), {'epsilon': epsilon})
        return super(ShrinkageFunction, cls).__new__(x)

    def exact(self, theta, gamma=None, total=False):
        theta = torch.abs(theta)
        if total:
            return self.exact_total(theta, gamma)
        else:
            return self._exact(theta, gamma, total=False)

    def exact_total(self, theta, gamma=None, total=True):
        return self._exact(theta, gamma, total=True)

    def prox(self, theta, gamma):
        theta, sgn = self._parse_theta(theta)
        return self._prox(theta, gamma, sgn=sgn)

    def _parse_theta(self, theta):
        self.sz = sz = theta.shape
        theta = theta.reshape(1, -1)
        self.edges = sz[0]
        sgn = torch.sign(theta)
        theta = torch.abs(theta).squeeze()
        return theta, sgn


class Log(object):
    def __init__(self, super, epsilon=1e-8):
        pass

    def _prox(self, gamma):
        print(gamma)

    def _exact(self, theta, gamma=None, total=True):
        theta = theta + self.epsilon
        y = torch.log(theta)
        return torch.sum(y) if total else y


class Snowflake(object):
    def __init__(self, super, epsilon=1e-8):
        pass

    def _prox(self, theta, gamma, sgn=None):
        if sgn is None:
            sgn = torch.sign(theta)
        coeffs = torch.zeros([self.edges, 3])
        coeffs[:, 0] = self.epsilon
        coeffs[:, 1] = -1 * theta
        coeffs[:, 2] = gamma / 2 - theta * self.epsilon

        candidates = torch.zeros([self.edges, 4])

        candidates[:, 1:] = self._cubic_real_roots(coeffs)
        candidates = candidates**2
        idx = torch.argmin(0.5 * (candidates - theta[:,None])**2 +
                        gamma * self._exact(candidates, total=False), axis=1)
        rho = candidates[np.arange(self.edges), idx]
        rho = sgn * rho
        return rho.squeeze()

    def _exact(self, theta, gamma=None, total=True):
        y = (2 * torch.sqrt(theta) - 2 * self.epsilon *
             torch.log(self.epsilon + torch.sqrt(theta)))
        return torch.sum(y) if total else y

    def _cubic_real_roots(self, coeffs):
        #   solve for real roots of N cubic polynomials of real coefficients written as
        #   0 = x^3 + p x^2 + q x^2 + r
        #   using a factorization of
        #   y^3 + ay + b = 0
        #   where
        #   x = y-p/3
        if isinstance(coeffs, np.ndarray):
            coeffs = torch.Tensor(coeffs)
        p = coeffs[:, 0]
        q = coeffs[:, 1]
        r = coeffs[:, 2]
        N = coeffs.shape[0]
        solutions = torch.zeros([3, coeffs.shape[0]])
        a = (1 / 3) * (3 * q - p**2)
        b = (1 / 27) * (2 * p**3 - 9 * p * q + 27 * r)
    # each pair [a(i) b(i)] are the coefficients of the i-th polynomial in y
        inner_determinant = linalg.around((b**2) / 4 + (a**3) / 27, 6)
        def to_root(pm):
            return np.abs((-b.numpy() / 2) + pm * np.sqrt((inner_determinant.numpy() + 0j)))
        A = to_root(1)
        A = np.sign(A) * np.abs(np.cbrt(A))
        B = to_root(-1)
        B = np.sign(B) * np.abs(np.cbrt(B))
        y_1 = A + B
        y_1 = np.real(y_1)
        y_1 = torch.Tensor(y_1)
        single_root = inner_determinant > 0
        solutions[0, single_root] = y_1[single_root]

        two_roots = inner_determinant == 0
        two_roots_bpos = two_roots * (b > 0)
        two_roots_bneg = two_roots * (b < 0)
        two_roots_b0 = two_roots * (b == 0)

        assert(all(two_roots_bpos + two_roots_bneg + two_roots_b0 == two_roots))
        if any(two_roots_bpos):
            tmp_a = a[two_roots_bpos]
            solutions[0, two_roots_bpos] = -2 * torch.sqrt(-tmp_a / 3)
            solutions[1, two_roots_bpos] = torch.sqrt(-tmp_a / 3)
            solutions[2, two_roots_bpos] = torch.sqrt(-tmp_a / 3)
        if any(two_roots_bneg):
            tmp_a = a[two_roots_bneg]
            solutions[0, two_roots_bneg] = 2 * torch.sqrt(-tmp_a / 3)
            solutions[1, two_roots_bneg] = -torch.sqrt(-tmp_a / 3)
            solutions[2, two_roots_bneg] = -torch.sqrt(-tmp_a / 3)
        if any(two_roots_b0):
            solutions[:, two_roots_b0] = 0

        three_roots = inner_determinant < 0
        if any(three_roots):
            tmp_a = a[three_roots]
            tmp_b = b[three_roots]
            phi = torch.acos(-1 * torch.sign(tmp_b) *
                             torch.sqrt(((tmp_b**2) / 4)
                                        / ((-tmp_a**3) / 27)))
            solutions[0, three_roots] = 2 * \
                torch.sqrt(-tmp_a / 3) * torch.cos(phi / 3 + (0) / 3)
            solutions[1, three_roots] = 2 * \
                torch.sqrt(-tmp_a / 3) * \
                torch.cos(phi / 3 + (2 * np.pi) / 3)
            solutions[2, three_roots] = 2 * \
                torch.sqrt(-tmp_a / 3) * \
                torch.cos(phi / 3 + (4 * np.pi) / 3)
        mask = linalg.around(solutions, 6) == 0
        solutions = solutions - p / 3
        solutions[(mask)] = 0
        return linalg.around(solutions, 6).T
