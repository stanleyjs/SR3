import numpy as np
from SR3.math import optimization


def test_cubicrealroots():
    op = optimization.ShrinkageFunction(optimization.Snowflake)

    def check_solution(coeffs):

        solution = np.roots(coeffs)
        solution = np.where(np.imag(solution) == 0, np.real(solution), 0)
        solution = np.sort(solution)
        assert(np.allclose(np.sort(op._cubic_real_roots(
            coeffs[None, 1:]).squeeze()), solution, atol=1e-5))
    # I am not sure how to test this exhaustively but
    # here are the conditions for polynomials that we need to check for:
    # this polynomial has 1 (positive) real root 0.6823
    # and two imaginary roots.z
    singleroot = np.array([1, 0, 1, -1])
    check_solution(singleroot)
    # the following polynomials test the three conditions where
    # inner_determinant==0 is true, b>0, b<0, b==0.
    # There are two roots. one is repeated.
    # 1 is repeated and the other is -1.
    tworoots_bpos = np.array([1, -1, -1, 1])
    check_solution(tworoots_bpos)
    # 0 is repeated and 1 is the other.
    tworoots_bneg = np.array([1, -1, 0, 0])
    check_solution(tworoots_bpos)

    # The following polynomial has the condition
    # inner_determinant ==0 and b==0.
    # We set to zero as there is some weird imaginary behavior that pops out.
    tworoots_b0 = np.array([1, 1, 1 / 3, 1 / 27])
    assert(np.allclose(
        op._cubic_real_roots(tworoots_b0[None, 1:]), np.zeros(3)))

    # this polynomial has three nonzero real roots
    threeroots = np.array([1, -6, -2, -2])
    check_solution(threeroots)
