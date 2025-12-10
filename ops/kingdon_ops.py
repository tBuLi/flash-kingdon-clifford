import itertools
from kingdon import MultiVector
from sympy import Symbol

def number_of_wgp_terms(X: MultiVector, Y: MultiVector) -> int:
    """ Returns the number of turns in the weighted gp between X and Y."""
    tot = 0
    for g1, g2 in itertools.product(X.grades, Y.grades):
        Z = X.grade(g1) * Y.grade(g2)
        tot += len(Z.grades)
    return tot

def wgp(X: MultiVector, Y: MultiVector, weights: MultiVector[int]) -> MultiVector:
    """ 
    Compute the weighted geometric product between X and Y. 
    The multivectors are mutiplied grade-wise, and a unique weight 
    is applied to each grade in the output.
    """
    tot = 0
    i = 0
    for gx, gy in itertools.product(X.grades, Y.grades):
        Z = X.grade(gx) * Y.grade(gy)
        for gz in Z.grades:
            tot += weights[i] * Z.grade(gz)
            i += 1
    return tot

def wgp_grad(X: MultiVector, Y: MultiVector, weights: MultiVector[int], go: MultiVector) -> MultiVector[int]:
    """Gradient of the weighted geometric product `wgp` with respect to the inputs and weights."""
    syms: list[Symbol] = [*X.values(), *Y.values(), *weights.e]
    wgp_output = wgp(X, Y, weights)
    go_wgp = wgp_output.sp(~go)  # sp -> scalar product
    return [go_wgp.map(lambda v: v.diff(s)) for s in syms]