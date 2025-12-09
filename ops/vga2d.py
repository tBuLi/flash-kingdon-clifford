"""
Kingdon uses symbolic optimization and CSE to generate efficient code for GA expressions.
This makes it easy to implement GA kernels for arbitrary algebras in both torch and triton,
which are of optimal computational efficiency without requiring manual optimization.

Hence, using kingdon will make this project more scalable and easier to maintain.
"""
from kingdon import Algebra, MultiVector
from sympy import symbols, Symbol
import triton

VGA2D = Algebra(2)
X = VGA2D.multivector(name='x')
Y = VGA2D.multivector(name='y')
ws = symbols('w:10')
weights = VGA2D.scalar(e=ws)
gate = VGA2D.scalar(name='gate')
go = VGA2D.multivector(name='go')

# Compile the weighted geometric product and its gradient kernels
@VGA2D.compile(symbolic=True)
def weighted_gp(X, Y, weights: MultiVector[10]) -> MultiVector:
    w0,w1,w2,w3,w4,w5,w6,w7,w8,w9 = weights
    X0, X1, X2 = (X.grade(g) for g in range(VGA2D.d + 1))
    Y0, Y1, Y2 = (Y.grade(g) for g in range(VGA2D.d + 1))
    return w0*X0*Y0 + w3*(X1|Y1) + w7*X2*Y2 \
         + w1*X0*Y1 + w4*X1*Y0 + w5*X1*Y2 + w8*X2*Y1 \
         + w2*X0*Y2 + w6*(X1^Y1) + w9*X2*Y0

@VGA2D.compile(symbolic=True, codegen_symbolcls=Symbol)
def weighted_gp_grad(x, y, weights: MultiVector[10], go) -> MultiVector[18]:
    """Compute the gradient of the weighted geometric product with respect to the inputs and weights."""
    syms: list[Symbol] = [*x.values(), *y.values(), *weights.e]
    wgp_output = weighted_gp(x, y, weights)
    go_wgp = wgp_output.sp(~go)  # sp -> scalar product
    return [go_wgp.map(lambda v: v.diff(s)) for s in syms]
    
# Extract the compiled function for inputs X, Y, weights (and go <-> gradient output).
weighted_gp_func = weighted_gp[X, Y, weights].func
weighted_gp_grad_func = weighted_gp_grad[X, Y, weights, go].func
# Decorate with triton.jit to convert the compiled GA expression into a triton kernel.
weighted_gp_kernel = triton.jit(weighted_gp_func)
weighted_gp_grad_kernel = triton.jit(weighted_gp_grad_func)
gate_kernel = triton.jit(VGA2D.gp[X, gate].func)  # X * gate
