"""
Kingdon uses symbolic optimization and CSE to generate efficient code for GA expressions.
This makes it easy to implement GA kernels for arbitrary algebras in both torch and triton,
which are of optimal computational efficiency without requiring manual optimization.

Hence, using kingdon will make this project more scalable and easier to maintain.
"""
from kingdon import Algebra
from sympy import symbols, Symbol
import triton
from .kingdon_ops import wgp, wgp_grad, number_of_wgp_terms

VGA3D = Algebra(3)
X = VGA3D.multivector(name='x')
Y = VGA3D.multivector(name='y')
ws = symbols(f'w:{number_of_wgp_terms(X, Y)}')
weights = VGA3D.scalar(e=ws)
gate = VGA3D.scalar(name='gate')
go = VGA3D.multivector(name='go')

# Mark the weighted geometric product and its gradient for compilation.
weighted_gp = VGA3D.compile(wgp, symbolic=True)
weighted_gp_grad = VGA3D.compile(wgp_grad, symbolic=True, codegen_symbolcls=Symbol)
# Extract the compiled function for inputs X, Y, weights (and go <-> gradient output).
weighted_gp_func = weighted_gp[X, Y, weights].func
weighted_gp_grad_func = weighted_gp_grad[X, Y, weights, go].func
# Decorate with triton.jit to convert the compiled GA expression into a triton kernel.
weighted_gp_kernel = triton.jit(weighted_gp_func)
weighted_gp_grad_kernel = triton.jit(weighted_gp_grad_func)
gate_kernel = triton.jit(VGA3D.gp[X, gate].func)  # X * gate
