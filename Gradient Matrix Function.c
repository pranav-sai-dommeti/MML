import sympy as sp
from sympy import display

# Define the function that computes gradient
def Gradient(f, x):
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            mat_2 = sp.zeros(x.shape[0], x.shape[1])
            for k in range(x.shape[0]):
                for l in range(x.shape[1]):
                    mat_2[k, l] = sp.diff(f[i, j], x[k, l])
            display(mat_2)

# Define symbols
x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')

# Define the matrix-valued function f and the variable matrix x
f = sp.Matrix([[sp.sin(x0 + 2*x1), 2*x1 + x3],
               [2*x0 + x2, sp.cos(2*x1 + x3)]])
x = sp.Matrix([[x0, x1],
               [x2, x3]])

# Compute the gradient
Gradient(f, x)

#compute gradient tensor
from sympy import *

# Declare symbols again (if needed)
x0, x1, x2, x3 = symbols("x0 x1 x2 x3")

# Define a 2×2 matrix-valued function
f = Matrix([
    [x0**2 * x1 * x2, x1**2 * x2 * x3],
    [x3**2 * x1 * x3, x2**3 * x1]
])

# Compute the gradient matrix: each element is a 2×2 matrix of partial derivatives
gradient_f = Matrix(2, 2, lambda i, j: Matrix([
    [f[i, j].diff(x0), f[i, j].diff(x1)],
    [f[i, j].diff(x2), f[i, j].diff(x3)]
]))

# Display the 2×2 matrix where each entry is a 2×2 matrix of partial derivatives
display(gradient_f)

