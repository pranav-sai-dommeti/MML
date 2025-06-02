import sympy as sp

x, y = sp.symbols('x y')
f = x - y + 2*(x**2) + 2*x*y + 2*(y**2)

grad = sp.Matrix([sp.diff(f, var) for var in (x, y)])
Hessian = sp.Matrix([[sp.diff(grad[i], var) for var in (x, y)] for i in range(2)])

print("Gradient Vector:")
display(grad)

print("\nHessian Matrix:")
display(Hessian)

x_val = sp.Matrix([[0], [0]])
x_new = x_val.copy()

for i in range(2):
    grad_val = grad.subs({x: x_val[0, 0], y: x_val[1, 0]})
    Hessian_val = Hessian.subs({x: x_val[0, 0], y: x_val[1, 0]})
    h_inv = Hessian_val.inv()
    x_new = x_val - h_inv * grad_val
    print(f"\nIteration {i+1}:")
    print("Gradient at this step:")
    display(grad_val)
    print("Updated x:")
    display(x_new)
    x_val = x_new

print("\nFinal result:")
print("Optimized x:")
display(x_new)

min_val = f.subs({x: x_new[0, 0], y: x_new[1, 0]})
print("The minimum value of the function is:", min_val)
