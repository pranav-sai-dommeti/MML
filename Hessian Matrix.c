import sympy as sp
import numpy as np

x, y, z = sp.symbols('x y z', real=True)
f1 = -(x**3) + 3*x*z + 2*y - (y**2) - 3*(z**2)

grad_f = sp.Matrix([sp.diff(f1, var) for var in (x, y, z)])
display(grad_f)

Hessian = sp.Matrix([[sp.diff(grad_f[i], var) for var in (x, y, z)] for i in range(3)])
display(Hessian)

stationary_points = sp.solve([grad_f[i] for i in range(3)], (x, y, z))
real_stationary_points = [point for point in stationary_points if all(coord.is_real for coord in point)]
print("\nReal Stationary Points:", real_stationary_points)

for i, point in enumerate(real_stationary_points):
    print(f"\nAnalysis for Real Stationary Point {i+1}: {point}")
    H_at_point = Hessian.subs({x: point[0], y: point[1], z: point[2]})
    display(H_at_point)
    H_at_point_num = np.array(H_at_point, dtype=np.float64)
    eigvals = np.linalg.eigvals(H_at_point_num)
    print("Eigenvalues:", eigvals)
    if any(e == 0 for e in eigvals):
        print("Test inconclusive")
    elif all(e > 0 for e in eigvals):
        print("Local Minimum.")
    elif all(e < 0 for e in eigvals):
        print("Local Maximum.")
    else:
        print("Saddle Point.")
