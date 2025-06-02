# Doolittle LU Decomposition Method
import numpy as np

def doolittle(A, B):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U

# Input for Doolittle Method
A1 = np.array([[2, -5, 3],
               [-1, 2, -1],
               [3, -1, 2]], dtype=float)

B1 = np.array([2, 1, 3], dtype=float)

L1, U1 = doolittle(A1, B1)
y1 = np.linalg.solve(L1, B1)
x1 = np.linalg.solve(U1, y1)

print("===== DOOLITTLE METHOD =====")
print(f"The lower triangular matrix L is:\n{L1}")
print(f"\nThe upper triangular matrix U is:\n{U1}")
print(f"\nSolution X:\n{x1}")
