# Cholesky Decomposition Method
import numpy as np

def cholesky(A, B):
    n = len(A)
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum(L[i, k] ** 2 for k in range(j)))
            else:
                L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] for k in range(j))) / L[j, j]
    
    y = np.linalg.solve(L, B)
    x = np.linalg.solve(L.T, y)
    
    return L, L.T, x

# Input for Cholesky Method (symmetric positive-definite matrix)
A2 = np.array([[4, 12, -16],
               [12, 37, -43],
               [-16, -43, 98]], dtype=float)

B2 = np.array([1, 2, 3], dtype=float)

L2, U2, x2 = cholesky(A2, B2)

print("\n===== CHOLESKY METHOD =====")
print(f"The lower triangular matrix L is:\n{L2}")
print(f"\nThe upper triangular matrix U (L.T) is:\n{U2}")
print(f"\nSolution X:\n{x2}")
