import numpy as np
import sympy as sp

def fibonacci(n):
    sequence = []
    n0, n1 = 1, 1
    for i in range(n):
        sequence.append(n0)
        n0, n1 = n1, n0 + n1
    return sequence

def fibonacci_search(f, a, b, n):
    x = sp.Symbol('x')
    fib = fibonacci(n + 1)
    f_lambda = sp.lambdify(x, f, 'numpy')

    for k in range(1, n):
        Lk = (fib[n - k] / fib[n - k + 1]) * (b - a)
        x1 = b - Lk
        x2 = a + Lk

        f1 = f_lambda(x1)
        f2 = f_lambda(x2)

        if f1 > f2:
            a = x1
        else:
            b = x2

        print(f"Iteration {k}: a={a:.5f}, b={b:.5f}, x1={x1:.5f}, x2={x2:.5f}")

    return (a + b) / 2

x = sp.Symbol('x')
f = x**2 - 2.6*x + 2
a, b = -2, 3
n = 20

min_x = fibonacci_search(f, a, b, n)
print("Approximate minimum at x =", min_x)
