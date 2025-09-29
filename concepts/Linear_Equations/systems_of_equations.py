"""
Section 1.1 - Systems of Linear Equations
-----------------------------------------
Cases:
1. Unique Solution
2. No Solution
3. Infinite Solutions
"""

import numpy as np
import matplotlib.pyplot as plt

# Helper function: print system
def print_system(A, b):
    m, n = A.shape
    for i in range(m):
        row = " + ".join([f"{A[i, j]}*x{j+1}" for j in range(n)])
        print(f"{row} = {b[i]}")
    print()

# Helper function: analyze system
def analyze_system(A, b):
    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(np.column_stack([A, b]))
    n = A.shape[1]

    if rank_A == rank_Ab == n:
        return "✅ Unique solution"
    elif rank_A == rank_Ab < n:
        return "♾️ Infinite solutions"
    else:
        return "❌ No solution"

# Helper function: plot 2D system
def plot_system(A, b, title):
    x_vals = np.linspace(-5, 5, 200)
    plt.figure()
    for i in range(A.shape[0]):
        if A[i, 1] != 0:  # avoid div by zero
            y_vals = (b[i] - A[i, 0]*x_vals) / A[i, 1]
            plt.plot(x_vals, y_vals, label=f"Eqn {i+1}")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Case 1: Unique Solution ===
print("=== Unique Solution ===")
A1 = np.array([[1, 1],
               [2, -1]])
b1 = np.array([3, 0])

print_system(A1, b1)
print("Analysis:", analyze_system(A1, b1))

# Solve
x1 = np.linalg.solve(A1, b1)
print("Solution:", x1)

# Plot
plot_system(A1, b1, "Unique Solution (lines intersect at one point)")

# === Case 2: No Solution ===
print("\n=== No Solution ===")
A2 = np.array([[1, 1],
               [1, 1]])
b2 = np.array([2, 4])

print_system(A2, b2)
print("Analysis:", analyze_system(A2, b2))

# np.linalg.solve would fail, so use lstsq
x2, residuals, rank, s = np.linalg.lstsq(A2, b2, rcond=None)
print("Least squares solution:", x2, "Residuals:", residuals)

# Plot
plot_system(A2, b2, "No Solution (parallel lines)")

# === Case 3: Infinite Solutions ===
print("\n=== Infinite Solutions ===")
A3 = np.array([[1, 1],
               [2, 2]])
b3 = np.array([2, 4])

print_system(A3, b3)
print("Analysis:", analyze_system(A3, b3))

# lstsq gives one valid solution
x3, residuals, rank, s = np.linalg.lstsq(A3, b3, rcond=None)
print("One solution from infinite set:", x3, "Residuals:", residuals)

# Plot
plot_system(A3, b3, "Infinite Solutions (overlapping lines)")
