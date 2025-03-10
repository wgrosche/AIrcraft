import casadi as ca
import numpy as np

def chebyshev_points(N):
    return np.cos(np.pi * np.arange(N + 1) / N)

def chebyshev_polynomials(N, t):
    T = [ca.MX.ones(t.shape), t]
    for n in range(2, N + 1):
        Tn = 2 * t * T[-1] - T[-2]
        T.append(Tn)
    return T

# Step 3: Derivatives of Chebyshev polynomials

def chebyshev_derivatives(T):
    dT = []
    for poly in T:
        dT.append(ca.jacobian(poly, t))
    return dT

# Number of collocation points
N = 10

# Generate collocation points
t = ca.MX.sym('t')
tau = chebyshev_points(N)
T = chebyshev_polynomials(N, t)
dT = chebyshev_derivatives(T)

print("Chebyshev-Gauss-Lobatto points:", tau)
print("Chebyshev polynomials:", T)
print("Chebyshev polynomial derivatives:", dT)

# Placeholder: Use these polynomials for flat output parameterization and dynamic constraints
