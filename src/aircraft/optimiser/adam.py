import numpy as np

alpha = 0.01

beta_1 = 0.9
beta_2 = 0.999

def func(x):
    return np.sum(x**2)

theta = np.array([1.0, 2.0, 3.0])

m0 = 0
v0 = 0
t = 0

converged = False

while not converged:
    t += 1
    grad = 2 * theta
    m = beta_1 * m0 + (1 - beta_1) * grad
    v = beta_2 * v0 + (1 - beta_2) * grad**2
    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)
    theta = theta - alpha * m_hat / (np.sqrt(v_hat) + 1e-8)
    converged = np.linalg.norm(grad) < 1e-6
    print(f"Iteration {t}: theta = {theta}, grad = {grad}")
    m0 = m
    v0 = v
