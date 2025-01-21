"""
Development of an adapted batched sgd approach
"""

def adam_step()


# Initialization
x = initial_guess
alpha = initial_step_size
m = number_of_samples
k = top_k_candidates
tolerance = convergence_criterion

while not converged:
    # Step 1: Generate m candidate points
    candidates = []
    for _ in range(m):
        grad = compute_gradient(f, x)
        candidate_x = x - alpha * grad
        candidates.append((candidate_x, f(candidate_x)))

    # Step 2: Select the top-k points with the smallest objective values
    candidates.sort(key=lambda item: item[1])  # Sort by f(x)
    top_candidates = candidates[:k]

    # Step 3: Reduce step size
    alpha = alpha * reduction_factor

    # Step 4: Evaluate new candidate and compare
    new_candidate = x - alpha * compute_gradient(f, x)
    if f(new_candidate) < max(top_candidates, key=lambda item: item[1])[1]:
        x = new_candidate  # Update to the better candidate
    else:
        x = min(top_candidates, key=lambda item: item[1])[0]  # Use the best previous candidate

    # Check for convergence
    if np.linalg.norm(compute_gradient(f, x)) < tolerance:
        converged = True




