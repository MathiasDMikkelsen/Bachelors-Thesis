import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import jeppes_method

# Parameters for the SWF function
xi = 1.0
theta = 1.0
n = 5  # Number of households
G = 1.0  # Parameter G (can be chosen)
phi = np.array([1.0] * n)  # Participation factor (modify as needed)
w = np.array([1.0] * n)  # Household wages (modify as needed)

def swf_obj(x):
    """
    Objective function for social welfare maximization.

    Args:
        x (np.ndarray): A 1D array containing tau_w (vector), tau_z and l (l_1, l_2, ..., l_n).
                        tau_w is x[0:n], tau_z is x[n], l is x[n+1:2*n+1].

    Returns:
        float: Negative social welfare (since we are minimizing).
    """
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]

    # Ensure l_vector sums to 1
    if not np.isclose(np.sum(l), 1.0):
        return 1e10  # High penalty if l_vector doesn't sum to 1

    utilities, aggregate_polluting, first_converged = jeppes_method.main_solve_print(tau_w, tau_z, l, n=n)

    # Print convergence status for these tax policy parameter candidates
    print(f"\nCandidate tau_w = {tau_w}, tau_z = {tau_z:.4f} | First Optimizer Convergence: {first_converged}")

    if not first_converged:
        # Penalize non-convergence.
        return 1e10

    welfare = np.sum(utilities) - xi * aggregate_polluting**theta
    return -welfare  # Negative because we use a minimizer.

# Initial guess for tau_w, tau_z and l_vector
initial_guess = np.array([0.15] * n + [3.0] + [0.2] * n)  # tau_w (vector), tau_z, l1, l2, ..., ln

# Bounds for tau_w, tau_z and l_vector
bounds = [(0.0, 1.0)] * n + [(0.1, 100.0)] + [(0.0, 1.0)] * n  # tau_w (vector), tau_z bounds, l_vector bounds

# Constraints for l_vector (sum to 1) and tau_w constraint (sum tau_w * phi * w = G)
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x[n+1:2*n+1]) - 1.0},  # l_vector sum to 1
    {'type': 'eq', 'fun': lambda x: np.sum(x[0:n] * phi * w) - G}  # tau_w * phi * w sum to G
]

# Optimize over tau_w, tau_z and l_vector
res = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

opt_tau_w = res.x[0:n]
opt_tau_z = res.x[n]
opt_l = res.x[n+1:2*n+1]
max_welfare = -res.fun

print("\n=== Social Welfare Optimization ===")
print("Optimal tau_w:", opt_tau_w)
print("Optimal tau_z:", opt_tau_z)
print("Optimal l_vector:", opt_l)
print("Maximized Social Welfare:", max_welfare)
print("Optimizer Convergence:", res.success)
