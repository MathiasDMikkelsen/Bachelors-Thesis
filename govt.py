import numpy as np
import importlib
from scipy.optimize import minimize
import complete_equi

importlib.reload(complete_equi)

# Parameters for the SWF function
xi = 7.0
theta = 1.0

def swf_obj(x):
    """
    Objective function for social welfare maximization.

    Args:
        x (np.ndarray): A 1D array containing tau_w, tau_z and l (l_1, l_2, l_3, l_4, l_5).
                        tau_w is x[0], tau_z is x[1] and l is x[2:7].

    Returns:
        float: Negative social welfare (since we are minimizing).
    """
    tau_w = x[0]
    tau_z = x[1]
    l = x[2:7]

    # Ensure l_vector sums to 1
    if not np.isclose(np.sum(l), 1.0):
        return 1e10  # High penalty if l_vector doesn't sum to 1

    utilities, aggregate_polluting, first_converged = complete_equi.main_solve(tau_w, tau_z, l, n=5)

    # Print convergence status for these tax policy parameter candidates
    print(f"\nCandidate tau_w = {tau_w:.4f}, tau_z = {tau_z:.4f} | First Optimizer Convergence: {first_converged}")

    if not first_converged:
        # Penalize non-convergence.
        return 1e10

    welfare = np.sum(utilities) - xi * aggregate_polluting**theta
    return -welfare  # Negative because we use a minimizer.

# Initial guess for tau_w, tau_z and l_vector
initial_guess = np.array([0.15, 3.0, 0.2, 0.2, 0.2, 0.2, 0.2])  # tau_w, tau_z, l1, l2, l3, l4, l5

# Bounds for tau_w, tau_z and l_vector
bounds = [(0.0, 0.5), (0.1, 100.0)] + [(0.0, 1.0)] * 5  # tau_w, tau_z bounds, l_vector bounds

# Constraints for l_vector (sum to 1)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x[2:7]) - 1.0})

# Optimize over tau_w, tau_z and l_vector
res = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

opt_tau_w = res.x[0]
opt_tau_z = res.x[1]
opt_l = res.x[2:7]
max_welfare = -res.fun

print("\n=== Social Welfare Optimization ===")
print("Optimal tau_w:", opt_tau_w)
print("Optimal tau_z:", opt_tau_z)
print("Optimal l_vector:", opt_l)
print("Maximized Social Welfare:", max_welfare)
print("Optimizer Convergence:", res.success)