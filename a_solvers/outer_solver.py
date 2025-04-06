import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import a_solvers.inner_solver as solver
from a_solvers.inner_solver import n, phi

# Parameters for the outer objective
G = 5.0           # Lump-sum (or other transfer) remains fixed
theta = 1.0       # degree of pollution disutility
# xi will be the parameter over which we vary in this analysis.

# We hold tau_w constant (all zeros)
tau_w_const = np.zeros(n)

def swf_for_tau_z(tau_z, xi):

    sol, res, conv = solver.solve(tau_w_const, tau_z, G)
    if not conv:
        return -1e10  # Penalize non-convergence
    # Retrieve utilities and aggregate pollution measure from the results
    utilities = res.get('utilities', np.zeros(n))
    agg_polluting = res.get('z_c', 0) + res.get('z_d', 0)
    welfare = np.sum(utilities) - xi * (agg_polluting**theta)
    return welfare

def objective_tau_z(x, xi):

    tau_z = x[0]
    return -swf_for_tau_z(tau_z, xi)

def find_optimal_tau_z(xi):
    """
    For a given xi, find the optimal tau_z in [0.1, 3.0] that maximizes social welfare.
    Returns the optimal tau_z.
    """
    x0 = np.array([0.5])
    bounds = [(0.1, 3.0)]
    res = minimize(objective_tau_z, x0, args=(xi,), method='SLSQP', bounds=bounds)
    if res.success:
        return res.x[0]
    else:
        print(f"Optimization failed for xi = {xi:.2f}: {res.message}")
        return np.nan

# --- Vary xi from 0.1 to 1.0 and record corresponding optimal tau_z ---
xi_values = np.linspace(0.1, 0.6, 100)
optimal_tau_z_array = np.zeros_like(xi_values)

for idx, xi in enumerate(xi_values):
    optimal_tau_z_array[idx] = find_optimal_tau_z(xi)
    print(f"xi = {xi:.2f}, optimal tau_z = {optimal_tau_z_array[idx]:.4f}")

# --- Plot optimal tau_z as a function of xi ---
plt.figure(figsize=(8,6))
plt.plot(xi_values, optimal_tau_z_array, 'b-', linewidth=2)
plt.xlabel(r'$\xi$', fontsize=14)
plt.ylabel(r'Optimal $\tau_z$', fontsize=14)
plt.title(r'Optimal $\tau_z$ as a Function of $\xi$', fontsize=16)
plt.tight_layout()
plt.savefig("optimal_tau_z_vs_xi_no_ic.pdf")
plt.show()