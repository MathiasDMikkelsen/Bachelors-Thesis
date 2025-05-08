import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import a_inner_solver as solver
from a_inner_solver import n, phi, t  # t is assumed defined in inner_solver



# Parameters for the outer objective
G = 5.0           # Fixed lump-sum component in the model
theta = 1.0       # Degree of pollution disutility

# We hold tau_w constant (given values)
tau_w_const = np.array([0.015, 0.072, 0.115, 0.156, 0.24])

def swf_for_tau_z(tau_z, xi):
    """
    For a given tau_z and xi, solve the inner equilibrium (with fixed tau_w and G)
    and calculate social welfare as:
    
        SWF = sum_i U_i - xi*(z_c + z_d)^theta
    
    If the inner solver does not converge, return a large penalty.
    """
    sol, res, conv = solver.solve(tau_w_const, tau_z, G)
    if not conv:
        return -1e10  # Penalize non-convergence
  
    utilities = res.get('utilities', np.zeros(n))
    agg_polluting = res.get('z_c', 0) + res.get('z_d', 0)
    welfare = np.sum(utilities) - xi * (agg_polluting**theta)
    return welfare

def objective_tau_z(x, xi):
    tau_z = x[0]
    return -swf_for_tau_z(tau_z, xi)

def find_optimal_tau_z(xi):
    x0 = np.array([0.5])
    bounds = [(0.1, 30.0)]
    res = minimize(objective_tau_z, x0, args=(xi,), method='SLSQP', bounds=bounds)
    if res.success:
        return res.x[0]
    else:
        print(f"Optimization failed for xi = {xi:.2f}: {res.message}")
        return np.nan

# --- Vary xi and compute optimal tau_z ---
xi_values = np.linspace(0.1, 10.0, 100)
optimal_tau_z_array = np.zeros_like(xi_values)

# Prepare arrays for government revenue and lump-sum transfer.
gov_revenue_array = np.zeros_like(xi_values)
lump_sum_array = np.zeros_like(xi_values)   # Will be computed per xi

for idx, xi in enumerate(xi_values):
    tau_z_opt = find_optimal_tau_z(xi)
    optimal_tau_z_array[idx] = tau_z_opt
    print(f"xi = {xi:.2f}, optimal tau_z = {tau_z_opt:.4f}")
    
    # Evaluate the inner solver at the optimal tau_z
    sol, res, conv = solver.solve(tau_w_const, tau_z_opt, G)
    if conv:
        # Compute total government revenue:
        # revenue = sum(tau_w_const * w * phi*(t - l_agents)) + tau_z_opt*(z_c+z_d)
        l_agents = res.get('l_agents', np.full(n, t))
        revenue = np.sum(tau_w_const * res["w"] * phi * (t - l_agents)) \
                  + tau_z_opt * (res.get("z_c", 0) + res.get("z_d", 0))
        gov_revenue_array[idx] = revenue
        
        # Compute lump-sum transfer as: tax revenue from environment minus G:
        lump_sum = tau_z_opt * (res.get("z_c", 0) + res.get("z_d", 0)) - G
        lump_sum_array[idx] = lump_sum
    else:
        gov_revenue_array[idx] = np.nan
        lump_sum_array[idx] = np.nan

# --- Plotting: 1 row, 3 columns ---
fig, axs = plt.subplots(1, 3, figsize=(16, 5))

# Left subplot: Optimal tau_z vs. xi
axs[0].plot(xi_values, optimal_tau_z_array, 'b-', linewidth=2)
axs[0].set_xlabel(r'$\xi$', fontsize=12)
axs[0].set_ylabel(r'Optimal $\tau_z$', fontsize=12)
axs[0].set_title(r'Optimal $\tau_z$ vs. $\xi$', fontsize=14)

# Middle subplot: Total Government Revenue vs. xi
axs[1].plot(xi_values, gov_revenue_array, 'g-', linewidth=2)
axs[1].set_xlabel(r'$\xi$', fontsize=12)
axs[1].set_ylabel('Total Government Revenue', fontsize=12)
axs[1].set_title(r'Gov. Revenue vs. $\xi$', fontsize=14)

# Right subplot: Lump-Sum Transfer vs. xi
axs[2].plot(xi_values, lump_sum_array, 'r-', linewidth=2)
axs[2].set_xlabel(r'$\xi$', fontsize=12)
axs[2].set_ylabel('Lump-Sum Transfer', fontsize=12)
axs[2].set_title(r'Lump-Sum vs. $\xi$', fontsize=14)

plt.tight_layout()
plt.savefig("optimal_tau_z_government_revenue_lumpsum.pdf")
plt.show()