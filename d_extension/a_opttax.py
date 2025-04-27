# plot_tau_z_comparison.py (Adding third scenario with optimal fixed tau_w)

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.optimize import minimize
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

a_solvers_dir = os.path.join(project_root, "a_solvers")

# ensure both are on sys.path
for p in (project_root, a_solvers_dir):
    if p not in sys.path:
        sys.path.insert(0, p)
import a_solvers.outer_solver_ext as outer_solver  # Imports the outer solver with maximize_welfare(G, xi)
import a_solvers.inner_solver_ext as solver       # Import inner solver directly too
# Removed theta from import, defined locally below
from a_solvers.inner_solver_ext import n, alpha, beta, gamma, d0, phi, t as T  # Import necessary params
import os  # For saving figure
import matplotlib as mpl   # only needed once, before any figures are created
mpl.rcParams.update({
    "text.usetex": True,
    "font.family":  "serif",
    "font.serif":  ["Palatino"],      # this line makes Matplotlib insert \usepackage{mathpazo}
    "text.latex.preamble": r"""
        \PassOptionsToPackage{sc}{mathpazo}  % give mathpazo the 'sc' option
        \linespread{1.5}
        \usepackage[T1]{fontenc}
    """,
})

# --- Simulation Parameters ---
G_value = 5.0
# Ensure theta is defined locally
theta = 1.0

# Define the fixed τ_w set for scenario 2
fixed_tau_w_preexisting = np.array([0.015, 0.072, 0.115, 0.156, 0.24])

# --- Compute the fixed τ_w for scenario 3 by solving for ξ = 0.1 ---
fixed_tau_w_optimal_xi01 = np.array([-1.22559844, -0.11142818,  0.17570669,  0.36519415,  0.62727242])

# Define the range and number of ξ values to test
xi_values = np.linspace(0.1, 0.25, 10)
# --- End Simulation Parameters ---

# --- Function to optimize ONLY τ_z for FIXED τ_w (Unchanged) ---
def maximize_welfare_fixed_w(G, xi, fixed_tau_w_arr):
    """
    Optimizes social welfare by choosing only tau_z, given fixed G, xi, and tau_w.
    """
    def swf_obj_fixed_w(tau_z_scalar, G_val, xi_val, fw_arr):
        tau_z = tau_z_scalar[0] if isinstance(tau_z_scalar, (list, np.ndarray)) else tau_z_scalar
        try:
            solution, results, converged = solver.solve(fw_arr, tau_z, G_val)
            if not converged:
                return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            valid_utilities = utilities
            welfare = np.sum(valid_utilities) - 5*xi_val * (agg_polluting**theta)
            return -welfare
        except Exception:
            return 1e10

    tau_z_bounds = [(1e-6, 100.0)]
    initial_tau_z_guess = [1.0]
    try:
        result = minimize(swf_obj_fixed_w,
                          initial_tau_z_guess,
                          args=(G, xi, fixed_tau_w_arr),
                          method='SLSQP',
                          bounds=tau_z_bounds,
                          options={'disp': False, 'ftol': 1e-15})
        if result.success:
            return result.x[0], -result.fun
        else:
            print("FAILURE in fixed-w τ_z optimization")
            return None, None
    except Exception:
        return None, None

# Lists to store results for all 3 scenarios
tau_z_optimal_w_results = []
tau_z_fixed_preexisting_results = []
tau_z_fixed_optimal_xi01_results = []
valid_xi_optimal_w = []
valid_xi_fixed_preexisting = []
valid_xi_fixed_optimal_xi01 = []

# --- Scenario 1: Optimal τ_w and τ_z ---
print("Running Scenario 1: Variable τ_w...")
print("-" * 30)
for xi_val in xi_values:
    try:
        opt_tau_w, opt_tau_z, max_welfare_val = outer_solver.maximize_welfare(G_value, xi_val)
        tau_z_optimal_w_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_optimal_w.append(xi_val)
    except Exception as e:
        print(f"    Error in Scenario 1 for ξ = {xi_val:.4f}: {e}")
        tau_z_optimal_w_results.append(np.nan)
        valid_xi_optimal_w.append(xi_val)
print("Scenario 1 finished.")

# --- Scenario 2: Fixed τ_w (Pre-existing), Optimal τ_z ---
print("\nRunning Scenario 2: Fixed τ_w (Pre-existing)...")
print(f"(Using fixed τ_w = {fixed_tau_w_preexisting})")
print("-" * 30)
for xi_val in xi_values:
    try:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_preexisting)
        tau_z_fixed_preexisting_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_fixed_preexisting.append(xi_val)
    except Exception as e:
        print(f"    Error in Scenario 2 for ξ = {xi_val:.4f}: {e}")
        tau_z_fixed_preexisting_results.append(np.nan)
        valid_xi_fixed_preexisting.append(xi_val)
print("Scenario 2 finished.")

# --- Scenario 3: Fixed τ_w (Optimal at ξ=0.1), Optimal τ_z ---
print("\nRunning Scenario 3: Fixed τ_w (Optimal at ξ=0.1)...")
print(f"(Using fixed τ_w = {fixed_tau_w_optimal_xi01})")
print("-" * 30)
for xi_val in xi_values:
    try:
        opt_tau_z, _ = maximize_welfare_fixed_w(G_value, xi_val, fixed_tau_w_optimal_xi01)
        tau_z_fixed_optimal_xi01_results.append(opt_tau_z if opt_tau_z is not None else np.nan)
        valid_xi_fixed_optimal_xi01.append(xi_val)
    except Exception as e:
        print(f"    Error in Scenario 3 for ξ = {xi_val:.4f}: {e}")
        tau_z_fixed_optimal_xi01_results.append(np.nan)
        valid_xi_fixed_optimal_xi01.append(xi_val)
print("Scenario 3 finished.")

print("-" * 30)
print("Simulations complete.")

# --- Plotting (3 lines, updated labels) ---

# Convert lists to numpy arrays
tau_z_optimal_w_results = np.array(tau_z_optimal_w_results)
valid_xi_optimal_w     = np.array(valid_xi_optimal_w)
tau_z_fixed_preexisting_results = np.array(tau_z_fixed_preexisting_results)
valid_xi_fixed_preexisting      = np.array(valid_xi_fixed_preexisting)
tau_z_fixed_optimal_xi01_results = np.array(tau_z_fixed_optimal_xi01_results)
valid_xi_fixed_optimal_xi01      = np.array(valid_xi_fixed_optimal_xi01)

plt.figure(figsize=(7, 5))

# Plot each scenario if data is valid
mask1 = ~np.isnan(tau_z_optimal_w_results)
mask2 = ~np.isnan(tau_z_fixed_preexisting_results)
mask3 = ~np.isnan(tau_z_fixed_optimal_xi01_results)

if np.any(mask1):
    plt.plot(valid_xi_optimal_w[mask1],
             tau_z_optimal_w_results[mask1],
             linestyle='-', label='Variable $\\tau_w$')

if np.any(mask2):
    plt.plot(valid_xi_fixed_preexisting[mask2],
             tau_z_fixed_preexisting_results[mask2],
             linestyle='--', label='Fixed $\\tau_w$ (Pre-existing)')

if np.any(mask3):
    plt.plot(valid_xi_fixed_optimal_xi01[mask3],
             tau_z_fixed_optimal_xi01_results[mask3],
             linestyle=':', label='Fixed $\\tau_w$ (Optimal at $\\xi=0.1$)')

plt.xlim(0.1, 0.25)
plt.xlabel(r'Environmental preference ($\xi$)', fontsize=14)
plt.ylabel(r'Optimal environmental tax ($\tau_z$)', fontsize=14)
plt.legend(loc='best')
plt.grid(True, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
plt.tight_layout()

plt.savefig("d_extension/a_opttax.pdf", bbox_inches='tight')
